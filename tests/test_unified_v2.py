"""Tests for the unified Model Forge v2.0 feasibility execution:

- Direct ComfyUI workflow support in media_providers.py + activation_plan
- Approval-gated local engine start/stop (loopback 127.0.0.1, minimal env,
  PID provenance) + local_engine_clis
- Expanded read-only plugin_api.py routes (/browse, POST /probe with metadata
  SSRF block, /jobs, /engines/local-cli)
- Headless smoke test for unified model-forge/desktop/plugin.js
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

from model_forge import media_providers, pull, tools
from model_forge.registry import ModelEntry
from model_forge.targets import ComputeTarget


def _hermes_fakes(monkeypatch, tmp_path, *, config=None):
    store = {"cfg": dict(config or {})}
    plugins = types.ModuleType("plugins")
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_data_dir = lambda name: tmp_path / name
    plugins.plugin_storage = storage
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)

    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.load_config_readonly = lambda: dict(store["cfg"])
    cfg_mod.load_config = lambda: store["cfg"]
    cfg_mod.save_config = lambda c: store.__setitem__("cfg", c)
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.config = cfg_mod
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg_mod)
    return store


def test_detect_media_protocol_comfyui_vs_openai():
    proto_comfy = media_providers.detect_media_protocol(
        "http://127.0.0.1:8188",
        kind="image",
        get_json=lambda u: {"system": {"comfyui-version": "0.3"}, "devices": []},
        has_route=lambda root, route: False,
    )
    assert proto_comfy == "comfyui"

    proto_openai = media_providers.detect_media_protocol(
        "http://127.0.0.1:8080/v1",
        kind="image",
        get_json=lambda u: {"system": {}},
        has_route=lambda root, route: route == "/v1/images/generations",
    )
    assert proto_openai == "openai"


def test_resolve_comfy_lane_maps_catalog_and_edit_i2v():
    assert media_providers.resolve_comfy_lane(
        "qwen-image-2.1", kind="image") == "qwen-image-2.1"
    assert media_providers.resolve_comfy_lane(
        "gen/krea2", kind="image") == "qwen-image-2.1"
    assert media_providers.resolve_comfy_lane(
        "qwen-image-2.1", kind="image", has_image_input=True
    ) == "qwen-image-2.1-edit"
    assert media_providers.resolve_comfy_lane(
        "ltx-2.5", kind="video") == "h3-t2v"
    assert media_providers.resolve_comfy_lane(
        "ltx-2.5", kind="video", has_image_input=True) == "h3-i2v"


def test_run_comfyui_media_reads_output_and_guards_view_origin(tmp_path):
    img_file = tmp_path / "out.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")

    class FakeRunner:
        workflows = {"qwen-image-2.1": types.SimpleNamespace(output_kind="image")}

        def __init__(self, root):
            self.root = root

        def run(self, lane, params):
            return {"outputs": [{"filename": "out.png", "path": str(img_file)}]}

    res = media_providers.run_comfyui_media(
        "http://127.0.0.1:8188", "a sunset", kind="image",
        model="qwen-image-2.1", runner_factory=FakeRunner,
    )
    assert res["ok"] is True
    assert res["protocol"] == "comfyui"
    assert res["bytes"].startswith(b"\x89PNG")

    # Remote /view fetch works only on configured ComfyUI origin
    class RemoteRunner:
        workflows = {"qwen-image-2.1": types.SimpleNamespace(output_kind="image")}

        def __init__(self, root):
            self.root = root

        def run(self, lane, params):
            return {"outputs": [{"filename": "out.png",
                                 "view_url": "http://evil.example.com/view?f=out.png"}]}

    blocked = media_providers.run_comfyui_media(
        "http://127.0.0.1:8188", "a sunset", kind="image",
        model="qwen-image-2.1", runner_factory=RemoteRunner,
        get_bytes=lambda u: b"SHOULD_NOT_FETCH",
    )
    assert blocked["ok"] is False
    assert "outside configured root" in blocked["error"]


def test_direct_comfyui_activation_and_unified_settings_mirror(tmp_path, monkeypatch):
    store = _hermes_fakes(monkeypatch, tmp_path)
    remote = ComputeTarget(
        id="gpu-box", label="GPU Box",
        base_url="http://gpu-box.example.com:8188", kind="lan",
        vram_gb=24.0, ram_gb=64.0,
    )
    plan = tools.activation_plan(
        ModelEntry(id="ltx-2.5", name="LTX-2.5", modality="video",
                   engines=("comfyui",), source="Lightricks/LTX-2"),
        remote,
        available_engines=["comfyui"],
    )
    assert plan["ok"] is True
    assert plan["action"] == "persist_media_provider"
    assert plan["protocol"] == "comfyui"
    assert plan["base_url"] == "http://gpu-box.example.com:8188"  # NO /v1

    saved = tools.write_media_provider_config(
        "video", base_url=plan["base_url"], default_model="ltx-2.5")
    assert saved["provider"] == "model-forge"
    mf_settings = store["cfg"]["plugins"]["entries"]["model-forge"]["settings"]
    assert mf_settings["video_base_url"] == "http://gpu-box.example.com:8188"
    assert mf_settings["video_default_model"] == "ltx-2.5"


def test_engine_start_plan_approval_spawn_and_provenance_stop(tmp_path, monkeypatch):
    _hermes_fakes(monkeypatch, tmp_path)

    # Missing binary -> clear refusal
    missing = tools.engine_start_plan("ollama", which_fn=lambda b: None)
    assert missing["ok"] is False
    assert missing["action"] == "engine_not_installed"

    # llama.cpp -> defers to Hermes Local Models
    core = tools.engine_start_plan("llamacpp", which_fn=lambda b: "/usr/bin/x")
    assert core["ok"] is False
    assert core["action"] == "use_core_local_models"

    # vLLM requires safe model_id (rejects flag injection)
    bad_vllm = tools.engine_start_plan("vllm", model_id="--evil",
                                       which_fn=lambda b: "/usr/bin/vllm")
    assert bad_vllm["ok"] is False
    assert bad_vllm["action"] == "model_id_required"

    # Ollama plan -> strictly loopback 127.0.0.1:11434
    plan = tools.engine_start_plan("ollama", which_fn=lambda b: "/usr/bin/ollama")
    assert plan["ok"] is True
    assert plan["argv"] == ["ollama", "serve"]
    assert plan["env_extra"]["OLLAMA_HOST"] == "127.0.0.1:11434"

    # Policy hook escalates execute=True with exact command + loopback host
    monkeypatch.setattr(shutil, "which", lambda b: f"/usr/bin/{b}")
    hook = tools.policy_hook(
        "model_forge_install",
        {"action": "engine_start", "engine": "ollama", "execute": True},
    )
    assert hook["action"] == "approve"
    assert hook["rule_key"] == "model-forge:engine_start"
    assert "ollama serve" in hook["message"]
    assert "127.0.0.1" in hook["message"]

    # Spawn uses shell=False and minimal env (no leaked secret!)
    monkeypatch.setenv("SUPER_SECRET_KEY", "must_never_leak")
    spawned = []

    class FakeProc:
        pid = os.getpid()  # alive PID so provenance check passes

    def fake_popen(argv, **kw):
        spawned.append((argv, kw))
        return FakeProc()

    started = tools.start_local_engine("ollama", which_fn=lambda b: "/usr/bin/ollama",
                                       popen_fn=fake_popen)
    assert started["ok"] is True
    assert started["action"] == "engine_started"
    assert spawned[0][1]["shell"] is False
    assert "SUPER_SECRET_KEY" not in spawned[0][1]["env"]
    assert spawned[0][1]["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"

    # Stop kills ONLY Model Forge-managed PID and refuses unmanaged engines
    killed = []
    stopped = tools.stop_local_engine("ollama", kill_fn=lambda pid, sig: killed.append((pid, sig)))
    assert stopped["ok"] is True
    assert killed and killed[0][0] == os.getpid()

    unmanaged = tools.stop_local_engine("exo")
    assert unmanaged["ok"] is False
    assert unmanaged["action"] == "not_managed_by_model_forge"


def test_plugin_api_probe_blocks_metadata_and_lists_jobs(tmp_path, monkeypatch):
    _hermes_fakes(monkeypatch, tmp_path)
    # Load dashboard/plugin_api.py with a minimal FastAPI shim so tests need no FastAPI install.
    fastapi = types.ModuleType("fastapi")

    class _Router:
        def get(self, p): return lambda fn: fn
        def post(self, p): return lambda fn: fn

    fastapi.APIRouter = _Router
    fastapi.Body = lambda default=None: default
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)

    repo = Path(__file__).resolve().parent.parent
    try:
        from _repo import repo_root
        repo = repo_root()
    except Exception:
        pass
    api_path = repo / "model-forge" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("mf_plugin_api_test", api_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Metadata SSRF hosts and userinfo URLs are refused
    assert mod._probe_endpoint("http://169.254.169.254/latest")["ok"] is False
    assert mod._probe_endpoint("http://metadata.google.internal")["ok"] is False
    assert mod._probe_endpoint("http://user:pass@127.0.0.1:11434")["ok"] is False
    assert mod._probe_endpoint("file:///etc/passwd")["ok"] is False

    # Jobs listing reads persisted job files newest-first
    pull._write_job("aaa111", {"job_id": "aaa111", "state": "running", "model": "qwen3:8b"})
    jobs_res = mod._jobs()
    assert jobs_res["ok"] is True
    assert any(j["job_id"] == "aaa111" for j in jobs_res["jobs"])

    clis_res = mod._local_cli()
    assert clis_res["ok"] is True
    assert "ollama" in clis_res["clis"]


def test_unified_desktop_gui_headless_smoke():
    node = shutil.which("node")
    if node is None:
        return
    repo = Path(__file__).resolve().parent.parent
    try:
        from _repo import repo_root
        repo = repo_root()
    except Exception:
        pass
    smoke = repo / "tests" / "smoke_unified_desktop.js"
    if not smoke.is_file():
        return
    res = subprocess.run([node, str(smoke)], capture_output=True, text=True,
                         cwd=str(repo))
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    assert "ALL UNIFIED DESKTOP GUI SMOKE CHECKS PASSED" in res.stdout
