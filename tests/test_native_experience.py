"""Tests for v1.3 native experience backend capabilities:

- Ollama streaming HTTP pull (live + offline fallback + non-curated tags)
- Exo native placement install plan
- Hugging Face keyword search in discovery + model_forge_browse
- Native Hermes activation for image, video, vision, and live-served models
- Enriched cockpit_data (pull_credentials, active_wiring, engine_catalog)
"""

from __future__ import annotations

import json
import sys
import time
import types

from model_forge import discovery, pull, tools
from model_forge.engines.base import EngineStatus
from model_forge.registry import HardwareProfile, ModelEntry
from model_forge.targets import ComputeTarget, add_target


def _hermes_fakes(monkeypatch, tmp_path, *, config=None, secret=""):
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

    def _save(c):
        store["cfg"] = c

    cfg_mod.save_config = _save
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.config = cfg_mod

    setup_mod = types.ModuleType("hermes_cli.model_setup")
    setup_calls = []

    def _persist(model, provider, base_url="", api_mode=""):
        row = {"default": model, "provider": provider,
               "base_url": base_url, "api_mode": api_mode}
        store["cfg"]["model"] = row
        setup_calls.append(row)
        return row

    setup_mod.persist_model_selection = _persist
    hermes_cli.model_setup = setup_mod

    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg_mod)
    monkeypatch.setitem(sys.modules, "hermes_cli.model_setup", setup_mod)

    sec = types.ModuleType("agent.secret_scope")
    sec.get_secret_str = lambda name, default="": secret if name == "HF_TOKEN" else ""
    agent = types.ModuleType("agent")
    agent.secret_scope = sec
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.secret_scope", sec)
    return store, setup_calls


def test_ollama_install_plan_live_and_offline():
    local = ComputeTarget(id="local", label="This machine",
                          base_url="http://127.0.0.1:8080", kind="local")
    entry = ModelEntry(id="qwen3:8b", name="Qwen3 8B", modality="chat",
                       engines=("ollama",), source="qwen3:8b")
    live = tools.install_plan(entry, local, available_engines=["ollama"])
    assert live["ok"] is True
    assert live["action"] == "ollama_pull"
    assert live["model"] == "qwen3:8b"
    assert live["endpoint"].endswith("/api/pull")

    offline = tools.install_plan(entry, local, available_engines=[])
    assert offline["ok"] is True
    assert offline["action"] == "run_cli_on_target"
    assert offline["command"] == ["ollama", "pull", "qwen3:8b"]

    hf_entry = ModelEntry(id="hf-gguf", name="HF GGUF", modality="chat",
                          engines=("ollama",),
                          source="unsloth/Qwen3-8B-GGUF")
    hf_plan = tools.install_plan(hf_entry, local, available_engines=["ollama"])
    assert hf_plan["model"] == "hf.co/unsloth/Qwen3-8B-GGUF"


def test_ollama_background_pull_job_and_remote_record(tmp_path, monkeypatch):
    _hermes_fakes(monkeypatch, tmp_path)

    def fake_pull(model, base_url, *, on_progress=None):
        if on_progress:
            on_progress({"status_text": "pulling layer",
                         "done_bytes": 500, "total_bytes": 1000})
        return {"ok": True, "status_text": "success",
                "done_bytes": 1000, "total_bytes": 1000}

    started = pull.start_ollama_pull(
        "qwen3:8b", "http://gpu-box.example.com:11434",
        target_id="gpu-box", pull_fn=fake_pull,
    )
    assert started["ok"] is True
    jid = started["job_id"]
    deadline = time.time() + 3.0
    status = {}
    while time.time() < deadline:
        status = pull.job_status(jid)
        if status.get("state") in ("done", "error"):
            break
        time.sleep(0.02)
    assert status["state"] == "done"
    assert status["done_bytes"] == 1000
    assert status["total_bytes"] == 1000
    rows = pull.pull_entries("gpu-box")
    assert any(r["model_id"] == "qwen3:8b" and r["source"] == "observed"
               for r in rows)


def test_model_forge_install_direct_ollama_and_exo(tmp_path, monkeypatch):
    _hermes_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        tools, "detect_engines",
        lambda base_url="", **kw: [
            EngineStatus(id="ollama", display_name="Ollama", available=True,
                         endpoint="http://127.0.0.1:11434",
                         modalities=("chat", "vision")),
            EngineStatus(id="exo", display_name="Exo", available=True,
                         endpoint="http://127.0.0.1:52415",
                         modalities=("chat",)),
        ],
    )
    monkeypatch.setattr(
        pull, "start_ollama_pull",
        lambda model, base_url, target_id="local": {
            "ok": True, "job_id": "abcdef012345",
            "poll": "model_forge_install with job_id=abcdef012345",
        },
    )
    out = json.loads(tools.model_forge_install(
        {"model_id": "qwen3:8b", "engine": "ollama"}))
    assert out["ok"] is True
    assert out["action"] == "ollama_pull"
    assert out["job_id"] == "abcdef012345"
    assert out["endpoint"] == "http://127.0.0.1:11434/api/pull"

    exo_out = json.loads(tools.model_forge_install(
        {"model_id": "mlx-community/Qwen3-8B", "engine": "exo"}))
    assert exo_out["ok"] is True
    assert exo_out["action"] == "exo_place"
    assert exo_out["endpoint"].endswith("/place_instance")

    bad = json.loads(tools.model_forge_install(
        {"model_id": "bad\nmodel", "engine": "ollama"}))
    assert bad["ok"] is False
    assert bad["action"] == "install_refused"


def test_discovery_keyword_query_and_browse_filter(tmp_path, monkeypatch):
    _hermes_fakes(monkeypatch, tmp_path)
    url = discovery.search_url("text-to-image", limit=5, query="flux")
    assert "search=flux" in url
    assert "sort=downloads" in url
    assert "pipeline_tag=text-to-image" in url

    default_url = discovery.search_url("text-to-image", limit=5)
    assert "sort=lastModified" in default_url
    assert "search=" not in default_url

    seen_urls = []

    def fake_fetch(u):
        seen_urls.append(u)
        return [
            {"id": "org/flux-low", "lastModified": "2026-11-20",
             "downloads": 100, "pipeline_tag": "text-to-image"},
            {"id": "org/flux-top", "lastModified": "2026-11-10",
             "downloads": 50000, "pipeline_tag": "text-to-image"},
        ]

    res = discovery.fresh_models("image", query="flux",
                                 cache_dir=tmp_path, fetch=fake_fetch)
    assert res["live_discovery"] == "ok"
    assert res["models"][0]["id"] == "org/flux-top"
    assert any("search=flux" in u for u in seen_urls)

    monkeypatch.setattr(tools, "detect_engines", lambda base_url="", **kw: [])
    monkeypatch.setattr(discovery, "fetch_json", fake_fetch)
    browsed = json.loads(tools.model_forge_browse(
        {"modality": "image_edit", "query": "flux"}))
    assert browsed["query"] == "flux"
    assert all("flux" in (m["id"] + m["name"]).lower()
               for m in browsed["models"])


def test_activate_image_video_vision_and_live_served_models(tmp_path, monkeypatch):
    store, setup_calls = _hermes_fakes(monkeypatch, tmp_path)
    remote = ComputeTarget(
        id="gpu-box", label="GPU Box",
        base_url="http://gpu-box.example.com:8080", kind="lan",
        vram_gb=24.0, ram_gb=64.0,
    )
    add_target(remote, base_dir=tmp_path)
    monkeypatch.setattr(tools, "get_target",
                        lambda tid=None: (remote if tid == "gpu-box"
                                          else ComputeTarget(
                                              id="local", label="This machine",
                                              base_url="http://127.0.0.1:8080",
                                              kind="local")))
    monkeypatch.setattr(
        tools, "detect_engines",
        lambda base_url="", **kw: [
            EngineStatus(id="ollama", display_name="Ollama", available=True,
                         endpoint="http://127.0.0.1:11434",
                         modalities=("chat", "vision"),
                         models_loaded=("qwen3:8b",)),
            EngineStatus(id="vllm", display_name="vLLM-Omni", available=True,
                         endpoint="http://gpu-box.example.com:8080",
                         modalities=("chat", "vision", "image", "image_edit"),
                         models_loaded=("Qwen/Qwen-Image-2.1",)),
            EngineStatus(id="comfyui", display_name="ComfyUI", available=True,
                         endpoint="http://gpu-box.example.com:8080",
                         modalities=("image", "video")),
        ],
    )

    # 1. Image model activation writes image_gen.provider + model-forge-image
    img_plan = json.loads(tools.model_forge_activate(
        {"model_id": "qwen-image-2.1", "target": "gpu-box", "apply": True}))
    assert img_plan["ok"] is True
    assert img_plan["action"] == "persist_media_provider"
    assert img_plan["applied"] is True
    assert store["cfg"]["image_gen"]["provider"] == "model-forge"
    img_cfg = store["cfg"]["plugins"]["entries"]["model-forge-image"]["settings"]
    assert img_cfg["base_url"] == "http://gpu-box.example.com:8080/v1"
    assert img_cfg["default_model"] == "qwen-image-2.1"

    # Approval prompt names image_gen + model-forge-image honestly
    hook = tools.policy_hook(
        "model_forge_activate",
        {"model_id": "qwen-image-2.1", "target": "gpu-box", "apply": True},
    )
    assert hook["action"] == "approve"
    assert "image_gen" in hook["message"]
    assert "model-forge-image" in hook["message"]

    # 2. Video model activation writes video_gen.provider + model-forge-video
    vid_plan = json.loads(tools.model_forge_activate(
        {"model_id": "ltx-2.5", "target": "gpu-box", "apply": True}))
    assert vid_plan["ok"] is True
    assert vid_plan["applied"] is True
    assert store["cfg"]["video_gen"]["provider"] == "model-forge"
    vid_cfg = store["cfg"]["plugins"]["entries"]["model-forge-video"]["settings"]
    assert vid_cfg["base_url"] == "http://gpu-box.example.com:8080/v1"

    # 3. Vision model writes auxiliary.vision alongside chat/custom_providers
    vis_plan = json.loads(tools.model_forge_activate(
        {"model_id": "minicpm-v-4.6", "target": "gpu-box", "apply": True}))
    assert vis_plan["ok"] is True
    assert vis_plan["applied"] is True
    assert vis_plan["auxiliary_vision"]["applied"] is True
    assert store["cfg"]["auxiliary"]["vision"]["model"] == "minicpm-v-4.6"

    # 4. Live-served non-curated Ollama model wires to Ollama's real :11434 port
    live_plan = json.loads(tools.model_forge_activate(
        {"model_id": "qwen3:8b", "target": "local", "apply": True}))
    assert live_plan["ok"] is True
    assert live_plan["applied"] is True
    assert live_plan["base_url"] == "http://127.0.0.1:11434/v1"
    assert setup_calls[-1]["default"] == "qwen3:8b"
    assert setup_calls[-1]["base_url"] == "http://127.0.0.1:11434/v1"

    # 5. Unserved non-curated id still fails closed
    refused = json.loads(tools.model_forge_activate(
        {"model_id": "not-really-served:99b", "target": "local"}))
    assert "error" in refused


def test_cockpit_data_includes_wiring_credentials_and_engines(tmp_path, monkeypatch):
    _hermes_fakes(
        monkeypatch, tmp_path,
        config={
            "model": {"default": "qwen3:8b", "provider": "custom",
                      "base_url": "http://127.0.0.1:11434/v1"},
            "image_gen": {"provider": "model-forge"},
            "plugins": {"entries": {
                "model-forge-image": {"settings": {
                    "base_url": "http://gpu-box.example.com:8080/v1",
                    "default_model": "qwen-image-2.1",
                }},
            }},
            "custom_providers": [{
                "name": "model-forge-local",
                "base_url": "http://127.0.0.1:11434/v1",
                "models": {"qwen3:8b": {}},
            }],
        },
        secret="hf_secret_never_leaked",
    )
    monkeypatch.setattr(tools, "detect_engines", lambda base_url="", **kw: [])
    monkeypatch.setattr(tools, "probe_hardware",
                        lambda: HardwareProfile(platform="linux"))
    data = tools.cockpit_data()
    assert data["ok"] is True
    assert data["pull_credentials"]["HF_TOKEN"] is True
    assert data["pull_credentials"]["MODELSCOPE_API_TOKEN"] is False
    # Secret value must never appear anywhere in cockpit_data
    assert "hf_secret_never_leaked" not in json.dumps(data)
    assert data["active_wiring"]["ok"] is True
    assert data["active_wiring"]["chat"]["model"] == "qwen3:8b"
    assert data["active_wiring"]["image"]["provider"] == "model-forge"
    assert data["active_wiring"]["image"]["base_url"] == "http://gpu-box.example.com:8080/v1"
    ids = {e["id"] for e in data["engine_catalog"]}
    assert {"llamacpp", "ollama", "vllm", "sglang", "comfyui",
            "strata", "maya", "exo"} <= ids


def test_desktop_gui_headless_smoke():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if node is None:
        return
    repo = Path(__file__).resolve().parent.parent
    try:
        from _repo import repo_root
        repo = repo_root()
    except Exception:
        pass
    smoke = repo / "tests" / "smoke_desktop_gui.js"
    if not smoke.is_file():
        return
    res = subprocess.run([node, str(smoke)], capture_output=True, text=True,
                         cwd=str(repo))
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    assert "ALL DESKTOP GUI SMOKE CHECKS PASSED" in res.stdout

