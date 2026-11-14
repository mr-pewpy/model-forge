"""Handler, schema, hardware, and pull tests.

The ``model_forge_*`` tool handlers are thin JSON wrappers around the pure
planners (covered in test_planners.py) plus lazy imports of Hermes internals
(``plugins.plugin_storage``, ``hermes_cli.*``) and stdlib urllib. We stub those
internals in ``sys.modules`` and point urllib at canned responses so no test
touches the network, the real plugin data dir, or the live Hermes config.
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest

from model_forge import hardware, pull, schemas
from model_forge.broker import EngineLifecycle
from model_forge.registry import models_for
from model_forge.targets import (
    ComputeTarget,
    add_target,
    get_target,
    load_targets,
)

_REMOTE = ComputeTarget(id="gpu-box", label="GPU Box", kind="lan",
                     base_url="http://gpu-box.example.com:8080",
                     vram_gb=48.0, ram_gb=96.0)
_LOCAL = ComputeTarget(id="local", label="This machine",
                       base_url="http://127.0.0.1:8080", kind="local")


def _entry(mid):
    return next(m for m in models_for() if m.id == mid)


# --- shared stubs -----------------------------------------------------------

def _stub_hermes_modules(monkeypatch, data_dir: Path):
    """Point the lazy Hermes imports the handlers reach for at fakes."""
    plugins = types.ModuleType("plugins")
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_data_dir = lambda name: data_dir / name
    plugins.plugin_storage = storage
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)

    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.load_config_readonly = lambda: {}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg_mod)

    # agent.file_safety is the core read-block chokepoint: default it to a
    # no-op (nothing blocked), monkeypatch raise to simulate blocked paths.
    agent_pkg = sys.modules.get("agent") or types.ModuleType("agent")
    fs_mod = types.ModuleType("agent.file_safety")
    fs_mod.raise_if_read_blocked = lambda path: None
    agent_pkg.file_safety = fs_mod
    monkeypatch.setitem(sys.modules, "agent", agent_pkg)
    monkeypatch.setitem(sys.modules, "agent.file_safety", fs_mod)


def _ok_engine(monkeypatch, modalities=("chat",)):
    """Make detect_engines report one available openai-compatible engine."""
    from model_forge.engines import base

    e = base.EngineStatus(id="openai-compatible", display_name="OpenAI-compat",
                          available=True,
                          endpoint="http://gpu-box.example.com:8080/v1",
                          modalities=modalities)
    monkeypatch.setattr(
        "model_forge.tools.detect_engines",
        lambda base_url="": [e],
    )


# --- schemas.py -------------------------------------------------------------

def test_schema_shapes():
    # every tool is named, described, and is a well-formed JSON-schema object
    for name, spec in [("model_forge_status", schemas.STATUS),
                       ("model_forge_browse", schemas.BROWSE),
                       ("model_forge_target_add", schemas.TARGET_ADD),
                       ("model_forge_install", schemas.INSTALL),
                       ("model_forge_generate_media", schemas.GENERATE_MEDIA),
                       ("model_forge_activate", schemas.ACTIVATE),
                       ("model_forge_storage", schemas.STORAGE),
                       ("model_forge_broker", schemas.BROKER)]:
        assert spec["name"] == name
        assert spec["description"].strip()
        assert spec["parameters"]["type"] == "object"
        assert isinstance(spec["parameters"].get("properties", {}), dict)


def test_browse_modality_enum_is_complete():
    from model_forge.registry import MODALITIES

    enum = set(schemas.BROWSE["parameters"]["properties"]["modality"]["enum"])
    assert set(MODALITIES) == enum


def test_activate_requires_model_id():
    assert schemas.ACTIVATE["parameters"]["required"] == ["model_id"]
    assert schemas.GENERATE_MEDIA["parameters"]["required"] == ["modality"]
    assert schemas.TARGET_ADD["parameters"]["required"] == ["id", "base_url"]


def test_storage_schema_shape():
    spec = schemas.STORAGE
    props = spec["parameters"]["properties"]
    assert props["action"]["enum"] == ["list", "delete"]
    assert "entry" in props  # entry identifier or exact path for delete
    assert spec["parameters"].get("required", []) == []  # list is the default


# --- hardware.py ------------------------------------------------------------

def test_probe_hardware_returns_profile(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    prof = hardware.probe_hardware()
    assert prof.gpus == ()
    assert prof.total_vram_gb == 0.0
    # platform labels may carry an arch suffix (e.g. "darwin-arm64")
    assert prof.platform.split("-")[0] in ("linux", "darwin", "macos",
                                           "windows")
    assert prof.ram_gb >= 0.0


def test_probe_hardware_parses_nvidia(monkeypatch):
    import subprocess

    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    fake = subprocess.CompletedProcess(args=hardware._NVIDIA_SMI,
                                       stdout="NVIDIA A100, 40960\n",
                                       stderr="", returncode=0)
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: fake)
    prof = hardware.probe_hardware()
    assert len(prof.gpus) == 1
    assert prof.gpus[0]["name"] == "NVIDIA A100"
    assert prof.gpus[0]["vram_gb"] == 40.0
    assert prof.total_vram_gb == 40.0


def test_probe_hardware_bad_nvidia_lines(monkeypatch):
    import subprocess

    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    fake = subprocess.CompletedProcess(
        args=hardware._NVIDIA_SMI,
        stdout="NotA,GPU,xx\nbad line without comma\n",
        stderr="", returncode=0)
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: fake)
    prof = hardware.probe_hardware()
    # no line parses into a float VRAM -> no GPUs
    assert prof.gpus == ()


def test_probe_hardware_nvidia_error(monkeypatch):
    import subprocess

    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise subprocess.SubprocessError("nvidia-smi died")
    monkeypatch.setattr(hardware.subprocess, "run", boom)
    prof = hardware.probe_hardware()
    assert prof.gpus == ()


def test_ram_gb_reads_sysconf(monkeypatch):
    monkeypatch.setattr(hardware.os, "sysconf",
                        lambda key: {"SC_PHYS_PAGES": 4194304,
                                     "SC_PAGE_SIZE": 4096}[key])
    assert hardware._ram_gb() == pytest.approx(16.0)


def test_ram_gb_degrades_to_zero(monkeypatch):
    def raise_oserror(key):
        raise OSError("no sysconf here")
    monkeypatch.setattr(hardware.os, "sysconf", raise_oserror)
    assert hardware._ram_gb() == 0.0


# --- pull.py ----------------------------------------------------------------

def test_download_plan_validation():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        assert pull.download_plan("https://x/y.bin", str(root / "y.bin"),
                                  allowed_roots=[root])["ok"] is True
    assert pull.download_plan("ftp://x/y", "/abs/y.bin")["ok"] is False
    assert pull.download_plan("https://x/y.bin", "rel/y.bin")["ok"] is False


def test_download_plan_confinement(tmp_path):
    root = tmp_path / "downloads"
    root.mkdir()
    # inside the root: fine
    ok = pull.download_plan("https://x/y.bin", str(root / "a" / "y.bin"),
                            allowed_roots=[root])
    assert ok["ok"] is True
    # outside the root: refused (this is the arbitrary-overwrite fix)
    out = pull.download_plan("https://x/y.bin", str(tmp_path / "victim.bin"),
                             allowed_roots=[root])
    assert out["ok"] is False and "dest must be inside" in out["error"]
    # traversal after the root is resolved and refused too
    trav = pull.download_plan("https://x/y.bin",
                              str(root / ".." / "victim.bin"),
                              allowed_roots=[root])
    assert trav["ok"] is False


def test_download_plan_refuses_symlink_dest(tmp_path):
    root = tmp_path / "downloads"
    root.mkdir()
    outside = tmp_path / "victim.bin"
    outside.write_bytes(b"precious")
    link = root / "y.bin"
    link.symlink_to(outside)
    res = pull.download_plan("https://x/y.bin", str(link),
                             allowed_roots=[root])
    assert res["ok"] is False


def test_download_plan_refuses_symlink_parent_escape(tmp_path):
    root = tmp_path / "downloads"
    root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    link_dir = root / "sub"
    link_dir.symlink_to(outside_dir)
    res = pull.download_plan("https://x/y.bin", str(link_dir / "y.bin"),
                             allowed_roots=[root])
    assert res["ok"] is False


def test_download_plan_user_roots(tmp_path):
    root = tmp_path / "downloads"
    root.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    # a user-configured root accepts its subtree
    ok = pull.download_plan("https://x/y.bin", str(models / "y.bin"),
                            allowed_roots=[root, models])
    assert ok["ok"] is True


def test_job_status_invalid_ids(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    # job_id is model-supplied: only the 12-hex shape start_download makes
    for bad in ("../../auth", "", "x" * 12, "AABBCCDD0011",
                "aabbccdd001", "aabbccdd00112", "aabbccdd001a/../x",
                "aabbccdd001a.json"):
        got = pull.job_status(bad)
        assert got["state"] == "unknown", bad
        assert "invalid job id" in got["error"], bad


def test_job_status_missing_valid_id(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    got = pull.job_status("aabbccdd0011")
    assert got["state"] == "unknown"
    assert "aabbccdd0011" in got["error"]


def test_start_download_validates(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    # bad url is rejected before any thread is spawned
    res = pull.start_download("not-a-url", "/tmp/x.bin")
    assert res["ok"] is False


def test_start_download_and_poll(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    # fake the network: urlopen returns a body of b"hello"
    import io
    import urllib.request

    class _Resp(io.BytesIO):
        status = 200

        def __init__(self, data):
            super().__init__(data)
            self.headers = {"Content-Length": "5"}

        def read(self, n=-1):
            return super().read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _Resp(b"hello"))
    # dest must sit inside the stubbed plugin downloads root
    dest = tmp_path / "model-forge" / "downloads" / "y.bin"
    res = pull.start_download("https://huggingface.co/x/y.bin", str(dest))
    assert res["ok"] is True and res["job_id"]
    import time
    for _ in range(100):
        st = pull.job_status(res["job_id"])
        if st.get("state") in ("done", "error"):
            break
        time.sleep(0.02)
    assert st["state"] == "done"
    assert st["done_bytes"] == 5
    assert dest.read_bytes() == b"hello"


def test_run_sha256_mismatch(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    import io
    import urllib.request

    class _Resp(io.BytesIO):
        status = 200

        def __init__(self, data):
            super().__init__(data)
            self.headers = {"Content-Length": "5"}

        def read(self, n=-1):
            return super().read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(b"hello"))
    dest = str(tmp_path / "m.bin")
    pull._run("aabbccdd0011", "https://huggingface.co/x/y.bin", dest,
              "0" * 64)  # wrong digest
    st = pull.job_status("aabbccdd0011")
    assert st["state"] == "error"
    assert "sha256 mismatch" in st["error"]
    assert not Path(dest).exists()


def test_run_resume_range(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    import io
    import urllib.request

    seen = {}

    class _Resp(io.BytesIO):
        status = 206

        def __init__(self, data):
            super().__init__(data)
            self.headers = {"Content-Length": "3"}

        def read(self, n=-1):
            return super().read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def _fake(req, timeout=None):
        seen["range"] = req.headers.get("Range")
        return _Resp(b"llo")

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    dest = str(tmp_path / "r.bin")
    Path(dest + ".part").write_bytes(b"he")  # pre-existing partial
    pull._run("aabbccdd0022", "https://huggingface.co/x/y.bin", dest, "")
    assert seen["range"] == "bytes=2-"
    assert Path(dest).read_bytes() == b"hello"


# --- tools.py: target_add ---------------------------------------------------

def test_target_add_validation_and_persist(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    # id required / not 'local'
    err = json.loads(tools.model_forge_target_add({"id": "local",
                                                   "base_url": "http://x:1"}))
    assert "error" in err
    # base_url must be http(s)
    err = json.loads(tools.model_forge_target_add(
        {"id": "gpu-box", "base_url": "ftp://x:1"}))
    assert "error" in err
    # happy path persists to the stubbed data dir
    out = json.loads(tools.model_forge_target_add(
        {"id": "gpu-box", "label": "GPU Box",
         "base_url": "http://gpu-box.example.com:8080", "vram_gb": "48"}))
    assert out["added"] == "gpu-box"
    assert any(t["id"] == "gpu-box" for t in out["targets"])


# --- tools.py: browse -------------------------------------------------------

def test_browse_unknown_target(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    out = json.loads(tools.model_forge_browse({"target": "nope",
                                               "modality": "image"}))
    assert "error" in out


def test_browse_lists_and_prices(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    # point get_target at our temp store by monkeypatching the handler's view
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    out = json.loads(tools.model_forge_browse({"target": "gpu-box",
                                               "modality": "image"}))
    assert out["target"] == "gpu-box"
    # budget = declared VRAM + (RAM - 8 GB) = 48 + (96 - 8) = 136
    assert out["hardware_budget_gb"] == 136.0
    assert isinstance(out["models"], list)
    for m in out["models"]:
        assert "id" in m and "fit" in m and "modality" in m


# --- tools.py: status -------------------------------------------------------

def test_status_lists_targets_and_engines(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "load_targets",
                        lambda: load_targets(base_dir=tmp_path))
    _ok_engine(monkeypatch, ("image", "video"))
    out = json.loads(tools.model_forge_status())
    assert isinstance(out["targets"], list)
    assert out["pull_credentials"] == {"HF_TOKEN": False,
                                       "MODELSCOPE_API_TOKEN": False}
    by_id = {t["id"]: t for t in out["targets"]}
    assert "local" in by_id and "gpu-box" in by_id
    eng = by_id["gpu-box"]["engines"][0]
    assert eng["available"] is True and "image" in eng["modalities"]


def test_status_remote_hw_uses_declared_budget(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "load_targets",
                        lambda: load_targets(base_dir=tmp_path))
    monkeypatch.setattr(tools, "detect_engines", lambda base_url="": [])
    out = json.loads(tools.model_forge_status())
    remote = next(t for t in out["targets"] if t["id"] == "gpu-box")
    # remote target -> declared VRAM, not a live probe
    assert remote["hardware"]["total_vram_gb"] == 48.0
    assert remote["hardware"]["ram_gb"] == 96.0


def test_status_error_path(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    def boom():
        raise RuntimeError("targets exploded")
    monkeypatch.setattr(tools, "load_targets", boom)
    out = json.loads(tools.model_forge_status())
    assert "error" in out and "targets exploded" in out["error"]


# --- tools.py: install ------------------------------------------------------

def test_install_unknown_model(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    out = json.loads(tools.model_forge_install(
        {"model_id": "nope", "target": "local"}))
    assert "error" in out


def test_install_job_id_poll(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    (tmp_path / "model-forge" / "jobs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "model-forge" / "jobs" / "aabbccdd0033.json").write_text(
        json.dumps({"job_id": "aabbccdd0033", "state": "done"}))
    out = json.loads(tools.model_forge_install({"job_id": "aabbccdd0033"}))
    assert out["job_id"] == "aabbccdd0033" and out["state"] == "done"
    # a traversal-shaped job_id is rejected before it reaches the jobs dir
    bad = json.loads(tools.model_forge_install({"job_id": "../../auth"}))
    assert bad["state"] == "unknown" and "invalid job id" in bad["error"]


def test_install_strata_runs_cli(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LOCAL, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    out = json.loads(tools.model_forge_install(
        {"model_id": "qwen3.8-flash-next", "target": "local",
         "engine_model": "deepseek41:ist-das-2.00bpw"}))
    assert out["ok"] is True and out["action"] == "run_cli_on_target"


def test_install_maya_pull(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    # stub the Maya HTTP pull to return a canned job row
    monkeypatch.setattr(tools.maya_adapter, "pull",
                        lambda model, detach=False, base_url="":
                        {"ok": True, "job": "maya-1"})
    out = json.loads(tools.model_forge_install(
        {"model_id": "glm-5.3-flash", "target": "gpu-box"}))
    assert out["action"] == "maya_pull"
    assert out["pull_response"] == {"ok": True, "job": "maya-1"}


def test_install_maya_no_response(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools.maya_adapter, "pull",
                        lambda model, detach=False, base_url="": None)
    out = json.loads(tools.model_forge_install(
        {"model_id": "glm-5.3-flash", "target": "gpu-box"}))
    assert out["ok"] is False and "pull_response" not in out


# --- tools.py: activate -----------------------------------------------------

def test_activate_unknown_model(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    out = json.loads(tools.model_forge_activate(
        {"model_id": "nope", "target": "local"}))
    assert "error" in out


def test_activate_plan_only(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _ok_engine(monkeypatch, ("chat", "image"))
    # the media hint is emitted only when the endpoint serves the route
    monkeypatch.setattr(tools, "route_exists",
                        lambda root, path, timeout=2.0: True)
    out = json.loads(tools.model_forge_activate(
        {"model_id": "glm-5.3-flash", "target": "gpu-box"}))
    assert out["ok"] is True and out["applied"] is False
    assert "hint" in out
    # image modality present AND the route served -> media setup hint
    assert "media_setup" in out and "image_gen" in out["media_setup"]


def test_activate_apply_writes(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _ok_engine(monkeypatch, ("chat",))
    setup = types.ModuleType("hermes_cli.model_setup")
    setup.persist_model_selection = (
        lambda model, provider, base_url="": {"model": model})
    monkeypatch.setitem(sys.modules, "hermes_cli.model_setup", setup)
    out = json.loads(tools.model_forge_activate(
        {"model_id": "glm-5.3-flash", "target": "gpu-box", "apply": True}))
    assert out["applied"] is True
    # engine_model not passed -> the plan's model is the registry entry id
    assert out["saved_model_section"] == {"model": "glm-5.3-flash"}


# --- tools.py: generate_media -----------------------------------------------

def _fake_urlopen(monkeypatch, payload: bytes, status=200, exc=None):
    import io
    import urllib.request

    class _Resp(io.BytesIO):
        def __init__(self, data, status):
            super().__init__(data)
            self.status = status
            self.headers = {}

        def read(self, n=-1):
            return super().read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def _fake(req, timeout=None):
        if exc is not None:
            raise exc
        return _Resp(payload, status)

    monkeypatch.setattr(urllib.request, "urlopen", _fake)


def test_generate_media_unknown_target(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "nope"}))
    assert "error" in out


def test_generate_media_3d_needs_args(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://gpu-box.example.com:8080/jobs"})
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box"}))
    assert out["action"] == "needs_args"


def test_generate_media_3d_submit(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://gpu-box.example.com:8080/jobs"})
    _fake_urlopen(monkeypatch, json.dumps({"id": "3djob-1"}).encode())
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box", "model": "gen/3d",
         "image": "https://example.com/x.png"}))
    assert out["ok"] is True and out["action"] == "submitted"
    assert out["submitted"]["id"] == "3djob-1"


def test_generate_media_3d_submit_failure(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://gpu-box.example.com:8080/jobs"})
    _fake_urlopen(monkeypatch, b"", exc=RuntimeError("conn refused"))
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box", "model": "gen/3d",
         "image": "https://example.com/x.png"}))
    assert out["action"] == "submit_failed"


def test_generate_media_3d_http_error_surfaces_status_and_body(tmp_path,
                                                               monkeypatch):
    import io
    import urllib.error
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://gpu-box.example.com:8080/jobs"})
    exc = urllib.error.HTTPError(
        "http://gpu-box.example.com:8080/jobs", 502, "Bad Gateway",
        hdrs=None, fp=io.BytesIO(b'{"detail":"engine exploded"}'))
    _fake_urlopen(monkeypatch, b"", exc=exc)
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box", "model": "gen/3d",
         "image": "https://example.com/x.png"}))
    # fail closed, but the detail names the HTTP status and a response
    # snippet instead of a generic submit_failed
    assert out["ok"] is False and out["action"] == "submit_failed"
    assert out["http_status"] == 502
    assert "502" in out["error"]
    assert "engine exploded" in out["error"]
    assert out["response_snippet"] == '{"detail":"engine exploded"}'


def test_generate_media_music_http_error_surfaces_status_and_body(tmp_path,
                                                                  monkeypatch):
    import io
    import urllib.error
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit",
                         "submit": "http://gpu-box.example.com:8080"
                                   "/v1/audio/music"})
    exc = urllib.error.HTTPError(
        "http://gpu-box.example.com:8080/v1/audio/music", 500,
        "Internal Server Error", hdrs=None,
        fp=io.BytesIO(b"<html>gateway refused the prompt</html>"))
    _fake_urlopen(monkeypatch, b"", exc=exc)
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "music", "target": "gpu-box", "model": "gen/music",
         "prompt": "lofi"}))
    assert out["ok"] is False and out["action"] == "submit_failed"
    assert out["http_status"] == 500
    assert "500" in out["error"]
    assert "gateway refused the prompt" in out["error"]


def test_generate_media_music_needs_args(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://gpu-box.example.com:8080/audio"})
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "music", "target": "gpu-box"}))
    assert out["action"] == "needs_args"


def test_generate_media_music_done(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://gpu-box.example.com:8080/audio"})
    _fake_urlopen(monkeypatch, json.dumps({"url": "http://x/a.mp3"}).encode())
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "music", "target": "gpu-box", "model": "gen/music",
         "prompt": "lofi"}))
    assert out["ok"] is True and out["action"] == "done"
    assert out["result"]["url"] == "http://x/a.mp3"
    # the synchronous wait is stated in the output
    assert "blocked" in out["note"]


def test_generate_media_route_missing(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": False, "action": "route_missing"})
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box", "model": "gen/3d",
         "image": "https://example.com/x.png"}))
    assert out["action"] == "route_missing"


# --- tools.py: _as_data_url (confined) --------------------------------------

def test_as_data_url_passthrough():
    from model_forge import tools

    assert tools._as_data_url("https://x/y.png") == "https://x/y.png"
    assert tools._as_data_url("data:image/png;base64,AA") \
        == "data:image/png;base64,AA"


def _png(tmp_path, name="x.png", data=None):
    p = tmp_path / name
    p.write_bytes(data if data is not None
                  else b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    return p


def test_as_data_url_local_file(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    p = _png(tmp_path)
    got = tools._as_data_url(str(p))
    assert got.startswith("data:image/png;base64,")
    assert base64.b64decode(got.split(",", 1)[1]) == p.read_bytes()


def test_as_data_url_jpeg_and_webp(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    j = tmp_path / "x.jpg"
    j.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 8)
    assert tools._as_data_url(str(j)).startswith("data:image/jpeg;base64,")
    w = tmp_path / "x.webp"
    w.write_bytes(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4)
    assert tools._as_data_url(str(w)).startswith("data:image/webp;base64,")


def test_as_data_url_missing_path_errors(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    # the old code passed raw strings through — that let any file be named
    with pytest.raises(ValueError):
        tools._as_data_url("/does/not/exist/x.png")


def test_as_data_url_blocked_path_errors(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    import agent.file_safety as fs

    def blocked(path):
        raise ValueError("read blocked: secrets area")
    monkeypatch.setattr(fs, "raise_if_read_blocked", blocked)
    from model_forge import tools

    with pytest.raises(ValueError, match="read blocked"):
        tools._as_data_url(str(tmp_path / "auth.json"))


def test_as_data_url_suffix_and_magic_enforced(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    # wrong suffix
    bad_suffix = tmp_path / "x.txt"
    bad_suffix.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="unsupported image type"):
        tools._as_data_url(str(bad_suffix))
    # right suffix, wrong content (magic bytes)
    fake = _png(tmp_path, "fake.png", data=b"#!/bin/sh\nrm -rf ~\n")
    with pytest.raises(ValueError, match="magic bytes"):
        tools._as_data_url(str(fake))


def test_as_data_url_size_cap(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    big = _png(tmp_path, "big.png",
               data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024)
    monkeypatch.setattr(tools, "_MAX_IMAGE_BYTES", 512)
    with pytest.raises(ValueError, match="too large"):
        tools._as_data_url(str(big))


def test_as_data_url_file_scheme_validated(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    p = _png(tmp_path)
    got = tools._as_data_url("file://" + str(p))
    assert got.startswith("data:image/png;base64,")
    # file:// does not smuggle past validation
    with pytest.raises(ValueError):
        tools._as_data_url("file:///does/not/exist/x.png")


def test_generate_media_invalid_image(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://gpu-box.example.com:8080/jobs"})
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box", "model": "gen/3d",
         "image": "/etc/shadow"}))
    assert out["action"] == "invalid_image"


# --- tools.py: policy_hook (approval gate) ----------------------------------

def test_policy_hook_target_add_requires_approval():
    from model_forge import tools

    d = tools.policy_hook("model_forge_target_add",
                          {"id": "evil", "label": "evil",
                           "base_url": "https://evil.example:8080"})
    assert d["action"] == "approve"
    assert "https://evil.example:8080" in d["message"]
    assert d["rule_key"] == "model-forge:target_add"


def test_policy_hook_activate_apply_requires_approval():
    from model_forge import tools

    d = tools.policy_hook("model_forge_activate",
                          {"model_id": "m1", "target": "gpu-box",
                           "apply": True})
    assert d["action"] == "approve"
    assert "model_id" in d["message"]
    assert d["rule_key"] == "model-forge:activate"


def test_policy_hook_plan_only_passes():
    from model_forge import tools

    # plan-only activate and every other call need no approval
    assert tools.policy_hook("model_forge_activate",
                             {"model_id": "m1"}) is None
    assert tools.policy_hook("model_forge_activate",
                             {"model_id": "m1", "apply": False}) is None
    assert tools.policy_hook("model_forge_status", {}) is None
    assert tools.policy_hook("model_forge_browse", {}) is None
    assert tools.policy_hook("terminal", {"command": "ls"}) is None
    assert tools.policy_hook("model_forge_install", {"model_id": "m"}) is None


def test_policy_hook_storage_delete_requires_approval(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    downloads = tmp_path / "model-forge" / "downloads"
    downloads.mkdir(parents=True)
    f = downloads / "weights.bin"
    f.write_bytes(b"12345")
    from model_forge import tools

    # delete escalates with the exact path and size being deleted
    for ref in (str(f), "plugin-downloads:weights.bin"):
        d = tools.policy_hook("model_forge_storage",
                              {"action": "delete", "entry": ref})
        assert d["action"] == "approve", ref
        assert str(f) in d["message"], ref
        assert "5 bytes" in d["message"], ref
        assert d["rule_key"] == "model-forge:storage_delete"
    # an unlisted ref still escalates (fail closed); the handler refuses it
    d = tools.policy_hook("model_forge_storage",
                          {"action": "delete", "entry": "nope"})
    assert d["action"] == "approve"
    assert "nope" in d["message"]


def test_policy_hook_storage_list_passes():
    from model_forge import tools

    # listing is read-only: no approval, and no storage walk either
    assert tools.policy_hook("model_forge_storage", {}) is None
    assert tools.policy_hook("model_forge_storage",
                             {"action": "list"}) is None


# --- honesty fixes: required args, job_id shape, fresh strings, modality ----

def test_install_and_activate_require_model_id(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    # empty args must say what is required — never leak a literal 'None'
    out = json.loads(tools.model_forge_install({}))
    assert out["error"] == "model_id required (or job_id to poll a download)"
    assert "None" not in json.dumps(out)
    out = json.loads(tools.model_forge_install({"model_id": ""}))
    assert "model_id required" in out["error"]
    assert "None" not in json.dumps(out)
    out = json.loads(tools.model_forge_activate({}))
    assert "model_id required" in out["error"]
    assert "None" not in json.dumps(out)


def test_generate_media_3d_job_id_is_shape_validated(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://gpu-box.example.com:8080/jobs"})

    def _no_call(req, timeout=None):
        raise AssertionError("an invalid job_id must never reach the target")

    monkeypatch.setattr("urllib.request.urlopen", _no_call)
    for bad in ("../secret", "..", "a/b", "x" * 129, "id 1", "ünicode",
                "job;rm", "a%2e%2e", "-bad\u2013dash"):
        out = json.loads(tools.model_forge_generate_media(
            {"modality": "3d", "target": "gpu-box", "job_id": bad}))
        assert out["ok"] is False and out["action"] == "invalid_job_id", bad
        assert "job_id" in out["error"]
    # a well-formed id (letters, digits, _ and -) is forwarded and polled
    _fake_urlopen(monkeypatch, json.dumps({"status": "done"}).encode())
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box", "job_id": "job-1_A"}))
    assert out["job"] == {"status": "done"}


def test_generate_media_music_audio_bytes_saved(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit",
                         "submit": "http://gpu-box.example.com:8080"
                                   "/v1/audio/music"})
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 8
    _fake_urlopen(monkeypatch, wav)
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "music", "target": "gpu-box", "model": "gen/music",
         "prompt": "lofi"}))
    # audio bytes are saved under the plugin media dir, not JSON-decoded
    assert out["ok"] is True and out["action"] == "done"
    saved = Path(out["audio"])
    assert saved.is_file() and saved.read_bytes() == wav
    assert str(tmp_path / "model-forge" / "media") in out["audio"]
    assert saved.suffix == ".wav"
    assert out["bytes"] == len(wav)
    assert "blocked" in out["note"] and str(saved) in out["note"]
    # other non-JSON bodies are saved too (honest .bin, no JSONDecodeError)
    _fake_urlopen(monkeypatch, b"<html>gateway error page</html>")
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "sfx", "target": "gpu-box", "model": "gen/sfx",
         "prompt": "boom"}))
    assert out["ok"] is True and out["action"] == "done"
    saved = Path(out["audio"])
    assert saved.suffix == ".bin"
    assert saved.read_bytes() == b"<html>gateway error page</html>"


def test_browse_fresh_falsey_strings_skip_network(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "detect_engines", lambda base_url="": [])

    def no_network(*a, **k):
        raise AssertionError("the live layer must not run for a falsey arg")

    monkeypatch.setattr("model_forge.discovery.fresh_models", no_network)
    for falsey in (False, "false", "FALSE", "0", "no", " No "):
        out = json.loads(tools.model_forge_browse(
            {"target": "gpu-box", "modality": "3d", "fresh": falsey}))
        assert out["live_discovery"] == "disabled", falsey
        assert out["fresh"] == []
    # a truthy string still enables the live layer
    monkeypatch.setattr(
        "model_forge.discovery.fresh_models",
        lambda modality=None: {"live_discovery": "ok", "models": []})
    out = json.loads(tools.model_forge_browse(
        {"target": "gpu-box", "modality": "3d", "fresh": "true"}))
    assert out["live_discovery"] == "ok"


def test_browse_invalid_modality_names_valid_ones(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    out = json.loads(tools.model_forge_browse({"modality": "banana"}))
    assert "error" in out and "models" not in out
    assert "banana" in out["error"]
    assert "chat" in out["error"] and "music" in out["error"]


def test_activate_apply_refuses_placeholder_ids(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _ok_engine(monkeypatch, ("chat",))

    def _forbidden(*a, **k):
        raise AssertionError("a placeholder id must never be persisted")

    setup = types.ModuleType("hermes_cli.model_setup")
    setup.persist_model_selection = _forbidden
    monkeypatch.setitem(sys.modules, "hermes_cli.model_setup", setup)
    # catalog marker: refuse apply with an honest message
    out = json.loads(tools.model_forge_activate(
        {"model_id": "core-gguf-catalog", "target": "gpu-box", "apply": True}))
    assert out["ok"] is False
    assert out["action"] == "placeholder_not_activatable"
    assert "catalog marker" in out["detail"]
    assert out["applied"] is False
    # stub entries are refused the same way
    out = json.loads(tools.model_forge_activate(
        {"model_id": "world-model-slot", "target": "gpu-box", "apply": True}))
    assert out["ok"] is False and out["action"] == "placeholder_not_activatable"
    assert "placeholder" in out["detail"]
    assert out["applied"] is False
    assert "hint" not in out   # no 'apply=true' nudge on a refusal


def test_activate_media_setup_gated_on_route_probe(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    # engine advertises image/video but the endpoint serves neither route
    # (ComfyUI-style modality listing)
    _ok_engine(monkeypatch, ("chat", "image", "video"))
    monkeypatch.setattr(tools, "route_exists",
                        lambda root, path, timeout=2.0: False)
    out = json.loads(tools.model_forge_activate(
        {"model_id": "glm-5.3-flash", "target": "gpu-box"}))
    assert out["ok"] is True
    assert "media_setup" not in out
    assert "/v1/images/generations" in out["media_setup_note"]
    assert "/v1/videos/generations" in out["media_setup_note"]
    # partial serving: only the image route answers the probe
    monkeypatch.setattr(
        tools, "route_exists",
        lambda root, path, timeout=2.0: path == "/v1/images/generations")
    out = json.loads(tools.model_forge_activate(
        {"model_id": "glm-5.3-flash", "target": "gpu-box"}))
    assert "image_gen" in out["media_setup"]
    assert "video_gen" not in out.get("media_setup", {})
    assert "/v1/videos/generations" in out["media_setup_note"]


# --- schema honesty ---------------------------------------------------------

def test_generate_media_schema_honest_about_gateway():
    desc = schemas.GENERATE_MEDIA["description"]
    assert "media gateway" in desc
    assert "/v1/videos/generations" in desc
    assert "/v1/audio/music" in desc and "/v1/audio/sfx" in desc
    # the ComfyUI-native flow is stated as a known limit, not implied support
    assert "ComfyUI" in desc and "not implemented" in desc


def test_install_schema_example_is_a_real_repo_path():
    desc = schemas.INSTALL["parameters"]["properties"]["engine_model"][
        "description"]
    assert "model.safetensors" not in desc    # the 404-prone example is gone
    assert "diffusion_models/your-weights.safetensors" in desc
    assert "omit engine_model" in desc


# --- tools.py: engine_install (approved vendor installers) ------------------

_OLLAMA_CMD = "curl -fsSL https://ollama.com/install.sh | sh"
_STRATA_CMD = ("curl -fsSL https://github.com/Niko1221/Strata/releases/"
               "latest/download/install.sh | sh")


def _installer_spy(monkeypatch, result=None, raise_exc=None):
    """Fake installer runner: records the exact commands it is asked to run.
    Tests never run a real installer."""
    calls = []

    def runner(command, timeout_s):
        calls.append(command)
        if raise_exc is not None:
            raise raise_exc
        return result or {"returncode": 0, "output_tail": "install ok"}

    monkeypatch.setattr("model_forge.tools._run_installer", runner)
    return calls


def _simulate_approval_gate(tool_name, args, decision, dispatch):
    """Emulate the Hermes human-approval gate contract for a pre_tool_call
    approve directive: only an explicit user approval lets the call reach
    the handler; denial, timeout or a gate error blocks the call BEFORE the
    handler runs (fail closed) — nothing executes."""
    from model_forge import tools
    directive = tools.policy_hook(tool_name, args)
    if directive is not None and directive.get("action") == "approve" \
            and decision != "approve":
        return {"blocked": True, "directive": directive}
    return {"blocked": False, "result": dispatch(args)}


def _poll_until_done(job_id, rounds=100):
    import time
    from model_forge import tools
    st = {}
    for _ in range(rounds):
        st = json.loads(tools.model_forge_install({"job_id": job_id}))
        if st.get("state") in ("done", "error", "timeout"):
            return st
        time.sleep(0.02)
    return st


def test_engine_install_plan_shows_exact_command_before_execute(tmp_path,
                                                                monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    calls = _installer_spy(monkeypatch)
    out = json.loads(tools.model_forge_install(
        {"action": "engine_install", "engine": "ollama"}))
    # plan first: the exact vendor command, and nothing ran
    assert out["ok"] is True and out["action"] == "vendor_installer"
    assert out["command"] == _OLLAMA_CMD
    assert out["executed"] is False and "job_id" not in out
    assert "execute=true" in out["hint"] and "denial" in out["hint"]
    assert calls == []
    # planning is read-only: no approval gate for it
    assert tools.policy_hook("model_forge_install",
                             {"action": "engine_install",
                              "engine": "ollama"}) is None
    assert tools.policy_hook("model_forge_install",
                             {"action": "engine_install", "engine": "ollama",
                              "execute": False}) is None


def test_policy_hook_engine_install_approve_directive():
    from model_forge import tools
    d = tools.policy_hook("model_forge_install",
                          {"action": "engine_install", "engine": "ollama",
                           "execute": True})
    assert d["action"] == "approve"
    assert d["rule_key"] == "model-forge:engine_install"
    assert _OLLAMA_CMD in d["message"]        # the exact command shown
    assert "ollama" in d["message"]
    # the Strata documented install command appears verbatim as well
    d = tools.policy_hook("model_forge_install",
                          {"action": "engine_install", "engine": "strata",
                           "execute": True})
    assert d["rule_key"] == "model-forge:engine_install"
    assert _STRATA_CMD in d["message"]
    # a model-install call still needs no approval (historical contract)
    assert tools.policy_hook("model_forge_install",
                             {"model_id": "m"}) is None


def test_engine_install_gate_denial_fails_closed(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    calls = _installer_spy(monkeypatch)
    out = _simulate_approval_gate(
        "model_forge_install",
        {"action": "engine_install", "engine": "ollama", "execute": True},
        "deny", tools.model_forge_install)
    assert out["blocked"] is True
    assert out["directive"]["rule_key"] == "model-forge:engine_install"
    assert calls == []                        # denial runs nothing
    jobs_dir = tmp_path / "model-forge" / "jobs"
    assert not jobs_dir.exists() or list(jobs_dir.glob("*.json")) == []


def test_engine_install_gate_timeout_fails_closed(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    calls = _installer_spy(monkeypatch)
    for decision in ("timeout", "gate_error"):
        out = _simulate_approval_gate(
            "model_forge_install",
            {"action": "engine_install", "engine": "strata", "execute": True},
            decision, tools.model_forge_install)
        assert out["blocked"] is True, decision
    assert calls == []                        # a timed-out gate runs nothing


def test_engine_install_approved_flow_runs_and_polls(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    from model_forge.engines.base import EngineStatus
    calls = _installer_spy(monkeypatch)
    monkeypatch.setattr(
        tools, "detect_engines",
        lambda base_url="": [EngineStatus(
            id="ollama", display_name="Ollama", available=True,
            endpoint="http://127.0.0.1:11434", detail="running")])
    out = _simulate_approval_gate(
        "model_forge_install",
        {"action": "engine_install", "engine": "ollama", "execute": True},
        "approve", tools.model_forge_install)
    assert out["blocked"] is False
    res = json.loads(out["result"])
    assert res["ok"] is True and res["action"] == "engine_install_started"
    assert res["executed"] is True and res["job_id"]
    assert res["command"] == _OLLAMA_CMD
    # poll through the shared jobs store (model_forge_install job_id)
    st = _poll_until_done(res["job_id"])
    assert st["state"] == "done"
    assert st["kind"] == "engine_install" and st["engine"] == "ollama"
    assert st["command"] == _OLLAMA_CMD and st["returncode"] == 0
    # completion re-runs engine detection and reports it honestly
    assert st["detection"]["detected"] is True
    assert calls == [_OLLAMA_CMD]             # the approved command only


def test_engine_install_failure_and_timeout_reported_honestly(tmp_path,
                                                              monkeypatch):
    import subprocess
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    # a non-zero installer exit is an error, never a false 'done'
    _installer_spy(monkeypatch, result={"returncode": 2,
                                       "output_tail": "permission denied"})
    res = json.loads(tools.model_forge_install(
        {"action": "engine_install", "engine": "ollama", "execute": True}))
    st = _poll_until_done(res["job_id"])
    assert st["state"] == "error" and "2" in st["error"]
    assert st["output_tail"] == "permission denied"
    assert "detection" not in st              # nothing is assumed installed
    # the hard timeout kills the job record as 'timeout', not 'done'
    _installer_spy(monkeypatch,
                   raise_exc=subprocess.TimeoutExpired(cmd="installer",
                                                       timeout=1.0))
    res = json.loads(tools.model_forge_install(
        {"action": "engine_install", "engine": "strata", "execute": True}))
    st = _poll_until_done(res["job_id"])
    assert st["state"] == "timeout"
    assert "hard timeout" in st["error"] and "nothing is assumed installed" \
        in st["error"]


def test_engine_install_llamacpp_guides_to_core_runtime(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    calls = _installer_spy(monkeypatch)
    monkeypatch.setattr(
        tools, "_core_llamacpp_status",
        lambda: {"present": True,
                 "signals": ["the hermes CLI is on PATH (Hermes core is "
                             "installed)"]})
    out = json.loads(tools.model_forge_install(
        {"action": "engine_install", "engine": "llamacpp"}))
    assert out["ok"] is True and out["action"] == "use_core_local_models"
    assert out["executed"] is False and "command" not in out
    assert "never installs llama.cpp binaries" in out["detail"]
    assert "Local Models" in out["detail"] and "hermes model" in out["detail"]
    assert out["core_runtime"]["present"] is True
    # even with execute=true: no command exists, nothing executes, and the
    # gate is not asked to approve an installer run
    assert tools.policy_hook("model_forge_install",
                             {"action": "engine_install", "engine": "llamacpp",
                              "execute": True}) is None
    out = json.loads(tools.model_forge_install(
        {"action": "engine_install", "engine": "llamacpp", "execute": True}))
    assert out["executed"] is False and calls == []
    # no managed runtime detected: honest absence, still core guidance
    monkeypatch.setattr(tools, "_core_llamacpp_status",
                        lambda: {"present": False, "signals": []})
    out = json.loads(tools.model_forge_install(
        {"action": "engine_install", "engine": "llamacpp"}))
    assert "No managed llama.cpp runtime was detected" in out["detail"]
    assert "Local Models" in out["detail"]
    assert calls == []


def test_engine_install_unknown_engine_never_runs(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    calls = _installer_spy(monkeypatch)
    for engine in ("banana", "ollama; rm -rf /", ""):
        out = json.loads(tools.model_forge_install(
            {"action": "engine_install", "engine": engine, "execute": True}))
        assert out["ok"] is False, engine
        assert out["action"] in ("unknown_engine", "needs_engine"), engine
        # an unknown or malformed engine id escalates nothing and runs nothing
        assert tools.policy_hook("model_forge_install",
                                 {"action": "engine_install", "engine": engine,
                                  "execute": True}) is None
    assert calls == []
    # unknown actions name the valid ones instead of falling through
    out = json.loads(tools.model_forge_install({"action": "banana"}))
    assert "error" in out and "engine_install" in out["error"]


def test_generate_media_3d_job_poll_http_error_surfaces_status_and_body(
        tmp_path, monkeypatch):
    import io
    import urllib.error
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://gpu-box.example.com:8080/jobs"})
    exc = urllib.error.HTTPError(
        "http://gpu-box.example.com:8080/jobs/3djob-1", 503,
        "Service Unavailable", hdrs=None,
        fp=io.BytesIO(b'{"detail":"queue is full"}'))
    _fake_urlopen(monkeypatch, b"", exc=exc)
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box", "job_id": "3djob-1"}))
    # the poll path surfaces the HTTP status and a response snippet, the
    # same honesty as the submit paths (not a generic error payload)
    assert out["ok"] is False and out["action"] == "poll_failed"
    assert out["http_status"] == 503
    assert out["response_snippet"] == '{"detail":"queue is full"}'
    assert "503" in out["error"] and "queue is full" in out["error"]


# --- model_forge_broker: weight residency + workflow scopes ------------------

class _RecordingAdapter(EngineLifecycle):
    """Engine lifecycle adapter that records every call into a shared log."""

    def __init__(self, engine_id, calls):
        self.engine_id = engine_id
        self.calls = calls

    def ensure_loaded(self, model_id, **kwargs):
        self.calls.append(("ensure_loaded", self.engine_id, model_id))
        return {"ok": True, "reason": "warmed"}

    def release(self, model_id):
        self.calls.append(("release", self.engine_id, model_id))
        return {"ok": True, "reason": "evicted"}

    def probe(self):
        return {"ok": True, "state": "observed", "resident": []}


def _residency_stack(target, calls, budget=48.0, adapters=None):
    """A real residency stack (bookings, policy, scopes) with recording
    engine adapters — everything through the broker library's public API."""
    from model_forge import tools
    from model_forge.broker import EvictionPlanner, Orchestrator, ResidencyLedger
    bookings = tools.bookings_from_registry()
    ledger = ResidencyLedger(bookings=bookings)
    ledger.set_budget(target.id, budget)
    if adapters is None:
        adapters = {engine: _RecordingAdapter(engine, calls)
                    for engine in ("ollama", "comfyui", "media-gateway")}
    planner = EvictionPlanner(ledger)
    orchestrator = Orchestrator(ledger, planner, adapters)
    return {"target_id": target.id, "ledger": ledger, "planner": planner,
            "orchestrator": orchestrator, "bookings": bookings,
            "budget_gb": budget}


def _book_chat_resident(stack):
    """A claim-time chat resident with its own honest booking (catalog ids
    cover the registry; this fixture models a running chat model)."""
    spec = {"vram_gb": 6.0, "source": "catalog-vram-floor", "load_s": None}
    stack["ledger"].book("chat-a", spec)
    stack["bookings"]["chat-a"] = spec
    stack["ledger"].record_resident("gpu-box", "chat-a", engine="ollama",
                                    lane="chat")


def _broker_env(monkeypatch, tmp_path, stack):
    """Shared handler environment: remote target, isolated scope state, and
    the residency stack behind the broker_stack seam."""
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "broker_stack", lambda target, **kw: stack)
    monkeypatch.setattr(tools, "_ACTIVE_SCOPES", {})
    monkeypatch.setattr(tools, "_MEDIA_SCOPE_JOBS", {})
    return tools


def test_register_wires_broker_tool_and_hook(tmp_path, monkeypatch):
    import importlib.util
    init_path = (Path(__file__).resolve().parent.parent
                 / "model-forge" / "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "model_forge_plugin_pkg", init_path,
        submodule_search_locations=[str(init_path.parent)])
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "model_forge_plugin_pkg", mod)
    spec.loader.exec_module(mod)

    class _Ctx:
        def __init__(self):
            self.tools = {}
            self.hooks = []

        def register_tool(self, name, toolset=None, schema=None, handler=None):
            self.tools[name] = (toolset, schema, handler)

        def register_hook(self, name, handler):
            self.hooks.append((name, handler))

    ctx = _Ctx()
    mod.register(ctx)
    assert len(ctx.tools) == 8
    toolset, schema, handler = ctx.tools["model_forge_broker"]
    assert toolset == "model_forge"
    assert schema is mod.model_forge.schemas.BROKER
    assert callable(handler)
    assert ("pre_tool_call", mod.model_forge.tools.policy_hook) in ctx.hooks
    # the catalog manifest lists the tool too
    manifest = (init_path.parent / "plugin.yaml").read_text()
    assert "model_forge_broker" in manifest


def test_broker_schema_shape_is_verbs_not_knobs():
    spec = schemas.BROKER
    props = spec["parameters"]["properties"]
    assert spec["name"] == "model_forge_broker"
    assert props["action"]["enum"] == ["status", "claim", "release", "plan",
                                      "bench"]
    assert spec["parameters"]["required"] == ["action"]
    # minimal, self-explanatory surface: the five verbs plus what to act on
    assert set(props) == {"action", "model_id", "modality", "scope_id",
                         "target"}
    assert "unload_policy" not in spec["description"]
    # bench is a verb with a model id, nothing more — no window knobs
    assert "bench" in spec["description"]
    assert "bench" in props["action"]["description"]


def test_bookings_helper_maps_model_entry_footprints():
    from model_forge import tools
    from model_forge.broker import ResidencyLedger
    from model_forge.registry import ModelEntry
    entries = (
        ModelEntry(id="fx-img", name="FX Image", modality="image",
                   engines=("vllm",), source="x/fx-img", size_gb=20.0,
                   vram_gb_min=16.0),
        ModelEntry(id="fx-moe", name="FX MoE", modality="chat",
                   engines=("maya",), source="x/fx-moe", size_gb=66.0,
                   ram_gb_min=64.0, engine_managed_fit=True),
    )
    bookings = tools.bookings_from_registry(entries)
    # the published VRAM floor becomes the nominal booked ceiling, honestly
    # labeled as a published floor — never as a measured peak
    assert bookings["fx-img"] == {"vram_gb": 16.0,
                                 "source": tools.BOOKING_SOURCE,
                                 "load_s": None}
    # no declared floor -> stays unbooked: a guessed 0 GB would be a guessed
    # fit, and the planner refuses to gate it until a measured peak lands
    assert "fx-moe" not in bookings
    # the dict is exactly what the residency layer takes
    ledger = ResidencyLedger(bookings=bookings)
    assert ledger.booked_vram_gb("fx-img") == 16.0
    assert ledger.booked_vram_gb("fx-moe") is None
    # the full catalog books every curated entry with a declared floor, and
    # only those — engine-managed entries and stubs stay unbooked
    all_bookings = tools.bookings_from_registry()
    from model_forge.registry import models_for
    assert set(all_bookings) == {m.id for m in models_for()
                                 if float(m.vram_gb_min or 0.0) > 0}


def test_broker_status_report_shape_with_fakes(tmp_path, monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=48.0)
    _book_chat_resident(stack)
    stack["ledger"].record_pull("gpu-box", "krea-2-turbo",
                                "/models/krea.safetensors", 1234)
    tools = _broker_env(monkeypatch, tmp_path, stack)

    out = json.loads(tools.model_forge_broker(
        {"action": "status", "target": "gpu-box"}))
    assert out["ok"] is True and out["action"] == "status"
    assert out["pulled"] == [{"model_id": "krea-2-turbo",
                              "path": "/models/krea.safetensors",
                              "size_bytes": 1234}]
    assert out["resident"] == [{"model_id": "chat-a", "engine": "ollama",
                                "lane": "chat"}]
    booked = {row["model_id"]: row for row in out["booked"]}
    assert booked["krea-2-turbo"]["vram_gb"] == 16.0
    assert booked["krea-2-turbo"]["source"] == tools.BOOKING_SOURCE
    assert out["budget"] == {"nominal_gb": 48.0, "booked_now_gb": 6.0,
                             "headroom_gb": 42.0}
    assert out["unbooked_residents"] == []
    # honest labels: published floors as ceilings, never a measured claim
    assert "published" in out["note"] and "measured" in out["note"]
    assert calls == []            # read-only: no engine call was driven


def test_broker_status_marks_unbooked_engine_managed_rows(tmp_path,
                                                          monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=48.0)
    _book_chat_resident(stack)
    tools = _broker_env(monkeypatch, tmp_path, stack)

    out = json.loads(tools.model_forge_broker(
        {"action": "status", "target": "gpu-box"}))
    rows = {row["model_id"]: row for row in out["unbooked"]}
    # the engine-managed chat tiers declare no VRAM floor: they stay
    # unbooked, with an honest hint instead of a guessed 0 GB ceiling
    assert set(rows) == {"qwen3.8-flash-next", "deepseek-v4.1-flash",
                         "minimax-m3.1-flash", "glm-5.3-flash"}
    for row in rows.values():
        assert row["vram_gb"] is None
        assert row["hint"] == "unbooked — run bench after first load"
    # never presented as booked ceilings (and stubs/markers stay out)
    booked_ids = {row["model_id"] for row in out["booked"]}
    assert not set(rows) & booked_ids
    assert "world-model-slot" not in rows and "core-gguf-catalog" not in rows

    # a measured peak books one — the hint row is gone, the label is honest
    stack["ledger"].record_measured_peak("glm-5.3-flash", 30.0)
    out = json.loads(tools.model_forge_broker(
        {"action": "status", "target": "gpu-box"}))
    assert "glm-5.3-flash" not in {row["model_id"]
                                   for row in out["unbooked"]}
    booked = {row["model_id"]: row for row in out["booked"]}
    assert booked["glm-5.3-flash"]["vram_gb"] == 30.0
    assert booked["glm-5.3-flash"]["label"] == "measured peak (bench)"


def test_broker_claim_runs_plan_steps_in_order_with_honest_progress(
        tmp_path, monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=24.0)
    _book_chat_resident(stack)
    tools = _broker_env(monkeypatch, tmp_path, stack)

    out = json.loads(tools.model_forge_broker(
        {"action": "claim", "target": "gpu-box", "model_id": "krea-2-turbo",
         "scope_id": "shot-1"}))
    assert out["ok"] is True and out["action"] == "claim"
    assert out["scope_id"] == "shot-1"
    # the plan steps run in order: releases first, then the pre-warm load
    assert [(s["action"], s["model_id"]) for s in out["steps"]] == [
        ("release", "chat-a"), ("load", "krea-2-turbo")]
    assert calls == [("release", "ollama", "chat-a"),
                     ("ensure_loaded", "comfyui", "krea-2-turbo")]
    # honest progress states — plain 'loading' with no invented timing
    assert [p["state"] for p in out["progress"]] == ["queued", "loading",
                                                    "ready"]
    assert out["progress"][-1]["scope_id"] == "shot-1"
    # the policy reasons ride along
    assert "image generation" in out["plan"]["release_steps"][0]["reason"]


def test_broker_claim_by_modality_uses_first_curated_model(tmp_path,
                                                           monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=48.0)
    tools = _broker_env(monkeypatch, tmp_path, stack)
    out = json.loads(tools.model_forge_broker(
        {"action": "claim", "target": "gpu-box", "modality": "image",
         "scope_id": "shot-m"}))
    assert out["ok"] is True
    assert out["model_id"] == "krea-2-turbo"   # first curated image model
    assert out["scope_id"] == "shot-m"


def test_broker_claim_fails_closed_over_budget(tmp_path, monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=8.0)
    _book_chat_resident(stack)
    tools = _broker_env(monkeypatch, tmp_path, stack)

    out = json.loads(tools.model_forge_broker(
        {"action": "claim", "target": "gpu-box", "model_id": "krea-2-turbo"}))
    assert out["ok"] is False and out["action"] == "claim_failed"
    assert out["scope_id"] is None
    assert "exceeds the nominal budget" in out["error"]
    assert "fail-closed" in out["error"]
    assert calls == []            # a refused claim executes nothing


def test_broker_release_restores_claim_time_residents(tmp_path, monkeypatch):
    # the vision gesture at handler level: with a chat model resident, claim
    # image generation, then release — chat comes back, in order
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=24.0)
    _book_chat_resident(stack)
    tools = _broker_env(monkeypatch, tmp_path, stack)

    out = json.loads(tools.model_forge_broker(
        {"action": "claim", "target": "gpu-box", "model_id": "krea-2-turbo",
         "scope_id": "chat-to-image"}))
    assert out["ok"] is True
    out = json.loads(tools.model_forge_broker(
        {"action": "release", "scope_id": "chat-to-image"}))
    assert out["ok"] is True and out["action"] == "release"
    assert calls == [
        ("release", "ollama", "chat-a"),
        ("ensure_loaded", "comfyui", "krea-2-turbo"),
        ("release", "comfyui", "krea-2-turbo"),
        ("ensure_loaded", "ollama", "chat-a")]
    residents = {r.model_id for r in stack["ledger"].residents("gpu-box")}
    assert residents == {"chat-a"}     # exactly the claim-time set is back


def test_broker_plan_dry_run_executes_nothing(tmp_path, monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=24.0)
    _book_chat_resident(stack)
    tools = _broker_env(monkeypatch, tmp_path, stack)

    out = json.loads(tools.model_forge_broker(
        {"action": "plan", "target": "gpu-box", "model_id": "krea-2-turbo"}))
    assert out["ok"] is True and out["action"] == "plan"
    assert out["dry_run"] is True
    would = out["would"]
    assert would["ok"] is True
    assert [(s["action"], s["model_id"]) for s in would["release_steps"]] == \
        [("release", "chat-a")]
    assert [(s["action"], s["model_id"]) for s in would["load_steps"]] == \
        [("load", "krea-2-turbo")]
    assert would["release_steps"][0]["reason"]
    assert calls == []            # dry run: nothing loaded, released, moved
    assert stack["ledger"].is_resident("gpu-box", "chat-a")
    assert not stack["ledger"].is_resident("gpu-box", "krea-2-turbo")
    # an impossible claim previews the refusal instead of executing anything
    stack["ledger"].set_budget("gpu-box", 2.0)
    out = json.loads(tools.model_forge_broker(
        {"action": "plan", "target": "gpu-box", "model_id": "krea-2-turbo"}))
    assert out["would"]["ok"] is False
    assert "exceeds the nominal budget" in out["would"]["reason"]
    assert calls == []


def test_bench_records_measured_peak_and_gates_fit(tmp_path, monkeypatch):
    # wiring: action=bench through the handler with an injected VRAM sampler
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=48.0)
    tools = _broker_env(monkeypatch, tmp_path, stack)
    monkeypatch.setattr(tools, "_BENCH_INTERVAL_S", 0.0)
    mid = "qwen3.8-flash-next"     # engine-managed: no published VRAM floor
    from model_forge.broker import LoadRequest

    def request():
        return LoadRequest(model_id=mid, engine="strata", lane="chat",
                           modality="chat")

    # before: the planner refuses to gate it — no booked ceiling at all
    plan = stack["planner"].plan_load(_REMOTE.id, request())
    assert plan.ok is False and "no booked footprint" in plan.reason

    readings = [4.0, 12.5, 9.0, 7.0, 12.4, 6.0]
    seen = {"i": 0}

    def fake_sampler():
        value = readings[seen["i"] % len(readings)]
        seen["i"] += 1
        return value

    out = json.loads(tools.model_forge_broker(
        {"action": "bench", "model_id": mid, "target": "gpu-box"},
        sampler=fake_sampler))
    assert out["ok"] is True and out["action"] == "bench"
    assert out["peak_gb"] == 12.5          # the peak, not the last reading
    assert out["label"] == "measured peak (bench)"
    assert out["booking"] == {"model_id": mid, "vram_gb": 12.5,
                              "source": "measured",
                              "label": "measured peak (bench)"}
    assert out["gates_fit"] is True
    assert "planner" in out["note"] and "never downgraded" in out["note"]
    assert out["samples"] == len(readings)
    # the measured peak is the booked ceiling and the planner gates fit on it
    assert stack["ledger"].booked_vram_gb(mid) == 12.5
    plan = stack["planner"].plan_load(_REMOTE.id, request())
    assert plan.ok is True
    # a measured peak is never downgraded by a published footprint
    row = stack["ledger"].book(mid, {"vram_gb": 2.0,
                                     "source": tools.BOOKING_SOURCE})
    assert "kept" in row
    assert stack["ledger"].booked_vram_gb(mid) == 12.5
    assert calls == []            # bench drives no engine, only measures


def test_bench_fails_soft_without_a_gpu(tmp_path, monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=48.0)
    tools = _broker_env(monkeypatch, tmp_path, stack)
    from model_forge.registry import HardwareProfile
    monkeypatch.setattr(tools, "probe_hardware",
                        lambda: HardwareProfile(gpus=(), total_vram_gb=0.0,
                                                ram_gb=32.0,
                                                platform="linux"))
    out = json.loads(tools.model_forge_broker(
        {"action": "bench", "model_id": "qwen3.8-flash-next",
         "target": "gpu-box"}))
    assert out["ok"] is False and out["action"] == "bench"
    assert "no GPU is present" in out["error"]
    # fail soft: nothing was measured and nothing was booked
    assert stack["ledger"].booking("qwen3.8-flash-next") is None
    assert calls == []


def test_bench_window_without_a_valid_reading_fails_closed(tmp_path,
                                                           monkeypatch):
    # a booked ceiling must be an observation, never a guess
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=48.0)
    tools = _broker_env(monkeypatch, tmp_path, stack)
    monkeypatch.setattr(tools, "_BENCH_INTERVAL_S", 0.0)
    out = json.loads(tools.model_forge_broker(
        {"action": "bench", "model_id": "krea-2-turbo", "target": "gpu-box"},
        sampler=lambda: None))
    assert out["ok"] is False
    assert "no valid VRAM readings" in out["error"]
    # the published booking is untouched — no guessed peak was booked
    booking = stack["ledger"].booking("krea-2-turbo")
    assert booking.source == tools.BOOKING_SOURCE


def test_bench_requires_a_model_id(tmp_path, monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=48.0)
    tools = _broker_env(monkeypatch, tmp_path, stack)
    out = json.loads(tools.model_forge_broker({"action": "bench",
                                               "target": "gpu-box"}))
    assert "model_id" in out["error"] and "bench" in out["error"]


def test_live_bench_sampler_parses_nvidia_smi_used_memory(monkeypatch):
    # the live sampler reads real GPU memory in the hardware-probe style:
    # nvidia-smi CSV query, MiB readings summed across GPUs into GB
    import subprocess
    from model_forge import tools
    monkeypatch.setattr(tools.shutil, "which",
                        lambda _: "/usr/bin/nvidia-smi")
    fake = subprocess.CompletedProcess(args=tools._NVIDIA_SMI_USED,
                                       stdout="1024\n2048\n",
                                       stderr="", returncode=0)
    monkeypatch.setattr(tools.subprocess, "run",
                        lambda *a, **kw: fake)
    assert tools.gpu_vram_used_gb() == 3.0


def test_generate_media_claims_and_releases_lane(tmp_path, monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=48.0)
    _book_chat_resident(stack)
    tools = _broker_env(monkeypatch, tmp_path, stack)
    monkeypatch.setattr(
        tools, "media_plan",
        lambda modality, base_url: {"ok": True, "action": "submit",
                                    "submit": "http://gpu-box.example.com"
                                              ":8080/v1/audio/music"})
    _fake_urlopen(monkeypatch, json.dumps({"url": "http://x/a.mp3"}).encode())

    out = json.loads(tools.model_forge_generate_media(
        {"modality": "music", "target": "gpu-box", "model": "gen/music",
         "prompt": "lofi"}))
    assert out["ok"] is True and out["action"] == "done"
    # one workflow scope around the job: claim the lane, release on completion
    assert calls == [("ensure_loaded", "media-gateway", "gen/music"),
                     ("release", "media-gateway", "gen/music")]
    lane = out["lane_claim"]
    assert lane["state"] == "released"
    assert [p["state"] for p in lane["progress"]] == ["queued", "loading",
                                                     "ready"]
    # the gateway model books the modality's largest published floor as its
    # nominal stand-in ceiling — honestly labeled as a stand-in
    spec = stack["bookings"]["gen/music"]
    assert spec["vram_gb"] == 20.0
    assert "stand-in" in spec["source"] and "ace-step" in spec["source"]


def test_generate_media_releases_lane_on_submit_failure(tmp_path, monkeypatch):
    # failure-path release: the scope ends and claim-time residents return
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=48.0)
    _book_chat_resident(stack)
    tools = _broker_env(monkeypatch, tmp_path, stack)
    monkeypatch.setattr(
        tools, "media_plan",
        lambda modality, base_url: {"ok": True, "action": "submit",
                                    "submit": "http://gpu-box.example.com"
                                              ":8080/v1/audio/music"})
    _fake_urlopen(monkeypatch, b"", exc=RuntimeError("conn refused"))

    out = json.loads(tools.model_forge_generate_media(
        {"modality": "music", "target": "gpu-box", "model": "gen/music",
         "prompt": "lofi"}))
    assert out["ok"] is False and out["action"] == "submit_failed"
    assert out["lane_claim"]["state"] == "released"
    assert calls == [("ensure_loaded", "media-gateway", "gen/music"),
                     ("release", "media-gateway", "gen/music")]
    assert stack["ledger"].is_resident("gpu-box", "chat-a")
    assert not stack["ledger"].is_resident("gpu-box", "gen/music")


def test_generate_media_claim_failure_fails_closed(tmp_path, monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=5.0)   # over budget
    tools = _broker_env(monkeypatch, tmp_path, stack)
    monkeypatch.setattr(
        tools, "media_plan",
        lambda modality, base_url: {"ok": True, "action": "submit",
                                    "submit": "http://gpu-box.example.com"
                                              ":8080/v1/audio/music"})

    def _no_call(req, timeout=None):
        raise AssertionError("a refused claim must never reach the target")

    monkeypatch.setattr("urllib.request.urlopen", _no_call)
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "music", "target": "gpu-box", "model": "gen/music",
         "prompt": "lofi"}))
    assert out["ok"] is False and out["action"] == "claim_failed"
    assert "exceeds the nominal budget" in out["error"]
    assert "nothing was submitted" in out["detail"]
    assert calls == []


def test_generate_media_3d_holds_scope_until_job_completes(tmp_path,
                                                           monkeypatch):
    calls = []
    stack = _residency_stack(_REMOTE, calls, budget=48.0)
    _book_chat_resident(stack)
    tools = _broker_env(monkeypatch, tmp_path, stack)
    monkeypatch.setattr(
        tools, "media_plan",
        lambda modality, base_url: {"ok": True, "action": "submit_job",
                                    "submit": "http://gpu-box.example.com"
                                              ":8080/jobs"})
    _fake_urlopen(monkeypatch, json.dumps({"id": "3djob-9"}).encode())
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box", "model": "gen/3d",
         "image": "https://example.com/x.png"}))
    assert out["ok"] is True and out["action"] == "submitted"
    assert out["lane_claim"]["state"] == "held"
    assert "3djob-9" in tools._MEDIA_SCOPE_JOBS

    # a still-running job keeps the lane claimed (never guess completion)
    _fake_urlopen(monkeypatch, json.dumps({"status": "running"}).encode())
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box", "job_id": "3djob-9"}))
    assert out["lane_claim"]["state"] == "held"
    assert "release" not in [c[0] for c in calls]
    assert "3djob-9" in tools._MEDIA_SCOPE_JOBS

    # terminal state: the scope releases and claim-time residents return
    _fake_urlopen(monkeypatch, json.dumps({"status": "done"}).encode())
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box", "job_id": "3djob-9"}))
    assert out["job"] == {"status": "done"}
    assert out["lane_claim"]["state"] == "released"
    assert "3djob-9" not in tools._MEDIA_SCOPE_JOBS
    assert calls == [("ensure_loaded", "media-gateway", "gen/3d"),
                     ("release", "media-gateway", "gen/3d")]
    assert {r.model_id for r in stack["ledger"].residents("gpu-box")} == \
        {"chat-a"}


def test_broker_stack_threads_injectable_engine_seams(tmp_path, monkeypatch):
    # the engine adapters take injectable HTTP/process seams — the real
    # stack wires them through, so nothing here touches the network
    from model_forge import tools
    http_calls = []

    def fake_http(method, url, body=None, timeout=None):
        http_calls.append((method, url, body))
        if url.endswith("/system_stats"):
            # ComfyUI readiness fingerprint
            return {"system": {"comfyui_version": "test"}, "devices": []}
        return {"models": []}

    spawn_calls = []

    class _Proc:
        def terminate(self):
            pass

        def kill(self):
            pass

        def poll(self):
            return None

    def fake_runner(argv):
        # engine processes may only be spawned through this seam — recorded,
        # never real
        spawn_calls.append(list(argv))
        return _Proc()

    stack = tools.broker_stack(_REMOTE, http=fake_http, runner=fake_runner,
                               which=lambda binary: "/fake/" + binary,
                               fresh=True)
    assert stack["budget_gb"] == 48.0
    assert stack["ledger"].booked_vram_gb("krea-2-turbo") == 16.0
    _book_chat_resident(stack)
    tools_env = _broker_env(monkeypatch, tmp_path, stack)

    # claim image generation: the resident chat model is evicted through the
    # Ollama adapter's real warm/evict call shape, on the fake HTTP seam
    out = json.loads(tools_env.model_forge_broker(
        {"action": "claim", "target": "gpu-box", "model_id": "krea-2-turbo",
         "scope_id": "seam-1"}))
    assert out["ok"] is True
    evict = [body for _, url, body in http_calls
             if isinstance(body, dict) and body.get("keep_alive") == 0]
    assert evict and evict[0]["model"] == "chat-a"
    # release: the claim-time chat model is warmed back through the same seam
    out = json.loads(tools_env.model_forge_broker(
        {"action": "release", "scope_id": "seam-1"}))
    assert out["ok"] is True
    warm = [body for _, url, body in http_calls
            if isinstance(body, dict) and body.get("model") == "chat-a"
            and body.get("keep_alive") not in (0, None)]
    assert warm
    # krea-2-turbo is ComfyUI-served: its readiness + cache-free contract
    # rides the HTTP seam and spawns no engine process
    assert any(url.endswith("/system_stats") for _, url, _ in http_calls)
    assert spawn_calls == []