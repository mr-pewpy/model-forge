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
                       ("model_forge_storage", schemas.STORAGE)]:
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