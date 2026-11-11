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

_LXC = ComputeTarget(id="ai-lxc", label="AI LXC", kind="lan",
                     base_url="http://192.168.1.62:8080",
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


def _ok_engine(monkeypatch, modalities=("chat",)):
    """Make detect_engines report one available openai-compatible engine."""
    from model_forge.engines import base

    e = base.EngineStatus(id="openai-compatible", display_name="OpenAI-compat",
                          available=True,
                          endpoint="http://192.168.1.62:8080/v1",
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
                       ("model_forge_activate", schemas.ACTIVATE)]:
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


# --- hardware.py ------------------------------------------------------------

def test_probe_hardware_returns_profile(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    prof = hardware.probe_hardware()
    assert prof.gpus == ()
    assert prof.total_vram_gb == 0.0
    assert prof.platform in ("linux", "darwin", "macos", "windows")
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
    assert pull.download_plan("https://x/y.bin", "/abs/y.bin")["ok"] is True
    assert pull.download_plan("ftp://x/y", "/abs/y.bin")["ok"] is False
    assert pull.download_plan("https://x/y.bin", "rel/y.bin")["ok"] is False


def test_job_status_unknown(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    got = pull.job_status("does-not-exist")
    assert got["state"] == "unknown"
    assert "does-not-exist" in got["error"]


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
    res = pull.start_download("https://huggingface.co/x/y.bin",
                              str(tmp_path / "y.bin"))
    assert res["ok"] is True and res["job_id"]
    import time
    for _ in range(100):
        st = pull.job_status(res["job_id"])
        if st.get("state") in ("done", "error"):
            break
        time.sleep(0.02)
    assert st["state"] == "done"
    assert st["done_bytes"] == 5
    assert (tmp_path / "y.bin").read_bytes() == b"hello"


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
    pull._run("j1", "https://huggingface.co/x/y.bin", dest,
              "0" * 64)  # wrong digest
    st = pull.job_status("j1")
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
    pull._run("j2", "https://huggingface.co/x/y.bin", dest, "")
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
        {"id": "ai-lxc", "base_url": "ftp://x:1"}))
    assert "error" in err
    # happy path persists to the stubbed data dir
    out = json.loads(tools.model_forge_target_add(
        {"id": "ai-lxc", "label": "AI LXC",
         "base_url": "http://192.168.1.62:8080", "vram_gb": "48"}))
    assert out["added"] == "ai-lxc"
    assert any(t["id"] == "ai-lxc" for t in out["targets"])


# --- tools.py: browse -------------------------------------------------------

def test_browse_unknown_target(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    out = json.loads(tools.model_forge_browse({"target": "nope",
                                               "modality": "image"}))
    assert "error" in out


def test_browse_lists_and_prices(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    # point get_target at our temp store by monkeypatching the handler's view
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    out = json.loads(tools.model_forge_browse({"target": "ai-lxc",
                                               "modality": "image"}))
    assert out["target"] == "ai-lxc"
    # budget = declared VRAM + (RAM - 8 GB) = 48 + (96 - 8) = 136
    assert out["hardware_budget_gb"] == 136.0
    assert isinstance(out["models"], list)
    for m in out["models"]:
        assert "id" in m and "fit" in m and "modality" in m


# --- tools.py: status -------------------------------------------------------

def test_status_lists_targets_and_engines(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "load_targets",
                        lambda: load_targets(base_dir=tmp_path))
    _ok_engine(monkeypatch, ("image", "video"))
    out = json.loads(tools.model_forge_status())
    assert isinstance(out["targets"], list)
    assert out["pull_credentials"] == {"HF_TOKEN": False,
                                       "MODELSCOPE_API_TOKEN": False}
    by_id = {t["id"]: t for t in out["targets"]}
    assert "local" in by_id and "ai-lxc" in by_id
    eng = by_id["ai-lxc"]["engines"][0]
    assert eng["available"] is True and "image" in eng["modalities"]


def test_status_remote_hw_uses_declared_budget(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "load_targets",
                        lambda: load_targets(base_dir=tmp_path))
    monkeypatch.setattr(tools, "detect_engines", lambda base_url="": [])
    out = json.loads(tools.model_forge_status())
    lxc = next(t for t in out["targets"] if t["id"] == "ai-lxc")
    # remote target -> declared VRAM, not a live probe
    assert lxc["hardware"]["total_vram_gb"] == 48.0
    assert lxc["hardware"]["ram_gb"] == 96.0


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
    (tmp_path / "model-forge" / "jobs" / "jx.json").write_text(
        json.dumps({"job_id": "jx", "state": "done"}))
    out = json.loads(tools.model_forge_install({"job_id": "jx"}))
    assert out["job_id"] == "jx" and out["state"] == "done"


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
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    # stub the Maya HTTP pull to return a canned job row
    monkeypatch.setattr(tools.maya_adapter, "pull",
                        lambda model, detach=False, base_url="":
                        {"ok": True, "job": "maya-1"})
    out = json.loads(tools.model_forge_install(
        {"model_id": "glm-5.3-flash", "target": "ai-lxc"}))
    assert out["action"] == "maya_pull"
    assert out["pull_response"] == {"ok": True, "job": "maya-1"}


def test_install_maya_no_response(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools.maya_adapter, "pull",
                        lambda model, detach=False, base_url="": None)
    out = json.loads(tools.model_forge_install(
        {"model_id": "glm-5.3-flash", "target": "ai-lxc"}))
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
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _ok_engine(monkeypatch, ("chat", "image"))
    out = json.loads(tools.model_forge_activate(
        {"model_id": "glm-5.3-flash", "target": "ai-lxc"}))
    assert out["ok"] is True and out["applied"] is False
    assert "hint" in out
    # image modality present -> media setup hint for image_gen
    assert "media_setup" in out and "image_gen" in out["media_setup"]


def test_activate_apply_writes(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _ok_engine(monkeypatch, ("chat",))
    setup = types.ModuleType("hermes_cli.model_setup")
    setup.persist_model_selection = (
        lambda model, provider, base_url="": {"model": model})
    monkeypatch.setitem(sys.modules, "hermes_cli.model_setup", setup)
    out = json.loads(tools.model_forge_activate(
        {"model_id": "glm-5.3-flash", "target": "ai-lxc", "apply": True}))
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
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://192.168.1.62:8080/jobs"})
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "ai-lxc"}))
    assert out["action"] == "needs_args"


def test_generate_media_3d_submit(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://192.168.1.62:8080/jobs"})
    _fake_urlopen(monkeypatch, json.dumps({"id": "3djob-1"}).encode())
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "ai-lxc", "model": "gen/3d",
         "image": "https://example.com/x.png"}))
    assert out["ok"] is True and out["action"] == "submitted"
    assert out["submitted"]["id"] == "3djob-1"


def test_generate_media_3d_submit_failure(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://192.168.1.62:8080/jobs"})
    _fake_urlopen(monkeypatch, b"", exc=RuntimeError("conn refused"))
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "ai-lxc", "model": "gen/3d",
         "image": "https://example.com/x.png"}))
    assert out["action"] == "submit_failed"


def test_generate_media_music_needs_args(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://192.168.1.62:8080/audio"})
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "music", "target": "ai-lxc"}))
    assert out["action"] == "needs_args"


def test_generate_media_music_done(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": True, "action": "submit_job",
                         "submit": "http://192.168.1.62:8080/audio"})
    _fake_urlopen(monkeypatch, json.dumps({"url": "http://x/a.mp3"}).encode())
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "music", "target": "ai-lxc", "model": "gen/music",
         "prompt": "lofi"}))
    assert out["ok"] is True and out["action"] == "done"
    assert out["result"]["url"] == "http://x/a.mp3"


def test_generate_media_route_missing(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "media_plan",
                        lambda modality, base_url:
                        {"ok": False, "action": "route_missing"})
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "ai-lxc", "model": "gen/3d",
         "image": "https://example.com/x.png"}))
    assert out["action"] == "route_missing"


# --- tools.py: _as_data_url -------------------------------------------------

def test_as_data_url_passthrough():
    from model_forge import tools

    assert tools._as_data_url("https://x/y.png") == "https://x/y.png"
    assert tools._as_data_url("data:image/png;base64,AA") \
        == "data:image/png;base64,AA"
    assert tools._as_data_url("/does/not/exist/x.png") \
        == "/does/not/exist/x.png"


def test_as_data_url_local_file(tmp_path):
    from model_forge import tools

    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n")
    got = tools._as_data_url(str(p))
    assert got.startswith("data:image/png;base64,")
    assert base64.b64decode(got.split(",", 1)[1]) == b"\x89PNG\r\n"


def test_as_data_url_jpeg_mime(tmp_path):
    from model_forge import tools

    p = tmp_path / "x.jpg"
    p.write_bytes(b"\xff\xd8")
    got = tools._as_data_url(str(p))
    assert got.startswith("data:image/jpeg;base64,")