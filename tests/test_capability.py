"""Capability cross-check tests: engine-aware fit verdicts, honest install
engine routing, phantom-'python' truth-telling, RAM-floor surfacing, the
remote hf argv fix, and route_missing honesty.

Pure planners (fit_verdict / fit_report / install_plan) are exercised
directly; handler tests stub Hermes internals and engine detection so
nothing touches the network or the live config.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from model_forge.engines.base import EngineStatus
from model_forge.registry import (
    MODELS, PHANTOM_ENGINES, HardwareProfile, fit_report, fit_verdict,
    engine_match, models_for,
)
from model_forge.targets import ComputeTarget, add_target, get_target

_BIG = HardwareProfile(gpus=({"name": "RTX 4090", "vram_gb": 24.0},),
                       total_vram_gb=24.0, ram_gb=64.0, platform="linux")
_LOW_RAM = HardwareProfile(gpus=({"name": "RTX 4090", "vram_gb": 24.0},),
                           total_vram_gb=24.0, ram_gb=16.0, platform="linux")
_REMOTE = ComputeTarget(id="gpu-box", label="GPU Box", kind="lan",
                     base_url="http://gpu-box.example.com:8080",
                     vram_gb=48.0, ram_gb=96.0)
_LOCAL = ComputeTarget(id="local", label="This machine",
                       base_url="http://127.0.0.1:8080", kind="local")

PHANTOM_PHRASE = "custom Python environment the plugin does not manage"


def _entry(mid):
    return next(m for m in models_for() if m.id == mid)


def _stub_hermes_modules(monkeypatch, data_dir: Path):
    plugins = types.ModuleType("plugins")
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_data_dir = lambda name: data_dir / name
    plugins.plugin_storage = storage
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)


def _detected(monkeypatch, ids):
    """Make tools.detect_engines report exactly these available engine ids."""
    sts = [EngineStatus(id=i, display_name=i, available=True,
                        endpoint="http://gpu-box.example.com:8080/v1")
           for i in ids]
    monkeypatch.setattr("model_forge.tools.detect_engines",
                        lambda base_url="": sts)


# --- engine-aware fit verdicts ----------------------------------------------

def test_fit_verdict_none_default_is_hardware_only():
    # available_engines=None keeps the historical strings exactly
    hunyuan = _entry("hunyuan3d-2")
    assert fit_verdict(hunyuan, _BIG) == "fits_gpu"
    mid = HardwareProfile(gpus=({"name": "GTX 1660", "vram_gb": 6.0},),
                          total_vram_gb=6.0, ram_gb=32.0, platform="linux")
    assert fit_verdict(_entry("deepseek-ocr-2"), mid) == "fits_memory"
    assert fit_verdict(_entry("world-model-slot"), _BIG) == "unavailable"


def test_fit_verdict_engine_available():
    hunyuan = _entry("hunyuan3d-2")          # engines: (comfyui,)
    assert fit_verdict(hunyuan, _BIG, ["comfyui"]) == "fits_gpu"
    # several engines listed: any match counts
    qwen = _entry("qwen-image-2.1")          # engines: (vllm, comfyui)
    assert fit_verdict(qwen, _BIG, ["comfyui", "llamacpp"]) == "fits_gpu"


def test_fit_verdict_engine_unavailable_beats_hardware_fit():
    hunyuan = _entry("hunyuan3d-2")
    # hardware says fits_gpu but no engine on the target can run it
    assert fit_verdict(hunyuan, _BIG, []) == "engine_unavailable"
    assert fit_verdict(hunyuan, _BIG, ["vllm"]) == "engine_unavailable"
    # a phantom-only entry can never be 'available'
    musicgen = _entry("musicgen-small")      # engines: (python,)
    assert fit_verdict(musicgen, _BIG, ["python", "comfyui"]) \
        == "engine_unavailable"


def test_engine_match_entry_order_and_phantom_skip():
    qwen = _entry("qwen-image-2.1")          # (vllm, comfyui)
    assert engine_match(qwen, ["comfyui", "vllm"]) == "vllm"   # entry order
    assert engine_match(qwen, ["comfyui"]) == "comfyui"
    assert engine_match(qwen, []) == ""
    trellis = _entry("trellis.2-4b")         # (comfyui, python)
    assert engine_match(trellis, ["python"]) == ""   # phantom never matches
    assert engine_match(trellis, ["python", "comfyui"]) == "comfyui"


def test_fit_report_engine_signals():
    hunyuan = _entry("hunyuan3d-2")
    r = fit_report(hunyuan, _BIG, ["comfyui"])
    assert r["verdict"] == "fits_gpu"
    assert r["engine_available"] is True and r["engine"] == "comfyui"
    r = fit_report(hunyuan, _BIG, [])
    assert r["engine_available"] is False and r["engine"] == ""
    assert r["verdict"] == "engine_unavailable"
    assert "no engine" in r["advisory"] and "comfyui" in r["advisory"]
    # phantom honesty rides along in the advisory
    r = fit_report(_entry("ace-step"), _BIG, [])
    assert PHANTOM_PHRASE in r["advisory"]


# --- ram_gb_min surfacing + engine_managed advisory --------------------------

def test_ram_gb_min_used_for_engine_managed_entries():
    flash = _entry("qwen3.8-flash-next")     # ram_gb_min=64, engine_managed
    assert flash.ram_gb_min == 64.0
    # 64 GB machine meets the floor: unchanged verdict
    assert fit_verdict(flash, _BIG, ["strata"]) == "engine_managed"
    # too little RAM must not hide behind 'engine_managed'
    assert fit_verdict(flash, _LOW_RAM, ["strata"]) == "needs_ram"
    assert fit_verdict(flash, _LOW_RAM) == "needs_ram"   # default mode too
    # unknown RAM (0) never raises a false advisory
    no_ram = HardwareProfile(gpus=(), total_vram_gb=0.0, ram_gb=0.0,
                             platform="linux")
    assert fit_verdict(flash, no_ram) == "engine_managed"


def test_fit_report_ram_floor_advisory():
    flash = _entry("qwen3.8-flash-next")
    r = fit_report(flash, _BIG, ["strata"])
    assert r["ram_gb_min"] == 64.0
    assert r["verdict"] == "engine_managed" and r["advisory"] == ""
    r = fit_report(flash, _LOW_RAM, ["strata"])
    assert r["verdict"] == "needs_ram"
    assert r["ram_gb_min"] == 64.0
    assert "64" in r["advisory"] and "16" in r["advisory"]
    assert "engine-managed" in r["advisory"]


def test_browse_rows_surface_fit_signals(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _detected(monkeypatch, ["comfyui"])
    out = json.loads(tools.model_forge_browse(
        {"target": "gpu-box", "modality": "3d", "fresh": False}))
    rows = {m["id"]: m for m in out["models"]}
    fit = rows["hunyuan3d-2"]["fit"]
    assert fit["engine_available"] is True and fit["engine"] == "comfyui"
    assert fit["verdict"] == "fits_gpu"       # 48 GB VRAM target
    # chat rows carry the RAM floor
    _detected(monkeypatch, ["maya"])
    out = json.loads(tools.model_forge_browse(
        {"target": "gpu-box", "modality": "chat", "fresh": False}))
    flash = next(m for m in out["models"] if m["id"] == "qwen3.8-flash-next")
    assert flash["fit"]["ram_gb_min"] == 64.0
    # target has 96 GB RAM -> floor met, engine matched -> engine_managed
    assert flash["fit"]["verdict"] == "engine_managed"


# --- install engine selection -----------------------------------------------

def test_install_plan_auto_selects_detected_engine():
    from model_forge.tools import install_plan
    qwen = _entry("qwen-image-2.1")           # (vllm, comfyui)
    plan = install_plan(qwen, _REMOTE, engine_model="model.safetensors",
                        available_engines=["comfyui"])
    assert plan["engine"] == "comfyui"        # NOT blind engines[0]=vllm
    assert plan["ok"] is True and plan["action"] == "run_on_target"
    # first match in entry order when several are detected
    plan = install_plan(qwen, _REMOTE, engine_model="model.safetensors",
                        available_engines=["comfyui", "vllm"])
    assert plan["engine"] == "vllm"


def test_install_plan_engine_override():
    from model_forge.tools import install_plan
    qwen = _entry("qwen-image-2.1")
    # override wins over detection, with an honest note when not detected
    plan = install_plan(qwen, _REMOTE, engine_model="model.safetensors",
                        engine="comfyui", available_engines=["vllm"])
    assert plan["engine"] == "comfyui"
    assert "not detected" in plan["engine_note"]
    # an override naming an engine the model cannot use is refused
    plan = install_plan(qwen, _REMOTE, engine="maya")
    assert plan["ok"] is False and plan["action"] == "engine_mismatch"
    assert "comfyui" in plan["detail"] or "vllm" in plan["detail"]


def test_install_plan_fallback_note_when_nothing_detected():
    from model_forge.tools import install_plan
    qwen = _entry("qwen-image-2.1")
    plan = install_plan(qwen, _REMOTE, engine_model="model.safetensors",
                        available_engines=[])
    assert plan["engine"] == "vllm"           # engines[0] fallback
    assert "no matching engine detected on this target" in plan["engine_note"]
    assert "install one or register a target that runs it" \
        in plan["engine_note"]
    assert "vllm" in plan["engine_note"]
    # no detection performed (None) keeps the historical quiet fallback
    plan = install_plan(qwen, _REMOTE, engine_model="model.safetensors")
    assert plan["engine"] == "vllm" and plan["engine_note"] == ""


def test_install_plan_needs_engine_model_intact():
    from model_forge.tools import install_plan
    plan = install_plan(_entry("qwen3.8-flash-next"), _LOCAL,
                        available_engines=["strata"])
    assert plan["ok"] is False and plan["action"] == "needs_engine_model"
    plan = install_plan(_entry("glm-5.3-flash"), _REMOTE, available_engines=[])
    assert plan["action"] == "maya_pull" and plan["model"] == "glm-maya-s"
    plan = install_plan(_entry("hunyuan3d-2"), _LOCAL,
                        available_engines=["comfyui"])
    assert plan["action"] == "needs_engine_model"   # local file pull needs path


def test_install_handler_auto_selects_detected_engine(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _detected(monkeypatch, ["comfyui"])
    out = json.loads(tools.model_forge_install(
        {"model_id": "qwen-image-2.1", "target": "gpu-box",
         "engine_model": "model.safetensors"}))
    assert out["ok"] is True and out["engine"] == "comfyui"
    assert out["action"] == "run_on_target"
    # explicit engine override flows through the handler too
    _detected(monkeypatch, ["vllm"])
    out = json.loads(tools.model_forge_install(
        {"model_id": "qwen-image-2.1", "target": "gpu-box",
         "engine": "comfyui", "engine_model": "model.safetensors"}))
    assert out["engine"] == "comfyui"
    assert "not detected" in out["engine_note"]


# --- phantom 'python' engine honesty ----------------------------------------

def test_phantom_python_notes_are_honest():
    phantom_ids = {m.id for m in MODELS if "python" in m.engines}
    assert phantom_ids == {"trellis.2-4b", "ace-step", "musicgen-small",
                           "audiogen", "stable-audio-open",
                           "chatterbox-turbo"}
    for m in MODELS:
        if "python" in m.engines:
            assert PHANTOM_PHRASE in m.note, m.id
            assert m.status in ("known", "validated")   # entries not dropped


def test_browse_engine_listing_labels_phantom(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _detected(monkeypatch, ["comfyui"])
    out = json.loads(tools.model_forge_browse(
        {"target": "gpu-box", "modality": "music", "fresh": False}))
    row = out["models"][0]
    assert row["engines"][0].startswith("python (")
    assert PHANTOM_PHRASE in row["engines"][0]
    # hybrid entries: real engines stay plain, phantom lane is labeled
    out = json.loads(tools.model_forge_browse(
        {"target": "gpu-box", "modality": "3d", "fresh": False}))
    trellis = next(m for m in out["models"] if m["id"] == "trellis.2-4b")
    assert trellis["engines"][0] == "comfyui"
    assert PHANTOM_PHRASE in trellis["engines"][1]


def test_install_plan_phantom_python_refuses_honestly():
    from model_forge.tools import install_plan
    plan = install_plan(_entry("musicgen-small"), _LOCAL)
    assert plan["ok"] is False and plan["action"] == "custom_env_unmanaged"
    assert PHANTOM_PHRASE in plan["detail"]
    assert "facebook/musicgen-small" in plan["detail"]   # where to get weights
    # neither an explicit override nor a matching detection makes it runnable
    plan = install_plan(_entry("musicgen-small"), _LOCAL, engine="python",
                        available_engines=["python"])
    assert plan["ok"] is False and plan["action"] == "custom_env_unmanaged"


# --- remote hf argv fix ------------------------------------------------------

def test_remote_hf_download_argv_is_wellformed():
    from model_forge.tools import install_plan
    plan = install_plan(_entry("hunyuan3d-2"), _REMOTE,
                        engine_model="checkpoints/x.safetensors")
    assert plan["command"] == ["hf", "download",
                               "Comfy-Org/Hunyuan-3D-2.0",
                               "checkpoints/x.safetensors"]
    # no engine_model -> whole-repo pull, not a stray positional '*'
    plan = install_plan(_entry("hunyuan3d-2"), _REMOTE)
    assert plan["command"] == ["hf", "download",
                               "Comfy-Org/Hunyuan-3D-2.0",
                               "--include", "*"]


# --- route_missing / ComfyUI honesty ----------------------------------------

def test_route_missing_says_comfyui_flow_not_implemented(monkeypatch):
    monkeypatch.setattr("model_forge.media.route_exists",
                        lambda root, path, timeout=2.0: False)
    from model_forge.media import media_plan
    plan = media_plan("3d", "http://gpu-box.example.com:8080")
    assert plan["ok"] is False and plan["action"] == "route_missing"
    assert "ComfyUI" in plan["detail"]
    assert "not implemented" in plan["detail"]
    assert "media gateway" in plan["detail"]
    assert "/v1/videos/generations" in plan["detail"]   # what does work


def test_generate_media_passes_honest_route_error(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr("model_forge.media.route_exists",
                        lambda root, path, timeout=2.0: False)
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "gpu-box"}))
    assert out["ok"] is False and out["action"] == "route_missing"
    assert "not implemented" in out["detail"]
    assert "media gateway" in out["detail"]


def test_hunyuan3d_note_carries_comfyui_caveat():
    note = _entry("hunyuan3d-2").note
    assert "ComfyUI" in note
    assert "not implemented" in note
    assert "media gateway" in note


# --- INSTALL schema ----------------------------------------------------------

def test_install_schema_has_optional_engine_arg():
    from model_forge import schemas
    props = schemas.INSTALL["parameters"]["properties"]
    assert props["engine"]["type"] == "string"
    assert props["engine_model"]["type"] == "string"   # override still there
    assert schemas.INSTALL["parameters"].get("required", []) == []


# --- engine_note honesty per case -------------------------------------------

def test_engine_note_phantom_fallback_matches_payload():
    from model_forge.tools import install_plan
    # fallback lands on a phantom engine: the note must not tell the caller
    # to install and serve it here — the payload says the opposite
    plan = install_plan(_entry("musicgen-small"), _LOCAL, available_engines=[])
    assert plan["engine"] == "python"
    assert PHANTOM_PHRASE in plan["engine_note"]
    assert "install and serve it first" not in plan["engine_note"]
    assert plan["action"] == "custom_env_unmanaged"
    assert PHANTOM_PHRASE in plan["detail"]     # note and payload agree
    # the same holds for an explicit phantom override
    plan = install_plan(_entry("musicgen-small"), _LOCAL, engine="python",
                        available_engines=[])
    assert PHANTOM_PHRASE in plan["engine_note"]
    assert "install and serve it first" not in plan["engine_note"]


def test_engine_note_real_engine_fallback_is_actionable():
    from model_forge.tools import install_plan
    plan = install_plan(_entry("glm-5.3-flash"), _REMOTE, available_engines=[])
    assert plan["engine"] == "maya"
    assert "no matching engine detected on this target" in plan["engine_note"]
    assert "install one or register a target that runs it" \
        in plan["engine_note"]


# --- HF repo file auto-resolve ----------------------------------------------

_GB = 1024 ** 3


def _fake_hf(siblings=None, exc=None, payload=None):
    """resolve_repo_weights fetch fake; records call URLs."""
    calls = []

    def fetch(url):
        calls.append(url)
        if exc is not None:
            raise exc
        if payload is not None:
            return payload
        return {"id": "org/repo", "siblings": siblings or []}

    fetch.calls = calls
    return fetch


def test_repo_weights_picks_largest_primary():
    from model_forge.tools import resolve_repo_weights
    fetch = _fake_hf([
        {"rfilename": "README.md", "size": 1000},
        {"rfilename": "config.json", "size": 300},
        {"rfilename": "model.bin", "size": 1 * _GB},
        {"rfilename": "model.safetensors", "size": 5 * _GB},
    ])
    plan = resolve_repo_weights("org/repo", fetch=fetch)
    assert plan["ok"] is True and plan["action"] == "resolved"
    assert plan["file"] == "model.safetensors"
    assert plan["size_bytes"] == 5 * _GB
    # model-info API with the blobs=true query parameter
    assert fetch.calls == [
        "https://huggingface.co/api/models/org/repo?blobs=true"]


def test_repo_weights_skips_component_files():
    from model_forge.tools import resolve_repo_weights
    # the component file is the largest file — it is still not the pick
    fetch = _fake_hf([
        {"rfilename": "text_encoders/clip.safetensors", "size": 6 * _GB},
        {"rfilename": "tokenizer/model.vocab", "size": 500_000},
        {"rfilename": "model.safetensors", "size": 2 * _GB},
    ])
    plan = resolve_repo_weights("org/repo", fetch=fetch)
    assert plan["ok"] is True
    assert plan["file"] == "model.safetensors"
    # smaller non-component weight wins over larger component weight
    fetch = _fake_hf([
        {"rfilename": "vae/diffusion_pytorch_model.safetensors",
         "size": 4 * _GB},
        {"rfilename": "unet/weights.safetensors", "size": 3 * _GB},
    ])
    plan = resolve_repo_weights("org/repo", fetch=fetch)
    assert plan["file"] == "unet/weights.safetensors"


def test_repo_weights_multi_component_pipeline():
    from model_forge.tools import resolve_repo_weights
    fetch = _fake_hf([
        {"rfilename": "model_index.json", "size": 400},
        {"rfilename": "unet/diffusion_pytorch_model.safetensors",
         "size": 5 * _GB},
    ])
    plan = resolve_repo_weights("org/repo", fetch=fetch)
    assert plan["ok"] is False and plan["action"] == "multi_file"
    assert "model_index.json" in plan["files"]
    assert "one file cannot satisfy" in plan["detail"]


def test_repo_weights_multi_component_dirs():
    from model_forge.tools import resolve_repo_weights
    fetch = _fake_hf([
        {"rfilename": "diffusion_models/dit.safetensors", "size": 5 * _GB},
        {"rfilename": "text_encoders/clip.safetensors", "size": 6 * _GB},
        {"rfilename": "vae/vae.safetensors", "size": 300 * 1024 ** 2},
    ])
    plan = resolve_repo_weights("org/repo", fetch=fetch)
    assert plan["ok"] is False and plan["action"] == "multi_file"
    assert "text_encoders/clip.safetensors" in plan["files"]
    assert "vae/vae.safetensors" in plan["files"]


def test_repo_weights_shards_are_multi_file():
    from model_forge.tools import resolve_repo_weights
    fetch = _fake_hf([
        {"rfilename": "model-00001-of-00002.safetensors", "size": 5 * _GB},
        {"rfilename": "model-00002-of-00002.safetensors", "size": 3 * _GB},
    ])
    plan = resolve_repo_weights("org/repo", fetch=fetch)
    assert plan["ok"] is False and plan["action"] == "multi_file"
    assert sorted(plan["files"]) == ["model-00001-of-00002.safetensors",
                                    "model-00002-of-00002.safetensors"]


def test_repo_weights_failsoft_when_api_unreachable():
    from model_forge.tools import resolve_repo_weights
    plan = resolve_repo_weights("org/repo",
                                fetch=_fake_hf(exc=OSError("no route")))
    assert plan["ok"] is False and plan["action"] == "needs_engine_model"
    assert "diffusion_models/your-weights.safetensors" in plan["detail"]
    assert "omit engine_model" in plan["detail"]
    assert "model.safetensors" not in plan["detail"]   # old example is gone
    # an unusable payload fails soft the same way
    plan = resolve_repo_weights("org/repo",
                                fetch=_fake_hf(payload={"not": "a repo"}))
    assert plan["ok"] is False and plan["action"] == "needs_engine_model"
    assert "omit engine_model" in plan["detail"]


def test_install_handler_auto_resolves_repo_file(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LOCAL, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _detected(monkeypatch, ["comfyui"])
    monkeypatch.setattr(
        tools, "_fetch_hf_json",
        _fake_hf([
            {"rfilename": "text_encoders/clip.safetensors", "size": 6 * _GB},
            {"rfilename": "checkpoints/hunyuan3d-dit-v2_fp16.safetensors",
             "size": 2 * _GB},
        ]))
    started = {}

    def fake_start(url, dest, sha256=""):
        started.update(url=url, dest=dest)
        return {"ok": True, "job_id": "aabbccdd0011",
                "poll": "model_forge_install with job_id=aabbccdd0011"}

    monkeypatch.setattr("model_forge.pull.start_download", fake_start)
    out = json.loads(tools.model_forge_install(
        {"model_id": "hunyuan3d-2", "target": "local"}))
    assert out["ok"] is True and out["action"] == "hf_download"
    assert out["resolved_file"] \
        == "checkpoints/hunyuan3d-dit-v2_fp16.safetensors"
    assert out["url"].endswith(
        "/checkpoints/hunyuan3d-dit-v2_fp16.safetensors")
    assert "auto-resolved" in out["engine_model_source"]
    assert started["url"] == out["url"]


def test_install_handler_failsoft_needs_engine_model(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LOCAL, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _detected(monkeypatch, ["comfyui"])
    monkeypatch.setattr(
        tools, "_fetch_hf_json",
        _fake_hf(exc=OSError("name resolution failed")))
    out = json.loads(tools.model_forge_install(
        {"model_id": "hunyuan3d-2", "target": "local"}))
    assert out["ok"] is False and out["action"] == "needs_engine_model"
    assert "diffusion_models/your-weights.safetensors" in out["detail"]
    assert "omit engine_model" in out["detail"]


def test_install_handler_multi_file_lists_components(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LOCAL, base_dir=tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _detected(monkeypatch, ["comfyui"])
    monkeypatch.setattr(
        tools, "_fetch_hf_json",
        _fake_hf([
            {"rfilename": "model_index.json", "size": 400},
            {"rfilename": "diffusion_models/dit.safetensors", "size": 5 * _GB},
        ]))
    out = json.loads(tools.model_forge_install(
        {"model_id": "hunyuan3d-2", "target": "local"}))
    assert out["ok"] is False and out["action"] == "multi_file"
    assert "model_index.json" in out["files"]


# --- engine provisioning: engine_install plans + discovery integration ------

def test_engine_install_plan_vendor_commands():
    from model_forge.tools import engine_install_plan
    plan = engine_install_plan("ollama")
    assert plan["ok"] is True and plan["action"] == "vendor_installer"
    assert plan["vendor"] == "Ollama" and plan["executes"] is True
    assert plan["command"] == "curl -fsSL https://ollama.com/install.sh | sh"
    maya = engine_install_plan("maya")
    assert maya["command"] == ("git clone https://github.com/mw00/project-maya "
                               "&& cd project-maya && ./setup.sh")
    assert any("git clone https://github.com/mw00/project-maya" in s
               for s in maya["steps"])
    # engine ids are normalized; empty and unknown ids are refused honestly
    assert engine_install_plan(" Strata ")["engine"] == "strata"
    bad = engine_install_plan("banana")
    assert bad["ok"] is False and bad["action"] == "unknown_engine"
    missing = engine_install_plan("")
    assert missing["ok"] is False and missing["action"] == "needs_engine"


def test_engine_install_commands_match_adapter_documented_installs(monkeypatch):
    # the planner's commands are exactly the installs the adapters document
    monkeypatch.setattr("model_forge.engines.strata.shutil.which",
                        lambda _: None)
    monkeypatch.setattr("model_forge.engines.strata.served_models",
                        lambda base_url="": [])
    monkeypatch.setattr("model_forge.engines.maya.catalog",
                        lambda base_url="": None)
    from model_forge.engines import maya, strata
    from model_forge.tools import engine_install_plan
    assert engine_install_plan("strata")["command"] in strata.detect("").detail
    detail = maya.detect("").detail
    assert "git clone https://github.com/mw00/project-maya" in detail
    assert "setup.sh" in detail
    assert engine_install_plan("maya")["command"].startswith(
        "git clone https://github.com/mw00/project-maya")


def test_engine_install_llamacpp_guides_to_core_runtime():
    from model_forge.tools import engine_install_plan
    present = engine_install_plan(
        "llamacpp",
        core_probe=lambda: {"present": True,
                            "signals": ["the hermes CLI is on PATH (Hermes "
                                        "core is installed)"]})
    assert present["ok"] is True
    assert present["action"] == "use_core_local_models"
    assert present["executes"] is False and "command" not in present
    assert "never installs llama.cpp binaries" in present["detail"]
    assert "Local Models" in present["detail"]
    assert "hermes model" in present["detail"]
    assert present["core_runtime"]["present"] is True
    absent = engine_install_plan("llamacpp",
                                 core_probe=lambda: {"present": False,
                                                     "signals": []})
    assert absent["action"] == "use_core_local_models"
    assert "No managed llama.cpp runtime was detected" in absent["detail"]
    assert "Local Models" in absent["detail"] and "downloads" in absent["detail"]


def test_install_plan_fallback_names_engine_install_for_provisionable_engines():
    from model_forge.tools import install_plan
    # maya fallback: the note names the engine_install action
    plan = install_plan(_entry("glm-5.3-flash"), _REMOTE, available_engines=[])
    note = plan["engine_note"]
    assert "no matching engine detected on this target" in note
    assert "install one or register a target that runs it" in note
    assert "action='engine_install', engine='maya'" in note
    # strata fallback likewise
    plan = install_plan(_entry("qwen3.8-flash-next"), _REMOTE,
                        available_engines=[])
    assert "action='engine_install', engine='strata'" in plan["engine_note"]
    # engines with no approved installer path keep the plain actionable note
    plan = install_plan(_entry("qwen-image-2.1"), _REMOTE, available_engines=[])
    assert "engine_install" not in plan["engine_note"]
    assert "install one or register a target that runs it" \
        in plan["engine_note"]
    # llama.cpp fallback names engine_install as honest core guidance
    plan = install_plan(_entry("core-gguf-catalog"), _REMOTE,
                        available_engines=[])
    assert "engine_install" in plan["engine_note"]
    assert "never installs llama.cpp binaries" in plan["engine_note"]
    assert "Local Models" in plan["engine_note"]
