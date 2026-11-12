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
_LXC = ComputeTarget(id="ai-lxc", label="AI LXC", kind="lan",
                     base_url="http://192.168.1.62:8080",
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
                        endpoint="http://192.168.1.62:8080/v1")
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
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _detected(monkeypatch, ["comfyui"])
    out = json.loads(tools.model_forge_browse(
        {"target": "ai-lxc", "modality": "3d", "fresh": False}))
    rows = {m["id"]: m for m in out["models"]}
    fit = rows["hunyuan3d-2"]["fit"]
    assert fit["engine_available"] is True and fit["engine"] == "comfyui"
    assert fit["verdict"] == "fits_gpu"       # 48 GB VRAM target
    # chat rows carry the RAM floor
    _detected(monkeypatch, ["maya"])
    out = json.loads(tools.model_forge_browse(
        {"target": "ai-lxc", "modality": "chat", "fresh": False}))
    flash = next(m for m in out["models"] if m["id"] == "qwen3.8-flash-next")
    assert flash["fit"]["ram_gb_min"] == 64.0
    # target has 96 GB RAM -> floor met, engine matched -> engine_managed
    assert flash["fit"]["verdict"] == "engine_managed"


# --- install engine selection -----------------------------------------------

def test_install_plan_auto_selects_detected_engine():
    from model_forge.tools import install_plan
    qwen = _entry("qwen-image-2.1")           # (vllm, comfyui)
    plan = install_plan(qwen, _LXC, engine_model="model.safetensors",
                        available_engines=["comfyui"])
    assert plan["engine"] == "comfyui"        # NOT blind engines[0]=vllm
    assert plan["ok"] is True and plan["action"] == "run_on_target"
    # first match in entry order when several are detected
    plan = install_plan(qwen, _LXC, engine_model="model.safetensors",
                        available_engines=["comfyui", "vllm"])
    assert plan["engine"] == "vllm"


def test_install_plan_engine_override():
    from model_forge.tools import install_plan
    qwen = _entry("qwen-image-2.1")
    # override wins over detection, with an honest note when not detected
    plan = install_plan(qwen, _LXC, engine_model="model.safetensors",
                        engine="comfyui", available_engines=["vllm"])
    assert plan["engine"] == "comfyui"
    assert "not detected" in plan["engine_note"]
    # an override naming an engine the model cannot use is refused
    plan = install_plan(qwen, _LXC, engine="maya")
    assert plan["ok"] is False and plan["action"] == "engine_mismatch"
    assert "comfyui" in plan["detail"] or "vllm" in plan["detail"]


def test_install_plan_fallback_note_when_nothing_detected():
    from model_forge.tools import install_plan
    qwen = _entry("qwen-image-2.1")
    plan = install_plan(qwen, _LXC, engine_model="model.safetensors",
                        available_engines=[])
    assert plan["engine"] == "vllm"           # engines[0] fallback
    assert "falling back" in plan["engine_note"]
    assert "vllm" in plan["engine_note"]
    # no detection performed (None) keeps the historical quiet fallback
    plan = install_plan(qwen, _LXC, engine_model="model.safetensors")
    assert plan["engine"] == "vllm" and plan["engine_note"] == ""


def test_install_plan_needs_engine_model_intact():
    from model_forge.tools import install_plan
    plan = install_plan(_entry("qwen3.8-flash-next"), _LOCAL,
                        available_engines=["strata"])
    assert plan["ok"] is False and plan["action"] == "needs_engine_model"
    plan = install_plan(_entry("glm-5.3-flash"), _LXC, available_engines=[])
    assert plan["action"] == "maya_pull" and plan["model"] == "glm-maya-s"
    plan = install_plan(_entry("hunyuan3d-2"), _LOCAL,
                        available_engines=["comfyui"])
    assert plan["action"] == "needs_engine_model"   # local file pull needs path


def test_install_handler_auto_selects_detected_engine(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _detected(monkeypatch, ["comfyui"])
    out = json.loads(tools.model_forge_install(
        {"model_id": "qwen-image-2.1", "target": "ai-lxc",
         "engine_model": "model.safetensors"}))
    assert out["ok"] is True and out["engine"] == "comfyui"
    assert out["action"] == "run_on_target"
    # explicit engine override flows through the handler too
    _detected(monkeypatch, ["vllm"])
    out = json.loads(tools.model_forge_install(
        {"model_id": "qwen-image-2.1", "target": "ai-lxc",
         "engine": "comfyui", "engine_model": "model.safetensors"}))
    assert out["engine"] == "comfyui"
    assert "not detected" in out["engine_note"]


# --- phantom 'python' engine honesty ----------------------------------------

def test_phantom_python_notes_are_honest():
    phantom_ids = {m.id for m in MODELS if "python" in m.engines}
    assert phantom_ids == {"trellis.2-4b", "ace-step", "musicgen-small",
                           "audiogen", "stable-audio-open",
                           "chatterbox-t3-turbo"}
    for m in MODELS:
        if "python" in m.engines:
            assert PHANTOM_PHRASE in m.note, m.id
            assert m.status in ("known", "validated")   # entries not dropped


def test_browse_engine_listing_labels_phantom(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    _detected(monkeypatch, ["comfyui"])
    out = json.loads(tools.model_forge_browse(
        {"target": "ai-lxc", "modality": "music", "fresh": False}))
    row = out["models"][0]
    assert row["engines"][0].startswith("python (")
    assert PHANTOM_PHRASE in row["engines"][0]
    # hybrid entries: real engines stay plain, phantom lane is labeled
    out = json.loads(tools.model_forge_browse(
        {"target": "ai-lxc", "modality": "3d", "fresh": False}))
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
    plan = install_plan(_entry("hunyuan3d-2"), _LXC,
                        engine_model="checkpoints/x.safetensors")
    assert plan["command"] == ["hf", "download",
                               "Comfy-Org/hunyuan3D_2.0_repackaged",
                               "checkpoints/x.safetensors"]
    # no engine_model -> whole-repo pull, not a stray positional '*'
    plan = install_plan(_entry("hunyuan3d-2"), _LXC)
    assert plan["command"] == ["hf", "download",
                               "Comfy-Org/hunyuan3D_2.0_repackaged",
                               "--include", "*"]


# --- route_missing / ComfyUI honesty ----------------------------------------

def test_route_missing_says_comfyui_flow_not_implemented(monkeypatch):
    monkeypatch.setattr("model_forge.media.route_exists",
                        lambda root, path, timeout=2.0: False)
    from model_forge.media import media_plan
    plan = media_plan("3d", "http://192.168.1.62:8080")
    assert plan["ok"] is False and plan["action"] == "route_missing"
    assert "ComfyUI" in plan["detail"]
    assert "not implemented" in plan["detail"]
    assert "weights-broker" in plan["detail"]
    assert "/v1/videos/generations" in plan["detail"]   # what does work


def test_generate_media_passes_honest_route_error(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_LXC, base_dir=tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr("model_forge.media.route_exists",
                        lambda root, path, timeout=2.0: False)
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "3d", "target": "ai-lxc"}))
    assert out["ok"] is False and out["action"] == "route_missing"
    assert "not implemented" in out["detail"]
    assert "weights-broker" in out["detail"]


def test_hunyuan3d_note_carries_comfyui_caveat():
    note = _entry("hunyuan3d-2").note
    assert "ComfyUI" in note
    assert "not implemented" in note
    assert "weights-broker" in note


# --- INSTALL schema ----------------------------------------------------------

def test_install_schema_has_optional_engine_arg():
    from model_forge import schemas
    props = schemas.INSTALL["parameters"]["properties"]
    assert props["engine"]["type"] == "string"
    assert props["engine_model"]["type"] == "string"   # override still there
    assert schemas.INSTALL["parameters"].get("required", []) == []
