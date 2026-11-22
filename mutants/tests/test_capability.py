"""Capability cross-check tests: engine-aware fit verdicts, honest install
engine routing, phantom-'python' truth-telling, RAM-floor surfacing, the
remote hf argv fix, and route_missing honesty.

Pure planners (fit_verdict / fit_report / install_plan) are exercised
directly; handler tests stub Hermes internals and engine detection so
nothing touches the network or the live config.
"""

from __future__ import annotations

import importlib.util
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

    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.load_config_readonly = lambda: {}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg_mod)


def _detected(monkeypatch, ids):
    """Make tools.detect_engines report exactly these available engine ids."""
    sts = [EngineStatus(id=i, display_name=i, available=True,
                        endpoint="http://gpu-box.example.com:8080/v1")
           for i in ids]
    monkeypatch.setattr("model_forge.tools.detect_engines",
                        lambda base_url="", **kw: sts)


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


def test_repo_weights_prefers_full_over_quant_regardless_of_size():
    # A2: full weights win even when a quantized variant is LARGER — a quant
    # is picked only on request (engine_model) or when no full weights exist.
    from model_forge.tools import resolve_repo_weights
    fetch = _fake_hf([
        {"rfilename": "model-f16.safetensors", "size": 2 * _GB},
        {"rfilename": "model-Q8_0.gguf", "size": 8 * _GB},
        {"rfilename": "model-Q4_K_M.gguf", "size": 4 * _GB},
    ])
    plan = resolve_repo_weights("org/repo", fetch=fetch)
    assert plan["ok"] is True
    assert plan["file"] == "model-f16.safetensors"
    assert plan["quantized"] is False
    assert "full weights preferred" in plan["detail"]


def test_repo_weights_falls_back_to_quant_only_when_no_full_weights():
    # a GGUF/quant-only repo still resolves (largest quant), labeled honestly
    from model_forge.tools import resolve_repo_weights
    fetch = _fake_hf([
        {"rfilename": "model-Q4_K_M.gguf", "size": 4 * _GB},
        {"rfilename": "model-Q8_0.gguf", "size": 8 * _GB},
    ])
    plan = resolve_repo_weights("org/repo", fetch=fetch)
    assert plan["ok"] is True
    assert plan["file"] == "model-Q8_0.gguf"
    assert plan["quantized"] is True
    assert "no full weights" in plan["detail"]


def test_is_quantized_names_quant_tokens_not_full_precision():
    from model_forge.tools import _is_quantized
    # quantized variants
    assert _is_quantized("model-Q4_K_M.gguf")
    assert _is_quantized("model-Q8_0.gguf")
    assert _is_quantized("model-int4.safetensors")
    assert _is_quantized("model-fp8.safetensors")
    assert _is_quantized("model-awq.safetensors")
    assert _is_quantized("weights/gptq/model.safetensors")
    # full precision (never treated as quant)
    assert not _is_quantized("model.safetensors")
    assert not _is_quantized("model-f16.gguf")
    assert not _is_quantized("model-bf16.safetensors")
    assert not _is_quantized("model-fp32.bin")


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

_OLLAMA_SHA = "25f64b810b947145095956533e1bdf56eacea2673c55a7e586be4515fc882c9f"
_STRATA_SHA = "1fcfbab133880249b7a6eaefb643f25b29a9b8f45624d2d8673e35948c329eda"


def test_engine_install_plan_vendor_commands():
    from model_forge.tools import engine_install_plan
    plan = engine_install_plan("ollama")
    assert plan["ok"] is True and plan["action"] == "vendor_installer"
    assert plan["vendor"] == "Ollama" and plan["executes"] is True
    # the pinned pipeline names the download + the SHA-256 verified in Python
    assert "https://ollama.com/install.sh" in plan["pipeline"]
    assert _OLLAMA_SHA in plan["pipeline"]
    assert "latest" not in plan["pipeline"] and "latest" not in plan["command"]
    # the shown command is the literal staged command the runner runs — no
    # fake sha256sum pipeline in the display
    assert "sha256sum" not in plan["pipeline"] and "sha256sum" not in plan["command"]
    assert plan["command"].startswith("sh ") and "install.sh" in plan["command"]
    assert plan["pinned"] == {"url": "https://ollama.com/install.sh",
                              "sha256": _OLLAMA_SHA}
    maya = engine_install_plan("maya")
    assert "git clone https://github.com/mw00/project-maya" in maya["pipeline"]
    assert "56b5216ab4464ac41c26b1be32eb63b98de53d0a" in maya["pipeline"]
    assert maya["command"].endswith("./setup.sh")
    assert any("56b5216ab4464ac41c26b1be32eb63b98de53d0a" in s
               for s in maya["steps"])
    # engine ids are normalized; empty and unknown ids are refused honestly
    assert engine_install_plan(" Strata ")["engine"] == "strata"
    bad = engine_install_plan("banana")
    assert bad["ok"] is False and bad["action"] == "unknown_engine"
    missing = engine_install_plan("")
    assert missing["ok"] is False and missing["action"] == "needs_engine"


def test_engine_install_commands_are_pinned_not_release_latest():
    # what executes comes only from the pinned catalog: a fixed Strata release
    # tag plus a SHA-256, and a fixed Maya commit — never a moving "latest"
    # ref and never a model-supplied string.
    from model_forge.tools import engine_install_plan
    strata = engine_install_plan("strata")
    assert ("https://github.com/Niko1221/Strata/releases/download/v0.2.15/"
            "install.sh") in strata["pipeline"]
    assert "latest" not in strata["pipeline"] and "latest" not in strata["command"]
    assert _STRATA_SHA in strata["pipeline"]
    assert strata["pinned"]["sha256"] == _STRATA_SHA
    maya = engine_install_plan("maya")
    assert "git clone https://github.com/mw00/project-maya" in maya["pipeline"]
    assert "setup.sh" in maya["command"]
    assert maya["pinned"]["commit"] == "56b5216ab4464ac41c26b1be32eb63b98de53d0a"


def test_engine_status_strings_advertise_no_unpinned_install(monkeypatch):
    """No engine adapter's status detail advertises an unpinned install: no
    ``releases/latest`` ref and no unpinned clone — the strings point at the
    pinned install path instead."""
    monkeypatch.setattr("model_forge.engines.strata.shutil.which",
                        lambda _: None)
    monkeypatch.setattr("model_forge.engines.strata.served_models",
                        lambda base_url="": [])
    monkeypatch.setattr("model_forge.engines.strata.strata_fingerprint",
                        lambda base_url="": False)
    monkeypatch.setattr("model_forge.engines.maya.catalog",
                        lambda base_url="": None)
    from model_forge.engines import maya, strata
    details = [strata.detect("").detail, strata.detect("http://x").detail,
               maya.detect("").detail, maya.detect("http://x").detail]
    for detail in details:
        assert "releases/latest" not in detail
        assert " && ./setup.sh" not in detail          # no unpinned clone+run
        assert "install.sh | sh" not in detail         # no pipe-to-shell
    # and the pinned provenance is named instead
    assert "1fcfbab133880249b7a6eaefb643f25b29a9b8f45624d2d8673e35948c329eda" \
        in strata.detect("").detail
    assert "56b5216ab4464ac41c26b1be32eb63b98de53d0a" in maya.detect("").detail


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


# --- cockpit models reuse the browse fit data --------------------------------

def test_cockpit_models_reuses_browse_fit_data(tmp_path, monkeypatch):
    # the Desktop pane's model section renders the model_forge_browse fit
    # data: cockpit rows must be exactly the browse rows (same fit blocks),
    # priced against the same target and detected engines.
    _stub_hermes_modules(monkeypatch, tmp_path)
    add_target(_REMOTE, base_dir=tmp_path)
    from model_forge import tools
    from model_forge.targets import load_targets as load_targets_store
    monkeypatch.setattr(tools, "load_targets",
                        lambda: load_targets_store(base_dir=tmp_path))
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(tools, "probe_hardware", lambda: _BIG)
    _detected(monkeypatch, ["comfyui"])
    # both priced against the local target with the same hardware + engines
    browse = json.loads(tools.model_forge_browse(
        {"modality": "3d", "fresh": False}))
    cockpit = tools.cockpit_models("3d")
    assert [m["id"] for m in cockpit["models"]] == \
        [m["id"] for m in browse["models"]]
    for a, b in zip(cockpit["models"], browse["models"]):
        assert a["fit"] == b["fit"]
        assert a["engines"] == b["engines"]
    # fit blocks carry the verdict + advisory the pane renders
    fit = cockpit["models"][0]["fit"]
    assert fit["verdict"] and "advisory" in fit


# --- model-forge-providers: native /model provider registration -------------

_REPO = Path(__file__).resolve().parent.parent
try:
    from _repo import repo_root as _repo_root
    _REPO = _repo_root()
except Exception:  # noqa: BLE001 — keep the literal path as the fallback
    pass


def _load_providers_pkg(monkeypatch):
    """Load model-forge-providers/__init__.py by file path against fakes for
    the provider registry (the test venv has no Hermes runtime)."""
    registered = []
    providers_mod = types.ModuleType("providers")
    base_mod = types.ModuleType("providers.base")

    class ProviderProfile:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            for attr, default in (
                    ("aliases", ()), ("display_name", ""), ("description", ""),
                    ("env_vars", ()), ("base_url", ""),
                    ("api_mode", "chat_completions"),
                    ("supports_model_listing", True),
                    ("supports_health_check", True),
                    ("fallback_models", ()), ("model_aliases", {}),
                    ("model_capabilities", {}), ("model_setup_flow", None)):
                self.__dict__.setdefault(attr, default)

    base_mod.ProviderProfile = ProviderProfile
    providers_mod.base = base_mod
    providers_mod.register_provider = registered.append
    monkeypatch.setitem(sys.modules, "providers", providers_mod)
    monkeypatch.setitem(sys.modules, "providers.base", base_mod)
    path = _REPO / "model-forge-providers" / "__init__.py"
    spec = importlib.util.spec_from_file_location("mf_providers", str(path))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "mf_providers", mod)
    spec.loader.exec_module(mod)
    return mod, registered


def _fake_model_setup(monkeypatch):
    mod = types.ModuleType("hermes_cli.model_setup")

    def persist_model_selection(model, provider, *, base_url=None,
                                api_mode=None):
        mod.calls.append({"model": model, "provider": provider,
                          "base_url": base_url, "api_mode": api_mode})
        return {"default": model, "provider": provider}

    mod.persist_model_selection = persist_model_selection
    mod.calls = []
    monkeypatch.setitem(sys.modules, "hermes_cli.model_setup", mod)
    return mod


def _fake_config_store(monkeypatch, cfg=None):
    mod = types.ModuleType("hermes_cli.config")
    state = json.loads(json.dumps(cfg or {}))

    def _load():
        return json.loads(json.dumps(state))

    def _save(new_cfg):
        state.clear()
        state.update(json.loads(json.dumps(new_cfg)))

    mod.load_config_readonly = _load
    mod.load_config = _load
    mod.save_config = _save
    mod.state = state
    monkeypatch.setitem(sys.modules, "hermes_cli.config", mod)
    return mod


def test_providers_package_registration_shape(monkeypatch):
    mod, registered = _load_providers_pkg(monkeypatch)
    by_name = {p.name: p for p in registered}
    assert set(by_name) == {"model-forge", "strata", "maya"}
    assert set(by_name) == set(mod.REGISTERED_SLUGS)
    for p in registered:
        # loopback rule: a base_url here would map every localhost endpoint
        # to this provider and disable local-server detection
        assert p.base_url == ""
        # the final *_BASE_URL env var is the user base-URL override
        assert p.env_vars[-1].endswith("_BASE_URL")
        assert p.fallback_models       # pickers always have a list
        assert p.model_capabilities    # declared per model id
        assert callable(p.model_setup_flow)
    assert callable(by_name["model-forge"].fetch_models)
    # engines that may be off keep the documented honest flags
    for name in ("strata", "maya"):
        assert by_name[name].supports_model_listing is False
        assert by_name[name].supports_health_check is False


def test_providers_capabilities_honest_against_registry(monkeypatch):
    mod, registered = _load_providers_pkg(monkeypatch)
    from model_forge.registry import MODELS
    from model_forge.tools import _CATALOG_MARKER_IDS

    def entry_for(mid):
        # "deepseek-v4.1-flash-1017" is the registry entry's name spelling,
        # kept verbatim as the maya row's existing fallback id
        if mid == "deepseek-v4.1-flash-1017":
            mid = "deepseek-v4.1-flash"
        for m in MODELS:
            if m.id == mid or mid in (m.engine_models or {}).values():
                return m
        return None

    for mid, caps in mod.MODEL_CAPABILITIES.items():
        entry = entry_for(mid)
        assert entry is not None, mid       # only real registry ids declared
        if caps.get("supports_vision") is True:
            # vision flags only where the registry knows them
            assert entry.modality in ("vision", "ocr") or "vision" in entry.note, mid
        if "supports_reasoning" in caps:
            assert "thinking" in entry.note, mid
        if "context_window" in caps:
            # context windows only where the registry states one
            assert str(caps["context_window"]) in entry.note, mid
        # unknown stays unknown: no False fillers anywhere
        for key, value in caps.items():
            assert value is not False, (mid, key)
    # unknown-but-honest: models the registry states nothing about carry no
    # invented fields at all
    for mid in ("qwen3.8-flash-next", "deepseek-v4.1-flash", "deepseek-maya-s"):
        assert mod.MODEL_CAPABILITIES[mid] == {}
    # per-model completeness: every id the package offers or persists is
    # declared
    ids = set(mod.MODEL_CAPABILITIES)
    for row in mod.CURATED_CHAT_VISION:
        ids.update(row["served_ids"])
    for p in registered:
        ids.update(p.fallback_models)
    assert ids <= set(mod.MODEL_CAPABILITIES)
    # the curated set is exactly the registry's activatable chat/vision ids
    # (catalog markers and stubs are not served model ids)
    registry_chatvision = {m.id for m in MODELS
                           if m.modality in ("chat", "vision")
                           and m.status != "stub"
                           and m.id not in _CATALOG_MARKER_IDS}
    assert set(mod.CURATED_IDS) == registry_chatvision
    # the served-id map matches the registry's engine_models exactly
    for row in mod.CURATED_CHAT_VISION:
        entry = entry_for(row["id"])
        assert set(row["served_ids"]) == \
            {row["id"]} | set((entry.engine_models or {}).values()), row["id"]


def test_providers_fetch_models_curated_plus_live(monkeypatch):
    mod, registered = _load_providers_pkg(monkeypatch)
    profile = next(p for p in registered if p.name == "model-forge")
    cfg_mod = _fake_config_store(monkeypatch, {
        "plugins": {"entries": {"model-forge": {"settings": {
            "engines_json": [{"label": "Workstation vLLM",
                              "base_url": "http://gpu-box.example.com:8000"}],
        }}}}})
    monkeypatch.delenv("MODEL_FORGE_BASE_URL", raising=False)
    # fake engines: the declared endpoint serves two extra model ids
    served = {"http://gpu-box.example.com:8000":
              ["minicpm-v-4.6", "extra-live-model"]}
    monkeypatch.setattr(mod, "_probe_served_models",
                        lambda root, timeout=8.0: served.get(root))
    got = profile.fetch_models()
    # curated chat/vision first, then live engine-served listings, deduped
    assert got[:len(mod.CURATED_IDS)] == list(mod.CURATED_IDS)
    assert "extra-live-model" in got
    assert got.count("minicpm-v-4.6") == 1
    # a caller-passed endpoint joins the probe set (the /model picker's
    # current custom endpoint)
    monkeypatch.setattr(mod, "_probe_served_models",
                        lambda root, timeout=8.0:
                        ["served-here"] if "8080" in root else served.get(root))
    before = json.loads(json.dumps(cfg_mod.state))
    got = profile.fetch_models(base_url="http://gpu-box.example.com:8080/v1")
    assert "served-here" in got and "extra-live-model" in got
    # a fetch never mutates config
    assert cfg_mod.state == before


def test_providers_fetch_models_falls_back_when_fetch_fails(monkeypatch):
    mod, registered = _load_providers_pkg(monkeypatch)
    profile = next(p for p in registered if p.name == "model-forge")
    cfg_mod = _fake_config_store(monkeypatch, {})
    monkeypatch.delenv("MODEL_FORGE_BASE_URL", raising=False)
    # nothing detectable: no listing at all, so pickers use fallback_models
    assert profile.fetch_models() is None
    # endpoints detectable but every engine is off (fake engine answers nothing)
    cfg_mod.load_config_readonly = lambda: {
        "plugins": {"entries": {"model-forge": {"settings": {
            "targets_json": [{"id": "gpu-box",
                              "base_url": "http://gpu-box.example.com:8080"}],
        }}}}}
    monkeypatch.setattr(mod, "_probe_served_models",
                        lambda root, timeout=8.0: None)
    assert profile.fetch_models() is None
    assert profile.fallback_models == mod.CURATED_IDS
    for p in registered:
        assert len(p.fallback_models) >= 1


def test_providers_setup_flow_persists_and_registers(monkeypatch, capsys):
    mod, registered = _load_providers_pkg(monkeypatch)
    cfg = {"custom_providers": [
        {"name": "gpu-box",
         "base_url": "http://gpu-box.example.com:8080/v1",
         "models": {"glm-maya-s": {}}}]}
    setup = _fake_model_setup(monkeypatch)
    cfg_mod = _fake_config_store(monkeypatch, cfg)
    # typed catalog id resolves to the served id the entry declares
    monkeypatch.setattr(mod, "_read_line", lambda prompt: "glm-5.3-flash")
    saved = mod.model_forge_setup_flow(cfg)
    assert saved == {"default": "glm-maya-s", "provider": "custom:gpu-box"}
    call = setup.calls[0]
    assert call["model"] == "glm-maya-s"
    assert call["provider"] == "custom:gpu-box"
    assert call["base_url"] == "http://gpu-box.example.com:8080/v1"
    assert call["api_mode"] == "chat_completions"
    # the custom_providers registration is re-merged (idempotent, declared)
    assert "glm-maya-s" in cfg_mod.state["custom_providers"][0]["models"]
    out = capsys.readouterr().out
    # reuses Model Forge's own browse/install/activate verbs
    for verb in ("model_forge_browse", "model_forge_install",
                 "model_forge_activate"):
        assert verb in out
    assert mod.NEXT_SESSION_NOTE in out
    assert "/model" in out
    # engine-bound flows (strata/maya rows) filter the catalog to their
    # engines' models
    flow = next(p for p in registered if p.name == "maya").model_setup_flow
    monkeypatch.setattr(mod, "_read_line", lambda prompt: "")
    flow(cfg, "current-model", None)
    out = capsys.readouterr().out
    assert "glm-5.3-flash" in out and "minicpm-v-4.6" not in out
    assert "current-model" in out and "No change." in out


def test_providers_setup_flow_refuses_unserved_model(monkeypatch, capsys):
    mod, _ = _load_providers_pkg(monkeypatch)
    setup = _fake_model_setup(monkeypatch)
    _fake_config_store(monkeypatch, {})
    monkeypatch.setattr(mod, "_read_line", lambda prompt: "glm-5.3-flash")
    cfg = {"custom_providers": []}
    assert mod.model_forge_setup_flow(cfg) is None
    assert setup.calls == []      # nothing persisted: fail closed, honestly
    out = capsys.readouterr().out
    assert "No change." in out
    for verb in ("model_forge_install", "model_forge_activate"):
        assert verb in out


def test_providers_setup_flow_number_pick(monkeypatch, capsys):
    mod, _ = _load_providers_pkg(monkeypatch)
    setup = _fake_model_setup(monkeypatch)
    _fake_config_store(monkeypatch, {})
    cfg = {"custom_providers": [
        {"name": "local", "base_url": "http://127.0.0.1:8080/v1",
         "models": {"qwen3.8-flash-next": {}}}]}
    monkeypatch.setattr(mod, "_read_line", lambda prompt: "1")
    mod.model_forge_setup_flow(cfg)
    assert setup.calls[0]["model"] == "qwen3.8-flash-next"
    assert setup.calls[0]["provider"] == "custom:local"
    assert "Saved model qwen3.8-flash-next on custom:local." \
        in capsys.readouterr().out


def test_persisted_slugs_round_trip_and_resolve(monkeypatch):
    mod, _ = _load_providers_pkg(monkeypatch)
    from model_forge.registry import ModelEntry
    from model_forge.tools import activation_plan

    # the reference's identity set — engine codenames are never resolvable
    for ok in ("custom", "custom:gpu-box", "custom:local", "strata", "maya",
               "model-forge", "llamacpp", "llama.cpp", "llama-cpp"):
        assert mod.is_resolvable_slug(ok), ok
    for bad in ("vllm", "ollama", "sglang", "comfyui", "exo", "custom:",
                "llamacppx"):
        assert not mod.is_resolvable_slug(bad), bad
    # legacy-entry identity: the slug keys on the normalized display name
    assert mod.custom_provider_slug("GPU Box") == "custom:gpu-box"
    assert mod.custom_provider_slug("gpu-box") == "custom:gpu-box"
    # every slug an activation persists round-trips: resolvable identity, and
    # the custom slug resolves to the entry declaring the persisted model
    cases = [("glm-5.3-flash", ["maya"]), ("qwen3.8-flash-next", ["strata"]),
             ("minicpm-v-4.6", ["vllm"]),
             ("qwen3.8-flash-next", ["openai-compatible"])]
    for mid, available in cases:
        entry = next(m for m in models_for() if m.id == mid)
        plan = activation_plan(entry, _REMOTE, available)
        assert mod.is_resolvable_slug(plan["provider"]), plan["provider"]
        cp = plan["custom_provider"]
        assert mod.is_resolvable_slug(cp["slug"]), cp["slug"]
        entry_names = {"custom:" + str(e.get("name") or "").lower()
                       .replace(" ", "-") for e in [cp["entry"]]}
        assert cp["slug"] in entry_names
        assert plan["model"] in cp["entry"]["models"]
    ollama = ModelEntry(id="ollama-chat", name="Ollama chat", modality="chat",
                        engines=("ollama",), source="a/b")
    plan = activation_plan(ollama, _REMOTE, ["ollama"])
    assert plan["provider"] == "custom"
    assert mod.is_resolvable_slug(plan["custom_provider"]["slug"])


def test_custom_provider_shape_parity_across_packages(monkeypatch):
    mod, _ = _load_providers_pkg(monkeypatch)
    from model_forge import tools

    entry_a = tools.custom_provider_entry(
        "glm-maya-s", name="gpu-box",
        base_url="http://gpu-box.example.com:8080/v1")
    entry_b = mod.custom_provider_entry(
        "glm-maya-s", name="gpu-box",
        base_url="http://gpu-box.example.com:8080/v1")
    assert entry_a == entry_b
    providers = [{"name": "gpu-box",
                  "base_url": "http://old-endpoint.example.com:9000/v1",
                  "models": {"x": {}}}]
    rows_a, merged_a = tools.merge_custom_provider_entry(entry_a, providers)
    rows_b, merged_b = mod.merge_custom_provider_entry(entry_b, providers)
    assert rows_a == rows_b and merged_a == merged_b
    assert tools._CUSTOM_PROVIDER_API_MODE == mod.API_MODE
    assert tools._NEXT_SESSION_NOTE == mod.NEXT_SESSION_NOTE


def _media_provider_fakes(monkeypatch):
    secret = types.ModuleType("agent.secret_scope")
    secret.get_secret_str = lambda name, default="": ""
    img = types.ModuleType("agent.image_gen_provider")
    img.DEFAULT_ASPECT_RATIO = "landscape"
    img.resolve_aspect_ratio = lambda v: "landscape"
    img.error_response = lambda **kw: {"success": False}
    img.success_response = lambda **kw: {"success": True}
    img.save_b64_image = lambda *a, **k: None

    class _ImageProvider:
        def name(self):
            raise NotImplementedError

    img.ImageGenProvider = _ImageProvider
    vid = types.ModuleType("agent.video_gen_provider")

    class _VideoBase:
        def __init__(self):
            self.calls = []

        def generate(self, prompt, **kwargs):
            self.calls.append(kwargs)
            return {"success": True}

    vid.OpenAICompatibleVideoGenProvider = _VideoBase
    vid.error_response = lambda **kw: {"success": False}
    vid.success_response = lambda **kw: {"success": True}
    agent_pkg = types.ModuleType("agent")
    agent_pkg.image_gen_provider = img
    agent_pkg.video_gen_provider = vid
    agent_pkg.secret_scope = secret
    fs_mod = types.ModuleType("agent.file_safety")
    fs_mod.raise_if_read_blocked = lambda path: None
    agent_pkg.file_safety = fs_mod
    openai_fake = types.ModuleType("openai")
    openai_fake.OpenAI = object
    httpx_fake = types.ModuleType("httpx")
    httpx_fake.get = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("httpx.get not stubbed in this test"))
    for name, m in (("agent", agent_pkg), ("agent.secret_scope", secret),
                    ("agent.image_gen_provider", img),
                    ("agent.video_gen_provider", vid),
                    ("agent.file_safety", fs_mod), ("openai", openai_fake),
                    ("httpx", httpx_fake)):
        monkeypatch.setitem(sys.modules, name, m)
    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.load_config_readonly = lambda: {}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg_mod)


def test_media_provider_vocabulary_matches_activate_hints(monkeypatch):
    # surface parity of state: the provider strings in the activate
    # media_setup hints are exactly the names the media providers register
    # and the model-forge provider row uses — one vocabulary everywhere.
    _media_provider_fakes(monkeypatch)
    from model_forge import tools

    def _load(pkg_dir, name):
        path = _REPO / pkg_dir / "__init__.py"
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        return mod

    img = _load("model-forge-image", "provider_image")
    vid = _load("model-forge-video", "provider_video")
    assert img.ModelForgeImageGenProvider().name == tools._MEDIA_PROVIDER_NAME
    assert vid.ModelForgeVideoGenProvider().name == tools._MEDIA_PROVIDER_NAME
    mod, registered = _load_providers_pkg(monkeypatch)
    row = next(p for p in registered if p.name == "model-forge")
    assert row.name == tools._MEDIA_PROVIDER_NAME
