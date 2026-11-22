"""Mutation-killing tests (A19 gate 7) — pin the exact behaviours the
surviving mutants in registry.py altered (dataclass defaults + two fit-advisory
branches), so those mutants are KILLED and the measured mutation score for the
pure modules reaches its target. See tests/mutation_harness.py.
"""

from __future__ import annotations

from model_forge.hardware import HardwareProfile
from model_forge.registry import (
    MODELS, ModelEntry, fit_report, fit_verdict, native_id_for,
)


# --- ModelEntry defaults (kills the const mutants on field defaults) -------

def test_model_entry_defaults_pinned():
    e = ModelEntry(id="x", name="X", modality="chat", engines=("maya",),
                   source="s")
    assert e.status == "known"          # default status is "known", not "stub"
    assert e.engine_models == {}        # default empty map
    assert e.engine_managed_fit is False
    assert e.released == ""
    assert e.note == ""


def test_hardware_profile_defaults_pinned():
    hw = HardwareProfile()
    assert hw.gpus == ()
    assert hw.total_vram_gb == 0.0
    assert hw.ram_gb == 0.0


def test_world_model_entry_is_stub():
    e = next(m for m in MODELS if m.modality == "world_model")
    assert e.status == "stub"
    assert e.engines == ()


# --- fit_verdict branches (kills the cmp mutant on fits_gpu) --------------

def test_fit_verdict_fits_gpu_branch():
    e = ModelEntry(id="x", name="X", modality="chat", engines=("llamacpp",),
                   source="s", size_gb=10.0, vram_gb_min=8.0)
    hw = HardwareProfile(total_vram_gb=16.0, ram_gb=64.0, platform="linux")
    assert fit_verdict(e, hw) == "fits_gpu"


def test_fit_verdict_engine_managed():
    e = ModelEntry(id="x", name="X", modality="chat", engines=("maya",),
                   source="s", size_gb=10.0, engine_managed_fit=True)
    hw = HardwareProfile(total_vram_gb=16.0, ram_gb=64.0, platform="linux")
    assert fit_verdict(e, hw) == "engine_managed"


# --- fit_report advisory labels (kills the const mutant on managed label) --

def test_fit_report_needs_ram_managed_label():
    e = ModelEntry(id="x", name="X", modality="chat", engines=("maya",),
                   source="s", size_gb=10.0, ram_gb_min=128.0,
                   engine_managed_fit=True)
    hw = HardwareProfile(total_vram_gb=0.0, ram_gb=64.0, platform="linux")
    r = fit_report(e, hw, ["maya"])
    assert r["verdict"] == "needs_ram"
    assert "engine-managed fit" in r["advisory"]


def test_fit_report_engine_unavailable_advisory():
    e = ModelEntry(id="x", name="X", modality="chat", engines=("maya",),
                   source="s", size_gb=10.0, vram_gb_min=8.0)
    hw = HardwareProfile(total_vram_gb=0.0, ram_gb=64.0, platform="linux")
    r = fit_report(e, hw, ["llamacpp"])  # maya not available
    assert r["verdict"] == "engine_unavailable"
    assert "no engine" in r["advisory"]


# --- native_id_for idempotence (kills const mutants near the reshape) -----

def test_native_id_for_unknown_id_identity():
    assert native_id_for("not/in/registry", "maya") == "not/in/registry"


# --- frozen dataclass + exact entry fields (kills remaining const mutants) --

def test_model_entry_is_frozen():
    e = ModelEntry(id="x", name="X", modality="chat", engines=("maya",),
                   source="s")
    raised = False
    try:
        e.id = "y"
    except Exception:
        raised = True
    assert raised, "ModelEntry must be frozen (assignment must raise)"


def test_hardware_profile_is_frozen():
    hw = HardwareProfile()
    raised = False
    try:
        hw.total_vram_gb = 9.0
    except Exception:
        raised = True
    assert raised, "HardwareProfile must be frozen (assignment must raise)"


def test_engine_managed_entries_pinned():
    # these curated entries declare engine-managed fit (engine sizes itself)
    for mid in ("core-gguf-catalog", "qwen3.8-flash-next", "deepseek-v4.1-flash",
                "minimax-m3.1-flash", "glm-5.3-flash"):
        e = next(m for m in MODELS if m.id == mid)
        assert e.engine_managed_fit is True, mid


def test_fit_report_needs_ram_unmanaged_label_empty():
    # engine_managed_fit=False -> the managed label is empty (the else branch)
    e = ModelEntry(id="x", name="X", modality="chat", engines=("llamacpp",),
                   source="s", size_gb=10.0, ram_gb_min=128.0,
                   engine_managed_fit=False)
    hw = HardwareProfile(total_vram_gb=0.0, ram_gb=64.0, platform="linux")
    r = fit_report(e, hw, ["llamacpp"])
    assert r["verdict"] == "needs_ram"
    assert "engine-managed fit" not in r["advisory"]


def test_world_model_stub_source_is_empty():
    e = next(m for m in MODELS if m.modality == "world_model")
    assert e.source == ""


def test_fit_verdict_exact_vram_boundary_is_fits_gpu():
    # total_vram_gb == vram_gb_min exactly -> fits_gpu (the >= boundary, not >)
    e = ModelEntry(id="x", name="X", modality="chat", engines=("llamacpp",),
                   source="s", size_gb=10.0, vram_gb_min=16.0)
    hw = HardwareProfile(total_vram_gb=16.0, ram_gb=64.0, platform="linux")
    assert fit_verdict(e, hw) == "fits_gpu"
