"""Registry integrity + fit-verdict tests (pure functions, no network)."""

from model_forge.hardware import HardwareProfile
from model_forge.registry import (
    MODALITIES, MODELS, fit_verdict, models_for,
)


def test_registry_ids_unique():
    ids = [m.id for m in MODELS]
    assert len(ids) == len(set(ids))


def test_registry_modalities_known():
    for m in MODELS:
        assert m.modality in MODALITIES, m.id


def test_every_modality_has_an_entry():
    covered = {m.modality for m in MODELS}
    assert covered == set(MODALITIES)


def test_models_for_filters():
    assert models_for(modality="3d")
    assert all(m.modality == "3d" for m in models_for(modality="3d"))
    assert models_for(engine="comfyui")
    assert all("comfyui" in m.engines for m in models_for(engine="comfyui"))


def test_fit_verdicts():
    big = HardwareProfile(gpus=({"name": "RTX 4090", "vram_gb": 24.0},),
                          total_vram_gb=24.0, ram_gb=64.0, platform="linux")

    hunyuan = next(m for m in MODELS if m.id == "hunyuan3d-2")
    assert fit_verdict(hunyuan, big) == "fits_gpu"
    # small real GPU: 4 GB VRAM + 8 GB RAM cannot hold the 5 GB model
    small = HardwareProfile(gpus=({"name": "GTX 1650", "vram_gb": 4.0},),
                            total_vram_gb=4.0, ram_gb=8.0, platform="linux")
    assert fit_verdict(hunyuan, small) == "too_big"

    flash = next(m for m in MODELS if m.id == "qwen3.8-flash-next")
    assert fit_verdict(flash, big) == "engine_managed"

    slot = next(m for m in MODELS if m.id == "world-model-slot")
    assert fit_verdict(slot, big) == "unavailable"

    # memory pool path: too big for VRAM alone, fits VRAM+RAM
    deepseek = next(m for m in MODELS if m.id == "deepseek-ocr-2")
    mid = HardwareProfile(gpus=({"name": "GTX 1660", "vram_gb": 6.0},),
                          total_vram_gb=6.0, ram_gb=32.0, platform="linux")
    assert fit_verdict(deepseek, mid) == "fits_memory"

    # no GPU at all: a GPU-bound model must not read as "fits_memory"
    nogpu = HardwareProfile(gpus=(), total_vram_gb=0.0, ram_gb=64.0, platform="linux")
    assert fit_verdict(hunyuan, nogpu) == "needs_gpu"


def test_strata_launch_cmd():
    from model_forge.engines import strata as strata_adapter

    cmd = strata_adapter.launch_cmd("deepseek41", "ist-das-2.00bpw", plan_only=True)
    assert cmd == ["strata", "launch", "--family", "deepseek41",
                   "--model", "ist-das-2.00bpw", "--plan"]
    assert strata_adapter.launch_cmd("qwen38", "pack") == [
        "strata", "launch", "--family", "qwen38", "--model", "pack"]
    try:
        strata_adapter.launch_cmd("", "pack")
        assert False, "empty family must raise"
    except ValueError:
        pass


def test_engine_status_defaults():
    from model_forge.engines.base import EngineStatus

    e = EngineStatus(id="x", display_name="X", available=False)
    assert e.models_loaded == () and e.modalities == ()


# --- registry truth: sources, gated repos, honest notes ----------------------

def test_model_entry_shape_backward_compatible():
    m = MODELS[0]
    for attr in ("id", "name", "modality", "engines", "source", "size_gb",
                 "vram_gb_min", "ram_gb_min", "engine_managed_fit", "status",
                 "note", "engine_models", "released"):
        assert hasattr(m, attr), attr


def test_sources_point_at_public_repos():
    # corrected against the HuggingFace model-info API (each verified live):
    # the old sources were private/placeholder namespaces that fail with 401
    expected = {
        "trellis.2-4b": "microsoft/TRELLIS.2-4B",
        "ace-step": "ACE-Step/acestep-v15-xl-turbo",
        "chatterbox-turbo": "ResembleAI/chatterbox-turbo",
        "hunyuan3d-2": "Comfy-Org/Hunyuan-3D-2.0",
    }
    for mid, src in expected.items():
        m = next(x for x in MODELS if x.id == mid)
        assert m.source == src, mid
    for m in MODELS:
        assert not m.source.startswith("tetherto/"), m.id
        # the old non-canonical hunyuan id redirects to the canonical repo;
        # nothing may keep referencing it
        assert m.source != "Comfy-Org/hunyuan3D_2.0_repackaged", m.id


def test_ace_step_sizes_match_verified_repo_footprint():
    # HF model-info API (blobs=true) 2026-09: the four model shards total
    # 19,953,378,268 bytes (~20 GB); the old 8 GB size fields were stale
    m = next(x for x in MODELS if x.id == "ace-step")
    assert m.size_gb == 20.0
    # VRAM floor can never honestly sit below the weights footprint
    assert m.vram_gb_min == 20.0


def test_chatterbox_name_and_id_match_public_repo():
    # the verified public repo is ResembleAI/chatterbox-turbo; no official
    # "t3" repo exists, so neither the id nor the name may claim one
    m = next(x for x in MODELS if x.id == "chatterbox-turbo")
    assert m.name == "Chatterbox Turbo"
    assert m.source == "ResembleAI/chatterbox-turbo"
    assert all("t3-turbo" not in x.id for x in MODELS)


def test_hunyuan3d_source_is_canonical_hf_id():
    # Comfy-Org/hunyuan3D_2.0_repackaged resolves (307) to the canonical
    # Comfy-Org/Hunyuan-3D-2.0, which serves the dit-v2 weights (split_files/)
    m = next(x for x in MODELS if x.id == "hunyuan3d-2")
    assert m.source == "Comfy-Org/Hunyuan-3D-2.0"


def test_gated_repos_are_noted():
    for mid in ("ltx-2.5", "stable-audio-open"):
        m = next(x for x in MODELS if x.id == mid)
        assert "gated repo: HF account + license acceptance required" in m.note, mid


def test_comfyui_notes_do_not_overclaim():
    for m in MODELS:
        if "comfyui" in m.engines:
            assert "download works" in m.note, m.id
            assert "generation is not plugin-driven yet" in m.note, m.id
            assert "workflow lane also works" not in m.note, m.id


def test_python_notes_state_unmanaged_env():
    for m in MODELS:
        if "python" in m.engines:
            assert "custom Python environment the plugin does not manage" in m.note, m.id
            assert "Model Forge cannot run it" in m.note, m.id


def test_status_values_are_known():
    for m in MODELS:
        assert m.status in ("validated", "known", "stub"), (m.id, m.status)
