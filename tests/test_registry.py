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
    small = HardwareProfile(gpus=(), total_vram_gb=0.0, ram_gb=8.0, platform="linux")

    hunyuan = next(m for m in MODELS if m.id == "hunyuan3d-2")
    assert fit_verdict(hunyuan, big) == "fits_gpu"
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
