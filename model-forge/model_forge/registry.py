"""Capability registry: modality -> engines -> models, with advisory hardware fit.

The registry is the plugin's manifest of what CAN run locally. Fit verdicts are
advisory estimates sized from published footprints; the engine that serves a
model does the authoritative memory sizing at launch. Core's managed llama.cpp
catalog (GGUF chat models, hardware-priced) is deliberately NOT duplicated here
— chat on llama.cpp defers to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Modalities, in demo-storyboard order.
MODALITIES: tuple[str, ...] = (
    "chat",
    "vision",
    "ocr",
    "image",
    "image_edit",
    "video",
    "3d",
    "music",
    "sfx",
    "voice_clone",
    "world_model",
)

MODALITY_LABELS: dict[str, str] = {
    "chat": "Chat / agent model",
    "vision": "Image understanding",
    "ocr": "Text extraction from images",
    "image": "Image generation",
    "image_edit": "Image editing / style transfer",
    "video": "Video generation (t2v / i2v)",
    "3d": "3D model generation (image to mesh)",
    "music": "Music generation",
    "sfx": "Sound effects",
    "voice_clone": "Voice / voice clone (TTS)",
    "world_model": "World simulation",
}


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    modality: str
    engines: tuple[str, ...]        # engine ids that can serve it
    source: str                     # HF repo id or engine-native id
    size_gb: float = 0.0            # download size, approximate
    vram_gb_min: float = 0.0        # min VRAM for a comfortable run
    ram_gb_min: float = 0.0         # min system RAM where relevant
    engine_managed_fit: bool = False  # engine sizes itself (Strata/Maya style)
    status: str = "known"           # validated | known | stub
    note: str = ""
    engine_models: dict = field(default_factory=dict)  # engine id -> engine-native model id


@dataclass(frozen=True)
class HardwareProfile:
    gpus: tuple[dict, ...] = ()
    total_vram_gb: float = 0.0
    ram_gb: float = 0.0
    platform: str = "unknown"

    @property
    def budget_gb(self) -> float:
        """Advisory pool: VRAM plus the RAM an engine may stream weights from."""
        return self.total_vram_gb + max(0.0, self.ram_gb - 8.0)


MODELS: tuple[ModelEntry, ...] = (
    # --- chat (llama.cpp GGUFs defer to core's Local Models catalog) ---
    ModelEntry(
        id="core-gguf-catalog",
        name="Hermes core GGUF catalog",
        modality="chat",
        engines=("llamacpp",),
        source="hermes:local-models",
        engine_managed_fit=True,
        status="validated",
        note="Use the built-in Local Models pane — core prices GGUFs for this machine.",
    ),
    ModelEntry(
        id="qwen3.8-flash-next",
        name="Qwen3.8 Flash Next (125B MoE)",
        modality="chat",
        engines=("strata", "maya"),
        source="unsloth/Qwen3.8-Flash-Next-GGUF",
        size_gb=66.0,
        ram_gb_min=64.0,
        engine_managed_fit=True,
        status="known",
        note="Strata streams experts RAM->GPU; 12 GB VRAM cards work as hot cache.",
    ),
    ModelEntry(
        id="deepseek-v4.1-flash",
        name="DeepSeek-V4.1-Flash-1017 (748B MoE)",
        modality="chat",
        engines=("strata", "maya"),
        source="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        size_gb=277.0,
        ram_gb_min=64.0,
        engine_managed_fit=True,
        status="known",
        note="Maya tiers run it from 128 GB machines upward; Strata on PC/Mac.",
        engine_models={"maya": "deepseek-maya-s"},
    ),
    ModelEntry(
        id="minimax-m3.1-flash",
        name="MiniMax-M3.1-Flash (428B MoE)",
        modality="chat",
        engines=("maya",),
        source="MiniMaxAI/MiniMax-M3.1-Flash",
        size_gb=93.0,
        ram_gb_min=64.0,
        engine_managed_fit=True,
        status="known",
        note="Maya-XS fits a 128 GB machine; vision + thinking levels; "
             "MiniMax Community License (check commercial terms).",
        engine_models={"maya": "minimax-maya-xs"},
    ),
    ModelEntry(
        id="glm-5.3-flash",
        name="GLM-5.3 Flash (321B MoE)",
        modality="chat",
        engines=("maya",),
        source="zai-org/GLM-5.3-Flash",
        size_gb=73.0,
        ram_gb_min=64.0,
        engine_managed_fit=True,
        status="known",
        note="Maya tiers (XS..L) self-select from your memory pool; vision capable.",
        engine_models={"maya": "glm-maya-s"},
    ),
    # --- vision / ocr ---
    ModelEntry(
        id="minicpm-v-4.6",
        name="MiniCPM-V 4.6",
        modality="vision",
        engines=("vllm",),
        source="openbmb/MiniCPM-V-4.6",
        size_gb=2.6,
        vram_gb_min=4.0,
        status="validated",
        note="Small vision sidecar; any OpenAI-compatible multimodal server works.",
    ),
    ModelEntry(
        id="deepseek-ocr-2",
        name="DeepSeek-OCR-2",
        modality="ocr",
        engines=("vllm",),
        source="deepseek-ai/DeepSeek-OCR-2",
        size_gb=6.8,
        vram_gb_min=10.0,
        status="validated",
        note="Max context 8192; needs trust-remote-code on vLLM.",
    ),
    # --- image / image edit ---
    ModelEntry(
        id="krea-2-turbo",
        name="Krea-2 Turbo",
        modality="image",
        engines=("comfyui",),
        source="Comfy-Org/Krea-2",
        size_gb=19.0,
        vram_gb_min=16.0,
        status="validated",
        note="Aesthetic/style-first default; 8-step turbo, cfg 1.0.",
    ),
    ModelEntry(
        id="qwen-image-2.1",
        name="Qwen-Image 2.1",
        modality="image",
        engines=("vllm-omni", "comfyui"),
        source="Qwen/Qwen-Image-2.1",
        size_gb=25.0,
        vram_gb_min=20.0,
        status="validated",
        note="Native 2K, typography, RGBA transparency.",
    ),
    ModelEntry(
        id="qwen-image-2.1-edit",
        name="Qwen-Image 2.1 Edit",
        modality="image_edit",
        engines=("vllm-omni",),
        source="Qwen/Qwen-Image-2.1",
        size_gb=25.0,
        vram_gb_min=20.0,
        status="validated",
        note="Up to 4 reference images; ordinal <image1>.. phrase binding.",
    ),
    # --- video ---
    ModelEntry(
        id="ltx-2.5",
        name="LTX-2.5 (t2v / i2v)",
        modality="video",
        engines=("comfyui",),
        source="Lightricks/LTX-2.5",
        size_gb=20.0,
        vram_gb_min=16.0,
        status="validated",
        note="i2v keeps subject identity; ComfyUI lane, torch >= 2.7.",
    ),
    ModelEntry(
        id="minimax-h3",
        name="MiniMax H3 (image/video)",
        modality="video",
        engines=("comfyui",),
        source="Comfy-Org/MiniMax-H3",
        size_gb=25.0,
        vram_gb_min=20.0,
        status="known",
        note="i2v/t2v; 1024x576 @ 24fps fits 24 GB with --novram.",
    ),
    # --- 3d ---
    ModelEntry(
        id="hunyuan3d-2",
        name="Hunyuan3D-2 (image to GLB)",
        modality="3d",
        engines=("comfyui",),
        source="Comfy-Org/hunyuan3D_2.0_repackaged",
        size_gb=5.0,
        vram_gb_min=10.0,
        status="validated",
        note="Single-view image to textured mesh; printable after repair.",
    ),
    ModelEntry(
        id="trellis.2-4b",
        name="TRELLIS.2-4B",
        modality="3d",
        engines=("comfyui", "python"),
        source="microsoft/TRELLIS.2",
        size_gb=8.0,
        vram_gb_min=12.0,
        status="known",
        note="Image to 3D, GLB/PLY output.",
    ),
    # --- audio family ---
    ModelEntry(
        id="ace-step",
        name="ACE-Step 1.5 turbo",
        modality="music",
        engines=("python",),
        source="tetherto/ACE-Step",
        size_gb=8.0,
        vram_gb_min=8.0,
        status="validated",
        note="Text-to-music; also serves on CPU via an OpenAI-style audio API.",
    ),
    ModelEntry(
        id="musicgen-small",
        name="MusicGen Small",
        modality="music",
        engines=("python",),
        source="facebook/musicgen-small",
        size_gb=2.0,
        vram_gb_min=4.0,
        status="known",
        note="Lightweight text-to-music baseline.",
    ),
    ModelEntry(
        id="audiogen",
        name="AudioGen",
        modality="sfx",
        engines=("python",),
        source="facebook/audiogen-medium",
        size_gb=4.0,
        vram_gb_min=6.0,
        status="known",
        note="Text-to-sound-effects.",
    ),
    ModelEntry(
        id="stable-audio-open",
        name="Stable Audio Open",
        modality="sfx",
        engines=("python",),
        source="stabilityai/stable-audio-open-1.0",
        size_gb=6.0,
        vram_gb_min=10.0,
        status="known",
        note="Open SFX and short music generation.",
    ),
    ModelEntry(
        id="chatterbox-t3-turbo",
        name="Chatterbox t3-turbo",
        modality="voice_clone",
        engines=("python",),
        source="tetherto/chatterbox",
        size_gb=3.0,
        vram_gb_min=6.0,
        status="validated",
        note="TTS with voice clone via reference audio.",
    ),
    # --- world model: honest empty slot ---
    ModelEntry(
        id="world-model-slot",
        name="World model (engine wanted)",
        modality="world_model",
        engines=(),
        source="",
        status="stub",
        note="No local open engine with a stable API exists yet; the adapter "
             "slot ships so the flow works the day one lands.",
    ),
)

_BY_ID = {m.id: m for m in MODELS}


def models_for(modality: str | None = None, engine: str | None = None) -> list[ModelEntry]:
    out = []
    for m in MODELS:
        if modality and m.modality != modality:
            continue
        if engine and engine not in m.engines:
            continue
        out.append(m)
    return out


def fit_verdict(m: ModelEntry, hw: HardwareProfile) -> str:
    """Advisory fit verdict: fits_gpu | fits_memory | needs_gpu | too_big |
    engine_managed | unknown | unavailable."""
    if m.status == "stub":
        return "unavailable"
    if m.engine_managed_fit:
        return "engine_managed"
    if not m.size_gb:
        return "unknown"
    if m.vram_gb_min and hw.total_vram_gb >= m.vram_gb_min:
        return "fits_gpu"
    if m.vram_gb_min and hw.total_vram_gb == 0:
        return "needs_gpu"
    if m.size_gb <= hw.budget_gb:
        return "fits_memory"
    return "too_big"
