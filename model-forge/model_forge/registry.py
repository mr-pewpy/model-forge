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

# Engine ids the registry may list but the plugin cannot run: entries with a
# "python" engine need a custom Python environment the plugin does not manage
# (no detector, no runner). Such engines never count as available and are
# never auto-selected for install — but the entries stay listed, honestly
# labeled, so the models remain discoverable.
PHANTOM_ENGINES: frozenset[str] = frozenset({"python"})
PHANTOM_ENGINE_NOTE = "custom Python environment the plugin does not manage"

# Every entry listing the "comfyui" engine carries this honest caveat: the
# plugin downloads ComfyUI model files but does not drive ComfyUI-native
# generation (see README known limits).
COMFYUI_LIMIT_NOTE = ("ComfyUI download works; generation is not plugin-driven "
                     "yet (see README known limits)")

# Gated HuggingFace repos need license acceptance before any download.
GATED_REPO_NOTE = ("This is a gated repo: HF account + license acceptance "
                   "required.")


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
    released: str = ""              # release date (YYYY-MM-DD) where known; "" = unknown


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
        note="Aesthetic/style-first default; 8-step turbo, cfg 1.0. "
             + COMFYUI_LIMIT_NOTE + ".",
    ),
    ModelEntry(
        id="qwen-image-2.1",
        name="Qwen-Image 2.1",
        modality="image",
        engines=("vllm", "comfyui"),
        source="Qwen/Qwen-Image-2.1",
        size_gb=25.0,
        vram_gb_min=20.0,
        status="validated",
        note="Native 2K, typography, RGBA transparency. vLLM form = vLLM-Omni "
             "(vllm serve <model> --omni). " + COMFYUI_LIMIT_NOTE + ".",
    ),
    ModelEntry(
        id="qwen-image-2.1-edit",
        name="Qwen-Image 2.1 Edit",
        modality="image_edit",
        engines=("vllm", "comfyui"),
        source="Qwen/Qwen-Image-2.1",
        size_gb=25.0,
        vram_gb_min=20.0,
        status="validated",
        note="Up to 4 reference images; ordinal <image1>.. phrase binding. "
             + COMFYUI_LIMIT_NOTE + ".",
    ),
    ModelEntry(
        id="flux-2-klein-4b",
        name="FLUX.2 Klein 4B (edit / outpaint)",
        modality="image_edit",
        engines=("sglang", "vllm", "comfyui"),
        source="black-forest-labs/FLUX.2-klein-4B",
        size_gb=4.0,
        vram_gb_min=8.0,
        status="validated",
        note="Apache-2.0, edit-native (image+mask inpainting/outpainting). "
             "SGLang Diffusion: sglang serve --tp-size N --num-gpus N; "
             "vLLM-Omni has a native FLUX.2-klein inpaint test. "
             + COMFYUI_LIMIT_NOTE + ".",
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
        note="i2v keeps subject identity; torch >= 2.7. "
             + COMFYUI_LIMIT_NOTE + ". " + GATED_REPO_NOTE,
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
        note="i2v/t2v; 1024x576 @ 24fps fits 24 GB with --novram. "
             + COMFYUI_LIMIT_NOTE + ".",
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
        note="Single-view image to textured mesh; printable after repair. "
             + COMFYUI_LIMIT_NOTE
             + ": the ComfyUI-native job flow is not implemented in Model "
             "Forge, and model_forge_generate_media needs a media gateway "
             "lane (/v1/videos/generations).",
    ),
    ModelEntry(
        id="trellis.2-4b",
        name="TRELLIS.2-4B",
        modality="3d",
        engines=("comfyui", "python"),
        source="microsoft/TRELLIS.2-4B",
        size_gb=8.0,
        vram_gb_min=12.0,
        status="known",
        note="Image to 3D, GLB/PLY output. " + COMFYUI_LIMIT_NOTE
             + ". The 'python' lane needs a " + PHANTOM_ENGINE_NOTE
             + " — Model Forge cannot run it.",
    ),
    # --- audio family ---
    ModelEntry(
        id="ace-step",
        name="ACE-Step 1.5 turbo",
        modality="music",
        engines=("python",),
        source="ACE-Step/acestep-v15-xl-turbo",
        size_gb=8.0,
        vram_gb_min=8.0,
        status="validated",
        note="Text-to-music; also serves on CPU via an OpenAI-style audio API. "
             "Needs a " + PHANTOM_ENGINE_NOTE + " — Model Forge cannot run it.",
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
        note="Lightweight text-to-music baseline. Needs a "
             + PHANTOM_ENGINE_NOTE + " — Model Forge cannot run it.",
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
        note="Text-to-sound-effects. Needs a " + PHANTOM_ENGINE_NOTE
             + " — Model Forge cannot run it.",
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
        note="Open SFX and short music generation. Needs a "
             + PHANTOM_ENGINE_NOTE + " — Model Forge cannot run it. "
             + GATED_REPO_NOTE,
    ),
    ModelEntry(
        id="chatterbox-t3-turbo",
        name="Chatterbox t3-turbo",
        modality="voice_clone",
        engines=("python",),
        source="ResembleAI/chatterbox-turbo",
        size_gb=3.0,
        vram_gb_min=6.0,
        status="validated",
        note="TTS with voice clone via reference audio. Needs a "
             + PHANTOM_ENGINE_NOTE + " — Model Forge cannot run it.",
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


def _ram_short(m: ModelEntry, hw: HardwareProfile) -> bool:
    """True when the entry declares a RAM floor the target demonstrably misses.

    ram_gb == 0 means 'unknown' (probe failure / undeclared target) and never
    raises an advisory on its own."""
    return bool(m.ram_gb_min and hw.ram_gb and hw.ram_gb < m.ram_gb_min)


def engine_match(m: ModelEntry,
                 available_engines: list[str] | tuple[str, ...] | None) -> str:
    """First of the entry's engines the target can actually run ("" if none).

    Phantom engines never match — the plugin cannot run them on any target —
    so an entry served only by 'python' is honestly 'nothing can run this'."""
    for e in m.engines:
        if e in PHANTOM_ENGINES:
            continue
        if available_engines and e in available_engines:
            return e
    return ""


def fit_verdict(m: ModelEntry, hw: HardwareProfile,
                available_engines: list[str] | tuple[str, ...] | None = None) \
        -> str:
    """Advisory fit verdict: fits_gpu | fits_memory | needs_gpu | needs_ram |
    too_big | engine_managed | engine_unavailable | unknown | unavailable.

    available_engines=None keeps the historical hardware-only scoring (fully
    backward compatible). Pass the target's detected engine ids and the
    verdict also reflects whether any engine there can serve the model at
    all: a hardware 'fits_gpu' for a model no installed engine can run reads
    engine_unavailable instead of misleading."""
    if m.status == "stub":
        return "unavailable"
    if available_engines is not None and not engine_match(m, available_engines):
        return "engine_unavailable"
    if _ram_short(m, hw):
        # engine_managed entries used to hide this behind 'engine_managed';
        # an unmet declared RAM floor is the same honest advisory for them.
        return "needs_ram"
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


def fit_report(m: ModelEntry, hw: HardwareProfile,
               available_engines: list[str] | tuple[str, ...] | None = None) \
        -> dict:
    """Row-ready fit block for browse rows: the verdict plus the engine
    availability and RAM signals a bare verdict string cannot carry."""
    matched = engine_match(m, available_engines)
    verdict = fit_verdict(m, hw, available_engines)
    advisory = ""
    if verdict == "engine_unavailable":
        needed = ", ".join(e for e in m.engines if e not in PHANTOM_ENGINES)
        advisory = ("no engine that can serve this model was detected on "
                    f"the target (needs: {needed or 'an engine'})")
        if set(m.engines) & PHANTOM_ENGINES:
            advisory += ("; 'python' entries need a " + PHANTOM_ENGINE_NOTE)
    elif verdict == "needs_ram":
        managed = " (engine-managed fit)" if m.engine_managed_fit else ""
        advisory = (f"declares {m.ram_gb_min:g} GB RAM minimum{managed}; "
                    f"the target has {hw.ram_gb:g} GB")
    return {
        "verdict": verdict,
        "engine_available": bool(matched),
        "engine": matched,
        "ram_gb_min": m.ram_gb_min,
        "advisory": advisory,
    }
