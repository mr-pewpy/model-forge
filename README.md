# Model Forge

**Local models for every modality — find your GPU, pick a model, and Hermes
runs it. No account, no API key, nothing leaves your machine.**

Model Forge is a Hermes Agent plugin that makes local AI a first-class
experience across every modality: chat, vision, OCR, image generation, image
editing, video, 3D meshes, music, sound effects, and voice. It finds your
hardware, shows you which models fit, maps each model to the right inference
engine (llama.cpp, Strata, Ollama, vLLM, ComfyUI, and more), and wires each
model into Hermes as a provider — so every native tool (chat,
`image_generate`, `video_generate`, `vision`, TTS) can use it.

## Why

Hermes already runs GGUF chat models locally (the built-in **Local Models**
pane handles those beautifully). But the moment you want an image, a video, a
3D model, or a voice, you are back to hand-installing engines and guessing at
VRAM numbers. Model Forge closes that gap: one registry of
**modality → engine → model → hardware fit**, so the flow is always the same:

> "I want X" → Model Forge picks what fits → pulls the model (or hands you
> the exact command) → you start the engine → Model Forge connects it to
> Hermes → you use it.

One honest caveat up front: **Model Forge does not start engines.** Install
either downloads model files, drives a running engine's own pull API, or
prints the exact command for you to run. Engines stay yours to launch.

## The demo (90 seconds, one gaming PC, zero cloud)

1. **Chat** — pick a chat model for your GPU; Hermes runs on it
2. **Image** — generate an image; Model Forge pulls the image model
3. **Edit** — edit that image with an image-edit model
4. **3D** — turn it into a printable 3D model
5. **Video** — generate a promo video from the 3D render (i2v)
6. **Voice** — give it a voice (TTS / voice clone) and a soundtrack (music, SFX)

Each step visibly downloads the model it needs.

## Tools

| Tool | What it does |
|------|--------------|
| `model_forge_status` | GPUs, VRAM, RAM, engines per compute target, pull credentials |
| `model_forge_browse` | Models by modality with an advisory fit verdict per target, plus a live HuggingFace "fresh" section (`fresh=false` = curated only) |
| `model_forge_target_add` | Register a compute target (LAN LXC, cloud GPU box) — asks for your approval first |
| `model_forge_install` | Pull a model: plugin-side HF download (sha256, resume, job polling), a running Maya's HTTP pull, or the exact engine CLI command to run yourself |
| `model_forge_activate` | Connect a running chat model as Hermes' default (plan-only; `apply=true` writes config after human approval) |
| `model_forge_generate_media` | 3D (image→GLB), music, SFX via engine job APIs |

Plus `tts/model_forge_tts.py` — a config-driven TTS/voice-clone command
provider for any OpenAI-style `/v1/audio/speech` endpoint (see `tts/`), and
a **cockpit desktop pane** (`model-forge-desktop/`) showing every compute
target with live engine probes and model counts.

## Model discovery

`model_forge_browse` has two layers:

- **Curated catalog** (`model_forge/registry.py`) — hand-maintained entries
  with approximate size, minimum VRAM, and an advisory **fit verdict** per
  target: `fits_gpu`, `fits_memory`, `needs_gpu`, `too_big`, `engine_managed`
  (Strata/Maya size themselves at launch), `unknown`, or `unavailable`
  (stub). Verdicts are estimates from published footprints — the serving
  engine does the authoritative memory sizing.
- **Fresh (live HuggingFace)** — newest public models per modality lane,
  sorted by last-modified date. Every row carries its date, download count,
  pipeline tag, and link, and is labeled **fresh/unvetted**: no fit
  estimate, no engine mapping, not installable — discovery only.

Pass `fresh=false` to skip the live layer (no network call, curated catalog
only). Live results are cached for one hour and deduplicated against the
curated entries. When HuggingFace is unreachable the fresh section fails
soft to an `unavailable` marker and the curated catalog keeps working.

To add a curated entry, append a `ModelEntry` to `MODELS` in
`model_forge/registry.py` — field guide and fit rules in
`docs/model-discovery.md`.

## Hardware support

`model_forge_status` probes each compute target. Every probe is fail-soft:
a missing tool or unreadable file degrades to "no data", never an error.

- **NVIDIA** — `nvidia-smi` (name + VRAM per GPU)
- **AMD** — `rocm-smi` and amdgpu sysfs (`mem_info_vram_total`); both are
  probed
- **Apple Silicon** — unified memory: the honest VRAM budget is total RAM;
  the GPU is labeled `unified memory` and the platform reads `darwin-arm64`
  so it is never mistaken for discrete VRAM
- **RAM** — POSIX `sysconf`, with a `GlobalMemoryStatusEx` fallback on
  Windows (0.0 only when both fail)

Remote targets use the `vram_gb` / `ram_gb` budgets you declare when you
register them.

## Engines

| Engine | Modalities | Status |
|--------|-----------|--------|
| llama.cpp / llama-server (incl. Hermes' managed runtime) | chat, vision | adapter done |
| Ollama | chat, vision, ocr | adapter done |
| Strata | chat, vision (125B MoE on consumer GPUs) | adapter done (detect + `strata launch` command) |
| Project Maya | chat, vision, ocr (321B–748B MoE, 1–16 GPUs) | adapter done (detect + Models-page API) |
| vLLM / vLLM-Omni | chat, vision, ocr, image, image_edit | adapter done (detect + omni route probe) |
| SGLang / SGLang Diffusion | chat, vision, image, image_edit | adapter done (detect + diffusion route probe) |
| ComfyUI | image, image_edit, video, 3d | adapter done (detect) |
| Any OpenAI-compatible endpoint | chat, vision, ocr, image | generic detector (never mislabels the engine) |
| Python engines (ACE-Step, Chatterbox, AudioGen, MusicGen) | music, sfx, voice_clone | adapter next |

GGUF chat models are served by Hermes' core Local Models catalog — Model Forge
never duplicates that pricing; it complements it.

## Bundled provider plugins

This repo also ships three provider packages:

- **`model-forge-providers/`** (`kind: model-provider`): registers `strata`
  and `maya` as first-class Hermes model providers (keyless, local — the LM
  Studio profile pattern), so `hermes model` / `--provider strata` can run
  Hermes directly on those engines.
- **`model-forge-image/`** (`kind: backend`): routes `image_generate` to ANY
  OpenAI-compatible images API (weights broker, vLLM-Omni, SGLang Diffusion)
  — endpoint via **Settings → Plugins** or `MODEL_FORGE_IMAGE_BASE_URL`.
- **`model-forge-video/`** (`kind: backend`): routes `video_generate` to any
  OpenAI video job API — endpoint via **Settings → Plugins** or
  `MODEL_FORGE_VIDEO_BASE_URL`.

## Pull credentials (Hugging Face / ModelScope)

**Settings → Plugins → Model Forge** has two optional masked fields:

- **Hugging Face token** (`HF_TOKEN`) — unlocks gated repos, full-speed pulls
- **ModelScope token** (`MODELSCOPE_API_TOKEN`) — authenticated mirror pulls

Both are optional; without them everything public still works. Tokens are
stored through Hermes' credential route (never in config files), read only
via the secret scope, and sent **only** to their own host family
(`huggingface.co` / `modelscope.cn` — enforced in code, unit-tested).
`model_forge_status` reports which tokens are configured (booleans only).

## Known limits

Honest scoping — current limits, and what does not work yet:

- **Install is registry-only.** Fresh (live) rows are discovery, not install
  targets; `model_forge_install` resolves curated `model_id`s only.
- **File pulls need the exact repo file path.** Pass `engine_model=` with
  the file inside the source repo (e.g. `model.safetensors`); the registry
  does not guess filenames.
- **music and sfx share one HuggingFace lane** (`text-to-audio`) — HF has
  no music-vs-sfx pipeline tags.
- **World model is an empty stub.** No local open engine with a stable API
  exists yet; the slot ships so the flow works the day one lands.
- **`model_forge_generate_media` speaks the weights-broker job API only**
  (`/v1/videos/generations`, `/v1/audio/music`, `/v1/audio/sfx`). The
  ComfyUI-native job flow (`/prompt` + `/history`) is not implemented.
- **No engine lifecycle.** Model Forge never installs or starts engines —
  it detects them, pulls model files, and emits run commands.

## Disclosure (what this plugin does on your machine)

- **Network:** probes compute-target engine endpoints (llama-server `/props`,
  Ollama `/api/tags`, Maya `/api/catalog`, ComfyUI `/system_stats`, vLLM
  `/version`, SGLang `/get_model_info`, OpenAI-compatible `/v1/models` +
  image-route existence). Model discovery queries the public HuggingFace API
  (`huggingface.co/api/models`, newest-first per modality; disabled with
  `fresh=false`) and caches answers for one hour. Model downloads stream
  from the URL the engine or registry names (HuggingFace / ModelScope
  resolve links). If pull credentials are configured, they are sent only to
  `huggingface.co` / `modelscope.cn` hosts.
- **Shell:** runs `nvidia-smi` and `rocm-smi` (read-only GPU queries) and
  checks `PATH` for the `strata` CLI. No engine installer or launcher is
  run by the plugin.
- **Files:** writes downloaded model files (with `.part` resume files),
  download job records, and discovery caches under its own plugin data dir;
  `download_roots` (**Settings → Plugins**) can widen where downloads land
  (symlinks refused). Reads: your Hermes config, read-only, for
  `download_roots`; and a local image file only when you pass one to
  `model_forge_generate_media` — behind the Hermes read-block check, image
  magic bytes, and a 25 MB cap.
- **Approvals:** two actions escalate to Hermes' human-approval gate and
  fail closed on denial — registering a compute target, and writing the
  default model config (`model_forge_activate` with `apply=true`).
- **Credentials:** none required. HF/ModelScope tokens are optional, stored
  through Hermes' credential route (`Settings → Plugins`), read only via the
  secret scope.
- **Telemetry:** none.

## Requirements

- Hermes Agent (see `requires_hermes` in `plugin.yaml`)
- Any supported engine for the modality you want; Model Forge tells you what
  your hardware supports before you download anything.

## Install

```bash
hermes plugins install <this repo>
# or copy the folder into ~/.hermes/plugins/model-forge
```

## Development

The repo holds two installable packages: `model-forge/` (general plugin,
tools) and `model-forge-providers/` (`kind: model-provider`).

```bash
python3 tests/run.py                # stdlib runner, no deps
pytest -q                           # same suite under pytest (root config: pytest.ini)
cd model-forge          # the tools plugin
hermes plugins doctor . --ci        # load-test in isolation
hermes plugins validate .           # the catalog CI check
cd ../model-forge-providers
hermes plugins doctor . --ci && hermes plugins validate .
```

MIT license. Built by [anai](https://anaisoftware.com).
