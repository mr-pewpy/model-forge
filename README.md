# Model Forge

**Local models for every modality — find your GPU, pick a model, and Hermes
runs it. No account, no API key, nothing leaves your machine.**

Model Forge is a Hermes Agent plugin that makes local AI a first-class
experience across every modality: chat, vision, OCR, image generation, image
editing, video, 3D meshes, music, sound effects, and voice. It finds your
hardware, shows you which models fit, drives the right inference engine
(llama.cpp, Strata, Ollama, vLLM, ComfyUI, and more), and wires each model
into Hermes as a provider — so every native tool (chat, `image_generate`,
`video_generate`, `vision`, TTS) can use it.

## Why

Hermes already runs GGUF chat models locally (the built-in **Local Models**
pane handles those beautifully). But the moment you want an image, a video, a
3D model, or a voice, you are back to hand-installing engines and guessing at
VRAM numbers. Model Forge closes that gap: one registry of
**modality → engine → model → hardware fit**, so the flow is always the same:

> "I want X" → Model Forge picks what fits → pulls the model → starts the
> engine → connects it to Hermes → you use it.

## The demo (90 seconds, one gaming PC, zero cloud)

1. **Chat** — pick a chat model for your GPU; Hermes runs on it
2. **Image** — generate an image; Model Forge pulls the image model and engine
3. **Edit** — edit that image with an image-edit model
4. **3D** — turn it into a printable 3D model
5. **Video** — generate a promo video from the 3D render (i2v)
6. **Voice** — give it a voice (TTS / voice clone) and a soundtrack (music, SFX)

Each step visibly downloads and self-hosts the model it needs.

## Tools

| Tool | What it does |
|------|--------------|
| `model_forge_status` | GPUs, VRAM, RAM, and which local engines are running |
| `model_forge_browse` | Models by modality with an advisory fit verdict for your machine |

More tools (`model_forge_install`, `model_forge_activate`) and the cockpit
desktop pane land in the next milestones — see [SCOPE.md](SCOPE.md).

## Engines

| Engine | Modalities | Status |
|--------|-----------|--------|
| llama.cpp / llama-server (incl. Hermes' managed runtime) | chat, vision | adapter done |
| Ollama | chat, vision, ocr | adapter done |
| Strata | chat, vision (125B MoE on consumer GPUs) | adapter done (detect + `strata launch`) |
| Project Maya | chat, vision, ocr (321B–748B MoE, 1–16 GPUs) | adapter done (detect + Models-page API) |
| vLLM / vLLM-Omni | chat, vision, ocr, image, image_edit | adapter done (detect + omni route probe) |
| SGLang / SGLang Diffusion | chat, vision, image, image_edit | adapter done (detect + diffusion route probe) |
| ComfyUI | image, image_edit, video, 3d | adapter done (detect) |
| Any OpenAI-compatible endpoint | chat, vision, ocr, image | generic detector (never mislabels the engine) |
| Python engines (ACE-Step, Chatterbox, AudioGen, MusicGen) | music, sfx, voice_clone | adapter next |

GGUF chat models are served by Hermes' core Local Models catalog — Model Forge
never duplicates that pricing; it complements it.

## Bundled provider plugin

This repo also ships **`model-forge-providers/`** (`kind: model-provider`):
registers `strata` and `maya` as first-class Hermes model providers (keyless,
local — the LM Studio profile pattern), so `hermes model` / `--provider strata`
can run Hermes directly on those engines.

## Pull credentials (Hugging Face / ModelScope)

**Settings → Plugins → Model Forge** has two optional masked fields:

- **Hugging Face token** (`HF_TOKEN`) — unlocks gated repos, full-speed pulls
- **ModelScope token** (`MODELSCOPE_API_TOKEN`) — authenticated mirror pulls

Both are optional; without them everything public still works. Tokens are
stored through Hermes' credential route (never in config files), read only
via the secret scope, and sent **only** to their own host family
(`huggingface.co` / `modelscope.cn` — enforced in code, unit-tested).
`model_forge_status` reports which tokens are configured (booleans only).

## Disclosure (what this plugin does on your machine)

- **Network:** probes compute-target engine endpoints (llama-server `/props`,
  Ollama `/api/tags`, Maya `/api/catalog`, ComfyUI `/system_stats`, vLLM
  `/version`, SGLang `/get_model_info`, OpenAI-compatible `/v1/models` +
  image-route existence) — engine APIs only, no external calls from the
  plugin itself. Model downloads are performed by the engines' own tooling
  when you ask for them. If pull credentials are configured, they are sent
  only to `huggingface.co` / `modelscope.cn` hosts.
- **Shell:** runs `nvidia-smi` (read-only GPU query) and checks `PATH` for
  engine CLIs (`strata`, later `maya`, `comfyui`). Future install/serve tools
  will run engine-provided installers/launchers and will say so here first.
- **Files:** reads no files outside its own plugin data dir today.
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
cd model-forge          # the tools plugin
hermes plugins doctor . --ci        # load-test in isolation
hermes plugins validate .           # the catalog CI check
python3 tests/run.py                # registry + adapter tests (stdlib runner)
cd ../model-forge-providers
hermes plugins doctor . --ci && hermes plugins validate .
```

MIT license. Built by [anai](https://anaisoftware.com).
