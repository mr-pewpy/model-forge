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
| llama.cpp / llama-server (incl. Hermes' managed runtime) | chat, vision | detection done |
| Ollama | chat, vision, ocr | detection done |
| Strata | chat, vision (125B MoE on consumer GPUs) | detection done |
| Project Maya | chat, vision (321B–748B MoE, 1–16 GPUs) | adapter next |
| vLLM / vLLM-Omni | vision, ocr, image, image_edit | adapter next |
| ComfyUI | image, video, 3d | adapter next |
| Python engines (ACE-Step, Chatterbox, AudioGen, MusicGen) | music, sfx, voice_clone | adapter next |

GGUF chat models are served by Hermes' core Local Models catalog — Model Forge
never duplicates that pricing; it complements it.

## Disclosure (what this plugin does on your machine)

- **Network:** probes `127.0.0.1` engine endpoints (llama-server `/props`,
  Ollama `/api/tags`, Strata `/v1/models`) — local only, no external calls
  from the plugin itself. Model downloads are performed by the engines' own
  tooling when you ask for them.
- **Shell:** runs `nvidia-smi` (read-only GPU query) and checks `PATH` for
  engine CLIs (`strata`, later `maya`, `comfyui`). Future install/serve tools
  will run engine-provided installers/launchers and will say so here first.
- **Files:** reads no files outside its own plugin data dir today.
- **Credentials:** none required. Secrets (if ever needed for remote engines)
  go through Hermes `requires_env` / secret scope only.
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

```bash
hermes plugins doctor . --ci        # load-test in isolation
hermes plugins validate . --install-deps   # the catalog CI check
python -m pytest tests/ -q          # registry + fit tests
```

MIT license. Built by [anai](https://anaisoftware.com).
