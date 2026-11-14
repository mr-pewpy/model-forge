# Model Forge

**Local models for every modality — find your hardware, pick a model, and
Hermes runs it. No account, no API key — models run on hardware you control.**

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

One honest caveat up front: **Model Forge does not install or launch engine
stacks.** Install
either downloads model files, drives a running engine's own pull API, prints
the exact command for you to run, or — behind the human-approval gate — runs
an engine vendor's official installer (`action=engine_install`). Engines
stay yours to launch. The one exception is weight residency: a claim or
release on a llama.cpp target starts and stops its `llama-server` process as
part of the swap (see "One GPU lane, many models" below).

## The demo (90 seconds, one gaming PC, zero cloud)

1. **Chat** — pick a chat model for your GPU; Hermes runs on it
2. **Image** — generate an image; Model Forge pulls the image model
3. **Edit** — edit that image with an image-edit model
4. **3D** — turn it into a printable 3D model
5. **Video** — generate a promo video from the 3D render (i2v)
6. **Voice** — give it a voice (TTS / voice clone) and a soundtrack (music, SFX)

Each step visibly downloads the model it needs. Media jobs (3D, music, SFX)
run through a user-configured OpenAI-compatible media gateway.

## Tools

| Tool | What it does |
|------|--------------|
| `model_forge_status` | GPUs, VRAM, RAM, engines per compute target, pull credentials |
| `model_forge_browse` | Models by modality with an advisory fit verdict per target, plus a live HuggingFace "fresh" section (`fresh=false` = curated only) |
| `model_forge_target_add` | Register a compute target (a remote host, a cloud instance) — asks for your approval first |
| `model_forge_install` | Pull a model: plugin-side HF download (sha256, resume, job polling), a running Maya's HTTP pull, or the exact engine CLI command to run yourself; provision an engine (`action=engine_install`) from its vendor's official installer after your approval |
| `model_forge_activate` | Connect a running chat model as Hermes' default (plan-only; `apply=true` writes config after human approval) |
| `model_forge_generate_media` | 3D (image→GLB), music, SFX — jobs through a user-configured OpenAI-compatible media gateway |
| `model_forge_storage` | Model weights on disk (downloads dir, Ollama models dir, HuggingFace hub cache): list names, sizes and modification times; delete a listed entry after your approval |
| `model_forge_broker` | Weight residency on a compute target: what weights are on disk and what is loaded now, what that books against the budget (`status`), and the claim/release verbs that run a multi-step swap (chat → image generation → back to chat) as one workflow scope; `plan` previews the steps first, `bench` measures a model's real VRAM peak and books it as its nominal ceiling |

Plus `tts/model_forge_tts.py` — a config-driven TTS/voice-clone command
provider for any OpenAI-style `/v1/audio/speech` endpoint (see `tts/`), and
a **cockpit desktop pane** (`model-forge-desktop/`) showing every compute
target with live engine probes and model counts.

## One GPU lane, many models (weight residency)

One card cannot hold a big chat model and an image model at once. Model
Forge makes the swap a first-class verb instead of manual load/unload
bookkeeping — verbs, not knobs:

- **`model_forge_broker` `claim`** takes the GPU lane for a modality or a
  model: the eviction policy runs (recently used models stay warm longest;
  image generation takes the whole card), what must go steps down, the
  requested model is pre-warmed, and you get a scope id with honest
  progress states — `queued`, `loading`, `ready`. Never a silent stall.
- **`release`** ends the scope and restores exactly what was resident when
  you claimed it.
- **`plan`** is a dry run of claim: the release/load steps with their
  reasons, nothing executed.
- **`bench`** measures what you run: a short VRAM measurement window on a
  loaded (or about-to-run) model records its observed peak and books it as
  the model's nominal ceiling — measured peaks replace published estimates
  and are never downgraded by them.
- **`status`** is a read-only residency summary: pulled weights with sizes,
  what is loaded now, booked ceilings, and budget usage.

**The orchestration model, honestly.** Hermes core owns `image_generate`
and `video_generate`, and Model Forge cannot wrap those tools. What Model
Forge provides is the verbs the agent calls *around* them: to run "chat
model resident → image generation → back to chat" as ONE workflow, the
agent claims the lane before the core tool call and releases it after —
the scope restores the chat model automatically. `model_forge_generate_media`
does the same automatically for its own jobs: it claims the lane before the
job is submitted and releases it when the job completes or fails (a refused
claim fails closed and submits nothing).

Fit is gated on booked ceilings — the catalog's published VRAM floors until
a measured peak replaces them — while live free VRAM only decides which
model loads first, never whether one fits. A model with no published VRAM
floor (the engine-managed chat tiers) stays unbooked: the planner refuses
to gate it until a measured peak lands, and `status` marks those rows
`unbooked — run bench after first load`. Engines that own their own
load/swap policy (llama-swap, ComfyUI, vLLM, media gateways) are recorded
honestly as claimed for the scope; the engine moves the weights itself on
first use, and the step reasons say so.

## Model discovery

`model_forge_browse` has two layers:

- **Curated catalog** (`model_forge/registry.py`) — hand-maintained entries
  with approximate size, minimum VRAM, and an advisory **fit verdict** per
  target, first matching rule wins: `unavailable` (stub), `engine_unavailable`
  (no engine on the target can serve it), `needs_ram`, `engine_managed`
  (Strata/Maya size themselves at launch), `unknown`, `fits_gpu`, `needs_gpu`,
  `fits_memory`, or `too_big`. Verdicts are estimates from published
  footprints — the serving
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
never duplicates that catalog; it complements it.

## Bundled provider plugins

This repo also ships three provider packages:

- **`model-forge-providers/`** (`kind: model-provider`): registers `strata`
  and `maya` as first-class Hermes model providers (keyless, local — the LM
  Studio profile pattern), so `hermes model` / `--provider strata` can run
  Hermes directly on those engines.
- **`model-forge-image/`** (`kind: backend`): routes `image_generate` to ANY
  OpenAI-compatible images API (an OpenAI-compatible media gateway,
  vLLM-Omni, SGLang Diffusion)
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
- **File pulls resolve repo filenames automatically where possible.** When
  a filename cannot be resolved, pass `engine_model=` with the exact file
  inside the source repo (e.g. `diffusion_models/your-weights.safetensors`;
  omit it and Model Forge resolves the file from the repo).
- **music and sfx share one HuggingFace lane** (`text-to-audio`) — HF has
  no music-vs-sfx pipeline tags.
- **World model is an empty stub.** No local open engine with a stable API
  exists yet; the slot ships so the flow works the day one lands.
- **Engine-managed chat tiers have no published VRAM floor.** They stay
  unbooked and the planner refuses to gate them until a measured peak lands
  — run `model_forge_broker` `action=bench` after the model's first load
  (`status` marks those rows `unbooked — run bench after first load`).
- **`model_forge_generate_media` speaks the media gateway job API only**
  (`/v1/videos/generations`, `/v1/audio/music`, `/v1/audio/sfx`). The
  ComfyUI-native job flow (`/prompt` + `/history`) is not implemented.
- **Engine lifecycle management is limited to weight residency.** Model
  Forge never installs or launches full engine stacks — it detects engines,
  pulls model files, and emits run commands. Claim and release do drive
  per-model load/unload calls on your compute targets (an Ollama warm/evict
  call, a llama-server process start/stop); engines that own their own
  load/swap policy are recorded honestly as claimed, not driven. Installing
  engine *software* is available for Ollama, Strata and Project Maya via
  `model_forge_install` with `action=engine_install`: the plan shows the
  vendor's exact official install command, and it runs as a background job
  only after you approve it at the human-approval gate. llama.cpp is never
  installed — Hermes core's managed runtime covers it.

## Disclosure (what this plugin does on your machine)

- **Network:** probes compute-target engine endpoints (llama-server `/props`,
  Ollama `/api/tags`, Maya `/api/catalog`, ComfyUI `/system_stats`, vLLM
  `/version`, SGLang `/get_model_info`, OpenAI-compatible `/v1/models` +
  image-route existence). Model discovery queries the public HuggingFace API
  (`huggingface.co/api/models`, newest-first per modality; disabled with
  `fresh=false`) and caches answers for one hour. Model downloads stream
  from the URL the engine or registry names (HuggingFace / ModelScope
  resolve links). Media jobs (`model_forge_generate_media`) POST prompts and
  image bytes to user-configured compute targets, and Maya pulls POST to the
  target's `/api/pull`. Weight residency claims and releases
  (`model_forge_broker`, and the automatic scope around
  `model_forge_generate_media` jobs) drive engine load and unload calls on
  your user-configured compute-target endpoints — for example an Ollama
  warm/evict call carrying `keep_alive`, or llama-server process starts and
  stops. The TTS helper (`tts/model_forge_tts.py`) POSTs the
  text to speak plus base64-encoded voice-clone reference audio to a
  user-configured `/v1/audio/speech` endpoint. If pull credentials are
  configured, they are sent only to `huggingface.co` / `modelscope.cn` hosts.
- **Shell:** runs `nvidia-smi` and `rocm-smi` (read-only GPU queries,
  including the `bench` action's VRAM measurement window) and
  checks `PATH` for the `strata` CLI. Hardware probes also read system state
  directly: amdgpu sysfs on Linux and the Windows memory API
  (`GlobalMemoryStatusEx`). `model_forge_install` with `action=engine_install`
  executes vendor-official install commands in a background process, but only
  after explicit user approval at the human-approval gate: Ollama
  (`curl -fsSL https://ollama.com/install.sh | sh`), Strata
  (`curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh`),
  and Project Maya (`git clone https://github.com/mw00/project-maya && cd project-maya && ./setup.sh`).
  llama.cpp is never installed — it defers to Hermes core's managed llama.cpp
  runtime (the built-in Local Models catalog and downloads). Outside of
  weight residency claims and releases (which may start and stop a
  `llama-server` process on this machine), no engine launcher is run by the
  plugin.
- **Background:** long model downloads and approved engine installs run on
  a daemon background thread (job records are pollable with
  `model_forge_install` `job_id`); installer jobs carry a hard timeout (900
  seconds) that kills the installer's process group, so nothing hangs
  unattended. These are the plugin's only background processes.
- **Files:** writes downloaded model files (with `.part` resume files),
  download job records, and discovery caches under its own plugin data dir;
  `download_roots` (**Settings → Plugins**) can widen where downloads land
  (symlinks refused). Reads: your Hermes config, read-only, for
  `targets_json`, `download_roots`, and the plugin's settings; a local image
  file only when you pass one to `model_forge_generate_media` — behind the
  Hermes read-block check, image magic bytes, and a 25 MB cap; and the TTS
  helper reads the `--in` text file and `--reference` audio file from
  whatever command-line paths you name. `model_forge_storage` reads file
  names, sizes and modification times — read-only — in the plugin downloads
  directory and in your engine-managed model directories (the Ollama models
  dir and the HuggingFace hub cache under your home directory);
  `model_forge_storage` with `action=delete` removes the user-approved files
  and folders from those roots.
- **Approvals:** four actions escalate to Hermes' human-approval gate and
  fail closed on denial or timeout — registering a compute target, writing
  the default model config (`model_forge_activate` with `apply=true`),
  deleting a model-storage entry (`model_forge_storage` with `action=delete`;
  the prompt names the exact path and size), and running a vendor engine
  installer (`model_forge_install` with `action=engine_install`,
  `execute=true`; the prompt shows the exact command, and denial or timeout
  runs nothing).
- **Credentials:** none required. HF/ModelScope tokens are optional, stored
  through Hermes' credential route (`Settings → Plugins`), read only via the
  secret scope.
- **Telemetry:** none.

## Requirements

- Hermes Agent 0.23.0 or newer (see `requires_hermes` in `plugin.yaml`)
- Any supported engine for the modality you want; Model Forge tells you what
  your hardware supports before you download anything.

## Install

Primary route — install from the plugin catalog:

```bash
hermes plugins install model-forge
```

Manual route — copy the package folder you want (e.g. `model-forge/`) into
`~/.hermes/plugins/`. The repo root has no manifest, so per-package folder
copies are the working manual path.

## Development

The repo holds five installable packages: `model-forge/` (tools + hooks),
`model-forge-image/` and `model-forge-video/` (media backends),
`model-forge-desktop/` (desktop pane), and `model-forge-providers/`
(`kind: model-provider`).

```bash
python3 tests/run.py                # stdlib runner, no deps; without pytest it
                                    # skips fixture-based tests
pytest -q                           # same suite under pytest (root config: pytest.ini)
cd model-forge          # the tools plugin
hermes plugins doctor . --ci        # load-test in isolation
hermes plugins validate .           # the catalog CI check
cd ../model-forge-providers
hermes plugins doctor . --ci && hermes plugins validate .
```

MIT license. Built by [anai](https://anaisoftware.com).
