# Model Forge v2 — the local-model resource layer (roadmap)

Where the plugin goes after the hackathon entry. v1 finds and pulls models;
v2 owns **local model resources** for Hermes: what weights live where, what
is loaded in VRAM right now, and what happens when a task needs a different
model. Everyday-user first throughout — productivity users, not power users.

## 1. Vision

Model Forge becomes the local-model resource layer for Hermes, on five
pillars:

1. **Discovery** — what models exist for a modality, and what fits this
   machine. (Largely done in v1.)
2. **Suggestion** — which model to use given what is available and what is
   pointed at: prefer what is already pulled, already warm, and inside the
   booked ceilings of the compute target in play.
3. **Model orchestration** — workflow scopes around a task: "with chat
   model → swap to image gen → restore chat" as one gesture, not three
   manual steps.
4. **Weight residency** — loading and unloading weights: what is resident,
   when it gets evicted, whose lane wakes. Not networking — the resource
   policy layer.
5. **Pulling + storage management** — downloads (done in v1), plus knowing
   where every byte lives and deleting with user approval.

**Everyday-user first.** The user says "make me an image"; Model Forge handles
residency. Verbs, not knobs — no `unload_policy=lru-600` anywhere in the UX.

**Official Hermes capacity.** Where Hermes core already does the job — the
managed llama.cpp runtime with its hardware-priced GGUF catalog — v2
integrates instead of duplicating, and engine provisioning rides official
installers and, where possible, an upstream Hermes engine-bundle feature.

## 2. Current state vs vision

Built (verified against the tree):

- **Discovery** — curated catalog (`registry.py`, `ModelEntry`: id, name,
  modality, engines, source, `size_gb`, `vram_gb_min`, `ram_gb_min`,
  `engine_managed_fit`, status, `engine_models`, `released`) plus a live
  HuggingFace "fresh" layer (`discovery.py`, 1-hour cache, fail-soft
  offline, `fresh=false` opt-out).
- **Suggestion (advisory)** — `fit_verdict` per compute target from
  registry footprints and declared budgets (`fits_gpu`, `fits_memory`,
  `engine_managed`, …). Estimates only; the serving engine does the
  authoritative sizing.
- **Pulling** — plugin-side HF/ModelScope streaming downloads with sha256
  verification, `.part` resume, destination confinement (`download_roots`,
  symlinks refused) and `job_id` progress polling; a running Maya's HTTP
  pull API (`/api/pull` + `/api/pull/status`); exact engine CLI commands
  for Strata and everything else (`strata launch` argv, `hf download`).

Missing (the v2 work):

- **Storage management** — no tool lists what is on disk with sizes and
  last-used, and nothing deletes with approval.
- **Weight residency** — nothing tracks residency or evicts; status only
  mirrors whatever an engine reports as loaded.
- **Model orchestration** — no workflow scopes, no pre-warm, no restore.
- **Engine install** — "Model Forge does not start engines" is the current
  honest rule; all engine adapters are detect-only.
- **HF filename auto-resolve** — file pulls resolve repo filenames
  automatically where possible; fall back to an exact `engine_model=` path
  inside the repo when resolution cannot determine the file. Smarter
  resolution (prefer full weights, quant variants only on request) continues
  as v2 work.

## 3. Serving policy proven under production load

These rules are distilled from production multi-GPU model serving. They
generalize to any machine; v2 applies them directly against raw engine APIs:

- **Fit against NOMINAL booked ceilings.** Live free VRAM only **orders**
  candidates (which to load first); it never **gates** them (whether they
  fit). Booked-vs-live comparisons cause false evictions — live free VRAM
  always reads lower than nominal and evicts models that actually fit.
- **A lane wake stops resident side models first.** Any "lane wake" (a big
  model claiming its lane) first stops resident side models; otherwise the
  side model OOMs the big lane.
- **Recency window (~600 s).** Models used within roughly the last ten
  minutes stay warm; recency outranks pure LRU near the top of the stack.
- **Image-gen entry evicts GPU residents every time.** Image generation
  wants the whole card; deterministic full eviction beats clever sharing.
- **Fire model loads sequentially.** Concurrent engine inits OOM — one
  load at a time.
- **Free GPU caches between back-to-back generative jobs.** Engines hold
  allocations after a job; an explicit cache free between consecutive
  generative jobs keeps the next one honest.
- **Big models need process stop/start, not engine sleep/wake.** Some
  engines' wake endpoints are broken upstream; process restart is the
  reliable path for large loads.
- **LLM-waiter priority.** While a chat request waits on a lane wake, refuse
  new generative jobs; the waiter owns the wake. Otherwise a gen job and a
  chat request fight over the swap.
- **Pull manifests must read the LFS sha256 nested in the lfs object**
  (`lfs.sha256` / `lfs.oid`) — not the pointer file's own hash — and
  retry expired signed URLs per round (HF resolve links expire mid-pull).

## 4. Hermes constraints

From the plugin catalog rules (already applied in the SCOPE risk register
and the README disclosure):

- **No bundled engine binaries (source-only).** Engine provisioning is
  plugin-orchestrated **official installers behind the human-approval
  gate**, and/or an upstream Hermes "engine bundle" framework feature.
  Precedent: Hermes core already ships a managed llama.cpp runtime with a
  hardware-priced catalog and downloads — the framework can own engines.
- **Background processes are allowed but must be disclosed in README.**
  The download job runner already is; serving-side engine processes extend
  the disclosure.
- **Fail cleanly unattended.** No hanging prompts. An unattended run
  degrades to a stated reason, never a stall.
- **Policy decisions route through the human-approval gate** — storage
  deletes, network targets, config writes — via the `pre_tool_call`
  approve contract this plugin already uses for `model_forge_target_add`
  and `model_forge_activate apply=true`.
- **Generalize for any machine through compute targets.** The policy layer
  books against declared budgets per target (local/lan/cloud), never
  host-local assumptions.

## 5. Architecture — four components

### (a) Residency ledger

What weights are pulled where — the plugin downloads dir plus
engine-managed dirs (Ollama's models dir, ComfyUI model subdirs, …) — and
what is loaded on which target. Booked ceilings per model come from
registry footprints (`size_gb`, `vram_gb_min`) plus a **measured-footprint
tool**: load a model, sample VRAM during a bench, record the peak.
Published footprints are advisory; measured peaks are the booking numbers.

### (b) Engine lifecycle adapters

Each adapter exposes `ensure_loaded` / `release` per engine:

| Engine | Lifecycle shape |
|--------|-----------------|
| Ollama | `keep_alive` preload to warm, `keep_alive=0` to evict |
| llama-swap | route control — it swaps on demand natively |
| llama.cpp / llama-server | process start/stop; Hermes core runtime integration where present |
| ComfyUI | workflow queue + explicit cache free between jobs |
| vLLM | process restart; sleep/wake is unreliable upstream |
| Maya / Strata | engine-native session/pull APIs |

### (c) Orchestrator

- **Workflow scopes with claim/release semantics** around a task: "with
  chat model → swap to image gen → restore chat" claims the GPU lane for
  the scope and restores on release.
- **Pre-warm on intent** — start loading before the user's click finishes.
- **Eviction = LRU + recency window (~600 s) + lane priority** — a full
  chat lane preempts side models.
- **Honest progress states** — `queued` / `loading (~Ns)` / `ready` —
  never a silent stall.

### (d) Storage tool — `model_forge_storage`

List models with sizes + last-used across the plugin dir and the
engine-managed dirs it can see; delete via the human-approval gate
(destructive by nature — approval is not optional). Fails closed on
denial.

## 6. UX contract

- **Verbs, not knobs.** "Make an image", "free up space", "use this
  model" — never eviction policies, cache modes, or VRAM math.
- **The user promise is orchestrated pre-warm + honest timing feedback.**
  Never zero-latency: weights physically move into VRAM and no policy
  erases that. Never a silent stall: a user who waits 40 s behind
  `loading (~35s) → ready` trusts the product; a user who waits 40 s at a
  frozen spinner files a bug.

## 7. Phased delivery

### P1 — user-path closure

- HF filename auto-resolve via the repo file-list API: `model_forge_install`
  resolves the model file without `engine_model=` (prefer full weights;
  quant variants only on request).
- `model_forge_storage`: list (sizes + last-used) and approved delete.
- Approved engine installs for Ollama + llama.cpp first (official
  installers, human-approval gate).

**Acceptance:** a curated model installs with no filename argument; a
storage delete requires approval and fails closed on denial; an unattended
install never hangs.

### P2 — residency core

- Residency ledger + Ollama / llama-swap / llama.cpp adapters +
  booked-ceiling fit + eviction (LRU + recency window + lane priority).

**Acceptance:** with a chat model resident, an image-gen load evicts per
policy and chat is restored after; fit gates on nominal booked ceilings
(live free VRAM only orders); loads fire sequentially.

### P3 — orchestration

- Workflow scopes with claim/release, pre-warm on intent, wiring into the
  chat and media flows (`image_generate`, `video_generate`,
  `model_forge_generate_media`).

**Acceptance:** "with chat model → image gen → restore chat" runs as one
scope with honest progress states through the swap.

### P4 — remaining adapters + framework

- ComfyUI / vLLM / Maya adapters to `ensure_loaded`/`release`; the
  measured-footprint tool; the upstream Hermes "engine bundle" proposal
  (formalizing the core llama.cpp precedent).

**Acceptance:** each adapter survives its engine's lifecycle quirks
(process restart where wake is unreliable, cache free between generative
jobs); measured peaks replace published footprints in bookings.

### P5 — broker consolidation (replace the .62 weights broker)

The weights broker on the .62 inference host (`192.168.1.62:8080`, Gitea
`charles/weights-broker`) predates this roadmap and still owns what v2 must
become: the TP2 vLLM ↔ ComfyUI container dance, the ComfyUI gen workflow
engine, and the stable OpenAI id/alias surface the fleet points at. P5 ports
those three jobs into Model Forge and retires the broker. The broker stays
LIVE and serving until the final phase; every replacement is verified before
its consumers flip, and the alias map must keep working at every step.

Current broker surface to preserve (verified 2026-11-19): 8 served ids at
`:8080` — the resident chat lane `Qwen/Qwen3.8-27B-FP8` (vLLM TP2 across
both 3090s, 262k ctx, served through the `current` symlink) plus 7 ComfyUI
gen lanes (`gen/qwen-image-2.1`, `-edit`, `-cutout`, `gen/moss-image`,
`gen/h3-t2v`, `gen/h3-i2v`, `gen/h3-r2v`) — with live consumers: Hermes
custom_providers + auxiliary.vision, the image-gen plugin, the house-scan
vision cron, and OpenViking on .63. De-registration order for the broker
itself follows the packer skill: survivor analysis, alias backfill for every
orphaned id, swap/verify before deleting, config deploy then weight deletes
last.

Three workstreams, each an acceptance gate:

- **(a) Docker runtime lifecycle.** Generalize `broker.py`'s
  `ProcessHandle` seam into a `RuntimeHandle` (process | container) and add
  a `DockerContainerLifecycle` adapter: `ensure_loaded` = `docker start` +
  health probe with a cold-boot budget; `release` = `docker stop` (instant
  VRAM free). Registry entries gain a `runtime: {kind, name, health_url,
  boot_budget_s, gpu_count}` section. Budgets: vLLM 90–170 s warm / ~8 min
  cold; ComfyUI minutes (ai-dock cold boot). The §3 LLM-waiter rule is
  implemented in the orchestrator. **Acceptance:** orchestrator
  claim(chat) → vllm-server up; claim(image) → lane evicted → comfyui up →
  job → idle release → lane restored.
- **(b) ComfyUI workflow runner.** Port the broker's workflow engine into
  `engines/comfyui.py`: template registry (reuse the broker's
  `workflows/*.json` verbatim), `param_map` injection (the broker's
  `_inject`: explicit map first, class-based seed/steps/prompt fallback;
  `size` → width/height), `/prompt` submit with 405 cold-boot retry,
  `/history` polling, output collection by `output_kind` (image/video/3d),
  data-URL → `/upload/image` normalization (nested `{"upload": {"path"}}`
  shape), explicit cache free between back-to-back jobs. Wire into
  `model_forge_generate_media` and the Hermes image/video providers.
  **Acceptance:** all 7 gen lanes run end-to-end through Model Forge with
  the same output shapes as today (images via `POST /v1/images/generations`
  semantics, video via the job API); old gen id aliases resolve.
- **(c) Facade + consumer migration.** A thin OpenAI facade in Model Forge:
  chat passthrough + `/v1/models` catalog + aliases (port the broker's
  route layer only — the orchestrator owns policy). Extend
  `model_forge_activate` to write Hermes `custom_providers` +
  `auxiliary.vision`. Flip consumers one at a time (Hermes providers →
  image plugin → vision cron → OpenViking .63), each verified before the
  next. **Acceptance:** all consumers green on Model Forge; the broker
  container stopped and its repo archived; freed resources reported.

Operational guardrails (from the broker skills): never `git reset --hard`
in `~/work/weights-broker` (a stash holds an old vLLM-Omni experiment);
infra ops through Gitea, product code through GitHub; commit with explicit
paths only; test every replacement on throwaway ports/containers first
(safe-adoption pattern: vLLM on :8001) and never disturb a live TP2 job.

## 8. Honest difficulty profile

- **The policy layer is proven.** These rules already run in a production
  multi-GPU serving deployment; the main work is expressing them against
  raw engine APIs — that work *is* the project, not the policy design.
- **Per-engine lifecycle quirks demand adapter-level honesty.** Sleep/wake
  is not a given — some engine wake endpoints are unreliable upstream — so
  process restart belongs in the adapter contract from day one, not as an
  escape hatch.
- **Single-GPU contention is physics.** One card cannot hold a big chat
  model and an image model at once. The orchestrator's job is honest
  timing and predictable eviction, not magic coexistence.
