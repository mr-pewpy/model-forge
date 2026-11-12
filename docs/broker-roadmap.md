# Model Forge v2 — the local-model resource layer (roadmap)

Where the plugin goes after the hackathon entry. v1 finds and pulls models;
v2 owns **local model resources** for Hermes: what weights live where, what
is loaded in VRAM right now, and what happens when a task needs a different
model. Normie-first throughout — productivity users, not power users.

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
4. **Weights broker** — loading and unloading weights: what is resident,
   when it gets evicted, whose lane wakes. Not networking — the resource
   policy layer.
5. **Pulling + storage management** — downloads (done in v1), plus knowing
   where every byte lives and deleting with user approval.

**Normie-first.** The user says "make me an image"; Model Forge handles
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
- **Weights broker** — nothing tracks residency or evicts; status only
  mirrors whatever an engine reports as loaded.
- **Model orchestration** — no workflow scopes, no pre-warm, no restore.
- **Engine install** — "Model Forge does not start engines" is the current
  honest rule; all engine adapters are detect-only.
- **HF filename auto-resolve** — file pulls demand an exact `engine_model=`
  path inside the repo; the registry does not guess filenames.

## 3. Reference implementation: the LAN weights broker

The developer already runs a LAN weights broker — a 2× RTX 3090 gateway
with a packer — and it is the proof the policy layer works. v2 is that
broker generalized off Docker onto raw engine APIs. Its battle rules port
directly:

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
  The download job runner already is; broker-side engine processes extend
  the disclosure.
- **Fail cleanly unattended.** No hanging prompts. An unattended run
  degrades to a stated reason, never a stall.
- **Policy decisions route through the human-approval gate** — storage
  deletes, network targets, config writes — via the `pre_tool_call`
  approve contract this plugin already uses for `model_forge_target_add`
  and `model_forge_activate apply=true`.
- **Generalize for any machine through compute targets.** The broker books
  against declared budgets per target (local/LAN/cloud), never host-local
  assumptions.

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
- **The normie promise is orchestrated pre-warm + honest timing feedback.**
  Never zero-latency: weights physically move into VRAM and no policy
  erases that. Never a silent stall: a user who waits 40 s behind
  `loading (~35s) → ready` trusts the product; a user who waits 40 s at a
  frozen spinner files a bug.

## 7. Phased delivery

### P1 — normie closure

- HF filename auto-resolve via the repo file-list API: `model_forge_install`
  resolves the model file without `engine_model=` (prefer full weights;
  quant variants only on request).
- `model_forge_storage`: list (sizes + last-used) and approved delete.
- Approved engine installs for Ollama + llama.cpp first (official
  installers, human-approval gate).

**Acceptance:** a curated model installs with no filename argument; a
storage delete requires approval and fails closed on denial; an unattended
install never hangs.

### P2 — broker core

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

## 8. Honest difficulty profile

- **The policy layer is proven.** The LAN broker already runs these rules;
  the main work is the port from Docker onto raw engine APIs — that port
  *is* the project, not the policy design.
- **Per-engine lifecycle quirks demand adapter-level honesty.** Sleep/wake
  is not a given — some engine wake endpoints are unreliable upstream — so
  process restart belongs in the adapter contract from day one, not as an
  escape hatch.
- **Single-GPU contention is physics.** One card cannot hold a big chat
  model and an image model at once. The orchestrator's job is honest
  timing and predictable eviction, not magic coexistence.
