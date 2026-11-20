# Model Forge — Product Master Plan

Requirements decomposition and build plan for Model Forge as the local-model product layer for Hermes.
Audience: the project owner and future contributors. Every current-state claim cites the file it was
verified against. Phases A–D define the build order; risks are stated honestly at the end.

## 1. Vision

Model Forge makes local AI a first-class Hermes experience: it finds the user's hardware, maps each
modality to a supported inference engine and a model that fits, provisions or connects that engine, and
wires the result into Hermes as a native model — with the same product experience on every surface
where a user meets Hermes. The core loop — "I want X" → determine what fits → pull or connect → use it —
is already proven (README.md). What this plan adds is product completeness and uniformity: every major
engine supported and usable two ways, a first-class guided path to any GPU, one Desktop settings
experience, feature parity across every Hermes surface, and reliability as the through-line (honest
states, fail-closed policy, approval gates, disclosures).

## 2. Pillar 1 — GPU discovery and pointing

### Current state (verified)

- Local hardware auto-detection covers NVIDIA, AMD (two probe paths), Apple Silicon unified memory, and
  system RAM, with every probe failing soft to "no data" (model_forge/hardware.py; README.md).
- Compute targets are the abstraction for "where inference runs": id, label, base URL, advisory kind,
  and user-declared VRAM/RAM budgets; the local target is always present and never removable; Desktop
  settings (`targets_json`) merge with tool-registered rows, settings winning on id conflicts
  (model_forge/targets.py; model-forge/plugin.yaml).
- `model_forge_target_add` registers a target and escalates to the human-approval gate first (rule key
  `model-forge:target_add`), showing the exact address (model_forge/tools.py).
- Remote-target hardware numbers are user-declared because they cannot be probed honestly over an engine
  API; fit verdicts price against them (model_forge/targets.py).
- The Desktop cockpit pane lists every target with live engine probes and declared budgets
  (model-forge-desktop/desktop/plugin.js).

### Gaps

- No guided product flow for pointing at a GPU the plugin cannot see (a cloud GPU box, another machine
  in the user's home). Registration today is a tool call or hand-edited `targets_json` JSON
  (model-forge/plugin.yaml).
- No pre-approval test/preview of a candidate target, and no edit or remove surface for registered
  targets (model_forge/targets.py).
- Declared budgets carry no provenance; nothing helps the user learn them from the remote machine.

### Workstreams

- **W1.1 Guided "point at a GPU" flow.** A first-class, step-wise path (Desktop settings and
  chat-driven) that collects the target's base URL, probes it read-only, shows what answered, collects
  declared budgets, then submits through `model_forge_target_add`'s approval gate. *Acceptance:* a user
  who cannot see the GPU locally can register it end-to-end without editing JSON; the probe preview runs
  before any approval prompt; the approval shows the exact address and runs nothing on denial.
- **W1.2 Target lifecycle management.** Edit, re-test, and remove registered targets (the local target
  remains permanent), reusing the approval gate for any change that redirects engine calls or media
  jobs. *Acceptance:* a registered target can be corrected or removed from the product surface; removal
  never touches the local target; every write action that redirects traffic passes the approval gate.
- **W1.3 Budget provenance.** Label each target's memory budget as declared or detected, and offer a
  read-only discovery helper (via the target's engine APIs) that suggests values the user confirms.
  *Acceptance:* every budget shown to the user is labeled with its source; the helper never writes a
  budget without user confirmation.

## 3. Pillar 2 — Complete inference-engine support

### Current state (verified)

- Detection adapters cover llama.cpp/Ollama (distinct fingerprints), Strata, Project Maya, ComfyUI, vLLM
  (with a vLLM-Omni route probe for diffusion lanes), SGLang (with a Diffusion route probe), plus a
  generic OpenAI-compatible detector that lists models but never claims engine identity
  (model_forge/engines/__init__.py; engines/vllm.py; engines/sglang.py; engines/openai_compat.py).
- The weight-residency broker has six lifecycle adapters — Ollama (keep-alive warm and evict),
  llama-swap (observational pass-through), llama.cpp (process start/stop), ComfyUI (readiness plus
  explicit cache free), vLLM (process stop/start; engine sleep/wake deliberately unused), Maya
  (engine-native readiness; release is an honest ledger acknowledgment) (model_forge/broker.py;
  model_forge/tools.py `_LIFECYCLE_ENGINES`).
- Provisioning is "plugin-provisioned": approved vendor-official installers for Ollama, Strata, and
  Project Maya, planned first and executed only behind the human-approval gate; llama.cpp is never
  installed — it defers to Hermes core's managed runtime (model_forge/tools.py `_ENGINE_INSTALLERS`;
  README.md).
- "User points at an HTTP API" exists as the generic detector (model_forge/engines/openai_compat.py)
  and probe ports for well-known defaults (model-forge-desktop/desktop/plugin.js).
- Strata and Maya are additionally registered as first-class Hermes providers
  (model-forge-providers/__init__.py).

### Gaps

- **Exo is now supported** — the shipped `model_forge/engines/exo.py` detector and `ExoLifecycle`
  (instance/deployment lifecycle) supersede the earlier "not supported anywhere in the tree" claim.
- Pointed endpoints are not first-class settings: there is no user-editable engines list in the config
  schema, so "connect my existing engine" is implicit probing rather than a product surface
  (model-forge/plugin.yaml).
- Approved installers cover three engines; vLLM, SGLang, ComfyUI, llama-swap, and Exo are connect-only
  or command-print (model_forge/tools.py).
- ComfyUI-native generation is plugin-driven — image/video run through the `/prompt` + `/history`
  workflow runner (local or connected ComfyUI); 3d/music/sfx ride a media gateway (README.md "Known
  limits"; model_forge/registry.py `COMFYUI_LIMIT_NOTE`).
- Python audio engines are honest "phantom" entries: discoverable but never selectable
  (model_forge/registry.py `PHANTOM_ENGINES`).

### Workstreams

- **W2.1 Exo support.** Detection (distinct fingerprint or honest generic fallback), registry engine
  mapping, and an observational or driveable lifecycle adapter matching Exo's actual load semantics.
  *Acceptance:* Exo appears in engine detection at a pointed endpoint, models map to it in
  browse/install, and any lifecycle claim is honest about what is and is not driven.
- **W2.2 First-class engines list.** A user-editable engines setting (per entry: label, base URL,
  expected engine, modalities) that feeds detection, fit verdicts, and model selection — the product
  form of "point at its HTTP API." *Acceptance:* a user-added endpoint is probed, listed, and usable for
  model selection without any tool call; entries persist in settings and appear in `model_forge_status`.
- **W2.3 Two ways per engine.** For every engine in the supported set (llama.cpp, Ollama, vLLM incl.
  Omni lanes, SGLang incl. Diffusion lanes, Maya, Strata, ComfyUI, llama-swap, Exo): either an approved
  installer exists or the product documents and offers the connect path, with no dead-end rows.
  *Acceptance:* each engine's product row states exactly how it gets onto a machine and how it gets
  connected; no engine resolves to "install something by hand" with no guidance.
- **W2.4 ComfyUI-native generation — DONE.** Implemented the `/prompt` + `/history` job flow
  (`ComfyWorkflowRunner` + `model_forge_generate_media`) so ComfyUI endpoints serve generation, not just
  downloads. *Acceptance (met):* an image job runs on a connected ComfyUI endpoint through the same
  claim/release scope as media jobs, with honest progress states — local **or** connected/remote, with
  fetchable `/view` output URLs for a remote ComfyUI.

## 4. Pillar 3 — Desktop product UX

### Current state (verified)

- Settings today: `targets_json`, `download_roots`, and two optional masked secret fields — Hugging Face
  and ModelScope tokens — stored through Hermes' credential route and sent only to their own host
  families (model-forge/plugin.yaml; README.md).
- The cockpit pane is a read-only view: targets plus live engine probes and model counts
  (model-forge-desktop/desktop/plugin.js).
- Storage inventory and approved deletion exist backend-side across the plugin downloads directory, the
  Ollama models directory, and the Hugging Face hub cache (model_forge/storage.py;
  model_forge/schemas.py `STORAGE`).

### Gaps

- No UI for the guided GPU-pointing flow (Pillar 1), engine management (Pillar 2), or model choice with
  fit verdicts; no storage management surface; no surfaced choice between pasting a pull token and
  referencing Hermes' existing secrets.

### Workstreams

- **W3.1 Settings shell.** Optional, coherent settings sections: GPU location (detect + guided
  pointing), engines (list, detection, provisioning), models (browse with per-target fit verdicts and
  one-action install), credentials (paste a token or reference an existing Hermes secret), storage
  (inventory and approved delete). *Acceptance:* every capability of the tool surface is reachable from
  Desktop settings without editing raw JSON; secret fields never display stored values.
- **W3.2 Fit-verdict model chooser.** Render `model_forge_browse` rows (advisory verdicts: `fits_gpu`,
  `fits_memory`, `needs_gpu`, `needs_ram`, `too_big`, `engine_managed`, `engine_unavailable`, `unknown`,
  `unavailable`) per target, with install actions and job progress (model_forge/registry.py
  `fit_verdict`). *Acceptance:* a user can pick a model for a chosen target and start its pull from the
  pane; verdicts are labeled advisory, matching the tool output.
- **W3.3 Storage surface.** List entries (name, path, size, modification time) and delete with the
  approval prompt naming exact path and size. *Acceptance:* deletion from Desktop passes the same
  approval gate and entry-confinement rules as the tool (model_forge/storage.py); a denied approval
  changes nothing.

## 5. Pillar 4 — Native integration and surface parity

### Current state (verified)

- "Wired like a native provider" today means: `model_forge_activate` persists
  `model.default`/`model.provider`/`base_url` through the documented
  `hermes_cli.model_setup.persist_model_selection` helper (plan-only by default; `apply=true` behind the
  approval gate) (model_forge/tools.py), and provider registration for Strata/Maya with deliberately
  empty `base_url` so local-server detection keeps working (model-forge-providers/__init__.py).
- Provider slug mapping is `strata`→`strata`, `maya`→`maya`, `llamacpp`→`llamacpp`, others→`custom`
  (model_forge/tools.py `_PROVIDER_FOR_ENGINE`) — consistent with the documented rule to map engines to
  `custom`/`custom:<target-id>` and never invent slugs (model-selection-integration.md).
- Per model-selection-integration.md: `/model` is core (plugins cannot override it; plugins may register
  their own commands and Telegram handlers); plugin-managed models become selectable by declaring a
  `custom_providers` entry (a typed model id routes to it automatically) or via live `/v1/models`
  probing of the current custom endpoint; `persist_model_selection` writes only the `model:` section,
  takes effect next session, and does not create `custom_providers` entries.

### Gaps

- Activation writes no `custom_providers` entry, so plugin-served models do not appear in the `/model`
  picker until the user configures one by hand — the documented practical flow (plugin
  provisions/activates the endpoint and writes a `custom_providers` entry → user taps `/model`) is not
  implemented end to end (model-selection-integration.md; model_forge/tools.py).
- Feature parity is unproven: there are no plugin surfaces for Telegram (model selection plus plugin
  commands), CLI/TUI beyond core's `hermes model`, or the dashboard; parity exists only where core
  renders the tools.
- Mobile: nothing beyond gateway-side work is required for the chat client, but no audit has verified
  each surface; on-device mobile inference is a named future lane, not a current commitment.

### Workstreams

- **W4.1 Custom-provider registration.** On activation, write the matching `custom_providers` entry
  (slug `custom` or `custom:<target-id>`, base URL, declared models) so `/model` selection works with no
  manual configuration; never invent provider slugs. *Acceptance:* after an approved activation, the
  model appears in the `/model` picker on the next session and switching to it routes to the endpoint;
  the write is shown in the approval prompt.
- **W4.2 Surface parity matrix.** Define and verify the capability matrix — model selection, model
  browsing with fit verdicts, install with job progress, broker status/claim/release, storage, settings
  — across Desktop, Telegram, CLI/TUI, and dashboard; implement the missing plugin-side pieces (Telegram
  handlers and plugin commands registered through the documented plugin API only). *Acceptance:* each
  surface either offers a capability or states, in-product, why not; no capability silently differs in
  behavior between two surfaces.
- **W4.3 Mobile readiness.** Verify every Pillar 4 capability is gateway-side (inherited by a future
  mobile client); document on-device mobile inference as a future lane with a named entry point when a
  mobile runtime exists. *Acceptance:* an audit shows no capability requires a specific desktop client;
  the future lane is recorded in the public docs without over-promising.

## 6. Pillar 5 — Product-grade reliability (the through-line)

### Current state (verified)

- Four user decisions escalate to the human-approval gate and fail closed on denial or timeout: target
  registration, default-model config writes, storage deletion, and vendor installer execution
  (model_forge/tools.py `policy_hook`, rule keys `model-forge:target_add` / `activate` / `storage_delete`
  / `engine_install`).
- Honest state vocabulary in the broker (`queued`, `loading`, `ready`; never a silent stall) and honest
  "unbooked" rows where no measured peak exists (README.md; model_forge/broker.py).
- Fail-soft observation everywhere: engine probes degrade to a stated reason, the live discovery layer
  fails soft offline, and installer jobs carry a hard timeout that kills the process group
  (model_forge/engines/__init__.py; model_forge/discovery.py; README.md).
- Storage deletion accepts only inventory-returned entries and re-verifies containment
  (model_forge/storage.py). A full disclosure section states network, shell, background, file, approval,
  credential, and telemetry behavior (README.md).

### Gaps and workstreams

- **W5.1 State vocabulary everywhere.** Every new surface (Desktop flows, Telegram, dashboard) renders
  the same honest states and reasons the tools return; no surface substitutes a spinner for a stated
  reason. *Acceptance:* a forced failure (unreachable target, denied approval, expired download) shows a
  stated reason on every surface.
- **W5.2 Approval coverage audit.** Any new write that redirects traffic, changes default config,
  destroys data, or executes vendor code goes through the approval gate before it ships. *Acceptance:* a
  review checklist maps each write action to its rule key or an explicit, documented reason it needs
  none.
- **W5.3 Disclosure upkeep.** Update README disclosure and catalog metadata for every feature in Phases
  A–D (new endpoints probed, new background jobs, new secret fields). *Acceptance:* the disclosure
  matches shipped behavior at each phase gate.

## 7. Phased build order

- **Phase A — Engine and GPU reach.** W2.1 (Exo), W2.2 (engines list), W2.3 (two ways per engine),
  W1.1–W1.2 (guided pointing + target lifecycle), W2.4 (ComfyUI generation). *Depends on:* nothing new —
  the compute-targets system and approved-installers framework already exist (model_forge/targets.py;
  model_forge/tools.py).
- **Phase B — Desktop UX.** W3.1–W3.3, plus W1.3. *Depends on:* Phase A backends (engines list, target
  flow) so the settings shell drives real capabilities.
- **Phase C — Surface parity.** W4.2, W5.1. *Depends on:* Phase A (stable tool semantics) and Phase B
  (the settings/data shapes the surfaces must mirror).
- **Phase D — Native-provider polish + mobile readiness.** W4.1, W4.3, W5.2–W5.3, and follow-through on
  the core engine-bundle proposal so provisioning rides a platform API instead of per-vendor installers
  (docs/engine-bundle-proposal.md). *Depends on:* Phase C — seams must be proven on at least two
  surfaces before the provider layer is polished, and the bundle interface should be shaped by that
  consumer experience.

## 8. Risks (honest)

- **Engine lifecycle quirks.** Engine sleep/wake is unreliable for some engines, so process restart is
  part of the adapter contract; Maya has no unload primitive; llama-swap accepts no eviction control;
  ComfyUI needs explicit cache frees between generative jobs (model_forge/broker.py;
  docs/broker-roadmap.md). New engines (Exo) may add another shape; adapters must state, not hide, what
  they drive.
- **Upstream API drift.** Engine fingerprints (`/api/catalog`, `/version`, `/get_model_info`, route
  probes), the Hugging Face discovery API, and vendor install commands all change outside our control
  (model_forge/engines/*; model_forge/tools.py `_ENGINE_INSTALLERS`; model_forge/discovery.py).
  Detection must degrade to honest "unknown" rows, never mislabel.
- **Surface-specific integration seams.** `/model` is core and cannot be overridden; picker behavior
  depends on per-adapter `send_model_picker` support; `custom_providers` matching keys on
  slash-normalized base URL plus api_mode and headers, and the picker only probes the current custom
  endpoint (model-selection-integration.md). Parity work must respect these seams rather than build
  plugin-side selectors for core's job.
- **Provisioning constraints.** Vendor installers assume unrestricted shell access; on restricted
  machines the provisioning path ends (docs/engine-bundle-proposal.md). The connect-a-running-engine
  path is the durable fallback and must never be a second-class experience.
- **Advisory fit verdicts.** Verdicts are estimates from published footprints; the serving engine does
  authoritative sizing (model_forge/registry.py). Product surfaces must keep the advisory label to avoid
  over-promising.
- **Single-GPU contention is physics.** One accelerator cannot hold a large chat model and an image
  model at once; the orchestrator's promise is predictable eviction and honest timing, not coexistence
  (docs/broker-roadmap.md).

## 9. Phase D status addendum (2026-11)

Short status notes only — the plan above is unchanged.

- **W4.1 Custom-provider registration — done.** An approved activation (`model_forge_activate`
  `apply=true`) writes the matching `custom_providers` entry (slug `custom` / `custom:<target-id>`,
  declaring the served model) so `/model` lists it with no manual configuration; the approval prompt
  names the entry before anything is written (model_forge/tools.py `write_custom_provider_entry`;
  shape parity-tested in model-forge-providers).
- **W4.3 Mobile readiness — done.** The gateway-side audit and the future-lane note are recorded in
  docs/surface-parity.md ("Mobile readiness (W4.3)"): every capability is gateway-side, no capability
  requires a specific desktop client, and on-device mobile inference is recorded as a named future
  lane without over-promising.
- **W5.2 Approval coverage audit — checklist delivered.** The write-action → rule-key /
  documented-reason map is now in docs/surface-parity.md ("Approval coverage (W5.2)"), verified
  against the tool handlers and `policy_hook`.
- **W5.3 Disclosure upkeep — current at this gate.** Both README disclosure sections and the five
  package descriptions were brought up to the shipped surface (dashboard API route, `/model-forge`
  command, user-declared endpoint probing, config reads and writes); it remains a standing obligation
  at every later phase gate.
- **Engine-bundle follow-through** (provisioning through a platform API instead of per-vendor
  installers) remains open (docs/engine-bundle-proposal.md).
