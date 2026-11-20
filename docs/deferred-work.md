# Model Forge — known limits and deferred work

Every known limit and deferred item in one place, so users and reviewers do
not have to reconstruct them from the README, the surface parity matrix, and
the product master plan. Each entry states four things: what the item is, the
honest engineering reason it is limited or deferred, what exists today
instead (the alternative users actually get), and what would unblock it.

The list is grouped as:

- **(A) Known limits** — shipped behavior that is deliberately less than a
  naive user might expect: a design decision, a platform seam, or an upstream
  constraint. These entries document what the product does not do; the
  alternative named for each is part of the product today.
- **(B) Deferred work** — capabilities the project intends to build, without
  a schedule. Nothing here is under active development or promised for a
  release date.

---

## (A) Known limits

### A1. Install resolves curated catalog entries only

- **What it is:** `model_forge_install` installs models from the curated
  catalog (`model_id`s in the registry). Live discovery rows are not install
  targets.
- **Why:** live rows come from the public Hugging Face listing and carry no
  fit estimate, no engine mapping, and no verified file set — installing one
  would mean guessing those.
- **What exists today:** live rows are labeled *fresh/unvetted* and serve
  discovery (date, downloads, pipeline tag, link); the curated catalog is the
  install surface, and any curated entry can be added by the project.
- **What would unblock it:** per-row vetting of live entries (fit data, file
  resolution) before they become installable.

### A2. File pulls fall back to an explicit file name

- **What it is:** when a repository file name cannot be resolved
  automatically, a file pull needs `engine_model=` naming the exact file
  inside the source repository.
- **Why:** repositories do not always publish a file list the resolver can
  read; a pull must never guess which file it is downloading.
- **What exists today:** resolution runs automatically where possible (via
  the repository's public file list), and the fallback error names the exact
  `engine_model=` value to pass. The resolver now prefers FULL (non-quantized)
  weights explicitly — a quantized variant is picked only when the repo offers
  no full weights, or when the caller names one via `engine_model=` ("quantized
  variants only on request") — and labels the pick honestly (`quantized`
  true/false).
- **What would unblock it:** broader file-list resolution for repositories
  that publish no readable file list (the fallback stays for those).

### A3. Music and SFX share one discovery lane

- **What it is:** music and sound-effect models appear in one
  `text-to-audio` lane instead of two modality lanes.
- **Why:** the upstream model taxonomy has no music-versus-SFX pipeline tag,
  so splitting the lane would mean mislabeling one side.
- **What exists today:** one honest shared lane; generation distinguishes
  music from SFX at the job API (`/v1/audio/music` versus `/v1/audio/sfx`).
- **What would unblock it:** an upstream taxonomy that separates the two
  modalities.

### A4. World model is an empty stub

- **What it is:** the world-model modality ships as a stub with no models.
- **Why:** no local open engine with a stable API exists for the modality
  yet; inventing rows would over-promise.
- **What exists today:** the slot is present so the flow works the day a
  real engine lands; browsing it states the stub honestly.
- **What would unblock it:** a local open world-model engine with a stable
  API.

### A5. Telegram output is plain-text rows

- **What it is:** the `/model-forge` command renders plain-text rows in
  chat, capped under the chat message length limit with an explicit
  truncation row.
- **Why:** interactive cards beyond the core approval and clarification
  flows are not used by this plugin; silent truncation would hide data.
- **What exists today:** complete data through the JSON subcommands
  (`cockpit`, `models`, `storage`) and the tools; the plain-text rows state
  when they are truncated.
- **What would unblock it:** a first-class rich-output path for plugin
  commands in chat clients.

### A6. Model switching is core (`/model`)

- **What it is:** choosing the active model uses the core `/model` picker and
  `hermes model`; Model Forge does not ship a model selector of its own.
- **Why:** plugins cannot override core's `/model`; building a competing
  selector would split the experience.
- **What exists today:** an approved activation writes the model config plus
  a `custom_providers` entry, so the activated model appears in the core
  picker with no manual configuration.
- **What would unblock it:** nothing to build — this is the platform seam by
  design.

### A7. Engine lifecycle management is limited to weight residency

- **What it is:** Model Forge does not install or launch full engine stacks:
  it detects engines, pulls model files, emits exact run commands, and drives
  per-model load/unload where an engine offers it. Approved vendor installers
  exist for Ollama, Strata, and Project Maya only; llama.cpp is never
  installed (core's managed runtime covers it); other engines are
  connect-only or command-print.
- **Why:** the plugin catalog is source-only (no bundled binaries), and
  engines stay the user's to launch — the disclosure states this up front.
- **What exists today:** `model_forge_install` with `action=engine_install`
  runs a vendor's PINNED install command (installer script SHA-256 checked;
  Maya checked out at a fixed commit) as a background job after the
  human-approval gate, and the bring-your-own-endpoint path connects any
  running engine without installing anything.
- **What would unblock it:** core-provisioned engine bundles (see B9) or
  additional vendor-official installers behind the same gate.

### A8. In-pane Desktop storage delete is not built

- **What it is:** the cockpit pane and the `/model-forge` command are
  read-only; they list storage (names, sizes, modification times) but do not
  delete.
- **Why:** a pane data read cannot ride the human-approval gate, and
  deletion of model weights is destructive — the gate is not optional.
- **What exists today:** the panes name the delete verb
  (`model_forge_storage` `action=delete`), where the approval prompt names
  the exact path and size before anything is removed and only
  inventory-returned entries are deletable.
- **What would unblock it:** a platform path for a pane action to ride the
  human-approval gate.

### A9. Live Desktop and dashboard rendering is untested in CI

- **What it is:** continuous integration verifies the data shapes, the
  command output rows, and the dashboard manifest and bundle; it does not
  render the Desktop pane or the dashboard tab.
- **Why:** rendering runs inside the Desktop app and the web dashboard, which
  the test harness cannot drive; a fabricated render test would prove
  nothing.
- **What exists today:** both renderers are verified by hand at each release
  gate, and both degrade to a stated reason when a data source fails — a
  failure state is never replaced with silent defaults.
- **What would unblock it:** a headless render harness for desktop panes and
  dashboard tabs.

### A10. Dashboard backend routes mount when the dashboard server starts

- **What it is:** the dashboard tab's backend route (the cockpit snapshot
  under `/api/plugins/model-forge/`) is imported and mounted by the
  dashboard server's startup sweep. Routes added by a later update of the
  package are served only after the server restarts (or an explicit plugin
  rescan).
- **Why:** the server imports a package's API module once, at startup, and
  deliberately does not re-import code for a package whose routes are already
  mounted.
- **What exists today:** the route is stable across updates, and the same
  cockpit data is available at any time through the `/model-forge` command
  and the Desktop pane — no data is ever exclusive to the dashboard tab.
- **What would unblock it:** platform-side hot reload of a plugin's dashboard
  API module on update.

### A11. The capability probe is skipped for the manifest-only desktop package

- **What it is:** the desktop package (`model-forge-desktop`) ships a
  manifest and a JavaScript pane with no Python package, and the plugin
  tooling's capability probe is skipped for it (`capability probe — skipped
  (no __init__.py)`); the runtime load check likewise expects a Python
  registration module.
- **Why:** a tooling limitation — the probe inspects a Python package's
  registrations, and a manifest-only plugin declares none.
- **What exists today:** the package is validated at manifest, desktop
  surface, and static-check level (`hermes plugins validate` passes), and its
  behavior is covered indirectly through the shared data shapes the other
  packages' tests assert.
- **What would unblock it:** plugin tooling that probes manifest-only
  (desktop) packages.

### A12. Engine-managed chat tiers stay unbooked until measured

- **What it is:** a model that publishes no VRAM floor (the engine-managed
  chat tiers, which size themselves at launch) has no booked ceiling; the
  planner refuses to gate fit on it.
- **Why:** a guessed ceiling would be a guessed fit, and the policy layer
  never guesses fits.
- **What exists today:** broker `status` marks those rows `unbooked — run
  bench after first load`, and `model_forge_broker` `action=bench` books the
  measured peak as the ceiling afterwards (measured peaks replace published
  estimates and are never downgraded by them).
- **What would unblock it:** a measured peak from `bench` after the model's
  first load — this is a user step, not missing code.

### A13. vLLM sleep/wake is deliberately unused

- **What it is:** the vLLM lifecycle adapter starts and stops the server
  process; the engine's sleep/wake endpoints are never called.
- **Why:** wake endpoints are unreliable upstream, so process restart is the
  reliable path for large loads — this is serving practice, not a shortcut.
- **What exists today:** claim/release drive a documented process
  stop/start, and the step reasons say so.
- **What would unblock it:** upstream sleep/wake that is reliable enough to
  trust for large loads.

### A14. Project Maya release is an honest no-op

- **What it is:** `release` on a Maya engine records the end of the workflow
  scope but drives no unload.
- **Why:** the engine exposes no unload primitive; the only removal route is
  destructive and is never used as a release, and an unload is never faked.
- **What exists today:** the release row is labeled `no-unload-primitive`
  with the reason stated; the engine keeps its own session policy.
- **What would unblock it:** an upstream, non-destructive unload or session
  end primitive.

### A15. Engines without model listing serve curated fallback models

- **What it is:** engine provider rows that declare no model listing serve
  their curated `fallback_models` in pickers instead of a live `/v1/models`
  listing.
- **Why:** the engine serves no listing route; a guessed list would be
  dishonest.
- **What exists today:** the fallback list is the curated catalog mirror
  (unknown fields omitted, never guessed), and the same rows state which
  models the engine serves when a listing is available.
- **What would unblock it:** an upstream model-listing route on those
  engines.

### A16. The Exo readiness wait is not consumed

- **What it is:** placing a model on an Exo cluster is asynchronous; the
  lifecycle adapter requests placement and reports the state
  `placement-requested` rather than blocking until the model is ready.
- **Why:** placement downloads and loads engine-natively; a verb that
  blocked on it would either hang or need to fake readiness. The state
  vocabulary promises `queued → loading → ready` only when those states are
  actually observed.
- **What exists today:** the state and its reason name the readiness route
  (`GET /instance/await`) so readiness can be watched, and a release stops
  the deployment through the engine's real unload primitive
  (`DELETE /instance/{id}`) even while placement is still in flight.
- **What would unblock it:** consuming the readiness wait in the lifecycle
  adapter so `ensure_loaded` can report `ready` when the engine says so.

### A17. Exo builds that predate the instance API are not detected

- **What it is:** an Exo endpoint that predates the current meta and instance
  API is reported as a generic OpenAI-compatible endpoint, not as Exo.
- **Why:** the detector fingerprints only Exo-specific surfaces (node
  identity, cluster state); generic routes such as `/healthcheck` are served
  by plenty of software and are deliberately never taken as proof of an
  engine identity.
- **What exists today:** the endpoint still works as a generic
  OpenAI-compatible engine (detection, fit, activation) and the row states
  its generic identity with a reason.
- **What would unblock it:** an Exo-specific surface those older builds
  serve, to use as an honest fingerprint.

### A18. Broker verbs that stop or evict an engine server are not gated

- **What it is:** starting an engine server process (llama-server, vLLM) from
  a weight-residency claim, swap, release or media-lane call now rides the
  human-approval gate (rule_key `model-forge:broker_server_start`), but
  STOPPING one is still ungated: a claim that evicts a resident model, or a
  release that ends a scope, stops the process without asking.
- **Why:** a stop is non-destructive — the weights stay on disk and the model
  can be re-loaded (a re-load that starts a server does pass the gate), on
  compute targets the user registered (each registration passed the gate) or
  configured in settings.
- **What exists today:** the disclosure states the process start/stop
  behavior; the verbs are non-destructive and fail closed on a refused claim.
  Stopping an EXTERNALLY-started server (a runtime the user started) is now
  gated (rule_key `model-forge:broker_server_stop`) — see B8. A plugin-owned
  server is still stopped without asking, by design: it was gated when Model
  Forge started it, so stopping it is the symmetric counterpart (weights stay
  on disk and a re-load that starts a server passes the gate again).
- **What would unblock it:** nothing further for the external case (B8 is
  implemented); a gate on plugin-owned stops would be redundant with the
  start gate.

### A19. Automated quality gates are partly short of their own targets

- **What it is:** the project's code-quality audit reports test coverage at
  93% (core) and 92% (provider packages) against a 100% target, and mutation
  testing as not measured.
- **Why:** the uncovered branches are mostly error-degradation paths that
  need a live engine to reach, and the mutation-testing tool did not run in
  the audit environment; the audit records both gaps rather than hiding them
  (`docs/quality-gates.md`).
- **What exists today:** 420 automated tests cover the data shapes, tool
  handlers, engine adapters and policy layer; the live smoke run at the last
  release gate exercised the remaining workflows by hand.
- **What would unblock it:** live-engine fixtures for the degradation
  branches and a working mutation-testing run.

### A20. The approval gate and the release handler re-evaluate the restore separately

- **What it is:** before a release that would start an engine server, the
  human-approval gate inspects the release's restore plan (models the scope
  would re-load) and asks for approval; the release handler then computes the
  same plan again when it runs. The two share their construction code
  (`_restore_requests` / `restore_plan`), but they are separate evaluations
  against live ledger state — so a concurrent claim that evicts a restored
  model inside the window between the gate's inspection and the handler can
  make the handler's recomputed plan start a server the gate did not see.
- **Why:** the gate is a pre-call hook over live state; the ledger can change
  between the hook and the handler, and closing that gap means re-verifying
  (or consuming the gated decision) inside the handler, which is not built.
- **What exists today:** the window is narrow — it requires an interleaved
  concurrent claim/eviction in the gap — and it can only restore models that
  were resident at claim time (the restore set is fixed then, never
  model-chosen). Every server-starting claim, swap, release and media-lane
  release still rides the gate (`rule_key
  model-forge:broker_server_start`), and any server a restore starts binds
  127.0.0.1 only. Only a state change inside the window escapes the gate.
- **What would unblock it:** handler-side re-verification of the restore
  against the gate's approval, or the handler consuming the exact plan the
  gate approved (a follow-up, not done today).

---

## (B) Deferred work (planned, not scheduled)

### B1. ComfyUI-native job generation

**Implemented.** Generation runs directly on a ComfyUI endpoint through the
`/prompt` + `/history` workflow flow — including a **connected** ComfyUI.

- **What it is:** running generation jobs directly on a ComfyUI endpoint
  (the `/prompt` + `/history` workflow flow), so a connected ComfyUI engine
  serves generation and not only downloads.
- **Why:** the media job surface was built against the OpenAI-compatible
  media gateway job API first; the ComfyUI workflow queue is a second job
  protocol with its own progress and cache semantics.
- **What exists today:** `ComfyWorkflowRunner` drives the workflow-queue job
  flow (submit `POST /prompt`, poll `GET /history/{id}`, collect outputs by
  kind, free the cache between jobs) through the bundled workflow templates,
  and `model_forge_generate_media` runs image/video through it inside the
  same claim/release scope as the gateway lanes — on a LOCAL **or** a
  CONNECTED/remote ComfyUI endpoint (detected at the target's `base_url`).
  Outputs are `data:` URLs when readable locally, else the fetchable ComfyUI
  `/view` URLs (a connected ComfyUI's `/output` is not readable here). Honest
  progress states throughout. 3d/music/sfx ride the media gateway job API.
- **Remaining:** a remote ComfyUI's output files are surfaced by `/view` URL
  rather than fetched and embedded locally; the media gateway keeps the
  3d/music/sfx lanes.

### B2. Broker residency rows on the cockpit surfaces

**Implemented.** Weight-residency rows render on both cockpit surfaces.

- **What it is:** rendering weight-residency rows (pulled, resident, booked,
  unbooked, budget usage) in the Desktop cockpit pane and the dashboard tab.
- **Why:** the cockpit surfaces were built around targets, engines, models
  and storage first; residency rendering on them was not built initially.
- **What exists today:** `cockpit_data()["residency"]` and the dashboard
  `/residency` API route carry the pulled / resident / booked / unbooked /
  budget rows, and both surfaces render a "Broker weight residency" section —
  the Desktop pane (`ResidencySection` in `plugin.js`) and the dashboard tab
  (`ResidencySection` in `dashboard/dist/index.js`) — with the same honest
  state vocabulary (booked provenance labels, unbooked → run bench, budget
  headroom). Verified by `tests/smoke_residency_renderers.js` (a Node smoke —
  the renderers are hand-verified per A9, so this is the automated check).
- **Remaining:** none for B2. (The dashboard `dist/index.js` is the served
  build artifact; if a frontend source exists elsewhere, port the
  `ResidencySection` there too.)

### B3. Bench sampling of remote compute targets

- **What it is:** the `bench` VRAM measurement window samples the machine
  Model Forge runs on (the same probe scope as the hardware report, NVIDIA
  memory via the vendor's standard query); sampling a model loaded on a
  remote compute target is deferred.
- **Why:** a remote engine API exposes no reliable VRAM sampler, and a
  booked ceiling must be an observation, never a guess — so bench fails soft
  with a stated reason instead of estimating ("run bench where the model is
  loaded").
- **What exists today:** run `bench` on the machine where the model is
  loaded; the recorded peak is booked as the model's nominal ceiling and
  gates fit from then on.
- **What would unblock it:** a per-target sampling path (and non-NVIDIA
  samplers) that observes the real footprint remotely.

### B4. AMD marketing-name sampling

- **What it is:** AMD GPUs are reported with generic labels (memory totals
  are sampled; the marketing name is not).
- **Why:** the AMD probe paths read memory totals only — there is no
  comparable name query to the NVIDIA probe's, and a guessed name would be a
  mislabel.
- **What exists today:** honest generic labels with correct VRAM budgets,
  which is what fit verdicts price against; names are cosmetic to fit.
- **What would unblock it:** an honest name source on AMD platforms
  (for example, a PCI device-name lookup) added to the hardware probe.

### B5. Engine-native reshaping of model aliases

- **What it is:** the short model aliases on the engine provider rows (for
  example `flash-next`) resolve to the curated catalog's registry-form model
  ids; reshaping them to an engine's native served names is deferred.
- **Why:** the catalog ids are the shared vocabulary across the registry,
  the providers mirror and the tools; engine-native names vary per engine
  and mapping them is a per-engine translation layer.
- **What exists today:** aliases work end to end against the registry ids,
  and served-id information is carried where the registry states it.
- **What would unblock it:** per-engine alias reshaping on top of the
  registry's served-id data.

### B6. Local adapters for the Python audio engines

- **What it is:** first-class adapters for the Python audio engines
  (ACE-Step, Chatterbox, AudioGen, MusicGen) for music, SFX and voice clone.
- **Why:** these engines live in custom Python environments the plugin does
  not manage, so they are listed as honest phantom entries — discoverable,
  never selectable — instead of pretending they run.
- **What exists today:** music, SFX and voice clone run through the
  user-configured media gateway lane and the TTS provider; the phantom rows
  state exactly what is missing.
- **What would unblock it:** adapters that drive those engines through a
  managed environment (an engine bundle would be the natural host — see B9).

### B7. Target lifecycle management

- **What it is:** editing, re-testing and removing registered compute
  targets, and a read-only probe preview of a candidate target before the
  approval prompt.
- **Why:** registration shipped first (with the approval gate) and the
  lifecycle surface after it was never built; the local target stays
  permanent by design.
- **What exists today:** registered targets come from the tool verb or the
  settings list (`targets_json`), which remains hand-editable; every target
  is probed read-only wherever engines are detected, so a wrong target
  degrades to a stated reason rather than a failure.
- **What would unblock it:** the target-lifecycle workstream (W1.2) and the
  pre-approval probe preview (W1.1) in the product master plan.

### B8. An approval gate for residency swaps on externally-started servers

**Implemented.** Model Forge now tracks honest server-process provenance and
gates the stop of an externally-started server.

- **What it is:** a gate on `claim`/`swap`/`release` when the swap would stop
  a server process the user started themselves (an externally-started
  runtime — e.g. a docker container brought up by compose).
- **Why:** the stop verb was ungated because a stop is non-destructive; the
  missing signal was provenance — whether the plugin or the user started the
  server. Stopping a server the user runs is the one case that should ask.
- **What exists today:** the container lifecycle records provenance — a
  container it starts is ``plugin`` (gated at start, symmetric to stop), one
  it adopts already-running is ``external``, and an unreadable prior state is
  fail-safe ``external``. ``stops_externally_started()`` on each engine
  lifecycle is the signal (process adapters answer ``False``: they only stop
  a handle they started). The policy gate escalates a stop of an external
  runtime to the human-approval gate (rule_key
  ``model-forge:broker_server_stop``); a plugin-owned stop stays ungated.
- **Scope:** the container adapter is the one runtime that can stop a
  user-started server (``docker stop``). Process adapters never stop a
  process they did not start, so they never need this gate.

### B9. Engine bundles (core-provisioned engine runtimes)

- **What it is:** manifest-declared, versioned engine bundles provisioned by
  Hermes core through its managed runtime, with a documented lifecycle API
  for plugins — so provisioning stops being one vendor-installer path per
  engine. The full proposal is
  [engine-bundle-proposal.md](engine-bundle-proposal.md) (problem, precedent,
  manifest sketch, API shape, non-goals), with a ready-to-post submission
  package in
  [engine-bundle-proposal-submission.md](engine-bundle-proposal-submission.md).
- **Why it is deferred:** it is a framework feature, not a plugin feature.
  The proposal is submitted to the framework maintainers as
  [issue #156547](https://github.com/NousResearch/hermes-agent/issues/156547)
  (2026-11-16) and is awaiting their response; the follow-through remains
  open in the product master plan (Phase D).
- **What exists today:** approved vendor-official installers behind the
  human-approval gate (A7) and the connect-a-running-engine path, which is a
  first-class experience and the durable fallback.
- **What would unblock it:** the framework maintainers adopting engine
  bundles — Model Forge's six-engine lifecycle adapter set is the
  reference-consumer contract for the API shape.

### B10. Remote-target swap weight records

- **What it is:** the swap verb's pre-flight verifies the incoming model's
  weights are already present before touching residency; for remote
  compute targets that check relies on recorded pulls, and installs do not
  yet write pull records for remote targets.
- **Why it is deferred:** the local path is solved (installed weights are
  read from the storage scan), and guessing remote state would violate the
  fail-closed rule.
- **What exists today:** remote swaps honestly refuse with a stated
  "weights not present — run the install verb first" reason instead of
  guessing, and nothing is evicted.
- **What would unblock it:** writing pull records for remote-target
  installs (planned as part of the remote-target workstream).

### B11. Weights-free engine pre-start

- **What it is:** the swap pre-flight warms the engine by confirming
  liveness; a stopped engine passes pre-flight and starts during the
  measured load phase.
- **Why it is deferred:** no supported engine exposes a way to start its
  server without naming the weights to serve, so a fully warm start before
  eviction is not possible without faking it.
- **What exists today:** the guarantee that nothing is evicted until the
  load is certain to begin; stopped-but-startable engines start within the
  measured load phase and the timings show it.
- **What would unblock it:** engine-side support for a weights-free warm
  process (or the engine bundles proposed in B9 exposing one).

---

Nothing in this file is hidden from users: every limit stated here is also
stated where the feature it concerns is documented.
