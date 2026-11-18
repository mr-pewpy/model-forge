# Model Forge — surface parity matrix

One Model Forge experience on every Hermes surface. This document is the
capability matrix for Phase C of the product master plan (workstream W4.2):
exactly what works where, rendered from the same honest state vocabulary
everywhere (W5.1), with the gaps named in-product and here.

Every capability below is either offered on a surface or states, in-product,
why not — no capability silently differs in behavior between two surfaces.

## The surfaces

| Surface | What it is |
|---|---|
| **Desktop** | The Desktop app: the Model Forge cockpit pane (`model-forge-desktop/desktop/plugin.js`, four sections: GPU & compute targets, Engines, Models, Installed model storage) plus chat with the agent and the `/model-forge` command. |
| **Telegram** | Gateway chat: the `/model-forge` command (appears in the bot menu) with plain-text rows, plus every Model Forge tool through the agent. |
| **CLI/TUI** | `hermes chat` and `hermes --tui`: the same `/model-forge` command and tools. Model switching is core's `/model` (and `hermes model`) — plugins cannot override it. |
| **Dashboard** | The web dashboard (`hermes dashboard`) Model Forge tab: `model-forge/dashboard/` (manifest.json + `dist/index.js` + `plugin_api.py`), a read-only snapshot of the same cockpit data. |
| **Mobile (future)** | Not a current client. See "Mobile readiness" below. |

## Capability matrix

| Capability | Desktop | Telegram | CLI/TUI | Dashboard |
|---|---|---|---|---|
| **Cockpit data** (targets + GPU hardware, engines, model fit rows, storage) | Pane sections + `/model-forge cockpit\|status` | `/model-forge cockpit\|status` (JSON / plain-text rows) | `/model-forge cockpit\|status` | Model Forge tab (same data via `plugin_api.py`) |
| **Model browsing with fit verdicts** | Pane "Models" section (curated `model_forge_browse` fit rows) + `model_forge_browse` tool | `/model-forge browse [modality]` rows, `/model-forge models` JSON, `model_forge_browse` tool | same as Telegram | Tab "Models" section (snapshot rows) |
| **Model selection / activation** | Tool verb `model_forge_activate` (plan-only default; `apply=true` rides the approval gate) in chat; `/model` picker is core | `model_forge_activate` tool via the agent | `model_forge_activate` tool; `hermes model` (core) | Not on the tab — activation is a tool verb with an approval gate; the tab is read-only by design |
| **Install with job progress** | `model_forge_install` tool in chat (job rows with honest progress states) | `model_forge_install` tool via the agent | `model_forge_install` tool | Not on the tab (read-only); install runs as an approved tool verb in chat |
| **Broker status** (pulled / resident / booked / unbooked / budget) | `model_forge_broker action=status` in chat | `/model-forge broker` rows + `model_forge_broker` tool | same as Telegram | Not on the tab yet (honest gap; run `/model-forge broker` in chat) |
| **Claim / release (GPU lane)** | `model_forge_broker action=claim\|release\|swap` tool in chat (`swap` is the one-call exchange with measured phase timings) | `/model-forge claim\|release\|swap` render guidance; the verbs are tool calls | same as Telegram | Not on the tab (read-only) |
| **Storage inventory** | Pane "Installed model storage" section (read-only) | `/model-forge storage` JSON + `model_forge_storage` tool | same as Telegram | Tab "Installed model storage" section (read-only) |
| **Storage delete** | Tool verb `model_forge_storage action=delete` only — rides the human-approval gate | same | same | Not offered (read-only); the tab names the delete verb honestly |
| **Storage resources** (on-demand resource view: memory usage + resident weights) | Pane "Installed model storage" → collapsed **Resource view** (loads one snapshot on expand; manual refresh; no polling or animation) + `/model-forge storage resources` text bars in chat | `/model-forge storage resources` text bars (JSON rows: `/model-forge resources`) | same as Telegram | Tab "Installed model storage" → collapsed **Resource view** (loads on expand; manual refresh; no polling or animation) |
| **Settings** (targets, engine endpoints, download roots, pull tokens) | Settings → Plugins → Model Forge (`config_schema` form; secret fields are masked and never display stored values) | Guided through the agent (`hermes config` / plugin settings) | `hermes config` / plugin settings | The tab is read-only; settings live in the Desktop/config surfaces |
| **Guided GPU pointing** (target registration) | Pointing invitation in the pane; `model_forge_target_add` (approval-gated) in chat | Invitation in `/model-forge cockpit\|status` rows; `model_forge_target_add` tool | same as Telegram | Invitation rendered on the tab; registration is the approval-gated tool verb |

## State vocabulary (identical on every surface, W5.1)

- **Fit verdicts** (advisory — kept labeled so nothing over-promises):
  `fits_gpu`, `fits_memory`, `needs_gpu`, `needs_ram`, `too_big`,
  `engine_managed`, `engine_unavailable`, `unknown`, `unavailable`.
- **Engine rows**: `available=true|false` plus a stated `detail` reason and
  `source=auto-detected|user-declared` — never a bare spinner.
- **Broker residency**: `queued → loading → ready`, with an explicit reason
  on failure; `unbooked` rows where no measured or published peak exists.
- **Storage**: `size_bytes` (on-disk size) and `last_modified` (filesystem
  time) in the `model_forge_storage` list shape.
- **Resource view (on-demand)**: `used_gb` (live observed usage from the
  hardware probe), `booked_now_gb` (the live claims — the booked ceilings of
  the models resident right now), `free_gb` (total minus used), each with a
  percentage of the total, plus honest `unavailable` rows with a stated
  reason when a probe fails. "Booked ceilings" (nominal catalog ceilings)
  and `booked_now` (live claims) are different quantities and never share a
  word. Design intent: resource visibility for weights and memory belongs
  with weight management — the view is opened when needed and is never a
  persistent display (no polling, no animation, no auto-refresh).

A forced failure (unreachable target, denied approval, expired download)
shows a stated reason on every surface; a command-side read that overruns its
budget returns an honest timeout row ("nothing was changed — retry later")
instead of hanging.

## Honest gaps

- **In-pane storage delete rides the approval-gate verb.** The panes and the
  `/model-forge` command are read-only by design: a pane data read cannot
  ride the human-approval gate. Deletion runs through
  `model_forge_storage action=delete`, where the gate names the exact path
  and size before anything is removed — the panes and the command name that
  verb instead of duplicating it.
- **Live desktop rendering is untested in CI.** The tests cover the data
  shapes, the command output rows, the dashboard manifest and bundle shape;
  the Desktop pane and dashboard tab rendering are verified by hand. Both
  renderers degrade to a stated reason when a data source fails — a failure
  state is never replaced with silent defaults.
- **`/model` is core.** Model switching uses the core `/model` picker (and
  `hermes model`); Model Forge writes activation through the documented
  persistence helper and never builds a plugin-side model selector.
- **Telegram output is plain-text rows.** Rows are capped under the 4096-char
  message limit with an honest truncation row; interactive cards beyond
  core's approval/clarify flows are not used.
- **Broker rows are chat-only today.** The cockpit panes show targets,
  engines, models and storage; residency rows live in `/model-forge broker`.
- **Commands are read-only.** Every mutating decision (target registration,
  install, activation writes, storage deletion, installer execution) is a
  tool verb that rides the human-approval gate — the same on every surface,
  fail closed on denial or timeout.

## Approval coverage (W5.2)

Every write, destructive, or policy-relevant action in the tool surface,
mapped to its rule key or its explicit, documented reason for needing no
gate (verified against the handlers in `model_forge/tools.py` and the
`policy_hook`):

| Action | Coverage |
|---|---|
| `model_forge_target_add` (registers where engine calls, media jobs and downloads may go) | Human-approval gate, rule key `model-forge:target_add`; the prompt shows the exact address; fail closed on denial or timeout |
| `model_forge_activate` with `apply=true` (writes `model.default` / `model.provider` / `base_url` **and** a `custom_providers` entry) | Gate, rule key `model-forge:activate`; the prompt names the model, target and the `custom_providers` slug |
| `model_forge_storage` with `action=delete` (destroys model weights on disk) | Gate, rule key `model-forge:storage_delete`; the prompt names the exact path and size; only inventory-returned entries are deletable |
| `model_forge_install` with `action=engine_install`, `execute=true` (runs a vendor installer in a background job) | Gate, rule key `model-forge:engine_install`; the prompt shows the exact command; commands come only from the fixed vendor catalog, never from a model-supplied string |
| `model_forge_install` model pulls (file writes) | No gate — documented reason: destinations are confined to the plugin downloads dir plus user-configured `download_roots` (symlinks refused); nothing existing is destroyed |
| `model_forge_broker` `claim` (loads a model; a vLLM/llama.cpp load **starts a server process**) | Gate when the claim would start a process-engine server, rule key `model-forge:broker_server_start`; a claim over an already-resident model starts nothing and needs no gate; the prompt names engine, model, target, port and the loopback-only bind |
| `model_forge_broker` `release` (restores what was resident at claim time; restoring a non-resident vLLM/llama.cpp model **re-starts a server process**) | Gate when the restore would re-start a process-engine server, rule key `model-forge:broker_server_start`; a release that restores nothing, restores an already-resident model, or restores only HTTP-native engines needs no gate |
| `model_forge_broker` `swap` (exchanges resident weights for one incoming model; a vLLM/llama.cpp load **starts a server process**) | Gate when the swap would start a process-engine server, rule key `model-forge:broker_server_start`; `dry_run=true` changes nothing and needs no gate |
| `model_forge_generate_media` (jobs to a media gateway; results saved under the plugin data dir; the job's GPU-lane claim and release may **start/re-start** a process-engine server) | Gate when the lane claim/release would start or re-start a process-engine server, rule key `model-forge:broker_server_start`; otherwise no gate — the endpoint is user-configured and generated files land only under the plugin data dir |
| `model-forge-providers` `hermes model` setup step (writes `model:` + `custom_providers`) | No gate — documented reason: it runs only from the interactive `hermes model` flow where the user's own pick is the decision; it is not reachable from a model-driven tool call |
| Everything else (`model_forge_status`, `model_forge_browse`, `model_forge_storage list`, `model_forge_broker status/plan/bench`, `/model-forge`, the cockpit panes, the dashboard tab) | Read-only |

## Mobile readiness (W4.3)

Every capability in this matrix is **gateway-side**: the `/model-forge`
command, the tools, the approval gate, and the dashboard API all run through
the gateway, and no capability requires a specific desktop client. A future
mobile chat client therefore inherits this matrix without per-client work.

**On-device mobile inference is a named future lane, not a current
commitment.** Its entry point, when a mobile runtime exists, is a Model
Forge engine adapter (`model_forge/engines/`) plus a compute-target kind for
the device — the same seams remote machines use today.
