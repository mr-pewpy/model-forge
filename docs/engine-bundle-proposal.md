# Engine Bundles — core-provisioned inference-engine runtimes for plugins

**Draft for review — not yet submitted.**

Audience: Hermes framework maintainers. From the maintainers of Model Forge, a plugin-catalog package
for local-model discovery, provisioning, and weight residency. Deliberately problem-first: roughly
nine parts context, one part proposal — the gap is the argument.

## 1. The problem: plugins that drive local models need engines they may not ship

A plugin that helps everyday users run local models must, at some point, run an inference engine. The
plugin catalog's rules and the platform's current shape leave it no good way to do that.

**Plugins cannot ship engines.** The catalog is source-only: no bundled binaries and no `.pyc`; a
native component must be built by the repository's own public CI with a build attestation. A plugin
package therefore cannot contain an inference engine, an installer payload, or a prebuilt server. The
rule is correct — it is a large part of why catalog plugins are trustworthy — but engine provisioning
must happen outside the plugin package.

**Plugins cannot assume engines exist.** Hermes core's managed runtime is llama.cpp-only (section 2).
The other engines users expect — vLLM, Ollama, LM Studio, llama-swap, and the generative engines —
are manual custom endpoints with no lifecycle, health, or switching. When a plugin tool runs on a
machine with no engine installed, the honest answer today is a failure row telling the user to go
install something.

**So plugins orchestrate vendor installers.** The workable pattern — the one Model Forge is built
around — is to run each vendor's official install command behind Hermes' human-approval gate: plan
before execute, fail closed on denial or timeout. It works, and it is fragmented by construction:

- **Per-vendor, per-fragment.** Each engine has its own installer idiom, install location, update
  mechanism, and failure modes; a plugin supporting three engines maintains three provisioning paths.
- **Inconsistent update and security story.** Vendor installers update on their own schedules, or not
  at all. Nothing centrally records which engine version is on the machine, how it got there, or
  whether a security fix has landed.
- **Dead ends on locked-down machines.** Vendor installers assume the user can run arbitrary shell
  commands and write outside the plugin's data directory. On managed or restricted machines, the
  provisioning path ends.
- **The everyday-user experience is "install three unrelated things by hand."** Product users have no
  mental model of "inference engine." The onboarding for a local media workflow is: find engine A's
  site, install it, find engine B's site, install it, return to Hermes, and hope the plugin detects
  both.

This is not a plugin-authoring failure. It is a missing platform capability: inference engines are a
shared dependency of the local-model plugin ecosystem, and today every plugin re-solves engine
provisioning alone.

## 2. The precedent: Hermes core already owns a managed engine runtime

Core does not treat engines as out of scope. It already runs one as a first-class managed capability
(per the plugin-author runtime map of `hermes_cli/local_runtime/`, ~4,900 lines, as of v0.22.x):

- **A managed llama.cpp runtime with PM-pinned engine builds** — cuda, vulkan, metal, hip, cpu —
  SHA-verified, with engine versions coming only from the package manager's lockfile.
- **One supervised server per Hermes process**: `ensure_local_runtime(config)` moves from config to
  installed binaries to a supervised llama-server in router mode, with preset entries generated from
  launch policy; one supervisor singleton is shared by every profile the process hosts.
- **A hardware-priced model catalog**: curated GGUF models with per-quant sizes, mmproj vision
  projectors, DFlash/MTP draft companions, quality scores, and sampling presets; estimator inputs
  measured on real files, a per-layer memory estimate with a physics check, and a live hardware
  budget probe.
- **Downloads as a managed surface**: Hugging Face search and download behind the catalog, per-tensor
  load progress, in-session context growth, and conservative router recovery.
- **Shared user surfaces**: the Desktop Local Models pane and the terminal `hermes model` command
  drive the same handlers.

The platform has proven it can own an engine end to end — pinning, verification, supervision,
hardware-aware sizing, and a user-facing install experience. What is missing is generality, not
capability: the managed path is llama.cpp-only, and the runtime map's standing rule to plugin authors
is to extend around it through engine HTTP APIs and documented provider hooks, never by importing its
internals.

## 3. What exists on the consumer side: one lifecycle contract across six engines

Model Forge's broker library (`model_forge/broker.py`) is a working consumer-side contract for
exactly the lifecycle surface this proposal would formalize. Its `EngineLifecycle` interface is three
methods — `ensure_loaded`, `release`, `probe` — with these properties:

- **Honest rows, never raises.** Every call returns a plain dict with `ok` and a stated `reason`; an
  engine quirk must not sink the caller, and an unattended run degrades to a stated reason, never a
  stall.
- **Fail-soft observation, fail-closed policy.** `probe` reports `state="unknown"` when an engine is
  unreachable and never raises. Policy decisions fail closed: a missing engine binary stops the plan
  with a stated reason — no argv is built, no half-started server.
- **Injectable seams.** HTTP functions, process runners, binary lookups, and clocks are all
  injectable; the module performs no I/O outside those seams.

One interface already covers six engines with genuinely different lifecycle shapes:

| Engine | Lifecycle shape |
|---|---|
| Ollama | keep-alive warm (`keep_alive`) and evict (`keep_alive=0`) over HTTP |
| llama-swap | observational pass-through — swaps on demand by route; no eviction control |
| llama.cpp (llama-server) | process start/stop on a weights path |
| ComfyUI | readiness plus explicit cache free between generative jobs (`POST /free`) |
| vLLM | process start/stop; engine sleep/wake is not used — wake endpoints are unreliable upstream |
| Maya | engine-native readiness over its model-management API; no unload primitive, so release is an honest ledger-side acknowledgment that never fakes an unload |

These adapters sit on a battle-tested residency and eviction policy distilled from production
multi-GPU serving: fit gates on nominal booked ceilings while live free VRAM only orders candidates,
a lane wake stops resident side models first, a recency window (~600 s) keeps recently used models
warm ahead of pure LRU, image-generation entry evicts GPU residents deterministically every time, and
model loads fire one at a time. The policy layer is proven; the remaining work is expressing it
against raw engine APIs.

The relevance here: engine lifecycle diversity is real, and one small, honest interface absorbs it. A
platform-level lifecycle API does not need to obscure that diversity — it needs to name it.

## 4. The proposal: engine bundles

**Engine bundles**: manifest-declared, versioned inference-engine runtimes that Hermes core provisions
and exposes to plugins through a documented lifecycle API.

- **Declared and versioned.** Each engine ships as a bundle with a manifest (section 5) recording
  name, version, upstream provenance, checksums, hardware requirements, and endpoint shape.
- **Core-owned provisioning on user approval.** Core downloads and installs bundle artifacts through
  its managed runtime path, on explicit user approval, with one-click install in Desktop. Plugins
  never handle binaries — consistent with the catalog's source-only rule.
- **A documented lifecycle API for plugins.** Load, release, probe, and endpoint discovery — the
  vocabulary Model Forge already implements — so plugins drive engines through one contract instead
  of per-vendor installer commands.
- **Hardware requirements per bundle**, declared in the manifest so core can reason about fit before
  provisioning and plugins can answer "will not run here" honestly instead of failing at runtime.

## 5. Sketch

**Bundle manifest** (illustrative fields):

```yaml
name: vllm
version: "0.x.y"
upstream: https://github.com/vllm-project/vllm   # provenance
artifacts:
  - platform: linux-x86_64-cuda
    url: ...
    sha256: ...
hardware:
  requires: [cuda]        # accelerator family
  vram_gb_min: ...
endpoints:
  base_url_default: http://127.0.0.1:8000
  health: /health
  models: /v1/models
```

**Plugin-facing API shape** — reuse the consumer-tested vocabulary:

```
ensure_loaded(model_id, **kwargs) -> dict   # honest row: ok + reason
release(model_id) -> dict
probe() -> dict                             # fail-soft: "unknown" when unreachable
endpoint(bundle_id) -> dict                 # endpoint discovery: base URL, health, model listing
```

**Approval and disclosure** — the conventions already exist in the plugin catalog and need no new
machinery: provisioning runs behind the human-approval gate with plan-before-execute and fails closed
on denial or timeout; shell commands, background processes, and network calls are disclosed in the
plugin README and submission; unattended runs degrade to a stated reason rather than hanging. Bundles
let a plugin's disclosure say "uses the vLLM engine bundle" instead of "runs the vendor's install
script."

## 6. Non-goals

- **Core does not manage GPU drivers or engine tuning.** Drivers stay with the operating system and
  vendor; engine configuration and tuning stay with the engines.
- **No third-party binaries inside plugin packages.** The catalog's source-only rule is unchanged;
  bundles are core-distributed, not plugin-distributed.
- **Engine-native configuration stays with the engines.** A bundle exposes endpoints and lifecycle,
  not a new configuration layer over the engine's own.

## 7. Ecosystem benefit and a concrete next step

- **Catalog plugins stop shelling out.** One provisioning path to maintain, review, and disclose —
  instead of one per vendor.
- **Users get one install path.** "Install three unrelated things by hand" becomes one approved
  action in Desktop.
- **Engines get a consistent security story.** Versioned bundles with checksums, centrally recorded
  installs, and updates that arrive through the platform rather than each vendor's installer.

We are not asking core to adopt a policy designed elsewhere: the residency and eviction rules in
section 3 are proven in production multi-GPU serving and generalize to any machine, and the lifecycle
contract is implemented and tested today. Model Forge can be the reference implementation consumer of
the interface — we would rather adapt our adapters to a core API than keep maintaining per-vendor
installer paths. We would welcome iterating on the manifest and API shape with the maintainers of
`local_runtime`: a draft interface reviewed against a real consumer is cheaper than one designed in
the abstract.
