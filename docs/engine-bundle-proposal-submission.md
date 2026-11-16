# Engine-bundle proposal — submission package

**Prepared for review — do not post without the maintainer-of-record's approval.**

## Venue note

Recommended venue: a free-form GitHub Issue on `NousResearch/hermes-agent`. Discussions are not
enabled on that repository (the repository API reports `has_discussions: false` and the
`/discussions` page returns 404), even though CONTRIBUTING.md's Community section references GitHub
Discussions for design proposals. In practice, design and RFC work is filed as issues there — for
example "Design proposal: default-off exec delivery for deterministic webhook handlers",
"RFC: remote plugin seam", and "Design proposal: make the live local gateway a first-class Desktop
backend". An issue titled in that convention matches how the maintainers already receive this kind
of proposal. Conventions and the exact next step are in the NOTES section below.

---

## Post title (paste into the issue title field)

```text
Design proposal: engine bundles — core-provisioned inference-engine runtimes for plugins
```

## Post body (paste into the issue body field)

```markdown
We maintain Model Forge, a plugin-catalog package for local-model discovery, provisioning, and weight
residency. We are raising this now because the lifecycle contract behind it is implemented and tested
today, so the API shape can be reviewed against a real consumer instead of designed in the abstract.

**The problem.** Plugins that drive local models need inference engines they may not ship. The plugin
catalog is source-only — no bundled binaries — core's managed runtime is llama.cpp-only, and the
remaining engines are manual custom endpoints with no lifecycle. The workable pattern today is for
plugins to orchestrate each vendor's official installer behind the human-approval gate. That is
fragmented by construction: one provisioning path per engine, an inconsistent update and security
story with no central record of installed engine versions, dead ends on locked-down machines where
vendor installers cannot run, and an everyday-user onboarding of "install three unrelated things by
hand."

**The precedent.** Core already owns a managed engine runtime end to end — pinned, checksum-verified,
supervised llama.cpp with hardware-aware sizing and a user-facing install experience; what is missing
is generality, not capability.

**The ask — engine bundles, manifest-declared and versioned:**

- Manifest-declared, versioned engine bundles: each records name, version, upstream provenance,
  checksums, hardware requirements, and endpoint shape.
- Core-owned provisioning on explicit user approval: core downloads and installs bundle artifacts
  through its managed runtime path (one-click install in Desktop); plugins never handle binaries,
  consistent with the catalog's source-only rule.
- A documented lifecycle API for plugins reusing the `ensure_loaded` / `release` / `probe` vocabulary
  Model Forge already implements, plus endpoint discovery.
- Per-bundle hardware requirements declared in the manifest, so a plugin can answer "will not run
  here" honestly instead of failing at runtime.

**Evidence.** Model Forge's broker library is a working consumer-side contract for exactly this
lifecycle surface: one interface covering six engines with genuinely different lifecycle shapes.

Model Forge can serve as the reference-implementation consumer — we would rather adapt our adapters
to a core API than keep maintaining per-vendor installer paths — and we would welcome iterating on
the manifest and API shape with the maintainers of `local_runtime`. Full proposal (problem,
precedent, manifest sketch, API shape, non-goals):
https://github.com/mr-pewpy/model-forge/blob/main/docs/engine-bundle-proposal.md
```

---

## NOTES (not part of the post — do not paste into the issue)

**Recommended venue and why.** A free-form GitHub Issue on `NousResearch/hermes-agent`, titled in the
repository's design-proposal convention. Rationale: GitHub Discussions are not enabled on the
repository (API `has_discussions: false`; `/discussions` returns 404), so the CONTRIBUTING.md pointer
to Discussions is not actionable; design proposals and RFCs in that tracker are issues (examples:
"Design proposal: default-off exec delivery for deterministic webhook handlers" #83050, "RFC:
remote plugin seam" #138924, "Design proposal: make the live local gateway a first-class Desktop
backend" #109891).

**Repository conventions found.**

- Title prefixes: existing design work uses "Design proposal: …" or "RFC: …". The Feature Request
  issue template (`feature_request.yml`) imposes a "[Feature]: " title prefix and required fields
  (Problem or Use Case, Proposed Solution, Feature Type dropdown — "Other" fits a platform
  capability); the post body above maps onto those fields if the template is used instead of a blank
  issue.
- Issue templates present: `bug_report.yml`, `feature_request.yml`, `setup_help.yml`. Blank issues
  are enabled (`config.yml`: `blank_issues_enabled: true`), and precedent design-proposal issues are
  free-form rather than template-driven.
- CONTRIBUTING.md requires searching existing issues and PRs before filing (duplicate check) —
  search "engine bundle", "local_runtime", "runtime" first and note any hits in the post if found.
- Labels on design/RFC issues (`type/feature`, `needs-decision`, `comp/plugins`, `comp/gateway`,
  `P3`) are applied by maintainers during triage; do not self-assign.
- Structure of precedent design-proposal issues: short motivation and design-summary sections, with
  the deep material linked (as this post links the full proposal). Some RFCs are opened as draft
  pull requests with a "What does this PR do?" body; that path suits changes with an accompanying
  implementation, not this proposal.

**Exact next step.** The maintainer-of-record (the human owner of this submission) reviews this
package and the full proposal, then personally opens the issue on `NousResearch/hermes-agent` by
pasting the title and body above — at a time of their choosing, after approval. This submission must
not be auto-posted: no agent, automation, or third party may post it on the maintainer-of-record's
behalf.
