# P5c — facade + consumer migration (task file)

Roadmap: docs/broker-roadmap.md "P5 — broker consolidation", workstream (c).
Porting source (READ-ONLY): /home/hermes/work/weights-broker —
broker/main.py route layer (create_app line 1263: /health, /v1/models 1534,
/v1/chat/completions 1607, /v1/images/generations 1649, /v1/videos 1682,
/v1/videos/{jid} 1698, /content 1714, /upload 1725, /v1/videos/generations
1739, /jobs/{jid} 1753, /assets 1765) and config.yaml `aliases:` (line 77)
+ `models:` advertisement (line 303).
NEVER `git reset --hard` in that repo.

## C1 — facade core (routes + aliases + chat passthrough)

- Stdlib-only OpenAI facade module (http.server + urllib seams; repo has no
  third-party deps and tests run in the repo .venv): chat passthrough
  (alias canonicalize BEFORE routing; rewrite to the lane served id;
  Content-Type: application/json explicitly — raw-body passthrough drops
  it), GET /v1/models catalog (registry + models advertisement enrichment +
  output_kind on gen entries + alias backfill so orphaned ids resolve),
  GET /health.
- model_forge/aliases.py: DEFAULT_ALIASES ported from config.yaml `aliases:`
  (chat legacy ids → Qwen/Qwen3.8-27B-FP8; gen legacy ids → current gen
  ids) + resolve(model_id, aliases=None) -> str seam.
- Owner files: model-forge/model_forge/facade.py (new),
  model-forge/model_forge/aliases.py (new), tests/test_facade.py (new),
  tests/test_aliases.py (new).

## C1b — gen routes + job API

- /v1/images/generations (sync {"data":[{"url"}]}), /v1/videos + GET
  /v1/videos/{id} + /content, /v1/videos/generations + /jobs/{jid},
  /upload, /assets/{jid}/{filename} — wired to B1's ComfyWorkflowRunner
  through the orchestrator claim/release.
- Owner files: model-forge/model_forge/facade.py, tests/test_facade.py.

## C2 — Hermes activation + consumer flips

- model_forge_activate (tools.py) writes Hermes custom_providers +
  auxiliary.vision (helpers custom_provider_entry /
  write_custom_provider_entry already exist).
- Flip consumers one at a time, each verified before the next: Hermes
  providers → image plugin → house-scan vision cron → OpenViking .63.

## C3 — decommission (parent-led)

- Broker stays live until every consumer is green on model-forge. Then:
  stop the weights-broker container on .62, archive the Gitea repo, report
  freed resources. Alias map must have worked at every step.

## Acceptance

All consumers green on model-forge; broker retired; freed resources
reported.
