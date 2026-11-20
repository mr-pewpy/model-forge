# P5a — Docker runtime lifecycle (task file)

Roadmap: docs/broker-roadmap.md "P5 — broker consolidation", workstream (a).
Porting source (READ-ONLY): /home/hermes/work/weights-broker (broker/main.py
ContainerControl at line 379, Lifecycle at 537; config.yaml `registry:`).
NEVER `git reset --hard` in that repo (stash holds an old vLLM-Omni experiment).

## A1 — RuntimeHandle + DockerContainerLifecycle

- Generalize broker.py's `ProcessHandle` (line ~748) seam into
  `RuntimeHandle` (kinds: process | container). Keep `ProcessHandle` as an
  alias so the existing llama.cpp / vLLM adapters and tests stay green.
- New `DockerContainerLifecycle(EngineLifecycle)`: `ensure_loaded` =
  `docker start` + health probe with cold-boot budget; `release` =
  `docker stop` (instant VRAM free); `probe` fail-soft. Docker access via an
  injectable runner (default: subprocess `docker ...` — the facade will run
  on .62 with the host docker CLI); unit tests use fakes, never real docker.
- Registry entries gain `runtime: {kind, name, health_url, boot_budget_s,
  gpu_count}` (registry.py ModelEntry optional field, validated). Factory
  maps kind=container entries to DockerContainerLifecycle.
- Budgets: vLLM 90–170 s warm / ~8 min cold; ComfyUI minutes.
- Owner files: model-forge/model_forge/broker.py, model-forge/model_forge/
  registry.py, tests/test_runtime_handle.py (new).

## A2 — LLM-waiter priority (roadmap §3 rule)

- While a chat claim waits on a lane wake, refuse new generative claims with
  an honest reason; the waiter owns the wake. Counter semantics mirror the
  broker's `llm_waiters` (broker/main.py Lifecycle.prepare_llm /
  prepare_gen).
- Owner files: model-forge/model_forge/broker.py, tests/test_llm_waiter.py
  (new).

## Acceptance (integration, verified on .62)

orchestrator claim(chat) → vllm-server up; claim(image) → lane evicted →
comfyui up → job → idle release → lane restored. Tested on throwaway
containers first (safe-adoption pattern; never disturb a live TP2 job).
