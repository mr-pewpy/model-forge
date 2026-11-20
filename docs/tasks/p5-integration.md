# P5 integration — facade shadow deploy + container-dance acceptance

**Model:** deepseek/deepseek-v4.1-flash
**Repo source:** `/home/hermes/model-forge` @ `220f72c` (+ the small stub-config patch below, uncommitted)
**Target host:** `root@192.168.1.62` (LXC `ai`, host python3 3.11.2 at `/usr/bin/python3`, host docker CLI)
**Status:** ✅ all acceptance runs a–f PASS against throwaway stub containers. The live
stack (`weights-broker` :8080, `vllm-server`, `weights-broker-comfyui-1`, `gluetun`) was
**never** stopped, restarted or touched.

---

## 1. Deploy layout on .62

```
/root/model-forge-shadow/
├── model_forge/                 # package copied from the repo (commit 220f72c + patch)
├── facade-shadow.json           # FACADE_CONFIG for the shadow deployment
├── stubs/
│   ├── vllm_stub.py             # throwaway vLLM/OpenAI stub  (port 9071)
│   └── comfy_stub.py            # throwaway ComfyUI stub       (port 9072)
├── stubctl/                     # marker dir: slowchat / slowcomfy toggle stub latency
└── assets/                      # ASSETS_DIR; assets/out/ is the runner's COMFY_OUTPUT_DIR
```

Package transfer: `tar` of `model_forge/` (excluding `__pycache__`) scp'd as one file,
extracted in place. **md5 verified** on 5 files against the local source — identical:
`serve.py f7415629…`, `facade.py 2eb9db60…`, `broker.py 8e3168ac…`,
`engines/comfyui.py 306a0b05…`, `aliases.py 38317a09…`.

### Service unit (shadow facade on :8081)

```bash
systemd-run --unit=mf-shadow --collect --working-directory=/root/model-forge-shadow \
  bash -c 'FACADE_CONFIG=/root/model-forge-shadow/facade-shadow.json \
           exec /usr/bin/python3 -m model_forge.serve'
```

Startup log line:

```
[model-forge-serve] listening on 127.0.0.1:8081 — lanes [comfyui:mf-test-comfy, vllm:mf-test-vllm] — target 'comfy', budget 48 GB
```

Control: `systemctl stop|start|status mf-shadow`, `journalctl -u mf-shadow`. No restart
policy on the unit or the stub containers.

### `facade-shadow.json`

```json
{
  "host": "127.0.0.1",
  "port": 8081,
  "LLM_BASE_URL": "http://127.0.0.1:9071/v1",
  "LLM_SERVED_ID": "Qwen/Qwen3.8-27B-FP8",
  "LLM_CANONICAL_ID": "Qwen/Qwen3.8-27B-FP8",
  "COMFYUI_BASE_URL": "http://127.0.0.1:9072",
  "ASSETS_DIR": "/root/model-forge-shadow/assets",
  "PUBLIC_BASE_URL": "http://127.0.0.1:8081",
  "CHAT_CONTAINER": "mf-test-vllm",
  "CHAT_HEALTH_URL": "http://127.0.0.1:9071/health",
  "GEN_CONTAINER": "mf-test-comfy",
  "GEN_HEALTH_URL": "http://127.0.0.1:9072/",
  "CHAT_BOOT_S": 30,
  "GEN_BOOT_S": 30,
  "COMFY_OUTPUT_DIR": "/root/model-forge-shadow/assets/out"
}
```

The last seven keys are the stub-config expressiveness seam (see §4). Everything else is
the existing documented config surface.

---

## 2. Throwaway stub design

Both stubs run detached with `--network host`, **no restart policy**, from a locally
present python-capable image (`vllm/vllm-openai:latest`, python 3.12.13) via
`--entrypoint python3` — **no image pulls** (network throttled). The stub scripts are
bind-mounted read-only; the output/assets dir is bind-mounted at the same host path so the
comfy stub writes where the runner resolves outputs.

```bash
IMG=vllm/vllm-openai:latest
docker run -d --name mf-test-vllm --network host \
  -v /root/model-forge-shadow/stubs:/stubs:ro \
  -v /root/model-forge-shadow/stubctl:/stubctl \
  --entrypoint python3 "$IMG" /stubs/vllm_stub.py

docker run -d --name mf-test-comfy --network host \
  -v /root/model-forge-shadow/stubs:/stubs:ro \
  -v /root/model-forge-shadow/stubctl:/stubctl \
  -v /root/model-forge-shadow/assets:/root/model-forge-shadow/assets \
  --entrypoint python3 "$IMG" /stubs/comfy_stub.py
```

Names `mf-test-vllm` / `mf-test-comfy` cannot collide with the live containers. Both are
created running then stopped, so the dance starts from a known stopped state.

**vLLM stub (9071):** `GET /health → 200 {}`; `GET /v1/models → OpenAI list`;
`POST /v1/chat/completions → {model, choices[0].message.content:"stub-echo:<model>"}` echoing
the request model id. Logs method/path/Content-Type/body to stdout. Marker
`/stubctl/slowchat` makes the chat route sleep 8 s (hold an LLM waiter open).

**ComfyUI stub (9072):** `GET /` and `/system_stats → 200`; `POST /prompt → {"prompt_id":"p1"}`
(and writes **real files** `assets/out/out.png` + `out.mp4` so the collected path exists);
`GET /history/p1 → completed entry` carrying **both** an `images` and a `videos` bucket (so the
same stub serves image and video lanes); `GET /history/slow → {}` (unfinished → honest
timeout); `POST /upload/image → {"name":"x.png"}`; `POST /free → {}`.

`_collect_outputs` resolves `os.path.join(output_dir, subfolder, filename)`, so the runner's
`output_dir` is set to `assets/out` (under `ASSETS_DIR`) — this is also what makes
`GET /assets/{jid}/{name}` serve the file (`_within_assets` confines serving to `assets_dir`).

**Critical stub detail — SIGTERM handler.** As container PID1 the process gets *no* default
kernel signal disposition, so without an explicit handler `docker stop` burned the full 10 s
SIGKILL grace. Because `DockerContainerLifecycle.release` issues `docker stop` **fire-and-forget**
(`subprocess.Popen`, no wait) that slow stop raced the lane-restore `docker start` and left the
restored chat container dead. Both stubs install `signal(SIGTERM/SIGINT, os._exit(0))` so
`docker stop` returns in ~0.1 s; the dance is then deterministic. (See §5 finding 2.)

---

## 3. Acceptance runs (all PASS)

All commands run on .62 against `http://127.0.0.1:8081`.

| # | Check | Command (abridged) | Response snippet | Result |
|---|-------|--------------------|------------------|--------|
| **a** | `/v1/models` → 8 public ids + alias keys resolve | `curl -s :8081/v1/models` | 18 entries: public `Qwen/Qwen3.8-27B-FP8` + 7 `gen/*`; aliases incl. `gen/krea2` with `alias_of: gen/qwen-image-2.1`; `research/qwen38-uncensored` present | **PASS** |
| **b** | legacy alias chat → stub gets CANONICAL id | `POST /v1/chat/completions {"model":"research/qwen38-uncensored"}` | resp `model:"Qwen/Qwen3.8-27B-FP8"`; vllm stub log: `POST /v1/chat/completions Content-Type=application/json BODY={"model": "Qwen/Qwen3.8-27B-FP8", …}`; containers before `(none)` → after `mf-test-vllm Up` | **PASS** |
| **c1** | chat wake: stopped container → facade starts + leaves up | `docker stop mf-test-vllm`; `POST /v1/chat/completions` | before: both Exited; `chat HTTP=200`; after: `mf-test-vllm Up 2s`, `mf-test-comfy Exited` | **PASS** |
| **c2** | gen dance: evict chat → comfy up → job → restore chat | `POST /v1/images/generations {"model":"gen/qwen-image-2.1"}` | `{"data":[{"url":"http://127.0.0.1:8081/assets/8f84dbe45ce4/out.png"}]}`; comfy stub log `POST /free`,`POST /prompt`,`GET /history/p1`; after idle release: `mf-test-comfy Exited(0)`, `mf-test-vllm Up 3s`; asset URL → `HTTP=200 CT=image/png bytes=22` | **PASS** |
| **d** | LLM-waiter refuses gen before any claim/eviction | touch `slowchat`; background slow chat; while in-flight `POST /v1/images/generations` | `GEN1 HTTP=503` `{"error":{"message":"image job … failed: refused: a chat request owns the lane wake — 1 LLM waiter(s) pending for qwen-image-2.1 (roadmap §3 …)","type":"busy"}}`; comfy still `Exited`; after chat done `GEN2 HTTP=200` | **PASS** |
| **e** | async video job API | `POST /v1/videos {"model":"gen/h3-t2v"}` → poll `GET /v1/videos/{id}` → `/content` | submit `{"id":"8ae91f561383","status":"queued","model":"h3-t2v"}`; poll …`completed` `data[0].url=…/assets/8ae91f561383/out.mp4`; content `HTTP=200 CT=video/mp4 bytes=14 "STUB-MP4-BYTES"` | **PASS** |
| **f** | sync image `_timeout` vs slow history → honest timeout | touch `slowcomfy`; `POST /v1/images/generations {"_timeout":1}` | `{"error":{"message":"image job … failed: history poll timed out after 3600.0s for slow","type":"generation_error"}}` `HTTP=504 time=6.0s` (returns, no hang) | **PASS** |

Container states were recorded after every phase (see c1/c2/d/e/f snippets). Live-stack
sanity after all runs: `weights-broker Up`, `vllm-server Up (healthy)`, `gluetun Up`, broker
`:8080/health → 200` — untouched throughout.

---

## 4. Repo patch — stub-config expressiveness (needed for this acceptance)

serve.py **hardcoded** the real container names, health URLs and boot budgets in
`TOPOLOGY_ENTRIES`, and the runner's `output_dir` was fixed at `/output`. There was no
env/JSON way to point the lanes at stub containers. The patch adds seven whitelisted
config fields (production-defaults byte-identical to the former constants) and threads them
through `topology_entries()`, `topology_bookings()` and `build_stack()`. Tests unaffected
(`test_serve.py` + `test_facade.py` = **80 passed**).

Files: `model-forge/model_forge/serve.py`, `model-forge/model_forge/facade.py`.
Authoritative diff: [`p5-stub-config.patch`](./p5-stub-config.patch) (226 lines). New
env/JSON keys → `FacadeConfig` fields:

| Env / JSON key | Field | Production default |
|---|---|---|
| `CHAT_CONTAINER` | `chat_container` | `vllm-server` |
| `CHAT_HEALTH_URL` | `chat_health_url` | `http://127.0.0.1:8000/health` |
| `GEN_CONTAINER` | `gen_container` | `weights-broker-comfyui-1` |
| `GEN_HEALTH_URL` | `gen_health_url` | `http://127.0.0.1:18188/` |
| `CHAT_BOOT_S` | `chat_boot_s` | `480.0` |
| `GEN_BOOT_S` | `gen_boot_s` | `600.0` |
| `COMFY_OUTPUT_DIR` | `comfy_output_dir` | `/output` |

Key hunks (full patch in the `.patch` file):

```diff
+# facade.py — new FacadeConfig fields
+    chat_container: str = "vllm-server"
+    chat_health_url: str = "http://127.0.0.1:8000/health"
+    gen_container: str = "weights-broker-comfyui-1"
+    gen_health_url: str = "http://127.0.0.1:18188/"
+    chat_boot_s: float = 480.0
+    gen_boot_s: float = 600.0
+    comfy_output_dir: str = "/output"

# serve.py — TOPOLOGY_ENTRIES constant becomes a config-driven factory
+def topology_entries(config: "FacadeConfig | None" = None) -> tuple[ModelEntry, ...]:
+    config = config or FacadeConfig()
+    return ( ModelEntry(… runtime={"name": config.chat_container,
+                                   "health_url": config.chat_health_url,
+                                   "boot_budget_s": float(config.chat_boot_s), …}),
+             ModelEntry(… runtime={"name": config.gen_container, …}) )
+TOPOLOGY_ENTRIES: tuple[ModelEntry, ...] = topology_entries()   # unchanged defaults

# serve.py — build_stack threads config through
-    ledger = ResidencyLedger(bookings=topology_bookings(), clock=clock)
+    entries = topology_entries(config)
+    ledger = ResidencyLedger(bookings=topology_bookings(config), clock=clock)
-    for entry in TOPOLOGY_ENTRIES:
+    for entry in entries:
-    runner = comfy_runner or ComfyWorkflowRunner(config.comfy_base_url, http=http)
+    runner = comfy_runner or ComfyWorkflowRunner(
+        config.comfy_base_url, http=http, output_dir=config.comfy_output_dir)

# serve.py — 7 new env keys in _ENV_TO_FIELD + 7 defaults in resolve_settings()
```

No commits made (per constraint). Apply with `git apply docs/tasks/p5-stub-config.patch`.

---

## 5. Findings / limitations (honest)

1. **Chat claim does not heal an externally-stopped chat container while the model is still
   ledger-resident.** `handle_chat` releases the chat scope with `restore=False`, so the
   chat model stays *resident in the ledger* after the first request. A later
   `docker stop mf-test-vllm` is then invisible: the planner emits **no load step**
   (`pending` is empty) and the request 502s on the dead upstream. This is exactly why
   acceptance c1 is run against a **freshly restarted facade** — with an empty ledger the
   first chat request produces the load step and wakes the container. In production the chat
   container runs continuously so it does not bite, and **any gen job naturally clears the
   chat residency** (its claim records the chat model released), after which the next chat
   re-wakes. Recommended follow-up: make the chat claim force an `ensure_loaded`/probe (parity
   with the broker's `prepare_llm`), or have `DockerContainerLifecycle.ensure_loaded` verify
   docker state instead of trusting its cached `_running`.
2. **`DockerContainerLifecycle.release` issues `docker stop` fire-and-forget** (`subprocess.Popen`,
   never waited). Harmless in production (a gen job runs for minutes ≫ the stop grace), but a
   sub-10 s job against a container that ignores SIGTERM can race the lane-restore `docker start`.
   Documented here because it forced the stub SIGTERM handler (see §2).
3. **Cosmetic:** the runner's timeout reason hardcodes `history_timeout_s` —
   `"history poll timed out after 3600.0s for slow"` — even when the honored budget was the
   request's smaller `_timeout` (1 s). The timeout really did use `min(_timeout, history_timeout_s)`;
   only the message text is misleading.

---

## 6. RUNBOOK — real cutover (next phase)

**Preconditions / checks (before touching anything):**
1. Confirm no live generative job on the broker: `docker logs --tail 50 weights-broker` and
   `curl -s http://127.0.0.1:8080/admin/status` → queue empty / no running job.
2. Note current live container names/ports: `weights-broker` (:8080), `vllm-server`
   (:8000, chat lane), `weights-broker-comfyui-1` (gen lane, :18188).

**Cutover:**
3. Stop the broker (this frees :8080 for the facade to take, if flipping ports):
   `docker stop weights-broker`   ← **the only live-container action; reversible.**
4. Write the production `FACADE_CONFIG` (real names + real budgets):
   ```json
   {
     "host":"0.0.0.0", "port":8080,
     "LLM_BASE_URL":"http://127.0.0.1:8000/v1", "LLM_SERVED_ID":"Qwen/Qwen3.8-27B-FP8",
     "COMFYUI_BASE_URL":"http://127.0.0.1:18188",
     "ASSETS_DIR":"/var/lib/model-forge/assets", "PUBLIC_BASE_URL":"http://192.168.1.62:8080",
     "CHAT_CONTAINER":"vllm-server", "CHAT_HEALTH_URL":"http://127.0.0.1:8000/health",
     "GEN_CONTAINER":"weights-broker-comfyui-1", "GEN_HEALTH_URL":"http://127.0.0.1:18188/",
     "CHAT_BOOT_S":480, "GEN_BOOT_S":600, "COMFY_OUTPUT_DIR":"/output"
   }
   ```
   Either take :8080 (above) or **keep the facade on :8081** and flip consumers to :8081 /
   a reverse proxy. `GEN_TARGET_ID` stays default `comfy`.
5. Relaunch the unit against the production JSON (same `systemd-run` as §1, new
   `FACADE_CONFIG`); confirm the startup log names the real containers.

**Real acceptance:**
6. `POST /v1/chat/completions` (fresh facade, empty ledger) → chat claim brings `vllm-server`
   up, request forwarded, lane left up.
7. `POST /v1/images/generations {"model":"gen/qwen-image-2.1"}` → image claim evicts the chat
   resident (`docker stop vllm-server`), starts `weights-broker-comfyui-1`, job runs, idle
   release restores `vllm-server`. Record `docker ps` after each phase.
8. Optional rollback rehearsal: re-run step 6 after `docker stop vllm-server`.

**Rollback (any time):**
9. `systemctl stop mf-shadow` (releases open scopes on shutdown) and
   `docker start weights-broker` to restore the broker on :8080.

---

## 7. REAL CUTOVER ACCEPTANCE (the .62 production window)

**Model:** deepseek/deepseek-v4.1-flash
**Repo source:** `/home/hermes/model-forge` @ `3b57bd3`
**Date:** 2026-11-20 (UTC) — executed live on the `ai` LXC (192.168.1.62)
**Result:** ✅ **all five real-acceptance checks PASS. No rollback. `mf-facade` left
running on :8080.** `weights-broker` remains stopped (facade owns :8080). `gluetun`
untouched throughout.

### 7.1 Package re-deploy (pre-window)

The shadow copy at `/root/model-forge-shadow/model_forge/` predated the heal +
wait-on-stop fixes, so the whole `model_forge/` package was re-deployed from the
`3b57bd3` repo tree (tar, `__pycache__` excluded) before the window; the old copy is
kept at `model_forge.pre-heal/`. **md5 verified identical to local source:**

| file | md5 (matches local) |
|---|---|
| `serve.py` | `f74156296013ccb96be9a17dee6eefb7` |
| `facade.py` | `2eb9db6028478542a6c81d88ce9c5426` |
| `broker.py` | `2b64d91584c36913b29dd0aadf0580ee` |
| `engines/comfyui.py` | `1e8b49102609ab85a089a18a5e89efcb` |
| `aliases.py` | `38317a098f2a19b2d890a5496d425321` |

(`broker.py` / `comfyui.py` differ from the §1 220f72c hashes — those are the newer
heal / wait-on-stop revisions. `serve.py` / `facade.py` / `aliases.py` match.)

### 7.2 Window timeline (UTC)

| T | event |
|---|---|
| `04:08:26Z` | pre-window idle recheck: `gen.busy=false`, `gen.container_running=false`, `llm.awake=true`; comfy container stopped |
| `04:08:40Z` | shadow window opens — `systemctl stop mf-shadow`; `docker rm -f mf-test-vllm mf-test-comfy` |
| `04:09:01Z` | **`docker stop weights-broker`** — :8080 goes dark (verified) |
| `04:09:02Z` | **`systemd-run --unit=mf-facade`** up on :8080; log `listening on 0.0.0.0:8080 — lanes [comfyui:weights-broker-comfyui-1, vllm:vllm-server]` |
| `04:10:08Z` | image job submitted (`gen/qwen-image-2.1`, 1024×1024) |
| `~04:10:10Z` | `docker stop vllm-server` (evict) → `weights-broker-comfyui-1` Up |
| `04:11:04–04:12:59Z` | ComfyUI on GPU1: 4.8 → 10.8 → 21.5 GB, 100% util; GPU0 ≈ 1 MiB |
| `~04:13Z` | job done; `qwen_image_21_00013_.png` written; idle release → comfy Exited, `vllm-server` started |
| `04:15:03Z` | `vllm-server /health → 200` (restore confirmed; both GPUs ≈ 22.6 GB) |

### 7.3 Accepted config (facade-real.json)

Real topology + real budgets. `host 0.0.0.0`, `port 8080`, `LLM_BASE_URL
http://127.0.0.1:8000/v1`, `LLM_SERVED_ID Qwen/Qwen3.8-27B-FP8`, `COMFYUI_BASE_URL
http://127.0.0.1:18188`, `CHAT_CONTAINER vllm-server`, `GEN_CONTAINER
weights-broker-comfyui-1`, `CHAT_BOOT_S 480`, `GEN_BOOT_S 600`,
`PUBLIC_BASE_URL http://192.168.1.62:8080`.

**Two deviations from the literal runbook/instruction, both deliberate:**

1. **`COMFY_OUTPUT_DIR=/data/broker/comfy/output`** — the real host path bound into
   the container at `/opt/ComfyUI/output` (`weights-broker-comfyui-1` volume).
2. **`ASSETS_DIR=/data/broker`** (not `/root/model-forge-shadow/assets`). The
   `GET /assets/{jid}/{name}` route refuses to serve any file that does not resolve
   under `assets_dir` (`FacadeConfig._within_assets`). Since the runner records the
   **on-disk ComfyUI output path** (`os.path.join(output_dir, sub, filename)`), the
   real output dir must be an ancestor-inclusive child of `assets_dir` for the asset
   URL to 200. `/data/broker` is that ancestor; the literal shadow assets dir would
   have 404'd every real gen asset. (The broker avoids this by *download*-ing outputs
   into `assets_dir` over HTTP; the facade reads them in place.) This is a genuine
   port difference worth a follow-up if `ASSETS_DIR` must stay at the shadow path.

**vLLM API key:** read from `/root/docker/weights-broker/.env` (`BROKER_VLLM_API_KEY`)
— sha256-identical to `/root/docker/local-ai/.env` `VLLM_API_KEY` (verified, not
printed) — injected via `systemd-run --setenv=BROKER_VLLM_API_KEY=…` (env-only per
§4). Value never echoed.

### 7.4 Real acceptance — per check

| # | Check | Evidence | Result |
|---|-------|----------|--------|
| **A** | `GET /v1/models` surface matches pre-flight snapshot | 18 entries: **8 public ids identical to the broker snapshot** (`Qwen/Qwen3.8-27B-FP8`, `gen/qwen-image-2.1`, `gen/qwen-image-2.1-edit`, `gen/qwen-image-2.1-cutout`, `gen/moss-image`, `gen/h3-t2v`, `gen/h3-i2v`, `gen/h3-r2v`) + 10 aliases incl. `openbmb/MiniCPM-V-4.6→Qwen/Qwen3.8-27B-FP8`, `gen/krea2→gen/qwen-image-2.1` | **PASS** |
| **B** | real vLLM chat + legacy-alias chat | `POST /v1/chat/completions {model:Qwen/Qwen3.8-27B-FP8}` → real completion, `model:Qwen/Qwen3.8-27B-FP8`; `vllm-server Up 5 hours` **before and after** (adopt-without-restart happy path). Alias `openbmb/MiniCPM-V-4.6` → resolved `model:Qwen/Qwen3.8-27B-FP8`, 200. `vllm-server` log: `POST /v1/chat/completions 200 OK` (auth forwarded) | **PASS** |
| **C** | real claim dance | `POST /v1/images/generations {gen/qwen-image-2.1}` → response `{"data":[{"url":"http://192.168.1.62:8080/assets/34db0586f283/qwen_image_21_00013_.png"}]}`. Mid-run: `vllm-server` **gone** from `docker ps`, `weights-broker-comfyui-1` Up, GPU0≈1 MiB / GPU1 rising to **21.5 GB @100%**. Asset URL `HTTP=200 CT=image/png bytes=955478` (`PNG 1024x1024 RGBA`). Idle release → comfy Exited + `vllm-server` restored (`Up`, healthy, both GPUs ≈22.6 GB) | **PASS** |
| **D** | chat on the restored lane | post-restore `POST /v1/chat/completions` → real completion, `model:Qwen/Qwen3.8-27B-FP8`, `finish_reason:length`(max_tokens=8); `system_fingerprint vllm-0.24.0-tp2` | **PASS** |
| **E** | LLM-waiter refuses gen; comfy not started | `docker stop vllm-server` → bg chat wakes it (`Up 11s, health: starting`); gen fired mid-boot → **`HTTP=503` in 0.0004 s**: `"refused: a chat request owns the lane wake — 1 LLM waiter(s) pending for qwen-image-2.1 (roadmap §3 …)"`; `weights-broker-comfyui-1` stayed **Exited**. Waiter's chat then returned normally (EXIT=0, real vLLM reply) | **PASS** |

### 7.5 Container / VRAM observations

- **Adopt-without-restart (B):** with `vllm-server` already healthy, the first chat
  produced **no load step** — the container stayed `Up 5 hours` across the request.
  The heal happy path behaves as intended on the real lane.
- **Eviction is real and total:** GPU0 dropped 22.7 GB → 1 MiB the instant
  `vllm-server` stopped; ComfyUI then grew GPU1 to 21.5 GB at 100% util.
- **Restore is warm and in-budget:** `vllm-server` was back `healthy` well inside the
  200 s target (up by ~04:13:28Z, `/health 200` at 04:15:03Z); no boot-budget pressure.
- **Observation (not a failure):** the *synchronous* image response was returned
  **after** the job + lane restore completed (~4 min after submit, job itself ~3 min),
  so the client blocks for the whole dance including reload. Worth documenting for
  consumer timeouts. (The §6 broker note's 240 s gateway figure would be tight here.)
- **Minor surface drift (cosmetic):** the facade's alias table maps
  `research/qwen38-uncensored` and `OBLITERATUS/Qwen3.8-27B-OBLITERATED` to
  `Qwen/Qwen3.8-27B-FP8` (the broker maps them to the separate research container,
  which is not in the facade's two-lane topology). Count and public ids match; only
  these research aliases' targets differ.

### 7.6 Rollback

**Not invoked — all checks PASS.** `mf-facade` remains running on :8080; broker
`weights-broker` remains stopped by design. Rollback remains available per §6:
`systemctl stop mf-facade && docker start weights-broker`.
No live container was removed, no image pruned, `gluetun` never touched, no git commit.

---

## 8. Consumer flips — the four live consumers onto model-forge (2026-11-20)

**Model:** deepseek/deepseek-v4.1-flash
**Endpoint:** `http://192.168.1.62:8080` — `mf-facade` active; `weights-broker` stopped (address unchanged,
so consumers already reached the facade). This phase refreshed each consumer's config to canonical ids
and re-verified it green, one at a time.

**Facade logging reality (correction to the task premise):** the facade handler overrides
`BaseHTTPRequestHandler.log_message` (`model_forge/facade.py:1118` — deliberately *quiet*), so
`journalctl -u mf-facade` carries **only** the two startup lines
(`listening on 0.0.0.0:8080 — lanes [comfyui:weights-broker-comfyui-1, vllm:vllm-server]`, and the
systemd `Started` line). There are **no per-request journal lines**. Request-landing evidence below
therefore comes from the **backend container logs** (`vllm-server` for chat/vision, `weights-broker-comfyui-1`
for gen) and the facade response itself.

### 8.1 Hermes providers (this host, profile `charles`)

**Found:** `custom_providers[0]` = `vLLM`, `base_url http://192.168.1.62:8080/v1`, `model
Qwen/Qwen3.8-27B-FP8`, `models` = the 8 canonical public ids (chat + 7 `gen/*`) matching the facade's
`/v1/models`, `context_length 262144`, `extra_body.chat_template_kwargs.enable_thinking=false` — already
canonical. `auxiliary.vision` pointed at **provider `nous` / `xiaomi/mimo-v2.6-pro`** (not the facade).

**Changed:** `model_forge_activate`'s registry has **no** entry whose id equals the facade's canonical
`Qwen/Qwen3.8-27B-FP8` (its chat ids are `qwen3.8-flash-next` etc.; the alias table maps *into*
`Qwen/Qwen3.8-27B-FP8` but no registry id *is* it), so `model_forge_activate(model_id=…)` cannot target
this facade. Used the documented operator path — the tool's **own in-process helpers**
`write_custom_provider_entry()` + `write_auxiliary_vision()` (the identical config surfaces the tool's
`facade=true` branch writes, through the same `hermes_cli.config` load/save; no dotted-index CLI writes):
- `custom_providers["vLLM"]` refreshed (merge-by-name preserved `api_key`, `extra_body`, `context_length`,
  and all `models`; added `api_mode: chat_completions`).
- `auxiliary.vision`: `model Qwen/Qwen3.8-27B-FP8`, `base_url http://192.168.1.62:8080/v1` (provider left
  `nous`; with a lane `base_url` set, `resolve_vision_provider_client()` takes the explicit endpoint
  override — `agent/auxiliary_client.py` "direct endpoint overrides beat provider selection").

**Evidence:**
- **(a) chat** — `POST /v1/chat/completions` `{"model":"Qwen/Qwen3.8-27B-FP8","messages":[{"role":"user","content":"…"}],"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}` → **200**, content `"CONSUMER1-CHAT-OK"`, `system_fingerprint vllm-0.24.0-tp2-c741c66b`.
- **(b) vision** — same route, `content:[{"type":"text",…},{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}]` → **200**, *"…the image is a solid red square. The dominant color is red."*
- **landing** — `vllm-server` log: `172.18.0.1:33978 - "POST /v1/chat/completions HTTP/1.1" 200 OK` (facade → vLLM).

**Verdict: PASS.**

### 8.2 Image-gen plugin (this host)

**Found:** `image_gen.provider = nous` — **not a registered image provider** (registered: `weightsbroker`,
`model-forge`, plus bundled `deepinfra/fal/krea/meta-ai/openai/openai-codex/openrouter/xai`). With `nous`
unregistered and more than one available provider, `get_active_provider()` returned **None** →
`image_generate` was non-functional. The `model-forge-image` plugin is present (provider name `model-forge`)
but had **no settings** configured.

**Changed:** `image_gen.provider = model-forge`; `plugins.entries.model-forge-image.settings.base_url =
http://192.168.1.62:8080/v1`; `…settings.default_model = gen/qwen-image-2.1` (canonical gen id).

**Evidence:** exercised the plugin's own path —
`ModelForgeImageGenProvider().generate("a single small red apple on a plain white background, centered",
aspect_ratio="square")` → `success:true` in **214.5 s**, `{"image":"http://192.168.1.62:8080/assets/9cca8dcd0dd2/qwen_image_21_00014_.png","model":"gen/qwen-image-2.1","provider":"model-forge"}`.
Asset fetch: **HTTP 200, `image/png`, 1,075,042 bytes, PNG 1024×1024 RGBA**. The full gen dance ran
(evict `vllm-server` → `weights-broker-comfyui-1` up → job → restore); `docker ps` after: `vllm-server Up (healthy)`.

**Verdict: PASS.**

### 8.3 House-scan vision cron (this host)

**Found:** the live house-scan cron is job `5bb6eba5d45e` *"New Home Alert — Baldwin & Mobile (hourly)"* —
`schedule 0 * * * *`, **`no_agent: true`**, script `run_house_scan_alert.sh` → `run_house_scan.py`,
with **`model: null`, `provider: null`, `base_url: null`**. The script's only network targets are
`https://www.sweethomealabama.homes/api-site/search/realTimeListings`, the Estately JSON-LD fallback,
`zillow.com` link building, and the listing-photo downloads — it makes **no LLM/vision call** and never
contacts `192.168.1.62`. No cron in either profile, and no historical job output, contains a house/vision
model call (the only vision-using historical crons were `gym-book-pipeline` and the X-audit *text* labeler,
both unrelated).

**Changed:** none — there is no model/base_url config to refresh and **no vision call shape to replay**.

**Verdict: N/A** — the consumer described does not exist. The house-scan job is a deterministic
scraper → filter → PDF pipeline, not a model consumer; nothing to flip or verify. (Honest finding, not a
fabricated green.) The only vision consumer on this host is the Hermes `auxiliary.vision` lane, already
flipped and verified in §8.1.

### 8.4 OpenViking (.63 = 192.168.1.63)

**Found:** `/root/.openviking/ov.conf` → `vlm = {"provider":"openai","model":"openbmb/MiniCPM-V-4.6",
"api_base":"http://192.168.1.62:8080/v1"}` (`openbmb/MiniCPM-V-4.6` is a **facade alias** → `Qwen/Qwen3.8-27B-FP8`).
`openviking.service` active; its VLM backend (`OpenAIVLM`) posts OpenAI `chat/completions` with
`image_url` data-URL content parts.

**Changed:** `ov.conf` `vlm.model` → `Qwen/Qwen3.8-27B-FP8` (canonical); `api_base` and `api_key` untouched
(JSON edited in place, validated). Backup: `/root/.openviking/ov.conf.bak-model-forge-flip-20261120-043553`.
`systemctl restart openviking` → active, root `HTTP 302`.

**Evidence:** from **.63**, `POST http://192.168.1.62:8080/v1/chat/completions`
`{"model":"openbmb/MiniCPM-V-4.6","messages":[{"role":"user","content":[{"type":"text",…},
{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}]}]}` → **200**, resolved
`model:"Qwen/Qwen3.8-27B-FP8"`, red description. Post-refresh, the **same call with the canonical id** →
**200**. `vllm-server` log shows the forwarded POSTs.

**Verdict: PASS.**

### 8.5 Cleanup / no-ops

- No git commit. `mf-facade`, `vllm-server`, `weights-broker-comfyui-1`, `gluetun` were never
  stopped/restarted/removed by this phase (the facade's *own* gen dance stopped+started `vllm-server` and
  `weights-broker-comfyui-1` as designed; both left healthy afterward). No `docker rm`/prune.
- Throwaway scratch (probe script, tiny test PNGs, fetched asset) removed from `cache/scratch`.

---

## 9. Lane parity — image lanes (2026-11-20, live on 192.168.1.62)

**Model:** xiaomi/mimo-v2.6-pro
**Scope:** the three remaining IMAGE gen lanes, one real end-to-end run each through
`POST http://192.168.1.62:8080/v1/images/generations` (sync, curl `--max-time 600`),
strictly sequential (each ran the full GPU dance: `vllm-server` evicted →
`weights-broker-comfyui-1` up → job → idle release restores vllm).

**Shared input image** (edit/cutout lanes): a 1885-byte 512×512 RGB PNG hand-built on
.62 with stdlib `zlib`/`struct` (two-tone blue/green), sent as `"image":
"data:image/png;base64,…"` (the facade maps `image_url`→`image` and uploads data-URLs
to ComfyUI's input dir per `engines/comfyui.py:_normalize_image_refs`).

### 9.1 Per lane

| lane | request (abridged) | verdict | output evidence |
|---|---|---|---|
| `gen/qwen-image-2.1-edit` | `{"model":"gen/qwen-image-2.1-edit","prompt":"make the sky purple","size":"1024x1024","image":"data:image/png;base64,<1885B PNG>"}` | **PASS** | response `{"created":1795150028,"data":[{"url":"http://192.168.1.62:8080/assets/d8d3685793f6/qwen_image_21_edit_00007_.png"}]}` — today's consumer shape. Asset `HTTP=200 CT=image/png bytes=164687`, `PNG 512×512 RGBA` (colortype 6), magic ok. Wall **164 s**. |
| `gen/qwen-image-2.1-cutout` | `{"model":"gen/qwen-image-2.1-cutout","prompt":"Remove the background, and output a PNG image","size":"1024x1024","image":"data:image/png;base64,<same PNG>"}` | **PASS** | response `{"created":1795150169,"data":[{"url":"http://192.168.1.62:8080/assets/a67e8bafde7c/qwen_image_21_edit_00008_.png"}]}`. Asset `HTTP=200 CT=image/png bytes=176583`, `PNG 512×512 RGBA` (colortype 6), magic ok. **Real alpha matte confirmed:** stdlib PNG decode shows the alpha channel varies (values include 0 and 1 — genuine transparency, not a flat opaque alpha). Wall **112 s**. |
| `gen/moss-image` | `{"model":"gen/moss-image","prompt":"a red circle on a white background","size":"1024x1024"}` | **PASS** | response `{"created":1795150479,"data":[{"url":"http://192.168.1.62:8080/assets/f732b0148567/moss_00298_.png"}]}`. Asset `HTTP=200 CT=image/png bytes=2245088`, `PNG 1024×1024 RGB` (colortype 2), magic ok. Wall **282 s**. |

All three response shapes are the same `{"created", "data":[{"url"}]}` envelope check C
(§7.4) accepted — today's consumers parse these unchanged.

### 9.2 Lane state after

After the third lane's idle release: `weights-broker-comfyui-1` **Exited(143)**,
`vllm-server` **Up, healthy**; `gluetun` untouched (Up 7 days); `weights-broker` still
stopped by design. Trivial chat through the facade:
`POST /v1/chat/completions {"model":"Qwen/Qwen3.8-27B-FP8",…,"max_tokens":8}` →
**200**, real completion, `model:"Qwen/Qwen3.8-27B-FP8"`, `finish_reason:"length"` —
restored lane verified.

### 9.3 Triage

None needed — all three lanes PASS; no model-forge bridge issue and no ComfyUI-stack
failure. No service stopped/restarted/removed by hand, no `docker rm`/prune, no git
commit, no secrets printed.

---

## 10. Lane parity — video lanes (2026-11-20, live on 192.168.1.62)

**Model:** xiaomi/mimo-v2.6-pro
**Scope:** the three h3 VIDEO lanes (`gen/h3-t2v`, `gen/h3-i2v`, `gen/h3-r2v`) via the
async job API `POST /v1/videos` → `GET /v1/videos/{id}` → `/content`, one lane at a time
(no concurrent gens). **Result: FAIL — `gen/h3-t2v` ran twice and failed twice, both
failures inside ComfyUI execution (not the bridge); `gen/h3-i2v` and `gen/h3-r2v` NOT
RUN — blocked on the h3 stack diagnosis** (same stack, same expected fate; per the ops
steer, one honest stack-level finding covers all three lanes instead of burning two more
30-min cycles).

### 10.1 API-shape parity (verified before/around the runs)

Today's shapes were first matched against the old broker source
(`/root/docker/weights-broker/broker/main.py:1682-1712,1753-1770`):

| surface | broker (today) | facade (observed) | match |
|---|---|---|---|
| `POST /v1/videos` | `{"id", "status":"queued", "model": <raw_id minus "gen/">}` (`main.py:1682-1696`) | `{"id": "9404b4a3e212", "status": "queued", "model": "h3-t2v"}` | ✅ (submit echo strips `gen/` exactly as the broker's `removeprefix("gen/")`) |
| `GET /v1/videos/{id}` | `in_progress` → `{"id","status":"completed","data":[{"url"}]}` / `failed`+`error` | `{"id","status":"in_progress"}` while running; `failed` + `error` string on failure | ✅ |
| `GET /jobs/{id}` | `{"id","kind","model","status","error","created","updated","params"}` (+`assets` on output); `params` decoded to object (`main.py:112`); row vocabulary `queued/running/done/failed` (`main.py:86,1237,1248`) | identical keys/values: `{"id":"36b49df203bb","kind":"gen","model":"h3-t2v","status":"running","error":null,"created":…,"updated":…,"params":{"prompt":…,"seconds":5.0}}` | ✅ |

Request bodies were the consumer shapes: `{"model":"gen/h3-t2v","prompt":"a red ball
bouncing on a wooden floor","seconds":5}` — **no `image` field** (t2v contract).

### 10.2 Per lane

| lane | verdict | evidence | wall time |
|---|---|---|---|
| `gen/h3-t2v` attempt 1 (job `9404b4a3e212`) | **FAIL** | poll1 at +90 s: `{"id","status":"failed","error":"comfyui execution error for 'h3-t2v': […execution_error node_id '14' node_type 'SamplerCustomAdvanced' … exception_message 'Allocation on device…'"}`. ComfyUI log: `torch.OutOfMemoryError: Allocation on device` inside `comfy_kitchen` `int8_linear` (`torch.cat(scaled_parts)`); memory summary device 0 **Allocated 22542 MiB / peak 22789 MiB, GPU reserved 23776 MiB of 24124 MB**; `[ERROR] Got an OOM, unloading all loaded models`; `Prompt executed in 42.80 seconds`. Boot log line: `Set vram state to: NORMAL_VRAM` | 90 s (job terminal at +57 s) |
| `gen/h3-t2v` attempt 2 (job `36b49df203bb`) — ONE retry after the fix below | **FAIL — stalled** | submit 05:10:47Z → `{"id":"36b49df203bb","status":"queued","model":"h3-t2v"}`; `Set vram state to: NO_VRAM` this boot; model loads with offload (`loaded partially … 19996.14 MB offloaded`, `Using RAM pressure cache`; h3 = **MiniMaxH3** — `minimax_h3_fl2va_pruned_int8_convrot.safetensors`, Qwen3VL-32B `nvfp4_awq` TE, `minimax_h3_video/audio` VAEs). GPU samples: 05:12–05:14Z GPU1 8190 MiB @ **100%** (load/TE phase), then 05:38Z and 05:52Z **GPU0 1 MiB 0% / GPU1 468 MiB 0%, container CPU 1.2%, no new ComfyUI log lines for minutes**; `GET /v1/videos/{id}` still `{"id","status":"in_progress"}` at +2430 s (27 polls). Expected ~16 min; observed **27–39 min with zero GPU/CPU activity = stalled execution**, declared FAIL per ops steer | ≥39 min (job row still `running`; see §10.5) |
| `gen/h3-i2v` | **NOT RUN — blocked on the h3 stack diagnosis** | same ComfyUI stack / same H3 workflow family as t2v; skipped to avoid burning 2 more 30-min cycles after the t2v stall | — |
| `gen/h3-r2v` | **NOT RUN — blocked on the h3 stack diagnosis** | same as i2v | — |

### 10.3 The fix applied before attempt 2 (stack config, not code)

Diagnosis of the attempt-1 OOM found the **VRAM-mode drift**, not a bridge fault:

- Launch chain: supervisor → `/opt/ai-dock/bin/supervisor-comfyui.sh` →
  `python main.py --disable-auto-launch $(cat /etc/comfyui_args.conf)`.
- `/etc/comfyui_args.conf` is a **`:ro` bind mount of host `/data/broker/comfy_args.conf`**
  (`RW=false` per `docker inspect`); the ai-dock preflight rewrite
  (`preflight.d/10-default.sh`: `printf "%s" "${COMFYUI_ARGS}" > /etc/comfyui_args.conf`)
  is **inert** against the ro mount, so the host file is authoritative across boots.
- Boot history (`docker logs -t`): `Set vram state to: NO_VRAM` on every boot
  Aug 25 → Sep 22 20:02Z; **`NORMAL_VRAM` from Sep 22 21:20:57Z on**. The host file was
  emptied **Sep 22 21:20:51** (now 0 bytes; a sibling `/data/broker/comfy_args.conf.novram.bak`,
  9 bytes, same minute, preserves the historical `--novram` content). Who emptied it is
  unidentified (no broker/facade code writes it — grep clean).
- H3 on 24 GB requires `--novram` (registry note, `registry.py:317`: *"1024x576 @ 24 fps
  fits 24 GB with --novram"*); the image lanes were unaffected because they fit in
  NORMAL_VRAM — the drift was latent until the first h3 run since August.
- **Fix (deployed host config only, no repo diff):** backed up the empty file to
  `/data/broker/comfy_args.conf.bak-pre-novram-restore-20261120`, then
  `printf "%s" "--novram" > /data/broker/comfy_args.conf`. Verified `Set vram state to:
  NO_VRAM` on attempt 2's boot. Nothing else changed: no service stopped/restarted by
  hand, no `docker rm`/prune, `gluetun` untouched, no git commit.

### 10.4 Triage — ComfyUI stack (H3 on current ComfyUI master), NOT a bridge regression

**Bridge hypothesis (VRAM fight / release-at-submit) ruled out with evidence.** Docker
events for job `9404b4a3e212` (attempt 1) reconstruct the claim/evict/release timeline:
`04:58:33Z vllm-server kill → die 04:58:34` → `weights-broker-comfyui-1 start 04:58:35`
→ (ComfyUI executes 04:58:46–04:59:29) → `comfy die 04:59:30` → `vllm-server start
04:59:34` → `health: healthy 05:00:55`. **vLLM was down for the job's entire life**; the
restore landed only after the job went terminal. 5 × 20 s live samples during attempt 2
confirmed the same discipline (`vllm-server` absent throughout; GPU0 1 MiB idle, GPU1
8.2 GB @ 100% during the active phase). This matches the code path
(`facade.py:_run_gen_job`: admit → claim → run → **release in `finally`**; the async
`/v1/videos` submit holds the claim for the whole background run). Bridge lifecycle
behaviour is correct in both attempts.

**Stack-side facts.** The failure is inside ComfyUI execution in both modes: with the
drifted NORMAL_VRAM it is a hard OOM at `SamplerCustomAdvanced` (22.5 GB allocated on
device — the known "H3 without `--novram`" failure), and with the restored known-good
`--novram` it becomes a **silent stall** (idle GPU/CPU, no error, no output). The H3
workflow is MiniMaxH3 with `comfy_kitchen` quantized ops (int8/nvfp4) — an upstream
ComfyUI-master stack. Current checkout `master @ bdb0816a292d`; the venv runs
`torch 2.7.1+cu128` and ComfyUI logs `Unsupported Pytorch detected. DynamicVRAM support
requires Pytorch version 2.8 or later … Falling back to legacy ModelPatcher` — i.e. the
tree and its torch are mid-transition upstream. Templates are byte-identical to the old
broker's (md5: `h3_t2v 5ad145407d3f1097289d587568a5e94c`,
`h3_i2v 7b3f9afb40ae893cc93bd726dbfbcc3d`, `h3_r2v b40f2a06e4ba43c5345138c77e3d5cbe`
vs `/root/docker/weights-broker/workflows/`) and the submitted params match the broker's
stored shape (`{"prompt","seconds":5.0}`). Honest caveat: this boot's ai-dock self-update
was **blocked** (`error: Your local changes to the following files would be overwritten
by merge`; `update-comfyui.sh: line 34: syntax error`), so the tree predates today — the
regression point (ComfyUI-tree update vs the Sep 22 args drift) cannot be narrowed
further without running H3 against a pinned older ComfyUI. The last successful h3 outputs
on record are `weightsbroker_h3_00001–00009_.mp4` (Aug 5–18, previous container
incarnation); **no h3 run is recorded against the current tree**, so the *lane's*
last-known-good state predates both suspects.

### 10.5 Lane state after / blockers

- At §10 write time: `weights-broker-comfyui-1` **Up (holding the stalled job)**,
  `vllm-server` **Exited** (evicted by the claim), `gluetun` untouched (Up 7 days),
  `weights-broker` stopped by design. Stalled job `36b49df203bb` holds the gen claim;
  the facade's runner history budget is `history_timeout_s=3600`
  (`engines/comfyui.py`), so the row fails and the claim releases by **~06:11Z**, after
  which the normal restore runs. **Verified after that point (06:13Z):** the runner's
  history poll timed out at 3600 s — job `36b49df203bb` →
  `{"id":"36b49df203bb","status":"failed","error":"history poll timed out after 3600s for c4be4e15-…"}`,
  the claim released on schedule, `weights-broker-comfyui-1` **Exited(143)**,
  `vllm-server` **Up, healthy**, `gluetun` untouched (Up 7 days), `weights-broker`
  stopped by design. Trivial chat through the facade:
  `POST /v1/chat/completions {"model":"Qwen/Qwen3.8-27B-FP8",…,"max_tokens":8}` →
  **200**, real completion `"OK"`, `model:"Qwen/Qwen3.8-27B-FP8"` — restored lane
  verified. (Note the stalled job did pin the gen lane for the full 3600 s window, as
  flagged in blocker 3 below.)
- **Blockers for the three h3 video lanes:** (1) MiniMaxH3 must be made to run again on
  the current ComfyUI master + torch 2.7.1 (pin an older tree, upgrade torch to ≥2.8,
  or fix the upstream nodes) — this is a **ComfyUI-stack** work item, not model-forge;
  (2) guard `/data/broker/comfy_args.conf` (whoever emptied it on Sep 22 is still
  unidentified; the preflight `${COMFYUI_ARGS}` rewrite being inert means a host-side
  write is the only mutator);
  (3) the facade's async-video default has **no caller-supplied budget** —
  `handle_video_submit` never passes `timeout_s`, so a stalled ComfyUI job pins the gen
  lane for the runner's full 3600 s history window (worth a follow-up knob, cosmetic
  here).
- No service stopped/restarted/removed by hand during this phase, no `docker rm`/prune,
  no git commit, no secrets printed.

---

## 11. h3 restoration — tree pin + lane parity (2026-11-20, live on 192.168.1.62)

**Model:** xiaomi/mimo-v2.6-pro
**Scope:** restore the three h3 VIDEO lanes by pinning the ComfyUI tree to the
known-good revision baked in `weights-broker-comfyui:mfflashfix`, persist the pin,
then run `gen/h3-t2v`, `gen/h3-i2v`, `gen/h3-r2v` end-to-end through the facade.
**Result: ✅ 3/3 lanes PASS with real mp4 bytes** (plus one bridge-side collection fix,
below). Image lanes regression-checked green before the video runs.

### 11.1 The baked sha (step 1)

`docker run --rm --entrypoint git weights-broker-comfyui:mfflashfix -C /opt/ComfyUI log -1`:

```
63e5bf594339f676b4b15e072529873e3448409e_2026-11-06 01:32:17 -0500_
Make the partner node team code owners of comfy_api_nodes (#17575)
```

Baked-tree checks (fresh container, no update script runs): `comfy_extras/nodes_minimax_h3.py`
present (37078 B); `comfy_kitchen/flash_attention.py:186` carries the mf-fix marker
`if True:  # mf-fix: convrot KV strides trip the cuda kernel; use torch fallback`;
baked `comfy/text_encoders/llama.py` md5 `4020ec32017003f32b121ab69d5fb339` (the
`contiguous()` sed fix applied — 4 occurrences). Baked git status: ` M llama.py` only
(plus volume-shading noise).

### 11.2 Live container state (step 2)

`weights-broker-comfyui-1` was `Exited(143)` (image `ltx25stable`; the compose file
already said `mfflashfix` — image drift between compose and the running container).
Started it for inspection: HEAD `master @ bdb0816a292d` (2026-11-07), ` M
comfy/text_encoders/llama.py`, no MERGE_HEAD (the section-10 merge mess had already
aborted), and the baked sha object **was already local** (`git cat-file -e` → yes) —
no network fetch needed. `bash -n /opt/ai-dock/bin/update-comfyui.sh` → **line 34
syntax error**: the mf-fix re-apply `sed` is an unquoted expression with parens
(`sed -i s/flash_attention_decode_gqa(xq, …`) — bash parses the `(`. The live
site-packages already carried the flash marker (count 1).

### 11.3 Pin + persist (step 3)

Inside the live container: `git merge --abort` (no-op, no merge to abort),
`git checkout -f 63e5bf59…` (local objects, detached HEAD), re-applied the llama.py
sed with proper quoting → md5 `4020ec32…` == baked. Fixed the update script by
quoting the sed expression (`bash -n` → clean; backup
`/opt/ai-dock/bin/update-comfyui.sh.bak-pre-syntax-fix` in the container layer).
`docker commit weights-broker-comfyui-1 weights-broker-comfyui:pinned-h3`.

Compose diff (`/root/docker/weights-broker/docker-compose.yaml`; backup
`docker-compose.yaml.bak-pre-h3pin-20261120`):

```diff
-    image: weights-broker-comfyui:mfflashfix
+    image: weights-broker-comfyui:pinned-h3
-      - COMFYUI_REF=master                     # H3 nodes are master-only (not in tags)
+      - COMFYUI_REF=63e5bf594339f676b4b15e072529873e3448409e                     # H3 nodes are master-only (not in tags)
```

Then `docker compose -f docker-compose.yaml up -d --pull never comfyui`
(service-level; recreate succeeded, other services untouched).

**Pitfall found — the first recreate did NOT hold the pin.** Boot came up at
`bdb0816a` again (reflog: `checkout: moving from 63e5bf594 to master`) even though
`printenv COMFYUI_REF` showed the sha. Root cause: ai-dock's
`/opt/ai-dock/etc/environment.sh` line 12 (`export COMFYUI_REF='master'`) is
**sourced at init** (`init.sh:487`), overriding the compose env, and
`init_write_environment` re-persists that same env — so the file is authoritative
and self-perpetuating; a compose-only `COMFYUI_REF` pin never reaches
`update-comfyui.sh`. Fix: set `COMFYUI_REF='<sha>'` in `environment.sh` too
(container layer), `git checkout -f <sha>` + sed again, re-commit
(`pinned-h3` re-tagged `ff472a944d1d`), compose recreate again. With
`COMFYUI_REF=<sha>` the update script's `git checkout ${ref}` keeps the pin at every
boot (the fixed `sed` is idempotent — the pattern no longer matches after applying).

### 11.4 Boot verification (after the second recreate)

| check | result |
|---|---|
| `git log -1 --format=%H` in `/opt/ComfyUI` | `63e5bf594339f676b4b15e072529873e3448409e` ✅ (pin survived boot) |
| `environment.sh` `COMFYUI_REF` | `'63e5bf59…'` ✅ (self-re-persisted) |
| `/etc/comfyui_args.conf` (`:ro` mount of `/data/broker/comfy_args.conf`) | `--novram` ✅ |
| flash marker `comfy_kitchen/flash_attention.py:186` | present (count 1) ✅ |
| `llama.py` md5 | `4020ec32017003f32b121ab69d5fb339` == baked ✅ |
| torch | `2.7.1+cu128` ✅ (mfflashfix-era stack) |
| `curl 127.0.0.1:18188/` from the host | `HTTP 200` ✅ (loopback-only, probed via ssh) |

### 11.5 Bridge-side collection fix — the real E2E blocker (mf_video_bucket_fix)

Triage of the first post-pin t2v run found the **actual reason no h3 lane ever
completed through the facade**: this ComfyUI tree records `SaveVideo` outputs in the
history entry's **`images`** bucket with `metadata.kind == "video"` (plus
`"animated": [true]`), while the facade's collector
(`engines/comfyui.py:_OUTPUT_BUCKETS["video"] = ("videos","results","gifs")`) never
looks there — `_poll_once` reports `running` forever and the runner rides to its
history budget. The same mismatch retro-explains §10's "stall" (see 11.7).

Facade code cannot change this without a restart (forbidden), so the fix is
ComfyUI-side, in the mf-fix tradition: `/data/broker/comfy/custom_nodes/mf_video_bucket_fix.py`
(**volume**-persisted, pinned tree untouched) patches
`PromptQueue.get_history` at read time to alias video-kind `images` items into a
`videos` bucket. Two load requirements discovered: the tree's custom-node loader
skips modules without `NODE_CLASS_MAPPINGS`/`comfy_entrypoint`
(`nodes.py:2347` — first attempt was `IMPORT FAILED`), and the check is
`is not None`, so an empty `NODE_CLASS_MAPPINGS = {}` suffices. Second boot:
`0.0 seconds: mf_video_bucket_fix.py` — imported clean, and jobs then collected.

### 11.6 Lane parity — per lane (one at a time; vllm-server Up+healthy before each submit)

Input image for i2v/r2v: stdlib-built 512×512 RGB PNG (1500 B, two-tone
blue/green), sent as `data:image/png;base64,…` per §9.

| lane | job | timeline (UTC) | verdict + evidence |
|---|---|---|---|
| `gen/h3-t2v` | `3435a622bcc0` | submit 07:33:47 → completed 07:49:30 (wall **15m43s**) | **PASS** — `{"id","status":"completed","data":[{"url":"http://192.168.1.62:8080/assets/3435a622bcc0/weightsbroker_h3_00012_.mp4"}]}`. Asset `HTTP 200 CT=video/mp4 1720304 bytes`, `ftyp` at offset 4 (`isom`); `GET /v1/videos/{id}/content` byte-identical (sha256 `ddc4f3b9dddf5ba3…`); job row `status:"done"` + `assets` list. |
| `gen/h3-i2v` | `c078c70d568c` | submit 07:51:25 → completed 08:08:30 (wall **17m05s**) | **PASS** — `data[0].url …/assets/c078c70d568c/weightsbroker_h3_00013_.mp4`. Asset `HTTP 200 CT=video/mp4 1772983 bytes`, `ftyp@4` ok; `/content` byte-identical. |
| `gen/h3-r2v` | `56e85dcdf30a` | submit 08:11:03 → completed 08:41:54 (wall **30m51s**) | **PASS** — `data[0].url …/assets/56e85dcdf30a/weightsbroker_h3_00014_.mp4`. Asset `HTTP 200 CT=video/mp4 2764986 bytes`, `ftyp@4` ok; `/content` byte-identical. (Heaviest lane: GPU1 peaked 13.3 GB, ~31 min — GPU stayed 100% throughout; the 25-min stall rule targets 0%-GPU no-progress, which never occurred.) |

Requests were the consumer shapes: `{"model":"gen/h3-*","prompt":"a red ball bouncing
on a wooden floor","seconds":5,"_timeout":1800}` (+ `"image": "data:image/png;base64,…"`
for i2v/r2v); submit echo `{"id","status":"queued","model":"h3-*"}` (strips `gen/`).
GPU dance held discipline per lane: `vllm-server` evicted for the whole job, restored
`healthy` after each idle release (`weights-broker-comfyui-1` ended `Exited(143)`).

**Superseded/failed attempts (recorded):** `24579b3ffd14` (t2v, 06:44:35Z submit) — real ComfyUI execution `Prompt executed in 00:14:45`, mp4
`weightsbroker_h3_00011_.mp4` written 06:59, but bridge-invisible pre-shim; declared
FAIL at the 25-min stall rule and the gen container stopped (job → honest
`failed: ComfyUI went unreachable while polling`). `f893f39576a5` (t2v) — killed
after 6 min when the first shim was `IMPORT FAILED` (missing `NODE_CLASS_MAPPINGS`).
`e304baca8170` — harness error: my submit piped the echo through `sed` and re-used
the i2v body (duplicate `h3-i2v`); killed at ~2 min (`failed: unreachable`), real
r2v body built and `model` field verified before resubmit.

### 11.7 Triage findings (cross-cutting)

1. **§10's "stall" was a misdiagnosis of a completed job.** `weightsbroker_h3_00010_.mp4`
   (mtime 05:25Z, 1720304 B — byte-size-identical to today's deterministic t2v output)
   proves attempt 2 finished ~15 min after its 05:10:47Z submit; the 05:38/05:52Z
   0%-GPU samples were post-completion idle. The job looked stalled only because the
   bucket mismatch hid the output until the 3600 s history timeout.
2. **Real E2E blocker = history-bucket drift** (tree `images`-bucket video vs facade
   `videos/results/gifs`) — fixed with `mf_video_bucket_fix.py` (11.5). Suggested
   follow-up: teach the facade's `_OUTPUT_BUCKETS`/`_collect_outputs` about
   `metadata.kind == "video"` (needs a facade restart — deferred, forbidden here).
3. **ai-dock boot env overrides compose env** (`environment.sh` sourced at init and
   re-persisted) — any `COMFYUI_REF` pin must live there, not only in compose (11.3).
4. §10.5 blocker 3 confirmed again: `handle_video_submit` ignores `_timeout` —
   doomed jobs pin the gen lane for the full 3600 s history window (worth the
   caller-supplied budget knob).
5. `/data/broker/comfy_args.conf` = `--novram` intact all day (the Sep 22 emptier is
   still unidentified).

### 11.8 State after / constraints held

`mf-facade` never stopped/restarted; `vllm-server` and `gluetun` never touched
manually (vllm only by the facade's own dance; left `Up, healthy`); `weights-broker`
stopped by design; the only container recreated was `weights-broker-comfyui-1`, via
`docker compose … up -d --pull never comfyui` (two recovers total: initial pin +
env.sh fix); no `docker rm`/prune (compose recreate manages its own container);
no `git reset --hard` in `~/work/weights-broker` (container `/opt/ComfyUI` git ops
only); no secrets printed; **no git commit**. Artifacts left in place:
`pinned-h3` image (`ff472a944d1d`), compose backup `docker-compose.yaml.bak-pre-h3pin-20261120`,
in-container backup `update-comfyui.sh.bak-pre-syntax-fix`,
`/data/broker/comfy/custom_nodes/mf_video_bucket_fix.py` (+ scratch copy),
`/data/broker/comfy/output/video/weightsbroker_h3_00011–00014_.mp4`.

---

## 12. Shim retirement — fixed collector proven live (2026-11-20, live on 192.168.1.62)

**Model:** xiaomi/mimo-v2.6-pro
**Scope:** retire the ComfyUI-side `mf_video_bucket_fix` shim (§11.5) and prove the
redeployed facade's fixed output collector (`engines/comfyui.py`: video collection now
scans the history `images` bucket for `metadata.kind == "video"` items) collects video
outputs without it. **Result: ✅ PASS — one real `gen/h3-t2v` completed with exactly one
real mp4 asset, shim retired, no rollback.**

- **Shim retired (kept as the rollback):** `/data/broker/comfy/custom_nodes/mf_video_bucket_fix.py`
  → `mf_video_bucket_fix.py.bak-retired-20261120` (rename 14:26:05Z; 2113 B, md5
  `fedd0a666e308f405fa5d88ad72094f9`, contents untouched). Loader takes effect at the
  next ComfyUI start — which was the h3 job's own boot (the dance starts
  `weights-broker-comfyui-1`), so the run below was the first shim-free boot.
- **h3 job `befe8932bd7f`** (`gen/h3-t2v`, `{"prompt":"a red ball bouncing on a wooden
  floor","seconds":5,"_timeout":1800}`): submit 14:26:27Z → `completed` 14:42:15Z, wall
  **15m48s** (ComfyUI `Prompt executed in 00:14:35`, sampling 20/20 steps @ ~40 s/it).
  `GET /v1/videos/{id}` → `{"status":"completed","data":[{"url":"http://192.168.1.62:8080/assets/befe8932bd7f/weightsbroker_h3_00015_.mp4"}]}`
  — exactly **one** data item. Asset `HTTP 200`, `Content-Type: video/mp4`,
  `Content-Length: 1720304`, `ftyp` magic at offset 4; `GET /v1/videos/{id}/content`
  byte-identical to the asset (sha256 `ddc4f3b9dddf5ba38c651f7521b13758b9a95c38def1a46fbbd68a4b2e307c67`).
  Job row `status:"done"` with `assets` list length **1** (no duplicated assets); exactly
  one new output file `…/output/video/weightsbroker_h3_00015_.mp4` (1720304 B). The
  collector found the video-kind `images`-bucket item with no shim — the §11.5 fix is
  effective.
- **Boot clean (shim not loaded):** the job's boot (container start 14:26:29Z) imported
  only `websocket_image_save.py`; the boot's own `Import times` block in
  `/var/log/supervisor/comfyui.log` lists that single module and there are **no
  IMPORT FAILED** lines in the current boot (the file's only `(IMPORT FAILED)` line and
  the `Skip … lack of NODE_CLASS_MAPPINGS` warning are old shim-era boots, lines
  34605/34609). **Pitfall:** raw `docker logs weights-broker-comfyui-1` is misleading —
  the ai-dock relay tails the persistent `/var/log/supervisor/*.log` and **replays** old
  content into the container log at boot (blocks timestamped 14:26:34Z listing
  `mf_video_bucket_fix.py` are stale pre-rename lines, flushed under new timestamps);
  additionally the custom-node loader skips any entry whose extension isn't `.py`
  (`nodes.py` `init_external_custom_nodes`), so the `.bak` can never be imported. Check
  the last `Import times` block of the supervisor log, not a raw log grep.
- **End state (verified 14:44Z):** `vllm-server` **Up, healthy** (restored by the idle
  release), `weights-broker-comfyui-1` **Exited(143)**, `gluetun` untouched (Up 7 days,
  healthy), `weights-broker` stopped by design, `GET /health` → `{"llm":true,"comfy_args":"novram-ok","ok":true}`,
  trivial chat `POST /v1/chat/completions` → **200** with a real completion
  (`Qwen/Qwen3.8-27B-FP8`). Constraints held: no `docker rm`/prune, no manual
  stop/restart of mf-facade/vllm-server/gluetun, no secrets printed, **no git commit**.
  Rollback if ever needed: rename `mf_video_bucket_fix.py.bak-retired-20261120` back to
  `mf_video_bucket_fix.py` (not needed — the collector fix works).
