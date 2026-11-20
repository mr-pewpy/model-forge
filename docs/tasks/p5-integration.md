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
