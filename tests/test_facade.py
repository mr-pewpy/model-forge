"""Facade route tests (model_forge/facade.py) — fakes, no sockets, no network.

Every external effect is a constructor seam, so these tests drive the real
route methods directly: a fake upstream captures the exact bytes and headers the
facade forwarded, fake probes exercise the fail-soft /health path, and a refusing
gate exercises the fail-closed 503. No port is ever bound.
"""

from __future__ import annotations

import io
import json

from model_forge.aliases import DEFAULT_ALIASES
from model_forge.facade import (
    CHAT_MODEL_ID,
    GEN_MODELS,
    MODELS_ADVERTISED,
    FacadeConfig,
    FacadeService,
    comfy_args_for,
    flip_comfy_args,
    make_handler,
)

PUBLIC_IDS = [CHAT_MODEL_ID, *GEN_MODELS]


# --- fakes ------------------------------------------------------------------

class FakeUpstream:
    """Records (raw_bytes, headers) per call; returns a fixed status/body."""

    def __init__(self, status=200, body=b'{"id":"cmpl-1","choices":[]}'):
        self.status = status
        self.body = body
        self.calls: list[tuple[bytes, dict]] = []

    def __call__(self, raw: bytes, headers: dict):
        self.calls.append((raw, headers))
        return self.status, self.body


def _service(upstream=None, gate=None, probes=None, **cfg):
    config = FacadeConfig(llm_canonical_id=CHAT_MODEL_ID,
                          llm_served_id="qwen38-27b-fp8-served", **cfg)
    return FacadeService(config, upstream=upstream, gate=gate, probes=probes)


# --- /v1/models listing -----------------------------------------------------

def test_catalog_lists_eight_public_ids():
    svc = _service(upstream=FakeUpstream())
    payload = svc.catalog()
    assert payload["object"] == "list"
    ids = {e["id"] for e in payload["data"]}
    for mid in PUBLIC_IDS:
        assert mid in ids, mid
    assert len(PUBLIC_IDS) == 8


def test_catalog_entry_shape_and_enrichment():
    svc = _service(upstream=FakeUpstream())
    by_id = {e["id"]: e for e in svc.catalog()["data"]}

    chat = by_id[CHAT_MODEL_ID]
    assert chat["object"] == "model"
    assert chat["owned_by"] == "vllm"
    assert chat["label"] == "Qwen3.8-27B-FP8"
    assert chat["engine"] == "vllm"
    assert chat["quantization"] == "fp8-full"
    assert chat["context_length"] == 262144
    assert chat["max_model_len"] == 262144  # vLLM-native mirror
    assert chat["capabilities"]["vision"] is True


def test_catalog_gen_entries_carry_output_kind():
    svc = _service(upstream=FakeUpstream())
    by_id = {e["id"]: e for e in svc.catalog()["data"]}
    for mid, kind in GEN_MODELS.items():
        assert by_id[mid]["output_kind"] == kind, mid
        assert by_id[mid]["owned_by"] == "comfyui"
        assert by_id[mid]["object"] == "model"


def test_catalog_every_alias_key_resolves_to_a_listed_target():
    svc = _service(upstream=FakeUpstream())
    payload = svc.catalog()
    by_id = {e["id"]: e for e in payload["data"]}
    listed = set(by_id)
    for alias, target in DEFAULT_ALIASES.items():
        # The alias key is present, points at its canonical id, and that
        # canonical id is itself advertised.
        assert alias in by_id, alias
        assert by_id[alias]["alias_of"] == target
        assert target in listed, target
        # Enrichment is looked up under the canonical id (rewrite-then-enrich).
        if target in MODELS_ADVERTISED:
            assert by_id[alias].get("label") == MODELS_ADVERTISED[target].get("label")


# --- chat passthrough -------------------------------------------------------

def test_chat_alias_canonicalization_and_served_rewrite_and_content_type():
    up = FakeUpstream()
    svc = _service(upstream=up)
    raw = json.dumps({"model": "research/qwen38-uncensored",
                      "messages": [{"role": "user", "content": "hi"}]}).encode()

    status, body = svc.handle_chat(raw)

    assert status == 200
    assert body == up.body  # upstream body returned verbatim
    assert len(up.calls) == 1
    sent_raw, headers = up.calls[0]
    sent = json.loads(sent_raw)
    # Alias canonicalized to the HF id, then rewritten to the served id.
    assert sent["model"] == "qwen38-27b-fp8-served"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    # The ported fix: Content-Type is set explicitly on the passthrough.
    assert headers["Content-Type"] == "application/json"


def test_chat_returns_upstream_status_and_body_verbatim():
    up = FakeUpstream(status=418, body=b'{"error":{"message":"teapot"}}')
    svc = _service(upstream=up)
    status, body = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())
    assert status == 418
    assert body == b'{"error":{"message":"teapot"}}'


def test_chat_non_chat_model_keeps_id_but_still_forwards():
    up = FakeUpstream()
    svc = _service(upstream=up)
    svc.handle_chat(json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    sent = json.loads(up.calls[0][0])
    assert sent["model"] == "gen/qwen-image-2.1"  # no canonical->served rewrite


def test_chat_invalid_json_is_400_and_touches_no_upstream():
    up = FakeUpstream()
    svc = _service(upstream=up)
    status, body = svc.handle_chat(b"{not json")
    assert status == 400
    assert json.loads(body)["error"]["type"] == "invalid_request_error"
    assert up.calls == []


# --- gate seam --------------------------------------------------------------

def test_gate_refusal_yields_503_with_honest_error():
    up = FakeUpstream()
    seen = {}

    def gate(model_id: str) -> dict:
        seen["model"] = model_id
        return {"ok": False, "reason": "lane busy: swapping weights, retry shortly"}

    svc = _service(upstream=up, gate=gate)
    status, body = svc.handle_chat(json.dumps({"model": "ocr/deepseek-ocr-2"}).encode())

    assert status == 503
    err = json.loads(body)["error"]
    assert err["type"] == "busy"
    assert err["message"] == "lane busy: swapping weights, retry shortly"
    # The gate saw the CANONICAL id, and no upstream call was made.
    assert seen["model"] == CHAT_MODEL_ID
    assert up.calls == []


def test_default_gate_allows():
    up = FakeUpstream()
    svc = _service(upstream=up)  # no gate -> trivial allow
    status, _ = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())
    assert status == 200
    assert len(up.calls) == 1


# --- /health fail-soft ------------------------------------------------------

def test_health_reflects_injected_probes():
    probes = {"llm": lambda: True, "comfy": lambda: True, "mode": lambda: "gen"}
    svc = _service(upstream=FakeUpstream(), probes=probes)
    health = svc.health()
    assert health["llm"] is True
    assert health["comfy"] is True
    assert health["mode"] == "gen"
    assert health["ok"] is True


def test_health_fail_soft_when_probe_raises():
    def boom():
        raise RuntimeError("probe exploded")

    probes = {"llm": boom, "comfy": lambda: True}  # mode intentionally omitted
    svc = _service(upstream=FakeUpstream(), probes=probes)
    health = svc.health()
    assert health["llm"] is False       # raising probe -> fail-soft default
    assert health["comfy"] is True
    assert health["mode"] == "unknown"  # missing probe -> default
    assert health["ok"] is False


# --- wiring (no ports bound) ------------------------------------------------

def test_make_handler_returns_configured_handler_class():
    from http.server import BaseHTTPRequestHandler

    handler = make_handler(_service(upstream=FakeUpstream()))
    assert issubclass(handler, BaseHTTPRequestHandler)
    assert handler.server_version.startswith("model-forge-facade")


# --- gate that raises is fail-closed ----------------------------------------

def test_gate_that_raises_yields_503_not_a_dropped_connection():
    up = FakeUpstream()

    def boom_gate(model_id: str) -> dict:
        raise RuntimeError("orchestrator exploded")

    svc = _service(upstream=up, gate=boom_gate)
    status, body = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())

    assert status == 503  # never propagates, never a silent forward
    err = json.loads(body)["error"]
    assert err["type"] == "busy"
    assert "orchestrator exploded" in err["message"]
    assert up.calls == []  # refused before the upstream


def test_gate_returning_non_dict_is_fail_closed():
    up = FakeUpstream()
    svc = _service(upstream=up, gate=lambda model_id: "not-a-dict")
    status, body = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())
    assert status == 503
    assert json.loads(body)["error"]["type"] == "busy"
    assert up.calls == []


# --- gen-lane alias inheritance (output_kind/kind) --------------------------

def test_catalog_gen_alias_inherits_output_kind_from_gen_models():
    svc = _service(upstream=FakeUpstream())
    by_id = {e["id"]: e for e in svc.catalog()["data"]}

    gen_aliases = {a: t for a, t in DEFAULT_ALIASES.items() if t in GEN_MODELS}
    assert gen_aliases  # sanity: krea2/ideogram-4/krea2-img2img exist
    for alias, target in gen_aliases.items():
        assert by_id[alias]["alias_of"] == target
        assert by_id[alias]["output_kind"] == GEN_MODELS[target], alias
        assert by_id[alias]["kind"] == GEN_MODELS[target], alias


# --- config precedence (from_env) -------------------------------------------

def test_from_env_reads_env_and_explicit_key_always_wins(monkeypatch):
    monkeypatch.setenv("BROKER_VLLM_API_KEY", "env-key")
    # The env read is the field default, so it applies with no explicit value.
    assert FacadeConfig().llm_api_key == "env-key"
    assert FacadeConfig.from_env().llm_api_key == "env-key"
    # An explicit value always wins — from_env is a pure pass-through.
    assert FacadeConfig(llm_api_key="explicit").llm_api_key == "explicit"
    assert FacadeConfig.from_env(llm_api_key="explicit").llm_api_key == "explicit"


# --- do_POST Content-Length handling ----------------------------------------

class _FakeHeaders:
    def __init__(self, values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _FakeWFile:
    def __init__(self):
        self.buf = b""

    def write(self, data):
        self.buf += data


def _drive_post(service, path, headers, body=b""):
    """Run the real do_POST with fakes (no socket); return status/body/headers."""
    handler_cls = make_handler(service)
    handler = handler_cls.__new__(handler_cls)  # bypass create_server/socket
    handler.path = path
    handler.headers = _FakeHeaders(headers)
    handler.rfile = io.BytesIO(body)
    handler.wfile = _FakeWFile()
    captured = {"status": None, "headers": {}}

    def send_response(status, message=None):
        captured["status"] = status

    def send_header(key, value):
        captured["headers"][key] = value

    handler.send_response = send_response
    handler.send_header = send_header
    handler.end_headers = lambda: None

    handler.do_POST()
    return captured["status"], handler.wfile.buf, captured["headers"]


def test_post_missing_content_length_is_411():
    up = FakeUpstream()
    svc = _service(upstream=up)
    status, body, _ = _drive_post(svc, "/v1/chat/completions", {})
    assert status == 411
    assert json.loads(body)["error"]["type"] == "invalid_request_error"
    assert up.calls == []  # never forwarded as a phantom empty body


def test_post_chunked_without_length_is_411():
    up = FakeUpstream()
    svc = _service(upstream=up)
    status, _, _ = _drive_post(svc, "/v1/chat/completions",
                               {"Transfer-Encoding": "chunked"})
    assert status == 411
    assert up.calls == []


def test_post_non_numeric_content_length_is_400():
    up = FakeUpstream()
    svc = _service(upstream=up)
    status, body, _ = _drive_post(svc, "/v1/chat/completions",
                                  {"Content-Length": "abc"})
    assert status == 400
    assert json.loads(body)["error"]["type"] == "invalid_request_error"
    assert up.calls == []


def test_post_valid_content_length_forwards_with_explicit_content_type():
    up = FakeUpstream()
    svc = _service(upstream=up)
    payload = json.dumps({"model": CHAT_MODEL_ID}).encode()
    status, body, sent = _drive_post(
        svc, "/v1/chat/completions",
        {"Content-Length": str(len(payload))}, body=payload)
    assert status == 200
    assert body == up.body
    assert sent["Content-Type"] == "application/json"
    assert len(up.calls) == 1


# ===========================================================================
# P5c-C1b — generation routes + job API
#
# Same discipline as above: every effect is a seam. A fake orchestrator records
# claim/release + waiter/admission order, a fake runner records workflow ids +
# free() calls, and the real in-memory JobStore backs the routes (it is stdlib
# and carries no external effect). ``dispatch`` runs the async video job body
# inline so the lifecycle is deterministic and no thread races the assertions.
# ===========================================================================

import os  # noqa: E402
import pytest  # noqa: E402

from model_forge.facade import FacadeService  # noqa: E402


class _Cfg:
    """Stand-in for a workflow registry entry (``.output_kind`` is all the
    routes read)."""

    def __init__(self, output_kind):
        self.output_kind = output_kind


def _workflows():
    return {
        "qwen-image-2.1": _Cfg("image"),
        "qwen-image-2.1-edit": _Cfg("image"),
        "h3-t2v": _Cfg("video"),
        "h3-i2v": _Cfg("video"),
    }


class FakeOrchestrator:
    """Records claim/release + waiter/admission calls; outcomes configurable."""

    def __init__(self, *, claim_ok=True, claim_reason="", admit_ok=True,
                 admit_reason="", waiter_ok=True, waiter_reason="",
                 waiter_token: str | None = "tok-1", gen_busy=False, log=None):
        self.claim_ok = claim_ok
        self.claim_reason = claim_reason
        self.admit_ok = admit_ok
        self.admit_reason = admit_reason
        self.waiter_ok = waiter_ok
        self.waiter_reason = waiter_reason
        self.waiter_token = waiter_token
        # The gen-busy read the chat path polls (a plain bool here; the real
        # orchestrator derives it from its open scopes).
        self.gen_busy = gen_busy
        self.log = log
        self.claims = []
        self.releases = []
        self.admits = []
        self.waiter_exits = []

    def _rec(self, event):
        if self.log is not None:
            self.log.append(event)

    def claim(self, scope_id, target_id, request):
        self.claims.append((scope_id, target_id, request))
        self._rec("claim")
        return {"ok": self.claim_ok, "reason": self.claim_reason}

    def release(self, scope_id, *, restore=True):
        self.releases.append((scope_id, restore))
        self._rec("release")
        return {"ok": True}

    def admit_gen(self, model_id=""):
        self.admits.append(model_id)
        self._rec("admit_gen")
        return {"ok": self.admit_ok, "reason": self.admit_reason}

    def waiter_enter(self):
        self._rec("waiter_enter")
        return {"ok": self.waiter_ok, "reason": self.waiter_reason,
                "token": self.waiter_token if self.waiter_ok else None}

    def waiter_exit(self, token):
        self.waiter_exits.append(token)
        self._rec("waiter_exit")

    @property
    def llm_waiters(self):
        return 0


class FakeRunner:
    """Records runner calls; returns a fixed result; counts free() calls."""

    def __init__(self, files=None, ok=True, reason="", log=None,
                 workflows=None):
        self.workflows = _workflows() if workflows is None else workflows
        self.files = list(files or [])
        self.ok = ok
        self.reason = reason
        self.log = log
        self.run_calls = []
        self.run_inputs = []   # kwargs the facade passed (incl. timeout_s)
        self.free_calls = 0

    def run(self, workflow_id, params, *, assets_dir, job_id,
            timeout_s=None):
        self.run_calls.append((workflow_id, params, assets_dir, job_id))
        self.run_inputs.append({"workflow_id": workflow_id, "params": params,
                                "assets_dir": assets_dir, "job_id": job_id,
                                "timeout_s": timeout_s})
        if self.log is not None:
            self.log.append("run")
        if not self.ok:
            return {"ok": False, "output_kind": "", "files": [],
                    "reason": self.reason, "meta": {}}
        return {"ok": True, "output_kind": "", "files": list(self.files),
                "reason": "", "meta": {}}

    def free(self):
        self.free_calls += 1
        if self.log is not None:
            self.log.append("free")
        return {"ok": True}


def _gen_service(tmp_path, *, orchestrator=None, runner=None, dispatch=None,
                 **cfg):
    config = FacadeConfig(assets_dir=str(tmp_path), **cfg)
    return FacadeService(config, orchestrator=orchestrator, runner=runner,
                         dispatch=dispatch)


def _inline(fn):
    """Deterministic dispatch: run the job body inline (no thread races)."""
    fn()


def _write(path, data=b"\x89PNG\r\n"):
    path.write_bytes(data)
    return str(path)


# --- image sync -------------------------------------------------------------

def test_image_sync_happy_path_order_shape(tmp_path):
    log = []
    out = _write(tmp_path / "shot.png")
    orch = FakeOrchestrator(log=log)
    runner = FakeRunner(files=[out], log=log)
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)

    raw = json.dumps({"model": "gen/qwen-image-2.1", "prompt": "a cat"}).encode()
    status, body = svc.handle_image_generations(raw)

    assert status == 200
    payload = json.loads(body)
    assert "created" in payload
    (item,) = payload["data"]
    jid = orch.claims[0][0]
    assert item["url"] == f"http://127.0.0.1:8080/assets/{jid}/shot.png"

    # Ordering contract (roadmap §3 LLM-waiter rule): admit_gen FIRST, then
    # claim -> run -> release, and free after (see facade docstrings). The
    # gen job must be admitted before claim executes any lane-wake side effect.
    assert log[:4] == ["admit_gen", "claim", "run", "release"]
    assert log.index("free") > log.index("run")
    # The claim targeted the configured gen target and the workflow id.
    scope, target, request = orch.claims[0]
    assert target == "comfy"
    assert request.model_id == "qwen-image-2.1"
    assert orch.admits == ["qwen-image-2.1"]


def test_image_sync_passes_alias_resolved_bare_workflow_id(tmp_path):
    out = _write(tmp_path / "a.png")
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[out])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)

    # gen/krea2 -> (alias) gen/qwen-image-2.1 -> (strip gen/) qwen-image-2.1
    raw = json.dumps({"model": "gen/krea2", "prompt": "x"}).encode()
    status, _ = svc.handle_image_generations(raw)

    assert status == 200
    assert runner.run_calls[0][0] == "qwen-image-2.1"  # bare workflow id
    assert orch.admits == ["qwen-image-2.1"]


def test_image_sync_admit_refusal_is_503_with_reason(tmp_path):
    log = []
    orch = FakeOrchestrator(
        admit_ok=False,
        admit_reason="chat waiter owns the lane wake; retry shortly",
        log=log)
    runner = FakeRunner(files=[str(tmp_path / "x.png")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)

    status, body = svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1"}).encode())

    assert status == 503
    err = json.loads(body)["error"]
    assert err["type"] == "busy"
    assert "chat waiter owns the lane wake" in err["message"]
    # Admit FIRST: a refused admission must NOT claim (no lane-wake side
    # effect) and must never reach the runner.
    assert orch.claims == []              # no claim side effect
    assert orch.releases == []            # nothing claimed -> nothing released
    assert runner.run_calls == []         # never reached the runner
    assert log == ["admit_gen"]           # admission checked before any claim


def test_image_sync_claim_refusal_is_503(tmp_path):
    orch = FakeOrchestrator(claim_ok=False, claim_reason="no free VRAM")
    runner = FakeRunner(files=[])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    status, body = svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    assert status == 503
    assert json.loads(body)["error"]["type"] == "busy"
    assert orch.releases == []  # nothing was claimed -> nothing to release


def test_image_sync_run_failure_is_504(tmp_path):
    orch = FakeOrchestrator()
    runner = FakeRunner(ok=False, reason="history poll timed out")
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    status, body = svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    assert status == 504
    assert json.loads(body)["error"]["type"] == "generation_error"
    assert "history poll timed out" in json.loads(body)["error"]["message"]


# --- request budget parity: body._timeout -> runner poll deadline -----------

def test_image_sync_forwards_request_budget_to_runner(tmp_path):
    # the sync image route honours the client's _timeout (broker ~1670) by
    # handing it to the runner as the poll budget.
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[_write(tmp_path / "o.png")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1", "_timeout": 30}).encode())
    assert runner.run_inputs[0]["timeout_s"] == 30.0


def test_image_sync_default_budget_is_1200_when_unset(tmp_path):
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[_write(tmp_path / "o.png")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    assert runner.run_inputs[0]["timeout_s"] == 1200.0


def test_non_finite_timeout_falls_back_to_default_budget(tmp_path):
    # JSON can carry 1e400 -> inf; Thread.join(inf) raises OverflowError, so
    # a non-finite _timeout must be treated as unparseable (default budget).
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[_write(tmp_path / "o.png")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1",
                    "_timeout": 1e400}).encode())
    assert runner.run_inputs[0]["timeout_s"] == 1200.0


def test_image_sync_short_budget_is_honest_504(tmp_path):
    # a small _timeout yields an honest 504 (the runner reports the timeout),
    # not a request that outlives its budget.
    orch = FakeOrchestrator()
    runner = FakeRunner(ok=False,
                        reason="history poll timed out after 3.0s for pid")
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    status, body = svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1", "_timeout": 3}).encode())
    assert status == 504
    assert runner.run_inputs[0]["timeout_s"] == 3.0
    assert "timed out" in json.loads(body)["error"]["message"]


def test_image_sync_short_timeout_with_real_runner_is_504(tmp_path):
    # end-to-end with the REAL runner + a fake clock: body._timeout caps the
    # poll loop, so a tiny budget returns an honest 504 without waiting the
    # runner's 3600 s history budget.
    from model_forge.engines.comfyui import ComfyWorkflowRunner

    hits = {"history": 0}

    def http(method, url, body=None, *, files=None, raw=False, timeout=10.0):
        if "/history/" in url:
            hits["history"] += 1
            return {}                 # never completes
        if url.endswith("/prompt"):
            return {"prompt_id": "pid1"}
        return {}                     # /free and anything else

    runner = ComfyWorkflowRunner("http://comfy:18188", http=http,
                                 sleep=lambda _s: None,
                                 poll_interval_s=1.0,
                                 history_timeout_s=3600.0)
    orch = FakeOrchestrator()
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    status, body = svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1", "_timeout": 3}).encode())
    assert status == 504
    assert hits["history"] == 3       # request budget, not the 3600 s ceiling
    assert "timed out" in json.loads(body)["error"]["message"]


# --- normalized params on the job record (broker parity) --------------------

def test_image_job_record_params_are_normalized(tmp_path):
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[_write(tmp_path / "o.png")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    svc.handle_image_generations(json.dumps(
        {"model": "gen/qwen-image-2.1", "prompt": "p", "n": 1, "_timeout": 5,
         "size": "1024x576"}).encode())
    jid = orch.claims[0][0]
    params = json.loads(svc.handle_job(jid)[1])["params"]
    assert "model" not in params and "n" not in params
    assert "_timeout" not in params
    assert params["size"] == "1024x576"
    assert params["width"] == 1024 and params["height"] == 576
    assert params["aspect_ratio"] == "16:9"


def test_video_job_record_params_are_normalized(tmp_path):
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[str(tmp_path / "v.mp4")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner,
                       dispatch=_inline)
    raw = json.dumps({"model": "gen/h3-t2v", "seconds": "5",
                      "size": "1024x576"}).encode()
    jid = json.loads(svc.handle_video_submit(raw)[1])["id"]
    params = json.loads(svc.handle_job(jid)[1])["params"]
    assert "model" not in params
    assert params["seconds"] == 5.0
    assert params["resolution"] == "1024x576"
    assert "width" not in params and "height" not in params


def test_job_record_note_flags_deferred_image_upload(tmp_path):
    # a deferred image ref is a known divergence from the broker's stored shape
    # (upload at run time, not route time) — the row carries a visible note.
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[_write(tmp_path / "o.png")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    svc.handle_image_generations(json.dumps(
        {"model": "gen/qwen-image-2.1-edit",
         "image_url": "http://ex.test/a.png"}).encode())
    jid = orch.claims[0][0]
    row = svc._jobs.get(jid)
    assert row["note"] and "run time" in row["note"]
    # image_url is mapped to image in the stored params
    assert row["params"]["image"] == "http://ex.test/a.png"
    assert "image_url" not in row["params"]


def test_job_record_note_absent_without_image_ref(tmp_path):
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[_write(tmp_path / "o.png")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1", "prompt": "p"}).encode())
    jid = orch.claims[0][0]
    assert svc._jobs.get(jid)["note"] is None


# --- video job lifecycle ----------------------------------------------------

def test_video_job_lifecycle_submit_status_content_jobs(tmp_path):
    orch = FakeOrchestrator()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    runner = FakeRunner(files=[str(video)])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner,
                       dispatch=_inline)

    raw = json.dumps({"model": "gen/h3-t2v", "prompt": "sunset",
                      "seconds": 5, "size": "1024x576"}).encode()
    status, body = svc.handle_video_submit(raw)

    assert status == 200
    submitted = json.loads(body)
    assert submitted["status"] == "queued"
    assert submitted["model"] == "h3-t2v"
    jid = submitted["id"]
    # Inline dispatch: the job ran to completion during submit.
    assert orch.admits == ["h3-t2v"]
    # Video param normalization (broker _openai_video_params).
    assert runner.run_calls[0][1]["seconds"] == 5.0
    assert runner.run_calls[0][1]["resolution"] == "1024x576"

    # GET /v1/videos/{jid} -> completed with a data url.
    st, b = svc.handle_video_status(jid)
    assert st == 200
    done = json.loads(b)
    assert done["status"] == "completed"
    assert done["data"] == [{"url": f"http://127.0.0.1:8080/assets/{jid}/clip.mp4"}]

    # GET /v1/videos/{jid}/content -> raw bytes + sane content-type.
    st, b, ctype = svc.handle_video_content(jid)
    assert st == 200
    assert b == b"\x00\x00\x00\x18ftypmp42"
    assert ctype == "video/mp4"

    # GET /jobs/{jid} -> broker-shaped record.
    st, b = svc.handle_job(jid)
    assert st == 200
    rec = json.loads(b)
    assert rec["id"] == jid and rec["kind"] == "gen"
    assert rec["model"] == "h3-t2v" and rec["status"] == "done"
    assert rec["params"]["seconds"] == 5.0
    assert rec["assets"] == [f"/assets/{jid}/clip.mp4"]


def test_video_status_not_found_is_404(tmp_path):
    svc = _gen_service(tmp_path, orchestrator=FakeOrchestrator(),
                       runner=FakeRunner(), dispatch=_inline)
    status, body = svc.handle_video_status("deadbeef")
    assert status == 404
    assert json.loads(body)["error"]["type"] == "not_found"


def test_video_submit_wrong_kind_is_400(tmp_path):
    orch = FakeOrchestrator()
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner(),
                       dispatch=_inline)
    # an image workflow submitted to the video route
    status, body = svc.handle_video_submit(
        json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    assert status == 400
    assert "not video" in json.loads(body)["error"]["message"]
    assert orch.claims == []  # refused before any job/claim


def test_video_generations_generic_shape(tmp_path):
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[str(tmp_path / "v.mp4")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner,
                       dispatch=_inline)
    status, body = svc.handle_video_generations(
        json.dumps({"model": "gen/h3-i2v", "prompt": "x"}).encode())
    assert status == 200
    payload = json.loads(body)
    assert payload["status"] == "queued"
    assert payload["model"] == "h3-i2v"
    assert payload["poll"] == f"/jobs/{payload['id']}"
    assert payload["asset"] == f"/assets/{payload['id']}"


# --- back-to-back free() and runner requirement -----------------------------

def test_runner_free_called_between_back_to_back_jobs(tmp_path):
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[str(tmp_path / "o.png")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    raw = json.dumps({"model": "gen/qwen-image-2.1"}).encode()
    svc.handle_image_generations(raw)
    svc.handle_image_generations(raw)
    assert len(runner.run_calls) == 2
    assert runner.free_calls == 2  # one cache free per job (broker behavior)


def test_gen_without_runner_is_503(tmp_path):
    svc = _gen_service(tmp_path, orchestrator=FakeOrchestrator())
    status, body = svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    assert status == 503
    assert json.loads(body)["error"]["type"] == "busy"


# --- bad input --------------------------------------------------------------

def test_image_unknown_model_is_404(tmp_path):
    svc = _gen_service(tmp_path, orchestrator=FakeOrchestrator(),
                       runner=FakeRunner())
    status, body = svc.handle_image_generations(
        json.dumps({"model": "gen/nope"}).encode())
    assert status == 404
    assert json.loads(body)["error"]["type"] == "invalid_request_error"


def test_image_wrong_kind_is_400(tmp_path):
    svc = _gen_service(tmp_path, orchestrator=FakeOrchestrator(),
                       runner=FakeRunner())
    status, body = svc.handle_image_generations(
        json.dumps({"model": "gen/h3-t2v"}).encode())
    assert status == 400
    assert "not image" in json.loads(body)["error"]["message"]


def test_gen_invalid_json_is_400_and_touches_no_job(tmp_path):
    orch = FakeOrchestrator()
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner())
    status, body = svc.handle_image_generations(b"{not json")
    assert status == 400
    assert orch.claims == []
    st2, _ = svc.handle_video_submit(b"[]")  # non-object JSON body
    assert st2 == 400


# --- raising gen-path seams are honest 5xx, never dropped connections -------

class _RaisingOrchestrator(FakeOrchestrator):
    """A seam that raises (vs. returning ok=False) for one method."""

    def __init__(self, which, **kw):
        super().__init__(**kw)
        self.which = which

    def admit_gen(self, *a, **k):
        if self.which == "admit_gen":
            raise RuntimeError("admit_gen exploded")
        return super().admit_gen(*a, **k)

    def claim(self, *a, **k):
        if self.which == "claim":
            raise RuntimeError("claim exploded")
        return super().claim(*a, **k)


class _RaisingRunner(FakeRunner):
    def run(self, *a, **k):
        raise RuntimeError("runner exploded")


@pytest.mark.parametrize("which", ["claim", "admit_gen"])
def test_image_sync_raising_orchestrator_seam_is_honest_5xx(tmp_path, which):
    orch = _RaisingOrchestrator(which)
    runner = FakeRunner(files=[str(tmp_path / "o.png")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    status, body = svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    assert status == 500
    err = json.loads(body)["error"]
    assert err["type"] == "generation_error"
    assert "exploded" in err["message"]


def test_image_sync_raising_runner_is_honest_5xx(tmp_path):
    orch = FakeOrchestrator()
    runner = _RaisingRunner(files=[])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    status, body = svc.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    assert status == 500
    err = json.loads(body)["error"]
    assert err["type"] == "generation_error"
    assert "runner exploded" in err["message"]


def test_image_sync_raising_seam_keeps_the_connection(tmp_path):
    # the route method returns an honest 500 JSON; the handler writes it out
    # (no exception escapes to drop the connection) — driven through do_POST.
    orch = _RaisingOrchestrator("admit_gen")
    runner = FakeRunner(files=[])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    payload = json.dumps({"model": "gen/qwen-image-2.1"}).encode()
    status, body, _ = _drive_post(
        svc, "/v1/images/generations",
        {"Content-Length": str(len(payload))}, body=payload)
    assert status == 500
    assert json.loads(body)["error"]["type"] == "generation_error"


# --- upload + assets serving ------------------------------------------------

def _multipart(filename, data, field="file", boundary="BOUNDARY123"):
    b = boundary.encode()
    body = (b"--" + b + b"\r\n"
            + b'Content-Disposition: form-data; name="' + field.encode()
            + b'"; filename="' + filename.encode() + b'"\r\n'
            + b"Content-Type: image/png\r\n\r\n" + data + b"\r\n"
            + b"--" + b + b"--\r\n")
    return body, f"multipart/form-data; boundary={boundary}"


def test_upload_stores_file_and_returns_broker_shape(tmp_path):
    svc = _gen_service(tmp_path, orchestrator=FakeOrchestrator(),
                       runner=FakeRunner())
    body, ctype = _multipart("ref.png", b"\x89PNGdata")
    status, resp = svc.handle_upload(body, ctype)
    assert status == 200
    payload = json.loads(resp)
    assert payload["name"].endswith("_ref.png")
    assert payload["path"] == os.path.join(str(tmp_path), "uploads", payload["name"])
    assert os.path.exists(payload["path"])
    assert open(payload["path"], "rb").read() == b"\x89PNGdata"


def test_upload_without_file_is_400(tmp_path):
    svc = _gen_service(tmp_path, orchestrator=FakeOrchestrator(),
                       runner=FakeRunner())
    status, body = svc.handle_upload(b"not multipart", "")
    assert status == 400
    assert json.loads(body)["error"]["type"] == "invalid_request_error"


def test_asset_serves_bytes_with_content_type(tmp_path):
    orch = FakeOrchestrator()
    out = tmp_path / "pic.png"
    out.write_bytes(b"\x89PNG\x00")
    runner = FakeRunner(files=[str(out)])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    svc.handle_image_generations(json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    jid = orch.claims[0][0]
    status, body, ctype = svc.handle_asset(jid, "pic.png")
    assert status == 200
    assert body == b"\x89PNG\x00"
    assert ctype == "image/png"
    st2, _, _ = svc.handle_asset(jid, "missing.png")
    assert st2 == 404


def test_asset_outside_assets_dir_is_not_served(tmp_path):
    # confinement: a job whose recorded file resolves OUTSIDE assets_dir must
    # not be served, even when the recorded name matches (defence in depth).
    outside = tmp_path.parent / "escape.png"
    outside.write_bytes(b"secret")
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[str(outside)])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner)
    svc.handle_image_generations(json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    jid = orch.claims[0][0]
    status, _, _ = svc.handle_asset(jid, "escape.png")
    assert status == 404        # resolves outside assets_dir -> refused


# --- chat waiter accounting -------------------------------------------------

def test_chat_waiter_enter_before_upstream_and_exit_after(tmp_path):
    log = []

    class _Upstream:
        def __call__(self, raw, headers):
            log.append("upstream")
            return 200, b'{"ok":true}'

    orch = FakeOrchestrator(log=log)
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner())
    svc._upstream = _Upstream()  # inject chat upstream (gen routes unused here)

    status, _ = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())
    assert status == 200
    # prepare_llm parity: reserve the waiter, wake the lane (claim), forward,
    # release (restore=False so the lane stays up), then exit the waiter.
    assert log == ["waiter_enter", "claim", "upstream", "release", "waiter_exit"]
    assert orch.waiter_exits == [orch.waiter_token]
    # the chat claim targeted the chat model on the shared target.
    scope, target, request = orch.claims[0]
    assert request.model_id == CHAT_MODEL_ID
    assert request.engine == "vllm" and request.lane == "chat"
    # released WITHOUT restoring prior residency: the lane stays up.
    assert orch.releases == [(scope, False)]


def test_chat_waiter_exit_in_finally_when_upstream_raises(tmp_path):
    log = []

    class _Boom:
        def __call__(self, raw, headers):
            log.append("upstream")
            raise RuntimeError("vllm died mid-request")

    orch = FakeOrchestrator(log=log)
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner())
    svc._upstream = _Boom()

    with pytest.raises(RuntimeError):
        svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())
    # The waiter still exits exactly once, and any taken scope is released
    # (restore=False keeps the lane up) even though the upstream raised.
    assert log == ["waiter_enter", "claim", "upstream", "release", "waiter_exit"]
    assert orch.waiter_exits == [orch.waiter_token]
    assert orch.releases and orch.releases[0][1] is False


def test_chat_waiter_refusal_is_503_and_touches_no_upstream(tmp_path):
    calls = []

    class _Upstream:
        def __call__(self, raw, headers):
            calls.append(1)
            return 200, b"{}"

    orch = FakeOrchestrator(waiter_ok=False,
                            waiter_reason="lane wake in progress")
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner())
    svc._upstream = _Upstream()
    status, body = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())
    assert status == 503
    assert json.loads(body)["error"]["type"] == "busy"
    assert "lane wake in progress" in json.loads(body)["error"]["message"]
    assert calls == []
    assert orch.waiter_exits == []  # no token was ever handed out


def test_chat_waiter_ok_without_token_is_fail_closed_503(tmp_path):
    # a seam returning {'ok': True, 'token': None} would leak the reservation
    # (handle_chat only calls waiter_exit when a token was handed out). Fail
    # closed: honest 503, no upstream call.
    calls = []

    class _Upstream:
        def __call__(self, raw, headers):
            calls.append(1)
            return 200, b"{}"

    orch = FakeOrchestrator(waiter_ok=True, waiter_token=None)
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner())
    svc._upstream = _Upstream()
    status, body = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())
    assert status == 503
    assert json.loads(body)["error"]["type"] == "busy"
    assert calls == []                    # refused before the upstream
    assert orch.waiter_exits == []        # no token to release -> no leak


def test_chat_waiter_seam_raising_is_fail_closed_503(tmp_path):
    class _BoomOrch(FakeOrchestrator):
        def waiter_enter(self):
            raise RuntimeError("orchestrator exploded")

    svc = _gen_service(tmp_path, orchestrator=_BoomOrch(), runner=FakeRunner())
    status, body = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())
    assert status == 503
    assert "orchestrator exploded" in json.loads(body)["error"]["message"]


# --- handler wiring for the new routes (no ports bound) ---------------------

def _drive_get(service, path):
    """Run the real do_GET with fakes (no socket); return status/body/ctype."""
    handler_cls = make_handler(service)
    handler = handler_cls.__new__(handler_cls)  # bypass create_server/socket
    handler.path = path
    handler.headers = _FakeHeaders({})
    handler.wfile = _FakeWFile()
    captured = {"status": None, "headers": {}}
    handler.send_response = lambda status, message=None: captured.__setitem__("status", status)
    handler.send_header = lambda k, v: captured["headers"].__setitem__(k, v)
    handler.end_headers = lambda: None
    handler.do_GET()
    return captured["status"], handler.wfile.buf, captured["headers"].get("Content-Type")


def test_handler_routes_image_generations_and_video_and_jobs(tmp_path):
    out = _write(tmp_path / "h.png")
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[out])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner,
                       dispatch=_inline)

    payload = json.dumps({"model": "gen/qwen-image-2.1"}).encode()
    status, body, _ = _drive_post(
        svc, "/v1/images/generations",
        {"Content-Length": str(len(payload))}, body=payload)
    assert status == 200
    assert "data" in json.loads(body)

    jid = json.loads(svc.handle_video_submit(
        json.dumps({"model": "gen/h3-t2v"}).encode())[1])["id"]
    st, b, ctype = _drive_get(svc, f"/v1/videos/{jid}")
    assert st == 200 and json.loads(b)["status"] == "completed"
    st, b, _ = _drive_get(svc, f"/jobs/{jid}")
    assert st == 200 and json.loads(b)["id"] == jid
    st, _, _ = _drive_get(svc, "/no/such/route")
    assert st == 404


# ===========================================================================
# P5c — chat path: prepare_llm parity (wait for gen jobs, wake the lane,
# leave it up). The real Orchestrator + recording engine adapters drive the
# wake; a fake clock drives the bounded wait loop.
# ===========================================================================

from model_forge.broker import (  # noqa: E402
    EngineLifecycle,
    EvictionPlanner,
    LoadRequest,
    Orchestrator,
    ResidencyLedger,
)


class _FakeClock:
    def __init__(self, now=1000.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += s


class _RecordingLifecycle(EngineLifecycle):
    """Records ensure_loaded/release (no processes, no network)."""

    def __init__(self, engine_id):
        self.engine_id = engine_id
        self.calls = []

    def ensure_loaded(self, model_id, **kwargs):
        self.calls.append(("ensure_loaded", model_id))
        return {"ok": True, "reason": "warmed"}

    def release(self, model_id):
        self.calls.append(("release", model_id))
        return {"ok": True, "reason": "evicted"}

    def probe(self):
        return {"ok": True, "state": "observed", "resident": []}


def _orch_for_chat(clock=None):
    """A real orchestrator over the shared 'comfy' target: the chat lane is
    booked to the vLLM adapter, a gen workflow to the comfyui adapter."""
    clock = clock or _FakeClock()
    ledger = ResidencyLedger(
        bookings={CHAT_MODEL_ID: 45.6, "image-x": 12.0}, clock=clock)
    ledger.set_budget("comfy", 48.0)
    vllm = _RecordingLifecycle("vllm")
    comfy = _RecordingLifecycle("comfyui")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"vllm": vllm, "comfyui": comfy}, clock=clock)
    return orch, ledger, vllm, comfy, clock


def _chat_svc(orch, clock=None, sleep=None, upstream=None, **cfg):
    config = FacadeConfig(llm_canonical_id=CHAT_MODEL_ID,
                          llm_served_id="qwen-served", gen_target_id="comfy",
                          **cfg)
    return FacadeService(config, upstream=upstream or FakeUpstream(),
                         orchestrator=orch, clock=clock, sleep=sleep)


def test_chat_wakes_the_lane_via_orchestrator_and_leaves_it_up():
    orch, ledger, vllm, _comfy, _clock = _orch_for_chat()
    svc = _chat_svc(orch)

    status, _ = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())

    assert status == 200
    # the wake ran: the claim drove the vLLM lane's ensure_loaded.
    assert ("ensure_loaded", CHAT_MODEL_ID) in vllm.calls
    assert ledger.is_resident("comfy", CHAT_MODEL_ID)
    # released with restore=False -> the lane is NOT stepped down, and the
    # scope no longer occupies the table.
    assert ("release", CHAT_MODEL_ID) not in vllm.calls
    assert orch._scopes == {}


def test_chat_exposes_release_with_restore_false(tmp_path):
    orch = FakeOrchestrator()
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner())
    svc._upstream = FakeUpstream()

    status, _ = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())

    assert status == 200
    assert orch.claims and orch.claims[0][2].model_id == CHAT_MODEL_ID
    assert orch.releases == [(f"chat-{orch.waiter_token}", False)]


def test_chat_waits_for_running_gen_job_then_wakes():
    clock = _FakeClock()
    orch, _ledger, vllm, _comfy, _ = _orch_for_chat(clock=clock)
    # a gen job is running: an open image scope holds the card.
    assert orch.claim("gen-1", "comfy",
                      LoadRequest("image-x", "comfyui", lane="image",
                                  modality="image"))["ok"] is True
    assert orch.gen_busy is True
    polls = []

    def sleep(s):
        polls.append(s)
        clock.advance(s)
        orch.release("gen-1")           # the gen job finishes during the wait

    svc = _chat_svc(orch, clock=clock, sleep=sleep, max_gen_wait_s=30.0)
    status, _ = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())

    assert status == 200
    assert polls == [0.25]              # polled once; budget not exhausted
    assert ("ensure_loaded", CHAT_MODEL_ID) in vllm.calls   # then it woke
    assert orch.gen_busy is False


def test_chat_wait_budget_exhaustion_is_honest_503():
    clock = _FakeClock()
    orch, _ledger, vllm, _comfy, _ = _orch_for_chat(clock=clock)
    orch.claim("gen-1", "comfy",
               LoadRequest("image-x", "comfyui", lane="image",
                           modality="image"))

    def sleep(s):
        clock.advance(s)

    svc = _chat_svc(orch, clock=clock, sleep=sleep, max_gen_wait_s=3.0)
    status, body = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())

    assert status == 503
    err = json.loads(body)["error"]
    assert err["type"] == "busy"
    assert "LLM wait budget exhausted" in err["message"]
    # the waiter exited exactly once; no chat claim was taken (the gen scope
    # is the only one open) and the lane was never woken.
    assert orch.llm_waiters == 0
    assert list(orch._scopes) == ["gen-1"]
    assert ("ensure_loaded", CHAT_MODEL_ID) not in vllm.calls


def test_chat_claim_refusal_is_503_and_exits_waiter(tmp_path):
    orch = FakeOrchestrator(claim_ok=False, claim_reason="no free VRAM")
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner())
    svc._upstream = FakeUpstream()

    status, body = svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())

    assert status == 503
    assert "no free VRAM" in json.loads(body)["error"]["message"]
    assert orch.waiter_exits == [orch.waiter_token]
    assert orch.releases == []          # claim refused -> nothing to release


def test_chat_upstream_raise_exits_waiter_and_releases_scope(tmp_path):
    orch = FakeOrchestrator()

    class _Boom:
        def __call__(self, raw, headers):
            raise RuntimeError("vllm died mid-request")

    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner())
    svc._upstream = _Boom()

    with pytest.raises(RuntimeError):
        svc.handle_chat(json.dumps({"model": CHAT_MODEL_ID}).encode())

    assert orch.waiter_exits == [orch.waiter_token]
    assert orch.releases == [(f"chat-{orch.waiter_token}", False)]


# ===========================================================================
# P5 follow-up — async video caller budget (body._timeout -> bounded job)
#
# The async video routes accept the sync image route's ``_timeout`` budget.
# On expiry the job row must end honestly (failed, 'timed out after Ns') and
# the orchestrator scope must be RELEASED — a stalled runner must not pin the
# gen lane for the runner's 3600 s default window.
# ===========================================================================

import threading  # noqa: E402
import time  # noqa: E402


class BlockingRunner:
    """A stalled-ComfyUI runner: ``run`` blocks past any budget and IGNORES
    the ``timeout_s`` seam (a black-holed poll can do exactly that), so the
    facade must bound the job and release the claim itself. The test lets the
    abandoned worker go via ``release_evt`` at teardown."""

    def __init__(self, workflows=None):
        self.workflows = _workflows() if workflows is None else workflows
        self.run_inputs = []
        self.started = threading.Event()
        self.release_evt = threading.Event()
        self.free_calls = 0

    def run(self, workflow_id, params, *, assets_dir, job_id, timeout_s=None):
        self.run_inputs.append({"workflow_id": workflow_id,
                                "params": params, "job_id": job_id,
                                "timeout_s": timeout_s})
        self.started.set()
        self.release_evt.wait()       # blocks until the test lets go
        return {"ok": True, "output_kind": "video", "files": [],
                "reason": "", "meta": {}}

    def free(self):
        self.free_calls += 1
        return {"ok": True}


def test_video_submit_timeout_fails_honestly_and_releases_claim(tmp_path):
    orch = FakeOrchestrator()
    runner = BlockingRunner()
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner,
                       dispatch=_inline)
    try:
        raw = json.dumps({"model": "gen/h3-t2v", "prompt": "p",
                          "_timeout": 0.05}).encode()
        start = time.monotonic()
        status, body = svc.handle_video_submit(raw)
        elapsed = time.monotonic() - start
        assert status == 200                    # async submit shape unchanged
        jid = json.loads(body)["id"]
        # the budget reached the runner seam too (its own poll loop stays
        # bounded) even though this runner ignores it
        assert runner.run_inputs[0]["timeout_s"] == 0.05
        # the job ends on the REQUEST budget, not the runner's 3600 s window
        assert elapsed < 2.0
        row = svc._jobs.get(jid)
        assert row["status"] == "failed"        # honest terminal state
        assert "timed out after 0.05s" in row["error"]
        assert orch.releases                    # claim released -> lane free
        assert runner.free_calls >= 1           # weights freed too
    finally:
        runner.release_evt.set()                # let the abandoned worker die


def test_video_generations_forwards_budget_and_drops_it_from_params(tmp_path):
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[str(tmp_path / "v.mp4")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner,
                       dispatch=_inline)
    raw = json.dumps({"model": "gen/h3-t2v", "prompt": "p",
                      "_timeout": 3}).encode()
    status, body = svc.handle_video_generations(raw)
    assert status == 200
    jid = json.loads(body)["id"]
    assert runner.run_inputs[0]["timeout_s"] == 3.0   # bounded poll budget
    # _timeout is transport, not a workflow param: it must not reach the
    # stored job params (image-route parity: 'n'/'_timeout' are dropped).
    params = json.loads(svc.handle_job(jid)[1])["params"]
    assert "_timeout" not in params


def test_video_submit_without_timeout_keeps_runner_history_budget(tmp_path):
    # absent _timeout -> today's behavior unchanged: no facade-side bound and
    # the runner uses its own history budget.
    orch = FakeOrchestrator()
    runner = FakeRunner(files=[str(tmp_path / "v.mp4")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner,
                       dispatch=_inline)
    jid = json.loads(svc.handle_video_submit(
        json.dumps({"model": "gen/h3-t2v"}).encode())[1])["id"]
    assert runner.run_inputs[0]["timeout_s"] is None
    assert svc._jobs.get(jid)["status"] == "done"


# --- prepare_gen: per-workload args flip, lane restart, LLM-lane adoption ----

class FakeLane:
    """Fake gen-lane adapter: records probe/release; state configurable."""

    def __init__(self, state="running", log=None):
        self.state = state
        self.log = log
        self.calls = []

    def probe(self):
        self.calls.append(("probe",))
        return {"ok": True, "state": self.state}

    def release(self, model_id):
        self.calls.append(("release", model_id))
        if self.log is not None:
            self.log.append("lane_stop")
        return {"ok": True, "reason": "stopped"}


class PrepOrchestrator(FakeOrchestrator):
    """FakeOrchestrator plus the prepare_gen seams (adapter_for/adopt_live)."""

    def __init__(self, *, lane_state="running", adopt_ok=True, **kw):
        super().__init__(**kw)
        self.lane = FakeLane(state=lane_state, log=kw.get("log"))
        self.adopt_ok = adopt_ok
        self.adopts = []

    def adapter_for(self, engine):
        return self.lane if engine == "comfyui" else None

    def adopt_live(self, target_id, model_id, engine, lane="chat"):
        self.adopts.append((target_id, model_id, engine, lane))
        self._rec("adopt_live")
        return {"ok": True, "adopted": self.adopt_ok,
                "reason": "fake adoption row"}


def test_comfy_args_policy_is_video_novram_image_empty():
    assert comfy_args_for("video") == "--novram"
    assert comfy_args_for("image") == ""
    assert comfy_args_for("") == "" and comfy_args_for(None) == ""


def test_flip_comfy_args_writes_video_novram_and_clears_for_image(tmp_path):
    f = tmp_path / "comfy_args.conf"
    f.write_text("")                              # leftover image-mode state
    row = flip_comfy_args(str(f), "video")
    assert row["ok"] is True and row["changed"] is True
    assert f.read_text().split() == ["--novram"]
    row = flip_comfy_args(str(f), "image")
    assert row["ok"] is True and row["changed"] is True
    assert f.read_text().strip() == ""


def test_flip_comfy_args_untouched_when_tokens_already_match(tmp_path):
    f = tmp_path / "comfy_args.conf"
    f.write_text("--novram\n")
    row = flip_comfy_args(str(f), "video")
    assert row["ok"] is True and row["changed"] is False
    assert "untouched" in row["reason"]


def test_flip_comfy_args_unconfigured_is_honest_noop():
    row = flip_comfy_args("", "video")
    assert row["ok"] is True and row["changed"] is False
    assert "not managed" in row["reason"]


def test_flip_comfy_args_write_failure_is_honest_not_a_raise(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    row = flip_comfy_args(str(blocker / "args.conf"), "video")
    assert row["ok"] is False and row["changed"] is False
    assert "failed" in row["reason"]


def test_prepare_gen_flips_args_restarts_lane_and_adopts_llm_lane(tmp_path):
    args = tmp_path / "comfy_args.conf"
    args.write_text("")                           # leftover image-mode state
    orch = PrepOrchestrator()
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner(),
                       comfy_args_file=str(args))

    row = svc.prepare_gen("h3-t2v", "video")

    assert row["ok"] is True
    assert args.read_text().split() == ["--novram"]
    # the lane was stopped so the claim's load rebinds the new args ...
    assert ("release", "h3-t2v") in orch.lane.calls
    # ... and the live LLM lane was adopted for stop/restore by claim/release
    assert orch.adopts == [("comfy", CHAT_MODEL_ID, "vllm", "chat")]


def test_prepare_gen_leaves_lane_alone_when_args_unchanged(tmp_path):
    args = tmp_path / "comfy_args.conf"
    args.write_text("--novram\n")                 # already video-mode
    orch = PrepOrchestrator()
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner(),
                       comfy_args_file=str(args))

    row = svc.prepare_gen("h3-t2v", "video")

    assert row["ok"] is True
    assert row["args"]["changed"] is False
    assert not any(c[0] == "release" for c in orch.lane.calls)
    # adoption still runs — it keys on the LLM lane, not on the args
    assert orch.adopts == [("comfy", CHAT_MODEL_ID, "vllm", "chat")]


def test_prepare_gen_skips_lane_restart_when_lane_not_running(tmp_path):
    args = tmp_path / "comfy_args.conf"
    args.write_text("")
    orch = PrepOrchestrator(lane_state="stopped")
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner(),
                       comfy_args_file=str(args))

    row = svc.prepare_gen("h3-t2v", "video")

    assert row["ok"] is True
    assert args.read_text().split() == ["--novram"]
    assert not any(c[0] == "release" for c in orch.lane.calls)
    assert "no restart needed" in row["lane"]["reason"]


def test_prepare_gen_without_orchestrator_seams_still_flips(tmp_path):
    args = tmp_path / "comfy_args.conf"
    args.write_text("")
    orch = FakeOrchestrator()       # no adapter_for / adopt_live seams
    svc = _gen_service(tmp_path, orchestrator=orch, runner=FakeRunner(),
                       comfy_args_file=str(args))

    row = svc.prepare_gen("h3-t2v", "video")

    assert row["ok"] is True
    assert args.read_text().split() == ["--novram"]
    assert "fail-soft" in row["lane"]["reason"]
    assert "fail-soft" in row["adopt"]["reason"]


def test_gen_job_orders_admit_then_prepare_then_claim(tmp_path):
    log = []
    args = tmp_path / "comfy_args.conf"
    args.write_text("--novram")    # video leftover; the image job must clear
    orch = PrepOrchestrator(log=log)
    runner = FakeRunner(files=[_write(tmp_path / "shot.png")], log=log)
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner,
                       dispatch=_inline, comfy_args_file=str(args))

    raw = json.dumps({"model": "gen/qwen-image-2.1", "prompt": "x"}).encode()
    status, _ = svc.handle_image_generations(raw)

    assert status == 200
    # admit_gen first (roadmap §3), then prepare_gen's lane stop + adoption,
    # then claim -> run -> release.
    assert log.index("admit_gen") < log.index("lane_stop")
    assert log.index("lane_stop") < log.index("adopt_live")
    assert (log.index("adopt_live") < log.index("claim") < log.index("run")
            < log.index("release"))
    assert args.read_text().strip() == ""


def test_video_submit_flips_args_to_novram(tmp_path):
    args = tmp_path / "comfy_args.conf"
    args.write_text("")                           # leftover image-mode state
    orch = PrepOrchestrator()
    runner = FakeRunner(files=[_write(tmp_path / "v.mp4")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner,
                       dispatch=_inline, comfy_args_file=str(args))

    status, _ = svc.handle_video_submit(
        json.dumps({"model": "gen/h3-t2v", "prompt": "x"}).encode())

    assert status == 200
    assert args.read_text().split() == ["--novram"]
    assert ("release", "h3-t2v") in orch.lane.calls


def test_gen_job_survives_an_args_write_failure(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    orch = PrepOrchestrator()
    runner = FakeRunner(files=[_write(tmp_path / "g.png")])
    svc = _gen_service(tmp_path, orchestrator=orch, runner=runner,
                       dispatch=_inline,
                       comfy_args_file=str(blocker / "args.conf"))

    raw = json.dumps({"model": "gen/qwen-image-2.1", "prompt": "x"}).encode()
    status, _ = svc.handle_image_generations(raw)

    # a failed flip degrades to the pre-flip behavior, never sinks the job
    assert status == 200
    assert orch.claims and orch.releases


