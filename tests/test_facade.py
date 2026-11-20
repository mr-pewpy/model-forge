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
