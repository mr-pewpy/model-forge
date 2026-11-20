"""OpenAI-compatible route facade — the thin layer consumers keep pointing at.

WHY this is separate from ``broker.py``: the orchestrator owns *policy* (what
weights are resident, when to swap, who waits). The facade owns *routes* — the
HTTP surface that Hermes providers, the image plugin, the vision cron and
OpenViking (.63) were written against. Keeping route plumbing apart from
orchestration policy is what lets the orchestrator change without rewriting the
contract that live clients depend on.

This module ports ONLY the broker's route shapes (``GET /health``,
``GET /v1/models``, ``POST /v1/chat/completions``). It is stdlib-only
(``http.server`` + ``urllib``) because the model-forge test venv carries no
third-party HTTP stack. Everything with a side effect is injectable — the
upstream chat call, the readiness probes, and the claim ``gate`` — so tests
drive the real code paths with fakes and bind no ports.

Two behaviours are ported deliberately, not by accident:

* **Alias canonicalization before routing.** The request's ``model`` is mapped
  through :mod:`model_forge.aliases` first, so pre-rename clients keep working.
* **Explicit ``Content-Type: application/json`` on the upstream passthrough.**
  The broker's httpx passthrough sent a raw ``content=`` body without a
  Content-Type and vLLM 400'd on it. The fix is replicated here: the header is
  set explicitly, never left to a client library's default.

The one orchestration seam is ``gate``: an async-agnostic callable invoked
before the upstream call that returns ``{'ok': bool, 'reason': str}``. Wave 2
plugs the orchestrator's claim/release + LLM-waiter counter in behind it
without touching the routes; the default gate trivially allows. Policy is
fail-closed (a refusing gate gets an honest 503); probes are fail-soft (a
raising probe reports a negative readiness, never an exception).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .aliases import DEFAULT_ALIASES, resolve

# ---------------------------------------------------------------------------
# Advertisement + public catalog
# ---------------------------------------------------------------------------

# The canonical chat lane id and the servable gen ids (7) — the 8 public ids
# live consumers enumerate. output_kind drives client routing (image vs video).
CHAT_MODEL_ID = "Qwen/Qwen3.8-27B-FP8"

GEN_MODELS: dict[str, str] = {
    "gen/qwen-image-2.1": "image",
    "gen/qwen-image-2.1-edit": "image",
    "gen/qwen-image-2.1-cutout": "image",
    "gen/moss-image": "image",
    "gen/h3-t2v": "video",
    "gen/h3-i2v": "video",
    "gen/h3-r2v": "video",
}

# Per-model advertisement metadata, ported from the broker config.yaml
# ``models:`` section (keyed by the EXACT advertised id). Unknown fields are
# ignored by OpenAI clients; Hermes reads context_length/max_model_len to size
# the context window, so the chat entry must carry it.
MODELS_ADVERTISED: dict[str, dict] = {
    "Qwen/Qwen3.8-27B-FP8": {
        "label": "Qwen3.8-27B-FP8",
        "description": (
            "Resident LLM lane — OBLITERATUS abliterated FP8 build (only chat "
            "model kept, 2026-11-19). vLLM TP2 across both 3090s, 262K context, "
            "native vision, tool calling, thinking control."
        ),
        "engine": "vllm",
        "context_length": 262144,
        "quantization": "fp8-full",
        "license": "apache-2.0",
        "capabilities": {
            "input": ["text", "image"],
            "output": ["text"],
            "vision": True,
            "tools": True,
            "thinking": True,
            "embeddings": False,
        },
    },
    "gen/moss-image": {
        "label": "MOSS-Image SFT (bf16)",
        "description": (
            "OpenMOSS MOSS-Image text-to-image, bf16 full precision. Native JSON "
            "layout control + text rendering. Reference sampling: 50 steps, "
            "CFG 3, Euler, SLG layer 22."
        ),
        "engine": "comfyui",
        "license": "apache-2.0",
        "capabilities": {"input": ["text"], "output": ["image"]},
    },
    "gen/qwen-image-2.1": {
        "label": "Qwen-Image-2.1",
        "description": (
            "Unified text-to-image + editing model (7B DiT, bf16) with PE-T2I "
            "prompt rewrite. Native 2K, RGBA transparency, typography. 25 "
            "steps, cfg 1."
        ),
        "engine": "comfyui",
        "capabilities": {"input": ["text"], "output": ["image"]},
    },
    "gen/qwen-image-2.1-edit": {
        "label": "Qwen-Image-2.1 Edit",
        "description": (
            "Image editing with up to 10 reference images, masks and identity "
            "hold (PE-I2I prompt rewrite). Output size follows the input image."
        ),
        "engine": "comfyui",
        "capabilities": {"input": ["text", "image"], "output": ["image"]},
    },
    "gen/qwen-image-2.1-cutout": {
        "label": "Qwen-Image-2.1 Cutout",
        "description": (
            "Transparent PNG cutouts / background removal. Literal prompts (no "
            "PE rewrite) — task phrasing like \"Remove the background, and "
            "output a PNG image\". Output carries a real alpha matte."
        ),
        "engine": "comfyui",
        "capabilities": {"input": ["text", "image"], "output": ["image"]},
    },
    "gen/h3-t2v": {
        "label": "MiniMax H3 text-to-video",
        "description": "Text-to-video. 1024x576, ~16 min for 5 s at 24 fps.",
        "engine": "comfyui",
        "capabilities": {"input": ["text"], "output": ["video"]},
    },
    "gen/h3-i2v": {
        "label": "MiniMax H3 image-to-video",
        "description": "Image-to-video (first-frame animation). 1024x576.",
        "engine": "comfyui",
        "capabilities": {"input": ["text", "image"], "output": ["video"]},
    },
    "gen/h3-r2v": {
        "label": "MiniMax H3 reference-to-video",
        "description": "Reference-image-to-video. 1024x576.",
        "engine": "comfyui",
        "capabilities": {"input": ["text", "image"], "output": ["video"]},
    },
}

# Enrichment keys copied through verbatim; context_length is coerced to int and
# mirrored as max_model_len (the vLLM-native name Hermes also reads).
_ENRICH_KEYS = ("label", "description", "engine", "quantization", "license",
                "capabilities")


def _owned_by(model_id: str) -> str:
    return "comfyui" if model_id in GEN_MODELS else "vllm"


def _enrich(entry: dict, meta: dict | None = None) -> dict:
    """Attach advertisement fields for an entry (broker ``enrich`` shape)."""
    meta = meta if meta is not None else MODELS_ADVERTISED.get(entry.get("id", ""))
    if not meta:
        return entry
    out = dict(entry)
    for key in _ENRICH_KEYS:
        if key in meta:
            out[key] = meta[key]
    if "context_length" in meta:
        out["context_length"] = int(meta["context_length"])
        out.setdefault("max_model_len", out["context_length"])
    return out


# ---------------------------------------------------------------------------
# Config + service
# ---------------------------------------------------------------------------

@dataclass
class FacadeConfig:
    """Injectable facade configuration — mirrors the broker's ``llm`` block.

    ``llm_canonical_id`` is the id advertised to clients (config ``llm.model``);
    ``llm_served_id`` is what the engine actually serves (config
    ``llm.served_id``). The chat passthrough rewrites canonical -> served before
    forwarding. ``llm_api_key`` falls back to ``BROKER_VLLM_API_KEY`` only when
    no explicit value is given — the env read is the field ``default_factory``,
    so an explicitly-passed ``llm_api_key`` always wins over the environment.
    """
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_served_id: str = CHAT_MODEL_ID
    llm_canonical_id: str = CHAT_MODEL_ID
    llm_api_key: str = field(default_factory=lambda: os.environ.get("BROKER_VLLM_API_KEY", ""))
    upstream_timeout_s: float = 600.0

    @classmethod
    def from_env(cls, **overrides) -> "FacadeConfig":
        """Build a config; ``llm_api_key`` defaults to ``BROKER_VLLM_API_KEY``.

        Pure construction over ``cls(**overrides)`` — the env lookup lives in
        the ``llm_api_key`` field default, so an ``llm_api_key`` passed here
        (or the dataclass constructor) always wins over the environment.
        """
        return cls(**overrides)


# An upstream call: (raw_body_bytes, headers) -> (status:int, body:bytes).
UpstreamCall = Callable[[bytes, dict], tuple[int, bytes]]
# A readiness probe: () -> value; may raise (fail-soft handled by the caller).
Probe = Callable[[], object]
# A claim gate: (canonical_model_id) -> {'ok': bool, 'reason': str}.
GateCall = Callable[[str], dict]


def _allow_gate(model_id: str) -> dict:  # noqa: ARG001 - seam signature
    return {"ok": True, "reason": ""}


class FacadeService:
    """Route logic, decoupled from the socket so tests drive it directly.

    Every external effect is a constructor seam: ``upstream`` performs the chat
    passthrough, ``probes`` feed /health, ``gate`` is the orchestrator seam.
    Methods return ``(status, body_bytes)`` — the handler only writes them out.
    """

    def __init__(self, config: FacadeConfig | None = None, *,
                 upstream: UpstreamCall | None = None,
                 gate: GateCall | None = None,
                 probes: dict[str, Probe] | None = None,
                 aliases: dict[str, str] | None = None,
                 models_advertised: dict[str, dict] | None = None):
        self.config = config or FacadeConfig.from_env()
        self.aliases = DEFAULT_ALIASES if aliases is None else aliases
        self.models = models_advertised if models_advertised is not None else MODELS_ADVERTISED
        self._upstream = upstream or self._urllib_upstream()
        self._gate = gate or _allow_gate
        self._probes = dict(probes or {})

    # -- upstream default (urllib; the passthrough fix lives in handle_chat) --

    def _urllib_upstream(self) -> UpstreamCall:
        url = self.config.llm_base_url.rstrip("/") + "/chat/completions"
        timeout = self.config.upstream_timeout_s

        def _post(raw: bytes, headers: dict) -> tuple[int, bytes]:
            req = urllib.request.Request(url, data=raw, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return int(getattr(r, "status", 200)), r.read()
            except urllib.error.HTTPError as e:
                return int(e.code), e.read()
            except Exception as e:  # unreachable upstream -> honest 502
                return 502, _error_bytes(f"upstream unreachable: {e}", "upstream_error")

        return _post

    # -- catalog -------------------------------------------------------------

    def catalog(self) -> dict:
        """The /v1/models payload: 8 public ids enriched + one alias entry each.

        Alias entries carry ``alias_of`` pointing at the resolved canonical id so
        a client that enumerates models still sees the legacy ids it pins, and
        the enrichment fields are looked up under the canonical id (the broker's
        "rewrite ids then enrich" order).
        """
        data: list[dict] = []
        data.append(_enrich({"id": CHAT_MODEL_ID, "object": "model", "created": 0,
                             "owned_by": "vllm"}))
        for mid, kind in GEN_MODELS.items():
            data.append(_enrich({"id": mid, "object": "model", "created": 0,
                                 "owned_by": "comfyui", "output_kind": kind,
                                 "kind": kind}))
        for alias, target in self.aliases.items():
            entry = {"id": alias, "object": "model", "created": 0,
                     "owned_by": _owned_by(target), "alias_of": target}
            # output_kind/kind live in GEN_MODELS, not MODELS_ADVERTISED, so a
            # gen-lane alias must inherit them explicitly — otherwise a client
            # routing on output_kind sees the alias as untyped (the canonical
            # id it points at is advertised with a kind).
            if target in GEN_MODELS:
                kind = GEN_MODELS[target]
                entry["output_kind"] = kind
                entry["kind"] = kind
            data.append(_enrich(entry, meta=self.models.get(target)))
        return {"object": "list", "data": data}

    # -- health --------------------------------------------------------------

    def _probe(self, name: str, default: object) -> object:
        fn = self._probes.get(name)
        if fn is None:
            if name == "llm":
                fn = self._default_llm_probe()
            else:
                return default
        try:
            return fn()
        except Exception:
            return default  # fail-soft: a dead probe is not a dead facade

    def _default_llm_probe(self) -> Probe:
        base = self.config.llm_base_url
        root = base[:-3] if base.endswith("/v1") else base
        url = root.rstrip("/") + "/health"

        def _probe() -> bool:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    return int(getattr(r, "status", 0)) == 200
            except Exception:
                return False

        return _probe

    def health(self) -> dict:
        """Lane readiness, fail-soft (mirrors the broker's /health loosely)."""
        llm = bool(self._probe("llm", False))
        comfy = bool(self._probe("comfy", False))
        mode = self._probe("mode", "unknown")
        return {"llm": llm, "comfy": comfy, "mode": mode, "ok": llm}

    # -- chat passthrough ----------------------------------------------------

    def handle_chat(self, raw: bytes) -> tuple[int, bytes]:
        """Alias-canonicalize, gate, rewrite to served id, forward verbatim."""
        try:
            body = json.loads(raw or b"{}")
        except Exception as e:
            return 400, _error_bytes(f"invalid JSON body: {e}", "invalid_request_error")
        if not isinstance(body, dict):
            return 400, _error_bytes("request body must be a JSON object",
                                     "invalid_request_error")

        model_id = str(body.get("model", ""))
        canonical = resolve(model_id, self.aliases)
        if canonical != model_id:
            body["model"] = canonical
            model_id = canonical

        # Orchestration seam: refuse BEFORE touching the upstream. Fail-closed —
        # a refusing gate yields an honest 503, never a silent forward. A gate
        # that RAISES (a faulting wave-2 orchestrator) is caught here too:
        # policy is fail-closed, so a broken gate is still an honest 503, never
        # a dropped connection nor a request that sneaks past the claim. A
        # malformed (non-dict) decision is treated the same way.
        try:
            decision = self._gate(model_id) or {}
            ok = bool(decision.get("ok", False))
            reason = decision.get("reason") or "lane unavailable"
        except Exception as e:
            return 503, _error_bytes(f"gate error: {e}", "busy")
        if not ok:
            return 503, _error_bytes(reason, "busy")

        # canonical -> served rewrite (rename pass; LiteLLM-style id mapping).
        if self.config.llm_served_id and body.get("model") == self.config.llm_canonical_id:
            body["model"] = self.config.llm_served_id

        out_raw = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}  # explicit — see module doc
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
        status, text = self._upstream(out_raw, headers)
        return int(status), text if isinstance(text, bytes) else str(text).encode()


def _error_bytes(message: str, err_type: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": err_type}}).encode()


# ---------------------------------------------------------------------------
# HTTP plumbing (stdlib) — thin: parse, delegate to the service, write out.
# ---------------------------------------------------------------------------

def make_handler(service: FacadeService):
    """Build a BaseHTTPRequestHandler subclass bound to ``service``."""

    class FacadeHandler(BaseHTTPRequestHandler):
        server_version = "model-forge-facade/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A002 - base signature; quiet
            pass

        def _send(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _not_found(self) -> None:
            self._send(404, _error_bytes(f"no route for {self.path}", "not_found"))

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, json.dumps(service.health()).encode())
            elif self.path == "/v1/models":
                self._send(200, json.dumps(service.catalog()).encode())
            else:
                self._not_found()

        def _body_length(self) -> int | None:
            """Content-Length as an int, or None after sending an honest error.

            A missing length (absent header or chunked transfer) and a
            non-numeric length are answered explicitly — never silently treated
            as an empty body, which would forward a request the client never
            sent.
            """
            raw = self.headers.get("Content-Length")
            if raw is None:
                self._send(411, _error_bytes(
                    "Content-Length header required", "invalid_request_error"))
                return None
            try:
                length = int(raw)
            except (TypeError, ValueError):
                self._send(400, _error_bytes(
                    f"invalid Content-Length: {raw!r}", "invalid_request_error"))
                return None
            if length < 0:
                self._send(400, _error_bytes(
                    f"invalid Content-Length: {raw!r}", "invalid_request_error"))
                return None
            return length

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._not_found()
                return
            length = self._body_length()
            if length is None:
                return  # already answered with 400/411
            raw = self.rfile.read(length) if length else b""
            try:
                status, body = service.handle_chat(raw)
            except Exception as e:  # last resort: never drop the connection
                self._send(500, _error_bytes(
                    f"internal facade error: {e}", "internal_error"))
                return
            self._send(status, body)

    return FacadeHandler


def build_server(config: FacadeConfig | None = None, host: str = "0.0.0.0",
                 port: int = 8080, **seams) -> ThreadingHTTPServer:
    """Construct (but do not start) a threading HTTP server for a facade."""
    service = FacadeService(config, **seams)
    return ThreadingHTTPServer((host, port), make_handler(service))
