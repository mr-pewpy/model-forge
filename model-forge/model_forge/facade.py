"""OpenAI-compatible route facade — the thin layer consumers keep pointing at.

WHY this is separate from ``broker.py``: the orchestrator owns *policy* (what
weights are resident, when to swap, who waits). The facade owns *routes* — the
HTTP surface that Hermes providers, the image plugin and other local
consumers were written against. Keeping route plumbing apart from
orchestration policy is what lets the orchestrator change without rewriting the
contract that live clients depend on.

This module ports the broker's route shapes: ``GET /health``,
``GET /v1/models``, ``POST /v1/chat/completions``, and the generation surface
(``POST /v1/images/generations`` sync, ``POST /v1/videos`` + ``GET
/v1/videos/{jid}`` + ``/content``, ``POST /v1/videos/generations``, ``GET
/jobs/{jid}``, ``POST /upload``, ``GET /assets/{jid}/{filename}``). It is
stdlib-only (``http.server`` + ``urllib``) because the model-forge test venv
carries no third-party HTTP stack. Everything with a side effect is injectable
— the upstream chat call, the readiness probes, the claim ``gate``, the
orchestrator, the ComfyUI ``runner``, the job store and the async dispatch —
so tests drive the real code paths with fakes and bind no ports.

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
import math
import mimetypes
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .aliases import DEFAULT_ALIASES, resolve

# The canonicalize-then-strip-'gen/' resolver is sibling B2's seam
# (model_forge.aliases.workflow_id_for). It may not be on disk when this module
# loads (the two are written in parallel), so fall back to the identical
# one-liner built from ``resolve`` — the observable contract (canonicalize,
# then drop the 'gen/' prefix to reach the bare runner workflow id) is what
# matters, not which module physically defines it.
try:  # pragma: no cover - exercised by whichever landing order wins
    from .aliases import workflow_id_for
except ImportError:
    def workflow_id_for(model_id: str,
                        aliases: dict[str, str] | None = None) -> str:
        return resolve(model_id, aliases).removeprefix("gen/")

# The orchestrator's claim seam takes a request object exposing ``model_id`` /
# ``engine`` / ``lane`` / ``modality``. broker.LoadRequest is that type, but
# importing broker at module load couples the facade to a file a sibling is
# editing in parallel, so degrade to a structurally-identical local record.
try:  # pragma: no cover - exercised by whichever landing order wins
    from .broker import LoadRequest as _LoadRequest
except Exception:
    @dataclass
    class _LoadRequest:  # type: ignore[no-redef]
        model_id: str
        engine: str
        lane: str = "chat"
        modality: str = "chat"
        weights_path: str | None = None
        estimated_s: float | None = None

# Broker-parity param prep is defined once in the ComfyUI engine — the image
# ``size`` -> aspect/width/height expansion and the video ``seconds``/``size``
# coercion. Reuse those PURE helpers (no upload side effect) so the params the
# facade stores on the job record match the runner's own normalization rather
# than duplicating the size table here. Degrade to None if the engine module is
# unavailable (whichever landing order wins); the facade then normalizes the
# control keys without an aspect label.
try:  # pragma: no cover - exercised by whichever landing order wins
    from .engines.comfyui import (_expand_size as _cf_expand_size,
                                  _expand_video_params as _cf_expand_video)
except Exception:  # pragma: no cover
    _cf_expand_size = None
    _cf_expand_video = None

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


# Broker wording parity for a wrong-kind gen request (broker/main.py ~1663 /
# ~1693): an image route says "not images", a video route "not videos".
_WANT_KIND_LABEL = {"image": "images", "video": "videos"}

# Recorded on a job row when request params diverge from the broker's stored
# shape — see FacadeService._defer_note / _normalize_gen_params.
_DEFER_NOTE = ("image references are uploaded at run time (not at job "
               "creation), so the stored params carry the raw ref, not the "
               "broker's pre-uploaded {'upload': {'path': ...}} shape")

# The chat path's prepare_llm parity tunables. A chat request polls the
# orchestrator's gen-busy state while it waits for in-flight gen jobs to drain;
# the poll interval is small enough to wake promptly and coarse enough not to
# spin. ``_CHAT_ENGINE`` is the engine id a chat claim targets (the vLLM lane).
GEN_WAIT_POLL_S = 0.25
_CHAT_ENGINE = "vllm"


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
    # How long a chat request waits for an in-flight gen job to drain before it
    # wakes the LLM lane (broker ``llm.max_gen_wait_s``, default 200 s < the
    # consumers' 240 s gateway timeout). On exhaustion the request gets an
    # honest 503 rather than a silent stall.
    max_gen_wait_s: float = 200.0
    # -- generation lane --------------------------------------------------
    # Where generated assets live on disk. ``/upload`` writes under
    # ``<assets_dir>/uploads`` and the runner resolves data-/http- image refs
    # there too (broker ``assets_dir``); ``/assets/{jid}/{name}`` serves the
    # runner's returned files. Overridable so tests and deployments point at
    # different volumes without touching the routes.
    assets_dir: str = field(default_factory=lambda: os.environ.get(
        "MODEL_FORGE_ASSETS_DIR",
        os.path.join(os.path.expanduser("~"), ".model-forge", "assets")))
    # The public origin baked into asset URLs the client fetches back
    # (broker ``public_base_url``). It must be the address the CONSUMER can
    # reach, not this process's bind address.
    public_base_url: str = "http://127.0.0.1:8080"
    # Compute target the gen claim/release scope opens against
    # (orchestrator ``claim(scope, target_id, request)``).
    gen_target_id: str = "comfy"
    # Comfy base for the default runner; the production ai-dock lane (broker
    # exposed it on 18188).
    comfy_base_url: str = "http://127.0.0.1:18188"
    # -- topology / runtime overrides (shadow + stub deployments) -----------
    # The container names, health URLs and cold-boot budgets of the two lanes
    # serve.py assembles. Defaults are neutral placeholder names; a
    # shadow/stub deployment points them at throwaway containers so the
    # container-dance mechanics can be exercised without touching the live
    # stack. Defaults are byte-identical to serve.py's former hardcoded
    # TOPOLOGY_ENTRIES, so behaviour is unchanged when nothing overrides them.
    chat_container: str = "chat-engine"
    chat_health_url: str = "http://127.0.0.1:8000/health"
    gen_container: str = "gen-engine"
    gen_health_url: str = "http://127.0.0.1:18188/"
    chat_boot_s: float = 480.0
    gen_boot_s: float = 600.0
    # Absolute base under which the runner resolves collected ComfyUI outputs.
    # The ai-dock image mounts its output dir at /output; a stub deployment
    # points this at a stub-writable dir under assets_dir so /assets serves it.
    comfy_output_dir: str = "/output"
    # ComfyUI args file the serve preflight checks for ``--novram`` (empty =
    # unchecked; env ``COMFY_ARGS_FILE``, FACADE_CONFIG settable). In one real
    # deployment it was silently emptied after a bind-mount change, and every
    # h3 video run OOMd afterwards (the h3 models REQUIRE ``--novram`` on
    # smaller cards). See :func:`comfy_args_state`.
    comfy_args_file: str = ""

    @classmethod
    def from_env(cls, **overrides) -> "FacadeConfig":
        """Build a config; ``llm_api_key`` defaults to ``BROKER_VLLM_API_KEY``.

        Pure construction over ``cls(**overrides)`` — the env lookup lives in
        the ``llm_api_key`` field default, so an ``llm_api_key`` passed here
        (or the dataclass constructor) always wins over the environment.
        """
        return cls(**overrides)


def comfy_args_state(path: str) -> str:
    """``--novram`` verdict for the ComfyUI args file: ``"novram-ok"`` /
    ``"novram-missing"`` / ``"unchecked"``.

    WHY: the h3 video models REQUIRE ``--novram`` on smaller cards, and the
    args file has been silently emptied in a real deployment — every h3 run
    OOMd afterwards and nothing surfaced the drift. serve.py logs this verdict
    at startup (an honest WARNING when the flag is confirmed absent) and
    ``GET /health`` reports it; neither hard-fails.

    Honest states only: no path configured — or a named file that cannot be
    read — is ``"unchecked"`` (the flag can be neither confirmed nor denied;
    the startup log carries the note naming the file). A readable file whose
    args lack the ``--novram`` token is ``"novram-missing"`` (confirmed OOM
    risk); one carrying it is ``"novram-ok"``.
    """
    if not path:
        return "unchecked"
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unchecked"
    return "novram-ok" if "--novram" in text.split() else "novram-missing"


# An upstream call: (raw_body_bytes, headers) -> (status:int, body:bytes).
UpstreamCall = Callable[[bytes, dict], tuple[int, bytes]]
# A readiness probe: () -> value; may raise (fail-soft handled by the caller).
Probe = Callable[[], object]
# A claim gate: (canonical_model_id) -> {'ok': bool, 'reason': str}.
GateCall = Callable[[str], dict]


# ---------------------------------------------------------------------------
# Job store (in-memory) — the record shape the broker's sqlite jobs table held
# ---------------------------------------------------------------------------

class JobStore:
    """Tiny stdlib job ledger mirroring the broker's ``JobStore`` columns.

    Rows carry the exact field set consumers read back through ``/jobs/{jid}``
    (id/kind/model/status/params/asset_path/error/created/updated). It is
    in-memory and process-local on purpose: the facade is a single process and
    the broker's durable table existed only so a restart could resume the
    worker queue — a concern this wave does not carry. Injectable so tests and
    a future durable store share the same route code.
    """

    _COLS = ("id", "kind", "model", "status", "params", "asset_path",
             "error", "note", "created", "updated")

    def __init__(self, clock: Callable[[], float] | None = None):
        self._clock = clock or time.time
        self._jobs: dict[str, dict] = {}

    def create(self, kind: str, model: str, params: dict,
               note: str | None = None) -> str:
        jid = uuid4().hex[:12]
        now = self._clock()
        self._jobs[jid] = {
            "id": jid, "kind": kind, "model": model, "status": "queued",
            "params": params, "asset_path": None, "error": None,
            "note": note, "created": now, "updated": now,
        }
        return jid

    def update(self, jid: str, **fields) -> None:
        row = self._jobs.get(jid)
        if row is None:
            return
        for k, v in fields.items():
            if k in ("status", "asset_path", "error"):
                row[k] = v
        row["updated"] = self._clock()

    def get(self, jid: str) -> dict | None:
        row = self._jobs.get(jid)
        return dict(row) if row is not None else None


def _default_dispatch(fn: Callable[[], None]) -> None:
    """Run a job body on a daemon thread — the async video routes' default.

    A real gen job blocks for minutes (ComfyUI cold boot + sampling), far past
    any HTTP client's read timeout, so submission enqueues and returns
    immediately while a background thread runs the claim/admit/run/release
    body. Daemon so a stuck job never pins process exit. Injectable so tests
    run the body inline (deterministic, no threads) with the same route code.
    """
    threading.Thread(target=fn, daemon=True).start()



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
                 models_advertised: dict[str, dict] | None = None,
                 orchestrator: object | None = None,
                 runner: object | None = None,
                 jobs: JobStore | None = None,
                 dispatch: Callable[[Callable[[], None]], None] | None = None,
                 sleep: Callable[[float], None] | None = None,
                 clock: Callable[[], float] | None = None):
        self.config = config or FacadeConfig.from_env()
        self.aliases = DEFAULT_ALIASES if aliases is None else aliases
        self.models = models_advertised if models_advertised is not None else MODELS_ADVERTISED
        self._upstream = upstream or self._urllib_upstream()
        # ``gate`` is an explicit override kept for tests. In production the
        # chat admission is driven by the orchestrator waiter seam; ``None``
        # means "use the orchestrator when present, otherwise allow".
        self._gate = gate
        self._probes = dict(probes or {})
        # Generation seams. ``orchestrator`` owns claim/release + waiter
        # admission; ``runner`` runs a workflow; ``jobs`` records lifecycles;
        # ``dispatch`` runs an async job body. All injectable so tests drive
        # the real routes with fakes and bind no ports.
        self._orchestrator = orchestrator
        self._runner = runner
        self._jobs = jobs or JobStore()
        self._dispatch = dispatch or _default_dispatch
        # Injectable time seams for the chat wait-for-gen-drain loop, so a test
        # drives it with a fake clock and records the polls it makes.
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic

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
        """Lane readiness, fail-soft (mirrors the broker's /health loosely).

        ``comfy_args`` surfaces the serve preflight's ``--novram`` verdict for
        the ComfyUI args file (see :func:`comfy_args_state`): ``novram-ok`` /
        ``novram-missing`` / ``unchecked``. It is read live so the field can
        never drift from the file on disk.
        """
        llm = bool(self._probe("llm", False))
        comfy = bool(self._probe("comfy", False))
        mode = self._probe("mode", "unknown")
        return {"llm": llm, "comfy": comfy, "mode": mode,
                "comfy_args": comfy_args_state(self.config.comfy_args_file),
                "ok": llm}

    # -- chat passthrough ----------------------------------------------------

    def _chat_admit(self, model_id: str) -> tuple[bool, str, str | None]:
        """Pre-upstream admission: (ok, reason, waiter_token).

        Order of precedence, and WHY:

        * An explicit ``gate`` override (tests) wins — it is the seam the
          wave-1 suite drives.
        * Otherwise, if an orchestrator is wired, ``waiter_enter()`` reserves
          the lane wake. This is the LLM-waiter rule: while a chat request
          waits on a lane wake, that waiter OWNS the wake, and the orchestrator
          (not the route) decides whether it may be admitted. The returned
          token is handed back to ``waiter_exit`` when the request finishes —
          the accounting must survive upstream errors, so the exit lives in a
          ``finally`` (see ``handle_chat``).
        * With neither seam, allow (wave-1 default).

        Fail-closed: a raising or malformed seam yields ``ok=False`` with an
        honest reason, never a dropped connection nor a request that sneaks
        past the claim.
        """
        try:
            if self._gate is not None:
                decision = self._gate(model_id) or {}
                return (bool(decision.get("ok", False)),
                        decision.get("reason") or "lane unavailable", None)
            if self._orchestrator is not None:
                entered = self._orchestrator.waiter_enter() or {}
                ok = bool(entered.get("ok"))
                token = entered.get("token")
                # Fail closed on an ok-without-token seam: ``handle_chat`` only
                # calls ``waiter_exit`` when a token was handed out, so an
                # ``{'ok': True, 'token': None}`` would leak the reservation
                # (the counter never decrements). Treat it as a gate failure.
                if ok and token is None:
                    return (False,
                            entered.get("reason")
                            or "lane admission failed: no waiter token reserved",
                            None)
                return ok, entered.get("reason") or "lane busy", token
            return True, "", None
        except Exception as e:  # broken seam -> honest refusal, fail-closed
            return False, f"gate error: {e}", None

    def _gen_busy(self) -> bool:
        """Whether the orchestrator reports an open generative scope.

        Fail-soft: an unwired or unreadable accessor reads as not-busy, so a
        broken seam never leaves a chat request waiting on a state it cannot
        observe (the caller then wakes the lane as before). The accessor is a
        plain read (``Orchestrator.gen_busy``) that never claims or evicts."""
        if self._orchestrator is None:
            return False
        accessor = getattr(self._orchestrator, "gen_busy", None)
        if accessor is None:
            return False
        try:
            return bool(accessor() if callable(accessor) else accessor)
        except Exception:
            return False

    def _wait_for_gen_drain(self) -> bool:
        """Wait for in-flight gen jobs to drain, bounded by ``max_gen_wait_s``.

        Polls the orchestrator's gen-busy state on the injectable clock/sleep.
        Returns True once no generative scope is open; False when the budget is
        exhausted with one still running (the caller answers an honest 503). A
        running gen job holds the GPU card, so the chat wake must wait for it
        to finish before it can bring the LLM lane up."""
        deadline = self._clock() + float(self.config.max_gen_wait_s)
        while self._gen_busy():
            if self._clock() >= deadline:
                return False
            self._sleep(GEN_WAIT_POLL_S)
        return True

    def handle_chat(self, raw: bytes) -> tuple[int, bytes]:
        """Alias-canonicalize, run prepare_llm parity, forward verbatim.

        prepare_llm parity (the broker's chat path), in order:

        1. reserve the LLM waiter (``waiter_enter``; a refusal is a 503);
        2. wait, bounded by ``max_gen_wait_s``, for any in-flight generative
           job to drain — a gen job holds the card, so the chat wake cannot
           proceed until it finishes (budget exhaustion -> honest 503);
        3. wake the chat lane: ``claim`` the chat lane on the orchestrator,
           which ensures the vLLM server is loaded (a claim refusal is a 503
           with its reason);
        4. forward upstream;
        5. ``release`` the scope with ``restore=False`` so the chat lane STAYS
           UP — the waiter owns the wake, and the lane stays resident until a
           gen entry evicts it.

        The waiter exits exactly once, in the ``finally``, on every path
        (success, upstream raise, claim refusal, budget exhaustion), and any
        scope taken is released there too. The explicit ``gate`` override
        (tests) keeps its wave-1 behavior: it decides admission and no
        orchestrator claim/wait runs.
        """
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

        # Admission seam: refuse BEFORE touching the upstream. Fail-closed.
        ok, reason, token = self._chat_admit(model_id)
        if not ok:
            return 503, _error_bytes(reason, "busy")

        orchestrator = self._orchestrator
        using_orchestrator = orchestrator is not None and self._gate is None
        scope: str | None = None
        claimed = False
        try:
            if using_orchestrator:
                # 2. wait for in-flight gen jobs to drain (bounded).
                if not self._wait_for_gen_drain():
                    return 503, _error_bytes(
                        "gen job in progress and LLM wait budget exhausted "
                        f"(waited {float(self.config.max_gen_wait_s):g}s)",
                        "busy")
                # 3. wake the chat lane: claim -> chat-engine ensure_loaded.
                scope = f"chat-{token}" if token else f"chat-{uuid4().hex[:12]}"
                request = _LoadRequest(model_id=model_id, engine=_CHAT_ENGINE,
                                       lane="chat", modality="chat")
                claim = orchestrator.claim(
                    scope, self.config.gen_target_id, request) or {}
                if not claim.get("ok"):
                    return 503, _error_bytes(
                        claim.get("reason") or "chat lane claim refused",
                        "busy")
                claimed = True

            # 4. canonical -> served rewrite (rename pass; LiteLLM-style id map).
            if self.config.llm_served_id and body.get("model") == self.config.llm_canonical_id:
                body["model"] = self.config.llm_served_id

            out_raw = json.dumps(body).encode()
            headers = {"Content-Type": "application/json"}  # explicit — see module doc
            if self.config.llm_api_key:
                headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
            status, text = self._upstream(out_raw, headers)
            return int(status), text if isinstance(text, bytes) else str(text).encode()
        finally:
            # 5. Release the chat scope WITHOUT restoring prior residency
            # (restore=False): the lane the waiter just woke stays UP until a
            # gen entry evicts it — never stopped and immediately re-started.
            if claimed and scope is not None:
                try:
                    self._orchestrator.release(scope, restore=False)
                except Exception:
                    pass
            # The waiter owns the wake: release the reservation no matter how
            # the request returned (error, timeout, success, refusal) so the
            # lane accounting never leaks a phantom waiter. ``waiter_exit`` is
            # idempotent and never raises by contract; the guard is belt-and-
            # braces against a misbehaving seam.
            if token is not None:
                try:
                    self._orchestrator.waiter_exit(token)
                except Exception:
                    pass


    # -- generation routes ---------------------------------------------------
    #
    # These port the broker's gen surface so the live consumers (Hermes image
    # plugin, video plugin) keep working against model-forge: the request and
    # response shapes, the alias/strip-'gen/' model resolution and the
    # self-hosted /assets serving are reproduced exactly — consumers parse
    # them and were written against the broker. The one thing added on top is
    # the orchestrator: every gen job opens a claim scope, passes the
    # LLM-waiter admission (``admit_gen``), runs the workflow, then releases;
    # a refusal is an honest 503, never a silently dropped job or connection.

    def _json(self, status: int, obj: object) -> tuple[int, bytes]:
        return status, json.dumps(obj).encode()

    def _read_json(self, raw: bytes) -> tuple[dict | None, tuple[int, bytes] | None]:
        try:
            body = json.loads(raw or b"{}")
        except Exception as e:
            return None, (400, _error_bytes(f"invalid JSON body: {e}",
                                            "invalid_request_error"))
        if not isinstance(body, dict):
            return None, (400, _error_bytes("request body must be a JSON object",
                                            "invalid_request_error"))
        return body, None

    def _asset_url(self, jid: str, path: str) -> str:
        base = self.config.public_base_url.rstrip("/")
        return f"{base}/assets/{jid}/{Path(path).name}"

    def _resolve_gen_workflow(self, model_id: str,
                              want_kind: str | None = None
                              ) -> tuple[str | None, tuple[int, bytes] | None]:
        """Resolve a requested model id to a bare runner workflow id.

        ``workflow_id_for`` canonicalizes the alias then strips the ``gen/``
        prefix (the broker's ``aliases.get(id, id).removeprefix('gen/')``). When
        the runner exposes its lane registry the id and (optionally) its
        ``output_kind`` are validated up front — an unknown model is an honest
        404, a wrong-kind request an honest 400, both before any job is created.
        """
        wf_id = workflow_id_for(model_id, self.aliases)
        workflows = getattr(self._runner, "workflows", None) if self._runner else None
        if workflows is None:
            return wf_id, None  # registry unknown (a fake) — let run() judge
        cfg = workflows.get(wf_id)
        if cfg is None:
            return None, (404, _error_bytes(
                f"unknown gen model '{wf_id}'; known: {sorted(workflows)}",
                "invalid_request_error"))
        kind = getattr(cfg, "output_kind", "")
        if want_kind and kind and kind != want_kind:
            want_label = _WANT_KIND_LABEL.get(want_kind, want_kind)
            return None, (400, _error_bytes(
                f"model '{wf_id}' generates {kind}, not {want_label}",
                "invalid_request_error"))
        return wf_id, None

    def _require_runner(self) -> tuple[int, bytes] | None:
        if self._runner is None:
            return 503, _error_bytes(
                "no generation runner configured on this facade", "busy")
        return None

    def _normalize_gen_params(self, body: dict, kind: str | None) -> dict:
        """Normalise request params to the broker's stored shape (route parity).

        Mirrors the broker's per-route param prep WITHOUT the image-upload
        side effect (the broker uploaded refs at route time via
        ``_normalize_image_refs``; here the runner fetches/upload ``image``
        references at run time instead — see :meth:`_defer_note`):

        * ``model`` is always dropped;
        * image lanes (``_openai_image_params``): ``n``/``_timeout`` dropped,
          ``size`` -> ``aspect_ratio`` + ``width``/``height``;
        * video lanes (``_openai_video_params``): ``seconds`` -> float,
          ``size`` -> ``resolution``;
        * the generic job route (broker ``generate``): drops ``model`` only
          (``_timeout`` too, see below);
        * then ``image_url`` -> ``image`` (both helpers, and the broker).

        ``_timeout`` is dropped on every route (video and generic included):
        it is a transport budget (the runner poll deadline), never a workflow
        param, so it must not reach the stored job params or the injection
        map.

        The size/second helpers are the runner's own pure functions, reused so
        the stored params cannot drift from what ``run`` later injects.
        """
        if kind == "video":
            params = {k: v for k, v in body.items()
                      if k not in ("model", "_timeout")}
            try:
                if _cf_expand_video is not None:
                    _cf_expand_video(params)
                elif params.get("size"):
                    params["resolution"] = params["size"]
            except (TypeError, ValueError):
                pass  # non-numeric seconds: run() reports it honestly
        elif kind == "image":
            params = {k: v for k, v in body.items()
                      if k not in ("model", "n", "_timeout")}
            if params.get("size") and _cf_expand_size is not None:
                _cf_expand_size(params)
        else:
            params = {k: v for k, v in body.items()
                      if k not in ("model", "_timeout")}
        if params.get("image_url"):
            params["image"] = params["image_url"]
        params.pop("image_url", None)
        return params

    @staticmethod
    def _has_image_ref(body: dict) -> bool:
        for key in ("image_url", "image"):
            val = body.get(key)
            if isinstance(val, str) and (val.startswith("data:")
                                         or val.startswith("http://")
                                         or val.startswith("https://")):
                return True
        return False

    def _defer_note(self, body: dict) -> str | None:
        """A job-record note when stored params diverge from the broker shape.

        The one remaining divergence is the image-ref upload: the broker
        normalised/uploaded refs at route time, this facade defers that to the
        runner's ``run``. Recorded on the row so the divergence is visible
        (never silently absorbed)."""
        return _DEFER_NOTE if self._has_image_ref(body) else None

    def _request_budget(self, body: dict, default: float | None = None
                        ) -> float | None:
        """The request's ``_timeout`` budget in seconds (float), or ``default``.

        One parse rule for every gen route: ``None``/absent/unparseable falls
        back to ``default`` (the sync image route passes its 1200 s default;
        the async video routes pass ``None`` — no budget, the runner's own
        history window applies, which is exactly today's behavior).
        """
        raw = body.get("_timeout")
        if raw is None:
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        # A non-finite budget (JSON can carry 1e400 -> inf) would blow up
        # Thread.join with OverflowError — treat it like unparseable.
        return value if math.isfinite(value) else default

    def _dispatch_job(self, jid: str, workflow_id: str, params: dict,
                      kind: str,
                      timeout_s: float | None = None) -> None:
        self._dispatch(
            lambda: self._run_gen_job(jid, workflow_id, params, kind,
                                      timeout_s=timeout_s))

    def _run_bounded(self, workflow_id: str, params: dict, run_kwargs: dict,
                     budget: float | None) -> dict:
        """Run the runner under a wall-clock budget; ``budget=None`` runs inline.

        WHY this is the facade's own deadline and not just the runner's
        ``timeout_s`` seam: ``ComfyWorkflowRunner.run`` bounds its poll COUNT,
        and each poll can block on an HTTP request timeout — a stalled or
        black-holed ComfyUI can hold the caller (and the claimed GPU lane) far
        past the request's ``_timeout``. A request budget is a WALL-clock
        promise (the broker blocked its worker for up to ``_timeout`` s before
        answering 504), so the call runs on a daemon worker thread joined for
        the budget and an overrun is an honest 'timed out after Ns' result —
        ``_run_gen_job``'s ``finally`` then releases the claim and frees the
        runner, unpinning the gen lane instead of holding it for the runner's
        3600 s default window. The budget is ALSO handed to ``run`` (the seam
        clamps to its history ceiling) so a well-behaved runner stays bounded
        itself. A daemon worker means an overrunning runner can never pin
        process exit; its late result is discarded (the row is already
        honestly terminal).
        """
        if budget is None:
            return self._runner.run(workflow_id, params, **run_kwargs) or {}
        box: dict = {}

        def _call():
            try:
                box["result"] = self._runner.run(workflow_id, params,
                                                 **run_kwargs)
            except Exception as e:   # noqa: BLE001 — captured into box; the
                # caller reads box["result"]/["exc"]. A timed-out worker's
                # late exception is DISCARDED once the row is terminal.
                box["exc"] = e

        worker = threading.Thread(target=_call, daemon=True)
        worker.start()
        worker.join(max(0.0, float(budget)))
        if worker.is_alive():
            return {"ok": False, "files": [],
                    "reason": (f"timed out after {float(budget):g}s (request "
                               f"_timeout budget expired before the runner "
                               f"reported)")}
        if "exc" in box:
            raise box["exc"]
        return box.get("result") or {}

    def _run_gen_job(self, jid: str, workflow_id: str, params: dict,
                     kind: str,
                     timeout_s: float | None = None) -> tuple[bool, int, str]:
        """Run one gen job: admit -> claim -> run -> release, then free.

        ORDER IS POLICY (roadmap §3, the LLM-waiter rule). ``admit_gen`` runs
        BEFORE ``claim``. The predecessor broker checks ``llm_waiters`` at the TOP of
        ``prepare_gen`` (its own route), before any
        stop/evict or lane wake; ``claim`` is what executes the gen plan
        (evicting the waiting chat resident, waking and loading ComfyUI). So a
        gen job MUST be refused before ``claim`` — claiming first and admitting
        second would let a gen job preempt the very chat request that owns the
        lane wake. A refusal is therefore an honest 503 with NO claim side
        effect (the original claim-then-admit order was wrong; the policy rule
        is the authority).

        Every transition is recorded on the job row so ``/v1/videos/{jid}``,
        ``/jobs/{jid}`` and the sync image route all read one source of truth.
        ``release`` runs in ``finally`` so a failed/completed/raising job never
        leaks a claim (fail-closed scope accounting). ``runner.free()`` also
        runs there: ``ComfyWorkflowRunner.run`` already frees before submit, but
        this facade-side free is a deliberate backstop — an injected/alternate
        runner, or a ``run`` that short-circuits (unknown template, bad params)
        before its own ``free``, would otherwise leave ComfyUI holding the
        previous job's weights and OOM the next one. It runs only once a claim
        was taken, so a refused job touches nothing.

        Returns ``(ok, http_status, reason)``; ``http_status`` is meaningful
        only on failure and lets the SYNC image route answer the right code
        (a policy refusal is a 503, a generation failure a 504)."""
        if self._orchestrator is None:
            self._jobs.update(jid, status="failed",
                              error="no orchestrator configured on this facade")
            return False, 503, "no orchestrator configured on this facade"
        scope = jid
        claimed = False
        try:
            # LLM-waiter admission FIRST: refuse before any claim side effect
            # (claim would evict the waiting chat resident and wake ComfyUI).
            admit = self._orchestrator.admit_gen(workflow_id) or {}
            if not admit.get("ok"):
                reason = (admit.get("reason")
                          or "admission refused: an LLM waiter owns the lane")
                self._jobs.update(jid, status="failed", error=reason)
                return False, 503, reason
            request = _LoadRequest(model_id=workflow_id, engine="comfyui",
                                   lane=kind, modality=kind)
            claim = self._orchestrator.claim(
                scope, self.config.gen_target_id, request) or {}
            if not claim.get("ok"):
                reason = claim.get("reason") or "claim refused"
                self._jobs.update(jid, status="failed", error=reason)
                return False, 503, reason
            claimed = True
            self._jobs.update(jid, status="running")
            run_kwargs: dict = {"assets_dir": self.config.assets_dir,
                                "job_id": jid}
            if timeout_s is not None:
                run_kwargs["timeout_s"] = timeout_s
            result = self._run_bounded(workflow_id, params, run_kwargs,
                                       timeout_s)
            if result.get("ok"):
                files = [str(p) for p in (result.get("files") or [])]
                self._jobs.update(jid, status="done",
                                  asset_path=json.dumps(files))
                return True, 200, ""
            reason = result.get("reason") or "generation failed"
            self._jobs.update(jid, status="failed", error=reason)
            return False, 504, reason
        except Exception as e:  # never let a job body take the process down
            reason = f"generation error: {type(e).__name__}: {e}"
            self._jobs.update(jid, status="failed", error=reason)
            return False, 500, reason
        finally:
            if claimed:
                try:
                    self._orchestrator.release(scope)
                except Exception:
                    pass
                try:
                    self._runner.free()
                except Exception:
                    pass

    def handle_image_generations(self, raw: bytes) -> tuple[int, bytes]:
        """OpenAI-compatible SYNC image generation -> ``{"data":[{"url"}]}``.

        Blocks until the job is terminal: the request's ``_timeout`` (default
        1200 s, broker main.py ~1670) IS the budget, honoured BOTH as the
        runner's poll deadline (clamped to the runner's own
        ``history_timeout_s``) AND as a facade wall-clock deadline (the sync
        route always passes a budget, so every sync image job runs on the
        bounded path and is cut at its budget instead of riding the runner's
        full history window) —
        the broker blocked on its worker for up to ``_timeout``s before
        answering 504. Hermes' image_generate tool and the image plugin read
        exactly this shape. Params are normalized to the broker's stored shape
        before the job is created (drop ``model``/``n``/``_timeout``,
        ``size`` -> width/height, ``image_url`` -> ``image``)."""
        err = self._require_runner()
        if err is not None:
            return err
        body, perr = self._read_json(raw)
        if perr is not None:
            return perr
        wf_id, werr = self._resolve_gen_workflow(str(body.get("model", "")),
                                                 want_kind="image")
        if werr is not None:
            return werr
        # One parse rule for every gen route (incl. non-finite rejection).
        budget = self._request_budget(body or {}, 1200.0)
        params = self._normalize_gen_params(body, kind="image")
        jid = self._jobs.create("gen", wf_id, params,
                                note=self._defer_note(body))
        ok, code, reason = self._run_gen_job(jid, wf_id, params, "image",
                                             timeout_s=budget)
        if not ok:
            err_type = "busy" if code == 503 else "generation_error"
            return code, _error_bytes(f"image job {jid} failed: {reason}",
                                      err_type)
        job = self._jobs.get(jid) or {}
        files = json.loads(job.get("asset_path") or "[]")
        urls = [self._asset_url(jid, p) for p in files]
        return self._json(200, {"created": int(time.time()),
                                "data": [{"url": u} for u in urls]})

    def handle_video_submit(self, raw: bytes) -> tuple[int, bytes]:
        """``POST /v1/videos`` — async submit -> ``{id,status:'queued',model}``.

        The body's optional ``_timeout`` (float seconds, sync image route
        parity) is the job's wall-clock budget: on expiry the row ends failed
        with an honest 'timed out after Ns' reason and the claim is released
        (the gen lane is not pinned for the runner's 3600 s default window).
        Absent ``_timeout`` keeps today's behavior (the runner's own history
        budget)."""
        err = self._require_runner()
        if err is not None:
            return err
        body, perr = self._read_json(raw)
        if perr is not None:
            return perr
        wf_id, werr = self._resolve_gen_workflow(str(body.get("model", "")),
                                                 want_kind="video")
        if werr is not None:
            return werr
        params = self._normalize_gen_params(body, kind="video")
        jid = self._jobs.create("gen", wf_id, params,
                                note=self._defer_note(body))
        self._dispatch_job(jid, wf_id, params, "video",
                           timeout_s=self._request_budget(body))
        return self._json(200, {"id": jid, "status": "queued", "model": wf_id})

    def handle_video_generations(self, raw: bytes) -> tuple[int, bytes]:
        """``POST /v1/videos/generations`` — generic job API (broker shape).

        No output_kind check here (the broker did none): the request is a
        workflow job, and the reply carries the ``poll``/``asset`` pointers a
        client follows. The optional ``_timeout`` budget behaves exactly as on
        ``POST /v1/videos`` (see :meth:`handle_video_submit`)."""
        err = self._require_runner()
        if err is not None:
            return err
        body, perr = self._read_json(raw)
        if perr is not None:
            return perr
        wf_id, werr = self._resolve_gen_workflow(str(body.get("model", "")))
        if werr is not None:
            return werr
        params = self._normalize_gen_params(body, kind=None)
        jid = self._jobs.create("gen", wf_id, params,
                                note=self._defer_note(body))
        self._dispatch_job(jid, wf_id, params, "video",
                           timeout_s=self._request_budget(body))
        return self._json(200, {"id": jid, "status": "queued", "model": wf_id,
                                "poll": f"/jobs/{jid}",
                                "asset": f"/assets/{jid}"})

    def handle_video_status(self, jid: str) -> tuple[int, bytes]:
        """``GET /v1/videos/{jid}`` -> queued/in_progress/completed/failed."""
        job = self._jobs.get(jid)
        if not job:
            return 404, _error_bytes("video job not found", "not_found")
        if job.get("status") == "done":
            files = json.loads(job.get("asset_path") or "[]")
            return self._json(200, {
                "id": jid, "status": "completed",
                "data": [{"url": self._asset_url(jid, p)} for p in files]})
        if job.get("status") == "failed":
            return self._json(200, {"id": jid, "status": "failed",
                                    "error": job.get("error")})
        return self._json(200, {"id": jid, "status": "in_progress"})

    def handle_video_content(self, jid: str) -> tuple[int, bytes, str]:
        """``GET /v1/videos/{jid}/content`` -> first existing asset's bytes."""
        job = self._jobs.get(jid)
        if job and job.get("asset_path"):
            for p in json.loads(job["asset_path"]):
                fp = Path(p)
                if fp.exists():
                    return 200, fp.read_bytes(), _content_type(fp)
        return (404, _error_bytes("video asset not found", "not_found"),
                "application/json")

    def handle_job(self, jid: str) -> tuple[int, bytes]:
        """``GET /jobs/{jid}`` — broker-shaped job record."""
        job = self._jobs.get(jid)
        if not job:
            return 404, _error_bytes("job not found", "not_found")
        resp = {k: job.get(k) for k in ("id", "kind", "model", "status",
                                        "error", "created", "updated")}
        resp["params"] = job.get("params")
        if job.get("asset_path"):
            files = json.loads(job["asset_path"])
            resp["assets"] = [f"/assets/{jid}/{Path(p).name}" for p in files]
        return self._json(200, resp)

    def handle_upload(self, raw: bytes, content_type: str) -> tuple[int, bytes]:
        """``POST /upload`` — store a multipart file under ``assets/uploads``.

        Returns the broker's ``{name, path}``. The runner later resolves a
        data-/http- image ref by reading this file and pushing it through
        ComfyUI's ``/upload/image`` (the nested ``{"upload":{"path"}}`` shape is
        the runner's concern, not this route's)."""
        parsed = _parse_multipart(raw, content_type)
        if parsed is None:
            return 400, _error_bytes(
                "multipart field 'file' (or 'image') with a filename required",
                "invalid_request_error")
        filename, data = parsed
        name = f"{uuid4().hex[:8]}_{Path(filename).name}"
        uploads = Path(self.config.assets_dir) / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        (uploads / name).write_bytes(data)
        return self._json(200, {"name": name, "path": str(uploads / name)})

    def _within_assets(self, path: Path) -> bool:
        """True when ``path`` resolves under ``assets_dir`` (confinement).

        Defence in depth: ``handle_asset`` already matches by the *recorded*
        filename (traversal-safe), but a recorded ``asset_path`` could still
        point anywhere on disk. Refuse to serve anything that resolves outside
        the assets dir — a symlink pointing out, or a job whose file was
        written elsewhere, returns 404 rather than leaking the file."""
        try:
            root = Path(self.config.assets_dir).resolve()
            resolved = path.resolve()
        except OSError:
            return False
        return resolved == root or root in resolved.parents

    def handle_asset(self, jid: str, filename: str) -> tuple[int, bytes, str]:
        """``GET /assets/{jid}/{filename}`` — serve a job's generated file.

        The recorded name must match AND the resolved file must live under the
        assets dir (``_within_assets``)."""
        job = self._jobs.get(jid)
        if job and job.get("asset_path"):
            for p in json.loads(job["asset_path"]):
                fp = Path(p)
                if fp.name == filename and fp.exists() and self._within_assets(fp):
                    return 200, fp.read_bytes(), _content_type(fp)
        return 404, _error_bytes("asset not found", "not_found"), "application/json"


def _error_bytes(message: str, err_type: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": err_type}}).encode()


def _content_type(path: Path) -> str:
    """Best-effort MIME type for a served asset (extension-guessed)."""
    ctype, _ = mimetypes.guess_type(str(path))
    return ctype or "application/octet-stream"


def _parse_multipart(raw: bytes, content_type: str) -> tuple[str, bytes] | None:
    """Extract the first file part from a multipart/form-data body.

    Stdlib-only: ``cgi`` was removed in 3.13 and this repo takes no third-party
    deps, so the boundary is split by hand. Returns ``(filename, data)`` for the
    first part carrying a ``filename=`` in its Content-Disposition (the broker
    accepted a ``file`` or ``image`` field — either is fine here), or ``None``
    when the body is not multipart / carries no file part."""
    if "multipart/form-data" not in (content_type or "") or "boundary=" not in content_type:
        return None
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    if not boundary:
        return None
    delim = b"--" + boundary.encode()
    for chunk in raw.split(delim):
        # Skip the preamble, the closing "--", and delimiter CRLF padding.
        part = chunk.lstrip(b"\r\n")
        if not part or part.startswith(b"--") or b"\r\n\r\n" not in part:
            continue
        head, _, body = part.partition(b"\r\n\r\n")
        if b"filename=" not in head:
            continue
        # The part body ends with a trailing CRLF before the boundary.
        if body.endswith(b"\r\n"):
            body = body[:-2]
        disp = head.decode("utf-8", "replace")
        marker = 'filename="'
        if marker not in disp:
            return None
        filename = disp.split(marker, 1)[1].split('"', 1)[0]
        return Path(filename).name or "upload", body
    return None


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

        def _send(self, status: int, body: bytes,
                  content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _not_found(self) -> None:
            self._send(404, _error_bytes(f"no route for {self.path}", "not_found"))

        def _segments(self) -> list[str]:
            path = self.path.split("?", 1)[0]
            return [p for p in path.split("/") if p]

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            seg = self._segments()
            try:
                if path == "/health":
                    self._send(200, json.dumps(service.health()).encode())
                elif path == "/v1/models":
                    self._send(200, json.dumps(service.catalog()).encode())
                elif len(seg) == 3 and seg[:2] == ["v1", "videos"]:
                    status, body = service.handle_video_status(seg[2])
                    self._send(status, body)
                elif (len(seg) == 4 and seg[:2] == ["v1", "videos"]
                        and seg[3] == "content"):
                    status, body, ctype = service.handle_video_content(seg[2])
                    self._send(status, body, ctype)
                elif len(seg) == 2 and seg[0] == "jobs":
                    status, body = service.handle_job(seg[1])
                    self._send(status, body)
                elif len(seg) == 3 and seg[0] == "assets":
                    status, body, ctype = service.handle_asset(seg[1], seg[2])
                    self._send(status, body, ctype)
                else:
                    self._not_found()
            except Exception as e:  # last resort: never drop the connection
                self._send(500, _error_bytes(
                    f"internal facade error: {e}", "internal_error"))

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
            path = self.path.split("?", 1)[0]
            length = self._body_length()
            if length is None:
                return  # already answered with 400/411
            raw = self.rfile.read(length) if length else b""
            content_type = self.headers.get("Content-Type", "") or ""
            try:
                if path == "/v1/chat/completions":
                    status, body = service.handle_chat(raw)
                elif path == "/v1/images/generations":
                    status, body = service.handle_image_generations(raw)
                elif path == "/v1/videos":
                    status, body = service.handle_video_submit(raw)
                elif path == "/v1/videos/generations":
                    status, body = service.handle_video_generations(raw)
                elif path == "/upload":
                    status, body = service.handle_upload(raw, content_type)
                else:
                    self._not_found()
                    return
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
