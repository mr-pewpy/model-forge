"""ComfyUI detection (github.com/comfyanonymous/ComfyUI) + the workflow runner.

Distinctive fingerprint: GET /system_stats. Modalities: diffusion workflows —
image, image edit, video (LTX/MiniMax H3), 3D (Hunyuan3D/TRELLIS).

``detect`` is the non-invasive probe used by ``detect_engines``. The
``ComfyWorkflowRunner`` below is the port of the production weights broker's
ComfyUI lane: it drives generative jobs through JSON workflow templates —
submit ``POST /prompt`` with an injected graph, poll ``GET /history/{id}``,
collect outputs by kind, free the cache between jobs. Templates + the lane
registry live in the ``model_forge.workflows`` package (ported verbatim); this
module is the engine that runs them.

Everything is stdlib-only. The HTTP callable is injectable (``http=``) and the
sleep is injectable (``sleep=``) so tests drive the whole flow with fakes and
never touch the network — the same seam style the broker lifecycle adapters in
``model_forge.broker`` use.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

from .base import EngineStatus, root_url
from ..workflows import (WorkflowCfg, load_template_json,
                         load_templates as _load_templates)

# Re-exported so the pinned public API is reachable as ``comfyui.WorkflowCfg``.
__all__ = ["WorkflowCfg", "ComfyWorkflowRunner", "inject_params", "detect",
           "DEFAULT_ROOT", "DEFAULT_GEN_ROOT"]

DEFAULT_ROOT = "http://127.0.0.1:8188"
# The production ai-dock ComfyUI lane (GPU 1). The broker exposed it on 18188;
# kept here so the ported runner documents the real endpoint it targets.
DEFAULT_GEN_ROOT = "http://127.0.0.1:18188"
_TIMEOUT_S = 2.0

# OpenAI ``size`` -> aspect-ratio labels, ported verbatim from the broker
# (broker/main.py create_app). Unknown sizes fall through as their own label
# and, where ``WxH`` parses, still yield width/height.
_SIZE_TO_ASPECT = {
    "1024x1024": "1:1", "1024x576": "16:9", "576x1024": "9:16",
    "1344x768": "16:9", "768x1344": "9:16", "1280x720": "16:9",
    "720x1280": "9:16", "768x768": "1:1", "832x1216": "3:4",
    "1216x832": "4:3",
}

# Which ComfyUI history output buckets a lane's ``output_kind`` collects.
# image: SaveImage -> ``images``. video: SaveVideo/VideoCombine -> ``videos``
# (newer) or the legacy ``results``/``gifs`` buckets. 3d: SaveGLB -> ``3d``
# (no current lane uses it, but the collector supports it so a future lane
# works without touching the runner).
_OUTPUT_BUCKETS = {
    "image": ("images",),
    "video": ("videos", "results", "gifs"),
    "3d": ("3d",),
}


# ---------------------------------------------------------------------------
# detection (unchanged surface)
# ---------------------------------------------------------------------------

def _get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def detect(base_url: str = "") -> EngineStatus:
    root = root_url(base_url) or DEFAULT_ROOT
    stats = _get_json(f"{root}/system_stats")
    if isinstance(stats, dict) and ("system" in stats or "devices" in stats):
        version = (stats.get("system") or {}).get("comfyui-version", "")
        gpus = [d.get("name", "") for d in stats.get("devices", [])][:3]
        return EngineStatus(
            id="comfyui",
            display_name="ComfyUI",
            available=True,
            endpoint=root,
            detail=(f"comfyui {version}" if version else "running")
                   + (f" | devices: {', '.join(g for g in gpus if g)}"
                      if gpus else ""),
            modalities=("image", "image_edit", "video", "3d"),
        )
    return EngineStatus(
        id="comfyui",
        display_name="ComfyUI",
        available=False,
        detail=(f"no ComfyUI answering at {root}"
                if base_url else
                "not running (default port 8188; start it or add a target "
                "for a remote ComfyUI)"),
        modalities=("image", "image_edit", "video", "3d"),
    )


# ---------------------------------------------------------------------------
# injectable HTTP seam (stdlib default)
# ---------------------------------------------------------------------------

def _encode_multipart(fields: dict, files: list) -> tuple[str, bytes]:
    """Encode a multipart/form-data body from form ``fields`` + ``files``.

    ``files`` is a list of ``(field, filename, content_bytes, content_type)``.
    Stdlib-only so the module has no third-party dependency (the broker used
    httpx's ``files=`` shorthand here)."""
    boundary = "----model-forge-" + uuid4().hex
    out: list[bytes] = []
    for name, val in (fields or {}).items():
        out.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{val}\r\n".encode("utf-8"))
    for field, filename, content, ctype in files:
        out.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode("utf-8"))
        out.append(content)
        out.append(b"\r\n")
    out.append(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", b"".join(out)


def _default_http(method: str, url: str, body=None, *, files=None, raw=False,
                  timeout: float = 10.0):
    """Default injectable HTTP seam: stdlib request/response, JSON by default.

    * JSON request when ``body`` is a dict (and no ``files``);
    * multipart when ``files`` is given (``body`` then carries the form fields);
    * ``raw=True`` returns the response bytes (used to fetch a remote image
      reference) instead of parsing JSON.

    Non-2xx responses raise ``urllib.error.HTTPError`` (stdlib behaviour), so
    the caller can inspect ``.code`` (the 405 cold-boot retry depends on it).
    An empty success body — ComfyUI's ``POST /free`` answers 200 with no body —
    parses to ``{}``, not an error."""
    headers: dict[str, str] = {}
    data = None
    if files:
        ctype, data = _encode_multipart(body, files)
        headers["Content-Type"] = ctype
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        blob = resp.read()
    if raw:
        return blob
    text = blob.decode("utf-8", "replace")
    return json.loads(text) if text.strip() else {}


# ---------------------------------------------------------------------------
# param injection (ported from ComfyBackend._inject + _openai_image_params)
# ---------------------------------------------------------------------------

def _resolve_param_value(value):
    """Unwrap an uploaded image reference to the filename ComfyUI expects.

    Image params are normalised to the NESTED shape ``{"upload": {"path":
    <name>}}`` (see ``_normalize_image_refs``). A node field such as
    ``LoadImage.image`` wants the bare filename, so injection reads
    ``value["upload"]["path"]``. Reading ``value["path"]`` here is the exact
    production bug the broker shipped (the nested key was ignored, so the
    dict was written into the graph and ComfyUI rejected it) — do not
    regress it."""
    if isinstance(value, dict) and "upload" in value:
        return (value.get("upload") or {}).get("path", "")
    return value


def _expand_size(params: dict) -> dict:
    """Port of the broker's ``_openai_image_params`` size handling: an OpenAI
    ``size`` ("1024x1024") adds an ``aspect_ratio`` label and — when it parses
    as ``WxH`` — ``width``/``height``. Unparseable sizes keep the label only;
    the original ``size`` key stays and is silently ignored by injection if no
    lane maps it."""
    size = params.get("size")
    if not size:
        return params
    params["aspect_ratio"] = _SIZE_TO_ASPECT.get(str(size), str(size))
    try:
        w, h = str(size).lower().split("x", 1)
        params["width"], params["height"] = int(w), int(h)
    except (ValueError, TypeError):
        pass
    return params


def _expand_video_params(params: dict) -> dict:
    """Port of the broker's ``_openai_video_params`` (broker/main.py): video
    lanes do NOT expand ``size`` into an aspect label + width/height — they
    coerce ``seconds`` to a float and map the OpenAI ``size`` to the lane's
    ``resolution`` input. This is deliberately different from the image prep
    (``_expand_size``/``_openai_image_params``): the H3 video lanes take a
    ``resolution`` string, not a width/height pair, so matching the broker here
    is what keeps ``size``/``seconds`` landing on the right nodes.

    ``seconds`` coerces to ``float`` because an OpenAI/Sora client may send it
    as a string ("5"); the lane's ``value`` input wants a number. A
    non-numeric value still raises, and ``run`` turns that into an honest
    ``ok=False`` row (never a blind submit).
    """
    if params.get("seconds") is not None:
        params["seconds"] = float(params["seconds"])
    if params.get("size"):
        params["resolution"] = params["size"]
    return params


def _coerce_numeric(value):
    """Coerce a numeric string (e.g. ``"42"``) to ``int`` for seed/steps.

    OpenAI-shaped clients sometimes send ``seed``/``steps`` as strings while
    ComfyUI wants ints; a clean coercion keeps those requests working. A
    non-numeric value is returned unchanged so a genuinely bad value still
    raises (and ``run`` turns that into an honest ``ok=False`` row)."""
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return value
    return value


def inject_params(graph: dict, param_map: dict | None, params: dict) -> dict:
    """Inject ``params`` into an API-format workflow graph, in place.

    Explicit ``param_map`` first — each entry is ``{"node": <id>, "field":
    <input>}``; the value lands on that node's input (created if absent, matching
    the broker). Then a class-based fallback for ``seed``/``steps``/``prompt``
    ONLY, so any workflow works out of the box: walk the graph, fill the first
    node whose inputs already carry ``seed``/``steps``, or the first
    ``CLIPTextEncode`` for ``prompt``. Anything unmapped and not one of those
    three is silently ignored (the broker's behaviour — the caller may send
    OpenAI-shaped keys the lane does not use)."""
    nodes = graph.get("nodes") or graph

    def set_field(node_id, field, value):
        node = nodes[str(node_id)]
        node.setdefault("inputs", {})
        node["inputs"][field] = value

    for param, value in params.items():
        if param in ("seed", "steps"):
            value = _coerce_numeric(value)
        spec = (param_map or {}).get(param)
        if isinstance(spec, dict):
            set_field(spec["node"], spec["field"], _resolve_param_value(value))
            continue
        if param == "seed":
            for nid, node in nodes.items():
                if "seed" in (node.get("inputs") or {}):
                    set_field(nid, "seed", int(value))
                    break
        elif param == "steps":
            for nid, node in nodes.items():
                if "steps" in (node.get("inputs") or {}):
                    set_field(nid, "steps", int(value))
                    break
        elif param == "prompt":
            for nid, node in nodes.items():
                if (node.get("class_type") == "CLIPTextEncode"
                        and "text" in (node.get("inputs") or {})):
                    set_field(nid, "text", str(value))
                    break
    return graph


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------

class ComfyWorkflowRunner:
    """Drive ComfyUI gen lanes through the ported workflow templates.

    Flow per ``run`` (mirrors the broker's GenWorker._process): normalise the
    request params (drop OpenAI control keys, expand ``size``, map
    ``image_url`` -> ``image``, upload image references), inject them into the
    lane's graph, ``POST /prompt`` (retrying the 405 that a cold ai-dock boot
    returns 100% of the time), poll ``GET /history/{id}`` until the execution
    completes or the timeout expires, then collect outputs by ``output_kind``.
    ``free`` is called before submit so back-to-back jobs do not OOM on the
    previous job's cached weights.

    ``alive`` is the caller's pre-flight probe: container-up is not
    service-up, and the broker polled ``alive`` before every gen entry. It is
    deliberately NOT part of ``run`` — during a cold boot ``/system_stats`` can
    lag the ``/prompt`` 405 window, and gating the run on it would defeat the
    retry that exists for exactly that window."""

    def __init__(self, base_url: str, *, http=None, sleep=None,
                 poll_interval_s: float = 2.0, submit_retries: int = 6,
                 retry_sleep_s: float = 10.0, history_timeout_s: float = 3600.0,
                 output_dir: str = "/output", workflows=None,
                 request_timeout_s: float = 60.0, fetch_timeout_s: float = 60.0):
        self.base = (base_url or "").rstrip("/")
        self._http = http or _default_http
        self._sleep = sleep or time.sleep
        self._poll_interval_s = poll_interval_s
        self._submit_retries = max(1, int(submit_retries))
        self._retry_sleep_s = retry_sleep_s
        self._history_timeout_s = history_timeout_s
        # Absolute base under which collected outputs are resolved. ComfyUI's
        # ai-dock image mounts the output dir at /output; override for other
        # deployments. The caller fetches the files from here.
        self._output_dir = output_dir
        self._request_timeout_s = request_timeout_s
        self._fetch_timeout_s = fetch_timeout_s
        self.workflows: dict[str, WorkflowCfg] = {}
        self.load_templates(workflows)

    # -- registry ---------------------------------------------------------

    def load_templates(self, source=None) -> dict[str, WorkflowCfg]:
        """Load the lane registry into ``self.workflows`` and return it.

        ``source`` is forwarded to ``model_forge.workflows.load_templates``:
        ``None`` uses the bundled templates, a directory resolves the bundled
        filenames against it, and a mapping is used verbatim."""
        self.workflows = _load_templates(source)
        return self.workflows

    # -- probes -----------------------------------------------------------

    def alive(self) -> bool:
        """Honest service probe: True only when ComfyUI answers ``/system_stats``
        with a real stats body. Fail-soft — a dead endpoint, timeout or any
        transport error is False, never a raise. A running container that has
        not finished booting the service answers nothing, so this distinguishes
        service-up from container-up."""
        try:
            stats = self._http("GET", f"{self.base}/system_stats", None,
                               timeout=5.0)
        except Exception:  # noqa: BLE001 — a probe never raises
            return False
        return isinstance(stats, dict) and ("system" in stats
                                            or "devices" in stats)

    def free(self) -> dict:
        """``POST /free`` — unload cached weights + free memory. Called between
        back-to-back jobs (ComfyUI holds allocations after a job and reclaims
        them when the queue next goes idle). Returns an honest row; never
        raises."""
        body = {"unload_models": True, "free_memory": True}
        try:
            self._http("POST", f"{self.base}/free", body,
                       timeout=self._request_timeout_s)
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {"ok": False, "cache_freed": False,
                    "reason": "ComfyUI cache free (POST /free) failed: "
                              f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "cache_freed": True,
                "reason": "ComfyUI cache freed (POST /free: unload_models + "
                          "free_memory)"}

    # -- image references -------------------------------------------------

    def _fetch_image_bytes(self, val: str) -> tuple[bytes, str]:
        """Resolve a ``data:`` or ``http(s)`` image reference to ``(bytes,
        ext)``. Remote fetches ride the injectable HTTP seam (``raw=True``) so
        tests fake them without a network."""
        if val.startswith("data:"):
            header, b64 = val.split(",", 1)
            data = base64.b64decode(b64)
            ext = ("png" if "image/png" in header
                   else "jpg" if "image/jpeg" in header
                   else "webp" if "image/webp" in header
                   else "bin")
            return data, ext
        data = self._http("GET", val, None, raw=True,
                          timeout=self._fetch_timeout_s)
        ext = Path(val.split("?")[0]).suffix.lstrip(".") or "bin"
        return data, ext

    def _upload_image(self, data: bytes, filename: str) -> dict:
        ctype = ("image/png" if filename.endswith(".png")
                 else "image/jpeg" if filename.endswith((".jpg", ".jpeg"))
                 else "application/octet-stream")
        return self._http(
            "POST", f"{self.base}/upload/image",
            {"type": "input", "overwrite": "true"},
            files=[("image", filename, data, ctype)],
            timeout=self._request_timeout_s)

    def _normalize_image_refs(self, params: dict, assets_dir) -> dict:
        """Upload any ``data:``/``http(s)`` image reference and rewrite it to
        the NESTED shape ``{"upload": {"path": <comfy input-dir name>}}``.

        ComfyUI's ``LoadImage`` only sees files that live in its input dir, so
        the reference is stashed under ``assets_dir/uploads`` (broker
        behaviour) and pushed through ``POST /upload/image``; the name ComfyUI
        returns is the input-dir path. The nested shape is the contract later
        waves read — index ``["upload"]["path"]``, not ``["path"]`` (that
        mismatch was a production bug; see ``_resolve_param_value``)."""
        for key, val in list(params.items()):
            if not isinstance(val, str) or not val:
                continue
            if not (val.startswith("data:") or val.startswith("http://")
                    or val.startswith("https://")):
                continue
            data, ext = self._fetch_image_bytes(val)
            name = f"{uuid4().hex[:8]}_{key}.{ext}"
            up = Path(assets_dir) / "uploads"
            up.mkdir(parents=True, exist_ok=True)
            saved = up / name
            saved.write_bytes(data)
            resp = self._upload_image(data, name)
            comfy_name = (resp.get("name") if isinstance(resp, dict) else "")
            params[key] = {"upload": {"path": comfy_name or name}}
        return params

    def _prepare_params(self, params: dict, assets_dir,
                        output_kind: str = "image") -> dict:
        """Normalise request params for a lane, matching the broker's OpenAI
        param handling *by output kind*.

        Image lanes follow ``_openai_image_params``: drop ``model``/``n``/
        ``_timeout``, expand ``size`` -> width/height (+ aspect label). Video
        lanes follow ``_openai_video_params``: drop ``model`` only, coerce
        ``seconds`` to float, map ``size`` -> ``resolution``. Both then map
        ``image_url`` -> ``image`` and normalise/upload image references. The
        kind split is the review item — a video request must not be handed
        width/height, and an image request must not be handed a resolution
        label.
        """
        if output_kind == "video":
            p = {k: v for k, v in (params or {}).items() if k != "model"}
            _expand_video_params(p)
        else:
            p = {k: v for k, v in (params or {}).items()
                 if k not in ("model", "n", "_timeout")}
            if p.get("size"):
                _expand_size(p)
        if p.get("image_url"):
            p["image"] = p["image_url"]
        p.pop("image_url", None)
        return self._normalize_image_refs(p, assets_dir)

    # -- submit / poll / collect -----------------------------------------

    def _submit(self, graph: dict, client_id: str) -> tuple[str | None, str]:
        """``POST /prompt`` with the 405 cold-boot retry. Returns
        ``(prompt_id, "")`` on success or ``(None, reason)`` honestly.

        A cold ai-dock boot answers 405 on ``/prompt`` for the first tens of
        seconds (100% of submits fail without this loop — verified in
        production). Retry up to ``submit_retries`` times, sleeping
        ``retry_sleep_s`` between attempts. A connection refusal is a DOWN
        container, not a booting one — reported honestly, never retried."""
        payload = {"prompt": graph, "client_id": client_id}
        for attempt in range(self._submit_retries):
            try:
                resp = self._http("POST", f"{self.base}/prompt", payload,
                                  timeout=self._request_timeout_s)
            except urllib.error.HTTPError as exc:
                if exc.code == 405:
                    # cold ai-dock boot: /prompt 405s until the service is
                    # ready — retry the whole window, then fail honestly.
                    if attempt < self._submit_retries - 1:
                        self._sleep(self._retry_sleep_s)
                        continue
                    return None, ("ComfyUI submit failed after "
                                  f"{self._submit_retries} attempts "
                                  "(405 cold-boot: service not ready)")
                return None, (f"ComfyUI submit failed: HTTP {exc.code} "
                              f"({exc.reason if hasattr(exc, 'reason') else 'error'})")
            except (urllib.error.URLError, OSError) as exc:
                return None, (f"ComfyUI unreachable at {self.base} "
                              f"(container down): {type(exc).__name__}: {exc}")
            # A 200 whose body still carries the 405 boot error (some proxies
            # mask the status): treat it the same as a 405.
            if isinstance(resp, dict) and resp.get("prompt_id"):
                return str(resp["prompt_id"]), ""
            body_text = json.dumps(resp) if not isinstance(resp, str) else resp
            if "405" in body_text:
                if attempt < self._submit_retries - 1:
                    self._sleep(self._retry_sleep_s)
                    continue
                # final attempt with the masked-405 still present: mirror the
                # HTTPError branch's exhaustion reason rather than reporting a
                # generic "unexpected response".
                return None, ("ComfyUI submit failed after "
                              f"{self._submit_retries} attempts "
                              "(405 cold-boot: service not ready)")
            return None, (f"ComfyUI submit failed: unexpected response "
                          f"{body_text[:300]}")
        return None, ("ComfyUI submit failed after "
                      f"{self._submit_retries} attempts (405 cold-boot)")

    def _collect_outputs(self, entry: dict, output_kind: str) -> list[dict]:
        """Collect a finished history entry's outputs for ``output_kind`` into
        flat rows carrying the node id + resolved absolute path. Only the
        buckets for this kind are read, so an image lane never confuses a
        stray ``images`` bucket in another output kind's node."""
        buckets = _OUTPUT_BUCKETS.get(output_kind, ())
        items: list[dict] = []
        for node_id, out in (entry.get("outputs") or {}).items():
            if not isinstance(out, dict):
                continue
            for bucket in buckets:
                for item in (out.get(bucket) or []):
                    if not isinstance(item, dict):
                        continue
                    fn = item.get("filename", "") or ""
                    if not fn:
                        # a bucket item with no filename is not a real output;
                        # skip it so it cannot count toward the done test.
                        continue
                    sub = item.get("subfolder", "") or ""
                    path = os.path.join(self._output_dir, sub, fn)
                    items.append({
                        "node": node_id, "kind": output_kind, "bucket": bucket,
                        "filename": fn, "subfolder": sub,
                        "type": item.get("type", "output"), "path": path,
                    })
        return items

    def _poll_once(self, prompt_id: str, output_kind: str):
        """One ``GET /history/{id}``. Returns ``(status, payload)`` with
        ``status`` in ``running``/``done``/``error``/``down``; ``down`` means
        the service vanished mid-job (an honest failure, not a retry).

        Known drift from the broker's poll (``broker/main.py`` ~288+): the
        broker also consults ``GET /queue`` when ``prompt_id`` is absent from
        history and fails fast on an errored prompt. This port does NOT — a
        prompt that errors *without* ever appearing in history (e.g. rejected
        during validation) leaves this poller reporting ``running`` until the
        full ``history_timeout_s`` expires. Left as-is for this wave; the
        fail-fast option is to query ``GET /queue`` after N empty history
        reads and return ``error`` (or ``down``) when the id is neither queued
        nor running, instead of burning the whole timeout.
        """
        try:
            hist = self._http("GET", f"{self.base}/history/{prompt_id}", None,
                              timeout=self._request_timeout_s)
        except urllib.error.HTTPError:
            return "running", None       # non-200 history read = still running
        except Exception as exc:  # noqa: BLE001 — transport died mid-job
            return "down", f"{type(exc).__name__}: {exc}"
        if not isinstance(hist, dict) or prompt_id not in hist:
            return "running", None
        entry = hist[prompt_id] or {}
        if (entry.get("status") or {}).get("status_str") == "error":
            msgs = (entry.get("status") or {}).get("messages")
            return "error", msgs
        items = self._collect_outputs(entry, output_kind)
        return ("done", items) if items else ("running", None)

    def _empty(self, output_kind, reason, meta=None):
        return {"ok": False, "output_kind": output_kind, "files": [],
                "reason": reason, "meta": meta or {}}

    # -- entry point ------------------------------------------------------

    def run(self, workflow_id: str, params: dict, *, assets_dir, job_id: str,
            timeout_s: float | None = None) -> dict:
        """Run one job end to end on ``workflow_id``.

        Returns ``{"ok", "output_kind", "files", "reason", "meta"}``.
        ``files`` are absolute paths under ComfyUI's output dir (the caller
        fetches them); ``meta`` carries the prompt id, node ids per output and
        the resolved output dir. Never raises: any failure is an honest
        ``ok=False`` with a ``reason``.

        ``timeout_s`` is an optional request budget (the OpenAI ``_timeout``
        the sync image route honours, broker main.py ~1670): the history poll
        deadline becomes ``min(timeout_s, history_timeout_s)`` so a short
        client budget is respected while the instance's own ceiling is still
        enforced. ``None`` keeps the historical ``history_timeout_s`` budget,
        so the parameter is fully backward compatible. An expired budget is an
        honest ``ok=False`` 'timed out' row, never a hang."""
        cfg = self.workflows.get(workflow_id)
        if cfg is None:
            return self._empty("", f"unknown workflow_id {workflow_id!r}; "
                                   f"known lanes: "
                                   f"{', '.join(sorted(self.workflows))}")
        try:
            graph = load_template_json(cfg)
        except Exception as exc:  # noqa: BLE001 — honest error, never a raise
            return self._empty(cfg.output_kind,
                               f"cannot read template for {workflow_id!r}: "
                               f"{type(exc).__name__}: {exc}")
        try:
            prepared = self._prepare_params(params, assets_dir,
                                            cfg.output_kind)
        except Exception as exc:  # noqa: BLE001
            return self._empty(cfg.output_kind,
                               f"invalid request params for {workflow_id!r}: "
                               f"{type(exc).__name__}: {exc}")
        try:
            inject_params(graph, cfg.param_map, prepared)
        except Exception as exc:  # noqa: BLE001 — honest error, never a raise
            return self._empty(cfg.output_kind,
                               f"cannot inject params into workflow "
                               f"{workflow_id!r}: {type(exc).__name__}: {exc}")

        # free cached weights from the previous job before submitting: the
        # next job otherwise OOMs (broker: GenWorker._process).
        self.free()

        prompt_id, err = self._submit(graph, client_id=job_id or "model-forge")
        if prompt_id is None:
            return self._empty(cfg.output_kind, err)

        poll_interval = self._poll_interval_s or 1.0
        budget = self._history_timeout_s
        if timeout_s is not None:
            try:
                budget = min(float(timeout_s), float(self._history_timeout_s))
            except (TypeError, ValueError):
                budget = self._history_timeout_s
        max_polls = max(1, int(budget / poll_interval))
        for _ in range(max_polls):
            self._sleep(self._poll_interval_s)
            status, payload = self._poll_once(prompt_id, cfg.output_kind)
            if status == "done":
                files = [it["path"] for it in payload]
                meta = {
                    "workflow_id": workflow_id, "prompt_id": prompt_id,
                    "client_id": job_id, "output_kind": cfg.output_kind,
                    "output_dir": self._output_dir, "outputs": payload,
                }
                return {"ok": True, "output_kind": cfg.output_kind,
                        "files": files,
                        "reason": f"workflow {workflow_id!r} produced "
                                  f"{len(files)} {cfg.output_kind} output(s)",
                        "meta": meta}
            if status == "error":
                return self._empty(cfg.output_kind,
                                   f"comfyui execution error for {workflow_id!r}: "
                                   f"{str(payload)[:500]}",
                                   {"prompt_id": prompt_id})
            if status == "down":
                return self._empty(cfg.output_kind,
                                   f"ComfyUI went unreachable while polling "
                                   f"{prompt_id}: {payload}",
                                   {"prompt_id": prompt_id})
        return self._empty(cfg.output_kind,
                           f"history poll timed out after "
                           f"{self._history_timeout_s}s for {prompt_id}",
                           {"prompt_id": prompt_id})
