"""Direct ComfyUI + OpenAI-compatible media execution for Hermes native tools.

Provides a single stdlib-only engine bridge shared by the unified ``model-forge``
package and the standalone ``model-forge-image`` / ``model-forge-video``
packages. When the user points their media endpoint at a raw ComfyUI server
(e.g. ``http://127.0.0.1:8188``), this module auto-detects ComfyUI via
``GET /system_stats`` and drives ``ComfyWorkflowRunner`` directly (reusing the
bundled workflow templates, SSRF-gated image fetches, and root-confined output
collection) — no separate facade daemon required. When the endpoint speaks the
OpenAI ``/v1/images`` or ``/v1/videos`` API (vLLM-Omni, SGLang Diffusion,
Model Forge facade, or any media gateway), requests ride OpenAI HTTP routes.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .aliases import resolve as resolve_alias
from .engines.base import root_url, route_exists
from .engines.comfyui import ComfyWorkflowRunner
from .workflows import load_templates

# Curated catalog id -> bundled ComfyUI workflow lane id (matching
# model_forge.workflows.load_templates() keys).
_CATALOG_TO_LANE = {
    "qwen-image-2.1": "qwen-image-2.1",
    "krea-2-turbo": "qwen-image-2.1",
    "qwen-image-2.1-edit": "qwen-image-2.1-edit",
    "flux-2-klein-4b": "qwen-image-2.1-edit",
    "moss-image": "moss-image",
    "minimax-h3": "h3-t2v",
    "ltx-2.5": "h3-t2v",
}

_EXT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "gif": "image/gif",
}


def _http_get_json(url: str, timeout: float = 4.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
            return data if isinstance(data, dict) else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _http_get_bytes(url: str, timeout: float = 30.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def detect_media_protocol(base_url: str, *, kind: str = "image",
                          get_json=None, has_route=None) -> str:
    """Return ``'comfyui'`` when ``base_url`` is a raw ComfyUI server (answers
    ``/system_stats`` without serving the OpenAI ``/v1`` media route), else
    ``'openai'``."""
    raw = str(base_url or "").strip()
    root = root_url(raw)
    if not root:
        return "openai"
    get_fn = get_json or _http_get_json
    route_fn = has_route or route_exists
    openai_route = ("/v1/videos" if kind == "video"
                    else "/v1/images/generations")
    stats = get_fn(f"{root}/system_stats")
    if isinstance(stats, dict) and ("system" in stats or "devices" in stats):
        if not route_fn(root, openai_route):
            return "comfyui"
    return "openai"


def comfy_lanes_for_kind(kind: str, workflows=None) -> list[str]:
    """Bundled ComfyUI workflow lane ids producing ``kind`` ('image'|'video')."""
    wf = workflows if isinstance(workflows, dict) else load_templates()
    return [
        lane_id for lane_id, cfg in wf.items()
        if getattr(cfg, "output_kind", "") == kind
    ]


def resolve_comfy_lane(model_id: str, *, kind: str = "image",
                       has_image_input: bool = False,
                       workflows=None) -> str:
    """Resolve a user or catalog model id to a bundled ComfyUI workflow lane."""
    wf = workflows if isinstance(workflows, dict) else load_templates()
    raw = str(model_id or "").strip()
    if raw:
        canon = resolve_alias(raw)
        stripped = canon[4:] if canon.startswith("gen/") else canon
        raw_stripped = raw[4:] if raw.startswith("gen/") else raw
        for cand in (raw, canon, stripped, raw_stripped):
            if cand in wf:
                if kind == "video" and has_image_input and cand == "h3-t2v" \
                        and "h3-i2v" in wf:
                    return "h3-i2v"
                if kind == "image" and has_image_input \
                        and cand == "qwen-image-2.1" \
                        and "qwen-image-2.1-edit" in wf:
                    return "qwen-image-2.1-edit"
                return cand
        mapped = (_CATALOG_TO_LANE.get(raw) or _CATALOG_TO_LANE.get(canon)
                  or _CATALOG_TO_LANE.get(stripped))
        if mapped:
            if kind == "video" and has_image_input and mapped == "h3-t2v" \
                    and "h3-i2v" in wf:
                return "h3-i2v"
            if kind == "image" and has_image_input \
                    and mapped == "qwen-image-2.1" \
                    and "qwen-image-2.1-edit" in wf:
                return "qwen-image-2.1-edit"
            if mapped in wf:
                return mapped
    lanes = comfy_lanes_for_kind(kind, wf)
    if not lanes:
        return ""
    if kind == "video" and has_image_input and "h3-i2v" in lanes:
        return "h3-i2v"
    if kind == "image" and has_image_input \
            and "qwen-image-2.1-edit" in lanes:
        return "qwen-image-2.1-edit"
    return lanes[0]


def _read_comfy_output_bytes(output: dict, root: str, *,
                             get_bytes=None) -> tuple[bytes, str, str]:
    """Return ``(raw_bytes, ext, view_url)`` for one ComfyWorkflowRunner output.

    Reads the local file when present; otherwise fetches ComfyUI's own
    ``/view`` endpoint strictly on ``root`` (never follows arbitrary hosts).
    """
    local_path = output.get("path")
    filename = str(output.get("filename") or "")
    ext = (Path(filename).suffix.lstrip(".").lower() if filename else "")
    view_url = str(output.get("view_url") or "")
    if local_path:
        p = Path(str(local_path))
        if p.is_file():
            data = p.read_bytes()
            ext = ext or p.suffix.lstrip(".").lower() or "png"
            return data, ext, view_url
    if not view_url and filename:
        q = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": str(output.get("subfolder") or ""),
            "type": str(output.get("type") or "output"),
        })
        view_url = f"{root}/view?{q}"
    if not view_url:
        raise RuntimeError("ComfyUI returned no readable output path or view URL")
    # Strict same-origin guard: only fetch /view from the configured ComfyUI root.
    if not view_url.startswith(root + "/"):
        raise RuntimeError("refused ComfyUI view URL outside configured root")
    fetch = get_bytes or _http_get_bytes
    data = fetch(view_url)
    ext = ext or "png"
    return data, ext, view_url


def run_comfyui_media(
    base_url: str,
    prompt: str,
    *,
    kind: str = "image",
    model: str = "",
    aspect_ratio: str = "",
    image_url: str = "",
    reference_image_urls: list[str] | None = None,
    runner_factory=None,
    get_bytes=None,
) -> dict[str, Any]:
    """Run an image or video workflow directly against a raw ComfyUI endpoint.

    Returns ``{"ok": True, "bytes": bytes, "ext": str, "mime": str,
    "model": str, "view_url": str, "protocol": "comfyui"}`` or
    ``{"ok": False, "error": str}``. Never raises."""
    root = root_url(base_url)
    if not root:
        return {"ok": False, "error": "ComfyUI base_url is required"}
    try:
        runner = (runner_factory(root) if runner_factory is not None
                  else ComfyWorkflowRunner(root))
        lane = resolve_comfy_lane(
            model, kind=kind,
            has_image_input=bool(image_url or reference_image_urls),
            workflows=runner.workflows,
        )
        if not lane:
            return {"ok": False,
                    "error": f"no bundled ComfyUI workflow template for {kind}"}
        params: dict[str, Any] = {"prompt": str(prompt or "").strip()}
        if aspect_ratio:
            params["aspect_ratio"] = aspect_ratio
        if image_url:
            params["image_url"] = image_url
        if reference_image_urls:
            params["reference_image_urls"] = list(reference_image_urls[:4])
        result = runner.run(lane, params)
        outputs = (result.get("outputs") if isinstance(result, dict)
                   else None) or []
        if not outputs:
            return {"ok": False,
                    "error": f"ComfyUI workflow {lane} produced no {kind} output"}
        raw, ext, view_url = _read_comfy_output_bytes(
            outputs[0], root, get_bytes=get_bytes)
        return {
            "ok": True,
            "bytes": raw,
            "ext": ext,
            "mime": _EXT_TO_MIME.get(ext, "application/octet-stream"),
            "model": lane,
            "view_url": view_url,
            "protocol": "comfyui",
        }
    except Exception as exc:  # noqa: BLE001 — honest provider error
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def openai_generate_image_stdlib(
    base_url: str, prompt: str, *, model: str = "", size: str = "1024x1024",
    extra_body: dict[str, Any] | None = None, timeout: float = 120.0,
) -> dict[str, Any]:
    """Pure-stdlib OpenAI POST /images/generations call (used by the unified
    model-forge package so it needs zero extra pip dependencies)."""
    base = str(base_url or "").rstrip("/")
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": "b64_json",
    }
    if model:
        payload["model"] = model
    if extra_body:
        payload.update(extra_body)
    req = urllib.request.Request(
        f"{base}/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local-endpoint",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    rows = data.get("data") if isinstance(data, dict) else None
    datum = rows[0] if isinstance(rows, list) and rows else {}
    return {
        "b64_json": datum.get("b64_json") or "",
        "url": datum.get("url") or "",
        "model": model or str(data.get("model") or ""),
    }


def openai_generate_video_stdlib(
    base_url: str, prompt: str, *, model: str = "",
    image_url: str = "", aspect_ratio: str = "16:9",
    duration: int | None = None, timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """Pure-stdlib OpenAI video job API (POST /videos -> poll -> /content)."""
    base = str(base_url or "").rstrip("/")
    payload: dict[str, Any] = {"prompt": prompt, "aspect_ratio": aspect_ratio}
    if model:
        payload["model"] = model
    if image_url:
        payload["image_url"] = image_url
    if duration is not None:
        payload["seconds"] = str(duration)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer local-endpoint",
    }
    req = urllib.request.Request(
        f"{base}/videos",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        job = json.loads(resp.read().decode("utf-8", "replace"))
    job_id = str(job.get("id") or "")
    if not job_id:
        raise RuntimeError(f"video endpoint returned no job id: {job!r}")
    deadline = time.monotonic() + timeout
    status = str(job.get("status") or "queued")
    while status not in ("completed", "succeeded", "failed", "cancelled"):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"video job {job_id} timed out")
        time.sleep(poll_interval)
        poll_req = urllib.request.Request(
            f"{base}/videos/{urllib.parse.quote(job_id, safe='')}",
            headers={"Authorization": "Bearer local-endpoint"},
        )
        with urllib.request.urlopen(poll_req, timeout=30.0) as resp:
            job = json.loads(resp.read().decode("utf-8", "replace"))
        status = str(job.get("status") or "")
    if status in ("failed", "cancelled"):
        raise RuntimeError(f"video job {job_id} {status}: {job.get('error')}")
    content_req = urllib.request.Request(
        f"{base}/videos/{urllib.parse.quote(job_id, safe='')}/content",
        headers={"Authorization": "Bearer local-endpoint"},
    )
    with urllib.request.urlopen(content_req, timeout=60.0) as resp:
        raw = resp.read()
    return {"bytes": raw, "ext": "mp4", "model": model or str(job.get("model") or "")}


def encode_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")
