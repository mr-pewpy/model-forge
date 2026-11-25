"""llama.cpp family detection: the managed server, or any llama-server you run.

llama-server fingerprints via /props (build_info + model fields); Ollama
answers /api/tags instead. Port list mirrors what core probes by default
(:8080) plus the common alternates — and Ollama's stock default port
(11434), which must be scanned or a plain Ollama install is invisible —
plain HTTP, no auth assumed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import EngineStatus, root_url

PROBE_PORTS = (8080, 8081, 8000, 8001, 11434)
_TIMEOUT_S = 2.0


def _get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _probe_root(root: str) -> EngineStatus | None:
    """One engine root: llama-server via /props, Ollama via /api/tags."""
    props = _get_json(f"{root}/props")
    if isinstance(props, dict) and ("build_info" in props or "model_path" in props):
        model = props.get("model_path") or props.get("model", "")
        return EngineStatus(
            id="llamacpp",
            display_name="llama.cpp (llama-server)",
            available=True,
            endpoint=root,
            detail=f"serving {model}" if model else "running",
            modalities=("chat", "vision"),
            models_loaded=(model,) if model else (),
        )
    tags = _get_json(f"{root}/api/tags")
    if isinstance(tags, dict) and "models" in tags:
        names = [m.get("name", "") for m in tags.get("models", [])][:3]
        return EngineStatus(
            id="ollama",
            display_name="Ollama",
            available=True,
            endpoint=root,
            detail=f"models: {', '.join(names)}" if names else "running",
            modalities=("chat", "vision", "ocr"),
            models_loaded=tuple(n for n in names if n),
        )
    return None


OLLAMA_DEFAULT_ROOT = "http://127.0.0.1:11434"


def ollama_endpoint(base_url: str = "") -> str:
    """Resolve an Ollama root URL: explicit base_url, else a live local
    Ollama instance on PROBE_PORTS, else the default port 11434."""
    if base_url:
        return root_url(base_url)
    for port in (11434, 8080, 8081, 8000, 8001):
        root = f"http://127.0.0.1:{port}"
        tags = _get_json(f"{root}/api/tags")
        if isinstance(tags, dict) and "models" in tags:
            return root
    return OLLAMA_DEFAULT_ROOT


def ollama_pull_stream(model: str, base_url: str = "", *, on_progress=None,
                       timeout: float = 300.0) -> dict:
    """POST /api/pull on an Ollama endpoint and consume NDJSON progress lines.

    Reports honest status/byte progress via ``on_progress(dict)`` as Ollama
    streams layers. Returns ``{"ok": True, ...}`` on completion or
    ``{"ok": False, "error": ...}`` on failure — never raises."""
    name = str(model or "").strip()
    if not name:
        return {"ok": False, "error": "model name is required"}
    root = root_url(base_url) or OLLAMA_DEFAULT_ROOT
    body = json.dumps({"name": name, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{root}/api/pull", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    last_status = ""
    done_bytes = 0
    total_bytes = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(frame, dict):
                    continue
                if frame.get("error"):
                    return {"ok": False, "model": name, "endpoint": root,
                            "done_bytes": done_bytes,
                            "total_bytes": total_bytes,
                            "error": str(frame["error"])}
                if frame.get("status"):
                    last_status = str(frame["status"])
                if isinstance(frame.get("completed"), (int, float)):
                    done_bytes = int(frame["completed"])
                if isinstance(frame.get("total"), (int, float)):
                    total_bytes = int(frame["total"])
                if callable(on_progress):
                    on_progress({
                        "status_text": last_status,
                        "done_bytes": done_bytes,
                        "total_bytes": total_bytes,
                    })
        return {"ok": True, "model": name, "endpoint": root,
                "status_text": last_status or "success",
                "done_bytes": done_bytes, "total_bytes": total_bytes}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "model": name, "endpoint": root,
                "done_bytes": done_bytes, "total_bytes": total_bytes,
                "error": f"{type(exc).__name__}: {exc}"}


def detect(base_url: str = "") -> EngineStatus:
    if base_url:
        root = root_url(base_url)
        return _probe_root(root) or EngineStatus(
            id="llamacpp",
            display_name="llama.cpp (llama-server)",
            available=False,
            detail=f"no llama-server or Ollama answering at {root}",
            modalities=("chat", "vision"),
        )
    for port in PROBE_PORTS:
        found = _probe_root(f"http://127.0.0.1:{port}")
        if found:
            return found
    return EngineStatus(
        id="llamacpp",
        display_name="llama.cpp (llama-server)",
        available=False,
        detail="no llama-server or Ollama on ports "
               + "/".join(str(p) for p in PROBE_PORTS)
               + " (the managed runtime starts with Hermes when enabled)",
        modalities=("chat", "vision"),
    )
