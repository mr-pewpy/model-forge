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
