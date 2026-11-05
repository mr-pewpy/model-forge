"""llama.cpp family detection: the managed server, or any llama-server you run.

llama-server fingerprints via /props (build_info + model fields); Ollama
answers /api/tags instead. Port list mirrors what core probes by default
(:8080) plus the common alternates — plain HTTP, no auth assumed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import EngineStatus

PROBE_PORTS = (8080, 8081, 8000, 8001)
_TIMEOUT_S = 2.0


def _get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def detect() -> EngineStatus:
    for port in PROBE_PORTS:
        props = _get_json(f"http://127.0.0.1:{port}/props")
        if isinstance(props, dict) and ("build_info" in props or "model_path" in props):
            model = props.get("model_path") or props.get("model", "")
            return EngineStatus(
                id="llamacpp",
                display_name="llama.cpp (llama-server)",
                available=True,
                endpoint=f"http://127.0.0.1:{port}",
                detail=f"serving {model}" if model else "running",
                modalities=("chat", "vision"),
                models_loaded=(model,) if model else (),
            )
        tags = _get_json(f"http://127.0.0.1:{port}/api/tags")
        if isinstance(tags, dict) and "models" in tags:
            names = [m.get("name", "") for m in tags.get("models", [])][:3]
            return EngineStatus(
                id="ollama",
                display_name="Ollama",
                available=True,
                endpoint=f"http://127.0.0.1:{port}",
                detail=f"models: {', '.join(names)}" if names else "running",
                modalities=("chat", "vision", "ocr"),
                models_loaded=tuple(n for n in names if n),
            )
    return EngineStatus(
        id="llamacpp",
        display_name="llama.cpp (llama-server)",
        available=False,
        detail="no llama-server on ports 8080/8081/8000/8001 "
               "(the managed runtime starts with Hermes when enabled)",
        modalities=("chat", "vision"),
    )
