"""Project Maya adapter (github.com/mw00/project-maya).

Maya runs GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 and MiniMax-M3.1-Flash on
1-16 GPUs and speaks two APIs: OpenAI-compatible at /v1, and an Ollama-style
model-management API — /api/tags, /api/ps, /api/show, /api/pull (+ status,
cancel), /api/delete, /api/catalog, /api/version. A request naming a model
that is not loaded auto-swaps to it. Detection keys on /api/catalog, which
plain Ollama does not serve.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import EngineStatus

DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
_TIMEOUT_S = 3.0


def _req(method: str, path: str, payload: dict | None = None) -> dict | None:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        DEFAULT_ENDPOINT + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


# --- Models-page API (what the desktop Models page drives) -------------------

def catalog() -> dict | None:
    """GET /api/catalog — every Maya model with size, plus the pull slot."""
    return _req("GET", "/api/catalog")


def loaded() -> list[str]:
    """GET /api/ps — names of the models currently loaded."""
    data = _req("GET", "/api/ps")
    if not isinstance(data, dict):
        return []
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def show(model: str) -> dict | None:
    return _req("POST", "/api/show", {"name": model})


def pull(model: str, detach: bool = True) -> dict | None:
    """POST /api/pull — start a download. detach=True returns immediately;
    poll pull_status() (the Models-page pattern)."""
    return _req("POST", "/api/pull", {"name": model, "detach": detach})


def pull_status() -> dict | None:
    return _req("GET", "/api/pull/status")


def pull_cancel() -> dict | None:
    return _req("POST", "/api/pull/cancel", {})


def delete(model: str) -> dict | None:
    """POST /api/delete — removes the model's config and its own files."""
    return _req("POST", "/api/delete", {"name": model})


# --- detection ----------------------------------------------------------------

def detect() -> EngineStatus:
    cat = catalog()
    if isinstance(cat, dict) and "models" in cat:
        running = loaded()
        names = [m.get("name", "") for m in cat.get("models", [])][:3]
        return EngineStatus(
            id="maya",
            display_name="Project Maya",
            available=True,
            endpoint=DEFAULT_ENDPOINT,
            detail=f"catalog: {', '.join(n for n in names if n)}"
                   + (f" | loaded: {', '.join(running)}" if running else ""),
            modalities=("chat", "vision", "ocr"),
            models_loaded=tuple(n for n in running if n),
        )
    return EngineStatus(
        id="maya",
        display_name="Project Maya",
        available=False,
        detail="not running (install: git clone "
               "https://github.com/mw00/project-maya && ./setup.sh — "
               "serves OpenAI-compatible /v1 on :8080 and auto-swaps models "
               "on request)",
        modalities=("chat", "vision", "ocr"),
    )
