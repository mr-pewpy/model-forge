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

from .base import EngineStatus, root_url

DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
_TIMEOUT_S = 3.0


def _root(base_url: str = "") -> str:
    return root_url(base_url) or DEFAULT_ENDPOINT


def _req(method: str, path: str, payload: dict | None = None,
         base_url: str = "") -> dict | None:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        _root(base_url) + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


# --- Models-page API (what the desktop Models page drives) -------------------

def catalog(base_url: str = "") -> dict | None:
    """GET /api/catalog — every Maya model with size, plus the pull slot."""
    return _req("GET", "/api/catalog", base_url=base_url)


def loaded(base_url: str = "") -> list[str]:
    """GET /api/ps — names of the models currently loaded."""
    data = _req("GET", "/api/ps", base_url=base_url)
    if not isinstance(data, dict):
        return []
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def show(model: str, base_url: str = "") -> dict | None:
    return _req("POST", "/api/show", {"name": model}, base_url=base_url)


def pull(model: str, detach: bool = True, base_url: str = "") -> dict | None:
    """POST /api/pull — start a download. detach=True returns immediately;
    poll pull_status() (the Models-page pattern)."""
    return _req("POST", "/api/pull", {"name": model, "detach": detach},
                base_url=base_url)


def pull_status(base_url: str = "") -> dict | None:
    return _req("GET", "/api/pull/status", base_url=base_url)


def pull_cancel(base_url: str = "") -> dict | None:
    return _req("POST", "/api/pull/cancel", {}, base_url=base_url)


def delete(model: str, base_url: str = "") -> dict | None:
    """POST /api/delete — removes the model's config and its own files."""
    return _req("POST", "/api/delete", {"name": model}, base_url=base_url)


# --- detection ----------------------------------------------------------------

def detect(base_url: str = "") -> EngineStatus:
    root = _root(base_url)
    cat = catalog(base_url)
    if isinstance(cat, dict) and "models" in cat:
        running = loaded(base_url)
        names = [m.get("name", "") for m in cat.get("models", [])][:3]
        return EngineStatus(
            id="maya",
            display_name="Project Maya",
            available=True,
            endpoint=root,
            detail=f"catalog: {', '.join(n for n in names if n)}"
                   + (f" | loaded: {', '.join(running)}" if running else ""),
            modalities=("chat", "vision", "ocr"),
            models_loaded=tuple(n for n in running if n),
        )
    return EngineStatus(
        id="maya",
        display_name="Project Maya",
        available=False,
        detail=("not running (install: git clone "
                "https://github.com/mw00/project-maya && ./setup.sh — "
                "serves OpenAI-compatible /v1 on :8080 and auto-swaps models "
                "on request)") if not base_url else
               f"no Maya answering at {root}",
        modalities=("chat", "vision", "ocr"),
    )
