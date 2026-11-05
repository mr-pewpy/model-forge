"""Generic OpenAI-compatible engine detection.

Any engine serving GET /v1/models — vLLM, SGLang, LiteLLM, llama-swap, a
weights broker, Strata, Maya's OpenAI path — speaks this API. We list its
models but never claim WHICH engine it is from the response alone (they are
indistinguishable). Engines with a distinctive fingerprint (llama-server
/props, Ollama /api/tags, Maya /api/catalog) are detected by their own
adapters and take precedence at the same endpoint.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import EngineStatus, route_exists, root_url

_TIMEOUT_S = 2.0


def served_models(base_url: str) -> list[str]:
    root = root_url(base_url)
    if not root:
        return []
    try:
        with urllib.request.urlopen(f"{root}/v1/models",
                                    timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return [m.get("id", "") for m in data.get("data", [])][:8]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def detect(base_url: str = "") -> EngineStatus:
    root = root_url(base_url)
    models = served_models(base_url)
    if models:
        modalities = ["chat", "vision", "ocr"]
        if route_exists(root, "/v1/images/generations"):
            modalities += ["image", "image_edit"]
        return EngineStatus(
            id="openai-compatible",
            display_name="OpenAI-compatible endpoint",
            available=True,
            endpoint=root,
            detail=f"serving {', '.join(models)} — engine identity unknown; "
                   "could be vLLM, Strata, llama-swap, a broker, ...",
            modalities=tuple(modalities),
            models_loaded=tuple(m for m in models if m),
        )
    return EngineStatus(
        id="openai-compatible",
        display_name="OpenAI-compatible endpoint",
        available=False,
        detail=(f"no /v1/models answering at {root}" if root
                else "no base_url to probe"),
        modalities=("chat", "vision", "ocr"),
    )
