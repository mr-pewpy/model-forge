"""vLLM detection (github.com/vllm-project/vllm).

Distinctive fingerprint: GET /version. vLLM also serves text/vision/OCR
models; with --omni it serves diffusion models too (vLLM-Omni: OpenAI images
API), detected via the /v1/images/generations route probe.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import EngineStatus, route_exists, root_url

DEFAULT_ROOT = "http://127.0.0.1:8000"
_TIMEOUT_S = 2.0


def _version(root: str) -> str:
    try:
        with urllib.request.urlopen(f"{root}/version",
                                    timeout=_TIMEOUT_S) as resp:
            return str(json.loads(resp.read().decode("utf-8", "replace"))
                       .get("version", ""))
    except (urllib.error.URLError, OSError, ValueError):
        return ""


def _models(root: str) -> list[str]:
    try:
        with urllib.request.urlopen(f"{root}/v1/models",
                                    timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return [m.get("id", "") for m in data.get("data", [])][:8]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def detect(base_url: str = "") -> EngineStatus:
    root = root_url(base_url) or DEFAULT_ROOT
    version = _version(root)
    if not version:
        return EngineStatus(
            id="vllm",
            display_name="vLLM",
            available=False,
            detail=(f"no vLLM answering at {root}" if base_url
                    else "not running (default port 8000)"),
            modalities=("chat", "vision", "ocr", "image", "image_edit"),
        )
    models = _models(root)
    omni = route_exists(root, "/v1/images/generations")
    modalities = ["chat", "vision", "ocr"]
    if omni:
        modalities += ["image", "image_edit"]
    return EngineStatus(
        id="vllm",
        display_name="vLLM" + ("-Omni" if omni else ""),
        available=True,
        endpoint=root,
        detail=f"vllm {version}"
               + (" | omni diffusion (images API)" if omni else "")
               + (f" | serving {', '.join(models)}" if models else ""),
        modalities=tuple(modalities),
        models_loaded=tuple(m for m in models if m),
    )
