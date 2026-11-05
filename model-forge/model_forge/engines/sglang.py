"""SGLang detection (github.com/sgl-project/sglang).

Distinctive fingerprint: GET /get_model_info. Covers the LLM server and
SGLang Diffusion (docs.sglang.io/docs/sglang-diffusion — `sglang serve`,
TP via --tp-size/--num-gpus, LoRA adapter management API); the diffusion
serving form is detected via the /v1/images/generations route probe.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import EngineStatus, route_exists, root_url

DEFAULT_ROOT = "http://127.0.0.1:30000"
_TIMEOUT_S = 2.0


def _model_info(root: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{root}/get_model_info",
                                    timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def detect(base_url: str = "") -> EngineStatus:
    root = root_url(base_url) or DEFAULT_ROOT
    info = _model_info(root)
    if not isinstance(info, dict):
        return EngineStatus(
            id="sglang",
            display_name="SGLang",
            available=False,
            detail=(f"no SGLang answering at {root}" if base_url
                    else "not running (default port 30000)"),
            modalities=("chat", "vision", "image", "image_edit"),
        )
    model = info.get("model_path") or info.get("model", "")
    diffusion = route_exists(root, "/v1/images/generations")
    modalities = ["chat", "vision"] if not diffusion else \
        ["image", "image_edit", "chat", "vision"]
    return EngineStatus(
        id="sglang",
        display_name="SGLang" + (" Diffusion" if diffusion else ""),
        available=True,
        endpoint=root,
        detail=(f"model {model}" if model else "running")
               + (" | diffusion (images API)" if diffusion else ""),
        modalities=tuple(modalities),
        models_loaded=(model,) if model else (),
    )
