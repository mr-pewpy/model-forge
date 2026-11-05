"""ComfyUI detection (github.com/comfyanonymous/ComfyUI).

Distinctive fingerprint: GET /system_stats. Modalities: diffusion workflows —
image, image edit, video (LTX/MiniMax H3), 3D (Hunyuan3D/TRELLIS). Job flow
(submit /prompt, poll /history) lands with the install/activate tools.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import EngineStatus, root_url

DEFAULT_ROOT = "http://127.0.0.1:8188"
_TIMEOUT_S = 2.0


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
