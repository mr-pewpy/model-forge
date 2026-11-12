"""Media generation planner — 3D, music, SFX job routing (pure).

These modalities have no native Hermes tool surface (image/video/TTS do), so
Model Forge exposes them through job-style engine APIs:

- 3d:    POST {endpoint}/v1/videos/generations {"model", "image"} -> {"id"}
         -> GET /jobs/{id} -> {status, data:[{url}]}   (media-gateway shape;
         ComfyUI Hunyuan3D/TRELLIS lanes behind it)
- music: POST {endpoint}/v1/audio/music {"model", "prompt", ...} -> audio
- sfx:   POST {endpoint}/v1/audio/speech-or-sfx-style {"model", "prompt"}

Route existence is probed before any submit (route_exists) — a target that
does not serve the modality gets an honest answer instead of a broken call.
"""

from __future__ import annotations

from .engines.base import route_exists, root_url

_ROUTE_FOR_MODALITY = {
    "3d": "/v1/videos/generations",
    "music": "/v1/audio/music",
    "sfx": "/v1/audio/sfx",
}


def media_plan(modality: str, base_url: str) -> dict:
    """What a media generation would do — no side effects (unit-tested)."""
    route = _ROUTE_FOR_MODALITY.get(modality)
    if route is None:
        return {"ok": False, "action": "unknown_modality",
                "detail": f"modality must be one of "
                          f"{sorted(_ROUTE_FOR_MODALITY)}"}
    root = root_url(base_url)
    if not root:
        return {"ok": False, "action": "no_endpoint",
                "detail": "no base_url on this target"}
    if not route_exists(root, route):
        return {"ok": False, "action": "route_missing",
                "detail": f"{root} does not serve {route} — no {modality} "
                          "lane on this endpoint. ComfyUI-native job flow is "
                          "not implemented yet; what works today is a "
                          "media gateway exposing the "
                          "/v1/videos/generations (3d), /v1/audio/music "
                          "(music) and /v1/audio/sfx (sfx) lanes."}
    return {
        "ok": True,
        "action": "submit_job" if modality == "3d" else "submit",
        "submit": root + route,
        "poll": root + "/jobs/{id}" if modality == "3d" else "",
        "detail": ("submit {model, image} then poll /jobs/{id} for the GLB"
                   if modality == "3d" else
                   f"submit {{model, prompt}} to {root}{route}"),
    }
