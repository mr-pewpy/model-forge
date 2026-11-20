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

from .aliases import workflow_id_for
from .engines.base import route_exists, root_url

_ROUTE_FOR_MODALITY = {
    "3d": "/v1/videos/generations",
    "music": "/v1/audio/music",
    "sfx": "/v1/audio/sfx",
    # A media gateway that serves the OpenAI images route can take an image
    # request over HTTP (the synchronous POST path). The local ComfyUI runner
    # (runner_plan) takes precedence whenever it is available.
    "image": "/v1/images/generations",
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
        if modality in ("3d", "music", "sfx"):
            detail = (
                f"{root} does not serve {route} — no {modality} lane on "
                "this endpoint. The 3d/music/sfx job flow rides a media "
                "gateway; a plain ComfyUI endpoint does not serve these "
                "lanes — the ComfyUI-native job flow for 3d/music/sfx is "
                "not implemented yet. What works today is a media gateway "
                "exposing the /v1/videos/generations (3d), /v1/audio/music "
                "(music) and /v1/audio/sfx (sfx) lanes.")
        else:
            detail = (
                f"{root} does not serve {route} — no {modality} lane on this "
                "endpoint. The local ComfyUI workflow runner serves the "
                f"{modality} lane when one is available on the target; a "
                "media gateway exposing the OpenAI images route can also take "
                "it over HTTP.")
        return {"ok": False, "action": "route_missing", "detail": detail}
    return {
        "ok": True,
        "action": "submit_job" if modality == "3d" else "submit",
        "submit": root + route,
        "poll": root + "/jobs/{id}" if modality == "3d" else "",
        "detail": ("submit {model, image} then poll /jobs/{id} for the GLB"
                   if modality == "3d" else
                   f"submit {{model, prompt}} to {root}{route}"),
    }


def runner_plan(modality: str, model_id: str, runner, aliases=None) -> dict:
    """What a LOCAL ComfyUI workflow-runner job would do — no side effects.

    The image/video generations a local ComfyUI can serve run through
    ``ComfyWorkflowRunner`` rather than the media-gateway HTTP path. This
    planner decides whether a lane exists for the request before anything is
    submitted:

    * resolve ``model_id`` to a runner workflow_id via
      :func:`model_forge.aliases.workflow_id_for` — this is what makes a
      pre-rename id (``gen/krea2``) land on the right lane
      (``qwen-image-2.1``);
    * the lane must exist in ``runner.workflows``;
    * the lane's ``output_kind`` must match the requested ``modality`` — a
      video request is never submitted to an image lane.

    ``runner=None`` (no local ComfyUI on the target) is an honest
    ``no_runner``; an unknown lane or a kind mismatch is an honest ``ok=False``
    the caller reports instead of submitting.
    """
    if runner is None:
        return {"ok": False, "action": "no_runner",
                "detail": "no ComfyUI workflow runner is available on this "
                          "target — a remote media-gateway keeps its HTTP path"}
    workflow_id = workflow_id_for(model_id, aliases)
    lanes = getattr(runner, "workflows", None) or {}
    cfg = lanes.get(workflow_id)
    if cfg is None:
        return {"ok": False, "action": "unknown_lane",
                "workflow_id": workflow_id,
                "detail": f"no ComfyUI lane {workflow_id!r} for model id "
                          f"{model_id!r}; known lanes: "
                          f"{', '.join(sorted(lanes))}"}
    kind = getattr(cfg, "output_kind", "")
    if kind != modality:
        return {"ok": False, "action": "kind_mismatch",
                "workflow_id": workflow_id, "output_kind": kind,
                "detail": f"lane {workflow_id!r} generates {kind}, "
                          f"not {modality}"}
    return {"ok": True, "action": "run_local", "workflow_id": workflow_id,
            "output_kind": kind,
            "detail": f"run lane {workflow_id!r} through the local ComfyUI "
                      "workflow runner"}
