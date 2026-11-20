"""ComfyUI workflow templates + the gen-lane registry.

The production weights broker on .62 drove every generative lane through a
ComfyUI (ai-dock, GPU 1) *workflow template*: a JSON graph in ComfyUI's API
format plus a small config that says which node/field each request parameter
lands on. Model Forge ports that lane here so the same seven lanes keep
working once the broker is replaced.

This package holds only data + the registry: the seven template JSONs are
copied VERBATIM from the broker's ``workflows/`` directory (byte-identical,
sha256-checked during the port) and ``WORKFLOWS`` mirrors the broker's
``config.yaml`` ``workflows:`` block. Keeping the templates as package files
(not paths into another checkout) means the runner is self-contained: it does
not need the broker tree to exist at runtime.

``WorkflowCfg`` lives here rather than in ``engines.comfyui`` because the
registry is the thing that carries it; ``engines.comfyui`` re-exports it so
the pinned public API (``comfyui.WorkflowCfg``) still holds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Directory holding the template JSONs (this package's own directory). The
# templates are package data, so the default source can never drift from the
# code that reads them.
TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class WorkflowCfg:
    """One generative lane: a ComfyUI template + how to inject parameters.

    ``output_kind`` selects which ComfyUI history output buckets the runner
    collects after execution: ``image`` (SaveImage -> ``images``), ``video``
    (SaveVideo -> ``videos``/``results``), or ``3d`` (SaveGLB -> ``3d``).
    ``param_map`` maps a request parameter name to ``{"node": <id>,
    "field": <input name>}``; anything not in the map falls back to the
    class-based defaults (seed/steps/prompt) inside the runner.
    """

    workflow_id: str
    template: str
    output_kind: str
    param_map: dict


# The seven lanes as configured in the broker's config.yaml `workflows:` block,
# in the same order. `template` is the bare filename; `load_templates` resolves
# it against a real directory. Keys here are the EXACT ids the broker exposed
# as model ids, so clients that send `model="h3-t2v"` keep working.
WORKFLOWS: dict[str, WorkflowCfg] = {
    "h3-t2v": WorkflowCfg(
        workflow_id="h3-t2v",
        template="h3_t2v.json",
        output_kind="video",
        param_map={
            "prompt": {"node": "104", "field": "prompt"},
            "seed": {"node": "15", "field": "noise_seed"},
            "steps": {"node": "9", "field": "steps"},
            "seconds": {"node": "111", "field": "value"},
        },
    ),
    "h3-i2v": WorkflowCfg(
        workflow_id="h3-i2v",
        template="h3_i2v.json",
        output_kind="video",
        param_map={
            "prompt": {"node": "104", "field": "prompt"},
            "seed": {"node": "15", "field": "noise_seed"},
            "steps": {"node": "9", "field": "steps"},
            "seconds": {"node": "111", "field": "value"},
            "image": {"node": "140", "field": "image"},
        },
    ),
    "h3-r2v": WorkflowCfg(
        workflow_id="h3-r2v",
        template="h3_r2v.json",
        output_kind="video",
        param_map={
            "prompt": {"node": "136", "field": "prompt"},
            "seed": {"node": "15", "field": "noise_seed"},
            "steps": {"node": "9", "field": "steps"},
            "seconds": {"node": "111", "field": "value"},
            "image": {"node": "140", "field": "image"},
        },
    ),
    "qwen-image-2.1": WorkflowCfg(
        workflow_id="qwen-image-2.1",
        template="qwen21_t2i.json",
        output_kind="image",
        param_map={
            "prompt": {"node": "5", "field": "prompt"},
            "seed": {"node": "8", "field": "seed"},
            "steps": {"node": "8", "field": "steps"},
            "cfg": {"node": "8", "field": "cfg"},
            "width": {"node": "7", "field": "width"},
            "height": {"node": "7", "field": "height"},
        },
    ),
    "qwen-image-2.1-edit": WorkflowCfg(
        workflow_id="qwen-image-2.1-edit",
        template="qwen21_edit.json",
        output_kind="image",
        param_map={
            "prompt": {"node": "6", "field": "prompt"},
            "seed": {"node": "9", "field": "seed"},
            "steps": {"node": "9", "field": "steps"},
            "cfg": {"node": "9", "field": "cfg"},
            "image": {"node": "4", "field": "image"},
            "resolution": {"node": "7", "field": "resolution"},
        },
    ),
    "qwen-image-2.1-cutout": WorkflowCfg(
        workflow_id="qwen-image-2.1-cutout",
        template="qwen21_edit_nope.json",
        output_kind="image",
        param_map={
            "prompt": {"node": "7", "field": "prompt"},
            "seed": {"node": "9", "field": "seed"},
            "steps": {"node": "9", "field": "steps"},
            "cfg": {"node": "9", "field": "cfg"},
            "image": {"node": "4", "field": "image"},
            "resolution": {"node": "7", "field": "resolution"},
        },
    ),
    "moss-image": WorkflowCfg(
        workflow_id="moss-image",
        template="moss_image_t2i.json",
        output_kind="image",
        param_map={
            "prompt": {"node": "4", "field": "text"},
            "seed": {"node": "7", "field": "seed"},
            "steps": {"node": "7", "field": "steps"},
            "cfg": {"node": "7", "field": "cfg"},
            "width": {"node": "6", "field": "width"},
            "height": {"node": "6", "field": "height"},
        },
    ),
}


def load_templates(source=None) -> dict[str, WorkflowCfg]:
    """Return the lane registry with each ``template`` resolved to a path.

    ``source`` may be:
      * ``None`` (default) — this package's own directory (the bundled,
        verbatim-ported templates);
      * a directory path — resolve the bundled template *filenames* against
        it (lets an operator point at the live broker's ``workflows/`` if it
        ever diverges);
      * a ``{workflow_id: WorkflowCfg}`` mapping — used verbatim (tests and
        callers that build their own lane set).

    A missing template file is NOT fatal here: the cfg is returned with its
    resolved path and the runner reports an honest error only when that lane
    is actually run. That keeps importing the registry (and listing lanes)
    free of local-filesystem assumptions.
    """
    if isinstance(source, dict):
        return dict(source)
    base = Path(source) if source else Path(TEMPLATE_DIR)
    out: dict[str, WorkflowCfg] = {}
    for wid, cfg in WORKFLOWS.items():
        out[wid] = WorkflowCfg(
            workflow_id=cfg.workflow_id,
            template=str(base / cfg.template),
            output_kind=cfg.output_kind,
            param_map=dict(cfg.param_map),
        )
    return out


def load_template_json(cfg: WorkflowCfg) -> dict:
    """Read a lane's template JSON (API format). Accepts either a
    ``{nodes: {...}}`` wrapper or a flat ``{node_id: node}`` map (the broker
    accepted both; the bundled templates are flat)."""
    with open(cfg.template, encoding="utf-8") as fh:
        return json.load(fh)
