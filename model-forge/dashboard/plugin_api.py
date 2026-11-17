"""Dashboard backend routes for Model Forge (mounted at
/api/plugins/model-forge/).

Read-only endpoints: the cockpit snapshot — exactly the data behind the
/model-forge command and the Desktop cockpit pane (compute targets with
hardware and engines, model fit rows, installed model storage) — and the
on-demand resource snapshot (/resources), which runs only when the tab's
collapsed Resource view is expanded (nothing polls or updates on its own).
Read-only by design: mutating verbs are the plugin's tools, which ride the
human-approval gate. Every read is bounded and fails soft to a stated
reason. Each request performs reads only — the plugin's config/settings
reads and the read-only engine probes behind the cockpit snapshot (the same
probes ``model_forge_status`` runs) — and starts no background jobs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi import APIRouter

# The plugin's own package (model_forge) sits next to this dashboard folder;
# the dashboard host imports this file standalone, so add the plugin root to
# the import path once, guarded and fail-soft.
_PLUGIN_ROOT = str(Path(__file__).resolve().parent.parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

router = APIRouter()

_SNAPSHOT_TIMEOUT_S = 25.0


def _snapshot() -> dict:
    try:
        from model_forge import tools
        return tools.cockpit_data()
    except Exception as exc:  # fail soft: the reason is the payload
        return {"ok": False,
                "error": f"cockpit snapshot unavailable: "
                         f"{type(exc).__name__}: {exc}"}


@router.get("/cockpit")
async def cockpit():
    """The read-only cockpit snapshot (targets, engines, models, storage)."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_snapshot), timeout=_SNAPSHOT_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {"ok": False,
                "error": f"cockpit snapshot timed out after "
                         f"{_SNAPSHOT_TIMEOUT_S:g} s (a probe or engine call "
                         "is not answering); nothing was changed — refresh "
                         "to retry"}
    except Exception as exc:  # fail soft: the reason is the payload
        return {"ok": False,
                "error": f"cockpit snapshot failed: "
                         f"{type(exc).__name__}: {exc}"}


def _resources() -> dict:
    try:
        from model_forge import tools
        return tools.cockpit_resources()
    except Exception as exc:  # fail soft: the reason is the payload
        return {"ok": False,
                "error": f"resource snapshot unavailable: "
                         f"{type(exc).__name__}: {exc}"}


@router.get("/resources")
async def resources():
    """The on-demand resource snapshot (memory usage rows, resident model
    blocks). Read-only and quiet: called only when the tab's collapsed
    Resource view is expanded or its manual refresh is clicked —
    nothing polls this route."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_resources), timeout=_SNAPSHOT_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {"ok": False,
                "error": f"resource snapshot timed out after "
                         f"{_SNAPSHOT_TIMEOUT_S:g} s (a probe or engine call "
                         "is not answering); nothing was changed — refresh "
                         "to retry"}
    except Exception as exc:  # fail soft: the reason is the payload
        return {"ok": False,
                "error": f"resource snapshot failed: "
                         f"{type(exc).__name__}: {exc}"}
