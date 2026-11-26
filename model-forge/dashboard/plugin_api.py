"""Dashboard and unified Desktop backend routes for Model Forge
(mounted at /api/plugins/model-forge/).

Read-only, bounded, fail-soft endpoints serving both the Hermes web dashboard
tab and the unified Desktop UI (``ctx.rest(...)``) — eliminating browser CORS
failures and removing any chat-session requirement:

- GET /cockpit — full cockpit snapshot (targets, GPU hardware, declared +
  detected engines, curated model fit rows, installed storage, residency,
  credential presence booleans, active Hermes wiring, engine catalog).
- GET /browse — live model catalog + optional Hugging Face keyword search
  priced against any selected compute target (``modality``, ``target``,
  ``engine``, ``query``, ``fresh``).
- POST /probe — read-only Python engine probe + ComfyUI VRAM budget suggestion
  for a user-supplied candidate ``base_url`` (bypasses browser CORS; refuses
  cloud metadata hosts).
- GET /jobs — recent download / Ollama pull / vendor-installer job records.
- GET /engines/local-cli — read-only PATH availability and managed-PID status
  for local inference engine CLIs.
- GET /residency — broker weight-residency rows.
- GET /resources — on-demand resource snapshot (called only when the collapsed
  Resource view is expanded or manually refreshed; nothing polls this route).

Mutating verbs stay approval-gated plugin tools (``model_forge_target_add``,
``model_forge_install``, ``model_forge_activate``, ``model_forge_storage``,
``model_forge_broker``).
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body

# The plugin's own package (model_forge) sits next to this dashboard folder;
# the dashboard host imports this file standalone, so add the plugin root to
# the import path once, guarded and fail-soft.
_PLUGIN_ROOT = str(Path(__file__).resolve().parent.parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

router = APIRouter()

_SNAPSHOT_TIMEOUT_S = 25.0
_BLOCKED_PROBE_HOSTS = frozenset({
    "169.254.169.254",
    "169.254.170.2",
    "169.254.169.253",
    "100.100.100.200",
    "metadata.google.internal",
    "metadata.goog",
})


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


def _browse(modality: Optional[str], target: Optional[str],
            engine: Optional[str], query: Optional[str],
            fresh: Optional[str]) -> dict:
    try:
        from model_forge import tools
        args: dict = {}
        if modality and modality != "all":
            args["modality"] = modality
        if target:
            args["target"] = target
        if engine:
            args["engine"] = engine
        if query:
            args["query"] = query
        if fresh is not None:
            args["fresh"] = fresh
        raw = tools.model_forge_browse(args)
        data = json.loads(raw)
        if isinstance(data, dict):
            data.setdefault("ok", "error" not in data)
            return data
        return {"ok": False, "error": "unexpected browse payload"}
    except Exception as exc:  # noqa: BLE001 — fail soft
        return {"ok": False,
                "error": f"browse unavailable: {type(exc).__name__}: {exc}"}


@router.get("/browse")
async def browse(modality: Optional[str] = None, target: Optional[str] = None,
                 engine: Optional[str] = None, query: Optional[str] = None,
                 fresh: Optional[str] = "false"):
    """Read-only model catalog + optional live Hugging Face search priced
    against the chosen target (default fresh=false for fast offline paging;
    pass fresh=true when the user searches Hugging Face)."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_browse, modality, target, engine, query, fresh),
            timeout=_SNAPSHOT_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {"ok": False,
                "error": f"browse timed out after {_SNAPSHOT_TIMEOUT_S:g} s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "error": f"browse failed: {type(exc).__name__}: {exc}"}


def _probe_endpoint(base_url: str) -> dict:
    url = str(base_url or "").strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return {"ok": False,
                "error": "base_url must be a valid http:// or https:// URL"}
    if parsed.username or parsed.password:
        return {"ok": False, "error": "URL userinfo is not allowed"}
    if parsed.hostname.lower() in _BLOCKED_PROBE_HOSTS:
        return {"ok": False, "error": "metadata hosts cannot be probed"}
    try:
        from model_forge import tools
        engines = [e for e in tools._target_engine_rows(url) if e.get("available")]
        suggestion = tools._suggest_budgets(url)
        return {
            "ok": True,
            "base_url": url.rstrip("/"),
            "engines": engines,
            "budget_suggestion": suggestion,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "error": f"probe failed: {type(exc).__name__}: {exc}"}


@router.post("/probe")
async def probe(payload: dict = Body(default={})):
    """Read-only backend probe of a candidate target or engine base_url
    (bypasses browser CORS and auto-suggests ComfyUI VRAM/RAM)."""
    base_url = str((payload or {}).get("base_url") or "")
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_probe_endpoint, base_url), timeout=15.0)
    except asyncio.TimeoutError:
        return {"ok": False, "error": "endpoint probe timed out after 15 s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "error": f"probe failed: {type(exc).__name__}: {exc}"}


def _jobs() -> dict:
    try:
        from model_forge import pull
        return {"ok": True, "jobs": pull.list_jobs(20)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "jobs": [],
                "error": f"jobs unavailable: {type(exc).__name__}: {exc}"}


@router.get("/jobs")
async def jobs():
    """Read-only list of active and recent pull/download/installer jobs."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(_jobs), timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "jobs": [],
                "error": f"jobs failed: {type(exc).__name__}: {exc}"}


def _local_cli() -> dict:
    try:
        from model_forge import tools
        return {"ok": True, "clis": tools.local_engine_clis()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "clis": {},
                "error": f"local CLI probe failed: {type(exc).__name__}: {exc}"}


@router.get("/engines/local-cli")
async def engines_local_cli():
    """Read-only PATH check + managed-PID status for local engine CLIs."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_local_cli), timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "clis": {},
                "error": f"local CLI probe failed: {type(exc).__name__}: {exc}"}


def _resources() -> dict:
    try:
        from model_forge import tools
        return tools.cockpit_resources()
    except Exception as exc:  # fail soft: the reason is the payload
        return {"ok": False,
                "error": f"resource snapshot unavailable: "
                         f"{type(exc).__name__}: {exc}"}


def _residency() -> dict:
    try:
        from model_forge import tools
        return tools.cockpit_residency()
    except Exception as exc:  # fail soft: the reason is the payload
        return {"ok": False,
                "error": f"residency snapshot unavailable: "
                         f"{type(exc).__name__}: {exc}"}


@router.get("/residency")
async def residency():
    """The read-only weight-residency rows (pulled / resident / booked /
    unbooked / budget) for the local compute target — the broker status data
    the cockpit surfaces render. Read-only and bounded."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_residency), timeout=_SNAPSHOT_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {"ok": False,
                "error": f"residency snapshot timed out after "
                         f"{_SNAPSHOT_TIMEOUT_S:g} s; nothing was changed — "
                         "refresh to retry"}
    except Exception as exc:  # fail soft: the reason is the payload
        return {"ok": False,
                "error": f"residency snapshot failed: "
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
