"""Live model discovery: fresh HuggingFace results beside the curated catalog.

The registry (registry.py) is a hand-maintained snapshot; this module adds a
freshness layer on top: it queries the public HuggingFace REST API for the
newest models in each modality's pipeline lane, sorted by last-modified date,
so browse can surface bleeding-edge releases alongside the curated entries.

Constraints honored here: stdlib only (urllib), answers cached on disk for
TTL_S seconds under plugin_data_dir("model-forge")/cache, and everything
fails SOFT — no network, no plugin storage, or a malformed API response just
yields ``live_discovery: "unavailable"`` and callers keep the curated catalog.

The HTTP fetch is injectable (``fetch=`` parameter, or the module-level
``fetch_json``) so tests never touch the network.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HF_API = "https://huggingface.co/api/models"
HTTP_TIMEOUT_S = 5.0
TTL_S = 3600  # one hour
DEFAULT_LIMIT = 8

# modality -> HF pipeline_tag lanes probed against the live API
# (GET /api/models?pipeline_tag=<tag> — `filter=<tag>` returns the same rows).
# Tags that return nothing today simply produce an honest empty lane.
MODALITY_TAGS: dict[str, tuple[str, ...]] = {
    "chat": ("text-generation",),
    "vision": ("image-text-to-text",),
    "ocr": ("image-to-text", "document-question-answering"),
    "image": ("text-to-image",),
    "image_edit": ("image-to-image",),
    "video": ("text-to-video", "image-to-video"),
    "3d": ("text-to-3d", "image-to-3d"),
    # HF has no text-to-music / text-to-sfx pipeline tags; both live under
    # text-to-audio (MusicGen, AudioGen, Stable Audio are all tagged that).
    "music": ("text-to-audio",),
    "sfx": ("text-to-audio",),
    "voice_clone": ("text-to-speech",),
    # honest empty lane, mirroring the registry's world-model stub slot
    "world_model": ("world-model",),
}


def search_url(pipeline_tag: str = "", limit: int = DEFAULT_LIMIT) -> str:
    """The freshness query: newest-first by last-modified, per pipeline lane."""
    params = {"sort": "lastModified", "direction": "-1", "limit": str(limit)}
    if pipeline_tag:
        params["pipeline_tag"] = pipeline_tag
    return HF_API + "?" + urllib.parse.urlencode(params)


def fetch_json(url: str, timeout: float = HTTP_TIMEOUT_S):
    """GET url and parse JSON (stdlib urllib). Raises on any network/parse
    error — callers treat exceptions as 'live discovery unavailable'."""
    req = urllib.request.Request(url, headers={"User-Agent": "model-forge"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _row(item: dict) -> dict:
    """Normalize one HF API model row to the fresh-result shape."""
    mid = str(item.get("id") or item.get("modelId") or "")
    last_modified = str(item.get("lastModified") or "")[:10]  # YYYY-MM-DD
    try:
        downloads = int(item.get("downloads") or 0)
    except (TypeError, ValueError):
        downloads = 0
    return {
        "id": mid,
        "last_modified": last_modified,
        "downloads": downloads,
        "pipeline_tag": str(item.get("pipeline_tag") or ""),
        "url": "https://huggingface.co/" + mid,
        "status": "fresh",
    }


def _sort_rows(rows: list[dict]) -> list[dict]:
    """last-modified desc, undated rows last, id tiebreak for determinism."""
    return sorted(rows, key=lambda r: (r["last_modified"] != "",
                                       r["last_modified"], r["id"]),
                  reverse=True)


def _cache_file(cache_dir: Path, key: str) -> Path:
    safe = re.sub(r"[^a-z0-9_-]", "_", key.lower())
    return cache_dir / ("hf_" + safe + ".json")


def _cache_dir() -> Path:
    from plugins.plugin_storage import plugin_data_dir
    return plugin_data_dir("model-forge") / "cache"


def _cache_read(cache_dir: Path | None, key: str, ttl: float,
                now: float) -> list[dict] | None:
    if cache_dir is None:
        return None
    try:
        blob = json.loads(_cache_file(cache_dir, key).read_text())
        if now - float(blob.get("fetched_at", 0)) >= ttl:
            return None  # expired
        rows = blob.get("results")
        return rows if isinstance(rows, list) else None
    except Exception:
        return None


def _cache_write(cache_dir: Path | None, key: str, rows: list[dict],
                 now: float) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_file(cache_dir, key).write_text(
            json.dumps({"fetched_at": now, "results": rows}))
    except Exception:
        pass  # cache is an optimization, never a failure


def fresh_models(modality: str | None = None, *, limit: int = DEFAULT_LIMIT,
                 cache_dir: Path | None = None, ttl: float = TTL_S,
                 now: float | None = None, fetch=None) -> dict:
    """Newest live HF models for a modality, last-modified desc.

    Returns {"live_discovery": "ok"|"unavailable", "models": [row, ...],
    "error": str} where each row is {id, last_modified, downloads,
    pipeline_tag, url, status:"fresh"} — unvetted live results, deliberately
    distinct from curated registry entries. Cached per modality for ``ttl``
    seconds under ``cache_dir`` (default: the plugin data dir); every failure
    degrades to ``unavailable`` so the curated catalog stays authoritative.
    """
    fetch = fetch or fetch_json
    now = time.time() if now is None else now
    key = modality or "all"
    try:
        cdir = Path(cache_dir) if cache_dir is not None else _cache_dir()
    except Exception:
        cdir = None
    cached = _cache_read(cdir, key, ttl, now)
    if cached is not None:
        return {"live_discovery": "ok", "models": _sort_rows(cached),
                "error": ""}
    try:
        tags = MODALITY_TAGS.get(modality, ()) if modality else ()
        rows: list[dict] = []
        seen: set[str] = set()
        for tag in (tags or ("",)):
            data = fetch(search_url(tag, limit))
            if not isinstance(data, list):
                raise ValueError(
                    f"unexpected HF response shape: {type(data).__name__}")
            for item in data:
                if not isinstance(item, dict):
                    continue
                row = _row(item)
                if row["id"] and row["id"] not in seen:
                    seen.add(row["id"])
                    rows.append(row)
        rows = _sort_rows(rows)[:limit]
        _cache_write(cdir, key, rows, now)
        return {"live_discovery": "ok", "models": rows, "error": ""}
    except Exception as exc:  # noqa: BLE001 — fail-soft is the contract
        return {"live_discovery": "unavailable", "models": [],
                "error": f"{type(exc).__name__}: {exc}"}
