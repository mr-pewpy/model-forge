"""Model-file downloads with resume + progress jobs.

Streams a single file from an http(s) URL (Hugging Face /resolve links,
ModelScope /resolve links, anything direct) to a destination path. Applies
pull credentials only for their own host family (see credentials.py).
Resume: a ``<dest>.part`` file continues from its size via a Range request
when the server supports it. Long downloads run in a thread with a JSON job
file under the plugin data dir; callers poll ``job_status(job_id)``.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .credentials import auth_headers

_CHUNK = 1 << 20  # 1 MiB
_TIMEOUT_S = 30.0
# The exact shape start_download generates (uuid4().hex[:12]). Anything else
# is rejected before it reaches the jobs path (no traversal, no odd bytes).
_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_DOWNLOAD_SUBDIR = "downloads"


def _jobs_dir() -> Path:
    from plugins.plugin_storage import plugin_data_dir
    d = plugin_data_dir("model-forge") / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def allowed_download_roots(extra_roots=None) -> list[Path]:
    """Where a model-requested download may land: the plugin's downloads dir
    plus any roots the USER lists in plugin settings (`download_roots`)."""
    from plugins.plugin_storage import plugin_data_dir
    roots = [plugin_data_dir("model-forge") / _DOWNLOAD_SUBDIR]
    roots += [Path(str(r)) for r in (extra_roots or []) if str(r).strip()]
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        settings = (entries.get("model-forge") or {}).get("settings") or {}
        roots += [Path(str(r)) for r in (settings.get("download_roots") or [])
                  if str(r).strip()]
    except Exception:
        pass
    return roots


def _confine_dest(dest: str, allowed_roots) -> tuple[Path | None, str]:
    """Resolve and confine a download destination. Returns (path, "") when
    safe, (None, reason) otherwise. Refuses: non-absolute paths, symlinked
    destination entries, and anything that resolves outside the allowed roots
    (which also defeats symlinked-parent escapes)."""
    p = Path(dest)
    if not p.is_absolute():
        return None, "dest must be an absolute file path"
    if p.is_symlink():
        return None, f"dest may not be a symlink: {dest}"
    try:
        resolved = p.resolve()
    except OSError as exc:
        return None, f"dest not usable: {exc}"
    for root in allowed_roots:
        try:
            root_resolved = Path(root).resolve()
        except OSError:
            continue
        if resolved.is_relative_to(root_resolved):
            return resolved, ""
    return None, ("dest must be inside the plugin downloads directory or a "
                  "configured download root (Settings -> Plugins -> Model "
                  "Forge -> download_roots)")


def download_plan(url: str, dest: str, allowed_roots=None) -> dict:
    """Validate a download request (pure). Destination confinement: a
    model-supplied dest may only land inside the plugin downloads dir or a
    user-configured download root (symlinks refused)."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"ok": False, "error": "url must be http(s)"}
    if not dest or not str(dest).startswith("/"):
        return {"ok": False, "error": "dest must be an absolute file path"}
    if allowed_roots is None:
        allowed_roots = allowed_download_roots()
    confined, err = _confine_dest(str(dest), allowed_roots)
    if confined is None:
        return {"ok": False, "error": err}
    return {"ok": True, "action": "download", "url": url, "dest": str(confined)}


def _write_job(job_id: str, payload: dict) -> None:
    (_jobs_dir() / f"{job_id}.json").write_text(
        json.dumps(payload), encoding="utf-8")


def job_status(job_id: str) -> dict:
    # job_id is model-supplied: gate it before it joins the jobs path.
    if not _JOB_ID_RE.fullmatch(str(job_id or "")):
        return {"state": "unknown", "error": f"invalid job id: {job_id!r}"}
    try:
        return json.loads(
            (_jobs_dir() / f"{job_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"state": "unknown", "error": f"no such job: {job_id}"}


def _run(job_id: str, url: str, dest: str, sha256: str) -> None:
    part = Path(str(dest) + ".part")
    final = Path(dest)
    state = {"job_id": job_id, "url": url, "dest": dest, "state": "running",
             "done_bytes": 0, "total_bytes": 0}
    try:
        final.parent.mkdir(parents=True, exist_ok=True)
        done = part.stat().st_size if part.exists() else 0
        headers = auth_headers(url)
        if done:
            headers["Range"] = f"bytes={done}-"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            if done and resp.status != 206:  # server restarted the file
                done = 0
            total = resp.headers.get("Content-Length")
            state["total_bytes"] = done + int(total) if total else 0
            mode = "ab" if done else "wb"
            with open(part, mode) as out:
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    state["done_bytes"] = done
                    if done % (8 * _CHUNK) < _CHUNK:
                        _write_job(job_id, state)
        if sha256:
            digest = hashlib.sha256()
            with open(part, "rb") as f:
                for chunk in iter(lambda: f.read(_CHUNK), b""):
                    digest.update(chunk)
            if digest.hexdigest().lower() != sha256.lower():
                part.unlink(missing_ok=True)
                state.update(state="error",
                             error=f"sha256 mismatch: {digest.hexdigest()}")
                _write_job(job_id, state)
                return
        part.replace(final)
        state.update(state="done", done_bytes=done)
    except Exception as exc:  # noqa: BLE001 — job file records the failure
        state.update(state="error", error=f"{type(exc).__name__}: {exc}")
    _write_job(job_id, state)


def start_download(url: str, dest: str, sha256: str = "") -> dict:
    """Kick the download in a background thread; returns the job row."""
    plan = download_plan(url, dest)
    if not plan.get("ok"):
        return plan
    job_id = uuid.uuid4().hex[:12]
    _write_job(job_id, {"job_id": job_id, "url": url, "dest": dest,
                        "state": "starting", "done_bytes": 0,
                        "total_bytes": 0})
    threading.Thread(target=_run, args=(job_id, url, dest, sha256),
                     daemon=True).start()
    return {"ok": True, "job_id": job_id, "poll": "model_forge_install with "
            f"job_id={job_id}"}


# --- pull records (B10): what weights are known present where ---------------
#
# The local target's weights are read from the storage scan every time. A
# REMOTE target cannot be scanned over an engine API, so installs RECORD what
# landed there: an engine-observed completion (Maya's /api/pull/status) or the
# user's own attestation after running the exact on-target command. The
# swap pre-flight reads these records instead of refusing every remote swap;
# a wrong record degrades to an honest load failure during the measured LOAD
# phase — it never destroys anything (fail-closed lives in the planner, not
# in a guess about remote disk state).

_RECORDS_FILE = "pull-records.json"
RECORD_OBSERVED = "observed"
RECORD_USER_CONFIRMED = "user_confirmed"


def _records_path() -> Path:
    from plugins.plugin_storage import plugin_data_dir
    return plugin_data_dir("model-forge") / _RECORDS_FILE


def _read_records() -> dict:
    try:
        raw = json.loads(_records_path().read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_records(records: dict) -> None:
    path = _records_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=1, sort_keys=True),
                    encoding="utf-8")


def record_pull_entry(target_id: str, model_id: str, *, path: str = "",
                      size_bytes=None,
                      source: str = RECORD_USER_CONFIRMED) -> dict:
    """Record weights present on a compute target (B10). ``source`` is the
    honest provenance: ``observed`` (an engine reported the pull done) or
    ``user_confirmed`` (the user attested the on-target install finished).
    Records never claim more than that — they say the weights are believed
    present, and the swap's LOAD phase states the real outcome."""
    target_id = str(target_id or "").strip()
    model_id = str(model_id or "").strip()
    if not target_id or not model_id:
        return {"ok": False, "error": "target_id and model_id are required"}
    try:
        size = int(size_bytes) if size_bytes is not None else None
    except (TypeError, ValueError):
        size = None
    row = {"target_id": target_id, "model_id": model_id,
           "path": str(path or "") or "(path not recorded)",
           "size_bytes": size,
           "source": (source if source in (RECORD_OBSERVED,
                                           RECORD_USER_CONFIRMED)
                      else RECORD_USER_CONFIRMED),
           "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime())}
    records = _read_records()
    records[f"{target_id}:{model_id}"] = row
    _write_records(records)
    return {"ok": True, **row}


def pull_entries(target_id: str | None = None) -> list[dict]:
    """Recorded pulls — for one target when ``target_id`` is given, else all.
    Newest key order stable (sorted)."""
    rows = [r for r in _read_records().values() if isinstance(r, dict)]
    if target_id is not None:
        rows = [r for r in rows if r.get("target_id") == target_id]
    return sorted(rows, key=lambda r: (str(r.get("target_id")),
                                       str(r.get("model_id"))))


def forget_pull_entry(target_id: str, model_id: str) -> dict:
    """Drop one record (e.g. after the user says the weights are gone)."""
    key = f"{str(target_id).strip()}:{str(model_id).strip()}"
    records = _read_records()
    if key not in records:
        return {"ok": False, "error": f"no recorded pull for {key!r}"}
    records.pop(key, None)
    _write_records(records)
    return {"ok": True, "forgotten": key}
