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
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .credentials import auth_headers

_CHUNK = 1 << 20  # 1 MiB
_TIMEOUT_S = 30.0


def _jobs_dir() -> Path:
    from plugins.plugin_storage import plugin_data_dir
    d = plugin_data_dir("model-forge") / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_plan(url: str, dest: str) -> dict:
    """Validate a download request (pure)."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"ok": False, "error": "url must be http(s)"}
    if not dest or not str(dest).startswith("/"):
        return {"ok": False, "error": "dest must be an absolute file path"}
    return {"ok": True, "action": "download", "url": url, "dest": dest}


def _write_job(job_id: str, payload: dict) -> None:
    (_jobs_dir() / f"{job_id}.json").write_text(
        json.dumps(payload), encoding="utf-8")


def job_status(job_id: str) -> dict:
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
