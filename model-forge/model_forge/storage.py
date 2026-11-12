"""Model-storage inventory and user-approved deletion.

Reads an inventory of model files on this machine: the plugin's downloads
directory plus the engine-managed model directories under the user home
(Ollama models dir, Hugging Face hub cache). Listing is read-only (names,
sizes, modification times) and fails soft — an absent or unreadable root is
skipped and reported, never raised.

Deletion accepts ONLY an entry the inventory returned — its ``id`` or its
exact ``path`` — never an arbitrary path, and re-verifies containment with
the pull._confine_dest rules: absolute, never a symlink, resolved strictly
inside one of the listed roots (which also defeats symlinked-parent
escapes). The human-approval gate sits one layer up in tools.policy_hook
(rule key ``model-forge:storage_delete``); by the time a handler reaches
delete_entry, the user has approved the exact path and size.
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_DOWNLOAD_SUBDIR = "downloads"


@dataclass(frozen=True)
class StorageRoot:
    """One directory the storage tool covers."""
    key: str    # stable machine id recorded on every entry
    label: str  # what the directory is
    path: Path


def storage_roots(home: Path | None = None) -> list[StorageRoot]:
    """Candidate roots: the plugin's downloads dir, plus the engine-managed
    model directories under the user home. Absent roots are fine here — the
    listing skips them fail-soft."""
    from plugins.plugin_storage import plugin_data_dir
    home = Path(home) if home is not None else Path.home()
    return [
        StorageRoot("plugin-downloads", "Model Forge downloads directory",
                    plugin_data_dir("model-forge") / _DOWNLOAD_SUBDIR),
        StorageRoot("ollama-models", "Ollama models directory",
                    home / ".ollama" / "models"),
        StorageRoot("huggingface-hub", "Hugging Face hub cache",
                    home / ".cache" / "huggingface" / "hub"),
    ]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(
        timespec="seconds")


def _tree_size(path: Path) -> int:
    """On-disk size of a directory tree (fail-soft per file: unreadable
    entries are skipped, not raised). Symlinks count as themselves and are
    never followed."""
    total = 0
    try:
        walk = os.walk(path, followlinks=False)
    except OSError:
        return 0
    for dirpath, _dirnames, filenames in walk:
        for fname in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, fname)).st_size
            except OSError:
                continue
    return total


def list_storage(roots=None, home: Path | None = None) -> dict:
    """Inventory every root: one entry per file/folder directly inside it.

    Entries carry name, absolute path, size in bytes (folders add up every
    file inside), last_modified (the filesystem modification time — usage is
    not tracked), and the root they came from. Absent or unreadable roots are
    skipped fail-soft and reported in roots_skipped."""
    roots = list(roots) if roots is not None else storage_roots(home=home)
    entries = []
    scanned = []
    skipped = []
    for root in roots:
        base = Path(os.path.abspath(str(root.path)))
        try:
            children = sorted(base.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            if isinstance(exc, FileNotFoundError):
                reason = "directory not present"
            elif isinstance(exc, PermissionError):
                reason = "directory not readable"
            else:
                reason = f"not usable: {exc}"
            skipped.append({"root": root.key, "label": root.label,
                            "path": str(base), "reason": reason})
            continue
        scanned.append({"root": root.key, "label": root.label,
                        "path": str(base)})
        for child in children:
            try:
                st = child.lstat()
            except OSError:
                continue  # fail-soft per entry
            if stat.S_ISLNK(st.st_mode):
                kind, size = "symlink", st.st_size
            elif stat.S_ISDIR(st.st_mode):
                kind, size = "dir", _tree_size(child)
            else:
                kind, size = "file", st.st_size
            entries.append({
                "id": f"{root.key}:{child.name}",
                "name": child.name,
                "path": str(base / child.name),
                "root": root.key,
                "root_label": root.label,
                "type": kind,
                "size_bytes": size,
                "last_modified": _iso(st.st_mtime),
            })
    return {
        "ok": True,
        "entries": entries,
        "roots": scanned,
        "roots_skipped": skipped,
        "note": ("size_bytes is on-disk size (a folder adds up every file "
                 "inside it); last_modified is the filesystem modification "
                 "time — usage is not tracked."),
    }


def _find_entry(entries: list[dict], ref: str) -> dict | None:
    return next((e for e in entries if e["id"] == ref or e["path"] == ref),
                None)


def describe_entry(ref: str, roots=None, home: Path | None = None) -> dict | None:
    """The inventory row for an entry id or exact listed path (else None).
    Read-only; used by the approval prompt to name what would be deleted."""
    roots = list(roots) if roots is not None else storage_roots(home=home)
    return _find_entry(list_storage(roots=roots)["entries"], str(ref or ""))


def _confine(path_str: str, allowed_roots) -> tuple[Path | None, str]:
    """Resolve and confine a delete target (the pull._confine_dest rules).
    Returns (path, "") when safe, (None, reason) otherwise. Refuses:
    non-absolute paths, symlinked entries, the root itself, and anything
    that resolves outside the allowed roots (which also defeats
    symlinked-parent escapes)."""
    p = Path(path_str)
    if not p.is_absolute():
        return None, "entry path must be absolute"
    if p.is_symlink():
        return None, f"refusing to delete a symlink: {path_str}"
    try:
        resolved = p.resolve()
    except OSError as exc:
        return None, f"entry path not usable: {exc}"
    for root in allowed_roots:
        try:
            root_resolved = Path(root).resolve()
        except OSError:
            continue
        if resolved != root_resolved \
                and resolved.is_relative_to(root_resolved):
            return resolved, ""
    return None, ("entry must sit strictly inside one of the listed storage "
                  "roots (the Model Forge downloads directory or an "
                  "engine-managed model directory)")


def delete_entry(ref: str, roots=None, home: Path | None = None) -> dict:
    """Delete one inventoried entry; reports the bytes freed.

    ref must be an entry id from list_storage or that entry's exact path —
    anything else is refused before it reaches the filesystem. Containment
    is re-verified here (fail closed) even though the ref matched the
    listing."""
    ref = str(ref or "")
    roots = list(roots) if roots is not None else storage_roots(home=home)
    entry = _find_entry(list_storage(roots=roots)["entries"], ref)
    if entry is None:
        return {"ok": False,
                "error": "ref is not an entry from the storage list: delete "
                         "accepts only an entry id or the exact path "
                         "returned by action=list (arbitrary paths are "
                         "refused)"}
    confined, err = _confine(entry["path"], [r.path for r in roots])
    if confined is None:
        return {"ok": False, "error": err, "entry": entry["path"]}
    try:
        st = confined.lstat()
    except OSError as exc:
        return {"ok": False, "error": f"entry is no longer present: {exc}",
                "entry": entry["path"]}
    is_dir = stat.S_ISDIR(st.st_mode)
    freed = _tree_size(confined) if is_dir else st.st_size
    try:
        if is_dir:
            shutil.rmtree(confined)
        else:
            confined.unlink()
    except OSError as exc:
        return {"ok": False, "error": f"delete failed: {exc}",
                "entry": entry["path"]}
    return {"ok": True, "deleted": entry["path"], "name": entry["name"],
            "root": entry["root"], "type": entry["type"],
            "freed_bytes": freed}
