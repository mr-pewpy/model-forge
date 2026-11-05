"""Compute targets: WHERE inference runs.

A target is a machine with GPUs — this one, an LXC on your Proxmox node, a
rented cloud box. Engines are probed over each target's HTTP API; only the
local machine gets CLI/`nvidia-smi` treatment. Hardware numbers for remote
targets are user-declared (we cannot probe a remote GPU's VRAM honestly over
an engine API), and fit verdicts use them as the budget.

Targets persist in the plugin data dir (survives plugin updates; no secrets
here — endpoint URLs only).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOCAL_ID = "local"
_FILE = "targets.json"


@dataclass(frozen=True)
class ComputeTarget:
    id: str
    label: str
    base_url: str               # engine root, e.g. http://192.168.1.62:8080
    kind: str = "local"         # local | lan | cloud (advisory label)
    vram_gb: float = 0.0        # declared budget for fit verdicts
    ram_gb: float = 0.0
    note: str = ""


def default_local() -> ComputeTarget:
    return ComputeTarget(
        id=LOCAL_ID, label="This machine", base_url="http://127.0.0.1:8080",
        kind="local", note="Engines probed on 8080/8081/8000/8001; "
                           "hardware detected live.",
    )


def _store_path(base_dir: Path | None) -> Path:
    if base_dir is not None:
        return Path(base_dir) / _FILE
    from plugins.plugin_storage import plugin_data_dir
    return plugin_data_dir("model-forge") / _FILE


def load_targets(base_dir: Path | None = None) -> list[ComputeTarget]:
    """Always contains the local target first; never raises."""
    try:
        raw = json.loads(_store_path(base_dir).read_text(encoding="utf-8"))
        targets = [ComputeTarget(**t) for t in raw if t.get("id")]
    except (OSError, ValueError, TypeError):
        targets = []
    if not any(t.id == LOCAL_ID for t in targets):
        targets.insert(0, default_local())
    return targets


def save_targets(targets: list[ComputeTarget], base_dir: Path | None = None) -> None:
    path = _store_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(t) for t in targets], indent=1), encoding="utf-8")


def add_target(t: ComputeTarget, base_dir: Path | None = None) -> list[ComputeTarget]:
    """Add or replace by id; the local target is never removable."""
    targets = [x for x in load_targets(base_dir) if x.id != t.id or x.id == LOCAL_ID]
    if t.id != LOCAL_ID:
        targets = [x for x in load_targets(base_dir) if x.id != t.id]
    targets.append(t)
    save_targets(targets, base_dir)
    return load_targets(base_dir)


def get_target(target_id: str | None, base_dir: Path | None = None) -> ComputeTarget | None:
    wanted = target_id or LOCAL_ID
    for t in load_targets(base_dir):
        if t.id == wanted:
            return t
    return None
