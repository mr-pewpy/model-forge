"""Compute targets: WHERE inference runs.

A target is a machine with GPUs — this machine, a remote server, or a
rented cloud box. Engines are probed over each target's HTTP API; only the
local machine gets CLI/`nvidia-smi` treatment. Hardware numbers for remote
targets are user-declared (we cannot probe a remote GPU's VRAM honestly over
an engine API), and fit verdicts use them as the budget.

Budget provenance (W1.3): every memory budget carries a source label —
``detected`` (sampled on this machine), ``suggested`` (an engine endpoint
reported it read-only and the user confirmed it), or ``declared`` (the user
typed it). Nothing writes a budget without the user confirming it.

Targets persist in the plugin data dir (survives plugin updates; no secrets
here — endpoint URLs only).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOCAL_ID = "local"
_FILE = "targets.json"

# Budget provenance labels (W1.3). "" means unknown/older row — reported as
# "declared" when a budget exists and "unset" when it does not.
SOURCE_DETECTED = "detected"
SOURCE_SUGGESTED = "suggested"
SOURCE_DECLARED = "declared"
_SOURCE_LABELS = {
    SOURCE_DETECTED: "detected on this machine",
    SOURCE_SUGGESTED: "suggested by an engine endpoint, confirmed by the user",
    SOURCE_DECLARED: "declared by the user",
}


def budget_label(source: str, gb: float) -> str:
    """Honest provenance label for one budget value (never a bare number)."""
    if not gb:
        return "unset (no budget declared — fit verdicts stay unpriced)"
    return _SOURCE_LABELS.get((source or "").strip(), "declared by the user")


@dataclass(frozen=True)
class ComputeTarget:
    id: str
    label: str
    base_url: str               # engine root, e.g. http://gpu-box.example.com:8080
    kind: str = "local"         # local | lan | cloud (advisory label)
    vram_gb: float = 0.0        # budget for fit verdicts (see vram_source)
    ram_gb: float = 0.0
    note: str = ""
    vram_source: str = ""       # detected | suggested | declared | ""
    ram_source: str = ""


_FIELDS = frozenset(ComputeTarget.__dataclass_fields__)


def default_local() -> ComputeTarget:
    return ComputeTarget(
        id=LOCAL_ID, label="This machine", base_url="http://127.0.0.1:8080",
        kind="local", note="Engines probed on 8080/8081/8000/8001; "
                           "hardware detected live.",
        vram_source=SOURCE_DETECTED, ram_source=SOURCE_DETECTED,
    )


def _store_path(base_dir: Path | None) -> Path:
    if base_dir is not None:
        return Path(base_dir) / _FILE
    from plugins.plugin_storage import plugin_data_dir
    return plugin_data_dir("model-forge") / _FILE


def load_targets(base_dir: Path | None = None) -> list[ComputeTarget]:
    """Always contains the local target first; never raises. Desktop settings
    (config_schema targets_json) merge in — they win on id conflicts, so the
    Desktop JSON editor and the target_add tool share one list. Unknown JSON
    keys are dropped, not fatal: a row written by a different version must
    never wipe the whole list."""
    try:
        raw = json.loads(_store_path(base_dir).read_text(encoding="utf-8"))
        targets = [ComputeTarget(**{k: v for k, v in t.items()
                                    if k in _FIELDS})
                   for t in raw if isinstance(t, dict) and t.get("id")]
    except (OSError, ValueError, TypeError):
        targets = []
    for t in _settings_targets():
        targets = [x for x in targets if x.id != t.id]
        targets.append(t)
    if not any(t.id == LOCAL_ID for t in targets):
        targets.insert(0, default_local())
    return targets


def _settings_targets() -> list[ComputeTarget]:
    """Targets declared in Desktop settings (config_schema targets_json)."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        settings = (entries.get("model-forge") or {}).get("settings") or {}
        rows = settings.get("targets_json") or []
        out = []
        for t in rows:
            if isinstance(t, dict) and t.get("id") and t.get("base_url"):
                out.append(ComputeTarget(
                    id=str(t["id"]), label=str(t.get("label") or t["id"]),
                    base_url=str(t["base_url"]),
                    kind=str(t.get("kind") or "lan"),
                    vram_gb=float(t.get("vram_gb") or 0.0),
                    ram_gb=float(t.get("ram_gb") or 0.0),
                    note=str(t.get("note") or ""),
                    vram_source=str(t.get("vram_source") or SOURCE_DECLARED),
                    ram_source=str(t.get("ram_source") or SOURCE_DECLARED)))
        return out
    except Exception:
        return []


def settings_target_ids() -> frozenset[str]:
    """Ids currently declared in Desktop settings (targets_json). Editing or
    removing these from the tool store would be invisible — settings win on id
    conflicts — so the lifecycle verbs refuse them with that reason."""
    return frozenset(t.id for t in _settings_targets())


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


def validate_target_fields(*, base_url: str = "", kind: str = "",
                           vram_gb=None, ram_gb=None) -> str:
    """Honest field validation shared by add/edit — returns an error string
    ('' when valid). Nothing is written on an invalid field."""
    if base_url and not (base_url.startswith("http://")
                         or base_url.startswith("https://")):
        return ("base_url must be http(s), e.g. "
                "http://gpu-box.example.com:8080")
    if kind and kind not in ("local", "lan", "cloud"):
        return f"kind must be one of ['local', 'lan', 'cloud'], got {kind!r}"
    for name, value in (("vram_gb", vram_gb), ("ram_gb", ram_gb)):
        if value is None:
            continue
        try:
            gb = float(value)
        except (TypeError, ValueError):
            return f"{name} must be a number, got {value!r}"
        if gb < 0.0:
            return f"{name} must be >= 0, got {gb:g}"
    return ""


def update_target(target_id: str, changes: dict,
                  base_dir: Path | None = None) -> tuple[ComputeTarget | None,
                                                         str]:
    """Edit a REGISTERED target's fields — never the local target, never a
    settings-declared row (settings win on id conflicts, so a store edit would
    be invisible; the caller is told to edit settings instead). Returns the
    updated target or (None, honest reason)."""
    if target_id == LOCAL_ID:
        return None, ("the local target is permanent by design — it cannot "
                      "be edited or removed")
    if target_id in settings_target_ids():
        return None, (f"{target_id!r} is declared in Desktop settings "
                      "(targets_json), which wins over the tool store — edit "
                      "it there (Settings -> Plugins -> Model Forge) instead")
    current = get_target(target_id, base_dir)
    if current is None:
        return None, f"unknown target: {target_id!r} (see model_forge_status)"
    fields = {k: v for k, v in changes.items() if k in _FIELDS and k != "id"}
    err = validate_target_fields(
        base_url=str(fields.get("base_url", "") or ""),
        kind=str(fields.get("kind", "") or ""),
        vram_gb=fields.get("vram_gb"), ram_gb=fields.get("ram_gb"))
    if err:
        return None, err
    merged = asdict(current)
    merged.update(fields)
    updated = ComputeTarget(**merged)
    targets = [updated if x.id == target_id else x
               for x in load_targets(base_dir)]
    save_targets(targets, base_dir)
    return updated, ""


def remove_target(target_id: str,
                  base_dir: Path | None = None) -> tuple[bool, str]:
    """Remove a REGISTERED target. The local target is permanent; a
    settings-declared row must be removed from settings. Returns (ok,
    honest reason)."""
    if target_id == LOCAL_ID:
        return False, ("the local target is permanent by design — it cannot "
                       "be edited or removed")
    if target_id in settings_target_ids():
        return False, (f"{target_id!r} is declared in Desktop settings "
                       "(targets_json) — remove it there (Settings -> "
                       "Plugins -> Model Forge); the tool store only holds "
                       "tool-registered targets")
    if get_target(target_id, base_dir) is None:
        return False, f"unknown target: {target_id!r} (see model_forge_status)"
    targets = [x for x in load_targets(base_dir) if x.id != target_id]
    save_targets(targets, base_dir)
    return True, ""
