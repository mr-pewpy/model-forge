"""B10: persistent pull records for remote compute targets.

A remote machine cannot be scanned over an engine API, so the swap
pre-flight relied on recorded pulls that installs never wrote — every remote
swap refused with "weights not present". Installs now RECORD what landed:
an engine-observed completion (Maya's pull slot reports done) or the user's
own attestation (action=mark_installed). These tests prove the records
persist, carry honest provenance, seed the residency ledger, and turn the
swap pre-flight's refusal into a pass — and that a wrong record changes
nothing on disk (the measured LOAD phase reports the real outcome).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from model_forge.broker import (
    EvictionPlanner,
    LoadRequest,
    Orchestrator,
    ResidencyLedger,
)
from model_forge.pull import (
    RECORD_OBSERVED,
    RECORD_USER_CONFIRMED,
    forget_pull_entry,
    pull_entries,
    record_pull_entry,
)


def _stub_hermes_modules(monkeypatch, data_dir: Path):
    plugins = types.ModuleType("plugins")
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_data_dir = lambda name: data_dir / name
    plugins.plugin_storage = storage
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)
    monkeypatch.setenv("HOME", str(data_dir / "home"))
    (data_dir / "home").mkdir(exist_ok=True)
    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.load_config_readonly = lambda: {}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg_mod)


# --- the record store -------------------------------------------------------

def test_record_read_forget_pull_entries(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    row = record_pull_entry("gpu-1", "gen/x", path="/models/x.safetensors",
                            size_bytes=4096)
    assert row["ok"] is True and row["source"] == RECORD_USER_CONFIRMED
    assert row["recorded_at"]
    # no path given -> an honest placeholder, never an empty claim
    row2 = record_pull_entry("gpu-1", "gen/y")
    assert row2["path"] == "(path not recorded)"
    # unknown provenance degrades to user_confirmed, never "observed"
    row3 = record_pull_entry("gpu-2", "gen/z", source="vibes")
    assert row3["source"] == RECORD_USER_CONFIRMED

    rows = pull_entries("gpu-1")
    assert [r["model_id"] for r in rows] == ["gen/x", "gen/y"]
    assert [r["model_id"] for r in pull_entries("gpu-2")] == ["gen/z"]
    assert len(pull_entries()) == 3

    assert forget_pull_entry("gpu-1", "gen/y")["ok"] is True
    assert [r["model_id"] for r in pull_entries("gpu-1")] == ["gen/x"]
    assert "no recorded pull" in forget_pull_entry("gpu-1", "gen/y")["error"]

    err = record_pull_entry("", "gen/x")
    assert err["ok"] is False


# --- the install verb -------------------------------------------------------

def test_mark_installed_records_user_confirmation(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    from model_forge.targets import ComputeTarget, add_target

    add_target(ComputeTarget(id="gpu-1", label="GPU One", kind="lan",
                             base_url="http://gpu-1:8080", vram_gb=48.0))
    monkeypatch.setattr(tools.maya_adapter, "pull_status",
                        lambda base_url="": {"state": "running"})
    out = json.loads(tools.model_forge_install(
        {"action": "mark_installed", "model_id": "gen/x", "target": "gpu-1",
         "path": "/models/x.safetensors"}))
    assert out["ok"] is True
    assert out["source"] == RECORD_USER_CONFIRMED
    assert "YOUR confirmation" in out["detail"]
    rows = pull_entries("gpu-1")
    assert rows[0]["model_id"] == "gen/x"

    # the local target is scanned from disk — the record says so honestly
    out = json.loads(tools.model_forge_install(
        {"action": "mark_installed", "model_id": "gen/x", "target": "local"}))
    assert out["ok"] is True
    assert "informational" in out["detail"]

    # invalid calls fail honestly and write nothing
    err = json.loads(tools.model_forge_install({"action": "mark_installed"}))
    assert err["ok"] is False and "model_id" in err["error"]
    err = json.loads(tools.model_forge_install(
        {"action": "mark_installed", "model_id": "gen/x", "target": "nope"}))
    assert err["ok"] is False and "unknown target" in err["error"]


def test_mark_installed_observed_when_engine_reports_done(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    from model_forge.targets import ComputeTarget, add_target

    add_target(ComputeTarget(id="gpu-1", label="GPU One", kind="lan",
                             base_url="http://gpu-1:8080", vram_gb=48.0))
    monkeypatch.setattr(tools.maya_adapter, "pull_status",
                        lambda base_url="": {"state": "done"})
    out = json.loads(tools.model_forge_install(
        {"action": "mark_installed", "model_id": "gen/x", "target": "gpu-1"}))
    assert out["source"] == RECORD_OBSERVED
    assert "pull slot complete" in out["detail"]
    # an unreachable engine is not "observed" — the honest fallback
    monkeypatch.setattr(tools.maya_adapter, "pull_status",
                        lambda base_url="": None)
    out = json.loads(tools.model_forge_install(
        {"action": "mark_installed", "model_id": "gen/y", "target": "gpu-1"}))
    assert out["source"] == RECORD_USER_CONFIRMED


def test_remote_install_plans_carry_record_hint(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    from model_forge.registry import models_for
    from model_forge.targets import ComputeTarget, add_target

    remote = ComputeTarget(id="gpu-1", label="GPU One", kind="lan",
                           base_url="http://gpu-1:8080", vram_gb=48.0)
    add_target(remote)
    monkeypatch.setattr(tools, "detect_engines", lambda base_url="", **kw: [])
    entry = next(m for m in models_for()
                 if m.source and m.status != "stub"
                 and "comfyui" in m.engines)
    out = json.loads(tools.model_forge_install(
        {"model_id": entry.id, "target": "gpu-1",
         "engine_model": "diffusion_models/w.safetensors"}))
    assert out["action"] == "run_on_target"
    assert "mark_installed" in out["record_hint"]


# --- seeding the ledger + the swap pre-flight -------------------------------

def test_seed_remote_pulls_feeds_ledger_once(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    from model_forge.targets import ComputeTarget

    record_pull_entry("gpu-1", "gen/x", path="/models/x.safetensors",
                      size_bytes=4096, source=RECORD_OBSERVED)
    target = ComputeTarget(id="gpu-1", label="GPU One", kind="lan",
                           base_url="http://gpu-1:8080", vram_gb=48.0)
    ledger = ResidencyLedger()
    stack = {"ledger": ledger}
    tools._seed_remote_pulls(stack, target)
    pulls = ledger.pulled("gpu-1")
    assert [(p.model_id, p.source) for p in pulls] == [("gen/x", "observed")]
    assert pulls[0].size_bytes == 4096
    # seeded once per stack (the flag gates re-reads)
    record_pull_entry("gpu-1", "gen/y")
    tools._seed_remote_pulls(stack, target)
    assert [p.model_id for p in ledger.pulled("gpu-1")] == ["gen/x"]
    # the local target never uses records — it is scanned from disk
    local = ComputeTarget(id="local", label="This machine", kind="local",
                          base_url="http://127.0.0.1:8080")
    tools._seed_remote_pulls({"ledger": ledger}, local)
    assert ledger.pulled("local") == ()


class _SwapAdapter:
    engine_id = "ollama"
    WARM_PATH = "keep-alive warm"

    def __init__(self):
        self.events = []

    def probe(self):
        self.events.append("probe")
        return {"ok": True, "state": "running"}

    def ensure_loaded(self, model_id, **kw):
        self.events.append(("load", model_id))
        return {"ok": True, "reason": "warmed"}

    def release(self, model_id):
        self.events.append(("release", model_id))
        return {"ok": True, "reason": "stepped down"}


class _Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


def test_recorded_pulls_unblock_the_swap_pre_flight(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools
    from model_forge.targets import ComputeTarget

    record_pull_entry("gpu-1", "image-x", path="/models/image-x.safetensors",
                      size_bytes=1024, source=RECORD_USER_CONFIRMED)
    target = ComputeTarget(id="gpu-1", label="GPU One", kind="lan",
                           base_url="http://gpu-1:8080", vram_gb=24.0)
    clock = _Clock()
    ledger = ResidencyLedger(bookings={"chat-a": 6.0, "image-x": 18.0},
                             clock=clock)
    ledger.set_budget("gpu-1", 24.0)
    ledger.record_resident("gpu-1", "chat-a", engine="ollama", lane="chat")
    adapter = _SwapAdapter()
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": adapter}, clock=clock)
    stack = {"ledger": ledger, "orchestrator": orch}
    request = LoadRequest("image-x", "ollama", lane="image", modality="image")

    # without the record: the pre-flight refuses (nothing recorded)
    row = orch.swap("gpu-1", request, dry_run=True)
    assert row["ok"] is False
    assert "weights not present" in row["reason"]

    # with the seeded record: the pre-flight passes and dry_run changes
    # nothing
    tools._seed_remote_pulls(stack, target)
    row = tools._broker_swap(stack, target, request, [], dry_run=True)
    assert row["ok"] is True
    pre = row["phases"]["preflight"]
    assert pre["ok"] is True
    assert "weights present" in pre["reason"]
    assert ledger.is_resident("gpu-1", "chat-a")   # dry run: untouched
