"""A20: handler-side re-verification of the release's restore plan against
the exact set the human-approval gate saw.

The gate (policy_hook) inspects the restore plan BEFORE the handler runs; the
handler recomputes it against live ledger state. These tests prove the
recomputed plan is now RE-VERIFIED against the gate-approved set and that a
grown plan (a concurrent claim/eviction in the gate-to-handler window) is
refused — fail-closed, nothing loaded — instead of silently starting a server
the gate never approved. ``approved_restore=None`` (no gate in the path)
keeps the pre-A20 behavior.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from model_forge.broker import (
    EngineLifecycle,
    EvictionPlanner,
    LoadRequest,
    Orchestrator,
    ResidencyLedger,
)


class FakeClock:
    def __init__(self, now: float = 1_000.0):
        self.now = float(now)

    def __call__(self) -> float:
        return self.now


class RecordingLifecycle(EngineLifecycle):
    def __init__(self, engine_id="recording"):
        self.engine_id = engine_id
        self.calls = []

    def ensure_loaded(self, model_id, **kwargs):
        self.calls.append(("ensure_loaded", model_id))
        return {"ok": True, "reason": "warmed"}

    def release(self, model_id):
        self.calls.append(("release", model_id))
        return {"ok": True, "reason": "stepped down"}

    def probe(self):
        return {"ok": True, "state": "running"}


def _stack(models):
    """A ready-to-claim stack: ``models`` maps model_id -> engine, with every
    model resident at claim time except the incoming image model."""
    clock = FakeClock()
    ledger = ResidencyLedger(
        bookings={**{m: 4.0 for m in models}, "image-x": 18.0},
        clock=clock)
    ledger.set_budget("local", 24.0)
    for m, engine in models.items():
        ledger.record_resident("local", m, engine=engine, lane="chat")
    adapters = {e: RecordingLifecycle(e) for e in set(models.values())}
    adapters.setdefault("ollama", RecordingLifecycle("ollama"))
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        adapters, clock=clock)
    claimed = orch.claim("shot-1", "local",
                         LoadRequest("image-x", "ollama", lane="image",
                                     modality="image"))
    assert claimed["ok"] is True
    return clock, ledger, adapters, orch


# --- orchestrator: the re-verification itself -------------------------------

def test_release_refuses_restore_plan_that_grew():
    # chat-a runs on a process engine (vllm); the gate approved only
    # chat-b's restore — the handler's recomputed plan now includes chat-a
    # (it was evicted in the window), so the release must refuse.
    clock, ledger, adapters, orch = _stack(
        {"chat-a": "vllm", "chat-b": "ollama"})
    released = orch.release("shot-1", approved_restore={"chat-b"},
                            gated_engines=("vllm",))
    assert released["ok"] is False
    assert "changed after the approval gate" in released["reason"]
    assert "chat-a" in released["reason"]
    assert "nothing was loaded" in released["reason"]
    # fail closed: the restore ran nothing
    assert not ledger.is_resident("local", "chat-a")
    assert not ledger.is_resident("local", "chat-b")
    assert ("ensure_loaded", "chat-a") not in adapters["vllm"].calls


def test_release_runs_when_restore_matches_approval():
    clock, ledger, adapters, orch = _stack(
        {"chat-a": "vllm", "chat-b": "ollama"})
    released = orch.release("shot-1",
                            approved_restore={"chat-a", "chat-b"},
                            gated_engines=("vllm",))
    assert released["ok"] is True
    assert ledger.is_resident("local", "chat-a")
    assert ledger.is_resident("local", "chat-b")


def test_release_only_gated_engines_are_verified():
    # chat-b's ollama restore is NOT in the approved set — but ollama starts
    # no server, so it is not the gate's business and the release proceeds.
    clock, ledger, adapters, orch = _stack(
        {"chat-a": "vllm", "chat-b": "ollama"})
    released = orch.release("shot-1", approved_restore={"chat-a"},
                            gated_engines=("vllm",))
    assert released["ok"] is True
    assert ledger.is_resident("local", "chat-b")


def test_release_empty_approval_refuses_process_restore():
    clock, ledger, adapters, orch = _stack({"chat-a": "vllm"})
    released = orch.release("shot-1", approved_restore=set(),
                            gated_engines=("vllm",))
    assert released["ok"] is False
    assert "chat-a" in released["reason"]
    # an empty approval does NOT bind non-gated restores
    clock, ledger, adapters, orch = _stack({"chat-b": "ollama"})
    released = orch.release("shot-1", approved_restore=set(),
                            gated_engines=("vllm",))
    assert released["ok"] is True


def test_release_without_manifest_keeps_legacy_behavior():
    clock, ledger, adapters, orch = _stack({"chat-a": "vllm"})
    released = orch.release("shot-1", approved_restore=None,
                            gated_engines=("vllm",))
    assert released["ok"] is True
    assert ledger.is_resident("local", "chat-a")


# --- the gate-to-handler wiring (tools.py) ----------------------------------

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


def test_policy_hook_records_approved_restore_and_release_consumes(
        tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    clock, ledger, adapters, orch = _stack(
        {"chat-a": "vllm", "chat-b": "ollama"})
    stack = {"orchestrator": orch, "ledger": ledger, "target_id": "local"}
    monkeypatch.setattr(tools, "broker_stack",
                        lambda target, **kw: stack)

    d = tools.policy_hook("model_forge_broker",
                          {"action": "release", "scope_id": "shot-1",
                           "target": "local"})
    assert d["rule_key"] == "model-forge:broker_server_start"
    # the manifest freezes exactly what the gate inspected
    assert tools._consume_approved_restore("shot-1") == {"chat-a", "chat-b"}

    # with the manifest consumed by the handler, a matching plan releases
    tools._record_approved_restore("shot-1", ["chat-a", "chat-b"])
    out = tools._broker_release(stack, "shot-1")
    assert out["ok"] is True
    assert tools._consume_approved_restore("shot-1") is None   # consumed


def test_release_handler_refuses_on_grown_plan(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    clock, ledger, adapters, orch = _stack(
        {"chat-a": "vllm", "chat-b": "ollama"})
    stack = {"orchestrator": orch, "ledger": ledger, "target_id": "local"}
    monkeypatch.setattr(tools, "broker_stack",
                        lambda target, **kw: stack)

    # The gate approved only chat-b (chat-a looked resident at gate time and
    # was evicted in the window) — the handler must refuse and say why.
    tools._record_approved_restore("shot-1", ["chat-b"])
    out = tools._broker_release(stack, "shot-1")
    assert out["ok"] is False
    assert out["action"] == "release_failed"
    assert "changed after the approval gate" in out["error"]
    assert not ledger.is_resident("local", "chat-a")


def test_release_handler_without_gate_runs(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    clock, ledger, adapters, orch = _stack({"chat-a": "vllm"})
    stack = {"orchestrator": orch, "ledger": ledger, "target_id": "local"}
    out = tools._broker_release(stack, "shot-1")
    assert out["ok"] is True     # no manifest: pre-A20 behavior


def test_media_inline_manifest_recorded_and_consumed(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    from model_forge.targets import ComputeTarget

    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 4.0}, clock=clock)
    ledger.set_budget("local", 24.0)
    ledger.record_resident("local", "chat-a", engine="vllm", lane="chat")
    adapters = {"vllm": RecordingLifecycle("vllm")}
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        adapters, clock=clock)
    stack = {"orchestrator": orch, "ledger": ledger, "target_id": "local"}
    monkeypatch.setattr(tools, "broker_stack",
                        lambda target, **kw: stack)
    target = ComputeTarget(id="local", label="This machine",
                           base_url="http://127.0.0.1:8080", kind="local")

    # submit hook: a process-engine resident exists, so the gate would
    # escalate — and the over-approximate restore set is frozen per target
    d = tools._media_would_start_server(
        {"model": "gen/x", "target": "local", "modality": "music"})
    assert d is not None and d["models"] == ["chat-a"]
    assert tools._consume_approved_restore("*local") == {"chat-a"}

    # no process resident -> the hook certifies an empty set (fail-closed on
    # any process restore that appears mid-call) and does not escalate
    ledger.record_released("local", "chat-a")
    d = tools._media_would_start_server(
        {"model": "gen/x", "target": "local", "modality": "music"})
    assert d is None
    assert tools._consume_approved_restore("*local") == frozenset()

    # the inline release consumes the per-target manifest
    tools._record_approved_restore("*local", ["chat-a"])
    out = tools._release_media_lane(target, "media-abc")
    assert tools._consume_approved_restore("*local") is None
    assert "scope_id" in out
