"""B8 — server provenance + the approval gate for stopping externally-started
engine servers.

Model Forge may only stop a server it started itself without asking. When a
swap/release would ``docker stop`` a runtime the USER started (an
externally-started container, e.g. one brought up by compose), that stop rides
the human-approval gate (rule_key ``model-forge:broker_server_stop``). The
signal is honest provenance tracked on the container lifecycle: a container
this adapter started is ``plugin`` (ungated to stop — gated at start), one it
adopted already-running is ``external`` (gated to stop). An unreadable prior
state is fail-safe ``external``.
"""

import types

from model_forge.broker import (
    DockerContainerLifecycle,
    EngineLifecycle,
    PlanStep,
)


class _FakeProc:
    def __init__(self, out=b""):
        self._out = out

    def communicate(self, timeout=None):
        return (self._out, b"")


class _FakeDockerRunner:
    """Records docker calls; ``inspect`` answers a configured running state
    (True / False / None for unreadable). ``stop`` flips the container to
    stopped so the release await terminates. Never touches real docker."""

    def __init__(self, running=None):
        self.calls = []
        self.running = running

    def __call__(self, args):
        self.calls.append(list(args))
        if args and args[0] == "inspect":
            if self.running is True:
                return _FakeProc(b"true\n")
            if self.running is False:
                return _FakeProc(b"false\n")
            return _FakeProc(b"")
        if args and args[0] == "stop":
            self.running = False
        return _FakeProc(b"")


def _docker_lifecycle(running):
    runner = _FakeDockerRunner(running=running)
    # A clock that advances on every read so any bounded wait terminates.
    ticks = {"n": 0}

    def clock():
        ticks["n"] += 1
        return float(ticks["n"])

    lc = DockerContainerLifecycle(
        "comfyui", container="comfy", health_url="http://127.0.0.1:1/health",
        boot_budget_s=0.0, runner=runner,
        sleep=lambda s: None, clock=clock)
    return lc, runner


def test_base_lifecycle_never_stops_an_external_server():
    # Process and HTTP adapters only stop what they started (or never stop a
    # server at all), so the base signal is False — no gate.
    class _P(EngineLifecycle):
        def ensure_loaded(self, model_id, **kwargs):
            return {}

        def release(self, model_id):
            return {}

        def probe(self):
            return {}

    assert _P().stops_externally_started("m") is False


def test_adopted_container_is_external_and_gated(monkeypatch):
    # A container already running (the user started it) is adopted as
    # external: stopping it WOULD stop a server the user started.
    monkeypatch.setattr("model_forge.broker._http_healthy",
                        lambda url, timeout=2.0: True)
    lc, runner = _docker_lifecycle(running=True)
    lc.ensure_loaded("krea-2-turbo")
    # adopted, never started by this adapter -> external
    assert lc.stops_externally_started() is True
    rel = lc.release("krea-2-turbo")
    assert rel["stopped_external"] is True
    assert rel["provenance"] == "external"


def test_plugin_started_container_is_plugin_owned(monkeypatch):
    # A container this adapter started is plugin-owned: stopping it is the
    # symmetric counterpart of an already-gated start — no extra gate.
    monkeypatch.setattr("model_forge.broker._http_healthy",
                        lambda url, timeout=2.0: True)
    lc, runner = _docker_lifecycle(running=False)
    lc.ensure_loaded("krea-2-turbo")
    assert lc.stops_externally_started() is False
    rel = lc.release("krea-2-turbo")
    assert rel["stopped_external"] is False
    assert rel["provenance"] == "plugin"


def test_unreadable_prior_state_is_fail_safe_external(monkeypatch):
    # If the container's prior state cannot be read, provenance is uncertain:
    # fail safe and treat it as external (a stop is gated).
    monkeypatch.setattr("model_forge.broker._http_healthy",
                        lambda url, timeout=2.0: True)
    lc, runner = _docker_lifecycle(running=None)
    lc.ensure_loaded("krea-2-turbo")
    assert lc.stops_externally_started() is True


# --- the approval gate (policy_hook) ----------------------------------------

class _Adapter:
    def __init__(self, external):
        self._external = external

    def stops_externally_started(self, model_id=""):
        return self._external


class _Planner:
    def __init__(self, release_steps):
        self._release_steps = release_steps

    def plan_load_many(self, target_id, requests, **kw):
        return types.SimpleNamespace(release_steps=self._release_steps,
                                     load_steps=())


def _gate_env(monkeypatch, *, external):
    from model_forge import tools
    target = types.SimpleNamespace(id="gpu-box", kind="lan")
    monkeypatch.setattr(tools, "get_target", lambda *_a, **_k: target)
    steps = (PlanStep(action="release", target_id="gpu-box",
                      model_id="krea-2-turbo", engine="comfyui",
                      reason="make room"),)
    stack = {"adapters": {"comfyui": _Adapter(external)},
             "planner": _Planner(steps),
             "orchestrator": types.SimpleNamespace(),
             "ledger": types.SimpleNamespace()}
    monkeypatch.setattr(tools, "broker_stack", lambda *_a, **_k: stack)
    return tools


def test_gate_escalates_a_swap_that_stops_an_external_server(monkeypatch):
    tools = _gate_env(monkeypatch, external=True)
    d = tools.policy_hook("model_forge_broker",
                          {"action": "swap", "target": "gpu-box",
                           "model_id": "krea-2-turbo"})
    assert d is not None and d["action"] == "approve"
    assert d["rule_key"] == "model-forge:broker_server_stop"
    assert "started yourself" in d["message"]
    assert "comfyui" in d["message"]
    assert "krea-2-turbo" in d["message"]
    assert "stops nothing" in d["message"]


def test_gate_allows_a_swap_that_stops_a_plugin_owned_server(monkeypatch):
    tools = _gate_env(monkeypatch, external=False)
    assert tools.policy_hook("model_forge_broker",
                             {"action": "swap", "target": "gpu-box",
                              "model_id": "krea-2-turbo"}) is None
