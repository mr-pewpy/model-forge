"""RuntimeHandle / DockerContainerLifecycle tests (roadmap §5a, P5a-A1).

Everything external is faked — the docker runner, the health endpoint, the
clock and the sleep seam — so the suite touches no real docker and no
network, and runs instantly even for cold-boot-budget cases. The registry
``runtime`` field is a small, validated data shape; its parse rules fail
closed (a container runtime without a name, a health url, or a positive
budget is refused with a clear error, never a guessed default).
"""

from __future__ import annotations

import pytest

import model_forge.broker as broker
from model_forge.broker import (
    DockerContainerLifecycle,
    ProcessHandle,
    RuntimeHandle,
    VllmServerLifecycle,
    runtime_lifecycle_for,
)
from model_forge.registry import ModelEntry


# --- fakes ------------------------------------------------------------------

class FakeClock:
    """Injectable monotonic clock; tests advance time explicitly."""

    def __init__(self, now: float = 1_000.0):
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeProcess:
    """A fake RuntimeHandle carrying NO ``kind`` — it must still satisfy the
    process seam (poll/terminate/wait) the adapters call."""

    def __init__(self):
        self.terminated = False
        self.waited = False
        self._running = True

    def poll(self):
        return None if self._running else 0

    def terminate(self):
        self.terminated = True
        self._running = False

    def wait(self, timeout=None):
        self.waited = True
        return 0


class FakeProcessRunner:
    """Injected process runner (llama.cpp / vLLM seam): records argv."""

    def __init__(self, proc=None):
        self.argvs = []
        self.proc = proc or FakeProcess()

    def __call__(self, argv, env=None):
        self.argvs.append(list(argv))
        return self.proc


class _FakeDockerProc:
    """A readable docker child handle: ``communicate`` yields the inspect
    stdout the adapter reads (a fake standing in for Popen(stdout=PIPE))."""

    def __init__(self, stdout):
        self._stdout = stdout

    def communicate(self, timeout=None):
        return self._stdout, b""


class FakeDockerRunner:
    """Injected docker runner: records subcommands, optionally raises, and
    returns a handle.

    ``inspect_state`` controls the readable ``docker inspect`` output the
    default runner emits (``True``/``False`` -> ``b"true"``/``b"false"``);
    ``None`` keeps the handle opaque (no ``communicate``) so the probe must
    fall back to tracked state, exactly like an unreadable real child."""

    def __init__(self, fail=None, inspect_state=None):
        self.calls = []
        self.fail = fail            # 'start' | 'stop' | 'inspect' | None
        self.inspect_state = inspect_state

    def __call__(self, args):
        sub = args[0]
        self.calls.append([str(a) for a in args])
        if self.fail == sub:
            raise RuntimeError(f"docker {sub} boom")
        if sub == "inspect" and self.inspect_state is not None:
            return _FakeDockerProc(b"true" if self.inspect_state else b"false")
        return object()


class FakeHealth:
    """Injectable health seam: healthy after ``after`` polls (0 = always)."""

    def __init__(self, after: int = 0):
        self.after = after
        self.calls = 0

    def __call__(self, url, timeout=2.0):
        self.calls += 1
        return self.calls > self.after


@pytest.fixture
def health(monkeypatch):
    """Patch the module health probe and return the fake."""
    fake = FakeHealth()
    # the fake accepts the real probe's (url, timeout=...) call shape
    monkeypatch.setattr(broker, "_http_healthy", fake)
    return fake


class StagedDockerRunner:
    """Injected docker runner whose ``docker inspect`` reports running for the
    first ``running_polls`` reads (after a stop), then stopped — the seam the
    release-wait tests drive. Non-inspect subcommands return an opaque handle
    (no ``communicate``), so the state is only ever read from inspect here."""

    def __init__(self, running_polls: int = 2):
        self.calls: list[list[str]] = []
        self.running_polls = running_polls
        self.inspect_calls = 0

    def __call__(self, args):
        sub = args[0]
        self.calls.append([str(a) for a in args])
        if sub == "inspect":
            self.inspect_calls += 1
            running = self.inspect_calls <= self.running_polls
            return _FakeDockerProc(b"true" if running else b"false")
        return object()


def _container(**overrides):
    kwargs = dict(engine_id="comfyui", container="weights-broker-comfyui-1",
                  health_url="http://127.0.0.1:18188/system_stats",
                  boot_budget_s=120.0)
    kwargs.update(overrides)
    return DockerContainerLifecycle(**kwargs)


# --- RuntimeHandle contract -------------------------------------------------

def test_runtime_handle_declares_kind_and_the_three_methods():
    for name in ("poll", "terminate", "wait"):
        assert hasattr(RuntimeHandle, name), name
    assert "kind" in getattr(RuntimeHandle, "__annotations__", {})
    assert getattr(RuntimeHandle, "__annotations__", {}).get("kind") == "str"


def test_process_handle_is_an_alias_of_the_runtime_handle_contract():
    assert ProcessHandle is RuntimeHandle


def test_fake_process_handle_still_satisfies_the_vllm_seam():
    """Regression: the existing llama.cpp / vLLM process seam is unchanged —
    a handle with only poll/terminate/wait (no ``kind``) still drives
    VllmServerLifecycle end to end."""
    proc = FakeProcess()
    runner = FakeProcessRunner(proc)
    vllm = VllmServerLifecycle(runner=runner, which=lambda n: f"/usr/bin/{n}",
                               exists=lambda p: True)
    assert not hasattr(proc, "kind")
    assert vllm.ensure_loaded("m", weights_path="/w")["ok"] is True
    assert runner.argvs
    assert vllm.probe()["state"] == "running"
    assert vllm.release("m")["ok"] is True
    assert proc.terminated is True and proc.waited is True


# --- DockerContainerLifecycle.ensure_loaded --------------------------------

def test_ensure_loaded_starts_container_and_confirms_health(health):
    runner = FakeDockerRunner()
    dcl = _container(runner=runner, sleep=lambda s: None)
    res = dcl.ensure_loaded("krea-2-turbo")
    assert res["ok"] is True
    assert res["engine"] == "comfyui"
    assert res["state"] == "ready"
    assert ["start", "weights-broker-comfyui-1"] in runner.calls
    assert health.calls == 1              # health checked, no sleep needed
    assert "loading (~120s)" == res["loading_state"]
    # A second ensure on an already-running container issues no new start.
    before = list(runner.calls)
    again = dcl.ensure_loaded("krea-2-turbo")
    assert again["ok"] is True and again["state"] == "ready"
    assert runner.calls == before
    assert "already running" in again["reason"]


def test_ensure_loaded_times_out_on_the_cold_boot_budget(monkeypatch):
    clock = FakeClock()
    runner = FakeDockerRunner()
    monkeypatch.setattr(broker, "_http_healthy", lambda url, timeout=2.0: False)
    dcl = _container(runner=runner, boot_budget_s=8.0, clock=clock,
                     sleep=clock.advance)
    res = dcl.ensure_loaded("ltx-2.5")
    assert res["ok"] is False
    assert "cold-boot budget" in res["reason"]
    assert "8s" in res["reason"]
    assert ["start", "weights-broker-comfyui-1"] in runner.calls
    # budget honoured exactly: slept up to (not past) the budget window
    assert 8.0 <= clock.now - 1_000.0 < 10.0


def test_cold_boot_timeout_never_reports_stopped(monkeypatch):
    """Finding 3: docker start succeeded, so the container runs (possibly
    holding partial VRAM). After the budget, probe must NOT say 'stopped'."""
    clock = FakeClock()
    runner = FakeDockerRunner()                 # opaque inspect -> tracked flag
    monkeypatch.setattr(broker, "_http_healthy", lambda url, timeout=2.0: False)
    dcl = _container(runner=runner, boot_budget_s=4.0, clock=clock,
                     sleep=clock.advance)
    res = dcl.ensure_loaded("ltx-2.5")
    assert res["ok"] is False
    assert res["state"] == "booting"
    assert dcl._running is True                 # container is up, not stopped
    probe = dcl.probe()
    assert probe["state"] == "booting"
    assert "stopped" not in probe["reason"]
    # A re-ensure on a booting container issues no new start (honest retry).
    before = list(runner.calls)
    dcl.ensure_loaded("ltx-2.5")
    assert runner.calls == before


def test_default_runner_captures_inspect_output_and_trims_env(monkeypatch):
    """Finding 1 + 6: the default runner is Popen(docker ...) with PIPEs so
    inspect state is readable, and a trimmed env (no provider keys)."""
    captured = {}

    class _Popen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs

        def communicate(self, timeout=None):
            return b"true\n", b""

    monkeypatch.setattr(broker.subprocess, "Popen", _Popen)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    assert broker._docker_state(broker._run_docker, "c") is True
    assert captured["argv"] == ["docker", "inspect", "-f",
                                "{{.State.Running}}", "c"]
    assert captured["kwargs"]["stdout"] is broker.subprocess.PIPE
    assert captured["kwargs"]["stderr"] is broker.subprocess.PIPE
    assert "OPENAI_API_KEY" not in captured["kwargs"]["env"]
    assert "PATH" in captured["kwargs"]["env"]


def test_ensure_loaded_start_failure_is_an_honest_row():
    runner = FakeDockerRunner(fail="start")
    dcl = _container(runner=runner, sleep=lambda s: None)
    res = dcl.ensure_loaded("krea-2-turbo")
    assert res["ok"] is False
    assert "docker start" in res["reason"]
    assert "boom" in res["reason"]


# --- DockerContainerLifecycle.release / probe -------------------------------

def test_release_issues_docker_stop_and_frees_the_gpu(health):
    runner = FakeDockerRunner()
    dcl = _container(runner=runner, sleep=lambda s: None)
    dcl.ensure_loaded("krea-2-turbo")
    res = dcl.release("krea-2-turbo")
    assert res["ok"] is True
    assert res["state"] == "released"
    assert ["stop", "weights-broker-comfyui-1"] in runner.calls
    assert "instantly" in res["reason"]


def test_release_stop_failure_is_an_honest_row():
    runner = FakeDockerRunner(fail="stop")
    dcl = _container(runner=runner)
    res = dcl.release("krea-2-turbo")
    assert res["ok"] is False
    assert "docker stop" in res["reason"]


def test_release_waits_until_the_container_reports_stopped():
    """Finding 2: ``docker stop`` frees VRAM only when the container actually
    exits, so release must WAIT for stopped — not return the moment the stop
    was issued. Here inspect reports running twice, then stopped: release
    returns only after the container reaches stopped, having polled past the
    running observations."""
    clock = FakeClock()
    runner = StagedDockerRunner(running_polls=2)
    dcl = _container(runner=runner, sleep=clock.advance, clock=clock,
                     stop_budget_s=30.0)
    res = dcl.release("krea-2-turbo")
    assert res["ok"] is True and res["state"] == "released"
    assert ["stop", "weights-broker-comfyui-1"] in runner.calls
    assert "confirmed stopped" in res["reason"]
    # it kept asking docker until stopped was reported: 3 inspect reads.
    assert runner.inspect_calls == 3
    # it slept between the two running observations — the wait was real.
    assert clock.now == 1_000.0 + 2 * dcl.POLL_INTERVAL_S


def test_release_overrun_reports_stopping_and_never_claims_released(health):
    """Finding 2: on stop-budget overrun, release must return an honest
    non-released row ('stopping') — never claim 'released' while the
    container still runs (its VRAM is not yet free). The injectable clock
    keeps the bounded wait instant."""
    clock = FakeClock()
    runner = StagedDockerRunner(running_polls=1_000)   # never reports stopped
    dcl = _container(runner=runner, sleep=clock.advance, clock=clock,
                     stop_budget_s=5.0)
    assert dcl.ensure_loaded("krea-2-turbo")["ok"] is True
    assert dcl._running is True
    res = dcl.release("krea-2-turbo")
    assert res["ok"] is False
    assert res["state"] == "stopping"
    assert "NOT released" in res["reason"] and "budget" in res["reason"]
    # honest: the tracked state is NOT reset — the container is still up.
    assert dcl._running is True
    # the stop budget was honoured exactly (bounded, never a hang).
    assert clock.now - 1_000.0 >= 5.0


def test_release_joins_the_stop_handle_when_docker_state_is_unreadable():
    """Release falls back to the stop handle's own join (``docker stop``
    blocks until exit) when ``docker inspect`` gives no readable state, then
    reports released — never a false 'released' while the stop still runs."""
    clock = FakeClock()

    class _Handle:
        def __init__(self):
            self.waited = False
            self._done = False

        def wait(self, timeout=None):
            self.waited = True
            self._done = True
            return 0

        def poll(self):
            return 0 if self._done else None

    handle = _Handle()

    class _Runner:
        def __init__(self):
            self.calls = []

        def __call__(self, args):
            self.calls.append([str(a) for a in args])
            if args[0] == "stop":
                return handle
            return object()               # opaque inspect -> unreadable

    runner = _Runner()
    dcl = _container(runner=runner, sleep=clock.advance, clock=clock,
                     stop_budget_s=30.0)
    res = dcl.release("krea-2-turbo")
    assert res["ok"] is True and res["state"] == "released"
    assert handle.waited is True


def test_probe_fails_soft_when_the_runner_raises():
    runner = FakeDockerRunner(fail="inspect")
    dcl = _container(runner=runner)
    res = dcl.probe()
    # A failed probe reports ok False, matching the sibling adapters
    # (ollama / llama-swap / comfyui) — it still fails soft (no raise).
    assert res["ok"] is False
    assert res["state"] == "unknown"
    assert res["resident"] == []
    assert "fail-soft" in res["reason"]


def test_probe_reads_docker_inspect_and_overrides_a_stale_running_flag(health):
    """Finding 1: with a readable inspect child, docker is authoritative —
    a container docker reports stopped is reported stopped even though the
    in-memory flag says running (and vice-versa)."""
    runner = FakeDockerRunner(inspect_state=False)
    dcl = _container(runner=runner, sleep=lambda s: None)
    dcl.ensure_loaded("krea-2-turbo")           # sets _running True in memory
    assert dcl._running is True
    probe = dcl.probe()
    assert probe["state"] == "stopped"          # docker says stopped wins
    assert probe["resident"] == []
    assert dcl._running is False                # flag reset from observation
    assert "docker inspect" in probe["reason"]
    # ...and the other direction: docker reports running though we tracked stop
    dcl.release("krea-2-turbo")
    assert dcl._running is False
    runner.inspect_state = True
    probe = dcl.probe()
    assert probe["state"] == "running"
    assert dcl._running is True                 # flag reset from observation


def test_probe_does_not_claim_docker_inspect_when_output_is_unreadable():
    runner = FakeDockerRunner()                 # opaque handle, no communicate
    dcl = _container(runner=runner)
    probe = dcl.probe()
    assert probe["state"] == "stopped"
    assert "docker inspect unreadable" in probe["reason"]
    assert "(docker inspect)" not in probe["reason"]


def test_probe_reports_tracked_state_when_output_is_unreadable(health):
    runner = FakeDockerRunner()
    dcl = _container(runner=runner, sleep=lambda s: None)
    assert dcl.probe()["state"] == "stopped"
    dcl.ensure_loaded("krea-2-turbo")
    probe = dcl.probe()
    assert probe["state"] == "running"
    assert probe["resident"] == ["krea-2-turbo"]
    dcl.release("krea-2-turbo")
    assert dcl.probe()["state"] == "stopped"


# --- registry runtime field: parse rules ------------------------------------

def _entry(**overrides):
    kwargs = dict(id="comfy-sidecar", name="ComfyUI sidecar", modality="image",
                  engines=("comfyui",), source="Comfy-Org/Krea-2")
    kwargs.update(overrides)
    return ModelEntry(**kwargs)


def test_runtime_field_defaults_to_none():
    assert _entry().runtime is None


def test_runtime_container_shape_parses():
    rt = {"kind": "container", "name": "comfyui-1",
          "health_url": "http://127.0.0.1:18188/system_stats",
          "boot_budget_s": 300.0, "gpu_count": 1}
    m = _entry(runtime=rt)
    assert m.runtime == rt


def test_runtime_process_shape_parses():
    m = _entry(runtime={"kind": "process"})
    assert m.runtime["kind"] == "process"


def test_runtime_gpu_count_defaults_are_the_callers():
    # gpu_count is optional in the shape; the factory supplies the default.
    m = _entry(runtime={"kind": "container", "name": "c",
                        "health_url": "http://127.0.0.1:1/h",
                        "boot_budget_s": 60.0})
    assert "gpu_count" not in m.runtime


def test_runtime_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        _entry(runtime={"kind": "pod", "name": "c",
                        "health_url": "http://x", "boot_budget_s": 1.0})


def test_runtime_container_requires_name():
    with pytest.raises(ValueError, match="name"):
        _entry(runtime={"kind": "container", "health_url": "http://x",
                        "boot_budget_s": 60.0})


def test_runtime_container_requires_health_url():
    with pytest.raises(ValueError, match="health_url"):
        _entry(runtime={"kind": "container", "name": "c",
                        "boot_budget_s": 60.0})


def test_runtime_container_requires_a_positive_boot_budget():
    with pytest.raises(ValueError, match="boot_budget_s"):
        _entry(runtime={"kind": "container", "name": "c",
                        "health_url": "http://x", "boot_budget_s": 0})
    with pytest.raises(ValueError, match="boot_budget_s"):
        _entry(runtime={"kind": "container", "name": "c",
                        "health_url": "http://x"})


def test_runtime_rejects_bad_gpu_count():
    with pytest.raises(ValueError, match="gpu_count"):
        _entry(runtime={"kind": "container", "name": "c",
                        "health_url": "http://x", "boot_budget_s": 60.0,
                        "gpu_count": 0})


def test_runtime_rejects_bool_boot_budget():
    # Finding 5: float(True) == 1.0 must NOT slip through as a budget — a bool
    # is not a number, same fail-closed treatment as gpu_count.
    with pytest.raises(ValueError, match="boot_budget_s"):
        _entry(runtime={"kind": "container", "name": "c",
                        "health_url": "http://x", "boot_budget_s": True})


def test_runtime_must_be_a_dict():
    with pytest.raises(ValueError, match="dict"):
        _entry(runtime="container")


# --- runtime_lifecycle_for factory ------------------------------------------

def test_runtime_lifecycle_for_builds_a_container_lifecycle():
    m = _entry(runtime={"kind": "container", "name": "comfyui-1",
                        "health_url": "http://127.0.0.1:18188/system_stats",
                        "boot_budget_s": 300.0})
    adapter = runtime_lifecycle_for(m)
    assert isinstance(adapter, DockerContainerLifecycle)
    assert adapter.engine_id == "comfyui"


def test_runtime_lifecycle_for_returns_none_for_process_and_missing():
    assert runtime_lifecycle_for(_entry()) is None
    assert runtime_lifecycle_for(_entry(runtime={"kind": "process"})) is None


def test_runtime_lifecycle_for_fails_closed_on_unmappable_shape():
    """Finding 7: parse validation is fail-closed, so the factory must be too
    — a runtime shape it cannot map raises with a clear reason, never None."""

    class _Bare:                       # a duck-typed entry with no validation
        id = "bare"
        engines = ("comfyui",)
        runtime = None

    bare = _Bare()
    setattr(bare, "runtime", "container")          # non-dict runtime
    with pytest.raises(ValueError, match="dict"):
        runtime_lifecycle_for(bare)
    setattr(bare, "runtime", {"kind": "pod", "name": "c"})   # unknown kind
    with pytest.raises(ValueError, match="kind"):
        runtime_lifecycle_for(bare)


def test_runtime_lifecycle_for_missing_keys_name_the_key():
    """A duck-typed entry (no parse validation) whose container runtime dict
    lacks ``name``/``health_url`` must fail with a clear ValueError NAMING the
    missing key — the promised fail-closed posture — never a bare KeyError
    from the constructor call."""
    class _Bare:                       # a duck-typed entry with no validation
        id = "bare"
        engines = ("comfyui",)
        runtime = None

    bare = _Bare()
    setattr(bare, "runtime", {"kind": "container", "health_url": "http://x",
                              "boot_budget_s": 60.0})          # no name
    with pytest.raises(ValueError, match="no name"):
        runtime_lifecycle_for(bare)
    setattr(bare, "runtime", {"kind": "container", "name": "c",
                              "boot_budget_s": 60.0})       # no health_url
    with pytest.raises(ValueError, match="no health_url"):
        runtime_lifecycle_for(bare)


def test_runtime_lifecycle_for_passes_seams_and_engine_id_override():
    clock = FakeClock()
    runner = FakeDockerRunner()
    m = _entry(engines=("comfyui",),
               runtime={"kind": "container", "name": "c",
                        "health_url": "http://127.0.0.1:1/h",
                        "boot_budget_s": 30.0})
    adapter = runtime_lifecycle_for(m, runner=runner, clock=clock,
                                    sleep=clock.advance, engine_id="comfy-c")
    assert adapter.engine_id == "comfy-c"
    # the factory wired the injected seams through: a boot uses them
    import model_forge.broker as b
    orig = b._http_healthy
    b._http_healthy = lambda url, timeout=2.0: True
    try:
        res = adapter.ensure_loaded("m")
    finally:
        b._http_healthy = orig
    assert res["ok"] is True and ["start", "c"] in runner.calls
