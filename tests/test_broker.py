"""Residency-broker tests: ledger, engine lifecycle adapters, eviction policy,
and the orchestrator facade.

Everything external is injected — fake clocks, fake HTTP functions, fake
process runners, recording engine adapters — so the suite runs with no
network and no processes, and every engine call shape is asserted from the
fakes' recorded calls. The policy cases mirror roadmap §3: nominal booked
ceilings gate fit (live free VRAM only orders), lane wakes stop side models
first, the ~600 s recency window keeps models warm at the boundary, image
generation evicts the whole card every time, and loads fire sequentially.
"""

from __future__ import annotations

from model_forge.broker import (
    Booking,
    ComfyUiLifecycle,
    EngineLifecycle,
    EvictionPlanner,
    ExoLifecycle,
    LlamaCppServerLifecycle,
    LlamaSwapLifecycle,
    LoadPlan,
    LoadRequest,
    MayaLifecycle,
    MeasuredFootprintRecorder,
    OllamaLifecycle,
    Orchestrator,
    ResidencyLedger,
    SWAP_HISTORY_LIMIT,
    VllmServerLifecycle,
    _minimal_engine_env,
    _non_loopback_bind,
    _spawn_process,
    is_image_generation,
    loading_state,
)


# --- fakes ------------------------------------------------------------------

class FakeClock:
    """Injectable monotonic clock; tests advance time explicitly."""

    def __init__(self, now: float = 1_000.0):
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeHttp:
    """Injected HTTP seam: records every call, returns canned JSON per URL
    suffix (an Exception value is raised instead)."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    def __call__(self, method, url, body=None, timeout=None):
        self.calls.append({"method": method, "url": url, "body": body,
                           "timeout": timeout})
        for suffix, resp in self.responses.items():
            if url.endswith(suffix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return {}


class FakeProcess:
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


class FakeRunner:
    """Injected process runner: records argv and env, returns one FakeProcess."""

    def __init__(self):
        self.argvs = []
        self.envs = []
        self.proc = FakeProcess()

    def __call__(self, argv, env=None):
        self.argvs.append(list(argv))
        self.envs.append(dict(env) if env is not None else None)
        return self.proc


class RecordingLifecycle(EngineLifecycle):
    """Engine adapter that records call shapes and can fail on demand."""

    def __init__(self, engine_id="recording", fail_load=(), fail_release=()):
        self.engine_id = engine_id
        self.calls = []
        self.fail_load = set(fail_load)
        self.fail_release = set(fail_release)

    def ensure_loaded(self, model_id, **kwargs):
        self.calls.append(("ensure_loaded", model_id, kwargs))
        if model_id in self.fail_load:
            return {"ok": False, "reason": f"fake load failure for {model_id}"}
        return {"ok": True, "reason": "warmed"}

    def release(self, model_id):
        self.calls.append(("release", model_id, {}))
        if model_id in self.fail_release:
            return {"ok": False, "reason": f"fake release failure for "
                                          f"{model_id}"}
        return {"ok": True, "reason": "evicted"}

    def probe(self):
        return {"ok": True, "state": "observed", "resident": []}


def _call_names(adapter):
    return [(c[0], c[1]) for c in adapter.calls]


# --- ledger -----------------------------------------------------------------

def test_ledger_tracks_pulls_residency_and_bookings():
    ledger = ResidencyLedger(bookings={"m-a": 8.0,
                                       "m-b": {"vram_gb": 4.5,
                                               "source": "measured",
                                               "load_s": 35.0}})
    assert ledger.booked_vram_gb("m-a") == 8.0
    assert ledger.booking("m-a").source == "registry"
    assert ledger.booked_vram_gb("m-b") == 4.5
    assert ledger.booking("m-b").source == "measured"
    assert ledger.estimated_load_s("m-b") == 35.0
    assert ledger.booked_vram_gb("m-unknown") is None

    ledger.record_pull("local", "m-a", "/dl/m-a.gguf", size_bytes=1234)
    pulls = ledger.pulled("local")
    assert len(pulls) == 1
    assert pulls[0].path == "/dl/m-a.gguf"
    assert pulls[0].size_bytes == 1234

    ledger.record_resident("local", "m-a", engine="ollama", lane="chat")
    assert ledger.is_resident("local", "m-a")
    assert ledger.residents("local")[0].engine == "ollama"
    assert ledger.booked_total_gb("local") == 8.0
    assert ledger.record_released("local", "m-a") is True
    assert not ledger.is_resident("local", "m-a")


def test_measured_booking_replaces_registry_but_never_downgraded():
    ledger = ResidencyLedger()
    ledger.book("m", 8.0, source="registry")
    ledger.book("m", 9.5, source="measured")
    assert ledger.booked_vram_gb("m") == 9.5
    # A Booking instance books too (the measured-peaks shape).
    ledger.book("m-c", Booking(3.0, source="measured", load_s=12.0))
    assert ledger.booked_vram_gb("m-c") == 3.0
    assert ledger.estimated_load_s("m-c") == 12.0
    # A registry footprint never downgrades a measured peak.
    row = ledger.book("m", 8.0, source="registry")
    assert row["ok"] and row["kept"].startswith("measured booking kept")
    assert ledger.booked_vram_gb("m") == 9.5


def test_ledger_rejects_bad_numbers_and_touch_updates_recency():
    clock = FakeClock()
    ledger = ResidencyLedger(clock=clock)
    assert ledger.book("m", float("nan"))["ok"] is False
    assert ledger.book("m", -1)["ok"] is False
    assert ledger.set_budget("t", "not-a-number")["ok"] is False
    assert ledger.booked_vram_gb("m") is None

    ledger.set_budget("t", 24.0)
    ledger.record_resident("t", "m", engine="ollama")
    first = ledger.last_used("t", "m")
    clock.advance(100.0)
    assert ledger.touch("t", "m") is True
    assert ledger.last_used("t", "m") == first + 100.0
    assert ledger.touch("t", "nope") is False


def test_free_vram_snapshot_is_recorded_for_ordering():
    ledger = ResidencyLedger()
    assert ledger.free_vram_gb("t") is None
    assert ledger.set_free_vram("t", 2.25)["ok"] is True
    assert ledger.free_vram_gb("t") == 2.25
    assert ledger.set_free_vram("t", -3.0)["ok"] is False


# --- adapter call shapes: Ollama -------------------------------------------

def test_ollama_ensure_loaded_warms_with_keep_alive():
    http = FakeHttp()
    ollama = OllamaLifecycle(base_url="http://127.0.0.1:11434", http=http,
                             keep_alive="5m")
    res = ollama.ensure_loaded("llama3")
    assert res["ok"] is True
    assert res["keep_alive"] == "5m"
    call = http.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "http://127.0.0.1:11434/api/generate"
    assert call["body"]["model"] == "llama3"
    assert call["body"]["keep_alive"] == "5m"
    assert call["body"]["prompt"] == ""
    assert call["body"]["stream"] is False


def test_ollama_release_uses_keep_alive_zero():
    http = FakeHttp()
    ollama = OllamaLifecycle(http=http, keep_alive="5m")
    res = ollama.release("llama3")
    assert res["ok"] is True
    assert res["keep_alive"] == 0
    call = http.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/generate")
    assert call["body"]["model"] == "llama3"
    assert call["body"]["keep_alive"] == 0


def test_ollama_probe_reads_api_ps_and_fails_soft():
    http = FakeHttp(responses={"/api/ps": {"models": [{"name": "llama3"},
                                                     {"name": "nomic-embed"}]}})
    ollama = OllamaLifecycle(http=http)
    res = ollama.probe()
    assert res["ok"] is True
    assert res["resident"] == ["llama3", "nomic-embed"]
    assert http.calls[0]["method"] == "GET"
    assert http.calls[0]["url"].endswith("/api/ps")

    down = OllamaLifecycle(http=FakeHttp(
        responses={"/api/ps": Exception("connection refused")}))
    res = down.probe()
    assert res["ok"] is False
    assert res["state"] == "unknown"
    assert "connection refused" in res["reason"]


def test_ollama_http_failure_returns_honest_rows_not_raises():
    http = FakeHttp(responses={"/api/generate": Exception("boom")})
    ollama = OllamaLifecycle(http=http)
    assert ollama.ensure_loaded("m")["ok"] is False
    assert ollama.release("m")["ok"] is False


# --- adapter call shapes: llama-swap (pass-through) -------------------------

def test_llama_swap_is_observational_pass_through():
    http = FakeHttp(responses={"/models": {"models": [{"name": "qwen"}]}})
    swap = LlamaSwapLifecycle(base_url="http://127.0.0.1:8999", http=http)
    loaded = swap.ensure_loaded("qwen")
    released = swap.release("qwen")
    # No eviction control: ensure_loaded/release must NOT drive the engine.
    assert http.calls == []
    for row in (loaded, released):
        assert row["ok"] is True
        assert row["state"] == "pass-through"
        assert "no explicit load or eviction control" in row["reason"]
        assert "on demand by route" in row["reason"]

    probe = swap.probe()
    assert probe["ok"] is True
    assert probe["resident"] == ["qwen"]
    assert http.calls[0]["method"] == "GET"
    assert http.calls[0]["url"].endswith("/models")

    down = LlamaSwapLifecycle(http=FakeHttp(
        responses={"/models": Exception("nope")}))
    assert down.probe()["state"] == "unknown"


# --- adapter call shapes: llama.cpp server ---------------------------------

def test_llamacpp_ensure_loaded_starts_process_with_argv():
    runner = FakeRunner()
    server = LlamaCppServerLifecycle(
        binary="llama-server", port=8080, extra_args=("--threads", "4"),
        runner=runner, which=lambda name: f"/usr/bin/{name}")
    res = server.ensure_loaded("qwen", weights_path="/models/qwen.gguf")
    assert res["ok"] is True
    assert runner.argvs == [["/usr/bin/llama-server", "-m", "/models/qwen.gguf",
                             "--host", "127.0.0.1", "--port", "8080",
                             "--threads", "4"]]
    assert res["argv"] == runner.argvs[0]


def test_llamacpp_release_terminates_process():
    runner = FakeRunner()
    server = LlamaCppServerLifecycle(runner=runner,
                                     which=lambda name: f"/usr/bin/{name}")
    server.ensure_loaded("qwen", weights_path="/models/qwen.gguf")
    res = server.release("qwen")
    assert res["ok"] is True
    assert runner.proc.terminated is True
    assert runner.proc.waited is True
    # Second release: nothing left to stop, honestly reported.
    assert "nothing to stop" in server.release("qwen")["reason"]


def test_llamacpp_probe_reports_process_state():
    runner = FakeRunner()
    server = LlamaCppServerLifecycle(runner=runner,
                                     which=lambda name: f"/usr/bin/{name}")
    assert server.probe() == {"ok": True, "engine": "llamacpp",
                              "state": "idle", "resident": []}
    server.ensure_loaded("qwen", weights_path="/models/qwen.gguf")
    probe = server.probe()
    assert probe["state"] == "running"
    assert probe["resident"] == ["qwen"]
    server.release("qwen")
    assert server.probe()["state"] == "idle"


def test_llamacpp_missing_binary_fails_closed():
    runner = FakeRunner()
    server = LlamaCppServerLifecycle(runner=runner, which=lambda name: None)
    res = server.ensure_loaded("qwen", weights_path="/models/qwen.gguf")
    assert res["ok"] is False
    assert "fail-closed" in res["reason"]
    assert "not found" in res["reason"]
    # Fail closed: no argv built, nothing spawned.
    assert runner.argvs == []


def test_llamacpp_refuses_to_guess_weights_path():
    runner = FakeRunner()
    server = LlamaCppServerLifecycle(runner=runner,
                                     which=lambda name: f"/usr/bin/{name}")
    res = server.ensure_loaded("qwen")
    assert res["ok"] is False
    assert "fail-closed" in res["reason"]
    assert runner.argvs == []


# --- adapter call shapes: ComfyUI (readiness + cache-free) ------------------

def test_comfyui_ensure_loaded_is_readiness_not_a_weights_pin():
    http = FakeHttp(responses={"/system_stats": {"system": {"comfyui-version":
                                                           "0.3.0"}}})
    comfy = ComfyUiLifecycle(base_url="http://127.0.0.1:8188", http=http)
    res = comfy.ensure_loaded("sdxl")
    assert res["ok"] is True
    assert res["state"] == "ready"
    assert res["contract"] == "readiness + cache-free"
    assert res["cache_freed"] is False
    # Honest behavior label: ComfyUI loads models per-workflow — this is a
    # readiness + cache-free contract, never a weights pin.
    assert "per-workflow" in res["reason"]
    assert "not a weights pin" in res["reason"]
    # First job: readiness only — GET /system_stats, nothing freed yet.
    assert [c["method"] for c in http.calls] == ["GET"]
    assert http.calls[0]["url"].endswith("/system_stats")


def test_comfyui_back_to_back_jobs_free_cache_between_jobs():
    http = FakeHttp(responses={"/system_stats": {"devices": [{"name": "gpu"}]}})
    comfy = ComfyUiLifecycle(http=http)
    comfy.ensure_loaded("sdxl")           # job 1: nothing held before it
    res = comfy.ensure_loaded("sdxl")     # job 2: back-to-back — free first
    assert res["ok"] is True
    assert res["cache_freed"] is True
    assert "back-to-back" in res["reason"]
    free_calls = [c for c in http.calls if c["url"].endswith("/free")]
    assert len(free_calls) == 1
    # The battle rule's call shape: POST /free with both reclaim flags.
    assert free_calls[0]["method"] == "POST"
    assert free_calls[0]["body"] == {"unload_models": True, "free_memory": True}
    # The free lands between the jobs: during job 2's ensure_loaded, after
    # its readiness check — not before job 1.
    assert http.calls[1]["url"].endswith("/system_stats")
    assert http.calls[2]["url"].endswith("/free")
    # A third consecutive job frees again: every back-to-back boundary.
    res3 = comfy.ensure_loaded("sdxl")
    assert res3["cache_freed"] is True
    assert len([c for c in http.calls if c["url"].endswith("/free")]) == 2


def test_comfyui_release_is_the_explicit_cache_free():
    http = FakeHttp(responses={"/system_stats": {"system": {}}})
    comfy = ComfyUiLifecycle(http=http)
    # Release with nothing held: an honest no-op, nothing sent.
    res = comfy.release("sdxl")
    assert res["ok"] is True
    assert res["cache_freed"] is False
    assert "nothing to free" in res["reason"]
    assert http.calls == []

    comfy.ensure_loaded("sdxl")
    http.calls.clear()
    res = comfy.release("sdxl")
    assert res["ok"] is True
    assert res["cache_freed"] is True
    assert res["contract"] == "explicit cache-free"
    assert "holds allocations" in res["reason"]
    call = http.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/free")
    assert call["body"] == {"unload_models": True, "free_memory": True}

    # With the cache freed, the next job is no longer back-to-back.
    http.calls.clear()
    res = comfy.ensure_loaded("sdxl")
    assert res["cache_freed"] is False
    assert [c["url"].rsplit("/", 1)[-1] for c in http.calls] == ["system_stats"]


def test_comfyui_http_failures_are_honest_rows_not_raises():
    down = ComfyUiLifecycle(http=FakeHttp(
        responses={"/system_stats": Exception("connection refused")}))
    res = down.ensure_loaded("sdxl")
    assert res["ok"] is False
    assert "connection refused" in res["reason"]
    assert down.probe()["state"] == "unknown"

    # A failing cache free is an honest row — never a raise.
    http = FakeHttp(responses={"/system_stats": {"system": {}},
                              "/free": Exception("boom")})
    comfy = ComfyUiLifecycle(http=http)
    comfy.ensure_loaded("sdxl")
    res = comfy.release("sdxl")
    assert res["ok"] is False
    assert "boom" in res["reason"]
    assert res["cache_freed"] is False


def test_comfyui_probe_reports_readiness_without_a_residency_listing():
    http = FakeHttp(responses={"/system_stats": {"system": {"comfyui-version":
                                                           "0.3.0"}}})
    comfy = ComfyUiLifecycle(http=http)
    res = comfy.probe()
    assert res["ok"] is True
    assert res["state"] == "observed"
    # ComfyUI exposes no weights-residency listing — honest empty resident.
    assert res["resident"] == []
    assert "per-workflow" in res["reason"]


# --- adapter call shapes: vLLM server (process stop/start only) -------------

def test_vllm_ensure_loaded_starts_process_with_argv():
    runner = FakeRunner()
    vllm = VllmServerLifecycle(
        binary="vllm", port=8000, extra_args=("--tensor-parallel-size", "2"),
        runner=runner, which=lambda name: f"/usr/bin/{name}",
        exists=lambda p: True)
    res = vllm.ensure_loaded("qwen", weights_path="/models/qwen")
    assert res["ok"] is True
    assert runner.argvs == [["/usr/bin/vllm", "serve", "/models/qwen",
                             "--host", "127.0.0.1", "--port", "8000",
                             "--tensor-parallel-size", "2"]]
    assert res["argv"] == runner.argvs[0]
    # Process restart only — engine sleep/wake must NOT be used.
    assert "sleep/wake" in res["lifecycle"]
    assert "process restart is the reliable path" in res["reason"]

    # The model id is the launch argument when no weights path is given (the
    # constructor's extra args still apply).
    res = vllm.ensure_loaded("meta/llama-3")
    assert res["ok"] is True
    assert runner.argvs[-1] == ["/usr/bin/vllm", "serve", "meta/llama-3",
                                "--host", "127.0.0.1", "--port", "8000",
                                "--tensor-parallel-size", "2"]


def test_vllm_release_stops_the_process():
    runner = FakeRunner()
    vllm = VllmServerLifecycle(runner=runner,
                               which=lambda name: f"/usr/bin/{name}",
                               exists=lambda p: True)
    vllm.ensure_loaded("qwen", weights_path="/models/qwen")
    res = vllm.release("qwen")
    assert res["ok"] is True
    assert runner.proc.terminated is True
    assert runner.proc.waited is True
    assert "stopped the vLLM server" in res["reason"]
    # Second release: nothing left to stop, honestly reported.
    assert "nothing to stop" in vllm.release("qwen")["reason"]


def test_vllm_probe_reports_process_state():
    runner = FakeRunner()
    vllm = VllmServerLifecycle(runner=runner,
                               which=lambda name: f"/usr/bin/{name}",
                               exists=lambda p: True)
    assert vllm.probe() == {"ok": True, "engine": "vllm", "state": "idle",
                            "resident": []}
    vllm.ensure_loaded("qwen", weights_path="/models/qwen")
    probe = vllm.probe()
    assert probe["state"] == "running"
    assert probe["resident"] == ["qwen"]
    vllm.release("qwen")
    assert vllm.probe()["state"] == "idle"


def test_vllm_absent_binary_fails_closed():
    runner = FakeRunner()
    vllm = VllmServerLifecycle(runner=runner, which=lambda name: None,
                               exists=lambda p: True)
    res = vllm.ensure_loaded("qwen", weights_path="/models/qwen")
    assert res["ok"] is False
    assert "fail-closed" in res["reason"]
    assert "not found" in res["reason"]
    # Fail closed: no argv built, nothing spawned.
    assert runner.argvs == []


def test_vllm_missing_weights_fails_closed():
    runner = FakeRunner()
    vllm = VllmServerLifecycle(runner=runner,
                               which=lambda name: f"/usr/bin/{name}",
                               exists=lambda p: False)
    res = vllm.ensure_loaded("qwen", weights_path="/models/gone")
    assert res["ok"] is False
    assert "fail-closed" in res["reason"]
    assert "not found" in res["reason"]
    assert runner.argvs == []


# --- adapter call shapes: Maya (engine-native, no unload primitive) ---------

def test_maya_ensure_loaded_maps_only_what_exists():
    http = FakeHttp(responses={"/api/ps": {"models": [{"name": "glm-flash"}]}})
    maya = MayaLifecycle(base_url="http://127.0.0.1:8080", http=http)
    res = maya.ensure_loaded("glm-flash")
    assert res["ok"] is True
    assert res["state"] == "resident"
    assert http.calls[0]["method"] == "GET"
    assert http.calls[0]["url"].endswith("/api/ps")

    # Not yet loaded: readiness over /api/show — loading is engine-native.
    http2 = FakeHttp(responses={"/api/ps": {"models": []},
                               "/api/show": {"name": "glm-flash"}})
    maya2 = MayaLifecycle(http=http2)
    res = maya2.ensure_loaded("glm-flash")
    assert res["ok"] is True
    assert res["state"] == "ready-on-demand"
    assert "auto-swap" in res["load_control"]
    assert "not a weights pin" in res["reason"]
    show = http2.calls[-1]
    assert show["method"] == "POST"
    assert show["url"].endswith("/api/show")
    assert show["body"] == {"name": "glm-flash"}


def test_maya_release_never_fakes_an_unload():
    http = FakeHttp()
    maya = MayaLifecycle(http=http)
    res = maya.release("glm-flash")
    assert res["ok"] is True
    # The honest label: Maya has no unload primitive and none is faked.
    assert res["unload_control"] == "no-unload-primitive"
    assert "no unload primitive" in res["reason"]
    assert "never faked" in res["reason"]
    # Nothing is sent to the engine — especially never /api/delete.
    assert http.calls == []


def test_maya_unknown_model_and_down_engine_are_honest_rows():
    http = FakeHttp(responses={"/api/ps": {"models": []}, "/api/show": None})
    maya = MayaLifecycle(http=http)
    res = maya.ensure_loaded("no-such-model")
    assert res["ok"] is False
    assert "does not know" in res["reason"]

    down = MayaLifecycle(http=FakeHttp(
        responses={"/api/ps": Exception("connection refused")}))
    res = down.ensure_loaded("glm-flash")
    assert res["ok"] is False
    assert "connection refused" in res["reason"]


def test_maya_probe_reads_api_ps_and_fails_soft():
    http = FakeHttp(responses={"/api/ps": {"models": [{"name": "a"},
                                                     {"name": "b"}]}})
    maya = MayaLifecycle(http=http)
    res = maya.probe()
    assert res["ok"] is True
    assert res["state"] == "observed"
    assert res["resident"] == ["a", "b"]
    assert http.calls[0]["url"].endswith("/api/ps")

    down = MayaLifecycle(http=FakeHttp(
        responses={"/api/ps": Exception("down")}))
    assert down.probe()["state"] == "unknown"

# --- adapter call shapes: Exo (instance placement; real unload primitive) ----

def test_exo_ensure_loaded_places_instance_or_observes_resident():
    http = FakeHttp(responses={
        "/state": {"topology": {}, "instances": {}, "deployments": {}},
        "/place_instance": {
            "message": "Command received.", "command_id": "cmd-1",
            "model_card": {"model_id": "mlx-community/Llama-3.2-1B"},
            "deployment_id": "dpl-1"},
    })
    exo = ExoLifecycle(base_url="http://127.0.0.1:52415", http=http)
    res = exo.ensure_loaded("mlx-community/Llama-3.2-1B")
    assert res["ok"] is True
    # honest state: placement requested, never claimed loaded — the engine
    # downloads and loads asynchronously
    assert res["state"] == "placement-requested"
    assert res["deployment_id"] == "dpl-1"
    assert "asynchronously" in res["reason"]
    assert "instance/await" in res["reason"]
    place = http.calls[-1]
    assert place["method"] == "POST"
    assert place["url"].endswith("/place_instance")
    # server-side placement: only the model id is sent, nothing invented
    assert place["body"] == {"model_id": "mlx-community/Llama-3.2-1B"}

    # Already placed: observed from GET /state — never re-launched.
    http2 = FakeHttp(responses={"/state": {"instances": {
        "inst-9": {"instance_id": "inst-9", "deployment_id": "dpl-9",
                   "shard_assignments": {"model_id": "glm"}}}}})
    exo2 = ExoLifecycle(http=http2)
    res = exo2.ensure_loaded("glm")
    assert res["ok"] is True and res["state"] == "resident"
    assert res["instance_id"] == "inst-9"
    assert all(c["method"] == "GET" for c in http2.calls)  # nothing launched


def test_exo_release_stops_by_deployment_id():
    http = FakeHttp(responses={
        "/place_instance": {"message": "Command received.",
                            "command_id": "cmd-1", "deployment_id": "dpl-1"},
        "/instance/dpl-1": {"message": "Command received.",
                            "command_id": "cmd-2", "instance_id": "dpl-1"},
    })
    exo = ExoLifecycle(http=http)
    exo.ensure_loaded("m")
    http.calls.clear()
    res = exo.release("m")
    assert res["ok"] is True
    # the real unload primitive: DELETE /instance/{deployment_id} —
    # documented to stop the deployment even while it waits to be placed
    assert res["unload_control"] == ExoLifecycle.UNLOAD_CONTROL
    assert "unload primitive" in res["reason"]
    assert len(http.calls) == 1
    call = http.calls[0]
    assert call["method"] == "DELETE"
    assert call["url"].endswith("/instance/dpl-1")


def test_exo_release_falls_back_to_state_lookup():
    http = FakeHttp(responses={
        "/state": {"instances": {
            "inst-7": {"instance_id": "inst-7", "deployment_id": "dpl-7",
                       "shard_assignments": {"model_id": "glm"}}}},
        "/instance/inst-7": {"message": "Command received.",
                             "command_id": "c", "instance_id": "inst-7"},
    })
    exo = ExoLifecycle(http=http)   # fresh adapter: no recorded deployment
    res = exo.release("glm")
    assert res["ok"] is True
    assert http.calls[-1]["method"] == "DELETE"
    assert http.calls[-1]["url"].endswith("/instance/inst-7")


def test_exo_release_nothing_running_is_an_honest_noop():
    http = FakeHttp(responses={"/state": {"instances": {}}})
    exo = ExoLifecycle(http=http)
    res = exo.release("glm")
    assert res["ok"] is True
    assert "nothing to stop" in res["reason"]
    # nothing deleted, nothing faked — the only call was the /state check
    assert [c["method"] for c in http.calls] == ["GET"]


def test_exo_failures_are_honest_rows_not_raises():
    down = ExoLifecycle(http=FakeHttp(
        responses={"/state": Exception("connection refused")}))
    res = down.ensure_loaded("m")
    assert res["ok"] is False and "connection refused" in res["reason"]
    # release cannot confirm what runs — refuse to guess (fail-closed)
    res = down.release("m")
    assert res["ok"] is False and "fail-closed" in res["reason"]

    no_ack = ExoLifecycle(http=FakeHttp(
        responses={"/state": {"instances": {}}, "/place_instance": None}))
    res = no_ack.ensure_loaded("m")
    assert res["ok"] is False and "acknowledgement" in res["reason"]

    del_fail = ExoLifecycle(http=FakeHttp(responses={
        "/place_instance": {"deployment_id": "dpl-1"},
        "/instance/dpl-1": Exception("boom")}))
    del_fail.ensure_loaded("m")
    res = del_fail.release("m")
    assert res["ok"] is False and "boom" in res["reason"]


def test_exo_probe_reads_state_and_fails_soft():
    http = FakeHttp(responses={"/state": {"instances": {
        "a": {"instance_id": "a",
              "shard_assignments": {"model_id": "m1"}},
        "b": {"instance_id": "b",
              "shard_assignments": {"model_id": "m2"}}}}})
    exo = ExoLifecycle(http=http)
    res = exo.probe()
    assert res["ok"] is True and res["state"] == "observed"
    assert res["resident"] == ["m1", "m2"]
    assert http.calls[0]["url"].endswith("/state")
    down = ExoLifecycle(http=FakeHttp(responses={"/state": Exception("down")}))
    assert down.probe()["state"] == "unknown"



# --- eviction planner: booked gating vs live-free ordering -----------------

def test_false_eviction_live_free_low_but_booked_fit_passes():
    """The false-eviction scenario: live free VRAM reads low, but the
    NOMINAL booked fit passes — so nothing is evicted."""
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 10.0, "chat-b": 12.0},
                             clock=clock)
    ledger.set_budget("local", 24.0)
    ledger.record_resident("local", "chat-a", engine="ollama", lane="chat")
    ledger.set_free_vram("local", 2.0)  # live free: 2 GB < 12 GB needed
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load("local", LoadRequest("chat-b", "ollama"))
    assert plan.ok is True
    assert plan.release_steps == ()      # no false eviction
    assert [s.model_id for s in plan.load_steps] == ["chat-b"]
    assert "nominal" in plan.reason


def test_nominal_fit_failure_still_evicts():
    """The gate is the booked ceiling — when the booked fit genuinely fails,
    eviction happens regardless of how free VRAM looks."""
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 10.0, "chat-b": 12.0},
                             clock=clock)
    ledger.set_budget("local", 20.0)     # 10 + 12 = 22 > 20
    ledger.record_resident("local", "chat-a", engine="ollama", lane="chat")
    ledger.set_free_vram("local", 18.0)  # live free looks fine — irrelevant
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load("local", LoadRequest("chat-b", "ollama"))
    assert plan.ok is True
    assert [s.model_id for s in plan.release_steps] == ["chat-a"]
    assert [s.model_id for s in plan.load_steps] == ["chat-b"]


def test_live_free_orders_load_steps_but_never_gates():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"big": 8.0, "small": 1.0, "mid": 2.0},
                             clock=clock)
    ledger.set_budget("local", 32.0)
    ledger.set_free_vram("local", 2.5)   # only small+mid fit live free
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load_many("local", [
        LoadRequest("big", "llamacpp"), LoadRequest("small", "llamacpp"),
        LoadRequest("mid", "llamacpp")])
    assert plan.ok is True
    # Ordering only: the fitting loads go first (smallest first), the big
    # one still loads — live free never gates.
    assert [s.model_id for s in plan.load_steps] == ["small", "mid", "big"]
    assert [s.action for s in plan.steps] == ["load", "load", "load"]


# --- eviction planner: lane wake -------------------------------------------

def test_lane_wake_stops_resident_side_models_first():
    clock = FakeClock()
    ledger = ResidencyLedger(
        bookings={"side-ocr": 4.0, "chat-b": 14.0, "big-chat": 16.0},
        clock=clock)
    ledger.set_budget("local", 30.0)
    # The side model is RECENT and chat-b is the pure-LRU victim — but a lane
    # wake stops side models first, so chat-b survives.
    ledger.record_resident("local", "side-ocr", engine="ollama", lane="ocr")
    ledger.record_resident("local", "chat-b", engine="ollama", lane="chat")
    clock.advance(900.0)                 # chat-b now far older …
    ledger.touch("local", "side-ocr")    # … while side-ocr was just used
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load("local",
                             LoadRequest("big-chat", "ollama", lane="chat"))
    assert plan.ok is True
    assert [s.model_id for s in plan.release_steps] == ["side-ocr"]
    assert "lane wake" in plan.release_steps[0].reason
    assert "side model" in plan.release_steps[0].reason
    assert "chat-b" not in [s.model_id for s in plan.release_steps]


# --- eviction planner: recency window --------------------------------------

def test_recency_window_boundary_keeps_recent_model_warm():
    clock = FakeClock()
    ledger = ResidencyLedger(
        bookings={"oldie": 8.0, "warmie": 8.0, "hot": 8.0, "incoming": 12.0},
        clock=clock)
    ledger.set_budget("local", 20.0)     # must free 16 GB: two victims
    # ages at now=1000: oldie 700s (outside), warmie exactly 600s (boundary,
    # inside the window), hot 5s (inside).
    clock.now = 300.0
    ledger.record_resident("local", "oldie", engine="ollama")
    clock.now = 400.0
    ledger.record_resident("local", "warmie", engine="ollama")
    clock.now = 995.0
    ledger.record_resident("local", "hot", engine="ollama")
    clock.now = 1_000.0
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load("local", LoadRequest("incoming", "ollama"))
    assert plan.ok is True
    assert [s.model_id for s in plan.release_steps] == ["oldie", "warmie"]
    # warmie sits exactly at the 600s boundary and counts as inside the
    # window: it is kept warm until last and only freed under forced need.
    assert "outside the 600s recency window" in plan.release_steps[0].reason
    assert "inside the 600s recency window" in plan.release_steps[1].reason
    assert "kept warm until last" in plan.release_steps[1].reason
    assert "hot" not in [s.model_id for s in plan.release_steps]


def test_recency_window_boundary_flips_just_past_window():
    clock = FakeClock()
    ledger = ResidencyLedger(
        bookings={"oldie": 8.0, "warmie": 8.0, "hot": 8.0, "incoming": 12.0},
        clock=clock)
    ledger.set_budget("local", 20.0)
    # warmie at age 600.1s — JUST outside the window now.
    clock.now = 300.0
    ledger.record_resident("local", "oldie", engine="ollama")
    clock.now = 399.9
    ledger.record_resident("local", "warmie", engine="ollama")
    clock.now = 995.0
    ledger.record_resident("local", "hot", engine="ollama")
    clock.now = 1_000.0
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load("local", LoadRequest("incoming", "ollama"))
    assert [s.model_id for s in plan.release_steps] == ["oldie", "warmie"]
    # Same eviction set, but warmie is now a plain LRU victim, not a warm
    # model forced out — the boundary is the honest difference.
    assert "outside the 600s recency window" in plan.release_steps[1].reason
    assert "kept warm until last" not in plan.release_steps[1].reason


def test_recency_window_forced_eviction_picks_least_recent_of_warm():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"warm-a": 8.0, "warm-b": 8.0,
                                       "incoming": 12.0}, clock=clock)
    ledger.set_budget("local", 20.0)     # one warm victim forced out
    # warm-a age 500s, warm-b age 50s — both inside the window.
    clock.now = 500.0
    ledger.record_resident("local", "warm-a", engine="ollama")
    clock.now = 950.0
    ledger.record_resident("local", "warm-b", engine="ollama")
    clock.now = 1_000.0
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load("local", LoadRequest("incoming", "ollama"))
    assert [s.model_id for s in plan.release_steps] == ["warm-a"]
    assert "kept warm until last" in plan.release_steps[0].reason


# --- eviction planner: image generation ------------------------------------

def test_image_generation_evicts_every_gpu_resident():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 6.0, "chat-b": 2.0,
                                       "image-x": 10.0}, clock=clock)
    ledger.set_budget("local", 24.0)     # 6 + 2 + 10 = 18 would nominally fit
    ledger.record_resident("local", "chat-a", engine="ollama", lane="chat")
    ledger.record_resident("local", "chat-b", engine="ollama", lane="chat")
    ledger.touch("local", "chat-a")      # chat-a is even the recent one
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load("local",
                             LoadRequest("image-x", "ollama",
                                         lane="image", modality="image"))
    assert plan.ok is True
    assert {s.model_id for s in plan.release_steps} == {"chat-a", "chat-b"}
    for step in plan.release_steps:
        assert "whole GPU" in step.reason
        assert "every time" in step.reason
    assert [s.model_id for s in plan.load_steps] == ["image-x"]


def test_image_generation_evicts_every_entry():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 6.0, "image-x": 10.0},
                             clock=clock)
    ledger.set_budget("local", 24.0)
    planner = EvictionPlanner(ledger, clock=clock)
    request = LoadRequest("image-x", "ollama", lane="image", modality="image")

    for _ in range(2):                   # every entry, not just the first
        ledger.record_resident("local", "chat-a", engine="ollama", lane="chat")
        plan = planner.plan_load("local", request)
        assert [s.model_id for s in plan.release_steps] == ["chat-a"]
        ledger.record_released("local", "chat-a")
        ledger.record_resident("local", "image-x", engine="ollama",
                               lane="image")
        ledger.record_released("local", "image-x")


def test_is_image_generation_matches_registry_modalities():
    assert is_image_generation("image") is True
    assert is_image_generation("image_edit") is True
    assert is_image_generation("image-gen") is True
    assert is_image_generation("chat", lane="image") is True
    assert is_image_generation("chat") is False
    assert is_image_generation("video") is False


# --- eviction planner: sequential plans and fail-closed gates --------------

def test_load_plan_is_sequential_ordered_serialized():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"a": 2.0, "b": 2.0, "c": 2.0},
                             clock=clock)
    ledger.set_budget("local", 24.0)
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load_many("local", [
        LoadRequest("a", "ollama"), LoadRequest("b", "ollama"),
        LoadRequest("c", "ollama")])
    assert plan.ok is True
    assert [s.action for s in plan.load_steps] == ["load", "load", "load"]
    assert [s.model_id for s in plan.load_steps] == ["a", "b", "c"]
    assert all("sequentially" in s.reason for s in plan.load_steps)
    # One serialized list — concurrency control is the caller's.
    assert isinstance(plan, LoadPlan)
    assert plan.steps == plan.release_steps + plan.load_steps


def test_plan_fails_closed_without_booked_footprint():
    ledger = ResidencyLedger()
    ledger.set_budget("local", 24.0)
    planner = EvictionPlanner(ledger)
    plan = planner.plan_load("local", LoadRequest("mystery", "ollama"))
    assert plan.ok is False
    assert "no booked footprint" in plan.reason
    assert "fail-closed" in plan.reason
    assert plan.steps == ()


def test_plan_fails_closed_without_declared_budget():
    ledger = ResidencyLedger(bookings={"m": 8.0})
    planner = EvictionPlanner(ledger)
    plan = planner.plan_load("local", LoadRequest("m", "ollama"))
    assert plan.ok is False
    assert "no nominal budget" in plan.reason
    assert "fail-closed" in plan.reason


def test_plan_fails_closed_when_model_exceeds_budget_alone():
    ledger = ResidencyLedger(bookings={"huge": 40.0})
    ledger.set_budget("local", 24.0)
    planner = EvictionPlanner(ledger)
    plan = planner.plan_load("local", LoadRequest("huge", "ollama"))
    assert plan.ok is False
    assert "exceeds the nominal budget" in plan.reason
    assert "fail-closed" in plan.reason


def test_unbooked_resident_is_released_first_honestly():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"incoming": 12.0}, clock=clock)
    ledger.set_budget("local", 20.0)
    ledger.record_resident("local", "mystery", engine="ollama", lane="chat")
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load("local", LoadRequest("incoming", "ollama"))
    assert plan.ok is True
    assert [s.model_id for s in plan.release_steps] == ["mystery"]
    assert "no booked footprint" in plan.release_steps[0].reason


# --- orchestrator: claim/release restore semantics -------------------------

def test_claim_release_restores_what_was_resident():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 6.0, "image-x": 18.0},
                             clock=clock)
    ledger.set_budget("local", 24.0)
    ledger.record_resident("local", "chat-a", engine="ollama", lane="chat")
    adapter = RecordingLifecycle("ollama")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": adapter}, clock=clock)

    claimed = orch.claim("shot-1", "local",
                         LoadRequest("image-x", "ollama", lane="image",
                                     modality="image", estimated_s=40.0))
    assert claimed["ok"] is True
    assert ledger.is_resident("local", "image-x")
    assert not ledger.is_resident("local", "chat-a")

    released = orch.release("shot-1")
    assert released["ok"] is True
    assert ledger.is_resident("local", "chat-a")     # restored
    assert not ledger.is_resident("local", "image-x")  # stepped down
    assert _call_names(adapter) == [
        ("release", "chat-a"), ("ensure_loaded", "image-x"),
        ("release", "image-x"), ("ensure_loaded", "chat-a")]
    assert orch.progress("shot-1")["state"] == "ready"


def test_claim_fails_closed_on_unknown_scope_and_double_claim():
    ledger = ResidencyLedger(bookings={"m": 4.0})
    ledger.set_budget("local", 24.0)
    adapter = RecordingLifecycle("ollama")
    orch = Orchestrator(ledger, EvictionPlanner(ledger), {"ollama": adapter})
    req = LoadRequest("m", "ollama")

    assert orch.claim("", "local", req)["ok"] is False
    assert orch.claim("s-1", "local", req)["ok"] is True
    second = orch.claim("s-1", "local", req)
    assert second["ok"] is False
    assert "already claimed" in second["reason"]
    assert orch.release("never-claimed")["ok"] is False


def test_release_releases_only_what_the_scope_brought_in():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 6.0, "chat-b": 4.0,
                                       "image-x": 18.0}, clock=clock)
    ledger.set_budget("local", 24.0)
    ledger.record_resident("local", "chat-a", engine="ollama", lane="chat")
    ledger.record_resident("local", "chat-b", engine="ollama", lane="chat")
    adapter = RecordingLifecycle("ollama")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": adapter}, clock=clock)

    orch.claim("shot-2", "local",
               LoadRequest("image-x", "ollama", lane="image",
                           modality="image"))
    assert {r.model_id for r in ledger.residents("local")} == {"image-x"}
    orch.release("shot-2")
    # Both claim-time residents come back — the restore set is exactly what
    # was resident at claim.
    assert {r.model_id for r in ledger.residents("local")} == {"chat-a",
                                                              "chat-b"}


# --- orchestrator: sequential loads and honest progress --------------------

def test_restore_loads_fire_sequentially_in_order():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"a": 2.0, "b": 2.0, "c": 2.0,
                                       "image-x": 10.0}, clock=clock)
    ledger.set_budget("local", 24.0)
    for m in ("a", "b", "c"):
        ledger.record_resident("local", m, engine="ollama", lane="chat")
    adapter = RecordingLifecycle("ollama")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": adapter}, clock=clock)

    orch.claim("shot-3", "local",
               LoadRequest("image-x", "ollama", lane="image",
                           modality="image"))
    adapter.calls.clear()
    orch.release("shot-3")
    # release image-x first, then one restore load at a time, in order.
    assert _call_names(adapter) == [
        ("release", "image-x"),
        ("ensure_loaded", "a"), ("ensure_loaded", "b"), ("ensure_loaded", "c")]


def test_progress_states_are_honest_strings():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 4.0, "image-x": 18.0},
                             clock=clock)
    ledger.set_budget("local", 24.0)
    ledger.record_resident("local", "chat-a", engine="ollama", lane="chat")
    adapter = RecordingLifecycle("ollama")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": adapter}, clock=clock)

    orch.claim("shot-4", "local",
               LoadRequest("image-x", "ollama", lane="image",
                           modality="image", estimated_s=35.0))
    states = [row["state"] for row in orch.history("shot-4")]
    assert states == ["queued", "loading (~35s)", "ready"]
    assert orch.progress("shot-4")["state"] == "ready"


def test_progress_without_estimate_says_plain_loading():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"m": 4.0}, clock=clock)
    ledger.set_budget("local", 24.0)
    adapter = RecordingLifecycle("ollama")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": adapter}, clock=clock)

    orch.prewarm("local", LoadRequest("m", "ollama"))
    states = [row["state"] for row in orch.history("local:m")]
    assert states == ["queued", "loading", "ready"]
    assert loading_state(35.0) == "loading (~35s)"
    assert loading_state(None) == "loading"
    assert loading_state(0.0) == "loading (~0s)"


def test_loading_estimate_falls_back_to_booked_load_s():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"m": {"vram_gb": 4.0,
                                             "load_s": 22.0}}, clock=clock)
    ledger.set_budget("local", 24.0)
    adapter = RecordingLifecycle("ollama")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": adapter}, clock=clock)

    orch.prewarm("local", LoadRequest("m", "ollama"))
    loading = orch.history("local:m")[1]
    assert loading["state"] == "loading (~22s)"
    assert loading["estimated_s"] == 22.0


# --- orchestrator: fail-closed mid-plan and adapter gaps -------------------

def test_claim_fails_closed_when_release_fails_mid_plan():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 6.0, "image-x": 18.0},
                             clock=clock)
    ledger.set_budget("local", 24.0)
    ledger.record_resident("local", "chat-a", engine="ollama", lane="chat")
    adapter = RecordingLifecycle("ollama", fail_release={"chat-a"})
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": adapter}, clock=clock)

    res = orch.claim("shot-5", "local",
                     LoadRequest("image-x", "ollama", lane="image",
                                 modality="image"))
    assert res["ok"] is False
    assert "fail-closed" in res["reason"]
    # The plan stopped at the failed release: the load never fired.
    assert _call_names(adapter) == [("release", "chat-a")]
    assert not ledger.is_resident("local", "image-x")
    assert orch.progress("shot-5")["state"].startswith("error: ")


def test_orchestrator_fails_closed_on_missing_adapter():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"m": 4.0}, clock=clock)
    ledger.set_budget("local", 24.0)
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": RecordingLifecycle("ollama")},
                        clock=clock)
    res = orch.prewarm("local", LoadRequest("m", "vllm"))
    assert res["ok"] is False
    assert "no engine adapter registered for 'vllm'" in res["reason"]
    assert orch.progress("local:m")["state"].startswith("error: ")


def test_plan_refusal_surfaces_as_error_progress_row():
    ledger = ResidencyLedger(bookings={"m": 4.0})  # no budget declared
    adapter = RecordingLifecycle("ollama")
    orch = Orchestrator(ledger, EvictionPlanner(ledger), {"ollama": adapter})
    res = orch.prewarm("local", LoadRequest("m", "ollama"))
    assert res["ok"] is False
    assert adapter.calls == []
    assert orch.progress("local:m")["state"].startswith("error: ")


# --- measured footprints: the bench-window write path ------------------------

class FakeSampler:
    """Injectable VRAM sampler: canned readings in order (the last reading
    repeats when the list runs out; an Exception value is raised; an empty
    list raises)."""

    def __init__(self, readings):
        self.readings = list(readings)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if not self.readings:
            raise LookupError("sampler exhausted")
        reading = self.readings[min(self.calls - 1, len(self.readings) - 1)]
        if isinstance(reading, Exception):
            raise reading
        return reading


def test_measured_recorder_books_the_observed_peak():
    clock = FakeClock()
    ledger = ResidencyLedger(clock=clock)
    ledger.book("chat-a", 8.0, source="published")
    sampler = FakeSampler([8.2, 9.5, 8.7])
    with MeasuredFootprintRecorder(sampler, "chat-a", ledger,
                                   clock=clock) as rec:
        rows = [rec.sample() for _ in range(3)]      # caller-run bench window
    assert all(r["ok"] for r in rows)
    assert [r["peak_gb"] for r in rows] == [8.2, 9.5, 9.5]
    # closing the window recorded the observed peak as the booked ceiling
    row = rec.recorded
    assert row["ok"] is True and row["vram_gb"] == 9.5
    assert row["source"] == "measured"
    assert sampler.calls == 3
    assert ledger.booked_vram_gb("chat-a") == 9.5
    booking = ledger.booking("chat-a")
    assert booking.is_measured is True
    assert booking.label == "measured peak (bench)"


def test_measured_recorder_is_fail_soft_on_sampler_errors():
    ledger = ResidencyLedger()
    sampler = FakeSampler([8.0, Exception("nvml gone"), float("nan"), 8.5])
    rec = MeasuredFootprintRecorder(sampler, "m", ledger)
    rows = [rec.sample() for _ in range(4)]
    assert rows[0]["ok"] is True and rows[0]["peak_gb"] == 8.0
    assert rows[1]["ok"] is False and "nvml gone" in rows[1]["reason"]
    assert rows[2]["ok"] is False and "not counted as a peak" in rows[2]["reason"]
    assert rows[3]["ok"] is True and rows[3]["peak_gb"] == 8.5
    assert rec.sample_count == 2                 # failed readings never count
    assert rec.peak_gb == 8.5


def test_measured_recorder_fails_closed_without_valid_readings():
    ledger = ResidencyLedger()
    rec = MeasuredFootprintRecorder(FakeSampler([]), "m", ledger)
    assert rec.sample()["ok"] is False
    row = rec.record()
    assert row["ok"] is False
    assert "fail-closed" in row["error"]
    assert "guessed peak" in row["error"]
    assert ledger.booked_vram_gb("m") is None     # nothing guessed, nothing booked
    assert ledger.booking("m") is None


def test_measured_peak_upgrades_published_booking():
    clock = FakeClock()
    ledger = ResidencyLedger(
        bookings={"chat-a": {"vram_gb": 8.0, "source": "published"}},
        clock=clock)
    assert ledger.booked_vram_gb("chat-a") == 8.0
    with MeasuredFootprintRecorder(FakeSampler([8.2, 9.6]), "chat-a", ledger,
                                   clock=clock) as rec:
        rec.sample()
        rec.sample()
    # measured peaks replace published footprints in the bookings (P4)
    assert ledger.booked_vram_gb("chat-a") == 9.6
    assert ledger.booking("chat-a").label == "measured peak (bench)"


def test_measured_booking_is_never_downgraded_by_published_updates():
    ledger = ResidencyLedger()
    ledger.record_measured_peak("chat-a", 9.5, load_s=35.0)
    assert ledger.booked_vram_gb("chat-a") == 9.5
    # a later published-footprint update must not downgrade the measured peak
    row = ledger.book("chat-a", {"vram_gb": 7.5, "source": "published"})
    assert row["ok"] is True and row["kept"].startswith("measured booking kept")
    assert ledger.booked_vram_gb("chat-a") == 9.5
    assert ledger.estimated_load_s("chat-a") == 35.0
    # …including a whole-dict refresh (the registry-to-bookings path)
    for model_id, spec in {"chat-a": {"vram_gb": 7.5,
                                      "source": "published"}}.items():
        ledger.book(model_id, spec)
    assert ledger.booked_vram_gb("chat-a") == 9.5
    # a NEWER measured peak (a fresh bench) may replace the older one
    row = ledger.record_measured_peak("chat-a", 10.5)
    assert row["ok"] is True and row["source"] == "measured"
    assert ledger.booked_vram_gb("chat-a") == 10.5


def test_booking_report_labels_measured_vs_published():
    ledger = ResidencyLedger(bookings={
        "pub-m": {"vram_gb": 4.0, "source": "published"},
        "reg-m": 2.0})
    ledger.record_measured_peak("meas-m", 9.0, load_s=33.0)
    report = ledger.booking_report()
    rows = {r["model_id"]: r for r in report}
    assert [r["model_id"] for r in report] == sorted(rows)   # stable order
    assert rows["pub-m"]["label"] == "published footprint (advisory)"
    assert rows["reg-m"]["label"] == "published footprint (advisory)"
    assert rows["meas-m"]["label"] == "measured peak (bench)"
    assert rows["meas-m"]["source"] == "measured"
    assert rows["meas-m"]["vram_gb"] == 9.0
    assert rows["meas-m"]["load_s"] == 33.0


def test_measured_peak_replaces_published_in_the_fit_gate():
    clock = FakeClock()
    ledger = ResidencyLedger(
        bookings={"chat-a": 8.0,
                  "chat-b": {"vram_gb": 10.0, "source": "published"}},
        clock=clock)
    ledger.set_budget("local", 20.0)
    ledger.record_resident("local", "chat-a", engine="ollama", lane="chat")
    planner = EvictionPlanner(ledger, clock=clock)
    # published 10 GB fits beside chat-a (8 + 10 <= 20): nothing evicted
    plan = planner.plan_load("local", LoadRequest("chat-b", "ollama"))
    assert plan.ok is True and plan.release_steps == ()
    # the bench measures the real peak at 13 GB — the measured peak replaces
    # the published footprint as the booking, and the fit gate follows it
    with MeasuredFootprintRecorder(FakeSampler([13.0]), "chat-b", ledger,
                                   clock=clock) as rec:
        rec.sample()
    assert ledger.booked_vram_gb("chat-b") == 13.0
    plan = planner.plan_load("local", LoadRequest("chat-b", "ollama"))
    assert plan.ok is True
    assert [s.model_id for s in plan.release_steps] == ["chat-a"]   # 8+13 > 20


# --- orchestrator: the end-to-end workflow gesture ---------------------------

def test_image_generation_gesture_end_to_end_with_fake_seams():
    """The P2/P3 acceptance gesture at library level, on the existing fake
    seams: chat model resident -> image-generation claim evicts GPU residents
    per policy -> image model loads with honest progress states -> release
    restores the chat model. No network, no processes — FakeHttp, FakeRunner,
    FakeClock only."""
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={
        "chat-a": {"vram_gb": 6.0, "source": "published", "load_s": 12.0},
        "image-x": {"vram_gb": 18.0, "source": "published", "load_s": 40.0}},
        clock=clock)
    ledger.set_budget("local", 24.0)
    http = FakeHttp()
    runner = FakeRunner()
    chat = OllamaLifecycle(base_url="http://127.0.0.1:11434", http=http,
                           keep_alive="5m")
    image = LlamaCppServerLifecycle(runner=runner,
                                    which=lambda name: f"/usr/bin/{name}")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": chat, "llamacpp": image}, clock=clock)

    # the chat model is resident (warmed over the fake HTTP seam)
    warmed = orch.prewarm("local", LoadRequest("chat-a", "ollama", lane="chat"))
    assert warmed["ok"] is True
    assert ledger.is_resident("local", "chat-a")
    assert http.calls[0]["body"]["keep_alive"] == "5m"

    # the image-generation claim evicts GPU residents per policy (whole card)
    claimed = orch.claim("shot", "local", LoadRequest(
        "image-x", "llamacpp", lane="image", modality="image",
        weights_path="/models/image-x.safetensors", estimated_s=40.0))
    assert claimed["ok"] is True
    assert not ledger.is_resident("local", "chat-a")
    assert ledger.is_resident("local", "image-x")
    evict_reason = claimed["plan"].release_steps[0].reason
    assert "whole GPU" in evict_reason and "every time" in evict_reason
    # the eviction was the keep_alive=0 evict call on the fake HTTP seam
    assert http.calls[1]["body"] == {"model": "chat-a", "prompt": "",
                                     "stream": False, "keep_alive": 0}
    # the image model loaded as a process on the fake runner seam
    assert runner.argvs == [["/usr/bin/llama-server", "-m",
                             "/models/image-x.safetensors", "--host",
                             "127.0.0.1", "--port", "8080"]]
    # honest progress states through the swap
    assert [r["state"] for r in orch.history("shot")] == [
        "queued", "loading (~40s)", "ready"]

    # release: the image model steps down, the chat model is restored
    released = orch.release("shot")
    assert released["ok"] is True
    assert ledger.is_resident("local", "chat-a")
    assert not ledger.is_resident("local", "image-x")
    assert runner.proc.terminated is True and runner.proc.waited is True
    assert http.calls[2]["body"]["keep_alive"] == "5m"   # chat re-warmed
    assert [r["state"] for r in orch.history("shot")] == [
        "queued", "loading (~40s)", "ready",
        "queued", "loading (~12s)", "ready"]
    assert released["progress"]["state"] == "ready"


# --- orchestrator: restore semantics for interleaved scopes ------------------

def test_release_restores_evicted_mid_scope_model_with_its_weights_path():
    """The evicted model was itself mid-scope: its request (weights path,
    load estimate) lives in another open scope. The release that restores it
    must carry the original weights path — a llama.cpp restore without one
    fails closed on a guessed path."""
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 6.0, "image-x": 18.0},
                             clock=clock)
    ledger.set_budget("local", 24.0)
    runner = FakeRunner()
    http = FakeHttp()
    chat = LlamaCppServerLifecycle(runner=runner,
                                   which=lambda name: f"/usr/bin/{name}")
    image = OllamaLifecycle(http=http)
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"llamacpp": chat, "ollama": image}, clock=clock)

    # s1 claims a big chat model on llama.cpp — process stop/start means
    # every load, restores included, needs the real weights path.
    assert orch.claim("s1", "local", LoadRequest(
        "chat-a", "llamacpp", lane="chat",
        weights_path="/models/chat-a.gguf", estimated_s=30.0))["ok"]
    assert [argv[2] for argv in runner.argvs] == ["/models/chat-a.gguf"]

    # s2 claims image generation: the image entry fully evicts chat-a even
    # though chat-a is s1's mid-scope model.
    assert orch.claim("s2", "local", LoadRequest(
        "image-x", "ollama", lane="image", modality="image"))["ok"]
    assert not ledger.is_resident("local", "chat-a")

    # s2 release restores the evicted mid-scope model with its ORIGINAL
    # weights path and honest load estimate — not a guessed one.
    released = orch.release("s2")
    assert released["ok"] is True
    assert ledger.is_resident("local", "chat-a")
    assert [argv[2] for argv in runner.argvs] == ["/models/chat-a.gguf"] * 2
    states = [r["state"] for r in orch.history("s2")]
    assert states[-3:] == ["queued", "loading (~30s)", "ready"]

    # s1 release afterwards steps down s1's own model and restores its set.
    released1 = orch.release("s1")
    assert released1["ok"] is True
    assert not ledger.is_resident("local", "chat-a")


def test_scope_release_steps_down_only_what_the_scope_brought_in():
    """Interleaved scopes: releasing the evicted model's own scope must not
    step down the OTHER live scope's model, and a release keeps its exact
    claim-time restore set."""
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 6.0, "image-x": 18.0},
                             clock=clock)
    ledger.set_budget("local", 24.0)
    adapter = RecordingLifecycle("ollama")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": adapter}, clock=clock)

    assert orch.claim("s1", "local",
                      LoadRequest("chat-a", "ollama", lane="chat"))["ok"]
    assert orch.claim("s2", "local", LoadRequest(
        "image-x", "ollama", lane="image", modality="image"))["ok"]
    assert {r.model_id for r in ledger.residents("local")} == {"image-x"}

    # s1 (whose chat model was evicted mid-scope) releases while s2 is live:
    # s2's image model must survive untouched.
    adapter.calls.clear()
    released = orch.release("s1")
    assert released["ok"] is True
    assert _call_names(adapter) == []
    assert ledger.is_resident("local", "image-x")

    # s2 release: its own model steps down, and the claim-time set restores —
    # including the model that was evicted from s1 mid-scope.
    released2 = orch.release("s2")
    assert released2["ok"] is True
    assert _call_names(adapter) == [("release", "image-x"),
                                    ("ensure_loaded", "chat-a")]
    assert {r.model_id for r in ledger.residents("local")} == {"chat-a"}


# --- orchestrator: the P4 adapters as plan steps -----------------------------

def test_comfyui_vllm_and_maya_run_as_orchestrator_plan_steps():
    """A plan can name the P4 adapters and the planner treats their bookings
    like any other engine's; each step drives the engine's own contract."""
    clock = FakeClock()
    ledger = ResidencyLedger(
        bookings={"sdxl": 8.0, "qwen": 8.0, "glm-flash": 8.0, "flux": 10.0},
        clock=clock)
    ledger.set_budget("local", 24.0)
    comfy_http = FakeHttp(
        responses={"/system_stats": {"system": {"comfyui-version": "0.3.0"}}})
    maya_http = FakeHttp(responses={"/api/ps": {"models": []},
                                   "/api/show": {"name": "glm-flash"}})
    runner = FakeRunner()
    adapters = {
        "comfyui": ComfyUiLifecycle(http=comfy_http),
        "vllm": VllmServerLifecycle(runner=runner,
                                    which=lambda name: f"/usr/bin/{name}",
                                    exists=lambda p: True),
        "maya": MayaLifecycle(http=maya_http),
    }
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        adapters, clock=clock)

    # The planner treats their bookings like any other engine's.
    r1 = orch.prewarm("local", LoadRequest(
        model_id="sdxl", engine="comfyui", lane="image", modality="image"))
    assert r1["ok"] is True
    assert r1["plan"].load_steps[0].engine == "comfyui"
    assert comfy_http.calls[0]["url"].endswith("/system_stats")

    r2 = orch.prewarm("local", LoadRequest(
        model_id="qwen", engine="vllm", lane="chat",
        weights_path="/models/qwen"))
    assert r2["ok"] is True
    assert r2["plan"].load_steps[0].engine == "vllm"
    assert runner.argvs == [["/usr/bin/vllm", "serve", "/models/qwen",
                             "--host", "127.0.0.1", "--port", "8000"]]

    r3 = orch.prewarm("local", LoadRequest(
        model_id="glm-flash", engine="maya", lane="chat"))
    assert r3["ok"] is True
    assert r3["plan"].load_steps[0].engine == "maya"
    assert maya_http.calls[-1]["url"].endswith("/api/show")

    assert {r.model_id for r in ledger.residents("local")} == {"sdxl", "qwen",
                                                              "glm-flash"}
    assert ledger.booked_total_gb("local") == 24.0

    # An image-generation entry releases every resident through its own
    # engine's release contract: ComfyUI frees the cache, vLLM stops the
    # process, Maya honestly reports no unload primitive.
    comfy_http.calls.clear()
    maya_http.calls.clear()
    r4 = orch.prewarm("local", LoadRequest(
        model_id="flux", engine="comfyui", lane="image", modality="image"))
    assert r4["ok"] is True
    release_engines = sorted(step.engine for step in r4["plan"].release_steps)
    assert release_engines == ["comfyui", "maya", "vllm"]
    assert runner.proc.terminated is True
    freed = [c for c in comfy_http.calls
             if c["url"].endswith("/free") and c["method"] == "POST"]
    assert freed and freed[0]["body"] == {"unload_models": True,
                                          "free_memory": True}
    # Maya's release step sent nothing to the engine — never a fake unload.
    assert maya_http.calls == []
    assert {r.model_id for r in ledger.residents("local")} == {"flux"}
    assert orch.progress("local:flux")["state"] == "ready"


# --- swap: the one-call weight exchange (hiccup-free by pre-flight) ----------

class SwapRecorder(EngineLifecycle):
    """Swap-test engine adapter: records the call ORDER (probes included),
    advances the injected fake clock per call (so phase timings are exact),
    fails on demand, and counts concurrent loads — swap choreography must
    never see two loads in flight."""

    def __init__(self, engine_id="ollama", clock=None, load_s=5.0,
                 release_s=2.0, probe_s=1.0, fail_load=(), fail_release=(),
                 probe_states=()):
        self.engine_id = engine_id
        self.clock = clock
        self.load_s = load_s
        self.release_s = release_s
        self.probe_s = probe_s
        self.fail_load = set(fail_load)
        self.fail_release = set(fail_release)
        self.probe_states = list(probe_states)
        self.events = []
        self.in_load = 0
        self.overlaps = 0

    def _tick(self, seconds):
        if self.clock is not None:
            self.clock.advance(seconds)

    def ensure_loaded(self, model_id, **kwargs):
        self.events.append(f"ensure_loaded:{model_id}")
        self.in_load += 1
        if self.in_load > 1:
            self.overlaps += 1
        self._tick(self.load_s)
        self.in_load -= 1
        if model_id in self.fail_load:
            return {"ok": False,
                    "reason": f"fake load failure for {model_id}"}
        return {"ok": True, "reason": "warmed"}

    def release(self, model_id):
        self.events.append(f"release:{model_id}")
        self._tick(self.release_s)
        if model_id in self.fail_release:
            return {"ok": False,
                    "reason": f"fake release failure for {model_id}"}
        return {"ok": True, "reason": "evicted"}

    def probe(self):
        self.events.append("probe")
        self._tick(self.probe_s)
        state = (self.probe_states.pop(0) if self.probe_states
                 else "observed")
        if state == "unknown":
            return {"ok": False, "state": "unknown", "resident": [],
                    "reason": "fake engine unreachable"}
        return {"ok": True, "state": state, "resident": []}


def _swap_setup(clock=None, budget=24.0, adapter=None, weights=("image-x",)):
    """A swap-ready world: booked models, a declared budget, residents
    chat-a (older) + chat-b (newer), recorded weights for the incoming
    models, and one recording adapter behind every engine."""
    clock = clock or FakeClock()
    ledger = ResidencyLedger(
        bookings={"chat-a": 6.0, "chat-b": 8.0, "chat-c": 12.0,
                  "image-x": 18.0},
        clock=clock)
    ledger.set_budget("gpu", budget)
    ledger.record_resident("gpu", "chat-a", engine="ollama", lane="chat")
    clock.advance(10.0)             # chat-b is the newer resident
    ledger.record_resident("gpu", "chat-b", engine="ollama", lane="chat")
    clock.advance(10.0)
    for mid in weights:
        ledger.record_pull("gpu", mid, f"/models/{mid}.safetensors", 1024)
    if adapter is None:
        adapter = SwapRecorder("ollama", clock=clock)
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": adapter}, clock=clock)
    return ledger, adapter, orch


def _image_request():
    return LoadRequest("image-x", "ollama", lane="image", modality="image",
                       weights_path="/models/image-x.safetensors")


def test_swap_history_ring_is_bounded_with_exact_record_shape():
    ledger = ResidencyLedger()
    for i in range(SWAP_HISTORY_LIMIT + 1):
        res = ledger.record_swap({
            "model_id": f"m{i}", "evicted": [f"old{i}"],
            "evict_seconds": 1.0, "load_seconds": 2.0,
            "ready_seconds": 0.5, "total_seconds": 3.5 + i,
            "ok": True, "warm_path": "warm"})
        assert res["ok"] is True and res["limit"] == SWAP_HISTORY_LIMIT
    history = ledger.swap_history()
    assert len(history) == SWAP_HISTORY_LIMIT          # bounded ring
    assert history[0]["model_id"] == "m1"              # the oldest dropped
    assert history[-1]["model_id"] == f"m{SWAP_HISTORY_LIMIT}"
    # exact record shape — the measured truth of one swap, nothing more
    assert set(history[-1]) == {"model_id", "evicted", "evict_seconds",
                                "load_seconds", "ready_seconds",
                                "total_seconds", "ok", "warm_path"}
    # a non-dict record is an honest error row, never a raise
    assert ledger.record_swap("nope")["ok"] is False


def test_swap_stats_are_measured_aggregates_never_estimates():
    ledger = ResidencyLedger()
    # nothing measured yet -> honest None, never a guessed 0
    assert ledger.swap_stats() == {"count": 0, "last_total_seconds": None,
                                   "average_total_seconds": None}
    ledger.record_swap({"model_id": "a", "evicted": [], "evict_seconds": 1.0,
                        "load_seconds": 2.0, "ready_seconds": 1.0,
                        "total_seconds": 4.0, "ok": True,
                        "warm_path": "keep_alive"})
    ledger.record_swap({"model_id": "b", "evicted": ["x"],
                        "evict_seconds": 2.0, "load_seconds": 4.0,
                        "ready_seconds": 2.0, "total_seconds": 8.0,
                        "ok": False, "warm_path": "process restart"})
    stats = ledger.swap_stats()
    assert stats == {"count": 2, "last_total_seconds": 8.0,
                     "average_total_seconds": 6.0}
    # a failed preflight (phases that never ran) still measures its total —
    # the None phase seconds are never counted as zeroes
    ledger.record_swap({"model_id": "c", "evicted": [],
                        "evict_seconds": None, "load_seconds": None,
                        "ready_seconds": None, "total_seconds": 1.0,
                        "ok": False, "warm_path": None})
    stats = ledger.swap_stats()
    assert stats["count"] == 3
    assert stats["last_total_seconds"] == 1.0
    assert stats["average_total_seconds"] == (4.0 + 8.0 + 1.0) / 3


def test_swap_preflight_failures_leave_residency_untouched():
    """The hardest guarantee: no evict/release call may happen before the
    preflight returns ok — every refusal path leaves residency untouched."""
    # (a) no booked footprint for the incoming model
    ledger, adapter, orch = _swap_setup()
    row = orch.swap("gpu", LoadRequest("mystery", "ollama", lane="chat"))
    assert row["ok"] is False
    assert "no booked footprint" in row["reason"]
    _assert_untouched(ledger, adapter, row)

    # (b) nominal fit fails even with the eviction applied
    ledger, adapter, orch = _swap_setup(budget=18.0)
    row = orch.swap("gpu", LoadRequest("chat-c", "ollama", lane="chat"),
                    release=["chat-a"])     # chat-b (8) + chat-c (12) > 18
    assert row["ok"] is False
    assert "nominal fit fails even with the eviction applied" in row["reason"]
    _assert_untouched(ledger, adapter, row)

    # (c) weights not present — a swap never begins a download mid-flight
    ledger, adapter, orch = _swap_setup(weights=())
    row = orch.swap("gpu", _image_request())
    assert row["ok"] is False
    assert "weights not present — run model_forge_install first" in \
        row["reason"]
    _assert_untouched(ledger, adapter, row)

    # (d) engine not live/reachable
    ledger, adapter, orch = _swap_setup()
    adapter.probe_states = ["unknown"]
    row = orch.swap("gpu", _image_request())
    assert row["ok"] is False
    assert "not live/reachable" in row["reason"]
    assert adapter.events == ["probe"]     # liveness probe only — zero evicts
    assert ledger.is_resident("gpu", "chat-a")
    assert ledger.is_resident("gpu", "chat-b")

    # (e) the incoming model cannot be its own eviction
    ledger, adapter, orch = _swap_setup()
    row = orch.swap("gpu", _image_request(), release=["image-x"])
    assert row["ok"] is False
    assert "cannot be named as its own eviction" in row["reason"]
    _assert_untouched(ledger, adapter, row)

    # (f) no nominal budget declared
    ledger, adapter, orch = _swap_setup()
    ledger._budgets.pop("gpu", None)      # as if the budget never existed
    row = orch.swap("gpu", _image_request())
    assert row["ok"] is False and "no nominal budget declared" in row["reason"]
    _assert_untouched(ledger, adapter, row)


def _assert_untouched(ledger, adapter, row):
    """Shared preflight-refusal assertions: zero evict/load calls, residency
    untouched, an honest resident state in the row, and an honest history
    record (nothing measured for phases that never ran)."""
    assert adapter.events == [] or adapter.events == ["probe"]
    assert [e for e in adapter.events
            if e.startswith(("release", "ensure_loaded"))] == []
    assert ledger.is_resident("gpu", "chat-a")
    assert ledger.is_resident("gpu", "chat-b")
    assert not ledger.is_resident("gpu", "image-x")
    assert "resident now:" in row["reason"]
    record = ledger.swap_history()[-1]
    assert record["ok"] is False and record["evicted"] == []
    assert record["evict_seconds"] is None     # never ran — never a guessed 0
    assert record["load_seconds"] is None
    assert record["ready_seconds"] is None


def test_swap_phase_ordering_evict_before_load_and_single_load():
    ledger, adapter, orch = _swap_setup()
    row = orch.swap("gpu", _image_request())
    assert row["ok"] is True
    # strict order: preflight probe -> BOTH evictions -> exactly ONE load ->
    # ready probe. Evict completes before load starts, never interleaved.
    assert adapter.events == ["probe", "release:chat-a", "release:chat-b",
                              "ensure_loaded:image-x", "probe"]
    assert adapter.overlaps == 0            # loads never concurrent
    assert adapter.in_load == 0
    assert [row["phases"][k]["name"] for k in
            ("preflight", "evict", "load", "ready")] == \
        ["PREFLIGHT", "EVICT", "LOAD", "READY"]
    assert row["evicted"] == ["chat-a", "chat-b"]    # LRU first per policy
    assert row["phases"]["load"]["steps"][0]["model_id"] == "image-x"


def test_swap_timings_come_from_the_injectable_clock():
    clock = FakeClock()
    ledger, adapter, orch = _swap_setup(clock=clock)
    row = orch.swap("gpu", _image_request())
    assert row["ok"] is True
    # the fake adapter advances the clock by exact amounts per call:
    # preflight probe 1s, two evictions 2s each, one load 5s, ready probe 1s
    timings = row["timings"]
    assert timings["preflight_seconds"] == 1.0
    assert timings["evict_seconds"] == 4.0
    assert timings["load_seconds"] == 5.0
    assert timings["ready_seconds"] == 1.0
    assert timings["total_seconds"] == 11.0
    # the measured row is what lands in history — measured, not estimated
    record = ledger.swap_history()[-1]
    assert record["total_seconds"] == 11.0
    assert record["evict_seconds"] == 4.0
    assert ledger.swap_stats()["average_total_seconds"] == 11.0
    # each phase carries its own measured seconds too
    assert row["phases"]["evict"]["seconds"] == 4.0
    assert row["phases"]["load"]["seconds"] == 5.0
    assert row["phases"]["ready"]["seconds"] == 1.0


def test_swap_dry_run_has_zero_side_effects():
    ledger, adapter, orch = _swap_setup()
    row = orch.swap("gpu", _image_request(), dry_run=True)
    assert row["ok"] is True and row["dry_run"] is True
    assert "nothing was changed" in row["note"]
    # zero side effects: only the read-only liveness probe ran
    assert adapter.events == ["probe"]
    assert ledger.is_resident("gpu", "chat-a")
    assert ledger.is_resident("gpu", "chat-b")
    assert not ledger.is_resident("gpu", "image-x")
    assert ledger.swap_history() == ()        # dry runs never record
    # the preflight plan rows come back
    plan = row["phases"]["preflight"]
    assert plan["ok"] is True
    assert [r["model_id"] for r in plan["evict_plan"]] == ["chat-a", "chat-b"]
    assert [r["model_id"] for r in plan["load_plan"]] == ["image-x"]
    assert row["swap_stats"]["count"] == 0


def test_swap_mid_flight_load_failure_states_what_is_resident():
    clock = FakeClock()
    adapter = SwapRecorder("ollama", clock=clock, fail_load=("image-x",))
    ledger, adapter, orch = _swap_setup(clock=clock, adapter=adapter)
    row = orch.swap("gpu", _image_request())
    assert row["ok"] is False
    # the row states exactly what is now resident — 'nothing' is honest —
    # and why: never a raise, never a silent stall
    assert row["resident_state"] == "nothing"
    assert row["resident_now"] == []
    assert "load of image-x failed" in row["reason"]
    assert "resident now: nothing" in row["reason"]
    assert row["evicted"] == ["chat-a", "chat-b"]
    # phase honesty: evict succeeded and is measured, load failed and is
    # measured, ready never ran — its seconds are None, never a guessed 0
    assert row["phases"]["evict"]["ok"] is True
    assert row["phases"]["load"]["ok"] is False
    assert "ready" not in row["phases"]
    assert row["timings"]["evict_seconds"] == 4.0
    assert row["timings"]["load_seconds"] == 5.0
    assert row["timings"]["ready_seconds"] is None
    record = ledger.swap_history()[-1]
    assert record["ok"] is False
    assert record["evicted"] == ["chat-a", "chat-b"]
    assert record["ready_seconds"] is None


def test_swap_eviction_failure_stops_before_load_honestly():
    adapter = SwapRecorder("ollama", fail_release=("chat-a",))
    ledger, adapter, orch = _swap_setup(adapter=adapter)
    row = orch.swap("gpu", _image_request())
    assert row["ok"] is False
    assert "eviction of chat-a failed" in row["reason"]
    assert "stopped before load" in row["reason"]
    assert "resident now:" in row["reason"]
    assert [e for e in adapter.events if e.startswith("ensure_loaded")] == []
    # the failed resident is still resident — the row says so honestly
    assert ledger.is_resident("gpu", "chat-a")
    assert row["evicted"] == []
    assert row["timings"]["load_seconds"] is None


def test_swap_warm_path_labels_warm_engine_vs_process_restart():
    # warm-capable engine: Ollama's keep_alive warm path is recorded verbatim
    clock = FakeClock()
    http = FakeHttp()
    ledger = ResidencyLedger(
        bookings={"chat-a": 6.0, "chat-c": 12.0}, clock=clock)
    ledger.set_budget("gpu", 20.0)
    ledger.record_resident("gpu", "chat-a", engine="ollama", lane="chat")
    ledger.record_pull("gpu", "chat-c", "/models/chat-c.gguf", 1024)
    ollama = OllamaLifecycle(http=http)
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": ollama}, clock=clock)
    row = orch.swap("gpu", LoadRequest("chat-c", "ollama", lane="chat"))
    assert row["ok"] is True
    assert "keep_alive" in row["warm_path"]
    assert "warm-swap" in row["warm_path"]

    # llama.cpp-style engine: process restart, never a claimed warm path —
    # and the stopped-but-startable engine passes the liveness preflight
    clock2 = FakeClock()
    runner = FakeRunner()
    ledger2 = ResidencyLedger(bookings={"image-y": 18.0}, clock=clock2)
    ledger2.set_budget("gpu", 24.0)
    ledger2.record_pull("gpu", "image-y", "/models/image-y.gguf", 1024)
    llamacpp = LlamaCppServerLifecycle(
        runner=runner, which=lambda name: f"/usr/bin/{name}")
    orch2 = Orchestrator(ledger2, EvictionPlanner(ledger2, clock=clock2),
                         {"llamacpp": llamacpp}, clock=clock2)
    row2 = orch2.swap("gpu", LoadRequest(
        "image-y", "llamacpp", lane="chat",
        weights_path="/models/image-y.gguf"))
    assert row2["ok"] is True
    assert row2["warm_path"].startswith("process restart")
    assert "no warm-swap surface" in row2["warm_path"]
    assert runner.argvs == [["/usr/bin/llama-server", "-m",
                             "/models/image-y.gguf", "--host", "127.0.0.1",
                             "--port", "8080"]]
    # the applied warm path is what lands in the measured history record
    assert ledger2.swap_history()[-1]["warm_path"] == row2["warm_path"]


def test_swap_image_entry_evicts_whole_card_despite_explicit_release():
    """Battle rule, every time: an image-generation entry evicts GPU
    residents deterministically — an explicit release list cannot keep a
    resident on the card."""
    ledger, adapter, orch = _swap_setup()
    row = orch.swap("gpu", _image_request(), release=["chat-a"])
    assert row["ok"] is True
    assert row["evicted"] == ["chat-a", "chat-b"]    # the whole card, LRU first
    assert not ledger.is_resident("gpu", "chat-a")
    assert not ledger.is_resident("gpu", "chat-b")
    assert ledger.is_resident("gpu", "image-x")
    reason = next(c["reason"] for c in row["phases"]["preflight"]["checks"]
                  if c["check"] == "release ids")
    assert "deterministic full eviction" in reason


def test_swap_explicit_release_evicts_only_named_ids_lru_first():
    clock = FakeClock()
    ledger, adapter, orch = _swap_setup(clock=clock, budget=26.0,
                                        weights=("image-x", "chat-c"))
    # explicit outgoing ids, least-recently used first; chat-b stays
    row = orch.swap("gpu", LoadRequest("chat-c", "ollama", lane="chat"),
                    release=["chat-b", "chat-a"])
    assert row["ok"] is True
    assert row["evicted"] == ["chat-a", "chat-b"]    # LRU first, not arg order
    assert ledger.is_resident("gpu", "chat-c")
    # a named id that is not resident is skipped with a stated reason
    adapter.events.clear()
    row2 = orch.swap("gpu", LoadRequest("image-x", "ollama", lane="image",
                                        modality="image"),
                     release=["ghost-model"])
    assert row2["ok"] is True
    reason = next(c["reason"] for c in
                  row2["phases"]["preflight"]["checks"]
                  if c["check"] == "release ids")
    assert "ghost-model" in reason and "nothing to evict for it" in reason


def test_swap_ready_probe_failure_is_an_honest_row():
    clock = FakeClock()
    adapter = SwapRecorder("ollama", clock=clock,
                           probe_states=["observed", "stopped"])
    ledger, adapter, orch = _swap_setup(clock=clock, adapter=adapter)
    row = orch.swap("gpu", _image_request())
    assert row["ok"] is False
    assert "ready probe could not confirm" in row["reason"]
    # the load did complete — the row says exactly what is resident now
    assert row["resident_state"] == "image-x"
    assert row["timings"]["load_seconds"] == 5.0
    assert row["timings"]["ready_seconds"] == 1.0
    assert row["phases"]["ready"]["ok"] is False
    assert ledger.swap_history()[-1]["ok"] is False


# --- engine-server confinement: loopback bind + minimal child env -----------

def test_llamacpp_server_binds_loopback_and_gets_minimal_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("HF_HOME", "/srv/hf-cache")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    runner = FakeRunner()
    server = LlamaCppServerLifecycle(
        binary="llama-server", port=8080, extra_args=("--threads", "4"),
        runner=runner, which=lambda name: f"/usr/bin/{name}")
    res = server.ensure_loaded("qwen", weights_path="/models/qwen.gguf")
    assert res["ok"] is True
    # the bind is explicit and loopback-only, ahead of the extra args
    assert runner.argvs[0] == ["/usr/bin/llama-server", "-m",
                               "/models/qwen.gguf", "--host", "127.0.0.1",
                               "--port", "8080", "--threads", "4"]
    assert res["reason"].endswith("(binds 127.0.0.1 only)")
    # the child sees exactly the minimal env — no provider keys, no tokens
    env = runner.envs[0]
    assert set(env) == {"PATH", "HOME", "LANG", "TMPDIR", "HF_HOME",
                        "CUDA_VISIBLE_DEVICES"}
    assert "OPENAI_API_KEY" not in env and "HF_TOKEN" not in env
    assert env["HF_HOME"] == "/srv/hf-cache"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_vllm_server_binds_loopback_and_gets_minimal_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "provider-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.delenv("HF_HOME", raising=False)
    runner = FakeRunner()
    vllm = VllmServerLifecycle(
        binary="vllm", port=8000, extra_args=("--tensor-parallel-size", "2"),
        runner=runner, which=lambda name: f"/usr/bin/{name}",
        exists=lambda p: True)
    res = vllm.ensure_loaded("qwen", weights_path="/models/qwen")
    assert res["ok"] is True
    assert runner.argvs[0] == ["/usr/bin/vllm", "serve", "/models/qwen",
                               "--host", "127.0.0.1", "--port", "8000",
                               "--tensor-parallel-size", "2"]
    assert res["reason"].startswith("started the vLLM server")
    assert "(binds 127.0.0.1 only)" in res["reason"]
    env = runner.envs[0]
    assert set(env) == {"PATH", "HOME", "LANG", "TMPDIR"}
    assert "ANTHROPIC_API_KEY" not in env and "HF_TOKEN" not in env


def test_minimal_engine_env_forwards_only_named_extras(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("HF_HOME", raising=False)
    env = _minimal_engine_env()
    assert set(env) == {"PATH", "HOME", "LANG", "TMPDIR",
                        "HIP_VISIBLE_DEVICES"}
    assert env["HIP_VISIBLE_DEVICES"] == "1"
    # a caller that names no extras gets the four base keys only
    assert set(_minimal_engine_env(extras=())) == {"PATH", "HOME", "LANG",
                                                   "TMPDIR"}


def test_llamacpp_hostile_extra_args_rebind_fails_closed():
    runner = FakeRunner()
    server = LlamaCppServerLifecycle(
        extra_args=("--host", "0.0.0.0"), runner=runner,
        which=lambda name: f"/usr/bin/{name}")
    res = server.ensure_loaded("qwen", weights_path="/models/qwen.gguf")
    assert res["ok"] is False
    assert "fail-closed" in res["reason"]
    assert "127.0.0.1 only" in res["reason"]
    assert runner.argvs == []          # nothing was spawned
    # per-call extra_args are inspected too, not just the constructor's
    res = server.ensure_loaded("qwen", weights_path="/models/qwen.gguf",
                               extra_args=("--host=0.0.0.0",))
    assert res["ok"] is False and runner.argvs == []


def test_vllm_hostile_extra_args_rebind_fails_closed():
    runner = FakeRunner()
    vllm = VllmServerLifecycle(runner=runner,
                               which=lambda name: f"/usr/bin/{name}",
                               exists=lambda p: True)
    res = vllm.ensure_loaded("qwen", weights_path="/models/qwen",
                             extra_args=("--listen", "::"))
    assert res["ok"] is False
    assert "fail-closed" in res["reason"]
    assert "127.0.0.1 only" in res["reason"]
    assert runner.argvs == []


def test_loopback_extra_args_are_allowed():
    # an explicit loopback bind never weakens the enforced loopback bind
    assert _non_loopback_bind(("--host", "127.0.0.1")) == ""
    assert _non_loopback_bind(("--host=localhost",)) == ""
    assert _non_loopback_bind(("--host", "::1")) == ""
    assert _non_loopback_bind(("--threads", "4")) == ""
    assert _non_loopback_bind(("--host", "0.0.0.0")) == "--host 0.0.0.0"
    assert _non_loopback_bind(("--host=::",)) == "--host=::"
    assert _non_loopback_bind(("--host",)) == "--host"   # missing value


def test_spawn_process_passes_env_to_popen(monkeypatch):
    seen = {}

    class _Popen:
        def __init__(self, argv, env=None):
            seen["argv"] = argv
            seen["env"] = env

    import model_forge.broker as broker_mod
    monkeypatch.setattr(broker_mod.subprocess, "Popen", _Popen)
    _spawn_process(["/usr/bin/llama-server", "--host", "127.0.0.1"],
                   {"PATH": "/usr/bin"})
    assert seen["argv"] == ["/usr/bin/llama-server", "--host", "127.0.0.1"]
    assert seen["env"] == {"PATH": "/usr/bin"}


# --- orchestrator.restore_plan: what a release WOULD load, without running --

def test_restore_plan_names_the_process_load_without_spawning():
    """The release's phase-2 restore plan can be inspected without executing:
    a scope whose claim-time resident is a non-resident process-engine model
    yields a llama.cpp load step — the exact signal the approval gate uses —
    while the fake runner seam is never touched."""
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 18.0, "image-x": 20.0},
                             clock=clock)
    ledger.set_budget("local", 24.0)
    runner = FakeRunner()
    chat = LlamaCppServerLifecycle(runner=runner,
                                   which=lambda name: f"/usr/bin/{name}")
    image = RecordingLifecycle("comfyui")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"llamacpp": chat, "comfyui": image}, clock=clock)

    assert orch.claim("s1", "local", LoadRequest(
        "chat-a", "llamacpp", lane="chat",
        weights_path="/models/chat-a.gguf", estimated_s=30.0))["ok"]
    assert orch.claim("s2", "local", LoadRequest(
        "image-x", "comfyui", lane="image", modality="image"))["ok"]
    assert not ledger.is_resident("local", "chat-a")

    before = len(runner.argvs)
    plan = orch.restore_plan("s2")
    assert plan is not None and plan.ok is True
    assert [s.engine for s in plan.load_steps] == ["llamacpp"]
    assert plan.load_steps[0].model_id == "chat-a"
    assert plan.load_steps[0].weights_path == "/models/chat-a.gguf"
    assert len(runner.argvs) == before        # planning spawned nothing
    # an unknown scope has nothing to restore
    assert orch.restore_plan("never-claimed") is None


def test_restore_plan_is_none_for_an_http_only_restore():
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-b": 6.0, "image-x": 20.0},
                             clock=clock)
    ledger.set_budget("local", 24.0)
    http = FakeHttp()
    chat = OllamaLifecycle(http=http)
    image = RecordingLifecycle("comfyui")
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {"ollama": chat, "comfyui": image}, clock=clock)
    assert orch.claim("s1", "local",
                      LoadRequest("chat-b", "ollama", lane="chat"))["ok"]
    assert orch.claim("s2", "local", LoadRequest(
        "image-x", "comfyui", lane="image", modality="image"))["ok"]
    assert not ledger.is_resident("local", "chat-b")
    plan = orch.restore_plan("s2")
    assert plan is not None
    assert [s.engine for s in plan.load_steps] == ["ollama"]  # HTTP, no process
