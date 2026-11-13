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
    EngineLifecycle,
    EvictionPlanner,
    LlamaCppServerLifecycle,
    LlamaSwapLifecycle,
    LoadPlan,
    LoadRequest,
    OllamaLifecycle,
    Orchestrator,
    ResidencyLedger,
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
    """Injected process runner: records argv, returns one FakeProcess."""

    def __init__(self):
        self.argvs = []
        self.proc = FakeProcess()

    def __call__(self, argv):
        self.argvs.append(list(argv))
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
                             "--port", "8080", "--threads", "4"]]
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
