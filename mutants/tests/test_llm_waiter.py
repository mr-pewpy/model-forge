"""LLM-waiter priority tests (roadmap §3, task P5a-A2).

The rule: while a chat request waits on a lane wake, new generative jobs
are refused — the waiter owns the wake. These tests pin the Orchestrator
waiter seam (``waiter_enter`` / ``waiter_exit`` / ``admit_gen`` /
``llm_waiters``) whose counter semantics mirror the predecessor broker's
``Lifecycle.llm_waiters`` (its ``prepare_llm`` / ``prepare_gen``), and the
pure-data expression of the
same rule inside ``EvictionPlanner``.

Everything is injected — fake clock, recording engine adapters — so the
suite runs with no network and no processes. The concurrency cases drive
the seam from real threads the way the facade's ``ThreadingHTTPServer``
does, and assert the accounting never double-counts under races.
"""

from __future__ import annotations

import threading
import time

from model_forge.broker import (
    EngineLifecycle,
    EvictionPlanner,
    LoadRequest,
    Orchestrator,
    ResidencyLedger,
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


class RecordingLifecycle(EngineLifecycle):
    """Engine adapter that records call shapes (no processes, no network).

    Honors the real lifecycle contract the orchestrator drives: probes
    report an observed/running state so a swap/claim preflight passes."""

    def __init__(self, engine_id: str = "recording"):
        self.engine_id = engine_id
        self.calls = []

    def ensure_loaded(self, model_id, **kwargs):
        self.calls.append(("ensure_loaded", model_id, kwargs))
        return {"ok": True, "reason": "warmed"}

    def release(self, model_id):
        self.calls.append(("release", model_id, {}))
        return {"ok": True, "reason": "evicted"}

    def probe(self):
        return {"ok": True, "state": "observed", "resident": []}


def _orch(adapter=None):
    """An orchestrator over a local target with a booked chat and image
    model, plus one recording adapter — the minimum a claim needs."""
    clock = FakeClock()
    ledger = ResidencyLedger(
        bookings={"chat-a": 20.0, "image-x": 12.0}, clock=clock)
    ledger.set_budget("local", 36.0)
    adapter = adapter or RecordingLifecycle()
    orch = Orchestrator(ledger, EvictionPlanner(ledger, clock=clock),
                        {adapter.engine_id: adapter}, clock=clock)
    return orch, ledger, adapter, clock


CHAT = LoadRequest("chat-a", "recording", lane="chat")
IMAGE = LoadRequest("image-x", "recording", lane="image", modality="image")


# --- waiter accounting ------------------------------------------------------

def test_waiter_enter_registers_a_unique_token_and_counts():
    orch, _ledger, _adapter, _clock = _orch()
    assert orch.llm_waiters == 0

    first = orch.waiter_enter()
    second = orch.waiter_enter()

    assert first["ok"] is True and second["ok"] is True
    assert isinstance(first["token"], str) and first["token"]
    assert isinstance(second["token"], str) and second["token"]
    assert first["token"] != second["token"]        # tokens are unique
    assert orch.llm_waiters == 2                     # concurrent waiters tracked
    # The reason names the policy honestly — the waiter owns the wake.
    assert "waiter" in first["reason"] and "owns the wake" in first["reason"]


def test_waiter_exit_decrements_exactly_once_per_token():
    orch, _ledger, _adapter, _clock = _orch()
    token = orch.waiter_enter()["token"]
    assert orch.llm_waiters == 1

    orch.waiter_exit(token)
    assert orch.llm_waiters == 0

    # Idempotent: a second exit for the same token is a no-op, never a raise.
    orch.waiter_exit(token)
    assert orch.llm_waiters == 0


def test_waiter_exit_unknown_token_is_a_noop_that_never_raises():
    orch, _ledger, _adapter, _clock = _orch()
    held = orch.waiter_enter()["token"]

    # An unknown token (never issued, or already exited) is a no-op.
    orch.waiter_exit("never-issued")
    assert orch.llm_waiters == 1
    orch.waiter_exit("")
    orch.waiter_exit(None)          # defensive: never raises either
    assert orch.llm_waiters == 1

    orch.waiter_exit(held)
    assert orch.llm_waiters == 0


def test_waiter_exit_only_removes_its_own_token():
    orch, _ledger, _adapter, _clock = _orch()
    a = orch.waiter_enter()["token"]
    orch.waiter_enter()             # a second, still-held waiter
    orch.waiter_exit(a)
    assert orch.llm_waiters == 1    # the other waiter is untouched


# --- gen admission ----------------------------------------------------------

def test_admit_gen_refuses_while_a_waiter_owns_the_wake():
    orch, _ledger, _adapter, _clock = _orch()
    orch.waiter_enter()

    row = orch.admit_gen("image-x")
    assert row["ok"] is False
    # Honest reason naming the policy, verbatim.
    assert "refused: a chat request owns the lane wake" in row["reason"]
    assert "waiter" in row["reason"] and "owns the wake" in row["reason"]
    assert "LLM-waiter priority" in row["reason"]


def test_admit_gen_is_open_at_zero_waiters():
    orch, _ledger, _adapter, _clock = _orch()
    row = orch.admit_gen("image-x")
    assert row["ok"] is True
    assert "admitted" in row["reason"]
    assert "admission" in row["reason"]
    # A bare call (no model id) is admitted the same way.
    assert orch.admit_gen()["ok"] is True


def test_admit_gen_reopens_only_after_every_waiter_exits():
    orch, _ledger, _adapter, _clock = _orch()
    a = orch.waiter_enter()["token"]
    b = orch.waiter_enter()["token"]

    orch.waiter_exit(a)
    assert orch.admit_gen("image-x")["ok"] is False   # b still waits
    orch.waiter_exit(b)
    assert orch.admit_gen("image-x")["ok"] is True


def test_admit_gen_never_claims_or_touches_the_ledger():
    """Admission is a gate, not a claim: it must not load anything."""
    orch, ledger, adapter, _clock = _orch()
    assert orch.admit_gen("image-x")["ok"] is True
    assert adapter.calls == []                        # no engine call
    assert ledger.residents("local") == ()            # nothing resident


# --- the waiter owns the wake (integration) ---------------------------------

def test_chat_claim_still_wakes_while_gen_admission_stays_refused():
    """A chat claim with a waiting waiter runs its normal lane wake; the
    waiter owns that wake, so gen admission stays refused throughout."""
    orch, ledger, adapter, _clock = _orch()
    token = orch.waiter_enter()["token"]

    result = orch.claim("chat-scope", "local", CHAT)
    assert result["ok"] is True
    # The chat lane wake genuinely ran — the model was loaded.
    assert ("ensure_loaded", "chat-a", {"weights_path": None}) in adapter.calls
    assert ledger.is_resident("local", "chat-a") is True

    # ...and gen is refused for exactly as long as the waiter waits.
    assert orch.admit_gen("image-x")["ok"] is False

    orch.waiter_exit(token)
    assert orch.admit_gen("image-x")["ok"] is True


# --- concurrency: honest accounting under races -----------------------------

def test_concurrent_enter_exit_never_double_counts():
    """Many threads enter/exit/decrement; with a double exit per token the
    count must still land at zero — exit is once-per-token, so interleavings
    cannot push the counter negative or leave a phantom waiter."""
    orch, _ledger, _adapter, _clock = _orch()
    threads_n, loops = 24, 200

    def worker():
        for _ in range(loops):
            token = orch.waiter_enter()["token"]
            orch.waiter_exit(token)
            orch.waiter_exit(token)     # the spurious second exit is a no-op

    threads = [threading.Thread(target=worker) for _ in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert orch.llm_waiters == 0


def test_concurrent_waiters_are_all_tracked_with_distinct_tokens():
    """N threads each hold a waiter open at once: the count reads N and every
    token is distinct — concurrent waiters are all tracked, none collapsed."""
    orch, _ledger, _adapter, _clock = _orch()
    n = 16
    tokens = []
    tokens_lock = threading.Lock()
    hold = threading.Event()

    def worker():
        token = orch.waiter_enter()["token"]
        with tokens_lock:
            tokens.append(token)
        hold.wait(timeout=10.0)
        orch.waiter_exit(token)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()

    deadline = time.time() + 10.0
    while (orch.llm_waiters < n or len(tokens) < n) and time.time() < deadline:
        time.sleep(0.001)

    assert orch.llm_waiters == n
    assert len(set(tokens)) == n                 # no token ever collides
    assert orch.admit_gen("image-x")["ok"] is False

    hold.set()
    for t in threads:
        t.join()
    assert orch.llm_waiters == 0


# --- pure-data suppression inside the planner -------------------------------

def test_planner_suppresses_gen_plan_while_waiters_pending():
    """The same rule as planner data: a generative plan is refused with an
    honest reason while a waiter owns the wake — nothing is loaded/released."""
    clock = FakeClock()
    ledger = ResidencyLedger(
        bookings={"chat-a": 20.0, "image-x": 12.0}, clock=clock)
    ledger.set_budget("local", 36.0)
    ledger.record_resident("local", "chat-a", engine="recording", lane="chat")
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load_many("local", [IMAGE], waiter_count=1)

    assert plan.ok is False
    assert "refused: a chat request owns the lane wake" in plan.reason
    assert "owns the wake" in plan.reason
    assert plan.release_steps == () and plan.load_steps == ()


def test_planner_gen_plan_is_open_without_waiters():
    """With no waiter, the gen plan is unchanged: the whole card is freed."""
    clock = FakeClock()
    ledger = ResidencyLedger(
        bookings={"chat-a": 20.0, "image-x": 12.0}, clock=clock)
    ledger.set_budget("local", 36.0)
    ledger.record_resident("local", "chat-a", engine="recording", lane="chat")
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load_many("local", [IMAGE], waiter_count=0)

    assert plan.ok is True
    assert [s.model_id for s in plan.release_steps] == ["chat-a"]
    assert [s.model_id for s in plan.load_steps] == ["image-x"]


def test_planner_waiter_suppression_does_not_touch_chat_plans():
    """A chat plan is never suppressed by waiters — only generative entries
    are gated; the waiter's own lane wake must always be plannable."""
    clock = FakeClock()
    ledger = ResidencyLedger(bookings={"chat-a": 20.0}, clock=clock)
    ledger.set_budget("local", 36.0)
    planner = EvictionPlanner(ledger, clock=clock)

    plan = planner.plan_load_many("local", [CHAT], waiter_count=3)

    assert plan.ok is True
    assert [s.model_id for s in plan.load_steps] == ["chat-a"]
