"""Model residency and lifecycle — the v2 weight-residency core (roadmap §5).

What weights live where, what is loaded in VRAM right now, and what happens
when a task needs a different model. Four pieces:

- ``ResidencyLedger`` — pulls, residency, booked ceilings per model (from a
  booking dict injected by callers: registry footprints or measured peaks)
  and live free-VRAM snapshots. Live free VRAM ORDERs candidates only.
- ``EngineLifecycle`` adapters — ``ensure_loaded`` / ``release`` / ``probe``
  per engine: Ollama (warm via a generate call with ``keep_alive``, evict
  with ``keep_alive=0``), llama-swap (observational pass-through — it swaps
  models on demand by route and exposes no eviction control), llama.cpp
  llama-server (process start/stop on a weights path; the process runner is
  injectable and a missing binary fails closed), ComfyUI (readiness plus the
  explicit cache free between back-to-back generative jobs via its
  ``POST /free`` route — ComfyUI loads models per-workflow, so this is a
  readiness + cache-free contract, never a weights pin), vLLM (process
  start/stop of the server with the model as a launch argument — engine
  sleep/wake is deliberately not used; wake endpoints are unreliable
  upstream and process restart is the reliable path), container engines (the
  docker dance: start the container and wait on its health endpoint within a
  caller-passed cold-boot budget, stop it to free the GPU instantly — the
  production ComfyUI sidecar pattern), Maya (engine-native
  readiness over its model-management API — Maya exposes no unload
  primitive, so release is a ledger-side acknowledgment and never fakes an
  unload), and Exo (engine-native instance placement — ``POST
  /place_instance`` launches a model on the clustered devices and ``DELETE
  /instance/{id}`` stops it by instance or deployment id, a real unload
  primitive; placement is asynchronous, so a launch is reported honestly as
  placement-requested, never as already loaded).
- ``EvictionPlanner`` — the serving policy proven under production load
  (roadmap §3) as
  pure plan data: fit gates on NOMINAL booked ceilings (live free VRAM only
  orders candidates — booked-vs-live comparisons cause false evictions), a
  lane wake stops resident side models first, a ~600 s recency window keeps
  recently-used models warm ahead of pure LRU, image-generation entry evicts
  GPU residents every time, and loads fire one at a time (the plan is an
  ordered, serialized list — concurrency control belongs to the caller).
- ``MeasuredFootprintRecorder`` — the measured-footprint write path
  (roadmap §5a / P4): given an injectable VRAM sampler and a model id,
  sample during a caller-run bench window and record the observed peak as
  that model's booked ceiling. Measured bookings are never downgraded by
  later published-footprint updates, and ledger reports label every booking
  honestly as measured or published.
- ``Orchestrator`` — claim/release workflow scopes (a scope claims the lane
  for a modality; ``release`` by default restores what was resident at claim
  time, while ``release(scope, restore=False)`` leaves residency untouched —
  the claim's target stays resident), pre-warm on intent, honest progress
  states (``queued`` / ``loading (~Ns)`` / ``ready``) as pure data, and
  ``swap`` — the one-call weight exchange in
  strict measured phases (PREFLIGHT / EVICT / LOAD / READY) with the hard
  guarantee that no evict/release call runs before the preflight passes.
  The orchestrator also carries the LLM-waiter priority seam (roadmap §3):
  ``waiter_enter`` / ``waiter_exit`` / ``llm_waiters`` account for chat
  requests blocked on a lane wake, and ``admit_gen`` refuses new generative
  jobs while any waiter exists — the waiter owns the wake. ``gen_busy`` is
  the cheap, honest read of whether any generative scope is open, so a chat
  request can wait for in-flight gen jobs to drain before it wakes the lane.

Everything external is injectable (HTTP functions, process runners, binary
lookups, clocks) so this module performs no I/O outside those seams.
Fail-soft where the roadmap says fail-soft: observational probes report
``unknown`` and never raise, and adapter calls return honest rows instead of
raising. Fail-closed where policy decisions are made: no booked footprint,
no declared budget, a missing engine binary, or a failed release mid-plan
all stop the plan with a stated reason — never a silent stall and never a
guessed fit.
"""

from __future__ import annotations

import abc
import collections
import json
import math
import os
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "RECENCY_WINDOW_S",
    "SWAP_HISTORY_LIMIT",
    "Booking",
    "ComfyUILifecycle",
    "ComfyUiLifecycle",
    "DockerContainerLifecycle",
    "EngineLifecycle",
    "EvictionPlanner",
    "ExoLifecycle",
    "LlamaCppServerLifecycle",
    "LlamaSwapLifecycle",
    "LoadPlan",
    "LoadRequest",
    "MayaLifecycle",
    "MeasuredFootprintRecorder",
    "OllamaLifecycle",
    "Orchestrator",
    "PlanStep",
    "ProcessHandle",
    "PullRecord",
    "ResidencyLedger",
    "ResidentRecord",
    "RuntimeHandle",
    "VllmLifecycle",
    "VllmServerLifecycle",
    "is_image_generation",
    "is_gen_lane",
    "loading_state",
    "runtime_lifecycle_for",
]

# Policy proven under production load: models used within roughly the last ten minutes stay
# warm; recency outranks pure LRU near the top of the stack.
RECENCY_WINDOW_S = 600.0

# The swap history ring: the last SWAP_HISTORY_LIMIT measured swap records
# (bounded — old records drop off; stats aggregate over the kept window).
SWAP_HISTORY_LIMIT = 20


# --- plan data ---------------------------------------------------------------

@dataclass(frozen=True)
class PlanStep:
    """One explicit step of a load plan, with an honest reason."""
    action: str                     # "release" | "load"
    target_id: str
    model_id: str
    engine: str
    reason: str
    lane: str = "chat"
    weights_path: str | None = None
    estimated_s: float | None = None


@dataclass(frozen=True)
class LoadPlan:
    """Ordered plan data: releases first, then loads. Pure data — applying a
    plan is the orchestrator's job, and loads fire sequentially."""
    target_id: str
    ok: bool
    reason: str
    release_steps: tuple[PlanStep, ...] = ()
    load_steps: tuple[PlanStep, ...] = ()

    @property
    def steps(self) -> tuple[PlanStep, ...]:
        """The serialized execution order: every release, then every load."""
        return tuple(self.release_steps) + tuple(self.load_steps)


@dataclass(frozen=True)
class LoadRequest:
    """A model the caller wants resident on a compute target."""
    model_id: str
    engine: str
    lane: str = "chat"
    modality: str = "chat"
    weights_path: str | None = None
    estimated_s: float | None = None


def loading_state(estimated_s: float | None) -> str:
    """Honest loading label: ``loading (~Ns)`` with a real estimate, plain
    ``loading`` when no duration is known — never invent timing."""
    if estimated_s is None:
        return "loading"
    return f"loading (~{max(0, int(round(estimated_s)))}s)"


def progress_row(state: str, *, model_id: str = "", target_id: str = "",
                 scope_id: str | None = None,
                 estimated_s: float | None = None) -> dict:
    """A progress state as pure data."""
    return {"state": state, "model_id": model_id, "target_id": target_id,
            "scope_id": scope_id, "estimated_s": estimated_s}


def is_image_generation(modality: str, lane: str = "") -> bool:
    """True for image-generation entries (``image``, ``image_edit``,
    ``image-gen``, …). Image generation wants the whole card — the policy
    answers with deterministic full eviction, not clever sharing."""
    for token in (modality, lane):
        norm = str(token or "").replace("-", "").replace("_", "") \
                             .replace(" ", "").lower()
        if norm.startswith("image"):
            return True
    return False


def is_gen_lane(modality: str, lane: str = "") -> bool:
    """True when a request names a generative (``image`` / ``video`` / ``gen``)
    lane — the lanes ``Orchestrator.gen_busy`` counts as an in-flight gen job.

    Generalizes ``is_image_generation`` to the whole generative surface: the
    facade's gen claims name their lane ``image`` or ``video`` (the video lane
    holds the card exactly as an image lane does), and a caller may use the
    umbrella ``gen``. Tokens normalize the same way (strip ``-``/``_``/spaces,
    lowercase) so ``image_edit`` / ``h3-t2v``-style lanes still match by their
    lane token."""
    for token in (modality, lane):
        norm = str(token or "").replace("-", "").replace("_", "") \
                             .replace(" ", "").lower()
        if norm == "gen" or norm.startswith("image") \
                or norm.startswith("video"):
            return True
    return False


def _finite_gb(value) -> float | None:
    try:
        gb = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(gb) or gb < 0.0:
        return None
    return gb


# --- residency ledger --------------------------------------------------------

@dataclass(frozen=True)
class Booking:
    """The nominal footprint booked for a model, in GB.

    ``source="registry"`` is a published footprint (advisory);
    ``source="measured"`` is a peak sampled during a bench on the machine —
    measured peaks are the booking numbers and never get downgraded by a
    registry value; ``source="engine_reported"`` is a remote engine's own
    memory report (B3) — a real observation, but lower-trust than a
    vendor-sampled local bench, and labeled so."""
    vram_gb: float
    source: str = "registry"
    load_s: float | None = None     # honest load-duration estimate, when known

    @property
    def is_measured(self) -> bool:
        """True for a bench-measured peak — the booking number that wins."""
        return self.source == "measured"

    @property
    def label(self) -> str:
        """Honest provenance label for ledger reports: a measured peak is
        never presented as an advisory published figure, or vice versa, and a
        remote engine report is never presented as a local bench."""
        if self.source == "measured":
            return "measured peak (bench)"
        if self.source == "engine_reported":
            return "engine-reported (remote)"
        return "published footprint (advisory)"


# Booking trust ranking: a higher-trust booking is never downgraded by a
# lower-trust one. measured (local vendor-sampled) > engine_reported (remote
# engine's own figure) > registry (published advisory). A newer booking of
# EQUAL trust replaces an older one (the latest bench wins).
_BOOKING_TRUST = {"measured": 3, "engine_reported": 2, "registry": 1}


@dataclass(frozen=True)
class PullRecord:
    """Weights on disk at a compute target, with size where known.

    ``source`` is honest provenance: "" (the local storage scan),
    ``observed`` (an engine reported the pull done) or ``user_confirmed``
    (the user attested the on-target install finished)."""
    model_id: str
    path: str
    size_bytes: int | None = None
    source: str = ""


@dataclass(frozen=True)
class ResidentRecord:
    """A model loaded in memory on a compute target.

    ``provenance`` is honest server-process provenance (B8): ``"plugin"``
    when Model Forge started the engine's server/runtime for this model, and
    ``"external"`` when the runtime was already running and Model Forge
    adopted it (a server the user started themselves — e.g. a docker
    container brought up by compose). Stopping an ``"external"`` runtime is
    stopping a process the user started, so it rides the approval gate; a
    ``"plugin"`` runtime was gated at start, so stopping it is symmetric and
    ungated. ``""`` means provenance is not recorded (an engine with no
    stoppable server)."""
    model_id: str
    engine: str
    lane: str
    loaded_at: float
    last_used: float
    provenance: str = ""


class ResidencyLedger:
    """Per compute target: what is pulled (paths + sizes where known), what
    is resident, booked ceilings per model, and live free-VRAM snapshots.

    The booking dict is injected by callers — registry footprints or
    measured values; this ledger only books and totals them. Live free VRAM
    is recorded for ORDERING decisions only: it never gates fit (live free
    always reads lower than nominal and causes false evictions).
    """

    def __init__(self, bookings=None, clock=None):
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._bookings: dict[str, Booking] = {}
        self._budgets: dict[str, float] = {}
        self._pulls: dict[str, dict[str, PullRecord]] = {}
        self._residents: dict[str, dict[str, ResidentRecord]] = {}
        self._free_vram: dict[str, float] = {}
        self._swap_history: collections.deque = collections.deque(
            maxlen=SWAP_HISTORY_LIMIT)
        for model_id, spec in (bookings or {}).items():
            self.book(model_id, spec)

    @property
    def clock(self):
        """The injectable monotonic clock (shared with the planner)."""
        return self._clock

    # -- bookings (nominal ceilings) --

    def book(self, model_id, vram_gb, source="registry", load_s=None) -> dict:
        """Book a nominal footprint for a model (GB).

        ``vram_gb`` may also be a booking dict (``{"vram_gb": …, "source": …,
        "load_s": …}``) or a ``Booking``. A higher-trust booking is never
        downgraded by a lower-trust one: measured (local bench) > engine
        reported (remote) > registry (advisory). A newer booking of equal
        trust replaces an older one (the latest bench wins)."""
        if isinstance(vram_gb, Booking):
            raw, source, load_s = vram_gb.vram_gb, vram_gb.source, load_s
            if load_s is None:
                load_s = vram_gb.load_s
        elif isinstance(vram_gb, dict):
            raw = vram_gb.get("vram_gb")
            source = str(vram_gb.get("source", source))
            load_s = vram_gb.get("load_s", load_s)
        else:
            raw = vram_gb
        gb = _finite_gb(raw)
        if gb is None:
            return {"ok": False, "model_id": model_id,
                    "error": "vram_gb must be a finite, non-negative number"}
        spec = Booking(vram_gb=gb, source=source, load_s=load_s)
        with self._lock:
            existing = self._bookings.get(model_id)
            if existing is not None:
                old_t = _BOOKING_TRUST.get(existing.source, 1)
                new_t = _BOOKING_TRUST.get(source, 1)
                if new_t < old_t:
                    return {"ok": True, "model_id": model_id,
                            "vram_gb": existing.vram_gb,
                            "source": existing.source,
                            "label": existing.label,
                            "kept": f"{existing.label} kept — a "
                                    f"{spec.label} figure is lower trust"}
            self._bookings[model_id] = spec
        return {"ok": True, "model_id": model_id, "vram_gb": spec.vram_gb,
                "source": spec.source, "label": spec.label}

    def booking(self, model_id) -> Booking | None:
        return self._bookings.get(model_id)

    def booked_vram_gb(self, model_id) -> float | None:
        b = self._bookings.get(model_id)
        return b.vram_gb if b else None

    def estimated_load_s(self, model_id) -> float | None:
        b = self._bookings.get(model_id)
        return b.load_s if b else None

    def record_measured_peak(self, model_id, peak_gb, load_s=None,
                             source="measured") -> dict:
        """The measured-footprint write path: book an observed bench peak as
        this model's ceiling. Routed through ``book`` so the no-downgrade
        invariant holds — a later lower-trust figure never replaces it. The
        default ``source="measured"`` is a local vendor-sampled bench;
        ``source="engine_reported"`` (B3) books a remote engine's own figure
        with its distinct provenance label."""
        return self.book(model_id, peak_gb, source=source, load_s=load_s)

    def booking_report(self) -> tuple[dict, ...]:
        """Every booked ceiling with an honest provenance label —
        ``measured peak (bench)`` vs ``published footprint (advisory)`` —
        so no report ever presents a measured booking as advisory or an
        advisory figure as measured."""
        with self._lock:
            items = sorted(self._bookings.items())
        return tuple(
            {"model_id": model_id, "vram_gb": b.vram_gb, "source": b.source,
             "label": b.label, "load_s": b.load_s}
            for model_id, b in items)

    def set_budget(self, target_id, vram_gb) -> dict:
        """Declare a target's nominal VRAM budget — the capacity every fit
        decision gates against (never live free VRAM)."""
        gb = _finite_gb(vram_gb)
        if gb is None:
            return {"ok": False, "target_id": target_id,
                    "error": "vram_gb must be a finite, non-negative number"}
        with self._lock:
            self._budgets[target_id] = gb
        return {"ok": True, "target_id": target_id, "vram_gb": gb}

    def budget_gb(self, target_id) -> float | None:
        return self._budgets.get(target_id)

    # -- pulls (weights on disk) --

    def record_pull(self, target_id, model_id, path, size_bytes=None,
                    source="") -> dict:
        if not str(path or "").strip():
            return {"ok": False, "model_id": model_id,
                    "error": "path must be a non-empty string"}
        try:
            size = int(size_bytes) if size_bytes is not None else None
        except (TypeError, ValueError):
            size = None
        rec = PullRecord(model_id=model_id, path=str(path), size_bytes=size,
                         source=str(source or ""))
        with self._lock:
            self._pulls.setdefault(target_id, {})[model_id] = rec
        return {"ok": True, "target_id": target_id, "model_id": model_id,
                "path": rec.path, "size_bytes": rec.size_bytes}

    def pulled(self, target_id) -> tuple[PullRecord, ...]:
        with self._lock:
            return tuple(self._pulls.get(target_id, {}).values())

    # -- residency (weights in memory) --

    def record_resident(self, target_id, model_id, engine, lane="chat",
                        provenance="") -> dict:
        now = self._clock()
        rec = ResidentRecord(model_id=model_id, engine=engine, lane=lane,
                             loaded_at=now, last_used=now,
                             provenance=str(provenance or ""))
        with self._lock:
            self._residents.setdefault(target_id, {})[model_id] = rec
        return {"ok": True, "target_id": target_id, "model_id": model_id,
                "engine": engine, "lane": lane, "provenance": rec.provenance}

    def record_released(self, target_id, model_id) -> bool:
        with self._lock:
            return self._residents.get(target_id, {}).pop(model_id, None) \
                   is not None

    def is_resident(self, target_id, model_id) -> bool:
        with self._lock:
            return model_id in self._residents.get(target_id, {})

    def residents(self, target_id) -> tuple[ResidentRecord, ...]:
        with self._lock:
            return tuple(self._residents.get(target_id, {}).values())

    def touch(self, target_id, model_id) -> bool:
        """Record a use: keeps the model inside the recency window."""
        now = self._clock()
        with self._lock:
            rec = self._residents.get(target_id, {}).get(model_id)
            if rec is None:
                return False
            self._residents[target_id][model_id] = ResidentRecord(
                model_id=rec.model_id, engine=rec.engine, lane=rec.lane,
                loaded_at=rec.loaded_at, last_used=now,
                provenance=rec.provenance)
            return True

    def last_used(self, target_id, model_id) -> float | None:
        with self._lock:
            rec = self._residents.get(target_id, {}).get(model_id)
            return rec.last_used if rec else None

    # -- live free VRAM (ordering only) --

    def set_free_vram(self, target_id, free_gb) -> dict:
        """Snapshot live free VRAM. ORDERING ONLY — never gate fit on it."""
        gb = _finite_gb(free_gb)
        if gb is None:
            return {"ok": False, "target_id": target_id,
                    "error": "free_gb must be a finite, non-negative number"}
        with self._lock:
            self._free_vram[target_id] = gb
        return {"ok": True, "target_id": target_id, "free_gb": gb}

    def free_vram_gb(self, target_id) -> float | None:
        return self._free_vram.get(target_id)

    # -- totals --

    def booked_total_gb(self, target_id) -> float:
        """Sum of booked ceilings over the residents (unbooked count 0 —
        see ``unbooked_residents`` for the accounting holes)."""
        with self._lock:
            recs = self._residents.get(target_id, {}).values()
            return sum(b.vram_gb for rec in recs
                       if (b := self._bookings.get(rec.model_id)) is not None)

    def unbooked_residents(self, target_id) -> tuple[str, ...]:
        with self._lock:
            return tuple(rec.model_id for rec in
                         self._residents.get(target_id, {}).values()
                         if rec.model_id not in self._bookings)

    # -- swap history (a bounded ring of measured swap records) --
    def record_swap(self, record) -> dict:
        """Record one executed swap in the bounded history ring
        (``SWAP_HISTORY_LIMIT`` entries — the oldest record drops off).

        The record is the measured truth of one swap: ``model_id``,
        ``evicted`` (the model ids actually evicted, in order), the measured
        per-phase seconds ``evict_seconds`` / ``load_seconds`` /
        ``ready_seconds`` (``None`` for a phase that never ran — never a
        guessed 0), the measured ``total_seconds``, ``ok``, and the
        ``warm_path`` that applied. Dry runs never record — they change
        nothing."""
        if not isinstance(record, dict):
            return {"ok": False,
                    "error": "swap record must be a dict row"}
        row = {
            "model_id": record.get("model_id"),
            "evicted": list(record.get("evicted") or ()),
            "evict_seconds": record.get("evict_seconds"),
            "load_seconds": record.get("load_seconds"),
            "ready_seconds": record.get("ready_seconds"),
            "total_seconds": record.get("total_seconds"),
            "ok": bool(record.get("ok")),
            "warm_path": record.get("warm_path"),
        }
        with self._lock:
            self._swap_history.append(row)
            kept = len(self._swap_history)
        return {"ok": True, "kept": kept, "limit": SWAP_HISTORY_LIMIT}

    def swap_history(self) -> tuple:
        """Every kept swap record, oldest first (bounded ring)."""
        with self._lock:
            return tuple(dict(row) for row in self._swap_history)

    def swap_stats(self) -> dict:
        """Aggregate swap stats over the bounded history: ``count`` (records
        kept), ``last_total_seconds`` and ``average_total_seconds`` in
        measured seconds — ``None`` when nothing has been measured yet, never
        an estimate."""
        with self._lock:
            rows = list(self._swap_history)
        if not rows:
            return {"count": 0, "last_total_seconds": None,
                    "average_total_seconds": None}
        totals = [r["total_seconds"] for r in rows
                  if isinstance(r.get("total_seconds"), (int, float))]
        last = rows[-1].get("total_seconds")
        return {
            "count": len(rows),
            "last_total_seconds": (float(last)
                                   if isinstance(last, (int, float))
                                   else None),
            "average_total_seconds": (sum(totals) / len(totals)
                                      if totals else None),
        }


# --- measured footprints ----------------------------------------------------

class MeasuredFootprintRecorder:
    """The measured-footprint recorder (roadmap §5a, P4 acceptance): sample
    VRAM through an INJECTED sampler during a caller-run bench window, then
    record the observed peak as that model's booked ceiling.

    This is the write path for measured bookings: ``record`` books the peak
    via ``ResidencyLedger.record_measured_peak`` (``source="measured"``) —
    through the same ``book`` gate that never lets a later published
    footprint downgrade a measured value. Ledger reports label the result
    ``measured peak (bench)`` so it is never confused with an advisory
    published figure.

    The bench window is the caller's: call ``sample()`` between bench
    iterations (the ``with`` block IS the window; closing it records the
    peak). A failing or non-finite sampler reading is skipped with an honest
    row, never a raise; a window with no valid reading fails closed — a
    booked ceiling must be an observation, never a guess."""

    def __init__(self, sampler, model_id, ledger, clock=None, source="measured"):
        self._sampler = sampler
        self._model_id = model_id
        self._ledger = ledger
        self._clock = clock or ledger.clock
        self._source = source
        self._peak: float | None = None
        self._samples = 0
        self._recorded: dict | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def peak_gb(self) -> float | None:
        """The highest valid reading seen so far (None until one lands)."""
        return self._peak

    @property
    def sample_count(self) -> int:
        """How many valid readings landed in the bench window."""
        return self._samples

    @property
    def recorded(self) -> dict | None:
        """The booking row from the last ``record()`` (None until recorded)."""
        return self._recorded

    def sample(self) -> dict:
        """Take one reading from the injected sampler and track the peak.
        Fail-soft: a raising or non-finite sampler is an honest row, never an
        exception — and never counted as a peak."""
        try:
            reading = self._sampler()
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {"ok": False, "model_id": self._model_id,
                    "peak_gb": self._peak,
                    "reason": f"VRAM sampler failed: "
                              f"{type(exc).__name__}: {exc}"}
        gb = _finite_gb(reading)
        if gb is None:
            return {"ok": False, "model_id": self._model_id,
                    "peak_gb": self._peak,
                    "reason": f"sampler returned {reading!r} — not a finite, "
                              f"non-negative VRAM reading; not counted as a "
                              f"peak"}
        self._samples += 1
        if self._peak is None or gb > self._peak:
            self._peak = gb
        return {"ok": True, "model_id": self._model_id, "gb": gb,
                "peak_gb": self._peak}

    def record(self, load_s=None) -> dict:
        """Record the observed peak as the model's booked ceiling. Fails
        closed with no valid readings — no guessed peak is ever booked."""
        if self._peak is None:
            row = {"ok": False, "model_id": self._model_id,
                   "error": "no valid VRAM readings in the bench window — "
                            "refusing to book a guessed peak (fail-closed)"}
            self._recorded = row
            return row
        row = self._ledger.record_measured_peak(self._model_id, self._peak,
                                                load_s=load_s,
                                                source=self._source)
        self._recorded = row
        return row

    # -- the caller-run bench window --

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._recorded is None:
            self.record()
        return None    # never swallow the caller's bench exception


# --- engine lifecycle adapters ----------------------------------------------

class EngineLifecycle(abc.ABC):
    """Per-engine residency control: warm, evict, and observe what is loaded.

    Every method returns a plain dict row (``ok`` + honest ``reason``) and
    never raises: an engine quirk must not sink the caller, and an unattended
    run degrades to a stated reason, never a stall. What a failed release
    means is a policy decision — the policy layer fails closed on it.

    ``WARM_PATH`` names, honestly, how this engine takes weights: a genuine
    warm-swap surface (Ollama ``keep_alive``, llama-swap routing, the ComfyUI
    workflow cache), a process restart, or engine-native placement. A swap
    records the label verbatim — it never claims a warm path the engine does
    not have."""

    engine_id: str = ""
    WARM_PATH: str = ("engine-native load (no warm-swap surface declared)")

    @abc.abstractmethod
    def ensure_loaded(self, model_id: str, **kwargs) -> dict:
        """Make ``model_id`` resident (warm path for this engine)."""

    @abc.abstractmethod
    def release(self, model_id: str) -> dict:
        """Evict ``model_id`` from memory."""

    @abc.abstractmethod
    def probe(self) -> dict:
        """Observational: what is resident right now. Fail-soft — an
        unreachable engine reports ``state="unknown"``, never raises."""

    def stops_externally_started(self, model_id: str = "") -> bool:
        """True only when ``release``/eviction would STOP an engine runtime
        the user started themselves (an externally-started server) — the B8
        gate signal.

        The default is ``False``: process adapters (llama.cpp, vLLM) only
        terminate a handle they started themselves and HTTP adapters never
        stop a server, so they can never stop an externally-started one. The
        container adapter overrides this: it can ``docker stop`` a container
        the user brought up (e.g. via compose), which is the case the
        approval gate must cover. ``model_id`` is accepted for a uniform
        signature; container adapters gate per-container, not per-model."""
        return False


def _http_json(method: str, url: str, body: dict | None = None,
               timeout: float = 10.0) -> dict:
    """The default injectable HTTP seam: stdlib JSON request/response."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", "replace")
    # An empty success body (e.g. ComfyUI's POST /free answers 200 with no
    # body) is a valid JSON response of ``{}`` here, not a parse error.
    return json.loads(text) if text.strip() else {}


class OllamaLifecycle(EngineLifecycle):
    """Ollama lifecycle: warm with a generate call carrying ``keep_alive``,
    evict with ``keep_alive=0`` — the only unload Ollama has.

    The HTTP function is injectable; the default is stdlib urllib."""

    engine_id = "ollama"
    WARM_PATH = ("warm-swap via keep_alive (the generate call parks the "
                 "weights resident; evict with keep_alive=0)")

    def __init__(self, base_url="http://127.0.0.1:11434", http=None,
                 keep_alive="5m"):
        self._root = (base_url or "").rstrip("/")
        self._http = http or _http_json
        self._keep_alive = keep_alive

    def ensure_loaded(self, model_id, *, keep_alive=None, **_ignored) -> dict:
        # An empty-prompt generate with a keep_alive duration is Ollama's
        # warm path: it loads the weights and parks them resident.
        body = {"model": model_id, "prompt": "", "stream": False,
                "keep_alive": self._keep_alive if keep_alive is None
                else keep_alive}
        try:
            self._http("POST", f"{self._root}/api/generate", body)
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {"ok": False, "engine": self.engine_id, "action": "load",
                    "model_id": model_id,
                    "reason": f"ollama warm call failed: "
                              f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "engine": self.engine_id, "action": "load",
                "model_id": model_id, "keep_alive": body["keep_alive"],
                "reason": f"warmed {model_id} with "
                          f"keep_alive={body['keep_alive']!r}"}

    def release(self, model_id) -> dict:
        body = {"model": model_id, "prompt": "", "stream": False,
                "keep_alive": 0}
        try:
            self._http("POST", f"{self._root}/api/generate", body)
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {"ok": False, "engine": self.engine_id, "action": "release",
                    "model_id": model_id,
                    "reason": f"ollama evict call failed: "
                              f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "engine": self.engine_id, "action": "release",
                "model_id": model_id, "keep_alive": 0,
                "reason": f"evicted {model_id} with keep_alive=0"}

    def probe(self) -> dict:
        try:
            data = self._http("GET", f"{self._root}/api/ps")
        except Exception as exc:  # noqa: BLE001 — probes fail soft
            return {"ok": False, "engine": self.engine_id, "state": "unknown",
                    "resident": [],
                    "reason": f"ollama /api/ps unreachable: "
                              f"{type(exc).__name__}: {exc}"}
        models = data.get("models") if isinstance(data, dict) else None
        resident = [m.get("name", "") for m in (models or [])
                    if isinstance(m, dict)]
        return {"ok": True, "engine": self.engine_id, "state": "observed",
                "resident": [n for n in resident if n]}


class LlamaSwapLifecycle(EngineLifecycle):
    """llama-swap is a pass-through: it swaps models on demand by route.

    Observational adapter only — llama-swap owns its own swap and reclaim
    policy and exposes no eviction control, so ``ensure_loaded``/``release``
    report state and NEVER drive the engine. Keep the ledger honest from
    ``probe``."""

    engine_id = "llama-swap"
    WARM_PATH = ("warm-swap via routing (llama-swap swaps models on demand "
                 "by route; no explicit load or eviction control)")
    NO_CONTROL = ("llama-swap swaps models on demand by route; "
                  "no explicit load or eviction control")

    def __init__(self, base_url="http://127.0.0.1:8999", http=None):
        self._root = (base_url or "").rstrip("/")
        self._http = http or _http_json

    def ensure_loaded(self, model_id, **_ignored) -> dict:
        return {"ok": True, "engine": self.engine_id, "action": "load",
                "model_id": model_id, "state": "pass-through",
                "reason": self.NO_CONTROL
                         + " (loading is engine-native on first use)"}

    def release(self, model_id) -> dict:
        return {"ok": True, "engine": self.engine_id, "action": "release",
                "model_id": model_id, "state": "pass-through",
                "reason": self.NO_CONTROL
                         + " (reclaim happens on the engine's own swap)"}

    def probe(self) -> dict:
        try:
            data = self._http("GET", f"{self._root}/models")
        except Exception as exc:  # noqa: BLE001 — probes fail soft
            return {"ok": False, "engine": self.engine_id, "state": "unknown",
                    "resident": [],
                    "reason": f"llama-swap /models unreachable: "
                              f"{type(exc).__name__}: {exc}"}
        if isinstance(data, dict):
            models = data.get("models") or []
        else:
            models = data if isinstance(data, list) else []
        resident = [m.get("name", "") for m in models if isinstance(m, dict)]
        return {"ok": True, "engine": self.engine_id, "state": "observed",
                "resident": [n for n in resident if n],
                "reason": self.NO_CONTROL}


class RuntimeHandle(Protocol):
    """The injectable runtime-handle contract — what a started engine answers
    to, whether it runs as a bare process (llama-server, the vLLM server) or
    as a container.

    ``kind`` names the runtime honestly ('process' | 'container') so a caller
    reading a handle knows what it holds; the three methods are the whole
    contract — ``poll`` for liveness, ``terminate`` to stop, ``wait`` to join.
    The process adapters below only ever call the three methods, so a handle
    that carries no ``kind`` still satisfies this contract structurally."""

    kind: str
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


# Back-compat alias: ProcessHandle WAS this contract (poll/terminate/wait),
# so the llama.cpp and vLLM adapters — and their fake-handle tests, which use
# handles with no ``kind`` — keep working unchanged.
ProcessHandle = RuntimeHandle


# The environment keys an engine server may inherit beyond the four base
# keys: forwarded only when the caller names them AND the parent has them.
# HF_HOME lets vLLM/llama.cpp reuse a downloaded-weights cache; the
# CUDA/HIP visibility keys pin which devices the server may use.
_ENGINE_ENV_EXTRAS = ("HF_HOME", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


def _minimal_engine_env(extras=_ENGINE_ENV_EXTRAS) -> dict:
    """The smallest environment an engine server child needs: PATH, HOME,
    LANG and TMPDIR, plus any caller-named engine extras (see
    ``_ENGINE_ENV_EXTRAS``) that are actually present in the parent.

    Nothing else from the parent is forwarded. Credentials are deliberately
    excluded — no HF_TOKEN and no provider API keys reach the child: a model
    pull goes through the pull path (pull.py), never through a spawned
    server's environment. ``tools.py`` keeps its own installer-side copy of
    this helper (the import direction is tools -> broker, so broker never
    imports tools).
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME") or os.path.expanduser("~"),
        "LANG": os.environ.get("LANG", "C"),
        "TMPDIR": os.environ.get("TMPDIR") or "/tmp",
    }
    for key in extras or ():
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


# Flags that (re)bind a server's listening address. A caller-supplied one of
# these in extra_args must resolve to loopback or the lifecycle refuses to
# start: an engine server is loopback-only by policy.
_BIND_FLAGS = ("--host", "--host-ip", "--listen", "--bind", "--bind-address")
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "ip6-localhost")


def _non_loopback_bind(extra_args) -> str:
    """The first caller-supplied bind flag in ``extra_args`` whose value does
    not resolve to loopback, as a shown token (``--host 0.0.0.0``), or ``''``
    when every bind flag is loopback or none is present.

    Covers both ``--host=value`` and ``--host value`` spellings. A bind flag
    with a missing value is treated as non-loopback (fail closed)."""
    args = list(extra_args or ())
    i = 0
    while i < len(args):
        token = str(args[i])
        flag, sep, inline = token.partition("=")
        if flag in _BIND_FLAGS:
            if sep:
                value = inline
                shown = token
            else:
                value = str(args[i + 1]) if i + 1 < len(args) else ""
                shown = f"{flag} {value}".strip()
                i += 1
            if value not in _LOOPBACK_HOSTS:
                return shown
        i += 1
    return ""


def _spawn_process(argv, env) -> ProcessHandle:
    """Default process runner for llama-server / vLLM: the one real spawn
    seam. The child gets exactly ``env`` — never the parent's environment, so
    no Hermes provider keys or tokens leak into the server (see
    ``_minimal_engine_env``). A handle answers ``poll()`` / ``terminate()`` /
    ``wait(timeout=…)``. A real ``Popen`` satisfies those three methods but
    carries no ``kind`` label — hence the ignore: the adapters only ever call
    the three methods, never ``kind`` (see ``RuntimeHandle``)."""
    return subprocess.Popen(list(argv), env=dict(env))  # type: ignore[return-value]


class LlamaCppServerLifecycle(EngineLifecycle):
    """llama.cpp llama-server is a process: start it on a weights path, stop
    it to evict (roadmap §3: big models need process stop/start, not engine
    sleep/wake). The process runner and the binary lookup are injectable so
    tests spawn nothing — and a missing binary fails closed: no argv is even
    built, no half-started server."""

    engine_id = "llamacpp"
    WARM_PATH = ("process restart (start llama-server on the weights; no "
                 "warm-swap surface)")

    def __init__(self, binary="llama-server", port=8080, extra_args=(),
                 runner=None, which=None, env_extras=None):
        self._binary = binary
        self._port = port
        self._extra_args = tuple(extra_args)
        self._runner = runner or _spawn_process
        self._which = which or shutil.which
        self._env_extras = tuple(
            _ENGINE_ENV_EXTRAS if env_extras is None else env_extras)
        self._handle: ProcessHandle | None = None
        self._serving: str | None = None

    def ensure_loaded(self, model_id, *, weights_path=None, port=None,
                      extra_args=None, **_ignored) -> dict:
        row = {"ok": False, "engine": self.engine_id, "action": "load",
               "model_id": model_id}
        if not str(weights_path or "").strip():
            return {**row, "reason": "llama.cpp needs an explicit weights "
                                     "path — refusing to guess (fail-closed)"}
        found = self._which(self._binary)
        if not found:
            return {**row, "reason": f"llama.cpp binary {self._binary!r} not "
                                     f"found on PATH — refusing to start "
                                     f"(fail-closed)"}
        extras = list(self._extra_args if extra_args is None else extra_args)
        hostile = _non_loopback_bind(extras)
        if hostile:
            return {**row, "reason": f"llama.cpp extra_args {hostile!r} would "
                                     f"rebind the server away from loopback — "
                                     f"refusing to start (fail-closed; the "
                                     f"server binds 127.0.0.1 only)"}
        bound_port = str(self._port if port is None else port)
        argv = [found, "-m", str(weights_path),
                "--host", "127.0.0.1", "--port", bound_port]
        argv += extras
        env = _minimal_engine_env(self._env_extras)
        try:
            handle = self._runner(argv, env)
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {**row, "reason": f"llama-server start failed: "
                                     f"{type(exc).__name__}: {exc}"}
        self._handle = handle
        self._serving = model_id
        return {"ok": True, "engine": self.engine_id, "action": "load",
                "model_id": model_id, "argv": argv,
                "reason": f"started llama-server for {model_id} on port "
                          f"{bound_port} (binds 127.0.0.1 only)"}

    def release(self, model_id) -> dict:
        if self._handle is None:
            return {"ok": True, "engine": self.engine_id, "action": "release",
                    "model_id": model_id,
                    "reason": f"no llama-server process for {model_id} — "
                              f"nothing to stop"}
        handle, self._handle, self._serving = self._handle, None, None
        try:
            handle.terminate()
            handle.wait(timeout=10.0)
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {"ok": False, "engine": self.engine_id, "action": "release",
                    "model_id": model_id,
                    "reason": f"llama-server stop failed: "
                              f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "engine": self.engine_id, "action": "release",
                "model_id": model_id,
                "reason": f"stopped llama-server for {model_id}"}

    def probe(self) -> dict:
        if self._handle is None:
            return {"ok": True, "engine": self.engine_id, "state": "idle",
                    "resident": []}
        running = self._handle.poll() is None
        return {"ok": True, "engine": self.engine_id,
                "state": "running" if running else "stopped",
                "resident": [self._serving] if (running and self._serving)
                           else []}


class ComfyUiLifecycle(EngineLifecycle):
    """ComfyUI lifecycle: readiness plus the explicit cache free between
    back-to-back generative jobs (roadmap §3 — engines hold allocations after
    a job; an explicit cache free between consecutive generative jobs keeps
    the next one honest).

    ComfyUI loads models per-workflow from its workflow queue and exposes no
    per-model weights pin or eviction control, so ``ensure_loaded`` is a
    READINESS + CACHE-FREE contract, never a weights pin: it confirms ComfyUI
    answers (``GET /system_stats``) and — when the previous generative job
    still holds allocations — signals the cache free first. ``release`` is
    the same explicit cache free after a job. The free rides ComfyUI's
    documented ``POST /free`` route (``{"unload_models": true,
    "free_memory": true}``); ComfyUI applies it when the queue next goes
    idle, so the call returns immediately.

    The HTTP function is injectable; the default is stdlib urllib."""

    engine_id = "comfyui"
    WARM_PATH = ("warm-swap via the workflow cache (readiness + explicit "
                 "cache-free between generative jobs)")
    FREE_BODY = {"unload_models": True, "free_memory": True}
    CONTRACT = "readiness + cache-free"
    RELEASE_CONTRACT = "explicit cache-free"

    def __init__(self, base_url="http://127.0.0.1:8188", http=None):
        self._root = (base_url or "").rstrip("/")
        self._http = http or _http_json
        self._allocations_held = False

    def _free_cache(self, model_id, action):
        """The explicit cache free (``POST /free``): ``None`` on success, an
        honest error row on failure — never raises."""
        try:
            self._http("POST", f"{self._root}/free", dict(self.FREE_BODY))
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {"ok": False, "engine": self.engine_id, "action": action,
                    "model_id": model_id, "cache_freed": False,
                    "reason": f"comfyui cache free (POST /free) failed: "
                              f"{type(exc).__name__}: {exc}"}
        return None

    def ensure_loaded(self, model_id, **_ignored) -> dict:
        row = {"ok": False, "engine": self.engine_id, "action": "load",
               "model_id": model_id, "contract": self.CONTRACT,
               "cache_freed": False}
        try:
            stats = self._http("GET", f"{self._root}/system_stats")
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {**row, "reason": f"comfyui readiness check "
                                     f"(GET /system_stats) failed: "
                                     f"{type(exc).__name__}: {exc}"}
        if not (isinstance(stats, dict)
                and ("system" in stats or "devices" in stats)):
            return {**row, "reason": "comfyui readiness check: unexpected "
                                     "/system_stats response — ComfyUI is "
                                     "not ready"}
        freed = False
        if self._allocations_held:
            err = self._free_cache(model_id, "load")
            if err is not None:
                return err
            freed = True
        self._allocations_held = True
        reason = ("ComfyUI is ready — readiness + cache-free only: ComfyUI "
                  "loads models per-workflow, not a weights pin")
        if freed:
            reason += ("; freed held allocations from the previous "
                       "generative job first (back-to-back generative jobs "
                       "free the cache between them)")
        return {"ok": True, "engine": self.engine_id, "action": "load",
                "model_id": model_id, "state": "ready",
                "contract": self.CONTRACT, "cache_freed": freed,
                "reason": reason}

    def release(self, model_id) -> dict:
        base = {"engine": self.engine_id, "action": "release",
                "model_id": model_id, "contract": self.RELEASE_CONTRACT}
        if not self._allocations_held:
            return {**base, "ok": True, "cache_freed": False,
                    "reason": "no held allocations from a previous "
                              "generative job — nothing to free"}
        err = self._free_cache(model_id, "release")
        if err is not None:
            return err
        self._allocations_held = False
        return {**base, "ok": True, "cache_freed": True,
                "reason": "explicit cache free between generative jobs "
                          "(POST /free: unload_models + free_memory) — "
                          "ComfyUI holds allocations after a job and "
                          "reclaims them when the queue next goes idle"}

    def probe(self) -> dict:
        try:
            stats = self._http("GET", f"{self._root}/system_stats")
        except Exception as exc:  # noqa: BLE001 — probes fail soft
            return {"ok": False, "engine": self.engine_id, "state": "unknown",
                    "resident": [],
                    "allocations_held": self._allocations_held,
                    "reason": f"comfyui /system_stats unreachable: "
                              f"{type(exc).__name__}: {exc}"}
        ready = isinstance(stats, dict) and ("system" in stats
                                             or "devices" in stats)
        return {"ok": ready, "engine": self.engine_id,
                "state": "observed" if ready else "unknown",
                "resident": [], "allocations_held": self._allocations_held,
                "reason": "ComfyUI loads models per-workflow and exposes no "
                          "weights-residency listing — residency is managed "
                          "through the job lifecycle (readiness + cache-free)"}


# Spelling aliases so callers can name the ComfyUI acronym either way.
ComfyUILifecycle = ComfyUiLifecycle


class VllmServerLifecycle(EngineLifecycle):
    """vLLM is a process: ``vllm serve <model>`` starts the server, stopping
    the process evicts (roadmap §3). PROCESS STOP/START ONLY — engine
    sleep/wake is deliberately not used: some engines' wake endpoints are
    unreliable upstream, and process restart is the reliable path for large
    loads.

    The model is the launch argument: a local weights path when one is given
    (and it must be on disk — missing weights fail closed), or the model id
    itself when vLLM is left to resolve it. The process runner and the binary
    lookup are injectable so tests spawn nothing — and a missing binary fails
    closed: no argv is even built, no half-started server."""

    engine_id = "vllm"
    WARM_PATH = ("process restart (process stop/start; engine sleep/wake is "
                 "not used — no warm-swap surface)")
    LIFECYCLE = "process stop/start (engine sleep/wake is not used)"

    def __init__(self, binary="vllm", port=8000, extra_args=(),
                 runner=None, which=None, exists=None, env_extras=None):
        self._binary = binary
        self._port = port
        self._extra_args = tuple(extra_args)
        self._runner = runner or _spawn_process
        self._which = which or shutil.which
        self._exists = exists or os.path.exists
        self._env_extras = tuple(
            _ENGINE_ENV_EXTRAS if env_extras is None else env_extras)
        self._handle: ProcessHandle | None = None
        self._serving: str | None = None

    def ensure_loaded(self, model_id, *, weights_path=None, port=None,
                      extra_args=None, **_ignored) -> dict:
        row = {"ok": False, "engine": self.engine_id, "action": "load",
               "model_id": model_id, "lifecycle": self.LIFECYCLE}
        weights = str(weights_path or "").strip()
        if weights:
            if not self._exists(weights):
                return {**row, "reason": f"vLLM weights not found at "
                                         f"{weights!r} — refusing to start "
                                         f"(fail-closed)"}
            serve_target = weights
        else:
            serve_target = str(model_id)
        found = self._which(self._binary)
        if not found:
            return {**row, "reason": f"vLLM binary {self._binary!r} not "
                                     f"found on PATH — refusing to start "
                                     f"(fail-closed)"}
        extras = list(self._extra_args if extra_args is None else extra_args)
        hostile = _non_loopback_bind(extras)
        if hostile:
            return {**row, "reason": f"vLLM extra_args {hostile!r} would "
                                     f"rebind the server away from loopback — "
                                     f"refusing to start (fail-closed; the "
                                     f"server binds 127.0.0.1 only)"}
        bound_port = str(self._port if port is None else port)
        argv = [found, "serve", serve_target,
                "--host", "127.0.0.1", "--port", bound_port]
        argv += extras
        env = _minimal_engine_env(self._env_extras)
        try:
            handle = self._runner(argv, env)
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {**row, "reason": f"vLLM server start failed: "
                                     f"{type(exc).__name__}: {exc}"}
        self._handle = handle
        self._serving = model_id
        return {"ok": True, "engine": self.engine_id, "action": "load",
                "model_id": model_id, "argv": argv,
                "lifecycle": self.LIFECYCLE,
                "reason": f"started the vLLM server for {model_id} on port "
                          f"{bound_port} (binds 127.0.0.1 only) (process "
                          f"stop/start — engine sleep/wake is not used; "
                          f"process restart is the reliable path)"}

    def release(self, model_id) -> dict:
        if self._handle is None:
            return {"ok": True, "engine": self.engine_id, "action": "release",
                    "model_id": model_id, "lifecycle": self.LIFECYCLE,
                    "reason": f"no vLLM server process for {model_id} — "
                              f"nothing to stop"}
        handle, self._handle, self._serving = self._handle, None, None
        try:
            handle.terminate()
            handle.wait(timeout=10.0)
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {"ok": False, "engine": self.engine_id, "action": "release",
                    "model_id": model_id, "lifecycle": self.LIFECYCLE,
                    "reason": f"vLLM server stop failed: "
                              f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "engine": self.engine_id, "action": "release",
                "model_id": model_id, "lifecycle": self.LIFECYCLE,
                "reason": f"stopped the vLLM server for {model_id} (process "
                          f"stop/start — engine sleep/wake is not used; "
                          f"process restart is the reliable path)"}

    def probe(self) -> dict:
        if self._handle is None:
            return {"ok": True, "engine": self.engine_id, "state": "idle",
                    "resident": []}
        running = self._handle.poll() is None
        return {"ok": True, "engine": self.engine_id,
                "state": "running" if running else "stopped",
                "resident": [self._serving] if (running and self._serving)
                           else []}


# Spelling alias matching the llama.cpp adapter's naming shape.
VllmLifecycle = VllmServerLifecycle


# --- container engines (the docker dance) ------------------------------------

def _docker_argv(args: list) -> list:
    return ["docker", *[str(a) for a in args]]


def _minimal_docker_env() -> dict:
    """The trimmed environment for the host ``docker`` CLI: PATH, HOME and
    TMPDIR it needs to find the binary, locate its config and use a writable
    temp dir, plus ``DOCKER_HOST`` when the parent explicitly points the client
    at a socket/host. Nothing else from the parent rides along — in particular
    the Hermes provider keys in the parent environment must NOT leak into the
    docker client (same hygiene as ``_minimal_engine_env``)."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME") or os.path.expanduser("~"),
        "TMPDIR": os.environ.get("TMPDIR") or "/tmp",
    }
    docker_host = os.environ.get("DOCKER_HOST")
    if docker_host:
        env["DOCKER_HOST"] = docker_host
    return env


def _run_docker(args):
    """Default docker runner: ``subprocess ['docker', <args>]`` — the one real
    docker seam. The facade runs on a host where the docker CLI is available;
    unit tests inject a fake and touch no docker.

    stdout/stderr are captured (PIPE) so the ``docker inspect`` path can read
    the container's state from the child; start/stop emit a few bytes at most,
    so the unread pipe never fills. The child gets the trimmed docker
    environment (``_minimal_docker_env``), never the full parent env."""
    return subprocess.Popen(_docker_argv(args), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=_minimal_docker_env())


def _docker_state(runner, container) -> bool | None:
    """Best-effort container running state from ``docker inspect``: ``True`` /
    ``False`` when the output is readable, ``None`` when it is not.

    Raises ONLY what the runner itself raises — the caller decides how to
    degrade. ``probe`` reports a raised runner fail-soft, and otherwise falls
    back to the adapter's last tracked state when the output is unreadable.
    Reading the child's stdout is optional because an injected fake runner
    may return an opaque handle with no ``communicate``."""
    proc = runner(["inspect", "-f", "{{.State.Running}}", container])
    communicate = getattr(proc, "communicate", None)
    if not callable(communicate):
        return None
    try:
        stdout, _ = communicate(timeout=5.0)
    except Exception:  # noqa: BLE001 — no readable output, not an error here
        return None
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    token = (stdout or "").strip().lower()
    if token == "true":
        return True
    if token == "false":
        return False
    return None


def _http_healthy(url, timeout: float = 2.0) -> bool:
    """Default health probe: a plain urllib GET, HTTP 200 means healthy.
    Any failure (connection refused while the engine is still booting, a
    timeout, a non-200) reads as not-yet-healthy, never an exception — the
    caller polls until the cold-boot budget expires."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(resp.getcode()) == 200
    except Exception:  # noqa: BLE001 — not-yet-up is a normal poll answer
        return False


class DockerContainerLifecycle(EngineLifecycle):
    """A containerized engine: start the container, wait for its health
    endpoint, stop the container to evict.

    Production parity (the predecessor broker's ``ContainerControl``): a side
    model (ComfyUI) runs as a container so the broker can stop it before a
    bigger lane wakes — a resident side model OOMs the waking lane (a
    tensor-parallel chat server books most of each accelerator at 0.95
    utilization). ``docker stop`` frees the GPU
    instantly, which is the whole reason to containerize the engine.

    ``ensure_loaded`` starts the container only when it is not already
    running, then polls ``health_url`` until it answers 200 or the cold-boot
    budget expires. ``release`` issues ``docker stop`` and then WAITS for the
    container to actually reach stopped, bounded by ``stop_budget_s``: VRAM
    is freed only when the container exits, so returning as soon as the stop
    was issued would race the next load's allocation (OOM on the 24 GB cards).
    Both drive the host docker CLI through the injectable ``runner``; tests
    inject fakes and NEVER call real docker.

    ``boot_budget_s`` is the cold-boot allowance the CALLER passes (vLLM
    ~90-170 s warm / ~8 min cold, ComfyUI minutes): this adapter never
    hardcodes a per-engine number, and a boot that overruns the budget is an
    honest failure, never a hang. ``sleep`` / ``clock`` are injected so the
    budget tests run instantly. ``gpu_count`` is carried honestly in the rows
    (the container is pinned to that many devices) — it is not a fit number.
    """

    WARM_PATH = ("container stop/start (docker stop frees the GPU instantly; "
                 "cold boot waits for the health endpoint)")
    LIFECYCLE = "container stop/start (docker stop frees the GPU instantly)"
    POLL_INTERVAL_S = 2.0

    def __init__(self, engine_id, *, container: str, health_url: str,
                 boot_budget_s: float, gpu_count: int = 1, runner=None,
                 sleep=None, clock=None, stop_budget_s: float = 60.0):
        self.engine_id = str(engine_id)
        self._container = str(container)
        self._health_url = str(health_url)
        self._boot_budget_s = float(boot_budget_s)
        self._gpu_count = int(gpu_count)
        self._runner = runner or _run_docker
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        # The release-side stop budget: ``release`` waits for the container to
        # actually reach stopped within this window before it reports the GPU
        # freed. Freeing VRAM happens only when the container exits, so a
        # release followed straight away by the next load would race the free
        # (OOM on the real 24 GB cards) if release returned as soon as
        # ``docker stop`` was issued. ``docker stop`` normally blocks until
        # exit, so this is an upper bound, not a per-release wait.
        self._stop_budget_s = float(stop_budget_s)
        # The last state this adapter observed. ``docker start`` against an
        # already-running container is the engine's own no-op, so tracking is
        # safe; ``probe`` still asks docker when it can, and resets this flag
        # from whatever docker reports. ``_booting`` is the honest distinct
        # state for a container that STARTED but never answered healthy: it is
        # up (possibly holding partial VRAM), so it is never reported stopped.
        self._running = False
        self._booting = False
        self._serving: str | None = None
        # B8 server provenance: True only when THIS adapter started the
        # container; False when it adopted a container the user already had
        # running (externally started — e.g. via compose) or when the prior
        # state was unreadable (fail-safe: treated as external, so a stop is
        # gated). Reset to False when the container is stopped.
        self._started_by_plugin = False

    def ensure_loaded(self, model_id, **_ignored) -> dict:
        row = {"ok": False, "engine": self.engine_id, "action": "load",
               "model_id": model_id, "container": self._container,
               "lifecycle": self.LIFECYCLE,
               # honest loading label while the container cold-boots
               "loading_state": loading_state(self._boot_budget_s)}
        if self._running and not self._booting:
            self._serving = model_id
            return {**row, "ok": True, "state": "ready",
                    "gpu_count": self._gpu_count,
                    "reason": f"container {self._container!r} is already "
                              f"running — nothing to start"}
        # A container left ``_booting`` from a prior overrun is already up:
        # re-issue no start, just retry readiness against the health endpoint.
        did_start = False
        if not self._running:
            # B8 provenance: was the container already running (externally
            # started — e.g. by the user / compose) before this load? Only a
            # container this load actually starts is plugin-owned; adopting a
            # running one, or an unreadable prior state, is treated as
            # external (fail-safe) so a later stop rides the gate.
            try:
                prior = _docker_state(self._runner, self._container)
            except Exception:  # noqa: BLE001 — unreadable -> fail-safe external
                prior = None
            self._started_by_plugin = (prior is False)
            try:
                self._runner(["start", self._container])
            except Exception as exc:  # noqa: BLE001 — honest row, never a raise
                return {**row, "reason": f"docker start {self._container!r} "
                                         f"failed: {type(exc).__name__}: {exc}"}
            self._running = True
            did_start = (prior is not True)
        start = self._clock()
        # Poll the health endpoint until it answers 200 or the cold-boot
        # budget expires. The budget is the caller's; overrunning it is an
        # honest failure, never a hang.
        while True:
            if _http_healthy(self._health_url):
                self._running = True
                self._booting = False
                self._serving = model_id
                started = ("started container" if did_start
                           else "container already up")
                return {**row, "ok": True, "state": "ready",
                        "gpu_count": self._gpu_count,
                        "reason": f"{started} {self._container!r} and it "
                                  f"answered healthy at {self._health_url} — "
                                  f"the engine is ready"}
            elapsed = self._clock() - start
            if elapsed >= self._boot_budget_s:
                # ``docker start`` succeeded: the container runs (and may hold
                # partial VRAM) even though it never answered healthy. Keep an
                # honest distinct state — never assert it stopped.
                self._running = True
                self._booting = True
                self._serving = model_id
                return {**row, "state": "booting", "reason":
                        f"cold-boot budget {self._boot_budget_s:g}s expired "
                        f"after {elapsed:g}s waiting for {self._health_url} "
                        f"— container {self._container!r} is up but never "
                        f"became healthy (may hold partial VRAM; the caller "
                        f"passes the real budget: vLLM warm ~90-170s / cold "
                        f"~8min, ComfyUI minutes)"}
            self._sleep(self.POLL_INTERVAL_S)

    def _await_stopped(self, handle) -> bool:
        """Wait for the container to reach stopped, bounded by
        ``stop_budget_s``.

        Returns ``True`` when the container is confirmed stopped (or when
        docker state is unreadable but the stop CLI's own join returned),
        ``False`` when the container is still running at budget — never a
        false ``released`` while it still holds VRAM. A failed inspect is
        fail-soft (treated as unreadable, not as running).
        """
        start = self._clock()
        joined = False
        while True:
            try:
                observed = _docker_state(self._runner, self._container)
            except Exception:  # noqa: BLE001 — unreadable, not proven running
                observed = None
            if observed is False:
                return True
            if observed is None:
                # ``docker inspect`` gave no readable state. Fall back to the
                # stop CLI's own contract: ``docker stop`` blocks until the
                # container exits, so join the handle ONCE within the budget.
                # A handle with no join seam (an opaque fake) is taken at
                # face value — the tracked flag already says we stopped it.
                if not joined:
                    joined = True
                    if not self._join(handle):
                        return False
                return True
            # observed is True — the container is still running. Wait it out
            # against the injectable clock so the budget is bounded.
            if self._clock() - start >= self._stop_budget_s:
                return False
            self._sleep(self.POLL_INTERVAL_S)

    def _join(self, handle) -> bool:
        """Join a stop handle within ``stop_budget_s``: ``True`` when it has
        finished (or exposes no join seam), ``False`` when it is still
        running. Never raises."""
        wait = getattr(handle, "wait", None)
        if not callable(wait):
            return True
        try:
            wait(timeout=self._stop_budget_s)
        except TypeError:
            try:
                wait()
            except Exception:  # noqa: BLE001 — honest, never a raise
                return True
        except Exception:  # noqa: BLE001 — honest, never a raise
            return True
        poll = getattr(handle, "poll", None)
        if callable(poll):
            try:
                return poll() is not None
            except Exception:  # noqa: BLE001
                return True
        return True

    def stops_externally_started(self, model_id: str = "") -> bool:
        """True when ``release`` would ``docker stop`` a container Model Forge
        did NOT start — an externally-started runtime (B8 gate signal)."""
        return bool(self._running and not self._started_by_plugin)

    def release(self, model_id) -> dict:
        external = bool(self._running and not self._started_by_plugin)
        base = {"engine": self.engine_id, "action": "release",
                "model_id": model_id, "container": self._container,
                "lifecycle": self.LIFECYCLE,
                # B8: whether this stop targets a server the user started
                "stopped_external": external,
                "provenance": ("external" if external else "plugin")}
        try:
            handle = self._runner(["stop", self._container])
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {**base, "ok": False, "state": "running",
                    "reason": f"docker stop {self._container!r} failed: "
                              f"{type(exc).__name__}: {exc} — the container "
                              f"may still hold VRAM"}
        # VRAM is freed only when the container exits: wait for it to reach
        # stopped within the stop budget before reporting the release. Never
        # claim ``released`` while the container is still running.
        if not self._await_stopped(handle):
            return {**base, "ok": False, "state": "stopping",
                    "reason": f"docker stop {self._container!r} was issued "
                              f"but the container had not reached stopped "
                              f"when the {self._stop_budget_s:g}s stop "
                              f"budget expired — NOT released (its VRAM is "
                              f"not yet free; the container is still "
                              f"stopping)"}
        self._running = False
        self._booting = False
        self._serving = None
        return {**base, "ok": True, "state": "released",
                "reason": f"stopped container {self._container!r} (docker "
                          f"stop) — confirmed stopped within the "
                          f"{self._stop_budget_s:g}s stop budget; its VRAM "
                          f"is freed instantly once the container exits"}

    def probe(self) -> dict:
        try:
            observed = _docker_state(self._runner, self._container)
        except Exception as exc:  # noqa: BLE001 — probes fail soft
            # Sibling adapters (ollama/llama-swap/comfyui) return ok False on
            # a failed probe; match that convention here.
            return {"ok": False, "engine": self.engine_id, "state": "unknown",
                    "resident": [], "container": self._container,
                    "reason": f"container {self._container!r} state probe "
                              f"failed: {type(exc).__name__}: {exc} — "
                              f"residency unknown (fail-soft)"}
        if observed is None:
            observed = self._running
            source = "tracked state — docker inspect unreadable"
        else:
            # Docker is authoritative: reset the in-memory flag from what it
            # actually reports, so a side container started/stopped outside
            # this adapter is reflected honestly.
            self._running = observed
            source = "docker inspect"
            if not observed:
                self._booting = False
        if not observed:
            state = "stopped"
        elif self._booting:
            state = "booting"
        else:
            state = "running"
        return {"ok": True, "engine": self.engine_id, "state": state,
                "resident": [self._serving] if (observed and self._serving)
                           else [],
                "container": self._container, "gpu_count": self._gpu_count,
                "reason": f"container {self._container!r} is {state} "
                          f"({source})"}


def runtime_lifecycle_for(entry, **seams):
    """Build the runtime lifecycle a registry entry declares, or ``None``.

    ``entry.runtime`` is the ModelEntry runtime dict: ``{"kind": "container",
    "name": …, "health_url": …, "boot_budget_s": …, "gpu_count": …}`` builds a
    ``DockerContainerLifecycle``; anything else (no runtime, or a
    ``kind="process"`` entry) returns ``None`` so process engines keep using
    the existing adapters (llama.cpp, vLLM).

    ``seams`` (``runner`` / ``sleep`` / ``clock``, plus an optional
    ``engine_id`` override — default: the entry's first engine id) pass
    straight through to the adapter. Fail closed: a container entry with no
    boot budget raises — this factory never invents a per-engine number, so a
    malformed registry surfaces loudly rather than at a guessed boot wait. A
    duck-typed entry (no ModelEntry parse validation) whose runtime dict is
    missing ``name``/``health_url`` likewise raises a clear ``ValueError``
    NAMING the missing key, never a bare ``KeyError`` from the constructor.
    """
    runtime = getattr(entry, "runtime", None)
    if runtime is None:
        return None
    if not isinstance(runtime, dict):
        raise ValueError(
            f"registry entry {getattr(entry, 'id', '?')!r}: runtime must be a "
            f"dict to build a lifecycle, got {type(runtime).__name__} — "
            f"refusing to guess (fail-closed)")
    kind = runtime.get("kind")
    if kind == "process":
        return None
    if kind != "container":
        raise ValueError(
            f"registry entry {getattr(entry, 'id', '?')!r}: runtime.kind must "
            f"be 'container' or 'process' to build a lifecycle, got {kind!r} "
            f"(fail-closed)")
    for key in ("name", "health_url"):
        if not runtime.get(key):
            raise ValueError(
                f"container runtime for {getattr(entry, 'id', '?')!r} "
                f"declares no {key} — refusing to guess a container name or "
                f"health URL (fail-closed)")
    engine_id = seams.pop("engine_id", None)
    if not engine_id:
        engines = getattr(entry, "engines", ()) or ()
        engine_id = engines[0] if engines else runtime.get("name", "")
    budget = runtime.get("boot_budget_s")
    if budget is None:
        raise ValueError(
            f"container runtime for {getattr(entry, 'id', '?')!r} declares no "
            f"boot_budget_s — refusing to invent a cold-boot budget "
            f"(fail-closed)")
    return DockerContainerLifecycle(
        engine_id, container=runtime["name"],
        health_url=runtime["health_url"], boot_budget_s=float(budget),
        gpu_count=int(runtime.get("gpu_count", 1)), **seams)


class MayaLifecycle(EngineLifecycle):
    """Maya's engine-native model-management API (roadmap §5b: Maya/Strata
    speak session/pull APIs). What honestly exists: ``GET /api/ps`` reports
    the loaded session, ``POST /api/show`` confirms a model is known, and a
    request naming a model auto-swaps to it — loading is engine-native, so
    ``ensure_loaded`` is a readiness check over those endpoints, never a
    weights pin. ``/api/pull`` downloads weights (pull management, not
    load).

    Maya exposes NO unload primitive. ``release`` manages only what exists:
    it sends nothing to the engine and answers with an honest ledger-side
    acknowledgment labeled ``no-unload-primitive`` — an unload is never
    faked. ``POST /api/delete`` removes a model's files (destructive) and is
    never used as a release.

    The HTTP function is injectable; the default is stdlib urllib."""

    engine_id = "maya"
    WARM_PATH = ("warm-swap via engine-native auto-swap (a request naming "
                 "the model loads it; no explicit load endpoint exists)")
    LOAD_CONTROL = ("engine-native auto-swap — Maya loads a model when a "
                    "request names it; no explicit load endpoint exists")
    NO_UNLOAD = ("Maya exposes no unload primitive — release is a "
                 "ledger-side acknowledgment only; nothing is sent to the "
                 "engine and an unload is never faked (POST /api/delete "
                 "removes a model's files and is destructive — never used "
                 "as a release)")

    def __init__(self, base_url="http://127.0.0.1:8080", http=None):
        self._root = (base_url or "").rstrip("/")
        self._http = http or _http_json

    def ensure_loaded(self, model_id, **_ignored) -> dict:
        row = {"ok": False, "engine": self.engine_id, "action": "load",
               "model_id": model_id, "load_control": self.LOAD_CONTROL}
        try:
            data = self._http("GET", f"{self._root}/api/ps")
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {**row, "reason": f"maya /api/ps unreachable: "
                                     f"{type(exc).__name__}: {exc}"}
        models = data.get("models") if isinstance(data, dict) else None
        loaded = [m.get("name", "") for m in (models or [])
                  if isinstance(m, dict)]
        if model_id in loaded:
            return {"ok": True, "engine": self.engine_id, "action": "load",
                    "model_id": model_id, "state": "resident",
                    "load_control": self.LOAD_CONTROL,
                    "reason": f"{model_id} is already loaded in Maya's "
                              f"session (GET /api/ps) — nothing to load"}
        try:
            shown = self._http("POST", f"{self._root}/api/show",
                               {"name": model_id})
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {**row, "reason": f"maya /api/show failed for {model_id}: "
                                     f"{type(exc).__name__}: {exc}"}
        if not isinstance(shown, dict) or not shown:
            return {**row, "reason": f"Maya does not know {model_id} "
                                     f"(POST /api/show returned nothing) — "
                                     f"not ready (fail-soft)"}
        return {"ok": True, "engine": self.engine_id, "action": "load",
                "model_id": model_id, "state": "ready-on-demand",
                "load_control": self.LOAD_CONTROL,
                "reason": f"{model_id} is known to Maya and loads "
                          f"engine-natively on the first request naming it "
                          f"(auto-swap); no explicit load endpoint exists — "
                          f"a readiness check, not a weights pin"}

    def release(self, model_id) -> dict:
        return {"ok": True, "engine": self.engine_id, "action": "release",
                "model_id": model_id,
                "unload_control": "no-unload-primitive",
                "reason": self.NO_UNLOAD}

    def probe(self) -> dict:
        try:
            data = self._http("GET", f"{self._root}/api/ps")
        except Exception as exc:  # noqa: BLE001 — probes fail soft
            return {"ok": False, "engine": self.engine_id, "state": "unknown",
                    "resident": [],
                    "reason": f"maya /api/ps unreachable: "
                              f"{type(exc).__name__}: {exc}"}
        models = data.get("models") if isinstance(data, dict) else None
        resident = [m.get("name", "") for m in (models or [])
                    if isinstance(m, dict)]
        return {"ok": True, "engine": self.engine_id, "state": "observed",
                "resident": [n for n in resident if n],
                "reason": self.NO_UNLOAD}


class ExoLifecycle(EngineLifecycle):
    """Exo's instance API (github.com/exo-explore/exo, docs/api.md §2).

    Exo places model instances across the devices it clusters. What
    honestly exists over HTTP: ``POST /place_instance`` asks the cluster to
    place and load a model (server-side placement; the answer is a command
    acknowledgement and creation is ASYNCHRONOUS — the model downloads and
    loads engine-natively, and clients wait for readiness through
    ``GET /instance/await?model_id=…``), and ``DELETE /instance/{id}`` stops
    a model by its instance id or its deployment id — a REAL unload
    primitive (it stops a deployment even while it waits to be placed
    again). ``release`` therefore drives that delete: an unload is never
    faked, and nothing destructive (custom-model delete) is ever used as a
    release.

    ``ensure_loaded`` is launch-or-observe: an instance already running for
    the model is reported ``resident`` and never re-launched; otherwise the
    placement is requested and reported ``placement-requested`` — the model
    is never claimed loaded before the engine says so. ``probe`` reads
    ``GET /state`` (instances keyed by id, each carrying
    ``shard_assignments.model_id``).

    The HTTP function is injectable; the default is stdlib urllib."""

    engine_id = "exo"
    WARM_PATH = ("engine-native instance placement (POST /place_instance; "
                 "creation is asynchronous)")
    LOAD_CONTROL = ("engine-native instance placement — POST /place_instance "
                    "asks the exo cluster to place and load the model; "
                    "creation is asynchronous (wait via GET /instance/await "
                    "before inference)")
    UNLOAD_CONTROL = ("instance delete (DELETE /instance/{id}) — a real "
                      "unload primitive: the cluster stops the deployment "
                      "and frees its memory")

    def __init__(self, base_url="http://127.0.0.1:52415", http=None):
        self._root = (base_url or "").rstrip("/")
        self._http = http or _http_json
        # model_id -> deployment_id from the launch acknowledgement: the id
        # DELETE /instance/{id} stops by, even before placement finishes.
        self._deployments: dict[str, str] = {}

    # Readiness wait (A16): after placement the adapter CONSUMES the
    # engine's readiness route (GET /instance/await) with this bounded wait,
    # so ensure_loaded can report "ready" when the engine actually confirms
    # it — and honestly stays "placement-requested" when the wait times out
    # or does not confirm. await_s=0 skips the wait entirely.
    AWAIT_S = 120.0

    @staticmethod
    def _await_confirms(resp) -> bool:
        """True only when the await answer POSITIVELY indicates readiness —
        an instance record or an explicit ready flag/status. An empty or
        ambiguous body confirms nothing: the state vocabulary promises
        ``ready`` only when the engine says so."""
        if not isinstance(resp, dict) or not resp:
            return False
        if resp.get("ready") is True:
            return True
        if resp.get("ready") is False:
            return False
        status = str(resp.get("status") or "").strip().lower()
        if status in ("pending", "queued", "loading"):
            return False
        if status in ("ready", "done", "completed", "running", "ok"):
            return True
        return bool(resp.get("instance_id") or resp.get("deployment_id")
                    or isinstance(resp.get("shard_assignments"), dict)
                    or resp.get("model_id"))

    def _await_ready(self, model_id: str, await_s: float) -> tuple[bool, str]:
        """Consume GET /instance/await (A16). Returns (confirmed, reason)."""
        if await_s <= 0:
            return False, ("readiness wait skipped (await_s=0) — the state "
                           "stays placement-requested")
        url = (f"{self._root}/instance/await?model_id="
               + urllib.parse.quote(str(model_id), safe=""))
        try:
            resp = self._http("GET", url, None, await_s)
        except Exception as exc:  # noqa: BLE001 — not confirmed is honest
            return False, (f"GET /instance/await did not confirm readiness "
                           f"within {await_s:g} s: {type(exc).__name__}: "
                           f"{exc}")
        if self._await_confirms(resp):
            return True, (f"GET /instance/await confirmed readiness for "
                          f"{model_id}")
        return False, (f"GET /instance/await answered without confirming "
                       f"readiness for {model_id} — the state stays "
                       f"placement-requested")

    @staticmethod
    def _instances(data) -> list[dict]:
        raw = data.get("instances") or {} if isinstance(data, dict) else {}
        items = raw.values() if isinstance(raw, dict) else (raw or [])
        return [i for i in items if isinstance(i, dict)]

    @staticmethod
    def _instance_model(inst: dict) -> str:
        shard = inst.get("shard_assignments")
        if isinstance(shard, dict) and shard.get("model_id"):
            return str(shard["model_id"])
        return str(inst.get("model_id") or "")

    @staticmethod
    def _instance_stop_id(inst: dict) -> str:
        # DELETE /instance/{id} accepts the instance id or the deployment id.
        return str(inst.get("instance_id") or inst.get("deployment_id")
                   or "")

    def ensure_loaded(self, model_id, await_s=None, **_ignored) -> dict:
        row = {"ok": False, "engine": self.engine_id, "action": "load",
               "model_id": model_id, "load_control": self.LOAD_CONTROL}
        try:
            state = self._http("GET", f"{self._root}/state")
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {**row, "reason": f"exo /state unreachable: "
                                     f"{type(exc).__name__}: {exc}"}
        for inst in self._instances(state):
            if self._instance_model(inst) == model_id:
                stop_id = (self._instance_stop_id(inst)
                           or self._deployments.get(model_id, ""))
                return {"ok": True, "engine": self.engine_id,
                        "action": "load", "model_id": model_id,
                        "state": "resident", "instance_id": stop_id,
                        "load_control": self.LOAD_CONTROL,
                        "reason": f"{model_id} already runs as an exo "
                                  f"instance (GET /state) — nothing to "
                                  f"launch"}
        try:
            ack = self._http("POST", f"{self._root}/place_instance",
                             {"model_id": model_id})
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {**row, "reason": f"exo POST /place_instance failed for "
                                     f"{model_id}: {type(exc).__name__}: "
                                     f"{exc}"}
        if not isinstance(ack, dict) or not ack:
            return {**row, "reason": f"exo POST /place_instance returned no "
                                     f"command acknowledgement for "
                                     f"{model_id} — not launched (fail-soft)"}
        deployment_id = str(ack.get("deployment_id") or "")
        if deployment_id:
            self._deployments[model_id] = deployment_id
        # A16: consume the readiness wait — the state is "ready" only when
        # the engine's await route positively confirms it.
        wait_s = self.AWAIT_S if await_s is None else float(await_s)
        confirmed, await_reason = self._await_ready(model_id, wait_s)
        instance_id = self._deployments.get(model_id, "")
        if confirmed:
            return {"ok": True, "engine": self.engine_id, "action": "load",
                    "model_id": model_id, "state": "ready",
                    "instance_id": instance_id,
                    "deployment_id": deployment_id,
                    "load_control": self.LOAD_CONTROL,
                    "reason": f"placement requested for {model_id} (command "
                              f"{ack.get('command_id') or 'accepted'}) — "
                              f"{await_reason}; the model is loaded"}
        return {"ok": True, "engine": self.engine_id, "action": "load",
                "model_id": model_id, "state": "placement-requested",
                "deployment_id": deployment_id,
                "load_control": self.LOAD_CONTROL,
                "reason": f"placement requested for {model_id} (command "
                          f"{ack.get('command_id') or 'accepted'}) — exo "
                          f"downloads and loads the model engine-natively "
                          f"and asynchronously; it is NOT loaded yet: "
                          f"{await_reason}"}

    def release(self, model_id) -> dict:
        base = {"engine": self.engine_id, "action": "release",
                "model_id": model_id, "unload_control": self.UNLOAD_CONTROL}
        stop_id = self._deployments.pop(model_id, "")
        if not stop_id:
            try:
                state = self._http("GET", f"{self._root}/state")
            except Exception as exc:  # noqa: BLE001 — honest row
                return {**base, "ok": False,
                        "reason": f"exo /state unreachable — cannot confirm "
                                  f"what runs {model_id}, refusing to guess "
                                  f"(fail-closed): {type(exc).__name__}: "
                                  f"{exc}"}
            for inst in self._instances(state):
                if self._instance_model(inst) == model_id:
                    stop_id = self._instance_stop_id(inst)
                    break
        if not stop_id:
            return {**base, "ok": True,
                    "reason": f"no exo instance runs {model_id} (GET "
                              f"/state) — nothing to stop"}
        try:
            self._http("DELETE", f"{self._root}/instance/{stop_id}")
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {**base, "ok": False, "instance_id": stop_id,
                    "reason": f"exo DELETE /instance/{stop_id} failed: "
                              f"{type(exc).__name__}: {exc}"}
        return {**base, "ok": True, "instance_id": stop_id,
                "reason": f"stopped exo instance {stop_id} (DELETE "
                          f"/instance/{stop_id}) — a real unload primitive; "
                          f"the cluster stops the model and frees its "
                          f"memory"}

    def probe(self) -> dict:
        try:
            state = self._http("GET", f"{self._root}/state")
        except Exception as exc:  # noqa: BLE001 — probes fail soft
            return {"ok": False, "engine": self.engine_id, "state": "unknown",
                    "resident": [],
                    "reason": f"exo /state unreachable: "
                              f"{type(exc).__name__}: {exc}"}
        if not isinstance(state, dict):
            return {"ok": False, "engine": self.engine_id, "state": "unknown",
                    "resident": [],
                    "reason": "exo /state returned no cluster state — "
                              "residency unknown"}
        resident = [self._instance_model(i) for i in self._instances(state)]
        return {"ok": True, "engine": self.engine_id, "state": "observed",
                "resident": [m for m in resident if m],
                "reason": "instances observed via GET /state; releases "
                          "drive " + self.UNLOAD_CONTROL}


# --- eviction planner -------------------------------------------------------

class EvictionPlanner:
    """The serving policy proven under production load (roadmap §3) as
    plan data.

    - Fit gates on NOMINAL booked ceilings. Live free VRAM only orders the
      load steps (which to load first); it never gates them — booked-vs-live
      comparisons cause false evictions.
    - A lane wake (a load claiming its lane) stops resident side models
      first — otherwise the side model OOMs the waking lane.
    - Recency window (``recency_window_s``, ~600 s): models used within the
      window stay warm ahead of pure LRU near the top of the stack; eviction
      order within a tier is least-recently-used first, and a warm model is
      freed only when the nominal fit gate still fails without it.
    - Image-generation entry evicts GPU residents every time — the whole
      card, deterministic full eviction, no clever sharing.
    - LLM-waiter priority (roadmap §3): while a chat request waits on a lane
      wake it OWNS the wake, so a generative plan is refused as pure data
      (an honest reason) when the caller passes ``waiter_count`` > 0 — the
      waiter owns the wake and two swaps must not fight over the card.
    - Loads fire sequentially: the plan is an ordered, serialized list;
      concurrency control is the caller's (concurrent engine inits OOM).

    Fit gaps fail closed: no booked footprint for a requested model, no
    declared budget for the target, or a request that exceeds the budget
    even on a clear card all return a refused plan with a stated reason.
    A resident with no booked footprint cannot be accounted in the nominal
    gate, so it is released first, honestly labeled.
    """

    def __init__(self, ledger, clock=None,
                 recency_window_s=RECENCY_WINDOW_S):
        self._ledger = ledger
        self._clock = clock or ledger.clock
        self._window = float(recency_window_s)

    def plan_load(self, target_id, request) -> LoadPlan:
        """Plan making one model resident."""
        return self.plan_load_many(target_id, [request])

    def plan_load_many(self, target_id, requests, *, waiter_count=0) -> LoadPlan:
        """Plan making a set of models resident (e.g. a scope restore).

        Pure planning: nothing is loaded, released, or recorded here.

        ``waiter_count`` is the current LLM-waiter count (default 0 — no
        waiter). When it is non-zero AND the request set contains a
        generative (image-generation) entry, the plan is REFUSED as pure
        data with an honest reason: the rule of roadmap §3 expressed in the
        planner's own style — a chat waiter owns the lane wake, so gen
        eviction/wake planning is suppressed while it waits. Callers with no
        waiter context (the normal claim/release path) pass nothing and get
        the policy unchanged; a chat plan is never suppressed."""
        reqs = list(requests)
        if not reqs:
            return LoadPlan(target_id, True, "nothing to load", (), ())
        want_ids = {r.model_id for r in reqs}
        image_entry = any(is_image_generation(r.modality, r.lane)
                          for r in reqs)
        now = self._clock()

        # -- LLM-waiter priority: a waiting chat request owns the lane wake,
        # so generative eviction/wake planning is suppressed while it waits.
        if waiter_count > 0 and image_entry:
            return self._refuse(
                target_id,
                f"refused: a chat request owns the lane wake — {waiter_count} "
                f"LLM waiter(s) pending (roadmap §3 LLM-waiter priority: the "
                f"waiter owns the wake; generative eviction/wake planning is "
                f"suppressed while any chat request waits)")


        # -- fail-closed gate inputs: nominal bookings and a declared budget.
        missing = [r.model_id for r in reqs
                   if self._ledger.booked_vram_gb(r.model_id) is None]
        if missing:
            return self._refuse(target_id,
                                f"no booked footprint for "
                                f"{', '.join(sorted(set(missing)))} — fit is "
                                f"gated on nominal booked ceilings and none "
                                f"is booked (fail-closed); book a registry "
                                f"footprint or a measured peak first")
        budget = self._ledger.budget_gb(target_id)
        if budget is None:
            return self._refuse(target_id,
                                f"no nominal budget declared for compute "
                                f"target {target_id!r} — fit cannot be gated "
                                f"on booked ceilings (fail-closed)")
        want_gb = sum(self._ledger.booked_vram_gb(r.model_id) for r in reqs)
        if want_gb > budget:
            return self._refuse(
                target_id,
                f"booked footprint {want_gb:g} GB exceeds the nominal budget "
                f"{budget:g} GB even on a clear card — cannot fit "
                f"(fail-closed)")

        residents = self._ledger.residents(target_id)
        staying = [rec for rec in residents if rec.model_id in want_ids]
        candidates = [rec for rec in residents if rec.model_id not in want_ids]
        pending = [r for r in reqs
                   if not self._ledger.is_resident(target_id, r.model_id)]
        need_gb = sum(self._ledger.booked_vram_gb(r.model_id)
                      for r in pending)

        # -- choose victims per policy.
        if image_entry:
            # Image generation wants the whole card: every GPU resident goes,
            # every time — least-recently-used first.
            victims = sorted(candidates, key=lambda rec: rec.last_used)
        elif not pending:
            return LoadPlan(target_id, True,
                            "already resident — nothing to release or load",
                            (), ())
        else:
            victims = self._select_victims(
                candidates, staying, {r.lane for r in reqs}, need_gb, budget,
                now)

        # -- explicit release steps, each with an honest policy reason.
        release_steps = tuple(
            PlanStep(action="release", target_id=target_id,
                     model_id=rec.model_id, engine=rec.engine,
                     reason=self._release_reason(rec, want_ids, reqs, now,
                                                 image_entry),
                     lane=rec.lane)
            for rec in victims)

        # -- ordered, serialized load steps: live free VRAM orders only.
        load_steps = tuple(
            PlanStep(action="load", target_id=target_id, model_id=r.model_id,
                     engine=r.engine,
                     reason=(f"load {r.model_id} via {r.engine} — loads fire "
                             f"sequentially (concurrent engine inits OOM)"),
                     lane=r.lane, weights_path=r.weights_path,
                     estimated_s=(r.estimated_s if r.estimated_s is not None
                                  else self._ledger.estimated_load_s(
                                      r.model_id)))
            for r in self._order_loads(target_id, pending))

        if image_entry:
            reason = (f"image-generation entry claims the whole GPU — "
                      f"{len(release_steps)} resident(s) released every time, "
                      f"then {len(load_steps)} load(s) one at a time")
        else:
            reason = (f"nominal booked fit passes ({need_gb:g} GB into "
                      f"{budget:g} GB; live free VRAM orders only, never "
                      f"gates) — {len(release_steps)} release(s), "
                      f"{len(load_steps)} load(s), one load at a time")
        return LoadPlan(target_id, True, reason, release_steps, load_steps)

    # -- internals --

    @staticmethod
    def _refuse(target_id, reason) -> LoadPlan:
        return LoadPlan(target_id, False, reason, (), ())

    def _select_victims(self, candidates, staying, wake_lanes, need_gb,
                        budget, now):
        """Eviction order: unbooked residents first (their VRAM cannot be
        accounted in the nominal gate), then side models of the waking lanes
        (a lane wake stops resident side models first), then least-recently
        used outside the recency window, then — kept warm until last —
        models inside the window."""
        def booked(rec):
            return self._ledger.booked_vram_gb(rec.model_id)

        def tier(rec):
            age = now - rec.last_used
            if booked(rec) is None:
                return 0
            if rec.lane not in wake_lanes:
                return 1
            if age > self._window:
                return 2
            return 3

        ordered = sorted(candidates,
                         key=lambda rec: (tier(rec), rec.last_used))
        # Unbooked residents always go: their VRAM cannot be accounted in
        # the nominal gate, so no fit decision may leave them resident.
        victims = [rec for rec in ordered if tier(rec) == 0]
        remaining = sum(booked(rec) or 0.0 for rec in staying + candidates)
        for rec in ordered:
            if rec in victims:
                continue
            if remaining + need_gb <= budget:
                break
            victims.append(rec)
            remaining -= booked(rec) or 0.0
        return victims

    def _release_reason(self, rec, want_ids, reqs, now, image_entry) -> str:
        if image_entry:
            return (f"image generation claims the whole GPU — deterministic "
                    f"full eviction of every resident, every time "
                    f"({rec.model_id}, lane {rec.lane!r})")
        if self._ledger.booked_vram_gb(rec.model_id) is None:
            return (f"{rec.model_id} has no booked footprint — its VRAM "
                    f"cannot be accounted in the nominal fit gate, so it is "
                    f"released first")
        wake_lanes = sorted({r.lane for r in reqs})
        if rec.lane not in self._wake_lanes(reqs):
            lanes = ", ".join(repr(l) for l in wake_lanes)
            return (f"lane wake: {', '.join(sorted(want_ids))} claims "
                    f"lane(s) {lanes} — resident side model {rec.model_id} "
                    f"(lane {rec.lane!r}) is stopped first so it cannot OOM "
                    f"the waking lane")
        age = now - rec.last_used
        if age > self._window:
            return (f"least-recently used (last used {age:.0f}s ago, outside "
                    f"the {self._window:.0f}s recency window) — freed so the "
                    f"nominal booked fit for "
                    f"{', '.join(sorted(want_ids))} passes")
        return (f"inside the {self._window:.0f}s recency window — kept warm "
                f"until last; freed only because the nominal booked fit for "
                f"{', '.join(sorted(want_ids))} still fails without it")

    @staticmethod
    def _wake_lanes(reqs) -> set:
        return {r.lane for r in reqs}

    def _order_loads(self, target_id, pending):
        """Order load steps by live free VRAM — which to load first, never
        whether they load. Models that fit the live free snapshot go first
        (smallest first); the rest follow in request order."""
        free = self._ledger.free_vram_gb(target_id)
        if free is None:
            return list(pending)
        fits = [r for r in pending
                if (self._ledger.booked_vram_gb(r.model_id) or 0.0) <= free]
        fits.sort(key=lambda r: self._ledger.booked_vram_gb(r.model_id))
        fits_ids = {id(r) for r in fits}
        rest = [r for r in pending if id(r) not in fits_ids]
        return fits + rest


# --- orchestrator ------------------------------------------------------------

class Orchestrator:
    """Ties the ledger, engine adapters, and eviction planner together.

    Workflow scopes: ``claim`` takes the lane for a modality and remembers
    what was resident on the target; ``release`` (default) steps the scope's
    models down and restores exactly that set, while
    ``release(scope_id, restore=False)`` releases the scope WITHOUT touching
    residency — the claim's target stays resident (the lane the waiter just
    woke stays up until a gen entry evicts it). ``prewarm`` loads on intent,
    before the user's click lands. Progress states (``queued`` /
    ``loading (~Ns)`` / ``ready``) are pure data rows recorded per scope or
    pre-warm key — no network I/O happens anywhere outside the injected
    adapter calls, and every engine call runs strictly one at a time
    (concurrency control is the caller's). A failed engine call mid-plan
    stops the run and is reported: fail closed, never a silent stall.
    """

    def __init__(self, ledger, planner, adapters, clock=None):
        self._ledger = ledger
        self._planner = planner
        self._adapters = dict(adapters)
        self._clock = clock or ledger.clock
        self._lock = threading.Lock()
        self._scopes: dict[str, dict] = {}
        self._history: dict[str, list] = {}
        self._current: dict[str, dict] = {}
        # The scope table has its OWN lock, separate from ``self._lock``: the
        # ``gen_busy`` accessor is polled by the facade while a gen job runs,
        # and reading the scope table must never block behind a slow engine
        # load (the same reason the waiter seam has its own lock). No code
        # path ever holds ``self._scopes_lock`` and ``self._lock`` at once.
        self._scopes_lock = threading.Lock()
        # Per-target load specs of successfully loaded models (weights path,
        # load estimate) — a scope release restores claim-time residents with
        # the fidelity their original load had, even when the evicted model
        # belongs to another still-open scope whose request is not visible
        # from the releasing scope.
        self._load_specs: dict[tuple[str, str], dict] = {}
        # LLM-waiter priority (roadmap §3): a chat request blocked on a lane
        # wake OWNS that wake, so new generative jobs are refused until it
        # gets in. Waiters are tracked as a set of opaque tokens, so a token's
        # exit is idempotent (removing a token that is not held is a no-op)
        # and concurrent waiters never double-count. The count has its own
        # lock, separate from ``self._lock``: the facade serves concurrent
        # requests (ThreadingHTTPServer), so an admission check or an exit
        # may arrive from another thread while a claim holds ``self._lock``
        # for a long engine load — sharing that lock would make an admission
        # check block behind a slow engine call.
        self._waiter_lock = threading.Lock()
        self._llm_waiters: set[str] = set()
        self._waiter_seq = 0

    # -- LLM-waiter priority (roadmap §3) --

    @property
    def llm_waiters(self) -> int:
        """How many chat requests currently wait on a lane wake.

        This is the model-forge proxy for the predecessor broker's
        ``Lifecycle.llm_waiters`` counter (its ``prepare_llm`` increments on
        entry and its ``finally`` decrements on exit; ``prepare_gen`` refuses
        while it is non-zero). A waiter is a
        chat request that has entered the lane-wake path and not yet left it.
        The count is read under the waiter lock, so a concurrent enter/exit
        never shows a torn or half-applied value."""
        with self._waiter_lock:
            return len(self._llm_waiters)

    def waiter_enter(self) -> dict:
        """Register one LLM (chat) waiter that is blocked on a lane wake.

        Mirrors ``prepare_llm``'s entry: it bumps the waiter count and hands
        back a token the caller holds for the duration of the wait. The token
        is unique (a monotonic sequence under the waiter lock), so concurrent
        waiters are all tracked independently and each exit removes exactly
        its own waiter. Registration cannot fail — the set is unbounded and
        nothing external is touched — so ``ok`` is always ``True`` here; the
        ``ok``/``reason`` shape matches the module's dict-row convention.

        Returns ``{'ok': True, 'token': <str>, 'reason': <str>}``."""
        with self._waiter_lock:
            self._waiter_seq += 1
            token = f"llm-waiter-{self._waiter_seq}"
            self._llm_waiters.add(token)
            pending = len(self._llm_waiters)
        return {
            "ok": True, "token": token,
            "reason": f"registered LLM waiter {token} — {pending} waiter(s) "
                      f"now own the lane wake; new generative jobs are "
                      f"refused while any chat request waits (roadmap §3 "
                      f"LLM-waiter priority: the waiter owns the wake)",
        }

    def waiter_exit(self, token: str) -> None:
        """Deregister one LLM waiter. Idempotent and never raises.

        Mirrors ``prepare_llm``'s ``finally: self.llm_waiters -= 1``: the
        waiter count drops by one when a token leaves. Because waiters are a
        set, removing a token that is not held (an unknown token, or a
        double exit of one already released) is a no-op — the count can never
        go negative and interleaved enter/exit pairs never double-count. A
        ``None``/empty token is likewise a no-op, never a raise: a
        late/duplicated exit must never sink the caller."""
        with self._waiter_lock:
            self._llm_waiters.discard(token)
        return None

    def admit_gen(self, model_id: str = "") -> dict:
        """Admit a generative job, or refuse it while a chat waiter owns the
        lane wake (roadmap §3 LLM-waiter priority).

        This is the REFUSAL side of the rule, mirroring ``prepare_gen``'s
        ``if self.llm_waiters > 0: raise RuntimeError(...)`` — except the
        refusal is returned as an honest row, never raised. While any chat
        request waits on a lane wake, a generative job is refused: it must
        NOT start or queue a lane wake of its own, because the waiter owns
        the wake and two swaps fighting over the card is exactly what the
        rule prevents.

        Admission is a gate, NOT a claim. When it returns ``ok=True`` it has
        done nothing but answer — the existing ``claim``/``release`` scope
        performs the actual lane claim (lane wake / eviction), unchanged.
        Likewise a chat ``claim`` never consults this seam: a chat request is
        the waiter, so it always gets its wake.

        Returns ``{'ok': bool, 'reason': str}`` with an honest reason naming
        the policy."""
        pending = self.llm_waiters
        if pending > 0:
            for_model = f" for {model_id}" if str(model_id or "").strip() else ""
            return {
                "ok": False,
                "reason": f"refused: a chat request owns the lane wake — "
                          f"{pending} LLM waiter(s) pending{for_model} "
                          f"(roadmap §3 LLM-waiter priority: the waiter owns "
                          f"the wake; no generative job is started or queued "
                          f"while a chat request waits on its lane wake)",
            }
        lead = f"admitted for {model_id}: " if str(model_id or "").strip() \
               else "admitted: "
        return {
            "ok": True,
            "reason": f"{lead}no chat waiter owns the lane wake — admission "
                      f"only; the claim/release scope performs the actual "
                      f"lane claim (admission alone never claims)",
        }

    def adopt_live(self, target_id, model_id, engine, lane="chat") -> dict:
        """Record a live-but-unrecorded runtime as resident (provenance
        ``"external"``) so later plans evict and restore it.

        WHY: the ledger is process-local. A lane started OUTSIDE this
        process (compose, a human at the shell) is invisible to it, so a
        gen claim plans NO release step for that lane — it is never stopped
        — and snapshots an empty restore set, so nothing brings it back
        afterwards. That is the observed failure mode this seam closes: the
        LLM lane had to be stopped and restored BY HAND around every gen
        run. Adopting records the observed runtime as resident BEFORE the
        claim computes its plan, so the claim's eviction stops the lane and
        the release's restore phase starts it again — the choreography the
        planner already implements, fed with an honest picture of reality.

        Honest provenance: the resident is recorded ``"external"`` — this
        process did not start it (``ResidentRecord.provenance``). Idempotent:
        an already-recorded residency is left exactly as it is (an existing
        record is never restamped or downgraded). Fail-soft: a missing model
        id/engine, an unknown engine, a raising probe, a probe that cannot
        confirm the runtime is up, or a probe naming a down state adopts
        nothing (``adopted: False``) — the ledger keeps its honest ignorance
        rather than inventing a resident. Never raises.

        Returns ``{'ok': True, 'adopted': bool, 'reason': str}``. Adoption
        is a heal, not a claim: it never starts, stops, or loads anything."""
        model_id = str(model_id or "").strip()
        engine = str(engine or "").strip()
        if not model_id or not engine:
            return {"ok": True, "adopted": False,
                    "reason": "nothing to adopt — a model id and an engine "
                              "name are both required (fail-soft)"}
        if self._ledger.is_resident(target_id, model_id):
            return {"ok": True, "adopted": False,
                    "reason": f"{model_id} is already recorded resident on "
                              f"{target_id!r} — residency left untouched"}
        adapter = self._adapters.get(engine)
        if adapter is None:
            return {"ok": True, "adopted": False,
                    "reason": f"no engine adapter registered for {engine!r} "
                              f"— runtime state cannot be observed "
                              f"(fail-soft)"}
        try:
            probe = adapter.probe()
        except Exception as e:  # noqa: BLE001 — probes fail soft
            return {"ok": True, "adopted": False,
                    "reason": f"probe for {engine!r} raised "
                              f"{type(e).__name__}: {e} — residency unknown, "
                              f"not adopted (fail-soft)"}
        if not isinstance(probe, dict) or not probe.get("ok"):
            return {"ok": True, "adopted": False,
                    "reason": f"probe for {engine!r} could not confirm the "
                              f"runtime state — not adopted (fail-soft)"}
        state = str(probe.get("state") or "").strip().lower()
        # Up = the READY_STATES a swap confirms on, plus ``booting`` (a
        # container that started but never answered healthy still holds
        # VRAM — exactly what a gen plan must evict).
        if state not in ("observed", "running", "ready", "booting"):
            return {"ok": True, "adopted": False,
                    "reason": f"runtime state is {state or 'unreadable'} — "
                              f"not up, nothing to adopt"}
        self._ledger.record_resident(target_id, model_id, engine=engine,
                                     lane=lane, provenance="external")
        return {"ok": True, "adopted": True,
                "reason": f"adopted the live runtime for {model_id} ({engine}"
                          f", lane {lane!r}) as resident on {target_id!r} "
                          f"with provenance 'external' — a later claim's "
                          f"plan will stop it and its release will restore "
                          f"it"}

    # -- gen-busy read (the facade's wait for in-flight gen jobs) --

    @property
    def gen_busy(self) -> bool:
        """True while a generative (image/video) workflow scope is open.

        Cheap and honest: the scope table is read under its own lock (never
        the plan lock), so a facade can poll this while a gen job runs without
        blocking behind a slow engine load. A generative job holds the GPU
        card, so the facade's chat path polls this and waits, bounded by its
        own budget, for in-flight gen jobs to drain before it wakes the LLM
        lane — the waiter owns the wake (roadmap §3). A plain read; it never
        claims, evicts, or records anything.
        """
        with self._scopes_lock:
            scopes = list(self._scopes.values())
        return any(is_gen_lane(getattr(s.get("request"), "modality", ""),
                               getattr(s.get("request"), "lane", ""))
                   for s in scopes)

    # -- workflow scopes --

    def claim(self, scope_id, target_id, request) -> dict:
        """Claim the lane for a modality as one workflow scope.

        Snapshots what is resident BEFORE any eviction — ``release`` restores
        exactly that set (including the claimed model if it was already
        resident at claim time)."""
        if not str(scope_id or "").strip():
            return self._fail(f"{target_id}:{request.model_id}", None,
                              "scope id must be a non-empty string",
                              target_id)
        with self._scopes_lock:
            already = scope_id in self._scopes
        if already:
            return self._fail(scope_id, scope_id,
                              f"scope {scope_id!r} is already claimed — "
                              f"release it first (fail-closed)",
                              target_id, request)
        restore = tuple(self._ledger.residents(target_id))
        # Residency heal (predecessor-broker hard rule: gate on CONTAINER state,
        # not the stale ledger/cache). If the ledger calls the REQUESTED model
        # resident but its runtime is actually down (e.g. a container stopped
        # outside this broker), the planner would emit NO load step, report
        # ready, and the upstream forward would 502. Consult the lifecycle
        # probe for the requested model and drop the stale residency so the
        # planner re-emits the load (the adapter's ensure_loaded is
        # idempotent: starting a running container is a no-op).
        self._heal_residency(target_id, {request.model_id})
        result = self._run_plan(
            self._planner.plan_load_many(target_id, [request]),
            key=scope_id, scope_id=scope_id, ready_model=request.model_id)
        if result["ok"]:
            with self._scopes_lock:
                self._scopes[scope_id] = {
                    "target_id": target_id,
                    "request": request,
                    "restore": restore,
                    # what this scope's plan actually loaded — the only models
                    # its release may step down
                    "brought_in": frozenset(
                        step.model_id for step in result["steps"]
                        if step.action == "load"),
                }
        return result

    def release(self, scope_id, *, restore=True, approved_restore=None,
                gated_engines=()) -> dict:
        """Release a claimed scope.

        With ``restore=True`` (the default) the scope's own models are stepped
        down and then exactly what was resident at claim time is restored
        (loads fire sequentially).

        With ``restore=False`` the scope is released WITHOUT touching
        residency: the scope's models are NOT stepped down and prior residents
        are NOT restored — the claim's target stays resident. This is the
        chat path's release: the waiter owns the wake, so after forwarding the
        request the LLM lane stays up (resident) until a gen entry evicts it,
        instead of being stopped and immediately re-started.

        ``approved_restore`` (A20) is the exact model-id set the
        human-approval gate inspected for this release, recorded by the
        pre-call hook. When provided, the handler RE-VERIFIES its recomputed
        restore plan against that set and fails closed — the restore is
        refused, nothing is loaded — if the plan would (re)start a gated
        engine server (``gated_engines``: the process-based engines whose
        load spawns a server) for a model the gate never saw. This consumes
        the gate's decision instead of re-deriving it, closing the pre-call
        TOCTOU window where a concurrent claim/eviction between the gate's
        inspection and this handler could grow the restore plan (A20).
        ``approved_restore=None`` (programmatic use, no gate in the path)
        keeps the pre-A20 behavior.
        """
        with self._scopes_lock:
            scope = self._scopes.get(scope_id)
        if scope is None:
            return self._fail(scope_id, scope_id,
                              f"no claimed scope {scope_id!r} — nothing to "
                              f"release (fail-closed)", "", None)
        target_id = scope["target_id"]

        # -- restore=False: drop the scope, leave residency exactly as it is.
        # The claim's target (and any other resident) stays up — the lane the
        # waiter just woke is not stopped; it stays resident until a gen entry
        # evicts it (the facade's chat release behavior).
        if not restore:
            with self._scopes_lock:
                self._scopes.pop(scope_id, None)
            return {
                "ok": True, "scope_id": scope_id, "target_id": target_id,
                "plan": None, "steps": (), "progress": self._current.get(scope_id),
                "reason": f"scope {scope_id!r} released without restore — "
                          f"residency is left as it is: the claim's target "
                          f"and any other resident stay up (the waiter owns "
                          f"the wake; the lane stays resident until a gen "
                          f"entry evicts it)",
            }

        # The scope's own models: what its claim plan loaded. Step-down is
        # scoped to those — releasing one live scope must never step down a
        # model another still-open scope brought in.
        brought_in = scope["brought_in"]

        # Phase 1: step down what this scope brought in that is still
        # resident (models evicted mid-scope have nothing to step down).
        down = [rec for rec in self._ledger.residents(target_id)
                if rec.model_id in brought_in]
        stepdown = LoadPlan(
            target_id, True,
            f"scope {scope_id!r} released",
            tuple(PlanStep(action="release", target_id=target_id,
                           model_id=rec.model_id, engine=rec.engine,
                           reason=(f"workflow scope {scope_id!r} released — "
                                   f"{rec.model_id} steps down so what was "
                                   f"resident at claim can be restored"),
                           lane=rec.lane)
                  for rec in down), ())
        r1 = self._run_plan(stepdown, key=scope_id, scope_id=scope_id,
                            emit_progress=False)
        if not r1["ok"]:
            return r1

        # Phase 2: restore the claim-time residents, per policy. Restore
        # fidelity comes from the load specs recorded when each model was
        # actually loaded — a restored model keeps its weights path and load
        # estimate even when the evicted model was itself mid-scope (its
        # request lives in another open scope the planner cannot see).
        #
        # This release and the approval gate share the SAME construction code
        # (``_restore_requests`` / ``restore_plan``), so both describe the
        # restore the same way. A20 closes the remaining pre-call TOCTOU: the
        # recomputed plan is re-verified against the EXACT set the gate
        # approved (``approved_restore``) before anything loads — a plan that
        # grew (a concurrent claim/eviction in the gate-to-handler window)
        # is refused, not silently executed.
        plan = self.restore_plan(scope_id) or LoadPlan(
            target_id, True, "nothing to restore", (), ())
        if approved_restore is not None:
            approved = set(approved_restore)
            gate_engines = set(gated_engines)
            unapproved = sorted(
                {s.model_id for s in plan.load_steps
                 if s.engine in gate_engines and s.model_id not in approved})
            if unapproved:
                return self._fail(
                    scope_id, scope_id,
                    "the restore plan changed after the approval gate "
                    f"inspected it — the recomputed restore would (re)start "
                    f"a server for {unapproved}, which the gate never saw; "
                    "the restore is refused and nothing was loaded "
                    "(fail-closed). The scope's own models were already "
                    "stepped down; re-run the release to approve the new "
                    "restore plan",
                    target_id, plan)
        r2 = self._run_plan(plan, key=scope_id, scope_id=scope_id)
        if r2["ok"]:
            with self._scopes_lock:
                self._scopes.pop(scope_id, None)
        return {"ok": r2["ok"], "scope_id": scope_id,
                "target_id": target_id, "plan": plan,
                "steps": r1["steps"] + r2["steps"],
                "progress": r2["progress"],
                "reason": (r2["reason"] if r2["ok"]
                           else r2.get("reason", plan.reason))}

    def _restore_requests(self, target_id, restore):
        """LoadRequests to restore a scope's claim-time residents, carrying
        the weights path and load estimate each model actually loaded with.
        Shared by ``release`` and ``restore_plan`` so the two can never
        diverge."""
        requests = []
        for rec in restore:
            spec = self._load_specs.get((target_id, rec.model_id), {})
            requests.append(LoadRequest(
                model_id=rec.model_id, engine=rec.engine, lane=rec.lane,
                modality="restore", weights_path=spec.get("weights_path"),
                estimated_s=spec.get("estimated_s")))
        return requests

    def restore_plan(self, scope_id):
        """What releasing ``scope_id`` WOULD load — the phase-2 restore plan,
        computed WITHOUT executing anything (the planner is pure: nothing is
        loaded, released or recorded).

        ``None`` when the scope is unknown — a release of an unknown scope
        fails closed and starts nothing, so there is nothing to inspect. The
        returned plan's ``load_steps`` name the engines a release would
        (re)start, so the approval gate can ask before a process-engine
        server (llama.cpp, vLLM) is spawned."""
        with self._scopes_lock:
            scope = self._scopes.get(scope_id)
        if scope is None:
            return None
        target_id = scope["target_id"]
        return self._planner.plan_load_many(
            target_id, self._restore_requests(target_id, scope["restore"]))

    # -- pre-warm and progress --

    def prewarm(self, target_id, request) -> dict:
        """Pre-warm on intent: start the load before the user's click
        finishes. Same plan and honest progress as a scoped load."""
        key = f"{target_id}:{request.model_id}"
        return self._run_plan(
            self._planner.plan_load_many(target_id, [request]),
            key=key, ready_model=request.model_id)

    # -- residency heal (gate on the runtime, not the stale ledger) --

    @staticmethod
    def _runtime_reports_down(adapter) -> bool:
        """True when an adapter's lifecycle probe says its runtime is NOT up.

        Only an explicit, healthy probe naming a down state (``stopped`` /
        ``idle``) counts — the honest "the container/process is not running"
        answers. Anything else keeps the optimistic resident assumption:
        an unregistered adapter, a probe that raises, a probe whose ``ok`` is
        False (``unknown`` — fail-soft), or any up state (``running`` /
        ``booting`` / ``ready`` / ``observed``). Gating on a real "down"
        answer — never on an absence of one — is the fail-soft choice: a
        transiently unreadable probe degrades to the old behavior (treat as
        resident) rather than firing a spurious load."""
        try:
            probe = adapter.probe()
        except Exception:  # noqa: BLE001 — a probe never sinks the caller
            return False
        if not isinstance(probe, dict) or not probe.get("ok"):
            return False
        state = str(probe.get("state") or "").strip().lower()
        return state in ("stopped", "idle")

    def _heal_residency(self, target_id, model_ids) -> tuple:
        """Drop ledger residency for any of ``model_ids`` whose runtime is down.

        Only the named models are checked — those a plan would otherwise treat
        as resident and skip loading. For each that the ledger calls resident
        and whose engine has a registered adapter, consult the lifecycle
        probe: a probe that reports the runtime stopped releases that model
        from the ledger so the planner treats it as NOT resident and re-emits
        its load step. Fail-soft: an engine with no adapter, a raising probe,
        or a probe that does not name a down state leaves residency untouched
        (the optimistic assumption). Returns the model ids healed, in order —
        for honest reporting/tests."""
        want = {m for m in (model_ids or ()) if m}
        if not want:
            return ()
        healed = []
        for rec in self._ledger.residents(target_id):
            if rec.model_id not in want:
                continue
            adapter = self._adapters.get(rec.engine)
            if adapter is None:
                continue
            if self._runtime_reports_down(adapter):
                self._ledger.record_released(target_id, rec.model_id)
                healed.append(rec.model_id)
        return tuple(healed)

    def touch(self, target_id, model_id) -> bool:
        """Record a use — keeps the model inside the recency window."""
        return self._ledger.touch(target_id, model_id)

    def adapter_for(self, engine):
        """The registered engine-lifecycle adapter for ``engine``, or ``None``.

        A read-only accessor so choreography outside the planner (the
        facade's ``prepare_gen``) can drive one lane's own lifecycle seam
        (``probe`` / ``release``) without reaching into the orchestrator's
        private adapter table. An unknown engine is ``None``, never a
        raise."""
        return self._adapters.get(str(engine or ""))

    def progress(self, key) -> dict | None:
        """The latest progress row for a scope id or pre-warm key."""
        return self._current.get(key)

    def history(self, key) -> tuple:
        """Every progress row recorded for a key, in order."""
        return tuple(self._history.get(key, []))

    # -- plan execution (strictly sequential) --

    def _run_plan(self, plan, key, scope_id=None, ready_model="",
                  emit_progress=True) -> dict:
        if not plan.ok:
            return self._fail(key, scope_id, plan.reason, plan.target_id,
                              plan, ())
        executed = []
        with self._lock:
            if emit_progress and plan.load_steps:
                self._record(key, progress_row(
                    "queued", target_id=plan.target_id, scope_id=scope_id))
            for step in plan.release_steps:
                adapter = self._adapters.get(step.engine)
                if adapter is None:
                    return self._fail(key, scope_id,
                                      f"no engine adapter registered for "
                                      f"{step.engine!r} — cannot release "
                                      f"{step.model_id} (fail-closed)",
                                      plan.target_id, plan, executed)
                res = adapter.release(step.model_id)
                if not res.get("ok"):
                    return self._fail(
                        key, scope_id,
                        f"release of {step.model_id} failed — "
                        f"{res.get('reason', 'no reason given')}; stopping "
                        f"the plan (fail-closed)",
                        plan.target_id, plan, executed)
                self._ledger.record_released(step.target_id, step.model_id)
                executed.append(step)
            for step in plan.load_steps:
                adapter = self._adapters.get(step.engine)
                if adapter is None:
                    return self._fail(key, scope_id,
                                      f"no engine adapter registered for "
                                      f"{step.engine!r} — cannot load "
                                      f"{step.model_id} (fail-closed)",
                                      plan.target_id, plan, executed)
                if emit_progress:
                    self._record(key, progress_row(
                        loading_state(step.estimated_s),
                        model_id=step.model_id, target_id=step.target_id,
                        scope_id=scope_id, estimated_s=step.estimated_s))
                res = adapter.ensure_loaded(step.model_id,
                                            weights_path=step.weights_path)
                if not res.get("ok"):
                    return self._fail(
                        key, scope_id,
                        f"load of {step.model_id} failed — "
                        f"{res.get('reason', 'no reason given')}; stopping "
                        f"the plan (fail-closed)",
                        plan.target_id, plan, executed)
                self._ledger.record_resident(step.target_id, step.model_id,
                                             engine=step.engine,
                                             lane=step.lane)
                self._load_specs[(step.target_id, step.model_id)] = {
                    "weights_path": step.weights_path,
                    "estimated_s": step.estimated_s,
                }
                executed.append(step)
            if emit_progress:
                self._record(key, progress_row(
                    "ready", model_id=ready_model, target_id=plan.target_id,
                    scope_id=scope_id))
        return {"ok": True, "scope_id": scope_id, "target_id": plan.target_id,
                "reason": plan.reason, "plan": plan,
                "steps": tuple(executed),
                "progress": self._current.get(key)}

    def _fail(self, key, scope_id, reason, target_id, plan=None,
              steps=()) -> dict:
        row = progress_row(f"error: {reason}", target_id=str(target_id or ""),
                           scope_id=scope_id)
        self._record(key, row)
        return {"ok": False, "scope_id": scope_id,
                "target_id": target_id, "reason": reason, "plan": plan,
                "steps": tuple(steps), "progress": row}

    def _record(self, key, row) -> None:
        self._history.setdefault(key, []).append(row)
        self._current[key] = row

    # -- swap: the one-call weight exchange -----------------------------------

    # What a READY probe must show to confirm a swap: an observed / running /
    # ready engine. "unknown" (fail-soft probe) and "stopped" (a dead
    # process) never confirm readiness.
    READY_STATES = ("observed", "running", "ready")

    def swap(self, target_id, request, release=None, dry_run=False) -> dict:
        """Exchange the resident weights for an incoming model in ONE
        choreographed call, in strict phases — each one measured by the
        injectable clock seam:

        - ``PREFLIGHT`` (no side effects): plan the eviction set and the load,
          fit-gate the incoming against the NOMINAL booked ceilings WITH the
          eviction applied (a refusal leaves residency untouched), verify the
          target engine is live/reachable — engine liveness/readiness only,
          the pre-warm: a stopped-but-startable server process passes and
          starts during LOAD, and weights are never loaded early — and verify
          the incoming weights are already present on the target (a swap must
          not begin a download mid-flight; it refuses with ``weights not
          present — run model_forge_install first`` instead).
        - ``EVICT``: only the planned residents, in policy order
          (least-recently used first), measured.
        - ``LOAD``: immediately after the final eviction, exactly ONE load,
          measured — through the engine's warm path where one exists (the
          adapter's ``ensure_loaded`` handles it; the applied warm path is
          recorded verbatim in the row).
        - ``READY``: probe to confirm, measured.

        The hard guarantee: no evict/release call runs before PREFLIGHT
        returns ok — a failed preflight leaves residency untouched. Mid-flight
        failures are honest rows, never a raise and never a hang: the row
        states exactly what is now resident (``"nothing"`` is an honest
        state) and why.

        ``release`` names explicit outgoing model ids (default: the eviction
        policy's own set); image-generation entries evict GPU residents
        deterministically every time regardless. ``dry_run=True`` returns the
        preflight plan rows with a ``nothing was changed`` note and has zero
        side effects. Every executed swap lands in the ledger's bounded swap
        history (``ResidencyLedger.swap_history`` / ``swap_stats``); dry runs
        never record."""
        t0 = self._clock()
        model_id = str(getattr(request, "model_id", "") or "").strip()
        phases: dict = {}
        checks: list = []
        preflight = {"name": "PREFLIGHT", "ok": False, "reason": "",
                     "seconds": None, "checks": checks,
                     "evict_plan": [], "load_plan": []}
        phases["preflight"] = preflight
        dry_note = ("dry run — nothing was changed (no weights were evicted "
                    "or loaded; the rows below are the preflight plan)")
        untouched_note = ("nothing was evicted or loaded — residency is "
                          "untouched")

        def _finish(ok, reason, *, evicted=(), warm_path=None,
                    evict_seconds=None, load_seconds=None,
                    ready_seconds=None, note=""):
            timings = {
                "preflight_seconds": preflight.get("seconds"),
                "evict_seconds": evict_seconds,
                "load_seconds": load_seconds,
                "ready_seconds": ready_seconds,
                "total_seconds": self._clock() - t0,
            }
            return self._swap_result(
                target_id, model_id, ok, reason,
                evicted=list(evicted), warm_path=warm_path,
                phases=phases, timings=timings, dry_run=bool(dry_run),
                note=note)

        def _refuse(reason, check):
            checks.append({"check": check, "ok": False, "reason": reason})
            preflight["reason"] = reason
            preflight["seconds"] = self._clock() - t0
            return _finish(False, reason,
                           note=dry_note if dry_run else untouched_note)

        # ---- PREFLIGHT: plan and verify — no side effects before ok ----
        if not model_id:
            return _refuse("swap needs an incoming model id — nothing to "
                           "swap in (fail-closed)", "incoming model")
        booked_gb = self._ledger.booked_vram_gb(model_id)
        if booked_gb is None:
            return _refuse(
                f"no booked footprint for {model_id} — fit is gated on "
                f"nominal booked ceilings and none is booked (fail-closed); "
                f"book a registry footprint or a measured peak first",
                "booked footprint")
        checks.append({"check": "booked footprint", "ok": True,
                       "reason": f"{model_id} books {booked_gb:g} GB "
                                 f"(nominal ceiling)"})
        budget = self._ledger.budget_gb(target_id)
        if budget is None:
            return _refuse(
                f"no nominal budget declared for compute target "
                f"{target_id!r} — fit cannot be gated on booked ceilings "
                f"(fail-closed)", "nominal budget")
        checks.append({"check": "nominal budget", "ok": True,
                       "reason": f"{budget:g} GB nominal budget declared"})

        plan = self._planner.plan_load_many(target_id, [request])
        if not plan.ok:
            return _refuse(plan.reason, "eviction + load plan")
        checks.append({"check": "eviction + load plan", "ok": True,
                       "reason": plan.reason})

        residents = self._ledger.residents(target_id)
        by_id = {rec.model_id: rec for rec in residents}
        image_entry = is_image_generation(getattr(request, "modality", ""),
                                          getattr(request, "lane", ""))
        explicit = [str(r or "").strip() for r in (release or ())]
        explicit = [r for r in explicit if r]
        if explicit and model_id in explicit:
            return _refuse(
                f"the incoming model {model_id} cannot be named as its own "
                f"eviction — nothing was evicted (fail-closed)",
                "release ids")
        policy_ids = [step.model_id for step in plan.release_steps]
        explicit_skipped = [r for r in explicit if r not in by_id]
        skipped_note = (
            f"; skipped {', '.join(sorted(explicit_skipped))} — not resident "
            f"on {target_id!r}, nothing to evict for it"
            if explicit_skipped else "")
        if explicit and image_entry:
            # Image-generation entry claims the whole card: deterministic
            # full eviction every time — the policy set is the eviction set.
            evict_ids = list(policy_ids)
            checks.append({
                "check": "release ids", "ok": True,
                "reason": (f"image-generation entry claims the whole GPU — "
                           f"the explicit release list is honored as part of "
                           f"the deterministic full eviction "
                           f"({len(evict_ids)} resident(s))"
                           f"{skipped_note}")})
        elif explicit:
            named = [r for r in explicit if r in by_id]
            # Explicit outgoing ids in policy order (least-recently used
            # first). Unbooked residents join them: their VRAM cannot be
            # accounted in the nominal gate (the planner's own rule).
            chosen = set(named)
            for rec in residents:
                if self._ledger.booked_vram_gb(rec.model_id) is None:
                    chosen.add(rec.model_id)
            evict_ids = [rec.model_id for rec in
                         sorted((by_id[m] for m in chosen),
                                key=lambda rec: rec.last_used)]
            reason_ids = (f"{len(named)} explicit release id(s) planned"
                          + skipped_note
                          + ("; unbooked residents join them (their VRAM "
                             "cannot be accounted in the nominal gate)"
                             if chosen else ""))
            checks.append({"check": "release ids", "ok": True,
                           "reason": reason_ids or
                                     "nothing resident to evict"})
        else:
            evict_ids = list(policy_ids)

        # Fit gate on nominal booked ceilings WITH the eviction applied —
        # live free VRAM only orders candidates, never gates.
        evict_set = set(evict_ids)
        keep_gb = sum(self._ledger.booked_vram_gb(rec.model_id) or 0.0
                      for rec in residents if rec.model_id not in evict_set)
        incoming_gb = 0.0 if model_id in by_id else booked_gb
        if keep_gb + incoming_gb > budget:
            return _refuse(
                f"nominal fit fails even with the eviction applied: kept "
                f"residents book {keep_gb:g} GB and {model_id} books "
                f"{incoming_gb:g} GB into a {budget:g} GB budget "
                f"(fail-closed); nothing was evicted",
                "nominal fit with eviction applied")
        checks.append({
            "check": "nominal fit with eviction applied", "ok": True,
            "reason": f"{keep_gb + incoming_gb:g} GB booked into a "
                      f"{budget:g} GB nominal budget with "
                      f"{len(evict_ids)} eviction(s) applied (live free VRAM "
                      f"orders only, never gates)"})

        # Engine liveness (the pre-warm: liveness/readiness ONLY — a stopped
        # but startable server process passes and starts during LOAD; weights
        # are never loaded before eviction).
        adapter = self._adapters.get(getattr(request, "engine", ""))
        if adapter is None:
            return _refuse(
                f"no engine adapter registered for "
                f"{getattr(request, 'engine', '')!r} — cannot verify engine "
                f"liveness or load {model_id} (fail-closed)",
                "engine liveness")
        probe = adapter.probe()
        state = str(probe.get("state") or "")
        if not probe.get("ok") or state == "unknown":
            probe_reason = probe.get("reason") or (f"probe state {state!r} "
                                                   f"confirms no liveness")
            return _refuse(
                f"engine {getattr(request, 'engine', '')!r} is not "
                f"live/reachable — {probe_reason}; the pre-warm is engine "
                f"liveness only and the swap stops here (fail-closed)",
                "engine liveness")
        live_reason = (f"engine {request.engine!r} answers its probe "
                       f"(state {state!r}) — pre-warm is engine "
                       f"liveness/readiness only; weights are never loaded "
                       f"before eviction")
        if state == "idle":
            live_reason += (f"; the engine process is stopped but startable "
                            f"— it starts during the measured LOAD phase")
        checks.append({"check": "engine liveness", "ok": True,
                       "reason": live_reason})

        # Weights already present: a swap must not begin a download mid-flight.
        present = any(p.model_id == model_id
                      for p in self._ledger.pulled(target_id))
        if not present:
            return _refuse(
                f"weights not present — run model_forge_install first "
                f"({model_id} has no recorded weights on {target_id!r}; a "
                f"swap must not begin a download mid-flight)",
                "weights present")
        checks.append({
            "check": "weights present", "ok": True,
            "reason": f"{model_id} weights are present on {target_id!r} — "
                      f"no download will begin mid-flight"})

        step_reason = {step.model_id: step.reason
                       for step in plan.release_steps}
        preflight["evict_plan"] = [
            {"model_id": mid, "engine": by_id[mid].engine,
             "lane": by_id[mid].lane,
             "reason": step_reason.get(
                 mid, f"explicit release requested — {mid} steps down for "
                      f"the incoming model")}
            for mid in evict_ids if mid in by_id]
        preflight["load_plan"] = [
            {"model_id": step.model_id, "engine": step.engine,
             "lane": step.lane, "weights_path": step.weights_path,
             "estimated_s": step.estimated_s, "reason": step.reason}
            for step in plan.load_steps]
        preflight["ok"] = True
        preflight["reason"] = (f"preflight ok: {len(evict_ids)} eviction(s) "
                               f"planned, {len(plan.load_steps)} load(s) "
                               f"planned (one at a time), engine live, "
                               f"weights present")
        preflight["seconds"] = self._clock() - t0

        if dry_run:
            warm = str(getattr(adapter, "WARM_PATH", "") or "")
            return _finish(True, preflight["reason"], warm_path=warm or None,
                           note=dry_note)

        # ---- EVICT: only the planned residents, policy order, measured ----
        with self._lock:
            t_evict = self._clock()
            evicted: list = []
            evict_steps: list = []
            for mid in evict_ids:
                rec = by_id.get(mid)
                if rec is None:
                    continue
                step_t = self._clock()
                ev_adapter = self._adapters.get(rec.engine)
                if ev_adapter is None:
                    evict_seconds = self._clock() - t_evict
                    phases["evict"] = {
                        "name": "EVICT", "ok": False, "seconds": evict_seconds,
                        "evicted": list(evicted), "steps": evict_steps,
                        "reason": f"no engine adapter registered for "
                                  f"{rec.engine!r}"}
                    return _finish(
                        False,
                        f"no engine adapter registered for {rec.engine!r} — "
                        f"cannot evict {mid}; the swap stopped before load "
                        f"(fail-closed) after evicting {len(evicted)} "
                        f"model(s)", evicted=evicted,
                        evict_seconds=evict_seconds)
                res = ev_adapter.release(mid)
                step_seconds = self._clock() - step_t
                evict_steps.append({
                    "model_id": mid, "engine": rec.engine,
                    "ok": bool(res.get("ok")), "seconds": step_seconds,
                    "reason": res.get("reason", "")})
                if not res.get("ok"):
                    evict_seconds = self._clock() - t_evict
                    phases["evict"] = {
                        "name": "EVICT", "ok": False, "seconds": evict_seconds,
                        "evicted": list(evicted), "steps": evict_steps,
                        "reason": res.get("reason", "")}
                    return _finish(
                        False,
                        f"eviction of {mid} failed — "
                        f"{res.get('reason', 'no reason given')}; the swap "
                        f"stopped before load (fail-closed) after evicting "
                        f"{len(evicted)} model(s)", evicted=evicted,
                        evict_seconds=evict_seconds)
                self._ledger.record_released(target_id, mid)
                evicted.append(mid)
            evict_seconds = self._clock() - t_evict
            phases["evict"] = {
                "name": "EVICT", "ok": True, "seconds": evict_seconds,
                "evicted": list(evicted), "steps": evict_steps,
                "reason": (f"{len(evicted)} resident(s) evicted in policy "
                           f"order (least-recently used first)"
                           if evicted else
                           "nothing to evict — the eviction set was empty")}

            # ---- LOAD: exactly one load, right after the last eviction ----
            t_load = self._clock()
            warm_path = str(getattr(adapter, "WARM_PATH", "") or
                            "engine-native load (no warm-swap surface "
                            "declared)")
            if not plan.load_steps:
                load_seconds = self._clock() - t_load
                phases["load"] = {
                    "name": "LOAD", "ok": True, "seconds": load_seconds,
                    "warm_path": warm_path, "steps": [],
                    "reason": f"{model_id} is already resident — nothing to "
                              f"load"}
            else:
                step = plan.load_steps[0]
                res = adapter.ensure_loaded(model_id,
                                            weights_path=step.weights_path)
                load_seconds = self._clock() - t_load
                phases["load"] = {
                    "name": "LOAD", "ok": bool(res.get("ok")),
                    "seconds": load_seconds, "warm_path": warm_path,
                    "steps": [{"model_id": model_id, "engine": step.engine,
                               "ok": bool(res.get("ok")),
                               "seconds": load_seconds,
                               "reason": res.get("reason", "")}],
                    "reason": res.get("reason", "")}
                if not res.get("ok"):
                    return _finish(
                        False,
                        f"load of {model_id} failed after evicting "
                        f"{len(evicted)} model(s) — "
                        f"{res.get('reason', 'no reason given')}",
                        evicted=evicted, evict_seconds=evict_seconds,
                        load_seconds=load_seconds, warm_path=warm_path)
                self._ledger.record_resident(target_id, model_id,
                                             engine=step.engine,
                                             lane=step.lane)
                self._load_specs[(target_id, model_id)] = {
                    "weights_path": step.weights_path,
                    "estimated_s": step.estimated_s}

            # ---- READY: probe to confirm, measured ----
            t_ready = self._clock()
            ready_probe = adapter.probe()
            ready_seconds = self._clock() - t_ready
            ready_state = str(ready_probe.get("state") or "")
            confirmed = (bool(ready_probe.get("ok"))
                         and ready_state in self.READY_STATES)
            ready_why = ready_probe.get("reason") or (f"probe state "
                                                      f"{ready_state!r}")
            phases["ready"] = {
                "name": "READY", "ok": confirmed, "seconds": ready_seconds,
                "probe": ready_probe,
                "reason": (f"ready confirmed (probe state {ready_state!r})"
                           if confirmed else
                           f"the ready probe could not confirm the load — "
                           f"{ready_why}")}
        if not confirmed:
            return _finish(
                False,
                f"the load completed but the ready probe could not confirm "
                f"it — {ready_why}",
                evicted=evicted, evict_seconds=evict_seconds,
                load_seconds=load_seconds, ready_seconds=ready_seconds,
                warm_path=warm_path)
        load_note = (f"{model_id} loaded via {warm_path}"
                     if plan.load_steps else
                     f"{model_id} is already resident (nothing to load)")
        return _finish(
            True,
            f"swap complete: {len(evicted)} evicted, {load_note}, "
            f"ready confirmed", evicted=evicted,
            evict_seconds=evict_seconds, load_seconds=load_seconds,
            ready_seconds=ready_seconds, warm_path=warm_path)

    def _swap_result(self, target_id, model_id, ok, reason, *, evicted,
                     warm_path, phases, timings, dry_run, note="") -> dict:
        """Build the swap row (pure data) and — unless this was a dry run —
        record the measured truth in the ledger's bounded swap history. The
        row always states exactly what is resident now (``"nothing"`` is an
        honest state); failures append it to the reason too."""
        residents = sorted(rec.model_id
                           for rec in self._ledger.residents(target_id))
        resident_state = ", ".join(residents) if residents else "nothing"
        if not ok:
            reason = f"{reason}; resident now: {resident_state}"
        if not dry_run:
            self._ledger.record_swap({
                "model_id": model_id,
                "evicted": list(evicted),
                "evict_seconds": timings.get("evict_seconds"),
                "load_seconds": timings.get("load_seconds"),
                "ready_seconds": timings.get("ready_seconds"),
                "total_seconds": timings.get("total_seconds"),
                "ok": bool(ok),
                "warm_path": warm_path,
            })
        row = {"ok": bool(ok), "action": "swap", "target_id": target_id,
               "model_id": model_id, "dry_run": bool(dry_run),
               "evicted": list(evicted), "warm_path": warm_path,
               "phases": phases, "timings": dict(timings),
               "resident_now": residents, "resident_state": resident_state,
               "reason": reason, "swap_stats": self._ledger.swap_stats()}
        if note:
            row["note"] = note
        return row
