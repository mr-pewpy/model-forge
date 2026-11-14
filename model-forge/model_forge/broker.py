"""Model residency and lifecycle — the v2 weight-residency core (roadmap §5).

What weights live where, what is loaded in VRAM right now, and what happens
when a task needs a different model. Four pieces:

- ``ResidencyLedger`` — pulls, residency, booked ceilings per model (from a
  booking dict injected by callers: registry footprints or measured peaks)
  and live free-VRAM snapshots. Live free VRAM ORDERs candidates only.
- ``EngineLifecycle`` adapters — ``ensure_loaded`` / ``release`` / ``probe``
  per engine: Ollama (warm via a generate call with ``keep_alive``, evict
  with ``keep_alive=0``), llama-swap (observational pass-through — it swaps
  models on demand by route and exposes no eviction control), and llama.cpp
  llama-server (process start/stop on a weights path; the process runner is
  injectable and a missing binary fails closed).
- ``EvictionPlanner`` — the battle-tested serving policy (roadmap §3) as
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
  for a modality and restores what was resident on release), pre-warm on
  intent, and honest progress states (``queued`` / ``loading (~Ns)`` /
  ``ready``) as pure data.

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
import json
import math
import shutil
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "RECENCY_WINDOW_S",
    "Booking",
    "EngineLifecycle",
    "EvictionPlanner",
    "LlamaCppServerLifecycle",
    "LlamaSwapLifecycle",
    "LoadPlan",
    "LoadRequest",
    "MeasuredFootprintRecorder",
    "OllamaLifecycle",
    "Orchestrator",
    "PlanStep",
    "PullRecord",
    "ResidencyLedger",
    "ResidentRecord",
    "is_image_generation",
    "loading_state",
]

# Battle-tested policy: models used within roughly the last ten minutes stay
# warm; recency outranks pure LRU near the top of the stack.
RECENCY_WINDOW_S = 600.0


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
    ``source="measured"`` is a peak sampled during a bench — measured peaks
    are the booking numbers and never get downgraded by a registry value."""
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
        never presented as an advisory published figure, or vice versa."""
        if self.source == "measured":
            return "measured peak (bench)"
        return "published footprint (advisory)"


@dataclass(frozen=True)
class PullRecord:
    """Weights on disk at a compute target, with size where known."""
    model_id: str
    path: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class ResidentRecord:
    """A model loaded in memory on a compute target."""
    model_id: str
    engine: str
    lane: str
    loaded_at: float
    last_used: float


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
        "load_s": …}``) or a ``Booking``. A measured booking always wins: a
        registry footprint never downgrades a measured peak."""
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
            if (existing is not None and existing.source == "measured"
                    and spec.source != "measured"):
                return {"ok": True, "model_id": model_id,
                        "vram_gb": existing.vram_gb, "source": existing.source,
                        "label": existing.label,
                        "kept": "measured booking kept — published footprints "
                                "are advisory"}
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

    def record_measured_peak(self, model_id, peak_gb, load_s=None) -> dict:
        """The measured-footprint write path: book an observed bench peak as
        this model's MEASURED ceiling. Routed through ``book`` so the
        no-downgrade invariant holds — a later published footprint never
        replaces a measured peak — while a newer measured peak may (the
        latest bench wins over an older one)."""
        return self.book(model_id, peak_gb, source="measured", load_s=load_s)

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

    def record_pull(self, target_id, model_id, path, size_bytes=None) -> dict:
        if not str(path or "").strip():
            return {"ok": False, "model_id": model_id,
                    "error": "path must be a non-empty string"}
        try:
            size = int(size_bytes) if size_bytes is not None else None
        except (TypeError, ValueError):
            size = None
        rec = PullRecord(model_id=model_id, path=str(path), size_bytes=size)
        with self._lock:
            self._pulls.setdefault(target_id, {})[model_id] = rec
        return {"ok": True, "target_id": target_id, "model_id": model_id,
                "path": rec.path, "size_bytes": rec.size_bytes}

    def pulled(self, target_id) -> tuple[PullRecord, ...]:
        with self._lock:
            return tuple(self._pulls.get(target_id, {}).values())

    # -- residency (weights in memory) --

    def record_resident(self, target_id, model_id, engine, lane="chat") -> dict:
        now = self._clock()
        rec = ResidentRecord(model_id=model_id, engine=engine, lane=lane,
                             loaded_at=now, last_used=now)
        with self._lock:
            self._residents.setdefault(target_id, {})[model_id] = rec
        return {"ok": True, "target_id": target_id, "model_id": model_id,
                "engine": engine, "lane": lane}

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
                loaded_at=rec.loaded_at, last_used=now)
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

    def __init__(self, sampler, model_id, ledger, clock=None):
        self._sampler = sampler
        self._model_id = model_id
        self._ledger = ledger
        self._clock = clock or ledger.clock
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
                                                load_s=load_s)
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
    means is a policy decision — the policy layer fails closed on it."""

    engine_id: str = ""

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


def _http_json(method: str, url: str, body: dict | None = None,
               timeout: float = 10.0) -> dict:
    """The default injectable HTTP seam: stdlib JSON request/response."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


class OllamaLifecycle(EngineLifecycle):
    """Ollama lifecycle: warm with a generate call carrying ``keep_alive``,
    evict with ``keep_alive=0`` — the only unload Ollama has.

    The HTTP function is injectable; the default is stdlib urllib."""

    engine_id = "ollama"

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


class ProcessHandle(Protocol):
    """The injectable process-handle contract: ``poll`` / ``terminate`` /
    ``wait`` — what a started llama-server answers to."""

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


def _spawn_process(argv) -> ProcessHandle:
    """Default process runner for llama-server: the one real spawn seam.
    A handle answers ``poll()`` / ``terminate()`` / ``wait(timeout=…)``."""
    return subprocess.Popen(list(argv))


class LlamaCppServerLifecycle(EngineLifecycle):
    """llama.cpp llama-server is a process: start it on a weights path, stop
    it to evict (roadmap §3: big models need process stop/start, not engine
    sleep/wake). The process runner and the binary lookup are injectable so
    tests spawn nothing — and a missing binary fails closed: no argv is even
    built, no half-started server."""

    engine_id = "llamacpp"

    def __init__(self, binary="llama-server", port=8080, extra_args=(),
                 runner=None, which=None):
        self._binary = binary
        self._port = port
        self._extra_args = tuple(extra_args)
        self._runner = runner or _spawn_process
        self._which = which or shutil.which
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
        argv = [found, "-m", str(weights_path),
                "--port", str(self._port if port is None else port)]
        argv += list(self._extra_args if extra_args is None else extra_args)
        try:
            handle = self._runner(argv)
        except Exception as exc:  # noqa: BLE001 — honest row, never a raise
            return {**row, "reason": f"llama-server start failed: "
                                     f"{type(exc).__name__}: {exc}"}
        self._handle = handle
        self._serving = model_id
        return {"ok": True, "engine": self.engine_id, "action": "load",
                "model_id": model_id, "argv": argv,
                "reason": f"started llama-server for {model_id} on port "
                          f"{argv[3]}"}

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


# --- eviction planner -------------------------------------------------------

class EvictionPlanner:
    """The battle-tested serving policy (roadmap §3) as plan data.

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

    def plan_load_many(self, target_id, requests) -> LoadPlan:
        """Plan making a set of models resident (e.g. a scope restore).

        Pure planning: nothing is loaded, released, or recorded here."""
        reqs = list(requests)
        if not reqs:
            return LoadPlan(target_id, True, "nothing to load", (), ())
        want_ids = {r.model_id for r in reqs}
        image_entry = any(is_image_generation(r.modality, r.lane)
                          for r in reqs)
        now = self._clock()

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
    what was resident on the target; ``release`` steps the scope's models
    down and restores exactly that set. ``prewarm`` loads on intent, before
    the user's click lands. Progress states (``queued`` / ``loading (~Ns)`` /
    ``ready``) are pure data rows recorded per scope or pre-warm key —
    no network I/O happens anywhere outside the injected adapter calls, and
    every engine call runs strictly one at a time (concurrency control is
    the caller's). A failed engine call mid-plan stops the run and is
    reported: fail closed, never a silent stall.
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
        # Per-target load specs of successfully loaded models (weights path,
        # load estimate) — a scope release restores claim-time residents with
        # the fidelity their original load had, even when the evicted model
        # belongs to another still-open scope whose request is not visible
        # from the releasing scope.
        self._load_specs: dict[tuple[str, str], dict] = {}

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
        if scope_id in self._scopes:
            return self._fail(scope_id, scope_id,
                              f"scope {scope_id!r} is already claimed — "
                              f"release it first (fail-closed)",
                              target_id, request)
        restore = tuple(self._ledger.residents(target_id))
        result = self._run_plan(
            self._planner.plan_load_many(target_id, [request]),
            key=scope_id, scope_id=scope_id, ready_model=request.model_id)
        if result["ok"]:
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

    def release(self, scope_id) -> dict:
        """Release a claimed scope: step the scope's models down, then
        restore what was resident at claim time (loads fire sequentially)."""
        scope = self._scopes.get(scope_id)
        if scope is None:
            return self._fail(scope_id, scope_id,
                              f"no claimed scope {scope_id!r} — nothing to "
                              f"release (fail-closed)", "", None)
        target_id = scope["target_id"]
        restore = scope["restore"]
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
        requests = []
        for rec in restore:
            spec = self._load_specs.get((target_id, rec.model_id), {})
            requests.append(LoadRequest(
                model_id=rec.model_id, engine=rec.engine, lane=rec.lane,
                modality="restore", weights_path=spec.get("weights_path"),
                estimated_s=spec.get("estimated_s")))
        plan = self._planner.plan_load_many(target_id, requests)
        r2 = self._run_plan(plan, key=scope_id, scope_id=scope_id)
        if r2["ok"]:
            self._scopes.pop(scope_id, None)
        return {"ok": r2["ok"], "scope_id": scope_id,
                "target_id": target_id, "plan": plan,
                "steps": r1["steps"] + r2["steps"],
                "progress": r2["progress"],
                "reason": (r2["reason"] if r2["ok"]
                           else r2.get("reason", plan.reason))}

    # -- pre-warm and progress --

    def prewarm(self, target_id, request) -> dict:
        """Pre-warm on intent: start the load before the user's click
        finishes. Same plan and honest progress as a scoped load."""
        key = f"{target_id}:{request.model_id}"
        return self._run_plan(
            self._planner.plan_load_many(target_id, [request]),
            key=key, ready_model=request.model_id)

    def touch(self, target_id, model_id) -> bool:
        """Record a use — keeps the model inside the recency window."""
        return self._ledger.touch(target_id, model_id)

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
