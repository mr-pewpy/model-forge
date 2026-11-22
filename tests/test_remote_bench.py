"""B3 remote bench — engine-reported VRAM + honest provenance (tests).

A remote engine's own memory report is an OBSERVATION (not a guess), so it is
booked — but with a distinct ``engine-reported (remote)`` provenance label,
lower-trust than a local vendor-sampled bench. The booking trust ranking
(measured > engine_reported > registry) keeps a higher-trust figure from being
downgraded. Fail-soft: an engine that reports no memory yields no booking.
"""

from __future__ import annotations

from model_forge.broker import Booking, ResidencyLedger
from model_forge.tools import (
    engine_reported_vram_gb, engine_reported_vram_sampler,
)

_GB = 1024 ** 3


class _Target:
    def __init__(self, base="http://remote:8080", kind="lan"):
        self.base_url = base
        self.kind = kind
        self.id = "remote"


# --- engine_reported_vram_gb: the engine's own observation ---------------

def test_engine_reported_vram_from_ollama_ps():
    # /api/ps size_vram sums across loaded models
    data = {"models": [{"size_vram": 2 * _GB}, {"size_vram": 1 * _GB}]}
    fetch = lambda url: data
    gb = engine_reported_vram_gb(_Target(), fetch=fetch)
    assert gb == pytest_approx(3.0)


def test_engine_reported_vram_from_comfyui_system_stats():
    # /system_stats devices[].gpu_total - gpu_free
    data = {"devices": [{"gpu_total": 8 * _GB, "gpu_free": 3 * _GB}]}
    fetch = lambda url: data
    # /api/ps returns nothing -> falls through to /system_stats
    def _fetch(url):
        return data if url.endswith("/system_stats") else {}
    gb = engine_reported_vram_gb(_Target(), fetch=_fetch)
    assert gb == pytest_approx(5.0)


def test_engine_reported_vram_none_when_engine_reports_nothing():
    fetch = lambda url: {}  # neither surface reports memory
    assert engine_reported_vram_gb(_Target(), fetch=fetch) is None


def test_engine_reported_vram_none_on_fetch_error():
    def _fetch(url):
        raise OSError("remote down")
    assert engine_reported_vram_gb(_Target(), fetch=_fetch) is None


def test_engine_reported_sampler_fail_soft():
    sampler, reason = engine_reported_vram_sampler(
        _Target(), fetch=lambda url: {})
    assert sampler is None
    assert "reports no GPU-memory figure" in reason


def test_engine_reported_sampler_reads():
    data = {"models": [{"size_vram": 4 * _GB}]}
    sampler, reason = engine_reported_vram_sampler(
        _Target(), fetch=lambda url: data)
    assert reason == ""
    assert sampler() == pytest_approx(4.0)


# --- booking trust ranking (B3 provenance) --------------------------------

def test_booking_label_engine_reported():
    b = Booking(vram_gb=5.0, source="engine_reported")
    assert b.label == "engine-reported (remote)"
    assert b.is_measured is False  # lower trust than a local bench


def test_engine_reported_books_but_never_downgrades_measured():
    ledger = ResidencyLedger()
    ledger.record_measured_peak("m", 9.5)  # local bench
    # a remote engine report must not downgrade the measured peak
    row = ledger.record_measured_peak("m", 7.0, source="engine_reported")
    assert row["ok"] and "kept" in row.get("kept", "")
    assert ledger.booked_vram_gb("m") == 9.5


def test_engine_reported_upgrades_registry():
    ledger = ResidencyLedger()
    ledger.book("m", 5.0, source="registry")  # advisory
    # an engine report is a real observation -> replaces the advisory
    row = ledger.record_measured_peak("m", 8.0, source="engine_reported")
    assert row["ok"] and row["source"] == "engine_reported"
    assert ledger.booked_vram_gb("m") == 8.0


def test_measured_upgrades_engine_reported():
    ledger = ResidencyLedger()
    ledger.record_measured_peak("m", 8.0, source="engine_reported")
    # a local bench is higher trust -> replaces the engine report
    row = ledger.record_measured_peak("m", 9.5)
    assert row["ok"] and row["source"] == "measured"
    assert ledger.booked_vram_gb("m") == 9.5


def test_engine_reported_replaces_older_engine_reported():
    ledger = ResidencyLedger()
    ledger.record_measured_peak("m", 8.0, source="engine_reported")
    ledger.record_measured_peak("m", 8.5, source="engine_reported")
    assert ledger.booked_vram_gb("m") == 8.5


def pytest_approx(v):
    # tiny local helper to avoid importing pytest at module scope
    class _Approx:
        def __init__(self, x):
            self.x = x

        def __eq__(self, other):
            return abs(float(other) - self.x) < 1e-6
    return _Approx(v)
