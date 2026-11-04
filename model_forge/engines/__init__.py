"""Engine adapters: how Model Forge finds and talks to local inference engines.

Detection is non-invasive: HTTP probes to well-known local endpoints and
`which` checks for CLIs. Nothing is installed or started at import time.
Adapters implement the same contract the catalog review allows: engines are
driven through their own public HTTP APIs / CLIs — never through Hermes core
internals.
"""

from __future__ import annotations

from .base import EngineStatus
from . import llamacpp, strata

__all__ = ["EngineStatus", "detect_engines"]


def detect_engines() -> list[EngineStatus]:
    """Probe every adapter; never raises."""
    out = []
    for detect in (llamacpp.detect, strata.detect):
        try:
            out.append(detect())
        except Exception as exc:  # a broken probe must not sink the tool call
            out.append(EngineStatus(
                id="unknown", display_name="probe error", available=False,
                detail=f"{type(exc).__name__}: {exc}",
            ))
    return out
