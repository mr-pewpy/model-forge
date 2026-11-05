"""Engine adapters: how Model Forge finds and talks to local inference engines.

Detection is non-invasive: HTTP probes to well-known local endpoints and
`which` checks for CLIs. Nothing is installed or started at import time.
Adapters implement the same contract the catalog review allows: engines are
driven through their own public HTTP APIs / CLIs — never through Hermes core
internals.
"""

from __future__ import annotations

from .base import EngineStatus, root_url
from . import llamacpp, maya, openai_compat, strata

__all__ = ["EngineStatus", "detect_engines"]


def detect_engines(base_url: str = "") -> list[EngineStatus]:
    """Probe every adapter at a compute target; never raises. When a
    distinctive engine already answered at the endpoint, the generic
    openai-compatible row for that endpoint is dropped (same server)."""
    out = []
    for detect in (llamacpp.detect, strata.detect, maya.detect,
                   openai_compat.detect):
        try:
            out.append(detect(base_url))
        except Exception as exc:  # a broken probe must not sink the tool call
            out.append(EngineStatus(
                id="unknown", display_name="probe error", available=False,
                detail=f"{type(exc).__name__}: {exc}",
            ))
    named_endpoints = {e.endpoint for e in out
                       if e.available and e.id in ("llamacpp", "ollama",
                                                   "strata", "maya")}
    return [e for e in out
            if not (e.id == "openai-compatible" and e.available
                    and e.endpoint in named_endpoints)]
