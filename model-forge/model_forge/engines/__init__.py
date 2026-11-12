"""Engine adapters: how Model Forge finds and talks to local inference engines.

Detection is non-invasive: HTTP probes to well-known local endpoints and
`which` checks for CLIs. Nothing is installed or started at import time.
Adapters implement the same contract the catalog review allows: engines are
driven through their own public HTTP APIs / CLIs — never through Hermes core
internals.
"""

from __future__ import annotations

import concurrent.futures

from .base import EngineStatus, root_url
from . import comfyui, llamacpp, maya, openai_compat, sglang, strata, vllm

__all__ = ["EngineStatus", "detect_engines"]

_DETECTORS = (llamacpp.detect, strata.detect, maya.detect, comfyui.detect,
              vllm.detect, sglang.detect, openai_compat.detect)
_NAMED_IDS = ("llamacpp", "ollama", "strata", "maya", "comfyui", "vllm",
              "sglang")


def _safe_detect(detect, base_url: str) -> EngineStatus:
    try:
        return detect(base_url)
    except Exception as exc:  # a broken probe must not sink the tool call
        return EngineStatus(
            id="unknown", display_name="probe error", available=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def detect_engines(base_url: str = "") -> list[EngineStatus]:
    """Probe every adapter at a compute target; never raises. When a
    distinctive engine already answered at the endpoint, the generic
    openai-compatible row for that endpoint is dropped (same server).

    The per-adapter probes run concurrently (one worker each): an unreachable
    target now costs the slowest single probe (a few seconds) instead of the
    sum of every adapter's dead-connection timeouts. ``executor.map`` keeps
    the result order identical to ``_DETECTORS`` regardless of which probe
    answers first, so row order and detection semantics are unchanged."""
    detectors = _DETECTORS
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(detectors)),
            thread_name_prefix="model-forge-probe") as pool:
        out = list(pool.map(lambda d: _safe_detect(d, base_url), detectors))
    named_endpoints = {e.endpoint for e in out
                       if e.available and e.id in _NAMED_IDS}
    return [e for e in out
            if not (e.id == "openai-compatible" and e.available
                    and e.endpoint in named_endpoints)]
