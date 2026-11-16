"""Engine adapters: how Model Forge finds and talks to local inference engines.

Detection is non-invasive: HTTP probes to well-known local endpoints and
`which` checks for CLIs. Nothing is installed or started at import time.
Adapters implement the same contract the catalog review allows: engines are
driven through their own public HTTP APIs / CLIs — never through Hermes core
internals.

Besides the compute target's own endpoint, ``detect_engines`` accepts
``extra_roots``: engine HTTP endpoints the user explicitly declared. Each
declared root is probed with the same fingerprint-first logic as the default
endpoint and its rows are reported with ``source="user-declared"``. With the
parameter absent, no declared root is probed and behavior is unchanged.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import replace

from .base import EngineStatus, root_url
from . import (comfyui, exo, llamacpp, maya, openai_compat, sglang, strata,
               vllm)

__all__ = ["EngineStatus", "detect_engines"]

_DETECTORS = (llamacpp.detect, strata.detect, maya.detect, comfyui.detect,
              vllm.detect, sglang.detect, exo.detect, openai_compat.detect)
_NAMED_IDS = ("llamacpp", "ollama", "strata", "maya", "comfyui", "vllm",
              "sglang", "exo")


def _safe_detect(detect, base_url: str) -> EngineStatus:
    try:
        return detect(base_url)
    except Exception as exc:  # a broken probe must not sink the tool call
        return EngineStatus(
            id="unknown", display_name="probe error", available=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _probe_detectors(base_url: str) -> list[EngineStatus]:
    """Every adapter's row at one base url, detector order preserved.

    The per-adapter probes run concurrently (one worker each): an unreachable
    target now costs the slowest single probe (a few seconds) instead of the
    sum of every adapter's dead-connection timeouts. ``executor.map`` keeps
    the result order identical to ``_DETECTORS`` regardless of which probe
    answers first, so row order and detection semantics are unchanged."""
    detectors = _DETECTORS
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(detectors)),
            thread_name_prefix="model-forge-probe") as pool:
        return list(pool.map(lambda d: _safe_detect(d, base_url), detectors))


def _drop_generic_for_named(rows: list[EngineStatus]) -> list[EngineStatus]:
    """Same-server rule: when a distinctive engine already answered at an
    endpoint, the generic openai-compatible row for that endpoint is dropped
    (same server). Duplicate generic rows for one endpoint collapse to the
    first — the same honest claim, reported once."""
    named_endpoints = {e.endpoint for e in rows
                       if e.available and e.id in _NAMED_IDS}
    out: list[EngineStatus] = []
    seen_generic: set[str] = set()
    for e in rows:
        if e.id == "openai-compatible" and e.available:
            if e.endpoint in named_endpoints or e.endpoint in seen_generic:
                continue
            seen_generic.add(e.endpoint)
        out.append(e)
    return out


def detect_engines(base_url: str = "",
                   extra_roots: list[str] | None = None) -> list[EngineStatus]:
    """Probe every adapter at a compute target; never raises.

    ``extra_roots`` is the list of engine HTTP endpoints the user explicitly
    declared. Each distinct declared root is probed with the same
    fingerprint-first logic as ``base_url`` and reported as rows with
    ``source="user-declared"`` (declared rows follow the base rows, in the
    order the roots were declared). With the parameter absent — or an empty
    list — no declared root is probed and the result is byte-identical to the
    single-endpoint behavior."""
    out = _drop_generic_for_named(_probe_detectors(base_url))
    declared: list[EngineStatus] = []
    seen: set[str] = set()
    for raw in (extra_roots or ()):
        root = root_url(raw if isinstance(raw, str) else "")
        if not root or root in seen:
            continue
        seen.add(root)
        rows = _drop_generic_for_named(_probe_detectors(root))
        declared.extend(replace(e, source="user-declared") for e in rows)
    return out + declared
