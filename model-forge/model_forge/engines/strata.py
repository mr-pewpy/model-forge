"""Strata detection and launch (github.com/Niko1221/Strata).

Strata serves OpenAI- and Anthropic-compatible APIs from 127.0.0.1:8080 and
deliberately mirrors much of llama-server's surface (/props, /slots, /metrics),
so a generic /v1/models answer NEVER identifies it — many engines serve that
route. "Strata" is claimed only on a Strata-specific fingerprint:

1. GET /v1/telemetry — the authoritative read-only snapshot behind
   `strata status` (schema v1: sampled_at / model / instance / work / quality);
2. on older servers, GET /v1/status — documented to expose `loaded` and
   `auto_load` (the `strata status` fallback chain reads the same routes).

Without one of those fingerprints, a server answering /v1/models on the
default port is reported as the generic OpenAI-compatible endpoint it
honestly is ("identity unknown"), with an id no registry entry lists — so
the row can never drive install engine auto-selection (the same no-mislabel
rule as engines/openai_compat.py). Strata's presence on a machine is shown
only by its own CLI on PATH.

Launch orchestration goes through the CLI: model families and pack ids come
from `strata launch`'s own prompts/docs — never invented here.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request

from .base import EngineStatus, root_url

DEFAULT_ROOT = "http://127.0.0.1:8080"
_DEFAULT_PORT = DEFAULT_ROOT.rsplit(":", 1)[-1]
_TIMEOUT_S = 2.0

# Top-level markers of the `strata status` telemetry schema v1 (see
# docs/TERMINAL_TELEMETRY_FIELDS.md upstream).
_TELEMETRY_MARKERS = ("sampled_at", "instance", "work", "quality")


def _get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def served_models(base_url: str = "") -> list[str]:
    root = root_url(base_url) or DEFAULT_ROOT
    try:
        with urllib.request.urlopen(f"{root}/v1/models", timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return [m.get("id", "") for m in data.get("data", [])][:5]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def strata_fingerprint(base_url: str = "") -> bool:
    """True only on a Strata-specific surface — never from generic /v1/models.

    Primary: GET /v1/telemetry must look like the schema-v1 snapshot
    (`strata status`'s authoritative read-only response). Fallback for older
    servers: GET /v1/status must expose the documented `loaded` state. Any
    other response — including any plain /v1/models list — is not proof."""
    root = root_url(base_url) or DEFAULT_ROOT
    telemetry = _get_json(f"{root}/v1/telemetry")
    if isinstance(telemetry, dict) and "model" in telemetry \
            and any(k in telemetry for k in _TELEMETRY_MARKERS):
        return True
    status = _get_json(f"{root}/v1/status")
    if isinstance(status, dict) and "loaded" in status \
            and ("auto_load" in status or "model" in status):
        return True
    return False


def launch_cmd(family: str, model: str, plan_only: bool = False) -> list[str]:
    """argv for `strata launch` — plan_only prints the memory plan without
    downloading. Callers execute this via the terminal; nothing runs here."""
    if not family or not model:
        raise ValueError("family and model are required (see `strata launch` docs)")
    cmd = ["strata", "launch", "--family", family, "--model", model]
    if plan_only:
        cmd.append("--plan")
    return cmd


def detect(base_url: str = "") -> EngineStatus:
    cli = shutil.which("strata")
    local = not base_url
    root = root_url(base_url) or DEFAULT_ROOT
    if not local:
        # Remote: /v1/models is served by many engines — never claim "Strata"
        # from it (openai_compat covers the endpoint generically).
        return EngineStatus(
            id="strata",
            display_name="Strata",
            available=False,
            detail=f"Strata is identified on the local machine only (via its "
                   f"CLI); {root} shows up as a generic openai-compatible "
                   "endpoint if it answers /v1/models",
            modalities=("chat", "vision"),
        )
    models = served_models(base_url)
    if models and strata_fingerprint(base_url):
        return EngineStatus(
            id="strata",
            display_name="Strata",
            available=True,
            endpoint=root,
            detail=f"serving {', '.join(models)}"
                   + ("" if cli else " (CLI not on PATH)"),
            modalities=("chat", "vision"),
            models_loaded=tuple(m for m in models if m),
        )
    if models:
        # A generic OpenAI-compatible server on the default port. Never label
        # it "Strata" (or any engine) from /v1/models alone; report the honest
        # generic row instead. Its id is deliberately "openai-compatible" —
        # no registry entry lists that id, so this row can never be
        # auto-selected for install (openai_compat's no-mislabel rule).
        return EngineStatus(
            id="openai-compatible",
            display_name="OpenAI-compatible endpoint",
            available=True,
            endpoint=root,
            detail=(f"generic OpenAI-compatible endpoint on port "
                    f"{_DEFAULT_PORT}, identity unknown — serving "
                    f"{', '.join(models)}; /v1/models never identifies the "
                    "engine (could be llama.cpp, vLLM, Strata, a broker, ...)"
                    + ("; the `strata` CLI is on PATH — `strata status` "
                       "confirms a running Strata server" if cli else "")),
            modalities=("chat", "vision", "ocr"),
            models_loaded=tuple(m for m in models if m),
        )
    return EngineStatus(
        id="strata",
        display_name="Strata",
        available=bool(cli),
        detail=("CLI found; run `strata launch` to start a model"
                if cli else
                "not installed (curl -fsSL https://github.com/Niko1221/Strata/"
                "releases/latest/download/install.sh | sh)"),
        modalities=("chat", "vision"),
    )
