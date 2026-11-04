"""Strata detection (github.com/Niko1221/Strata).

Strata serves OpenAI- and Anthropic-compatible APIs from 127.0.0.1:8080 and
ships a `strata` CLI (`strata launch`, `strata connect`). Detection only —
install/launch orchestration lands in the next milestone.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request

from .base import EngineStatus

ENDPOINT = "http://127.0.0.1:8080/v1/models"
_TIMEOUT_S = 2.0


def _serving() -> list[str]:
    try:
        with urllib.request.urlopen(ENDPOINT, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return [m.get("id", "") for m in data.get("data", [])][:3]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def detect() -> EngineStatus:
    cli = shutil.which("strata")
    models = _serving()
    if models:
        return EngineStatus(
            id="strata",
            display_name="Strata",
            available=True,
            endpoint="http://127.0.0.1:8080",
            detail=f"serving {', '.join(models)}"
                   + ("" if cli else " (CLI not on PATH)"),
            modalities=("chat", "vision"),
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
