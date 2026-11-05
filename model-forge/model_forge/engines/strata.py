"""Strata detection and launch (github.com/Niko1221/Strata).

Strata serves OpenAI- and Anthropic-compatible APIs from 127.0.0.1:8080 and
ships a `strata` CLI (`strata launch`, `strata connect`). Launch orchestration
goes through the CLI: model families and pack ids come from `strata launch`'s
own prompts/docs — never invented here.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request

from .base import EngineStatus

ENDPOINT = "http://127.0.0.1:8080/v1/models"
_TIMEOUT_S = 2.0


def served_models() -> list[str]:
    try:
        with urllib.request.urlopen(ENDPOINT, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return [m.get("id", "") for m in data.get("data", [])][:5]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def launch_cmd(family: str, model: str, plan_only: bool = False) -> list[str]:
    """argv for `strata launch` — plan_only prints the memory plan without
    downloading. Callers execute this via the terminal; nothing runs here."""
    if not family or not model:
        raise ValueError("family and model are required (see `strata launch` docs)")
    cmd = ["strata", "launch", "--family", family, "--model", model]
    if plan_only:
        cmd.append("--plan")
    return cmd


def detect() -> EngineStatus:
    cli = shutil.which("strata")
    models = served_models()
    if models:
        return EngineStatus(
            id="strata",
            display_name="Strata",
            available=True,
            endpoint="http://127.0.0.1:8080",
            detail=f"serving {', '.join(models)}"
                   + ("" if cli else " (CLI not on PATH)"),
            modalities=("chat", "vision"),
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
