"""Tool handlers — every handler returns a JSON string, even on error."""

from __future__ import annotations

import json

from .engines import detect_engines
from .hardware import probe_hardware
from .registry import (
    MODALITY_LABELS, models_for, fit_verdict,
)


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=1)


def model_forge_status(args: dict | None = None, **kwargs) -> str:
    """Hardware + engine inventory."""
    try:
        hw = probe_hardware()
        engines = detect_engines()
        return _dump({
            "hardware": {
                "platform": hw.platform,
                "gpus": list(hw.gpus),
                "total_vram_gb": hw.total_vram_gb,
                "ram_gb": hw.ram_gb,
            },
            "engines": [
                {
                    "id": e.id,
                    "name": e.display_name,
                    "available": e.available,
                    "endpoint": e.endpoint,
                    "detail": e.detail,
                    "modalities": list(e.modalities),
                }
                for e in engines
            ],
        })
    except Exception as exc:
        return _dump({"error": f"{type(exc).__name__}: {exc}"})


def model_forge_browse(args: dict | None = None, **kwargs) -> str:
    """Models for a modality, priced against this machine (advisory)."""
    args = args or {}
    modality = args.get("modality") or None
    engine = args.get("engine") or None
    try:
        hw = probe_hardware()
        entries = models_for(modality=modality, engine=engine)
        return _dump({
            "hardware_budget_gb": hw.budget_gb,
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "modality": m.modality,
                    "modality_label": MODALITY_LABELS.get(m.modality, m.modality),
                    "engines": list(m.engines),
                    "source": m.source,
                    "size_gb": m.size_gb,
                    "vram_gb_min": m.vram_gb_min,
                    "fit": fit_verdict(m, hw),
                    "status": m.status,
                    "note": m.note,
                }
                for m in entries
            ],
            "note": "Fit verdicts are advisory; engines do authoritative "
                    "memory sizing at launch. GGUF chat models: use Hermes' "
                    "built-in Local Models pane (hardware-priced catalog).",
        })
    except Exception as exc:
        return _dump({"error": f"{type(exc).__name__}: {exc}"})
