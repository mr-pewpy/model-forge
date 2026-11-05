"""Tool handlers — every handler returns a JSON string, even on error.

Handlers wrap pure planners (install_plan / activation_plan) so the routing
logic is unit-tested without touching the network or the Hermes config.
"""

from __future__ import annotations

import json

from .credentials import status as credentials_status
from .engines import detect_engines
from .engines import maya as maya_adapter
from .engines import strata as strata_adapter
from .engines.base import root_url
from .hardware import probe_hardware
from .registry import MODALITY_LABELS, ModelEntry, fit_verdict, models_for
from .targets import ComputeTarget, add_target, get_target, load_targets

_CHAT_MODALITIES = ("chat", "vision")
_PROVIDER_FOR_ENGINE = {
    "strata": "strata",
    "maya": "maya",
    "llamacpp": "llamacpp",
    "ollama": "custom",
    "vllm": "custom",
    "openai-compatible": "custom",
}


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=1)


def _err(exc: Exception) -> str:
    return _dump({"error": f"{type(exc).__name__}: {exc}"})


def _hw_for_target(target: ComputeTarget):
    if target.kind == "local":
        return probe_hardware()
    from .registry import HardwareProfile
    return HardwareProfile(gpus=(), total_vram_gb=target.vram_gb,
                           ram_gb=target.ram_gb, platform=target.kind)


# --- pure planners -----------------------------------------------------------

def install_plan(entry: ModelEntry, target: ComputeTarget,
                 engine_model: str = "") -> dict:
    """What model_forge_install would do — no side effects (unit-tested)."""
    engine = entry.engines[0] if entry.engines else ""
    engine_model = engine_model or entry.engine_models.get(engine, "")
    if entry.status == "stub":
        return {"ok": False, "action": "unavailable",
                "detail": "no engine exists for this modality yet"}
    if entry.id == "core-gguf-catalog" or engine == "llamacpp":
        return {"ok": True, "action": "use_core_local_models",
                "detail": "GGUF chat models are priced and installed by "
                          "Hermes' built-in Local Models pane (Settings -> "
                          "Providers -> Local Models, or `hermes model`); "
                          "Model Forge does not duplicate it"}
    if engine == "maya":
        if not engine_model:
            return {"ok": False, "action": "needs_engine_model",
                    "detail": "pass engine_model=<Maya setup id>, e.g. "
                              "glm-maya-s (see Maya docs/MODELS.md)"}
        return {"ok": True, "action": "maya_pull",
                "model": engine_model,
                "endpoint": root_url(target.base_url) + "/api/pull",
                "poll": root_url(target.base_url) + "/api/pull/status",
                "detail": "POST /api/pull {name, detach:true}; poll "
                          "/api/pull/status for byte progress"}
    if engine == "strata":
        if not engine_model or ":" not in engine_model:
            return {"ok": False, "action": "needs_engine_model",
                    "detail": "pass engine_model=<family>:<pack> (from "
                              "`strata launch` docs), e.g. deepseek41:"
                              "ist-das-2.00bpw"}
        family, _, pack = engine_model.partition(":")
        return {"ok": True, "action": "run_cli_on_target",
                "command": strata_adapter.launch_cmd(family, pack),
                "detail": f"run this ON {target.label} ({target.base_url}); "
                          "add --plan first to see the memory plan"}
    return {"ok": False, "action": "adapter_pending",
            "detail": f"engine '{engine}' install orchestration lands with "
                      "its adapter; model source: " + entry.source}


def activation_plan(entry: ModelEntry, target: ComputeTarget,
                    available_engines: list[str], engine_model: str = "") -> dict:
    """What model_forge_activate would do — no side effects (unit-tested)."""
    if entry.modality not in _CHAT_MODALITIES:
        return {"ok": False, "action": "not_a_chat_model",
                "detail": f"{entry.modality} models activate through their "
                          "own provider surface (image_gen/video_gen), not "
                          "as the chat model"}
    engine = next((e for e in entry.engines if e in available_engines), "")
    if not engine and "openai-compatible" in available_engines:
        # A generic endpoint can serve any of this entry's engines (remote
        # Strata, vLLM, a broker); connect it as a custom endpoint.
        engine = "openai-compatible"
    if not engine:
        return {"ok": False, "action": "engine_not_running",
                "detail": f"none of {list(entry.engines)} answered at "
                          f"{target.base_url} — install/serve first"}
    provider = _PROVIDER_FOR_ENGINE.get(engine, "custom")
    model_id = engine_model or entry.engine_models.get(engine) or entry.id
    return {
        "ok": True,
        "action": "persist_model_selection",
        "provider": provider,
        "model": model_id,
        "base_url": root_url(target.base_url) + "/v1",
        "detail": "writes model.default + model.provider (+ base_url) via "
                  "the documented hermes_cli.model_setup helper; takes "
                  "effect on your next session",
    }


# --- tools -------------------------------------------------------------------

def model_forge_status(args: dict | None = None, **kwargs) -> str:
    """Hardware + engine inventory across all compute targets."""
    try:
        targets_out = []
        for t in load_targets():
            hw = _hw_for_target(t)
            targets_out.append({
                "id": t.id,
                "label": t.label,
                "kind": t.kind,
                "base_url": t.base_url,
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
                        "models_loaded": list(e.models_loaded),
                    }
                    for e in detect_engines("" if t.kind == "local" else t.base_url)
                ],
            })
        return _dump({"targets": targets_out,
                      "pull_credentials": credentials_status()})
    except Exception as exc:
        return _err(exc)


def model_forge_browse(args: dict | None = None, **kwargs) -> str:
    """Models for a modality, priced against a compute target (advisory)."""
    args = args or {}
    try:
        target = get_target(args.get("target"))
        if target is None:
            return _dump({"error": f"unknown target: {args.get('target')}"})
        hw = _hw_for_target(target)
        entries = models_for(modality=args.get("modality") or None,
                             engine=args.get("engine") or None)
        return _dump({
            "target": target.id,
            "hardware_budget_gb": hw.budget_gb,
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "modality": m.modality,
                    "modality_label": MODALITY_LABELS.get(m.modality, m.modality),
                    "engines": list(m.engines),
                    "source": m.source,
                    "engine_models": m.engine_models,
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
        return _err(exc)


def model_forge_target_add(args: dict | None = None, **kwargs) -> str:
    """Register a compute target (this machine, a LAN LXC, a cloud box)."""
    args = args or {}
    try:
        tid = str(args.get("id", "")).strip()
        label = str(args.get("label", "")).strip() or tid
        base_url = str(args.get("base_url", "")).strip()
        kind = str(args.get("kind", "")).strip() or "lan"
        if not tid or tid == "local":
            return _dump({"error": "id is required and cannot be 'local'"})
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            return _dump({"error": "base_url must be http(s), e.g. "
                                   "http://192.168.1.62:8080"})
        t = ComputeTarget(
            id=tid, label=label, base_url=base_url, kind=kind,
            vram_gb=float(args.get("vram_gb") or 0.0),
            ram_gb=float(args.get("ram_gb") or 0.0),
            note=str(args.get("note", "")),
        )
        add_target(t)
        return _dump({
            "added": t.id,
            "targets": [
                {"id": x.id, "label": x.label, "kind": x.kind,
                 "base_url": x.base_url}
                for x in load_targets()
            ],
        })
    except Exception as exc:
        return _err(exc)


def model_forge_install(args: dict | None = None, **kwargs) -> str:
    """Pull a model on a compute target (Maya: HTTP pull; Strata: CLI)."""
    args = args or {}
    try:
        entry = next((m for m in models_for()
                      if m.id == args.get("model_id")), None)
        if entry is None:
            return _dump({"error": f"unknown model_id: {args.get('model_id')}"})
        target = get_target(args.get("target"))
        if target is None:
            return _dump({"error": f"unknown target: {args.get('target')}"})
        plan = install_plan(entry, target,
                            engine_model=str(args.get("engine_model", "")))
        if plan.get("action") == "maya_pull":
            result = maya_adapter.pull(plan["model"], detach=True,
                                       base_url=target.base_url)
            if result is None:
                plan["ok"] = False
                plan["error"] = ("no response from "
                                 f"{plan['endpoint']} — is Maya running "
                                 "on that target?")
            else:
                plan["pull_response"] = result
        return _dump(plan)
    except Exception as exc:
        return _err(exc)


def model_forge_activate(args: dict | None = None, **kwargs) -> str:
    """Connect a running chat/vision model as THIS Hermes' default model.

    Safe by default: pass apply=true to actually write model.* config
    (takes effect next session; revert with `hermes model`)."""
    args = args or {}
    try:
        entry = next((m for m in models_for()
                      if m.id == args.get("model_id")), None)
        if entry is None:
            return _dump({"error": f"unknown model_id: {args.get('model_id')}"})
        target = get_target(args.get("target"))
        if target is None:
            return _dump({"error": f"unknown target: {args.get('target')}"})
        base = "" if target.kind == "local" else target.base_url
        available = [e.id for e in detect_engines(base) if e.available]
        plan = activation_plan(entry, target, available,
                               engine_model=str(args.get("engine_model", "")))
        if plan.get("ok") and args.get("apply") is True:
            try:
                from hermes_cli.model_setup import persist_model_selection
                saved = persist_model_selection(
                    plan["model"], plan["provider"],
                    base_url=plan["base_url"])
                plan["applied"] = True
                plan["saved_model_section"] = saved
            except Exception as exc:
                plan["applied"] = False
                plan["error"] = f"{type(exc).__name__}: {exc}"
        else:
            plan["applied"] = False
            plan["hint"] = "re-run with apply=true to write the config"
        return _dump(plan)
    except Exception as exc:
        return _err(exc)
