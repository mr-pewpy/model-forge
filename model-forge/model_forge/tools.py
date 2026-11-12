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
from .media import media_plan
from .registry import (
    MODALITY_LABELS, PHANTOM_ENGINE_NOTE, PHANTOM_ENGINES, ModelEntry,
    fit_report, fit_verdict, models_for,
)
from .targets import ComputeTarget, add_target, get_target, load_targets

_CHAT_MODALITIES = ("chat", "vision")
_IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_MAX_IMAGE_BYTES = 25 * 1024 * 1024
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


def policy_hook(tool_name: str, args: dict | None = None, task_id: str = "",
                **kwargs) -> dict | None:
    """pre_tool_call policy: routes the two policy decisions through Hermes'
    human-approval gate.

    Registering a compute target decides where engine calls and media jobs
    (including image data) can be sent; rewriting model.default redirects all
    future conversations. Both are the USER'S decisions — the model must not
    make them alone. Escalating with ``action: approve`` shows the user the
    exact URL / config change; denial, timeout, or a gate error fails closed.
    """
    args = args or {}
    if tool_name == "model_forge_target_add":
        return {
            "action": "approve",
            "message": (
                "Model Forge wants to register a compute target. Engine "
                "probes, media jobs (including image data) and downloads "
                "may then be sent to this address:\n"
                f"  id: {args.get('id')}\n"
                f"  label: {args.get('label') or args.get('id')}\n"
                f"  base_url: {args.get('base_url')}"),
            "rule_key": "model-forge:target_add",
        }
    if tool_name == "model_forge_activate" and args.get("apply") is True:
        return {
            "action": "approve",
            "message": (
                "Model Forge wants to rewrite your default model config "
                "(model.default, model.provider, base_url) so future "
                "sessions chat through this endpoint:\n"
                f"  model_id: {args.get('model_id')}\n"
                f"  target: {args.get('target') or 'local'}\n"
                f"  engine_model: {args.get('engine_model') or '(registry '
                f'default)'}"),
            "rule_key": "model-forge:activate",
        }
    return None


def _err(exc: Exception) -> str:
    return _dump({"error": f"{type(exc).__name__}: {exc}"})


def _hw_for_target(target: ComputeTarget):
    if target.kind == "local":
        return probe_hardware()
    from .registry import HardwareProfile
    return HardwareProfile(gpus=(), total_vram_gb=target.vram_gb,
                           ram_gb=target.ram_gb, platform=target.kind)


# --- pure planners -----------------------------------------------------------

def _engine_listing(m: ModelEntry) -> list[str]:
    """Engine ids for a browse row, with honest labeling for phantom engines —
    'python' is never presented as if the plugin could run it."""
    return [f"{e} (needs a {PHANTOM_ENGINE_NOTE})" if e in PHANTOM_ENGINES
            else e for e in m.engines]


def _select_engine(entry: ModelEntry, requested: str,
                   available_engines: list[str] | None) -> tuple[str, str]:
    """(engine, honest note) for an install plan.

    An explicit override wins; otherwise the first of entry.engines detected
    on the target; otherwise entry.engines[0] with a fallback note when
    detection ran and found nothing. Phantom engines are never auto-selected.
    """
    requested = (requested or "").strip()
    detected = available_engines is not None
    if requested:
        if detected and requested not in available_engines:
            return requested, (f"caller-specified engine '{requested}' was "
                               "not detected on this target — install and "
                               "serve it first")
        return requested, ""
    for e in entry.engines:
        if e not in PHANTOM_ENGINES and available_engines \
                and e in available_engines:
            return e, ""
    if entry.engines:
        if detected:
            return entry.engines[0], (
                f"none of {list(entry.engines)} was detected on this target; "
                f"falling back to '{entry.engines[0]}' — install and serve it "
                "first")
        return entry.engines[0], ""
    return "", ""


def install_plan(entry: ModelEntry, target: ComputeTarget,
                 engine_model: str = "", engine: str = "",
                 available_engines: list[str] | None = None) -> dict:
    """What model_forge_install would do — no side effects (unit-tested).

    Engine selection: explicit `engine` override (validated against the
    entry), else the first of entry.engines detected on the target, else
    engines[0] with an honest fallback note when detection found nothing."""
    if entry.status == "stub":
        return {"ok": False, "action": "unavailable", "engine": "",
                "engine_note": "",
                "detail": "no engine exists for this modality yet"}
    requested = (engine or "").strip()
    if requested and requested not in entry.engines:
        return {"ok": False, "action": "engine_mismatch", "engine": requested,
                "engine_note": "",
                "detail": f"engine '{requested}' cannot serve {entry.id}; "
                          f"this model runs on {list(entry.engines)}"}
    chosen, engine_note = _select_engine(entry, requested, available_engines)

    def _plan(**kw):
        kw.setdefault("engine", chosen)
        kw.setdefault("engine_note", engine_note)
        return kw

    engine_model = engine_model or entry.engine_models.get(chosen, "")
    if entry.id == "core-gguf-catalog" or chosen == "llamacpp":
        return _plan(ok=True, action="use_core_local_models",
                     detail="GGUF chat models are priced and installed by "
                            "Hermes' built-in Local Models pane (Settings -> "
                            "Providers -> Local Models, or `hermes model`); "
                            "Model Forge does not duplicate it")
    if chosen == "maya":
        if not engine_model:
            return _plan(ok=False, action="needs_engine_model",
                         detail="pass engine_model=<Maya setup id>, e.g. "
                                "glm-maya-s (see Maya docs/MODELS.md)")
        return _plan(ok=True, action="maya_pull",
                     model=engine_model,
                     endpoint=root_url(target.base_url) + "/api/pull",
                     poll=root_url(target.base_url) + "/api/pull/status",
                     detail="POST /api/pull {name, detach:true}; poll "
                            "/api/pull/status for byte progress")
    if chosen == "strata":
        if not engine_model or ":" not in engine_model:
            return _plan(ok=False, action="needs_engine_model",
                         detail="pass engine_model=<family>:<pack> (from "
                                "`strata launch` docs), e.g. deepseek41:"
                                "ist-das-2.00bpw")
        family, _, pack = engine_model.partition(":")
        return _plan(ok=True, action="run_cli_on_target",
                     command=strata_adapter.launch_cmd(family, pack),
                     detail=f"run this ON {target.label} ({target.base_url}); "
                            "add --plan first to see the memory plan")
    if chosen == "python":
        return _plan(ok=False, action="custom_env_unmanaged",
                     detail=f"'{entry.name}' needs a {PHANTOM_ENGINE_NOTE} — "
                            "Model Forge cannot install or serve it. Get the "
                            f"weights from {entry.source or 'the model card'} "
                            "and run them in that environment yourself.")
    if chosen in ("comfyui", "vllm", "sglang"):
        if target.kind == "local" and entry.source:
            if not engine_model:
                return _plan(ok=False, action="needs_engine_model",
                             detail="pass engine_model=<file path inside the "
                                    "source repo> (e.g. model.safetensors), "
                                    "and optionally dest=<absolute path>")
            fname = engine_model.rstrip("/").rsplit("/", 1)[-1]
            return _plan(ok=True, action="hf_download",
                         url=f"https://huggingface.co/{entry.source}"
                             f"/resolve/main/{engine_model}",
                         dest_default=f"downloads/{entry.id}/{fname}",
                         detail="plugin-side download with a progress job "
                                "(credentials applied per host); pass dest to "
                                "land in an engine models dir directly, then "
                                "link/move into the right subdir "
                                "(ComfyUI: models/diffusion_models etc.)")
        cmd = ["hf", "download", entry.source]
        if engine_model:
            cmd.append(engine_model.rstrip("/"))
        else:
            cmd += ["--include", "*"]
        return _plan(ok=True, action="run_on_target",
                     command=cmd,
                     detail=f"run on {target.label} and place the files in the "
                            f"engine's models dir there; source: {entry.source}")
    return _plan(ok=False, action="adapter_pending",
                 detail=f"engine '{chosen}' install orchestration lands with "
                        "its adapter; model source: " + entry.source)


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
        base = "" if target.kind == "local" else target.base_url
        available = [e.id for e in detect_engines(base) if e.available]
        entries = models_for(modality=args.get("modality") or None,
                             engine=args.get("engine") or None)
        out = {
            "target": target.id,
            "hardware_budget_gb": hw.budget_gb,
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "modality": m.modality,
                    "modality_label": MODALITY_LABELS.get(m.modality, m.modality),
                    "engines": _engine_listing(m),
                    "source": m.source,
                    "engine_models": m.engine_models,
                    "size_gb": m.size_gb,
                    "vram_gb_min": m.vram_gb_min,
                    "fit": fit_report(m, hw, available),
                    "status": m.status,
                    "note": m.note,
                }
                for m in entries
            ],
            "note": "Fit verdicts are advisory and cross-check the engines "
                    "detected on the target (see each row's fit block); "
                    "engines do authoritative memory sizing at launch. GGUF "
                    "chat models: use Hermes' built-in Local Models pane "
                    "(hardware-priced catalog).",
        }
        out.update(_fresh_section(args, entries))
        return _dump(out)
    except Exception as exc:
        return _err(exc)


def _fresh_section(args: dict, entries: list[ModelEntry]) -> dict:
    """Live HuggingFace freshness block for browse: newest-first rows with
    dates, honest 'fresh/unvetted' labeling, fail-soft to an unavailable
    marker. Pass fresh=false for curated-only output (no network call)."""
    if args.get("fresh") is False:
        return {
            "live_discovery": "disabled",
            "fresh": [],
            "fresh_note": "Live discovery skipped (fresh=false); "
                          "curated catalog only.",
        }
    from . import discovery
    res = discovery.fresh_models(args.get("modality") or None)
    if res.get("live_discovery") != "ok":
        return {
            "live_discovery": res.get("live_discovery", "unavailable"),
            "fresh": [],
            "fresh_note": "Live HuggingFace discovery unavailable "
                          f"({res.get('error') or 'no network'}); "
                          "showing the curated catalog only.",
        }
    curated_sources = {m.source for m in entries if m.source}
    return {
        "live_discovery": "ok",
        "fresh": [r for r in res["models"] if r["id"] not in curated_sources],
        "fresh_note": "Fresh = live HuggingFace results, newest first "
                      "(unvetted: no fit estimate, no engine mapping). "
                      "Curated entries above are hand-maintained and "
                      "fit-priced.",
    }


def model_forge_target_add(args: dict | None = None, **kwargs) -> str:
    """Register a compute target (this machine, a remote host, a cloud GPU box)."""
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
                                   "http://gpu-box.example.com:8080"})
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
    """Pull a model on a compute target (Maya: HTTP pull; HF download: job;
    Strata: CLI). Pass job_id to poll a download."""
    args = args or {}
    try:
        from . import pull
        if args.get("job_id"):
            return _dump(pull.job_status(str(args["job_id"])))
        entry = next((m for m in models_for()
                      if m.id == args.get("model_id")), None)
        if entry is None:
            return _dump({"error": f"unknown model_id: {args.get('model_id')}"})
        target = get_target(args.get("target"))
        if target is None:
            return _dump({"error": f"unknown target: {args.get('target')}"})
        base = "" if target.kind == "local" else target.base_url
        available = [e.id for e in detect_engines(base) if e.available]
        plan = install_plan(entry, target,
                            engine_model=str(args.get("engine_model", "")),
                            engine=str(args.get("engine", "")),
                            available_engines=available)
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
        elif plan.get("action") == "hf_download":
            dest = str(args.get("dest") or "")
            if not dest:
                from plugins.plugin_storage import plugin_data_dir
                dest = str(plugin_data_dir("model-forge")
                           / plan["dest_default"])
            started = pull.start_download(plan["url"], dest,
                                          sha256=str(args.get("sha256", "")))
            plan.update(started)
            plan["dest"] = dest
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
        statuses = detect_engines(base)
        available = [e.id for e in statuses if e.available]
        plan = activation_plan(entry, target, available,
                               engine_model=str(args.get("engine_model", "")))
        if plan.get("ok"):
            mods = {m for e in statuses if e.available for m in e.modalities}
            base_v1 = root_url(target.base_url) + "/v1"
            media = {}
            if "image" in mods:
                media["image_gen"] = (
                    "hermes config set plugins.entries.model-forge-image"
                    f".settings.base_url {base_v1} && hermes config set "
                    "image_gen.provider model-forge")
            if "video" in mods:
                media["video_gen"] = (
                    "hermes config set plugins.entries.model-forge-video"
                    f".settings.base_url {base_v1} && hermes config set "
                    "video_gen.provider model-forge")
            if media:
                plan["media_setup"] = media
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


def model_forge_generate_media(args: dict | None = None, **kwargs) -> str:
    """Generate 3D models / music / SFX through a compute target's engine
    API. 3d submits a job (poll with job_id); music/sfx return audio."""
    args = args or {}
    try:
        import json as _json
        import urllib.request
        modality = str(args.get("modality", ""))
        target = get_target(args.get("target"))
        if target is None:
            return _dump({"error": f"unknown target: {args.get('target')}"})
        plan = media_plan(modality, target.base_url)
        if not plan.get("ok"):
            return _dump(plan)
        root = root_url(target.base_url)
        if modality == "3d":
            if args.get("job_id"):
                req = urllib.request.Request(
                    f"{root}/jobs/{args['job_id']}")
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    return _dump({"job": _json.loads(
                        resp.read().decode("utf-8", "replace"))})
            model = str(args.get("model", "")).strip()
            image = str(args.get("image", "")).strip()
            if not model or not image:
                return _dump({"ok": False, "action": "needs_args",
                              "detail": "3d needs model (e.g. gen/3d) and "
                                        "image (URL or local file path)"})
            try:
                image_ref = _as_data_url(image)
            except ValueError as exc:
                return _dump({"ok": False, "action": "invalid_image",
                              "error": str(exc)})
            body = {"model": model, "image": image_ref}
            if args.get("prompt"):
                body["prompt"] = str(args["prompt"])
            req = urllib.request.Request(
                plan["submit"], data=_json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=60.0) as resp:
                    out = _json.loads(resp.read().decode("utf-8", "replace"))
            except Exception as exc:  # noqa: BLE001
                return _dump({"ok": False, "action": "submit_failed",
                              "error": f"{type(exc).__name__}: {exc}"})
            return _dump({"ok": True, "action": "submitted",
                          "submitted": out,
                          "poll": f"model_forge_generate_media with "
                                  f"modality=3d, target={target.id}, "
                                  f"job_id={out.get('id', '')}"})
        # music / sfx: sync POST returning audio
        model = str(args.get("model", "")).strip()
        prompt = str(args.get("prompt", "")).strip()
        if not model or not prompt:
            return _dump({"ok": False, "action": "needs_args",
                          "detail": f"{modality} needs model and prompt"})
        body = {"model": model, "prompt": prompt}
        req = urllib.request.Request(
            plan["submit"], data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600.0) as resp:
                out = _json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            return _dump({"ok": False, "action": "submit_failed",
                          "error": f"{type(exc).__name__}: {exc}"})
        return _dump({"ok": True, "action": "done", "result": out})
    except Exception as exc:
        return _err(exc)


def _as_data_url(value: str) -> str:
    """Local image path -> data: URL (engines on other hosts need bytes).

    Model-supplied values are confined: URLs pass through, local paths must
    pass the Hermes read-block check and look like a real image file (suffix,
    magic bytes, size cap). Anything else raises ValueError instead of being
    passed through raw — a raw path could otherwise read any file the user
    can read and upload it to the engine host.
    """
    if value.startswith(("data:", "http://", "https://")):
        return value
    if value.startswith("file://"):
        value = value[len("file://"):]
    from agent.file_safety import raise_if_read_blocked
    raise_if_read_blocked(value)
    import base64
    from pathlib import Path
    p = Path(value)
    if not p.is_file():
        raise ValueError(
            f"image must be a URL, a data: URL, or an existing local image "
            f"file: {value!r}")
    suffix = p.suffix.lower()
    mime = _IMAGE_MIME_BY_SUFFIX.get(suffix)
    if mime is None:
        raise ValueError(
            f"unsupported image type {suffix!r} (allowed: "
            f"{sorted(_IMAGE_MIME_BY_SUFFIX)})")
    size = p.stat().st_size
    if size > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"image too large: {size} bytes (max {_MAX_IMAGE_BYTES})")
    head = p.open("rb").read(16)
    if not (head.startswith(b"\x89PNG\r\n\x1a\n")
            or head.startswith(b"\xff\xd8\xff")
            or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")):
        raise ValueError(
            "image content does not match its extension (magic bytes)")
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
