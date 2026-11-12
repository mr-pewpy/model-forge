"""Tool handlers — every handler returns a JSON string, even on error.

Handlers wrap pure planners (install_plan / activation_plan) so the routing
logic is unit-tested without touching the network or the Hermes config.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import uuid

from .credentials import status as credentials_status
from .engines import detect_engines
from .engines import maya as maya_adapter
from .engines import strata as strata_adapter
from .engines.base import route_exists, root_url
from .hardware import probe_hardware
from .media import media_plan
from .registry import (
    MODALITIES, MODALITY_LABELS, PHANTOM_ENGINE_NOTE, PHANTOM_ENGINES,
    ModelEntry, fit_report, fit_verdict, models_for,
)
from .targets import ComputeTarget, add_target, get_target, load_targets

_CHAT_MODALITIES = ("chat", "vision")
# model_forge_generate_media joins a model-supplied job id into a /jobs/ URL
# path on the compute target: only these characters, this length, may pass.
_JOB_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
# Engines that install by pulling a file out of a source repo on this machine.
_FILE_PULL_ENGINES = ("comfyui", "vllm", "sglang")
# What to tell the caller to pass to engine_model when a repo file must be
# named by hand.
_FILE_PULL_HINT = (
    "a file path inside the repo, e.g. "
    "diffusion_models/your-weights.safetensors; omit engine_model and "
    "Model Forge resolves it from the repo")
# Registry ids that are catalog bookkeeping, not servable model ids.
_CATALOG_MARKER_IDS = frozenset({"core-gguf-catalog"})
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".ckpt", ".pt", ".bin")
_WEIGHT_SUFFIX_RANK = {s: i for i, s in enumerate(_WEIGHT_SUFFIXES)}
# Paths that hold auxiliary components, never the primary weights file.
_COMPONENT_DIR_NAMES = ("vae", "clip")
_COMPONENT_NAME_FRAGMENTS = ("tokenizer", "processor", "config",
                             "model_index.json")
_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.\w+$")
# Files at or above this size count as "large" for multi-component detection.
_LARGE_FILE_BYTES = 100 * 1024 * 1024
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
    """pre_tool_call policy: routes the three policy decisions through
    Hermes' human-approval gate.

    Registering a compute target decides where engine calls and media jobs
    (including image data) can be sent; rewriting model.default redirects all
    future conversations; deleting a model-storage entry destroys model
    weights on this machine. All three are the USER'S decisions — the model
    must not make them alone. Escalating with ``action: approve`` shows the
    user the exact URL / config change / path and size; denial, timeout, or
    a gate error fails closed (the call is blocked before the handler runs).
    """
    args = args or {}
    if tool_name == "model_forge_storage" and args.get("action") == "delete":
        from . import storage
        ref = str(args.get("entry") or "")
        entry = storage.describe_entry(ref)
        if entry is not None:
            target = (f"  path: {entry['path']}\n"
                      f"  size: {entry['size_bytes']} bytes")
        else:
            target = (f"  path: {ref or '(no entry given)'}\n"
                      f"  size: unknown (not an entry from the storage "
                      f"list; the delete will be refused)")
        return {
            "action": "approve",
            "message": (
                "Model Forge wants to permanently delete a model-storage "
                "entry from this machine:\n" + target),
            "rule_key": "model-forge:storage_delete",
        }
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
    on the target; otherwise entry.engines[0] as the planned fallback when
    detection ran and found nothing. The auto-match never selects a phantom
    engine (the plugin cannot run one), but the fallback can land on a
    phantom when entry.engines[0] is one — the returned note and the install
    payload then say plainly that the plugin cannot install or serve it,
    instead of telling the caller to install and serve it here.
    """
    requested = (requested or "").strip()
    detected = available_engines is not None
    if requested:
        if requested in PHANTOM_ENGINES:
            return requested, (f"'{requested}' needs a "
                               f"{PHANTOM_ENGINE_NOTE} — Model Forge cannot "
                               "install or serve it")
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
        fallback = entry.engines[0]
        if detected:
            if fallback in PHANTOM_ENGINES:
                return fallback, (
                    f"'{fallback}' needs a {PHANTOM_ENGINE_NOTE} — Model "
                    "Forge cannot install or serve it; none of "
                    f"{list(entry.engines)} was detected on this target. Run "
                    "the weights in that environment yourself, or register a "
                    "target that runs a supported engine")
            return fallback, (
                "no matching engine detected on this target — install one or "
                "register a target that runs it (planning for the listed "
                f"engine '{fallback}')")
        return fallback, ""
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
                             detail="pass engine_model=" + _FILE_PULL_HINT +
                                    "; optionally pass dest=<absolute path>")
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
    if entry.id in _CATALOG_MARKER_IDS:
        return {"ok": False, "action": "placeholder_not_activatable",
                "detail": f"'{entry.id}' is a catalog marker for Hermes' "
                          "built-in Local Models catalog, not a model id a "
                          "running engine serves — it cannot be written to "
                          "model.default. Activate a concrete served model "
                          "instead (GGUF chat models are managed in the "
                          "built-in Local Models pane)"}
    if entry.status == "stub":
        return {"ok": False, "action": "placeholder_not_activatable",
                "detail": f"'{entry.id}' is a placeholder entry — no engine "
                          "serves this modality yet, so it cannot become the "
                          "default model. Activate a concrete served model"}
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
        modality = args.get("modality") or None
        if modality is not None and modality not in MODALITIES:
            return _dump({"error": f"unknown modality: {modality!r}; valid "
                                   f"modalities: {list(MODALITIES)}"})
        target = get_target(args.get("target"))
        if target is None:
            return _dump({"error": f"unknown target: {args.get('target')}"})
        hw = _hw_for_target(target)
        base = "" if target.kind == "local" else target.base_url
        available = [e.id for e in detect_engines(base) if e.available]
        entries = models_for(modality=modality,
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


def _fresh_disabled(value) -> bool:
    """True for an explicit opt-out of the live layer: false, 'false', '0'
    or 'no' (any case, whitespace tolerated). A bare string must not silently
    re-enable the network call."""
    if value is False:
        return True
    return isinstance(value, str) \
        and value.strip().lower() in ("false", "0", "no")


def _fresh_section(args: dict, entries: list[ModelEntry]) -> dict:
    """Live HuggingFace freshness block for browse: newest-first rows with
    dates, honest 'fresh/unvetted' labeling, fail-soft to an unavailable
    marker. Pass fresh=false for curated-only output (no network call)."""
    if _fresh_disabled(args.get("fresh")):
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
        model_id = str(args.get("model_id") or "").strip()
        if not model_id:
            return _dump({"error": "model_id required (or job_id to poll a "
                                   "download)"})
        entry = next((m for m in models_for() if m.id == model_id), None)
        if entry is None:
            return _dump({"error": f"unknown model_id: {model_id}"})
        target = get_target(args.get("target"))
        if target is None:
            return _dump({"error": f"unknown target: {args.get('target')}"})
        base = "" if target.kind == "local" else target.base_url
        available = [e.id for e in detect_engines(base) if e.available]
        engine_model = str(args.get("engine_model") or "")
        engine_arg = str(args.get("engine", ""))
        plan = install_plan(entry, target, engine_model=engine_model,
                            engine=engine_arg, available_engines=available)
        if (plan.get("action") == "needs_engine_model"
                and not engine_model.strip()
                and target.kind == "local"
                and entry.source
                and plan.get("engine") in _FILE_PULL_ENGINES):
            # No file named: resolve the primary weights file from the repo's
            # file list instead of making the caller guess a path.
            resolved = resolve_repo_weights(entry.source)
            if resolved.get("ok"):
                plan = install_plan(entry, target,
                                    engine_model=resolved["file"],
                                    engine=engine_arg,
                                    available_engines=available)
                plan["resolved_file"] = resolved["file"]
                plan["engine_model_source"] = (
                    f"auto-resolved from the {entry.source} file list on "
                    "Hugging Face")
            else:
                plan.update({k: v for k, v in resolved.items()
                             if k not in ("engine", "engine_note")})
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
        model_id = str(args.get("model_id") or "").strip()
        if not model_id:
            return _dump({"error": "model_id required"})
        entry = next((m for m in models_for() if m.id == model_id), None)
        if entry is None:
            return _dump({"error": f"unknown model_id: {model_id}"})
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
            root = root_url(target.base_url)
            base_v1 = root + "/v1"
            media = {}
            withheld = []
            for modality, key, route, plugin in (
                    ("image", "image_gen", "/v1/images/generations",
                     "model-forge-image"),
                    ("video", "video_gen", "/v1/videos/generations",
                     "model-forge-video")):
                if modality not in mods:
                    continue
                if not route_exists(root, route):
                    # An engine can list a modality it does not actually
                    # serve on an OpenAI-compatible route (ComfyUI lists
                    # image/video/3d but serves none of these endpoints),
                    # so the setup hint is emitted only after the route
                    # answers a probe.
                    withheld.append(f"{key} -> {root}{route}")
                    continue
                media[key] = (
                    f"hermes config set plugins.entries.{plugin}"
                    f".settings.base_url {base_v1} && hermes config set "
                    f"{key}.provider model-forge")
            if media:
                plan["media_setup"] = media
            if withheld:
                plan["media_setup_note"] = (
                    "an engine on this target lists these modalities but the "
                    "endpoint did not answer the matching route, so the "
                    "setup hint is withheld (run it manually only if your "
                    "endpoint serves that route): " + "; ".join(withheld))
        if plan.get("ok"):
            if args.get("apply") is True:
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
        else:
            plan["applied"] = False
        return _dump(plan)
    except Exception as exc:
        return _err(exc)


def model_forge_storage(args: dict | None = None, **kwargs) -> str:
    """List model-storage entries (downloads dir + engine-managed model
    dirs), or delete one listed entry (routed through the human-approval
    gate; see policy_hook)."""
    args = args or {}
    try:
        from . import storage
        action = str(args.get("action") or "list").strip() or "list"
        if action not in ("list", "delete"):
            return _dump({"error": f"unknown action: {action!r}; valid "
                                   f"actions: ['list', 'delete']"})
        if action == "list":
            return _dump(storage.list_storage())
        ref = str(args.get("entry") or "").strip()
        if not ref:
            return _dump({"error": "entry required for action=delete: pass "
                                   "an entry id from action=list or that "
                                   "entry's exact path"})
        return _dump(storage.delete_entry(ref))
    except Exception as exc:
        return _err(exc)


def model_forge_generate_media(args: dict | None = None, **kwargs) -> str:
    """Generate 3D models / music / SFX through a compute target's
    media-gateway endpoint. 3d submits a job (poll with job_id); music/sfx
    run synchronously and return audio (JSON result bodies are passed
    through)."""
    args = args or {}
    try:
        import json as _json
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
                job_id = str(args["job_id"])
                if not _JOB_TOKEN_RE.fullmatch(job_id):
                    return _dump({"ok": False, "action": "invalid_job_id",
                                  "error": "job_id must be 1-128 ASCII "
                                           "letters, digits, '_' or '-' "
                                           f"(got {job_id!r})"})
                req = urllib.request.Request(
                    f"{root}/jobs/{job_id}")
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
                return _dump(_submit_failure(exc))
            return _dump({"ok": True, "action": "submitted",
                          "submitted": out,
                          "poll": f"model_forge_generate_media with "
                                  f"modality=3d, target={target.id}, "
                                  f"job_id={out.get('id', '')}"})
        # music / sfx: synchronous POST. The caller waits while the engine
        # renders; gateways return audio bytes, some return a JSON body.
        model = str(args.get("model", "")).strip()
        prompt = str(args.get("prompt", "")).strip()
        if not model or not prompt:
            return _dump({"ok": False, "action": "needs_args",
                          "detail": f"{modality} needs model and prompt"})
        body = {"model": model, "prompt": prompt}
        req = urllib.request.Request(
            plan["submit"], data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=600.0) as resp:
                raw = resp.read()
                try:
                    content_type = resp.headers.get("Content-Type") or ""
                except Exception:  # noqa: BLE001 — headers are best-effort
                    content_type = ""
        except Exception as exc:  # noqa: BLE001
            return _dump(_submit_failure(exc))
        waited = time.monotonic() - started
        wait_note = (f"this call blocked {waited:.1f} s until the engine "
                     "finished rendering")
        try:
            out = _json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            out = None
        if out is not None:
            return _dump({"ok": True, "action": "done", "result": out,
                          "waited_seconds": round(waited, 3),
                          "note": wait_note})
        path = _save_generated_audio(modality, raw, content_type)
        return _dump({"ok": True, "action": "done", "audio": str(path),
                      "bytes": len(raw), "content_type": content_type,
                      "waited_seconds": round(waited, 3),
                      "note": wait_note + f"; audio saved to {path}"})
    except Exception as exc:
        return _err(exc)


def _submit_failure(exc: Exception) -> dict:
    """Error payload for a failed media-endpoint call (fail closed). An HTTP
    failure names its status and a short response snippet in the detail so
    the caller can act on it instead of a generic submit_failed."""
    out = {"ok": False, "action": "submit_failed"}
    status = getattr(exc, "code", None)
    if isinstance(status, int):
        out["http_status"] = status
    snippet = ""
    read = getattr(exc, "read", None)
    if callable(read):
        try:
            raw = read()
        except Exception:  # noqa: BLE001 — snippet is best-effort
            raw = b""
        if isinstance(raw, (bytes, bytearray)):
            snippet = bytes(raw).decode("utf-8", "replace").strip()[:200]
    if snippet:
        out["response_snippet"] = snippet
    if "http_status" in out:
        detail = f"media endpoint request failed: HTTP {out['http_status']}"
        reason = getattr(exc, "reason", None)
        if reason:
            detail += f" {reason}"
        if snippet:
            detail += f"; response: {snippet}"
    else:
        detail = f"{type(exc).__name__}: {exc}"
    out["error"] = detail
    return out


def _audio_suffix(raw: bytes, content_type: str) -> str:
    """File suffix for a generated audio payload: Content-Type first, then
    container sniffing, then an honest .bin."""
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    by_mime = {
        "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav",
        "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/ogg": ".ogg",
        "audio/flac": ".flac", "audio/mp4": ".m4a", "audio/aac": ".aac",
        "audio/webm": ".weba",
    }
    if mime in by_mime:
        return by_mime[mime]
    if raw.startswith(b"ID3") or (len(raw) > 2 and raw[0] == 0xFF
                                  and raw[1] & 0xE0 == 0xE0):
        return ".mp3"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WAVE":
        return ".wav"
    if raw.startswith(b"OggS"):
        return ".ogg"
    if raw.startswith(b"fLaC"):
        return ".flac"
    return ".bin"


def _save_generated_audio(modality: str, raw: bytes,
                          content_type: str):
    """Persist a non-JSON generation response under the plugin data dir
    (model-forge/media/) and return the path."""
    from pathlib import Path
    from plugins.plugin_storage import plugin_data_dir
    media_dir = plugin_data_dir("model-forge") / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    name = f"{modality}-{uuid.uuid4().hex[:12]}" \
           f"{_audio_suffix(raw, content_type)}"
    path = Path(media_dir) / name
    path.write_bytes(raw)
    return path


# --- Hugging Face repo file auto-resolve ------------------------------------

def _fetch_hf_json(url: str) -> dict:
    """GET a JSON document from Hugging Face (stdlib urllib)."""
    req = urllib.request.Request(url, headers={"User-Agent": "model-forge"})
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _is_weights_file(path: str) -> bool:
    return path.lower().endswith(_WEIGHT_SUFFIXES)


def _suffix_rank(path: str) -> int:
    """Preference rank for a weights file's suffix (lower = preferred)."""
    low = path.lower()
    for suffix, rank in _WEIGHT_SUFFIX_RANK.items():
        if low.endswith(suffix):
            return rank
    return 99


def _is_component_path(path: str) -> bool:
    """True for paths that hold auxiliary components (text encoders, VAEs,
    CLIP, tokenizer/processor/config files) — never the primary weights."""
    for part in path.lower().split("/"):
        if part in _COMPONENT_DIR_NAMES or part.startswith("text_encoder"):
            return True
        for fragment in _COMPONENT_NAME_FRAGMENTS:
            if fragment in part:
                return True
    return False


def _weights_plan(files: list[tuple[str, int]]) -> dict:
    """Pick the primary weights file from a repo file list (pure).

    files: (path, size_bytes) rows. Picks the largest weights-suffix file
    that is not an auxiliary component. If one file cannot honestly satisfy
    an install — a diffusers-style multi-component pipeline
    (model_index.json), large component weights in several directories, or
    sharded weights — returns a multi_file plan listing what was found
    instead of pretending one file suffices.
    """
    names = [p for p, _ in files]
    if any(p.lower().rsplit("/", 1)[-1] == "model_index.json" for p in names):
        return {"ok": False, "action": "multi_file",
                "detail": "model_index.json marks a multi-component model "
                          "pipeline — one file cannot satisfy an install; "
                          "pass engine_model (and dest) once per component",
                "files": sorted(names)}
    weights = [(p, s) for p, s in files if _is_weights_file(p)]
    large = [(p, s) for p, s in weights if s >= _LARGE_FILE_BYTES]
    comp_dirs = {p.rsplit("/", 1)[0].lower() for p, _ in large
                 if _is_component_path(p)}
    if len(comp_dirs) >= 2:
        return {"ok": False, "action": "multi_file",
                "detail": "large component weights in several directories ("
                          + ", ".join(sorted(comp_dirs))
                          + ") — one file cannot satisfy an install; pass "
                          "engine_model (and dest) once per component",
                "files": sorted(p for p, _ in large)}
    candidates = [(p, s) for p, s in weights if not _is_component_path(p)]
    shards = [(p, s) for p, s in candidates if _SHARD_RE.search(p)]
    if len(shards) >= 2:
        return {"ok": False, "action": "multi_file",
                "detail": "weights are split into "
                          f"{len(shards)} shard files — every shard is "
                          "required; pass engine_model once per shard",
                "files": sorted(p for p, _ in shards)}
    if not candidates:
        return {"ok": False, "action": "no_weights_found",
                "detail": "no primary weights file found in this repo; "
                          "weight files present: "
                          + (", ".join(sorted(p for p, _ in weights))
                             or "none"),
                "files": sorted(names)}
    pick = min(candidates,
               key=lambda row: (-row[1], _suffix_rank(row[0]), row[0]))
    return {"ok": True, "action": "resolved", "file": pick[0],
            "size_bytes": pick[1],
            "detail": "primary weights file picked from the repo file list"}


def resolve_repo_weights(repo_id: str, fetch=None) -> dict:
    """Resolve the primary weights file of a Hugging Face repo from its file
    list (model-info API, blobs=true). stdlib urllib; pass fetch= to inject
    a fake in tests. API unreachable or unusable fails soft: the caller gets
    the corrected needs_engine_model message instead of a crash."""
    fetch = fetch or _fetch_hf_json
    url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
    try:
        data = fetch(url)
    except Exception:  # noqa: BLE001 — fail soft to a manual path request
        return {"ok": False, "action": "needs_engine_model",
                "detail": "pass engine_model=" + _FILE_PULL_HINT}
    rows = data.get("siblings") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {"ok": False, "action": "needs_engine_model",
                "detail": "pass engine_model=" + _FILE_PULL_HINT}
    files = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("rfilename")
        if not isinstance(name, str) or not name:
            continue
        size = row.get("size")
        if not isinstance(size, int):
            lfs = row.get("lfs")
            size = lfs.get("size") if isinstance(lfs, dict) else None
        files.append((name, size if isinstance(size, int) else 0))
    plan = _weights_plan(files)
    plan["repo"] = repo_id
    return plan


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
