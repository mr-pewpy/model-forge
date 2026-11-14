"""Tool handlers — every handler returns a JSON string, even on error.

Handlers wrap pure planners (install_plan / activation_plan) so the routing
logic is unit-tested without touching the network or the Hermes config.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
import uuid

from .broker import (
    EngineLifecycle, EvictionPlanner, LlamaCppServerLifecycle,
    LlamaSwapLifecycle, LoadRequest, OllamaLifecycle, Orchestrator,
    ResidencyLedger,
)
from .credentials import status as credentials_status
from .engines import detect_engines
from .engines import llamacpp as llamacpp_adapter
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
# Engine provisioning: the vendor-official install commands Model Forge's
# engine_install action may plan and run. The command is never model-supplied
# — only these fixed strings can execute, and only after the human-approval
# gate approves the exact command (rule_key model-forge:engine_install).
# llama.cpp is deliberately absent: Hermes core manages a llama.cpp runtime
# (hardware-priced Local Models catalog and downloads), so engine_install
# returns guidance to it instead of installing binaries.
_ENGINE_INSTALLERS = {
    "ollama": {
        "vendor": "Ollama",
        "command": "curl -fsSL https://ollama.com/install.sh | sh",
        "steps": ["run the Ollama official install script "
                  "(curl -fsSL https://ollama.com/install.sh | sh)"],
    },
    "strata": {
        "vendor": "Strata",
        "command": ("curl -fsSL https://github.com/Niko1221/Strata/releases/"
                    "latest/download/install.sh | sh"),
        "steps": ["run the Strata official install script "
                  "(curl -fsSL https://github.com/Niko1221/Strata/releases/"
                  "latest/download/install.sh | sh)"],
    },
    "maya": {
        "vendor": "Project Maya",
        "command": ("git clone https://github.com/mw00/project-maya "
                    "&& cd project-maya && ./setup.sh"),
        "steps": ["git clone https://github.com/mw00/project-maya",
                  "./setup.sh (run from inside the cloned repository)"],
    },
}
# Hard timeout for an approved installer job: the process group is killed at
# the limit so nothing can hang unattended.
_ENGINE_INSTALL_TIMEOUT_S = 900.0


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=1)


def _installer_key(engine_id: str) -> str:
    """Normalized engine id for the approved-installer catalog."""
    return (engine_id or "").strip().lower()


def _installer_for(engine_id: str) -> dict | None:
    """Approved vendor installer spec for an engine id, or None. The catalog
    is the ONLY source of executable commands — model-supplied strings never
    reach a shell."""
    return _ENGINE_INSTALLERS.get(_installer_key(engine_id))


def policy_hook(tool_name: str, args: dict | None = None, task_id: str = "",
                **kwargs) -> dict | None:
    """pre_tool_call policy: routes the policy decisions through
    Hermes' human-approval gate.

    Registering a compute target decides where engine calls and media jobs
    (including image data) can be sent; rewriting model.default redirects all
    future conversations; deleting a model-storage entry destroys model
    weights on this machine; running a vendor engine installer (engine_install
    with execute=true) puts vendor software on this machine. All four are the
    USER'S decisions — the model must not make them alone. Escalating with
    ``action: approve`` shows the user the exact URL / config change / path
    and size / install command; denial, timeout, or a gate error fails closed
    (the call is blocked before the handler runs). Guidance-only engine
    plans (llama.cpp -> Hermes core's managed runtime) carry no command and
    execute nothing, so they do not escalate.
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
    if tool_name == "model_forge_install" \
            and args.get("action") == "engine_install" \
            and args.get("execute") is True:
        spec = _installer_for(str(args.get("engine") or ""))
        if spec is not None:
            return {
                "action": "approve",
                "message": (
                    "Model Forge wants to install the "
                    f"{spec['vendor']} inference engine on this machine by "
                    "running the vendor's official install command in a "
                    "background job:\n"
                    f"  command: {spec['command']}\n"
                    f"  engine: {_installer_key(str(args.get('engine') or ''))}\n"
                    "Progress is pollable with model_forge_install job_id, "
                    "and a hard timeout stops the installer if it runs too "
                    "long. Approve to run it — denying or letting this "
                    "prompt time out runs nothing."),
                "rule_key": "model-forge:engine_install",
            }
        return None   # no approved installer -> nothing can execute
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
            note = (
                "no matching engine detected on this target — install one or "
                "register a target that runs it (planning for the listed "
                f"engine '{fallback}')")
            hint = _engine_install_hint(fallback)
            return fallback, note + (" " + hint if hint else "")
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


# --- engine provisioning (approved vendor installers) -----------------------

def _core_llamacpp_status() -> dict:
    """Detect Hermes core's managed llama.cpp runtime — read-only signals
    only: a `hermes` CLI on PATH (core is installed and owns the runtime,
    its hardware-priced Local Models catalog and downloads) and/or a
    llama-server already answering locally. Reports exactly what was found,
    never more."""
    signals = []
    if shutil.which("hermes"):
        signals.append("the hermes CLI is on PATH (Hermes core is installed)")
    try:
        status = llamacpp_adapter.detect("")
    except Exception:  # noqa: BLE001 — detection is advisory, fail soft
        status = None
    if status is not None and status.available:
        signals.append(f"a llama.cpp server is answering at {status.endpoint}")
    return {"present": bool(signals), "signals": signals}


def engine_install_plan(engine_id: str, core_probe=None) -> dict:
    """What engine_install would do for an engine id — no side effects
    beyond read-only detection (unit-tested).

    Engines with an approved vendor installer return the exact command the
    installer would run; it executes only with execute=true after the
    human-approval gate approves that command. llama.cpp has no installer:
    the plan DETECTS whether Hermes core's managed llama.cpp runtime is
    present and returns honest guidance to core's Local Models catalog and
    downloads instead — Model Forge never installs llama.cpp binaries."""
    key = _installer_key(engine_id)
    if not key:
        return {"ok": False, "action": "needs_engine",
                "error": "engine required: pass engine=<engine id> "
                         f"(approved installers: {sorted(_ENGINE_INSTALLERS)}; "
                         "'llamacpp' returns guidance to Hermes core's "
                         "managed runtime)"}
    if key == "llamacpp":
        probe = core_probe or _core_llamacpp_status
        runtime = probe()
        if runtime.get("present"):
            found = "; ".join(runtime.get("signals") or ()) or "present"
            detail = ("llamacpp has no vendor installer — Model Forge never "
                      "installs llama.cpp binaries. Hermes core's managed "
                      f"llama.cpp runtime is present on this machine ({found}). "
                      "Use it: GGUF models are priced and installed through "
                      "the built-in Local Models catalog (Settings -> "
                      "Providers -> Local Models, or `hermes model`)")
        else:
            detail = ("llamacpp has no vendor installer — Model Forge never "
                      "installs llama.cpp binaries. No managed llama.cpp "
                      "runtime was detected on this machine. Use Hermes "
                      "core's Local Models (Settings -> Providers -> Local "
                      "Models, or `hermes model`): it manages the llama.cpp "
                      "runtime, its hardware-priced catalog and its "
                      "downloads; Model Forge detects the server "
                      "automatically once it starts")
        return {"ok": True, "action": "use_core_local_models", "engine": "llamacpp",
                "executes": False, "core_runtime": runtime, "detail": detail}
    spec = _installer_for(key)
    if spec is None:
        return {"ok": False, "action": "unknown_engine", "engine": key,
                "error": f"no approved engine installer for '{key}'; Model "
                         f"Forge can provision {sorted(_ENGINE_INSTALLERS)} "
                         "from their vendors' official installers and guides "
                         "'llamacpp' to Hermes core's managed runtime"}
    return {"ok": True, "action": "vendor_installer", "engine": key,
            "vendor": spec["vendor"], "command": spec["command"],
            "steps": list(spec["steps"]), "executes": True,
            "detail": f"the {spec['vendor']} vendor-official install "
                      "command; it runs only with execute=true after the "
                      "human-approval gate approves this exact command "
                      "(rule_key model-forge:engine_install)"}


def _engine_install_hint(engine_id: str) -> str:
    """Discovery integration: the 'no matching engine detected' fallback
    names the engine_install action for engines with an approved installer
    path — and for llama.cpp, the honest core-runtime pointer instead."""
    key = _installer_key(engine_id)
    if key in _ENGINE_INSTALLERS:
        return (f"Model Forge can provision '{key}' from its vendor's "
                "official installer: model_forge_install with "
                f"action='engine_install', engine='{key}' — plan first, "
                "then the human-approval gate shows the exact command "
                "before anything runs")
    if key == "llamacpp":
        return ("Model Forge never installs llama.cpp binaries — use Hermes "
                "core's managed llama.cpp runtime (the built-in Local Models "
                "catalog); model_forge_install with action='engine_install', "
                "engine='llamacpp' explains")
    return ""


def _confirm_engine(engine_id: str) -> dict:
    """Re-run engine detection after an installer exits — the install is
    confirmed by detection, never assumed from the installer's exit code."""
    try:
        statuses = detect_engines("")
    except Exception as exc:  # noqa: BLE001 — honest, not fatal
        return {"detected": False,
                "detail": f"engine detection failed after the install: "
                          f"{type(exc).__name__}: {exc}"}
    for e in statuses:
        if e.id == _installer_key(engine_id) and e.available:
            return {"detected": True, "endpoint": e.endpoint,
                    "detail": e.detail or "the engine answers detection"}
    return {"detected": False,
            "detail": "the engine did not answer detection after the "
                      "installer finished; engines are started by you — run "
                      "it and Model Forge will detect it"}


def _run_installer(command: str, timeout_s: float) -> dict:
    """Run the approved vendor install command in its own session. The hard
    timeout kills the whole process group so nothing can hang unattended;
    on timeout this raises subprocess.TimeoutExpired (the job record says
    'timeout', never a false 'done')."""
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()
        try:
            proc.wait(timeout=10.0)
        except Exception:  # noqa: BLE001 — already killed; nothing to do
            pass
        raise
    return {"returncode": proc.returncode,
            "output_tail": (out or b"").decode("utf-8", "replace")[-2000:]}


def _engine_install_job(job_id: str, plan: dict, runner=None) -> None:
    """Background installer job (daemon thread, the pull.py pattern): writes
    progress records to the shared jobs store (polled with
    model_forge_install job_id), enforces the hard timeout, and confirms the
    install by re-running engine detection."""
    from . import pull
    state = {"job_id": job_id, "kind": "engine_install",
             "engine": plan["engine"], "vendor": plan["vendor"],
             "command": plan["command"], "state": "running",
             "hard_timeout_s": _ENGINE_INSTALL_TIMEOUT_S}
    pull._write_job(job_id, state)
    try:
        runner = runner or _run_installer
        result = runner(plan["command"], _ENGINE_INSTALL_TIMEOUT_S)
        state["returncode"] = result.get("returncode")
        state["output_tail"] = result.get("output_tail", "")
        if result.get("returncode") == 0:
            state["detection"] = _confirm_engine(plan["engine"])
            state["state"] = "done"
            state["detail"] = (
                "the vendor installer exited successfully"
                + (" and the engine now answers detection"
                   if state["detection"].get("detected")
                   else "; see detection for the honest post-install state"))
        else:
            state["state"] = "error"
            state["error"] = (
                f"the vendor installer exited with code "
                f"{result.get('returncode')} — nothing is assumed installed")
    except subprocess.TimeoutExpired:
        state["state"] = "timeout"
        state["error"] = (
            f"the vendor installer exceeded the "
            f"{_ENGINE_INSTALL_TIMEOUT_S:.0f} s hard timeout and was "
            "stopped (process group killed) — nothing is assumed installed")
    except Exception as exc:  # noqa: BLE001 — the job record the failure
        state["state"] = "error"
        state["error"] = f"{type(exc).__name__}: {exc}"
    pull._write_job(job_id, state)


def start_engine_install(engine_id: str, runner=None) -> dict:
    """Start the approved vendor installer as a background job (daemon
    thread + jobs-store records). Call only after the human-approval gate
    approved the exact command — the plan (engine_install_plan) comes first."""
    from . import pull
    plan = engine_install_plan(engine_id)
    if not plan.get("command"):
        return {"ok": False, "action": plan.get("action", "unknown_engine"),
                "engine": plan.get("engine", _installer_key(engine_id)),
                "error": plan.get("error")
                         or f"engine '{engine_id}' has no vendor install "
                           "command — nothing was run"}
    job_id = uuid.uuid4().hex[:12]
    pull._write_job(job_id, {"job_id": job_id, "kind": "engine_install",
                             "engine": plan["engine"], "vendor": plan["vendor"],
                             "command": plan["command"], "state": "starting",
                             "hard_timeout_s": _ENGINE_INSTALL_TIMEOUT_S})
    threading.Thread(target=_engine_install_job, args=(job_id, plan, runner),
                     daemon=True).start()
    return {"ok": True, "job_id": job_id,
            "poll": f"model_forge_install with job_id={job_id}"}


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


def _engine_install_action(args: dict) -> dict:
    """engine_install action: plan first (the exact vendor command), execute
    only with execute=true — and that call is escalated to the human-approval
    gate by policy_hook (rule_key model-forge:engine_install) BEFORE this
    handler runs; denial, timeout or a gate error blocks the call entirely,
    so nothing runs without explicit approval. The command is never
    model-supplied: it comes only from the approved-installer catalog."""
    engine_id = str(args.get("engine") or "").strip()
    plan = engine_install_plan(engine_id)
    if args.get("execute") is not True:
        plan["executed"] = False
        if plan.get("command"):
            plan["hint"] = ("plan only — re-run with action='engine_install', "
                            f"engine='{plan['engine']}', execute=true to run "
                            "the command above; the human-approval gate "
                            "shows the exact command first, and denial or "
                            "timeout runs nothing")
        return plan
    # execute=true. For a command-bearing plan, policy_hook has already
    # escalated this exact call to the human-approval gate — reaching this
    # point means the user approved the exact command shown there.
    plan["executed"] = False
    if not plan.get("command"):
        return plan   # guidance-only (llama.cpp) or invalid: nothing to run
    started = start_engine_install(plan["engine"])
    if not started.get("ok"):
        plan.update({k: v for k, v in started.items() if k != "ok"})
        return plan
    plan.update(started)
    plan["executed"] = True
    plan["action"] = "engine_install_started"
    plan["detail"] = (
        "the approved vendor installer runs as a background job (hard "
        f"timeout {int(_ENGINE_INSTALL_TIMEOUT_S)} s); poll with "
        f"model_forge_install job_id={started['job_id']} — completion "
        "re-runs engine detection and reports it honestly")
    return plan


def model_forge_install(args: dict | None = None, **kwargs) -> str:
    """Pull a model on a compute target (Maya: HTTP pull; HF download: job;
    Strata: CLI), or provision an engine (action='engine_install': plan the
    vendor's official install command first; execute=true runs it behind the
    human-approval gate). Pass job_id to poll a download or engine install."""
    args = args or {}
    try:
        from . import pull
        if args.get("job_id"):
            return _dump(pull.job_status(str(args["job_id"])))
        action = str(args.get("action") or "model_install").strip()
        if action == "engine_install":
            return _dump(_engine_install_action(args))
        if action != "model_install":
            return _dump({"error": f"unknown action: {action!r}; valid "
                                   f"actions: ['model_install', "
                                   f"'engine_install']"})
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
    through). The GPU lane is claimed as one workflow scope before the job
    is submitted and released when the job completes or fails; a refused
    claim fails closed with an honest error and submits nothing."""
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
                try:
                    with urllib.request.urlopen(req, timeout=30.0) as resp:
                        job = _json.loads(
                            resp.read().decode("utf-8", "replace"))
                except Exception as exc:  # noqa: BLE001
                    # Same honesty as the submit paths: name the HTTP status
                    # and a response snippet instead of a generic error.
                    out = _submit_failure(exc)
                    out["action"] = "poll_failed"
                    return _dump(out)
                out = {"job": job}
                held = _MEDIA_SCOPE_JOBS.get(job_id)
                if held is not None:
                    scope_id = held["scope_id"]
                    history = [dict(row) for row in broker_stack(target)
                               ["orchestrator"].history(scope_id)]
                    claim = {"scope_id": scope_id,
                             "model_id": held.get("model_id"),
                             "modality": held.get("modality"),
                             "progress": history}
                    if _media_job_finished(job):
                        _MEDIA_SCOPE_JOBS.pop(job_id, None)
                        out["lane_claim"] = _lane_block(
                            claim,
                            released=_release_media_lane(target, scope_id),
                            note="job complete — the lane scope was released "
                                 "and what was resident at claim time is "
                                 "restored")
                    else:
                        out["lane_claim"] = _lane_block(
                            claim,
                            note="the job is still running (or its state is "
                                 "unclear) — the lane stays claimed; poll "
                                 "again and the scope releases when the job "
                                 "finishes, or end it now with "
                                 "model_forge_broker action=release, "
                                 f"scope_id={scope_id!r}")
                return _dump(out)
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
            # Claim the GPU lane around the job as one workflow scope: the
            # eviction policy runs, what must go steps down, and the scope is
            # released when the job completes or fails.
            claim = _claim_media_lane(target, modality, model)
            if not claim.get("ok"):
                return _dump({"ok": False, "action": "claim_failed",
                              "error": claim.get("error") or
                                       "the lane claim was refused",
                              "lane_claim": _lane_block(claim),
                              "detail": "the GPU lane claim was refused (fail "
                                        "closed) — nothing was submitted"})
            settled = []

            def _settle(note):
                rel = _release_media_lane(target, claim["scope_id"])
                settled.append(rel)
                return _lane_block(claim, released=rel, note=note)

            try:
                req = urllib.request.Request(
                    plan["submit"], data=_json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=60.0) as resp:
                        out = _json.loads(resp.read().decode("utf-8", "replace"))
                except Exception as exc:  # noqa: BLE001
                    failure = _submit_failure(exc)
                    failure["lane_claim"] = _settle(
                        "the job failed to start — the lane scope was "
                        "released and what was resident at claim time is "
                        "restored")
                    return _dump(failure)
                job_id = str(out.get("id", "") or "")
                if job_id and _JOB_TOKEN_RE.fullmatch(job_id):
                    _MEDIA_SCOPE_JOBS[job_id] = {
                        "scope_id": claim["scope_id"], "target_id": target.id,
                        "model_id": model, "modality": modality}
                    settled.append(True)   # held: the scope outlives this call
                    lane = _lane_block(
                        claim,
                        note="the lane stays claimed until the job "
                             f"completes; poll with job_id={job_id} — the "
                             "scope releases and what was resident at claim "
                             "time is restored when the job finishes")
                else:
                    lane = _settle(
                        "no pollable job id came back — the lane scope was "
                        "released immediately")
                return _dump({"ok": True, "action": "submitted",
                              "submitted": out,
                              "lane_claim": lane,
                              "poll": f"model_forge_generate_media with "
                                      f"modality=3d, target={target.id}, "
                                      f"job_id={out.get('id', '')}"})
            finally:
                if not settled:
                    _release_media_lane(target, claim["scope_id"])
        # music / sfx: synchronous POST. The caller waits while the engine
        # renders; gateways return audio bytes, some return a JSON body.
        model = str(args.get("model", "")).strip()
        prompt = str(args.get("prompt", "")).strip()
        if not model or not prompt:
            return _dump({"ok": False, "action": "needs_args",
                          "detail": f"{modality} needs model and prompt"})
        body = {"model": model, "prompt": prompt}
        # Claim the GPU lane around the job as one workflow scope; it is
        # released when the render completes or on any failure.
        claim = _claim_media_lane(target, modality, model)
        if not claim.get("ok"):
            return _dump({"ok": False, "action": "claim_failed",
                          "error": claim.get("error") or
                                   "the lane claim was refused",
                          "lane_claim": _lane_block(claim),
                          "detail": "the GPU lane claim was refused (fail "
                                    "closed) — nothing was submitted"})
        settled = []

        def _settle(note):
            rel = _release_media_lane(target, claim["scope_id"])
            settled.append(rel)
            return _lane_block(claim, released=rel, note=note)

        try:
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
                failure = _submit_failure(exc)
                failure["lane_claim"] = _settle(
                    "the job failed — the lane scope was released and what "
                    "was resident at claim time is restored")
                return _dump(failure)
            waited = time.monotonic() - started
            wait_note = (f"this call blocked {waited:.1f} s until the engine "
                         "finished rendering")
            lane = _settle(
                "the render completed — the lane scope was released and what "
                "was resident at claim time is restored")
            try:
                out = _json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                out = None
            if out is not None:
                return _dump({"ok": True, "action": "done", "result": out,
                              "waited_seconds": round(waited, 3),
                              "lane_claim": lane,
                              "note": wait_note})
            path = _save_generated_audio(modality, raw, content_type)
            return _dump({"ok": True, "action": "done", "audio": str(path),
                          "bytes": len(raw), "content_type": content_type,
                          "waited_seconds": round(waited, 3),
                          "lane_claim": lane,
                          "note": wait_note + f"; audio saved to {path}"})
        finally:
            if not settled:
                _release_media_lane(target, claim["scope_id"])
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


# --- weight residency: bookings, engine adapters, workflow scopes ------------

# Booked ceilings come from the curated catalog's published VRAM floors,
# honestly labeled wherever they surface: a floor is a published minimum used
# as the nominal ceiling until a measured peak replaces it (a measured booking
# is never downgraded by a published one).
BOOKING_SOURCE = "catalog-vram-floor"
BOOKING_SOURCE_NONE = "catalog-vram-floor (none declared)"
BOOKING_SOURCE_NOTE = (
    "booked ceilings are the curated catalog's published VRAM floors used as "
    "nominal ceilings — advisory published minimums, not measured peaks; a "
    "measured peak replaces one and is never downgraded. A model that "
    "declares no VRAM floor (engine-sized at launch) is booked at 0 GB so "
    "workflow scopes can still account for it.")


def bookings_from_registry(entries=None) -> dict:
    """Curated catalog ModelEntry footprints -> the bookings dict the
    residency layer takes.

    Each entry's published ``vram_gb_min`` floor becomes its nominal booked
    ceiling, with honest labeling (``BOOKING_SOURCE`` / ``BOOKING_SOURCE_NONE``
    for entries that declare no floor). Stdlib-only and pure: no I/O."""
    out = {}
    for m in (models_for() if entries is None else entries):
        try:
            floor = float(m.vram_gb_min or 0.0)
        except (TypeError, ValueError):
            floor = 0.0
        out[m.id] = {
            "vram_gb": floor,
            "source": BOOKING_SOURCE if floor > 0 else BOOKING_SOURCE_NONE,
            "load_s": None,
        }
    return out


class _PassThroughLifecycle(EngineLifecycle):
    """Observational pass-through for engines whose load/evict policy the
    engine itself owns (Model Forge has no explicit lifecycle control for
    them). The llama-swap honesty rule: report state, never drive the
    engine — loading is engine-native on first use, reclaim is engine-native
    later, and both are reported as such."""

    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self.no_control = (
            f"'{engine_id}' loads and reclaims models on its own; Model Forge "
            f"has no explicit load or eviction control for it")

    def ensure_loaded(self, model_id, **_ignored) -> dict:
        return {"ok": True, "engine": self.engine_id, "action": "load",
                "model_id": model_id, "state": "pass-through",
                "reason": self.no_control + " (loading is engine-native on "
                          "first use) — recorded as claimed for the scope"}

    def release(self, model_id) -> dict:
        return {"ok": True, "engine": self.engine_id, "action": "release",
                "model_id": model_id, "state": "pass-through",
                "reason": self.no_control + " (reclaim happens on the "
                          "engine's own schedule)"}

    def probe(self) -> dict:
        return {"ok": True, "engine": self.engine_id, "state": "unknown",
                "resident": [], "reason": self.no_control}

# Engines with explicit load/evict control (real lifecycle adapters), and the
# engines whose lifecycle the engine itself owns (observational pass-through).
_LIFECYCLE_ENGINES = ("ollama", "llama-swap", "llamacpp")
_PASS_THROUGH_ENGINES = ("comfyui", "vllm", "sglang", "strata", "maya",
                         "openai-compatible", "python", "media-gateway")


def _adapter_catalog(target, http=None, runner=None, which=None) -> dict:
    """Engine lifecycle adapters for a compute target. Real control where the
    engine exposes it (Ollama warm/evict calls, llama-swap observation,
    llama.cpp process start/stop) — every external seam (HTTP function,
    process runner, binary lookup) is injectable so nothing touches the
    network or spawns a process unless the caller allows it. Everything else
    is an honest pass-through. Load/unload calls go to the target's
    user-configured endpoint."""
    root = root_url(target.base_url) or "http://127.0.0.1"
    catalog = {
        "ollama": OllamaLifecycle(base_url=root, http=http),
        "llama-swap": LlamaSwapLifecycle(base_url=root, http=http),
        "llamacpp": LlamaCppServerLifecycle(runner=runner, which=which),
    }
    for engine_id in _PASS_THROUGH_ENGINES:
        catalog[engine_id] = _PassThroughLifecycle(engine_id)
    return catalog


_BROKER_LOCK = threading.Lock()
_BROKER_STACKS: dict = {}
# Workflow scopes opened around media jobs (media job id -> scope record), so
# an async job's scope is released when the job completes.
_MEDIA_SCOPE_JOBS: dict = {}
# Every open workflow scope (scope id -> record), for the read-only summary.
_ACTIVE_SCOPES: dict = {}


def _build_broker_stack(target, *, http=None, runner=None, which=None,
                        clock=None, adapters=None) -> dict:
    """Build the weight-residency stack for a compute target: the residency
    summary (bookings from the catalog's published VRAM floors, budget from
    the target's declared VRAM), the eviction policy, and the claim/release
    orchestrator with engine lifecycle adapters. Construction only performs
    I/O through the injected seams; pass adapters= to replace the adapter
    catalog wholesale (tests)."""
    bookings = bookings_from_registry()
    ledger = ResidencyLedger(bookings=bookings, clock=clock)
    try:
        budget = float(_hw_for_target(target).total_vram_gb or 0.0)
    except Exception:  # noqa: BLE001 — honest: unknown budget stays undeclared
        budget = 0.0
    if budget > 0:
        ledger.set_budget(target.id, budget)
    if adapters is None:
        adapters = _adapter_catalog(target, http=http, runner=runner,
                                    which=which)
    planner = EvictionPlanner(ledger, clock=clock)
    orchestrator = Orchestrator(ledger, planner, adapters, clock=clock)
    return {"target_id": target.id, "ledger": ledger, "planner": planner,
            "orchestrator": orchestrator, "bookings": bookings,
            "budget_gb": budget}


def broker_stack(target, **seams):
    """The weight-residency stack for a compute target, cached per target so
    workflow scopes survive across tool calls (claim, then release later).
    Any injected seam (http / runner / which / clock / adapters) or
    fresh=True builds an isolated stack that never enters the cache."""
    fresh = bool(seams.pop("fresh", False))
    if seams or fresh:
        return _build_broker_stack(target, **seams)
    with _BROKER_LOCK:
        stack = _BROKER_STACKS.get(target.id)
        if stack is None:
            stack = _build_broker_stack(target)
            _BROKER_STACKS[target.id] = stack
    return stack


def _step_rows(steps) -> list:
    return [{"action": s.action, "model_id": s.model_id, "engine": s.engine,
             "lane": s.lane, "reason": s.reason} for s in (steps or ())]


def _claim_engine(entry: ModelEntry) -> str:
    """The engine that will serve a claim's warm: the first engine Model Forge
    can actually run for the entry, else the entry's first engine (the
    pass-through adapter labels what that means), else ''."""
    for engine_id in entry.engines:
        if engine_id not in PHANTOM_ENGINES:
            return engine_id
    return entry.engines[0] if entry.engines else ""


def _claim_request(entry):
    engine = _claim_engine(entry)
    if not engine:
        return None, (f"'{entry.id}' has no engine that can serve it — "
                      "nothing to warm")
    return LoadRequest(model_id=entry.id, engine=engine, lane=entry.modality,
                       modality=entry.modality), ""


def _claim_entry(model_id: str = "", modality: str = ""):
    """What to claim: an explicit catalog model id, or a modality's default
    curated model (its first non-stub, non-catalog-marker entry)."""
    model_id = (model_id or "").strip()
    modality = (modality or "").strip()
    if not model_id and not modality:
        return None, ("pass model_id (a catalog model id from "
                      "model_forge_browse) or modality to claim the lane for")
    if modality and modality not in MODALITIES:
        return None, (f"unknown modality: {modality!r}; valid modalities: "
                      f"{list(MODALITIES)}")
    if model_id:
        entry = next((m for m in models_for() if m.id == model_id), None)
        if entry is None:
            return None, f"unknown model_id: {model_id}"
        return entry, ""
    for entry in models_for(modality=modality):
        if entry.status != "stub" and entry.id not in _CATALOG_MARKER_IDS:
            return entry, ""
    return None, (f"no curated {modality} model exists yet — claim a specific "
                  "model_id instead")


def _seed_local_pulls(stack: dict, target) -> None:
    """Record the local target's on-disk weights (the model storage
    inventory) in the residency summary — a read-only scan, fail-soft. Remote
    targets are never scanned: they report only what was recorded for them."""
    if target.kind != "local" or stack.get("pulls_seeded"):
        return
    stack["pulls_seeded"] = True
    try:
        from . import storage
        listing = storage.list_storage()
    except Exception:  # noqa: BLE001 — the summary degrades, never crashes
        return
    ledger = stack["ledger"]
    for row in listing.get("entries") or ():
        path = str(row.get("path") or "")
        if path:
            ledger.record_pull(target.id, str(row.get("name") or path),
                               path, row.get("size_bytes"))


def _broker_status(stack: dict, target) -> dict:
    """Honest read-only residency summary: pulled weights with sizes, what is
    resident now, booked ceilings, and current budget usage."""
    ledger = stack["ledger"]
    _seed_local_pulls(stack, target)
    budget = ledger.budget_gb(target.id)
    booked_now = ledger.booked_total_gb(target.id)
    return {
        "ok": True,
        "action": "status",
        "target": target.id,
        "pulled": [{"model_id": r.model_id, "path": r.path,
                    "size_bytes": r.size_bytes}
                   for r in ledger.pulled(target.id)],
        "resident": [{"model_id": r.model_id, "engine": r.engine,
                      "lane": r.lane} for r in ledger.residents(target.id)],
        "booked": [{"model_id": model_id, "vram_gb": spec["vram_gb"],
                    "source": spec["source"]}
                   for model_id, spec in sorted(stack["bookings"].items())],
        "budget": {
            "nominal_gb": budget,
            "booked_now_gb": round(booked_now, 3),
            "headroom_gb": (round(budget - booked_now, 3)
                            if budget is not None else None),
        },
        "unbooked_residents": list(ledger.unbooked_residents(target.id)),
        "active_scopes": [
            {"scope_id": sid, "model_id": rec.get("model_id"),
             "modality": rec.get("modality"), "kind": rec.get("kind")}
            for sid, rec in sorted(_ACTIVE_SCOPES.items())
            if rec.get("target_id") == target.id],
        "note": ("Read-only report. " + BOOKING_SOURCE_NOTE
                 + " Live free VRAM, when sampled, only orders which model "
                   "loads first — it never gates fit."),
    }


def _broker_plan(stack: dict, target, request) -> dict:
    """Dry run of a claim: the release/load steps with reasons — the policy
    planner is pure, so nothing is loaded, released or recorded here."""
    plan = stack["planner"].plan_load_many(target.id, [request])
    return {
        "ok": True,
        "action": "plan",
        "dry_run": True,
        "target": target.id,
        "model_id": request.model_id,
        "modality": request.modality,
        "would": {
            "ok": plan.ok,
            "reason": plan.reason,
            "release_steps": _step_rows(plan.release_steps),
            "load_steps": _step_rows(plan.load_steps),
        },
        "note": ("dry run — nothing was loaded, released or recorded; "
                 "action=claim executes exactly these steps and returns a "
                 "scope id"),
    }


def _broker_claim(stack: dict, target, request, scope_id: str = "") -> dict:
    scope_id = (scope_id or "").strip() or f"scope-{uuid.uuid4().hex[:8]}"
    orchestrator = stack["orchestrator"]
    result = orchestrator.claim(scope_id, target.id, request)
    ok = bool(result.get("ok"))
    out = {
        "ok": ok,
        "action": "claim" if ok else "claim_failed",
        "scope_id": scope_id if ok else None,
        "target": target.id,
        "model_id": request.model_id,
        "modality": request.modality,
        "steps": _step_rows(result.get("steps")),
        "progress": [dict(row) for row in orchestrator.history(scope_id)],
        "reason": result.get("reason", ""),
    }
    plan = result.get("plan")
    if plan is not None:
        out["plan"] = {"reason": plan.reason,
                       "release_steps": _step_rows(plan.release_steps),
                       "load_steps": _step_rows(plan.load_steps)}
    if ok:
        _ACTIVE_SCOPES[scope_id] = {"target_id": target.id,
                                    "model_id": request.model_id,
                                    "modality": request.modality,
                                    "kind": "workflow"}
        out["note"] = (
            f"the GPU lane is claimed for this scope; when the workflow is "
            f"done, release it with model_forge_broker action=release, "
            f"scope_id={scope_id!r} — that restores exactly what was "
            f"resident at claim time")
    else:
        out["error"] = result.get("reason", "the lane claim was refused")
        out["note"] = ("fail closed — the claim was refused and nothing is "
                       "held; see steps for anything already executed before "
                       "the refusal")
    return out


def _broker_release(stack: dict, scope_id: str) -> dict:
    orchestrator = stack["orchestrator"]
    result = orchestrator.release(scope_id)
    ok = bool(result.get("ok"))
    _ACTIVE_SCOPES.pop(scope_id, None)
    out = {
        "ok": ok,
        "action": "release" if ok else "release_failed",
        "scope_id": scope_id,
        "target": result.get("target_id", stack["target_id"]),
        "steps": _step_rows(result.get("steps")),
        "progress": [dict(row) for row in orchestrator.history(scope_id)],
        "reason": result.get("reason", ""),
    }
    if not ok:
        out["error"] = result.get("reason", "the scope could not be released")
    return out


def model_forge_broker(args: dict | None = None, **kwargs) -> str:
    """Weight residency and workflow scopes on a compute target. Verbs:
    status (read-only residency summary), claim (take the GPU lane for a
    modality or model — evict per policy, pre-warm, return a scope id with
    honest progress states), release (end a scope and restore what was
    resident at claim time), plan (dry run of claim — nothing executed)."""
    args = args or {}
    try:
        action = str(args.get("action") or "").strip()
        if action not in ("status", "claim", "release", "plan"):
            return _dump({"error": f"unknown action: {action!r}; valid "
                                   f"actions: ['status', 'claim', 'release', "
                                   f"'plan']"})
        if action == "release":
            scope_id = str(args.get("scope_id") or "").strip()
            if not scope_id:
                return _dump({"error": "scope_id required for action=release: "
                                       "pass the scope id returned by claim"})
            target_id = (str(args.get("target") or "").strip()
                         or _ACTIVE_SCOPES.get(scope_id, {}).get("target_id")
                         or "")
            target = get_target(target_id or None)
            if target is None:
                target = get_target(None)
            return _dump(_broker_release(broker_stack(target), scope_id))
        target = get_target(args.get("target"))
        if target is None:
            return _dump({"error": f"unknown target: {args.get('target')}"})
        stack = broker_stack(target)
        if action == "status":
            return _dump(_broker_status(stack, target))
        entry, err = _claim_entry(str(args.get("model_id") or ""),
                                 str(args.get("modality") or ""))
        if err:
            return _dump({"error": err})
        request, err = _claim_request(entry)
        if err:
            return _dump({"error": err})
        if action == "plan":
            return _dump(_broker_plan(stack, target, request))
        return _dump(_broker_claim(stack, target, request,
                                  scope_id=str(args.get("scope_id") or "")))
    except Exception as exc:
        return _err(exc)


# --- media job lane claims ---------------------------------------------------

_MEDIA_JOB_TERMINAL = frozenset({
    "done", "succeeded", "success", "completed", "finished",
    "error", "failed", "cancelled", "canceled"})


def _media_job_finished(job: dict) -> bool:
    """True when a media job record shows a terminal state. An unknown state
    reads as still running — the scope stays claimed and the output says how
    to end it (never guess completion)."""
    status = str((job or {}).get("status") or "").strip().lower()
    return status in _MEDIA_JOB_TERMINAL


def _media_stand_in(modality: str):
    """The curated entry whose published VRAM floor stands in as the nominal
    booked ceiling for a media-gateway model (a gateway model's own footprint
    is not published): the largest declared floor among the modality's
    curated entries, honestly labeled when booked."""
    best = None
    for entry in models_for(modality=modality):
        if entry.status == "stub":
            continue
        try:
            floor = float(entry.vram_gb_min or 0.0)
        except (TypeError, ValueError):
            continue
        if floor > 0 and (best is None or floor > best.vram_gb_min):
            best = entry
    return best


def _claim_media_lane(target, modality: str, model: str) -> dict:
    """Claim the GPU lane around one media job as a single workflow scope:
    run the eviction policy, step down what must go, and record the job's
    model as claimed for the scope (a media gateway loads it itself — the
    pass-through adapter says so)."""
    stack = broker_stack(target)
    ledger = stack["ledger"]
    stand_in = _media_stand_in(modality)
    if ledger.booked_vram_gb(model) is None and stand_in is not None:
        spec = {"vram_gb": float(stand_in.vram_gb_min),
                "source": f"{BOOKING_SOURCE} stand-in ({stand_in.id})",
                "load_s": None}
        ledger.book(model, spec)
        stack["bookings"][model] = spec
    request = LoadRequest(model_id=model, engine="media-gateway",
                          lane=modality, modality=modality)
    scope_id = f"media-{uuid.uuid4().hex[:8]}"
    result = stack["orchestrator"].claim(scope_id, target.id, request)
    ok = bool(result.get("ok"))
    out = {"ok": ok, "scope_id": scope_id if ok else None,
           "model_id": model, "modality": modality,
           "steps": _step_rows(result.get("steps")),
           "progress": [dict(row)
                        for row in stack["orchestrator"].history(scope_id)],
           "reason": result.get("reason", "")}
    if ok:
        _ACTIVE_SCOPES[scope_id] = {"target_id": target.id, "model_id": model,
                                    "modality": modality, "kind": "media-job"}
    else:
        out["error"] = result.get("reason", "the lane claim was refused")
    return out


def _release_media_lane(target, scope_id: str) -> dict:
    stack = broker_stack(target)
    result = stack["orchestrator"].release(scope_id)
    _ACTIVE_SCOPES.pop(scope_id, None)
    out = {"ok": bool(result.get("ok")), "scope_id": scope_id,
           "steps": _step_rows(result.get("steps")),
           "progress": [dict(row)
                        for row in stack["orchestrator"].history(scope_id)],
           "reason": result.get("reason", "")}
    if not out["ok"]:
        out["error"] = result.get("reason", "the scope could not be released")
    return out


def _lane_block(claim: dict, released: dict | None = None,
                note: str = "") -> dict:
    block = {"scope_id": claim.get("scope_id"),
             "model_id": claim.get("model_id"),
             "modality": claim.get("modality"),
             "progress": claim.get("progress") or []}
    if released is not None:
        block["state"] = "released"
        block["released"] = released
    else:
        block["state"] = "held"
    if note:
        block["note"] = note
    return block
