"""Tool handlers — every handler returns a JSON string, even on error.

Handlers wrap pure planners (install_plan / activation_plan) so the routing
logic is unit-tested without touching the network or the Hermes config.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
import uuid

from .aliases import resolve
from .broker import (
    EngineLifecycle, EvictionPlanner, LlamaCppServerLifecycle,
    LlamaSwapLifecycle, LoadRequest, MayaLifecycle, MeasuredFootprintRecorder,
    OllamaLifecycle, Orchestrator, ResidencyLedger, ComfyUiLifecycle,
    VllmServerLifecycle,
)
from .credentials import status as credentials_status
from .engines import detect_engines
from .engines import llamacpp as llamacpp_adapter
from .engines import maya as maya_adapter
from .engines import strata as strata_adapter
from .engines.base import route_exists, root_url
from .hardware import probe_hardware
from .media import media_plan, runner_plan
from .registry import (
    MODALITIES, MODALITY_LABELS, PHANTOM_ENGINE_NOTE, PHANTOM_ENGINES,
    ModelEntry, fit_report, fit_verdict, models_for, native_id_for,
)
from .targets import (SOURCE_DECLARED, ComputeTarget, add_target, budget_label,
                      get_target, load_targets, remove_target,
                      update_target, validate_target_fields)

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
# A quantized weights variant (A2: prefer FULL weights; a quant is picked only
# on request or when the repo offers no full weights). Conventional quant
# tokens: GGUF quants (q4, q4_k_m, Q8_0), int4/int8, fp8/fp4 (not fp16/bf16,
# which are full precision), and named methods (awq/gptq/exl/quant).
_QUANT_RE = re.compile(
    r"(?:^|[-_./])(?:"
    r"q[1-8](?:[_-][a-z0-9]+)?"
    r"|int[1-8]"
    r"|fp[48]"
    r"|nf4|awq|gptq|exl2?"
    r"|quant"
    r")(?=[-_./]|$)", re.IGNORECASE)
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
# --- /model integration (model-selection-integration.md) --------------------
# Persisted provider slugs are resolvable identities only: registered slugs
# ("strata" / "maya" — registered by the model-forge-providers package), the
# llamacpp aliases, or the custom class (custom / custom:<name>). Engine
# codenames are never slugs.
_CUSTOM_PROVIDER_API_MODE = "chat_completions"
# One state vocabulary everywhere: the activation plan, the approval prompt
# and the apply output all describe the persist with the same sentence.
_NEXT_SESSION_NOTE = "takes effect on your next session"
# The media providers (model-forge-image / model-forge-video) register under
# this provider name; the media_setup hints must say exactly this.
_MEDIA_PROVIDER_NAME = "model-forge"


def custom_provider_name(target_id: str) -> str:
    """Entry name whose custom:<name> identity is ``custom:<target-id>``:
    legacy ``custom_providers`` entries key their slug on the normalized
    display name (hermes_cli.providers.custom_provider_slug)."""
    return (target_id or "").strip().lower().replace(" ", "-")


def custom_provider_slug_for(target_id: str) -> str:
    """The resolvable ``custom:<target-id>`` identity for a compute target
    (slug rules: custom / custom:<name> / registered slugs / llamacpp
    aliases — never invented engine codenames)."""
    return "custom:" + custom_provider_name(target_id)


def custom_provider_entry(model_id: str, *, name: str, base_url: str) -> dict:
    """The ``custom_providers`` entry declaring a Model Forge-served model —
    the documented discovery surface that makes it selectable via ``/model``
    (a typed model id a custom_providers entry DECLARES routes to that entry
    automatically). Declared models merge on update; per-model metadata is
    written only where it is actually known."""
    return {
        "name": name,
        "base_url": base_url,
        "api_mode": _CUSTOM_PROVIDER_API_MODE,
        "models": {model_id: {}},
    }


def merge_custom_provider_entry(entry: dict, providers) -> tuple[list, dict]:
    """Merge *entry* into a ``custom_providers`` list and return
    ``(rows, merged)``. Identity: the entry's ``custom:<name>`` slug (its
    name), else slash-normalized base_url + api_mode — the reference's match
    rule. Declared models merge: a later activation or /v1/models probe never
    drops an already-declared model id."""
    rows = [r for r in (providers or []) if isinstance(r, dict)]
    name = str(entry.get("name") or "").strip()
    base = str(entry.get("base_url") or "").strip()
    mode = str(entry.get("api_mode") or "").strip()

    def _declared(models) -> dict:
        if isinstance(models, dict):
            return {str(k): (dict(v) if isinstance(v, dict) else {})
                    for k, v in models.items()}
        if isinstance(models, list):
            return {str(m): {} for m in models if isinstance(m, str)}
        return {}

    merged = None
    out = []
    for row in rows:
        row_name = str(row.get("name") or "").strip()
        row_base = str(row.get("base_url") or row.get("url") or "").strip()
        same_name = bool(name) and row_name == name
        same_route = (bool(base) and bool(row_base)
                      and row_base.rstrip("/").lower() == base.rstrip("/").lower()
                      and str(row.get("api_mode") or "").strip() == mode)
        if merged is None and (same_name or same_route):
            row = dict(row)
            declared = _declared(row.get("models"))
            for mid, meta in (entry.get("models") or {}).items():
                declared.setdefault(mid, dict(meta) if isinstance(meta, dict) else {})
            row["models"] = declared
            if same_name:
                # The slug identity moved with its target: point it at the
                # endpoint the activation just proved out.
                row["name"] = name
                row["base_url"] = base
                row["api_mode"] = mode
            merged = row
            out.append(row)
        else:
            out.append(row)
    if merged is None:
        merged = dict(entry)
        out.append(merged)
    return out, merged


def write_custom_provider_entry(entry: dict) -> dict:
    """Register the entry in config.yaml's ``custom_providers`` list (the
    /model discovery surface) and save. Raises on config failure — callers
    report the failure honestly instead of hiding it."""
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    rows, merged = merge_custom_provider_entry(entry, cfg.get("custom_providers"))
    cfg["custom_providers"] = rows
    save_config(cfg)
    return merged


def write_auxiliary_vision(model: str, base_url: str) -> dict:
    """Point Hermes' ``auxiliary.vision`` lane at a Model Forge facade: the
    vision lane's model + base_url, saved through the SAME in-process
    ``hermes_cli.config`` load/save API ``write_custom_provider_entry`` uses.

    Never a dotted ``hermes config set`` path: the CLI splits a key on '.',
    so ``auxiliary.vision.model`` would create three nested nodes instead of
    the flat ``auxiliary: {vision: {model: ...}}`` Hermes reads. Raises on
    config failure — callers report it honestly instead of hiding it."""
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    auxiliary = cfg.get("auxiliary")
    if not isinstance(auxiliary, dict):
        auxiliary = {}
    vision = auxiliary.get("vision")
    if not isinstance(vision, dict):
        vision = {}
    vision["model"] = model
    vision["base_url"] = base_url
    auxiliary["vision"] = vision
    cfg["auxiliary"] = auxiliary
    save_config(cfg)
    return dict(vision)


# Engine provisioning: the vendor-official installers Model Forge's
# engine_install action may plan and run. The command is never model-supplied
# — only these fixed, PINNED specs can execute, and only after the
# human-approval gate approves the exact command (rule_key
# model-forge:engine_install). Each entry pins its provenance so what runs
# cannot change while the pinned value stands still: an installer script is
# downloaded to a temp file and its SHA-256 checked against a pinned constant
# before anything runs; the Maya clone is checked out at a fixed commit and
# the checked-out HEAD re-verified before setup.sh runs. A checksum mismatch,
# a download failure or a checkout mismatch runs NOTHING. llama.cpp is
# deliberately absent: Hermes core manages a llama.cpp runtime
# (hardware-priced Local Models catalog and downloads), so engine_install
# returns guidance to it instead of installing binaries.
_ENGINE_INSTALL_ROOT = "${TMPDIR:-/tmp}/model-forge-installers"


def _pipeline(key: str, spec: dict) -> str:
    """Plain-wording description of the PINNED acquisition step an installer
    entry performs before it runs anything — download + checksum verification
    for a script, clone + pinned checkout + HEAD re-verification for git.
    Describes what verifies honestly (Python checks the SHA-256; there is no
    shell ``sha256sum`` pipeline), so the shown plan never advertises a
    command that does not run."""
    base = f"{_ENGINE_INSTALL_ROOT}/{key}"
    if spec["kind"] == "script":
        dest = f"{base}/{spec['url'].rsplit('/', 1)[-1]}"
        return (f"download {spec['url']} to {dest}, verify its SHA-256 "
                f"{spec['sha256']} in Python (a mismatch or download failure "
                f"runs nothing), then stage the verified local copy")
    dest = f"{base}/project-maya"
    return (f"git clone {spec['repo']} {dest}, git checkout "
            f"{spec['commit']}, re-verify the checked-out HEAD matches that "
            f"commit (a mismatch runs nothing), then stage the pinned "
            f"checkout")


def _installer_dest(key: str, spec: dict) -> str:
    """The on-disk path the pinned artifact is staged to for ``key``. Resolved
    through ``_install_root`` every time, so the path the plan and the
    approval message name is byte-for-byte the path the stager writes and the
    runner is handed — one root, never two."""
    base = os.path.join(_install_root(), key)
    if spec["kind"] == "script":
        return os.path.join(base, spec["url"].rsplit("/", 1)[-1])
    return os.path.join(base, "project-maya")


def _installer_command(spec: dict, dest: str) -> str:
    """The literal shell command the installer runner is handed for the
    staged artifact at ``dest`` — the local script for a script installer, or
    ``./setup.sh`` inside the pinned checkout for git. This is the ONE
    construction: the plan, the approval message and the stager all call it,
    so the approved text is exactly the command that runs (same string, not
    merely shell-equivalent)."""
    if spec["kind"] == "script":
        return f"{spec['runner']} {shlex.quote(dest)}"
    return f"cd {shlex.quote(dest)} && ./setup.sh"


def _installer_entry(key: str, *, vendor: str, kind: str, steps,
                     **pinned) -> dict:
    """One approved-installer catalog entry: vendor, acquisition kind, the
    pinned provenance fields, the human steps, and the plain-wording pinned
    pipeline. The literal executed command is NOT baked here — it is built on
    demand from the same ``_installer_command`` helper the stager uses, so the
    shown command and the run command can never be two different strings."""
    spec = {"vendor": vendor, "kind": kind, **pinned, "steps": list(steps)}
    spec["pipeline"] = _pipeline(key, spec)
    return spec


_ENGINE_INSTALLERS = {
    "ollama": _installer_entry(
        "ollama", vendor="Ollama", kind="script",
        url="https://ollama.com/install.sh",
        sha256="25f64b810b947145095956533e1bdf56eacea2673c55a7e586be4515fc882c9f",
        runner="sh",
        steps=["download the Ollama official install script and verify its "
               "SHA-256 (https://ollama.com/install.sh, "
               "25f64b810b947145095956533e1bdf56eacea2673c55a7e586be4515fc882c9f)",
               "run the verified local copy with sh — a checksum mismatch "
               "runs nothing"]),
    "strata": _installer_entry(
        "strata", vendor="Strata", kind="script",
        url=("https://github.com/Niko1221/Strata/releases/download/v0.2.15/"
             "install.sh"),
        sha256="1fcfbab133880249b7a6eaefb643f25b29a9b8f45624d2d8673e35948c329eda",
        runner="sh",
        steps=["download the Strata v0.2.15 release install script and verify "
               "its SHA-256 (https://github.com/Niko1221/Strata/releases/"
               "download/v0.2.15/install.sh, "
               "1fcfbab133880249b7a6eaefb643f25b29a9b8f45624d2d8673e35948c329eda)",
               "run the verified local copy with sh — a checksum mismatch "
               "runs nothing"]),
    "maya": _installer_entry(
        "maya", vendor="Project Maya", kind="git",
        repo="https://github.com/mw00/project-maya",
        commit="56b5216ab4464ac41c26b1be32eb63b98de53d0a",
        steps=["git clone https://github.com/mw00/project-maya",
               "git checkout 56b5216ab4464ac41c26b1be32eb63b98de53d0a and "
               "verify the checked-out HEAD — a mismatch runs nothing",
               "./setup.sh (run from inside the pinned checkout)"]),
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


def _install_root() -> str:
    """Resolved install root, read from the environment at call time. The plan,
    the approval message and the stager all resolve it through this one helper,
    so the shown path and the staged path never diverge."""
    return os.path.join(os.environ.get("TMPDIR") or "/tmp",
                        "model-forge-installers")


def _fetch_url(url: str, timeout: float = 60.0) -> bytes:
    """Download a pinned installer script. Seam: tests replace this so no
    network call happens; failures raise and the caller turns them into an
    honest error row."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _git_run(args: list[str], cwd: str | None = None):
    """Run one git command for the Maya pin. Seam: tests replace this so no
    real clone happens. The child gets the minimal environment (no Hermes
    credentials) and ``GIT_TERMINAL_PROMPT=0`` so a clone can never hang on
    an interactive credential prompt — it fails instead."""
    env = {**_minimal_env(), "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env=env)


def _proc_tail(proc) -> str:
    out = getattr(proc, "stdout", b"") or b""
    if isinstance(out, bytes):
        out = out.decode("utf-8", "replace")
    return out.strip()[-400:]


def _stage_installer(engine_id: str, *, downloader=None, git=None) -> dict:
    """Fetch and VERIFY the pinned installer artifact — the only thing the
    runner seam is ever handed. Returns ``{"ok": True, "artifact",
    "command"}`` or ``{"ok": False, "error"}``; never raises and never runs
    any part of the vendor installer. A checksum mismatch, a download
    failure or a checkout mismatch leaves nothing to run."""
    key = _installer_key(engine_id)
    spec = _ENGINE_INSTALLERS.get(key)
    if spec is None:
        return {"ok": False,
                "error": f"no approved installer for '{key}' — nothing was run"}
    downloader = downloader or _fetch_url
    git = git or _git_run
    base = os.path.join(_install_root(), key)
    dest = _installer_dest(key, spec)
    try:
        if spec["kind"] == "script":
            try:
                data = downloader(spec["url"])
            except Exception as exc:  # noqa: BLE001 — an honest error row
                return {"ok": False,
                        "error": (f"download of {spec['url']} failed: "
                                  f"{type(exc).__name__}: {exc} — nothing "
                                  "was run")}
            digest = hashlib.sha256(data).hexdigest()
            if digest != spec["sha256"]:
                return {"ok": False,
                        "error": (f"checksum mismatch for {spec['url']}: "
                                  f"expected {spec['sha256']}, got {digest} — "
                                  "nothing was run")}
            os.makedirs(base, exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(data)
            return {"ok": True, "artifact": dest,
                    "command": _installer_command(spec, dest)}
        # git: clone, check out the pinned commit, re-verify the HEAD.
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(base, exist_ok=True)
        clone = git(["clone", spec["repo"], dest])
        if getattr(clone, "returncode", 1) != 0:
            return {"ok": False,
                    "error": (f"git clone of {spec['repo']} failed: "
                              f"{_proc_tail(clone)} — nothing was run")}
        co = git(["-C", dest, "checkout", spec["commit"]])
        if getattr(co, "returncode", 1) != 0:
            shutil.rmtree(dest, ignore_errors=True)
            return {"ok": False,
                    "error": (f"git checkout of {spec['commit']} failed: "
                              f"{_proc_tail(co)} — nothing was run")}
        head = git(["-C", dest, "rev-parse", "HEAD"])
        got = _proc_tail(head)
        if getattr(head, "returncode", 1) != 0 or got != spec["commit"]:
            shutil.rmtree(dest, ignore_errors=True)
            return {"ok": False,
                    "error": (f"checkout mismatch for {spec['repo']}: "
                              f"expected HEAD {spec['commit']}, got "
                              f"{got or '(unknown)'} — nothing was run")}
        return {"ok": True, "artifact": dest,
                "command": _installer_command(spec, dest)}
    except Exception as exc:  # noqa: BLE001 — the job record carries it
        return {"ok": False,
                "error": f"{type(exc).__name__}: {exc} — nothing was run"}


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
        action = str(args.get("action") or "add").strip() or "add"
        if action in ("preview", "retest"):
            return None    # read-only probes: no gate, no write
        if action == "remove":
            tid = str(args.get("id") or "")
            try:
                row = get_target(tid)
                current = (_dump(_target_registration_row(row))
                           if row is not None else "")
            except Exception:  # noqa: BLE001 — uncertainty -> escalate text
                current = ""
            return {
                "action": "approve",
                "message": (
                    "Model Forge wants to REMOVE a registered compute "
                    "target. Engine probes and model fit no longer cover "
                    "this machine; nothing on it is touched or deleted:\n"
                    f"  id: {args.get('id')}\n"
                    + (f"  registration: {current}\n" if current else "") +
                    "Removing only forgets the registration. Approve to "
                    "remove it; denying or letting this prompt time out "
                    "changes nothing."),
                "rule_key": "model-forge:target_remove",
            }
        if action == "edit":
            tid = str(args.get("id") or "")
            try:
                current = get_target(tid)
                before = (_target_registration_row(current)
                          if current is not None else "(unknown target)")
            except Exception:  # noqa: BLE001 — uncertainty -> escalate text
                before = "(could not read the current registration)"
            return {
                "action": "approve",
                "message": (
                    "Model Forge wants to EDIT a registered compute target. "
                    "Engine probes, media jobs (including image data) and "
                    "downloads may then go to the changed address:\n"
                    f"  id: {args.get('id')}\n"
                    f"  before: {before}\n"
                    f"  changes: {json.dumps({k: v for k, v in args.items() if k not in ('action', 'id')})}"),
                "rule_key": "model-forge:target_edit",
            }
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
        slug = custom_provider_slug_for(str(args.get("target") or "local"))
        shown_engine_model = args.get("engine_model") or "(registry default)"
        return {
            "action": "approve",
            "message": (
                "Model Forge wants to rewrite your default model config "
                "(model.default, model.provider, base_url) and register the "
                "custom_providers entry " + slug + " (declaring the served "
                "model) so future sessions chat through this endpoint and "
                "/model can switch to it:\n"
                f"  model_id: {args.get('model_id')}\n"
                f"  target: {args.get('target') or 'local'}\n"
                f"  engine_model: {shown_engine_model}\n"
                f"  custom_providers: {slug} (declares the served model "
                f"id)"),
            "rule_key": "model-forge:activate",
        }
    if tool_name == "model_forge_install" \
            and args.get("action") == "engine_install" \
            and args.get("execute") is True:
        spec = _installer_for(str(args.get("engine") or ""))
        if spec is not None:
            key = _installer_key(str(args.get("engine") or ""))
            # The SAME construction the stager runs — the approved text and the
            # executed command are one string (see _installer_command).
            command = _installer_command(spec, _installer_dest(key, spec))
            if spec["kind"] == "script":
                pin = (f"  pinned sha256: {spec['sha256']}\n"
                       f"  source: {spec['url']}\n")
            else:
                pin = (f"  pinned commit: {spec['commit']}\n"
                       f"  source: {spec['repo']}\n")
            return {
                "action": "approve",
                "message": (
                    "Model Forge wants to install the "
                    f"{spec['vendor']} inference engine on this machine by "
                    "running its PINNED installer in a background job. The "
                    "pinned artifact is acquired and verified first (a "
                    "checksum or checkout mismatch runs nothing); the runner "
                    "then executes EXACTLY this command:\n"
                    f"  command: {command}\n"
                    f"  acquire: {spec['pipeline']}\n"
                    + pin +
                    f"  engine: {key}\n"
                    "Progress is pollable with model_forge_install job_id, "
                    "and a hard timeout stops the installer if it runs too "
                    "long. Approve to run it — denying or letting this "
                    "prompt time out runs nothing."),
                "rule_key": "model-forge:engine_install",
            }
        return None   # no approved installer -> nothing can execute
    if tool_name == "model_forge_broker" \
            and args.get("action") in ("claim", "swap"):
        # A claim/swap can start an engine SERVER process (vLLM/llama.cpp),
        # which is a model-callable side effect the model must not make
        # alone. dry_run swaps change nothing, so they never escalate.
        if not (args.get("action") == "swap"
                and _swap_dry_run(args.get("dry_run"))):
            detail = _broker_would_start_server(args)
            if detail is not None:
                return {
                    "action": "approve",
                    "message": (
                        "Model Forge wants to start an inference engine "
                        "SERVER process on a compute target (a "
                        f"{detail['engine']} server for a claim/swap). "
                        "Starting an engine server requires your approval:\n"
                        f"  engine: {detail['engine']}\n"
                        f"  model: {detail['model_id']}\n"
                        f"  target: {detail['target']}\n"
                        f"  port: {detail['port']}\n"
                        "  bind: 127.0.0.1 only — loopback, never all "
                        "interfaces\n"
                        "Approving starts the server; denying or letting "
                        "this prompt time out starts nothing."),
                    "rule_key": "model-forge:broker_server_start",
                }
    if tool_name == "model_forge_broker" and args.get("action") == "release":
        # A release restores what was resident at claim time; if that
        # restore would load a non-resident process-engine model, the
        # release SPAWNS a server process. Same gate as a starting claim.
        detail = _release_would_start_server(args)
        if detail is not None:
            return {
                "action": "approve",
                "message": (
                    "Model Forge wants to RELEASE a workflow scope, which "
                    "would RE-START an inference engine SERVER process on a "
                    "compute target to restore what was resident at claim "
                    f"time (a {detail['engine']} server). Starting an engine "
                    "server requires your approval:\n"
                    f"  engine: {detail['engine']}\n"
                    f"  model: {detail['model_id']}\n"
                    f"  target: {detail['target']}\n"
                    f"  port: {detail['port']}\n"
                    "  bind: 127.0.0.1 only — loopback, never all "
                    "interfaces\n"
                    "Approving restarts the server; denying or letting "
                    "this prompt time out starts nothing."),
                "rule_key": "model-forge:broker_server_start",
            }
    if tool_name == "model_forge_generate_media":
        # A media job claims the GPU lane around itself and releases it when
        # the job ends; the release restores what was resident at claim time.
        # If that restore would load a non-resident process-engine model, the
        # release SPAWNS a server process — gate it like a starting claim.
        detail = _media_would_start_server(args)
        if detail is not None:
            return {
                "action": "approve",
                "message": (
                    "Model Forge wants to run a media job whose GPU-lane "
                    "claim/release would start or RE-START an inference "
                    f"engine SERVER process (a {detail['engine']} server). "
                    "Starting an engine server requires your approval:\n"
                    f"  engine: {detail['engine']}\n"
                    f"  model: {detail['model_id']}\n"
                    f"  target: {detail['target']}\n"
                    f"  port: {detail['port']}\n"
                    "  bind: 127.0.0.1 only — loopback, never all "
                    "interfaces\n"
                    "Approving starts the server; denying or letting "
                    "this prompt time out starts nothing."),
                "rule_key": "model-forge:broker_server_start",
            }
    # B8: stopping an externally-started server (a runtime the user started
    # themselves, e.g. a docker container brought up by compose) rides the
    # approval gate too. This is provenance-scoped: a plugin-owned runtime
    # was gated at start, so stopping it is symmetric and ungated; only a
    # stop that would take down a server Model Forge did NOT start prompts.
    if (tool_name == "model_forge_broker"
            and args.get("action") in ("claim", "swap", "release")
            and not (args.get("action") == "swap"
                     and _swap_dry_run(args.get("dry_run")))):
        detail = _broker_would_stop_external(args)
        if detail is not None:
            return {
                "action": "approve",
                "message": (
                    "Model Forge wants to STOP an inference engine runtime "
                    "that YOU started yourself (an externally-started "
                    f"server: a {detail['engine']} runtime). This swap/"
                    "release would stop a process you run, so it needs your "
                    "approval:\n"
                    f"  engine: {detail['engine']}\n"
                    f"  model: {detail['model_id']}\n"
                    f"  target: {detail['target']}\n"
                    "  stops: a server Model Forge did NOT start\n"
                    "Approving stops the server (its weights stay on disk "
                    "and it can be re-loaded); denying or letting this "
                    "prompt time out stops nothing."),
                "rule_key": "model-forge:broker_server_stop",
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


# --- user-declared engine endpoints (Desktop settings) -----------------------

def _settings_engines() -> list[dict]:
    """Engine endpoints the user declared in Desktop Settings -> Plugins ->
    Model Forge (config_schema ``engines_json``): the bring-your-own-engine
    path — point Model Forge at any engine HTTP API.

    Returns normalized rows [{"label", "base_url", "source": "user-declared"}].
    Rows without a usable http(s) base_url are skipped fail-soft (a malformed
    settings row must never sink a tool call); anything unusual in the config
    read degrades to "no declared engines", never an error. Stdlib-only."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        settings = (entries.get("model-forge") or {}).get("settings") or {}
        rows = settings.get("engines_json") or []
    except Exception:  # noqa: BLE001 — outside a Hermes process, or unreadable
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        base_url = str(row.get("base_url") or "").strip().rstrip("/")
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            continue
        out.append({"label": str(row.get("label") or row.get("id") or base_url),
                    "base_url": base_url, "source": "user-declared"})
    return out


def declared_engine_roots() -> list[str]:
    """The extra_roots seam input: base URLs of user-declared engine endpoints
    (deduplicated, order kept)."""
    seen, out = set(), []
    for row in _settings_engines():
        if row["base_url"] not in seen:
            seen.add(row["base_url"])
            out.append(row["base_url"])
    return out


def _detect(base_url: str):
    """Engine detection with the user-declared engine endpoints folded in.

    ``engines.detect_engines`` takes ``extra_roots`` (user-declared engine
    base URLs) and labels the rows it finds there ``source: user-declared``;
    auto-detected rows are unchanged. ``extra_roots`` is passed only when the
    user actually declared endpoints, so the zero-config path is exactly the
    plain auto-detection call."""
    extra = declared_engine_roots()
    if extra:
        return detect_engines(base_url, extra_roots=extra)
    return detect_engines(base_url)


# --- the pointing invitation (honest empty state) ----------------------------

# What to tell a user who has no GPU and no compute targets yet. Product
# copy: name the exact next steps (the tool/action, and the settings field).
POINTING_INVITATION_STEPS = (
    "Add a cloud GPU or another machine as a compute target with "
    "model_forge_target_add (id, base_url, and vram_gb / ram_gb for fit "
    "estimates) — the human-approval gate confirms each target before it "
    "is used.",
    "Or declare an engine endpoint in Settings -> Plugins -> Model Forge "
    "('Declared engine endpoints', the engines_json field): each entry is "
    "a label plus the engine's HTTP API base URL, and Model Forge probes "
    "it everywhere engines are detected.",
)

def pointing_invitation(targets, hw, declared=None) -> dict | None:
    """The guided pointing flow's invitation — or None when the setup already
    has compute.

    Emitted exactly when hardware detection finds no GPU on this machine and
    no compute targets are registered (the built-in 'local' row does not
    count): instead of a bare empty state, the caller gets an honest
    invitation naming the exact next steps — register a compute target via
    model_forge_target_add, or declare an engine endpoint in settings. When
    engine endpoints are already declared, the invitation says so instead of
    telling the user to redo it. Pure: no I/O."""
    has_gpu = bool(getattr(hw, "gpus", ()) or getattr(hw, "total_vram_gb", 0.0) > 0)
    user_targets = [t for t in targets
                    if getattr(t, "id", None) not in ("", "local")]
    if has_gpu or user_targets:
        return None
    steps: list[str] = list(POINTING_INVITATION_STEPS)
    if declared:
        steps[1] = (steps[1] + f" ({len(declared)} endpoint(s) already "
                    f"declared — Model Forge probes them automatically.)")
    return {
        "reason": "No GPU was detected on this machine and no compute "
                  "targets are registered, so there is nothing to price "
                  "models against yet.",
        "next_steps": steps,
    }


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
    model_id = engine_model or native_id_for(entry.id, engine)
    base_v1 = root_url(target.base_url) + "/v1"
    slug = custom_provider_slug_for(target.id)
    cp_entry = custom_provider_entry(model_id, name=custom_provider_name(target.id),
                                     base_url=base_v1)
    return {
        "ok": True,
        "action": "persist_model_selection",
        "provider": provider,
        "model": model_id,
        "base_url": base_v1,
        "custom_provider": {
            "slug": slug,
            "entry": cp_entry,
            "applied": False,
            "write": ("registers the custom_providers entry " + slug
                      + " declaring '" + model_id + "' so /model can "
                      "select it"),
        },
        "detail": ("writes model.default + model.provider (+ base_url) via "
                  "the documented hermes_cli.model_setup helper and "
                  "registers the custom_providers entry " + slug + " "
                  "(declaring '" + model_id + "') so /model can switch to "
                  "it; " + _NEXT_SESSION_NOTE),
    }


# --- facade activation: the two Hermes consumer surfaces (P5c C2) -----------
#
# Pointing a Hermes install at a model-forge facade is ONE activation that must
# cover BOTH config surfaces the live consumers read:
#
#   * ``custom_providers`` — a LIST, merged in-process with the existing
#     merge_custom_provider_entry helper (identity: the ``custom:<name>`` slug,
#     else slash-normalized base_url + api_mode). The entry is computed by the
#     existing custom_provider_entry / merge_custom_provider_entry helpers; the
#     declared model list is refreshed from the facade's ``/v1/models`` when it
#     answers (fail-soft, with an honest 'models not refreshed' note otherwise).
#   * ``auxiliary.vision`` — the vision lane (model + base_url) the house-scan
#     vision cron reads.
#
# Both surfaces are written IN-PROCESS through the same ``hermes_cli.config``
# load/save API ``write_custom_provider_entry`` uses. Dotted ``hermes config
# set`` index writes were abandoned: the CLI splits a key on '.' (so a
# canonical model id like ``Qwen/Qwen3.8-27B-FP8`` shatters into nested keys),
# cannot grow a list (a fresh config becomes an unread MAPPING ``{'0': {...}}``,
# and appending at ``len(rows)`` raises IndexError). Model ids are canonicalized
# through the alias table first: the facade resolves legacy aliases anyway, so
# the config should name the canonical id ``/v1/models`` actually advertises.
_FACADE_MODELS_TIMEOUT_S = 2.0


def _fetch_facade_models(base_v1: str,
                         timeout: float = _FACADE_MODELS_TIMEOUT_S) -> list[str]:
    """Model ids advertised by a facade's ``/v1/models``. Seam: tests replace
    this so no network call happens. Raises on failure — the caller turns the
    failure into an honest 'models not refreshed' note, never a hang (the
    request carries a short timeout)."""
    url = base_v1.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", "replace"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("unexpected /v1/models payload (no data list)")
    ids: list[str] = []
    for row in data:
        mid = row.get("id") if isinstance(row, dict) else None
        if isinstance(mid, str) and mid:
            ids.append(mid)
    return ids


def _current_custom_providers() -> list[dict]:
    """The current ``custom_providers`` list — the merge helper needs the live
    list so an existing entry is updated in place, never duplicated. Fail-soft:
    an unreadable config reads as no providers, so an activation never crashes
    on config access."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        rows = cfg.get("custom_providers")
        return ([r for r in rows if isinstance(r, dict)]
                if isinstance(rows, list) else [])
    except Exception:  # noqa: BLE001 — outside Hermes, or unreadable config
        return []


def _facade_catalog_entry(requested: str, target: ComputeTarget):
    """Registry-free entry for a facade id the target's ``/v1/models`` catalog
    advertises (P5 §8: the facade chat id ``Qwen/Qwen3.8-27B-FP8`` is a
    CATALOG id, not a curated-registry entry — ``model_forge_activate`` used
    to refuse it and activation fell back to manual operator steps).

    Acceptance is fail-closed: the id must be present in the target's own
    catalog, or ``aliases.resolve`` must map it to a cataloged id (the facade
    resolves aliases anyway; the config writes the CANONICAL id). An
    unreachable catalog refuses with an honest reason rather than activating
    an id nobody could verify — the fetch stays fail-soft for the plan's
    model LIST, but presence is the acceptance criterion here.

    The returned entry is synthetic: modality chat (activation connects a
    chat/vision default) and engines covering what an OpenAI-compatible
    facade can front — the match only selects the provider slug, which is
    ``custom`` for all of them, and the plan path (plus the human-approval
    gate) is the registry one, unchanged. Returns ``(entry, refusal)`` —
    exactly one of the two is set.
    """
    base_v1 = root_url(target.base_url) + "/v1"
    canonical = resolve(requested)
    try:
        catalog = _fetch_facade_models(base_v1)
    except Exception as exc:  # noqa: BLE001 — honest refusal, never a raise
        return None, (f"unknown model_id: {requested!r} is not in the curated "
                      f"registry and {base_v1}/models could not be fetched to "
                      f"confirm it ({type(exc).__name__}: {exc}) — refusing "
                      f"an unverifiable id (fail-closed)")
    names = set(catalog)
    if requested not in names and canonical not in names:
        return None, (f"unknown model_id: {requested!r} is neither in the "
                      f"curated registry nor advertised by {base_v1}/models "
                      f"(its aliases resolve to {canonical!r}, also absent)")
    entry = ModelEntry(id=canonical, name=canonical, modality="chat",
                       engines=("openai-compatible", "vllm", "sglang"),
                       source=canonical)
    return entry, None


def _facade_plan_surfaces(plan: dict, *, providers, loader=None) -> dict:
    """Augment an ok activation *plan* for a FACADE target with both Hermes
    consumer surfaces (custom_providers + auxiliary.vision).

    Pure apart from the read-only ``/v1/models`` probe (injectable, fail-soft).
    The written model id is the CANONICAL id: a legacy alias is resolved
    through :mod:`model_forge.aliases` first, because the facade resolves
    aliases anyway and the config should name the id ``/v1/models`` advertises
    (naming what the endpoint actually serves, not a pre-rename id)."""
    base_v1 = plan["base_url"]
    model = resolve(str(plan.get("model") or ""))
    plan["model"] = model

    cp = plan["custom_provider"]
    cp_entry = dict(cp["entry"])
    cp_entry["models"] = {model: {}}
    note = ""
    try:
        refreshed = (loader or _fetch_facade_models)(base_v1)
        for mid in refreshed:
            cp_entry["models"].setdefault(resolve(str(mid)), {})
    except Exception as exc:  # noqa: BLE001 — fail soft, honest note
        note = (f"models not refreshed from {base_v1}/models: "
                f"{type(exc).__name__}: {exc}; the custom_providers entry "
                "declares only the activated model")
    cp["entry"] = cp_entry
    _, cp["merged"] = merge_custom_provider_entry(cp_entry, providers)
    cp["write"] = ("registers the custom_providers entry " + cp["slug"]
                   + " (merged into the list) declaring "
                   + ", ".join(cp_entry["models"]))
    if note:
        cp["models_note"] = note
    plan["auxiliary_vision"] = {
        "model": model,
        "base_url": base_v1,
        "write": ("sets auxiliary.vision.model + auxiliary.vision.base_url so "
                  "the vision lane chats through the facade endpoint"),
        "applied": False,
    }
    plan["facade"] = True
    return plan


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
    pinned = ({"url": spec["url"], "sha256": spec["sha256"]}
              if spec["kind"] == "script"
              else {"repo": spec["repo"], "commit": spec["commit"]})
    # Built by the same helper the stager runs, from the same artifact path —
    # the planned command IS the executed command, byte for byte.
    command = _installer_command(spec, _installer_dest(key, spec))
    return {"ok": True, "action": "vendor_installer", "engine": key,
            "vendor": spec["vendor"], "command": command,
            "pipeline": spec["pipeline"],
            "steps": list(spec["steps"]), "executes": True, "pinned": pinned,
            "detail": f"the {spec['vendor']} pinned install — the artifact is "
                      "verified against the pinned release tag / commit / "
                      "SHA-256 before it runs (command is the exact staged "
                      "command); it runs only with execute=true after the "
                      "human-approval gate approves "
                      "this exact command "
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
        statuses = _detect("")
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


def _minimal_env() -> dict:
    """The smallest environment an installer child needs: PATH, HOME, LANG
    and TMPDIR only. Nothing else from the parent is forwarded — no provider
    keys, tokens or other credentials reach the vendor install script. The
    caller names what the child needs; this helper adds nothing on its own."""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME") or os.path.expanduser("~"),
        "LANG": os.environ.get("LANG", "C"),
        "TMPDIR": os.environ.get("TMPDIR") or "/tmp",
    }


def _run_installer(command: str, timeout_s: float) -> dict:
    """Run the approved, verified installer command in its own session with a
    minimal environment (see ``_minimal_env`` — no Hermes credentials reach
    the vendor script). The hard timeout kills the whole process group so
    nothing can hang unattended; on timeout this raises
    subprocess.TimeoutExpired (the job record says 'timeout', never a false
    'done')."""
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            env=_minimal_env())
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


def _engine_install_job(job_id: str, plan: dict, runner=None,
                        stager=None) -> None:
    """Background installer job (daemon thread, the pull.py pattern): stages
    and VERIFIES the pinned artifact first, then writes progress records to
    the shared jobs store (polled with model_forge_install job_id), enforces
    the hard timeout, and confirms the install by re-running engine
    detection. A staging failure (checksum mismatch, download failure,
    checkout mismatch) is an honest error row that ran nothing."""
    from . import pull
    state = {"job_id": job_id, "kind": "engine_install",
             "engine": plan["engine"], "vendor": plan["vendor"],
             "command": plan["command"], "state": "running",
             "hard_timeout_s": _ENGINE_INSTALL_TIMEOUT_S}
    pull._write_job(job_id, state)
    try:
        stager = stager or _stage_installer
        staged = stager(plan["engine"])
        if not staged.get("ok"):
            state["state"] = "error"
            state["error"] = (staged.get("error")
                              or "the installer artifact could not be "
                                 "verified — nothing was run")
            pull._write_job(job_id, state)
            return
        state["artifact"] = staged.get("artifact")
        state["executed_command"] = staged["command"]
        runner = runner or _run_installer
        result = runner(staged["command"], _ENGINE_INSTALL_TIMEOUT_S)
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


def start_engine_install(engine_id: str, runner=None, stager=None) -> dict:
    """Start the approved vendor installer as a background job (daemon
    thread + jobs-store records). Call only after the human-approval gate
    approved the exact command — the plan (engine_install_plan) comes first.
    The job stages and verifies the pinned artifact before running anything."""
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
    threading.Thread(target=_engine_install_job,
                     args=(job_id, plan, runner, stager),
                     daemon=True).start()
    return {"ok": True, "job_id": job_id,
            "poll": f"model_forge_install with job_id={job_id}"}


# --- tools -------------------------------------------------------------------

def model_forge_status(args: dict | None = None, **kwargs) -> str:
    """Cockpit inventory: hardware + engine inventory across all compute
    targets, the user-declared engine endpoints (settings), the
    model-storage listing (the model_forge_storage shape: sizes and
    modification times), and — when hardware detection finds no GPU and no
    compute targets exist — the pointing invitation naming the exact next
    steps instead of a bare empty state."""
    try:
        declared = _settings_engines()
        targets = load_targets()
        out = {"targets": _target_rows(targets),
               "declared_engines": [dict(row) for row in declared],
               "storage": _storage_listing(),
               "pull_credentials": credentials_status()}
        local = next((t for t in targets if t.id == "local"), targets[0])
        invitation = pointing_invitation(
            targets, _hw_for_target(local), declared=declared)
        if invitation is not None:
            out["invitation"] = invitation
        return _dump(out)
    except Exception as exc:
        return _err(exc)


def _target_rows(targets) -> list[dict]:
    """Per-target hardware + engine rows (the status/cockpit data path)."""
    targets_out = []
    for t in targets:
        hw = _hw_for_target(t)
        targets_out.append({
            "id": t.id,
            "label": t.label,
            "kind": t.kind,
            "base_url": t.base_url,
            "budgets": {
                "vram_gb": t.vram_gb,
                "ram_gb": t.ram_gb,
                "vram_source": budget_label(t.vram_source, t.vram_gb),
                "ram_source": budget_label(t.ram_source, t.ram_gb),
            },
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
                    "source": getattr(e, "source", "") or "auto-detected",
                }
                for e in _detect("" if t.kind == "local" else t.base_url)
            ],
        })
    return targets_out


def _storage_listing() -> dict:
    """The model-storage inventory in the model_forge_storage list shape
    (entries with sizes and modification times) — the one data path behind
    model_forge_status, model_forge_storage, and the Desktop cockpit. Read-only."""
    from . import storage
    return storage.list_storage()


def _browse_model_rows(entries, hw, available) -> list[dict]:
    """Browse rows with fit data — shared by model_forge_browse and the
    Desktop cockpit data (the pane reuses exactly this fit data)."""
    return [
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
    ]


def model_forge_browse(args: dict | None = None, **kwargs) -> str:
    """Models for a modality, priced against a compute target (advisory).
    When hardware detection finds no GPU and no compute targets exist, the
    response carries the pointing invitation (named next steps) instead of
    leaving the caller at a bare empty state."""
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
        available = [e.id for e in _detect(base) if e.available]
        entries = models_for(modality=modality,
                             engine=args.get("engine") or None)
        out = {
            "target": target.id,
            "hardware_budget_gb": hw.budget_gb,
            "models": _browse_model_rows(entries, hw, available),
            "note": "Fit verdicts are advisory and cross-check the engines "
                    "detected on the target (see each row's fit block); "
                    "engines do authoritative memory sizing at launch. GGUF "
                    "chat models: use Hermes' built-in Local Models pane "
                    "(hardware-priced catalog).",
        }
        invitation = pointing_invitation(
            load_targets(),
            _hw_for_target(next((t for t in load_targets()
                                 if t.id == "local"), target)),
            declared=_settings_engines())
        if invitation is not None:
            out["invitation"] = invitation
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


def _target_engine_rows(base_url: str) -> list[dict]:
    """Engine probe rows for a target base URL (the preview/retest read path
    — probes only, never writes)."""
    return [
        {
            "id": e.id,
            "name": e.display_name,
            "available": e.available,
            "endpoint": e.endpoint,
            "detail": e.detail,
            "modalities": list(e.modalities),
            "models_loaded": list(e.models_loaded),
            "source": getattr(e, "source", "") or "auto-detected",
        }
        for e in _detect(base_url)
    ]


def _suggest_budgets(base_url: str) -> dict:
    """Read-only budget discovery (W1.3): ask the target's own engine APIs
    what memory it reports, and SUGGEST values the user confirms. Never
    writes a budget. One honest source today: ComfyUI /system_stats exposes
    device VRAM totals and system RAM. Anything else — no suggestion, never a
    guess."""
    root = root_url(base_url)
    if not root:
        return {"suggested": False,
                "note": "no base_url to probe — declare budgets yourself"}
    try:
        raw = _fetch_url(root + "/system_stats", timeout=5.0)
        stats = json.loads(raw.decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 — no data is an honest outcome
        return {"suggested": False,
                "note": ("no engine endpoint on this address reported memory "
                         "totals read-only — declare vram_gb/ram_gb yourself "
                         "(the values are labeled 'declared by the user')")}
    out = {"suggested": True,
           "source": "engine-reported (ComfyUI /system_stats)",
           "note": ("suggested values — pass them back as vram_gb/ram_gb to "
                    "confirm; nothing is written until you do")}
    try:
        vram = sum(float(d.get("vram_total") or 0)
                   for d in stats.get("devices") or ())
        if vram > 0:
            out["vram_gb"] = round(vram / (1024 ** 3), 2)
    except (TypeError, ValueError):
        pass
    try:
        ram = float((stats.get("system") or {}).get("ram_total") or 0)
        if ram > 0:
            out["ram_gb"] = round(ram / (1024 ** 3), 2)
    except (TypeError, ValueError):
        pass
    if "vram_gb" not in out and "ram_gb" not in out:
        return {"suggested": False,
                "note": ("the endpoint answered but reported no memory "
                         "totals — declare vram_gb/ram_gb yourself")}
    return out


def _target_registration_row(t: ComputeTarget) -> dict:
    """One target's registration row with honest budget provenance (W1.3)."""
    return {
        "id": t.id, "label": t.label, "kind": t.kind,
        "base_url": t.base_url,
        "vram_gb": t.vram_gb, "ram_gb": t.ram_gb, "note": t.note,
        "vram_source": budget_label(t.vram_source, t.vram_gb),
        "ram_source": budget_label(t.ram_source, t.ram_gb),
    }


def _target_preview(args: dict) -> dict:
    """action=preview — the guided-pointing probe preview (W1.1): validate
    the candidate address, probe it read-only, show exactly what answered and
    what a registration would store, and suggest budgets the user confirms.
    Writes nothing and never reaches the approval gate — the gate prompt
    comes later, on the action=add call."""
    tid = str(args.get("id", "")).strip()
    base_url = str(args.get("base_url", "")).strip()
    kind = str(args.get("kind", "")).strip() or "lan"
    if not tid or tid == "local":
        return {"ok": False, "action": "preview",
                "error": "id is required and cannot be 'local'"}
    err = validate_target_fields(base_url=base_url, kind=kind,
                                 vram_gb=args.get("vram_gb"),
                                 ram_gb=args.get("ram_gb"))
    if err:
        return {"ok": False, "action": "preview", "error": err}
    existing = get_target(tid)
    registration = {
        "id": tid,
        "label": str(args.get("label", "")).strip() or tid,
        "base_url": base_url, "kind": kind,
        "vram_gb": float(args.get("vram_gb") or 0.0),
        "ram_gb": float(args.get("ram_gb") or 0.0),
        "note": str(args.get("note", "")),
    }
    rows = _target_engine_rows(base_url)
    answered = [r for r in rows if r["available"]]
    return {
        "ok": True,
        "action": "preview",
        "writes": False,
        "registration": registration,
        "already_registered": existing is not None,
        "id_conflict_hint": (
            "" if existing is None or existing.base_url == base_url else
            f"id {tid!r} is already registered at {existing.base_url} — "
            f"action='add' would replace it with {base_url}; prefer "
            "action='edit' to correct the existing registration"),
        "probe": {
            "probed": base_url,
            "engines_answered": [r["id"] for r in answered],
            "engines": rows,
            "detail": (f"{len(answered)} engine endpoint(s) answered at "
                       f"{base_url}" if answered else
                       f"no known engine answered at {base_url} yet — the "
                       "registration is still valid (engines can start "
                       "later; every probe degrades to a stated reason)"),
        },
        "suggested_budgets": _suggest_budgets(base_url),
        "next": ("to register, re-run with action='add' and the same "
                 "fields; the human-approval gate shows the exact address "
                 "first, and denial registers nothing"),
    }


def _target_retest(args: dict) -> dict:
    """action=retest — read-only re-probe of a REGISTERED target's engines
    plus its recorded budgets (nothing is written)."""
    tid = str(args.get("id", "")).strip()
    t = get_target(tid or None) if tid else get_target(args.get("target"))
    if t is None:
        return {"ok": False, "action": "retest",
                "error": f"unknown target: {tid or args.get('target')!r} "
                         "(see model_forge_status)"}
    rows = _target_engine_rows("" if t.kind == "local" else t.base_url)
    return {
        "ok": True, "action": "retest", "writes": False,
        "target": _target_registration_row(t),
        "engines_answered": [r["id"] for r in rows if r["available"]],
        "engines": rows,
        "detail": ("read-only probe — budgets are unchanged; edit them with "
                   "action='edit' (the approval gate runs first)"),
    }


def _target_edit(args: dict) -> dict:
    """action=edit — correct a REGISTERED target (W1.2). policy_hook routed
    this exact call through the human-approval gate (rule_key
    model-forge:target_edit) BEFORE this runs; denial or timeout blocks it,
    so nothing changes without approval. The local target and
    settings-declared rows are refused with the honest reason."""
    tid = str(args.get("id", "")).strip()
    if not tid:
        return {"ok": False, "action": "edit", "error": "id is required"}
    current = get_target(tid)
    if current is None:
        return {"ok": False, "action": "edit",
                "error": f"unknown target: {tid!r} (see model_forge_status)"}
    changes = {}
    for key in ("label", "base_url", "kind", "note"):
        if key in args and args.get(key) is not None:
            changes[key] = str(args.get(key)).strip()
    for key in ("vram_gb", "ram_gb"):
        raw = args.get(key)
        if raw is not None:
            changes[key] = float(raw)
            src_key = key.replace("_gb", "_source")
            changes[src_key] = (str(args.get(src_key) or "").strip()
                                or SOURCE_DECLARED)
    updated, err = update_target(tid, changes)
    if updated is None:
        return {"ok": False, "action": "edit", "error": err}
    with _BROKER_LOCK:
        _BROKER_STACKS.pop(tid, None)   # rebuild with the new budgets
    return {
        "ok": True, "action": "edit", "edited": updated.id,
        "before": _target_registration_row(current),
        "after": _target_registration_row(updated),
        "detail": ("updated — engines are re-probed everywhere targets are "
                   "listed; if this target has a cached broker stack it is "
                   "rebuilt with the new budgets"),
    }


def _target_remove(args: dict) -> dict:
    """action=remove — drop a REGISTERED target (W1.2). policy_hook routed
    this exact call through the human-approval gate (rule_key
    model-forge:target_remove) BEFORE this runs; denial or timeout blocks it.
    The local target is permanent; settings-declared rows are refused with
    the honest reason (remove them in settings)."""
    tid = str(args.get("id", "")).strip()
    if not tid:
        return {"ok": False, "action": "remove", "error": "id is required"}
    ok, err = remove_target(tid)
    if not ok:
        return {"ok": False, "action": "remove", "error": err}
    with _BROKER_LOCK:
        _BROKER_STACKS.pop(tid, None)   # drop the removed target's stack
    return {
        "ok": True, "action": "remove", "removed": tid,
        "targets": [{"id": x.id, "label": x.label, "kind": x.kind,
                     "base_url": x.base_url} for x in load_targets()],
        "detail": ("removed — nothing was probed, downloaded or deleted; "
                   "this only forgets the registration"),
    }


def model_forge_target_add(args: dict | None = None, **kwargs) -> str:
    """Register a compute target (this machine, a remote host, a cloud GPU
    box), and manage registered ones. Actions: 'add' (default; rides the
    human-approval gate), 'preview' (read-only probe preview of a candidate —
    W1.1), 'edit' and 'remove' (registered targets; both ride the approval
    gate), 'retest' (read-only re-probe). Every write is fail-closed: denial
    or timeout at the gate changes nothing."""
    args = args or {}
    try:
        action = str(args.get("action") or "add").strip() or "add"
        if action == "preview":
            return _dump(_target_preview(args))
        if action == "retest":
            return _dump(_target_retest(args))
        if action == "edit":
            return _dump(_target_edit(args))
        if action == "remove":
            return _dump(_target_remove(args))
        if action != "add":
            return _dump({"error": f"unknown action: {action!r}; valid "
                                   f"actions: ['add', 'preview', 'edit', "
                                   f"'remove', 'retest']"})
        tid = str(args.get("id", "")).strip()
        label = str(args.get("label", "")).strip() or tid
        base_url = str(args.get("base_url", "")).strip()
        kind = str(args.get("kind", "")).strip() or "lan"
        if not tid or tid == "local":
            return _dump({"error": "id is required and cannot be 'local'"})
        err = validate_target_fields(base_url=base_url, kind=kind,
                                     vram_gb=args.get("vram_gb"),
                                     ram_gb=args.get("ram_gb"))
        if err:
            return _dump({"error": err})
        t = ComputeTarget(
            id=tid, label=label, base_url=base_url, kind=kind,
            vram_gb=float(args.get("vram_gb") or 0.0),
            ram_gb=float(args.get("ram_gb") or 0.0),
            note=str(args.get("note", "")),
            vram_source=(str(args.get("vram_source") or "").strip()
                         or (SOURCE_DECLARED
                             if args.get("vram_gb") is not None else "")),
            ram_source=(str(args.get("ram_source") or "").strip()
                        or (SOURCE_DECLARED
                            if args.get("ram_gb") is not None else "")),
        )
        add_target(t)
        return _dump({
            "added": t.id,
            "target": _target_registration_row(t),
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


def _install_mark_installed(args: dict) -> dict:
    """action=mark_installed — record that a model's weights are present on a
    compute target (B10), so swap pre-flights trust them there.

    A remote machine cannot be scanned over an engine API, so the record
    carries honest provenance: ``observed`` when the target's engine reports
    its pull slot complete (Maya /api/pull/status says done), else
    ``user_confirmed`` — the user's attestation that the exact on-target
    install step finished. A wrong record is not destructive: the swap's
    measured LOAD phase reports the real outcome and nothing on disk is
    touched by a record. No approval gate: this records a fact the user
    asserts; it redirects no traffic and deletes nothing."""
    from . import pull
    model_id = str(args.get("model_id") or "").strip()
    if not model_id:
        return {"ok": False, "action": "mark_installed",
                "error": "model_id is required"}
    target = get_target(args.get("target"))
    if target is None:
        return {"ok": False, "action": "mark_installed",
                "error": f"unknown target: {args.get('target')!r} "
                         "(see model_forge_status)"}
    source = pull.RECORD_USER_CONFIRMED
    try:
        status = maya_adapter.pull_status(base_url=target.base_url)
    except Exception:  # noqa: BLE001 — no observation is an honest outcome
        status = None
    observed = (isinstance(status, dict)
                and str(status.get("state") or "").strip().lower()
                in ("done", "completed", "finished"))
    if observed:
        source = pull.RECORD_OBSERVED
    row = pull.record_pull_entry(
        target.id, model_id,
        path=str(args.get("path") or ""),
        size_bytes=args.get("size_bytes"), source=source)
    row["action"] = "mark_installed"
    if target.kind == "local":
        row["detail"] = (
            "recorded; note the local target's swap pre-flight reads the "
            "on-disk storage scan, not records — this entry is "
            "informational")
        return row
    row["detail"] = (
        ("the engine reports its pull slot complete — recorded as "
         "observed" if observed else
         "the engine did not report a completed pull for this model — "
         "recorded as YOUR confirmation that the install step finished")
        + f"; swap pre-flights on {target.id!r} now trust these weights "
          "(the measured LOAD phase reports the real outcome if they are "
          "not actually there)")
    return row


def model_forge_install(args: dict | None = None, **kwargs) -> str:
    """Pull a model on a compute target (Maya: HTTP pull; HF download: job;
    Strata: CLI), or provision an engine (action='engine_install': plan the
    vendor's official install command first; execute=true runs it behind the
    human-approval gate). Pass job_id to poll a download or engine install.
    action='mark_installed' records weights present on a remote target (B10)
    — engine-observed or user-confirmed — so swaps there are not refused."""
    args = args or {}
    try:
        from . import pull
        if args.get("job_id"):
            return _dump(pull.job_status(str(args["job_id"])))
        action = str(args.get("action") or "model_install").strip()
        if action == "engine_install":
            return _dump(_engine_install_action(args))
        if action == "mark_installed":
            return _dump(_install_mark_installed(args))
        if action != "model_install":
            return _dump({"error": f"unknown action: {action!r}; valid "
                                   f"actions: ['model_install', "
                                   f"'engine_install', 'mark_installed']"})
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
        available = [e.id for e in _detect(base) if e.available]
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
        if (target.kind != "local"
                and plan.get("action") in ("run_on_target",
                                           "run_cli_on_target", "maya_pull")):
            plan["record_hint"] = (
                "when the files land on " + target.label + ", record them "
                "with model_forge_install action='mark_installed', "
                f"model_id={model_id!r}, target={target.id!r} — recorded "
                "pulls are what the swap pre-flight trusts on remote "
                "targets (recorded as your confirmation unless the engine "
                "reports the pull complete)")
        return _dump(plan)
    except Exception as exc:
        return _err(exc)


def model_forge_activate(args: dict | None = None, **kwargs) -> str:
    """Connect a running chat/vision model as THIS Hermes' default model.

    Safe by default: pass apply=true to actually write model.* config
    (takes effect next session; revert with `hermes model`).

    Pass facade=true when the target is an OpenAI-compatible model-forge
    facade: the plan and apply then cover BOTH config surfaces the live
    consumers read — the index-addressed ``custom_providers`` entry (model
    list refreshed from the target's ``/v1/models`` when reachable, fail-soft
    otherwise) AND ``auxiliary.vision`` (model + base_url for the vision lane).
    A legacy alias in the requested/served model id is written as its CANONICAL
    id (the facade resolves aliases anyway; config names what /v1/models
    advertises). Engine-native activations are unchanged. For facade=true a
    registry-less id is also accepted when the TARGET's own /v1/models catalog
    advertises it (or aliases.resolve maps it to a cataloged id) — fail-closed,
    an honest refusal otherwise.

    Tests exercise the REAL write path against real ``hermes_cli.config``
    load/save semantics (an isolated config in a temp home); the two facade
    surfaces are written IN-PROCESS, never through dotted ``hermes config set``
    CLI calls. The human-approval gate (policy_hook, apply=true) is
    unchanged."""
    args = args or {}
    try:
        requested = str(args.get("model_id") or "").strip()
        if not requested:
            return _dump({"error": "model_id required"})
        # A legacy alias resolves to its canonical id first (identity for a
        # canonical id): the registry is keyed on canonical ids and the config
        # should name the canonical id the facade advertises.
        model_id = resolve(requested)
        entry = next((m for m in models_for() if m.id == model_id), None)
        if entry is None:
            entry = next((m for m in models_for() if m.id == requested), None)
        if entry is None and args.get("facade") is True:
            # P5 §8: a facade activation may name a CATALOG id the curated
            # registry has no entry for (the facade chat id
            # Qwen/Qwen3.8-27B-FP8). Accept exactly what the target's own
            # /v1/models advertises (or what an alias resolves to there) —
            # fail-closed, with an honest refusal for anything else.
            catalog_target = get_target(args.get("target"))
            if catalog_target is None:
                return _dump({"error": f"unknown target: {args.get('target')}"})
            entry, refusal = _facade_catalog_entry(requested, catalog_target)
            if entry is None:
                return _dump({"error": refusal})
        if entry is None:
            return _dump({"error": f"unknown model_id: {requested}"})
        target = get_target(args.get("target"))
        if target is None:
            return _dump({"error": f"unknown target: {args.get('target')}"})
        base = "" if target.kind == "local" else target.base_url
        statuses = _detect(base)
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
                    f"{key}.provider {_MEDIA_PROVIDER_NAME}")
            if media:
                plan["media_setup"] = media
            if withheld:
                plan["media_setup_note"] = (
                    "an engine on this target lists these modalities but the "
                    "endpoint did not answer the matching route, so the "
                    "setup hint is withheld (run it manually only if your "
                    "endpoint serves that route): " + "; ".join(withheld))
        # A facade target covers BOTH consumer surfaces: the index-addressed
        # custom_providers entry and auxiliary.vision (see _facade_plan_surfaces).
        if plan.get("ok") and args.get("facade") is True:
            _facade_plan_surfaces(plan,
                                  providers=_current_custom_providers())
        if plan.get("ok"):
            if args.get("apply") is True:
                try:
                    from hermes_cli.model_setup import persist_model_selection
                    saved = persist_model_selection(
                        plan["model"], plan["provider"],
                        base_url=plan["base_url"],
                        api_mode=_CUSTOM_PROVIDER_API_MODE)
                    plan["applied"] = True
                    plan["saved_model_section"] = saved
                except Exception as exc:
                    plan["applied"] = False
                    plan["error"] = f"{type(exc).__name__}: {exc}"
                # The custom_providers registration is its own honest write:
                # a failure here is reported beside the model-section state,
                # never folded into it.
                if plan.get("facade"):
                    # Both consumer surfaces are written IN-PROCESS through the
                    # SAME hermes_cli.config load/save API the non-facade path
                    # uses. Index-addressed `hermes config set` writes cannot be
                    # used: the CLI splits a key on '.' (a canonical model id
                    # like Qwen/Qwen3.8-27B-FP8 shatters into nested keys) and
                    # cannot grow a list (a fresh config becomes an unread
                    # MAPPING {'0': {...}}, appending at len(rows) raises
                    # IndexError). Each surface keeps its own honest
                    # applied/error state.
                    try:
                        plan["custom_provider"]["saved"] = \
                            write_custom_provider_entry(
                                plan["custom_provider"]["entry"])
                        plan["custom_provider"]["applied"] = True
                    except Exception as exc:  # noqa: BLE001 — honest state
                        plan["custom_provider"]["applied"] = False
                        plan["custom_provider"]["error"] = \
                            f"{type(exc).__name__}: {exc}"
                    try:
                        plan["auxiliary_vision"]["saved"] = \
                            write_auxiliary_vision(
                                plan["auxiliary_vision"]["model"],
                                plan["auxiliary_vision"]["base_url"])
                        plan["auxiliary_vision"]["applied"] = True
                    except Exception as exc:  # noqa: BLE001 — honest state
                        plan["auxiliary_vision"]["applied"] = False
                        plan["auxiliary_vision"]["error"] = \
                            f"{type(exc).__name__}: {exc}"
                    plan["config_writes"] = [
                        {"key": "custom_providers",
                         "value": plan["custom_provider"]["slug"],
                         "ok": plan["custom_provider"]["applied"]},
                        {"key": "auxiliary.vision.model",
                         "value": plan["auxiliary_vision"]["model"],
                         "ok": plan["auxiliary_vision"]["applied"]},
                        {"key": "auxiliary.vision.base_url",
                         "value": plan["auxiliary_vision"]["base_url"],
                         "ok": plan["auxiliary_vision"]["applied"]},
                    ]
                else:
                    try:
                        plan["custom_provider"]["saved"] = \
                            write_custom_provider_entry(
                                plan["custom_provider"]["entry"])
                        plan["custom_provider"]["applied"] = True
                    except Exception as exc:
                        plan["custom_provider"]["applied"] = False
                        plan["custom_provider"]["error"] = \
                            f"{type(exc).__name__}: {exc}"
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
    dirs), delete one listed entry (routed through the human-approval gate;
    see policy_hook), or take the on-demand resource snapshot
    (action=resources: a read-only memory view per compute target — VRAM and
    RAM as used / booked_now / free, resident model blocks, queued loads,
    measured swap stats; runs only when asked)."""
    args = args or {}
    try:
        from . import storage
        action = str(args.get("action") or "list").strip() or "list"
        if action not in ("list", "delete", "resources"):
            return _dump({"error": f"unknown action: {action!r}; valid "
                                   f"actions: ['list', 'delete', 'resources']"})
        if action == "list":
            return _dump(storage.list_storage())
        if action == "resources":
            return _dump(storage_resources(args))
        ref = str(args.get("entry") or "").strip()
        if not ref:
            return _dump({"error": "entry required for action=delete: pass "
                                   "an entry id from action=list or that "
                                   "entry's exact path"})
        return _dump(storage.delete_entry(ref))
    except Exception as exc:
        return _err(exc)


# --- local ComfyUI workflow-runner wiring (image / video lanes) --------------

# Local ComfyUI-runner media jobs keyed by job id. The runner is synchronous,
# so a video job runs in a daemon thread while the tool returns immediately and
# the caller keeps the existing job_id polling contract.
_COMFY_MEDIA_JOBS: dict = {}
_COMFY_JOBS_LOCK = threading.Lock()
# The workflow inputs a local lane may consume; everything else in the tool
# args (modality/target/model/job_id) is control data, never a workflow input.
_LOCAL_GEN_PARAM_KEYS = ("prompt", "seed", "steps", "seconds", "size", "cfg",
                         "image", "image_url", "resolution",
                         "negative_prompt")


def _local_workflow_runner(target, *, http=None, **seams):
    """The local ComfyUI workflow runner for *target*, or None when the active
    stack does not provide one.

    A runner exists for a LOCAL target whose detected engines include a live
    ComfyUI, or for a CONNECTED/remote target whose ``base_url`` is a ComfyUI
    endpoint (the ported, bundled templates drive it over HTTP — B1). For a
    remote ComfyUI the outputs live on its host, so they are surfaced as the
    fetchable ``/view`` URLs the runner reports. A target with no ComfyUI is
    an honest ``None`` (the media-gateway HTTP path keeps its lanes). Read-only
    detection; a failure or no ComfyUI is an honest None, never a raise.
    Injectable — tests replace this module attribute so no probe runs:
    ``monkeypatch.setattr(tools, "_local_workflow_runner", lambda t, **k: fake)``.
    """
    if target is None:
        return None
    try:
        from .engines.comfyui import ComfyWorkflowRunner
        if getattr(target, "kind", "") == "local":
            for e in _detect(""):
                if e.id == "comfyui" and getattr(e, "available", False):
                    return ComfyWorkflowRunner(getattr(e, "endpoint", "") or "",
                                               http=http, **seams)
            return None
        # A CONNECTED/remote target: drive a ComfyUI at its base_url when one
        # is detected there (B1 — a connected ComfyUI serves generation, not
        # only downloads). Its /output dir is not readable here, so outputs
        # are surfaced as the fetchable /view URLs the runner reports.
        base = (getattr(target, "base_url", "") or "").rstrip("/")
        if not base:
            return None
        from .engines.comfyui import detect as _comfy_detect
        st = _comfy_detect(base)
        if (getattr(st, "available", False)
                and getattr(st, "id", "") == "comfyui"):
            return ComfyWorkflowRunner(
                getattr(st, "endpoint", "") or base, http=http, **seams)
    except Exception:  # noqa: BLE001 — no runner is an honest fallback
        return None
    return None


def _local_assets_dir() -> str:
    """Dir the runner stashes uploaded image references under (``uploads/``).
    The plugin data dir, with an honest temp fallback when it is unavailable."""
    try:
        from plugins.plugin_storage import plugin_data_dir
        d = plugin_data_dir("model-forge") / "media"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)
    except Exception:  # noqa: BLE001 — a temp dir keeps the job going
        import tempfile
        return tempfile.mkdtemp(prefix="model-forge-media-")


def _media_asset_url(path, *, embed: bool = True, fallback: str = "") -> str:
    """A reference for a runner output file: a ``data:`` URL when the file is
    readable locally and ``embed`` is set (images), else the absolute path the
    runner reported. For a REMOTE/connected ComfyUI the path lives on its
    host and is not readable here, so the ComfyUI ``/view`` URL the runner
    reported is used (``fallback``) — never a dead local path. Never raises —
    an unreadable file falls back to its path (or ``fallback``)."""
    from pathlib import Path
    try:
        p = Path(str(path))
        if embed and p.is_file():
            import base64
            mime = _IMAGE_MIME_BY_SUFFIX.get(p.suffix.lower(),
                                             "application/octet-stream")
            return (f"data:{mime};base64,"
                    + base64.b64encode(p.read_bytes()).decode())
    except OSError:
        pass
    return fallback or str(path)


def _local_gen_params(args: dict) -> dict:
    """The workflow inputs to forward from the tool args (whitelisted so the
    control keys — modality/target/model/job_id — never reach a lane)."""
    return {k: args[k] for k in _LOCAL_GEN_PARAM_KEYS
            if args.get(k) is not None}


def _generate_media_local(args: dict, target, modality: str, runner) -> dict:
    """Run one media job through the local ComfyUI workflow runner, inside the
    same claim/release scope the media-gateway path uses.

    Image lanes run synchronously (the caller waits; each output is a data:
    URL). Video lanes keep the job_id polling contract: the synchronous runner
    runs in a daemon thread and the lane stays claimed until the job settles.
    Either way the lane is claimed before submit and released on completion or
    failure — the existing scope semantics, preserved. A job_id argument is
    handled as a poll.
    """
    if args.get("job_id"):
        return _poll_local_media_job(args, target)
    model = str(args.get("model", "")).strip()
    if not model:
        return {"ok": False, "action": "needs_args",
                "detail": f"{modality} needs model (e.g. gen/qwen-image-2.1 "
                          "or a bare lane id)"}
    plan = runner_plan(modality, model, runner)
    if not plan.get("ok"):
        return plan
    workflow_id = plan["workflow_id"]
    params = _local_gen_params(args)
    if modality == "image":
        return _run_local_image(target, runner, workflow_id, model, params)
    return _submit_local_video(target, runner, workflow_id, model, params)


def _run_local_image(target, runner, workflow_id, model, params) -> dict:
    """Synchronous image lane: claim the scope, run the lane, release, and
    return the OpenAI images shape (``data: [{"url": ...}]``)."""
    claim = _claim_media_lane(target, "image", model)
    if not claim.get("ok"):
        return {"ok": False, "action": "claim_failed",
                "error": claim.get("error") or "the lane claim was refused",
                "lane_claim": _lane_block(claim),
                "detail": "the GPU lane claim was refused (fail closed) — "
                          "nothing was submitted"}
    settled: list = []

    def _settle(note):
        # Mark the scope settled BEFORE releasing: if the release itself
        # raises, the finally must not issue a second release for the same
        # scope (release idempotency is not assumed).
        settled.append(True)
        rel = _release_media_lane(target, claim["scope_id"])
        return _lane_block(claim, released=rel, note=note)

    try:
        job_id = "img-" + uuid.uuid4().hex[:10]
        result = runner.run(workflow_id, params,
                            assets_dir=_local_assets_dir(), job_id=job_id)
        ok = bool(result.get("ok"))
        lane = _settle(
            "the render completed — the lane scope was released and what was "
            "resident at claim time is restored" if ok else
            "the job failed — the lane scope was released and what was "
            "resident at claim time is restored")
        if not ok:
            return {"ok": False, "action": "submit_failed",
                    "output_kind": "image", "workflow_id": workflow_id,
                    "error": result.get("reason")
                             or "the local ComfyUI job failed",
                    "lane_claim": lane}
        files = result.get("files") or []
        urls = result.get("urls") or []
        data = [{"url": _media_asset_url(
            f, fallback=(urls[i] if i < len(urls) else ""))}
            for i, f in enumerate(files)]
        return {"ok": True, "action": "done", "output_kind": "image",
                "data": data,
                "files": files, "model": model, "workflow_id": workflow_id,
                "lane_claim": lane,
                "note": "ran the ComfyUI image lane; each output is a "
                        "data: URL (or the ComfyUI /view URL when the file "
                        "is not readable here — e.g. a connected/remote "
                        "ComfyUI)"}
    finally:
        if not settled:
            _release_media_lane(target, claim["scope_id"])


def _submit_local_video(target, runner, workflow_id, model, params) -> dict:
    """Async video lane: claim the scope, start the synchronous runner in a
    daemon thread, and return the OpenAI video-job submit shape (``{id,
    status}``). The lane stays claimed until the poller sees a terminal state."""
    claim = _claim_media_lane(target, "video", model)
    if not claim.get("ok"):
        return {"ok": False, "action": "claim_failed",
                "error": claim.get("error") or "the lane claim was refused",
                "lane_claim": _lane_block(claim),
                "detail": "the GPU lane claim was refused (fail closed) — "
                          "nothing was submitted"}
    job_id = "vid-" + uuid.uuid4().hex[:10]
    record = {"id": job_id, "status": "queued",
              "scope_id": claim["scope_id"], "model_id": model,
              "modality": "video", "workflow_id": workflow_id,
              "files": [], "data": [], "error": ""}
    with _COMFY_JOBS_LOCK:
        _COMFY_MEDIA_JOBS[job_id] = record
    _MEDIA_SCOPE_JOBS[job_id] = {"scope_id": claim["scope_id"],
                                 "target_id": target.id, "model_id": model,
                                 "modality": "video"}

    def _worker():
        try:
            res = runner.run(workflow_id, params,
                             assets_dir=_local_assets_dir(), job_id=job_id)
        except Exception as exc:  # noqa: BLE001 — the record carries it
            res = {"ok": False, "files": [],
                   "reason": f"{type(exc).__name__}: {exc}"}
        with _COMFY_JOBS_LOCK:
            rec = _COMFY_MEDIA_JOBS.get(job_id)
            if rec is None:
                return
            if res.get("ok"):
                files = res.get("files") or []
                urls = res.get("urls") or []
                rec["status"] = "completed"
                rec["files"] = files
                rec["data"] = [{"url": _media_asset_url(
                    f, embed=False,
                    fallback=(urls[i] if i < len(urls) else ""))}
                    for i, f in enumerate(files)]
            else:
                rec["status"] = "failed"
                rec["error"] = (res.get("reason")
                                or "the local ComfyUI job failed")

    threading.Thread(target=_worker, daemon=True).start()
    lane = _lane_block(claim, note="the lane stays claimed until the job "
                                   f"completes; poll with job_id={job_id} — "
                                   "the scope releases and what was resident "
                                   "at claim time is restored when the job "
                                   "finishes")
    return {"ok": True, "action": "submitted", "output_kind": "video",
            "submitted": {"id": job_id, "status": "queued",
                          "model": workflow_id},
            "lane_claim": lane,
            "poll": f"model_forge_generate_media with modality=video, "
                    f"target={target.id}, job_id={job_id}"}


def _poll_local_media_job(args: dict, target) -> dict:
    """Poll a local ComfyUI-runner job by job_id, in the OpenAI video-job shape
    (``{status, data: [{url}]}``). A terminal job releases its lane scope; a
    non-local job_id is an honest ``unknown_job``."""
    job_id = str(args.get("job_id") or "")
    if not _JOB_TOKEN_RE.fullmatch(job_id):
        return {"ok": False, "action": "invalid_job_id",
                "error": "job_id must be 1-128 ASCII letters, digits, '_' or "
                         f"'-' (got {job_id!r})"}
    with _COMFY_JOBS_LOCK:
        rec = _COMFY_MEDIA_JOBS.get(job_id)
        snapshot = dict(rec) if rec is not None else None
    if snapshot is None:
        return {"ok": False, "action": "unknown_job",
                "error": f"no local ComfyUI job {job_id!r} — it may have "
                         "already been collected, or it is not a local job"}
    status = snapshot["status"]
    terminal = status in ("completed", "failed")
    job = {"id": job_id, "model": snapshot.get("workflow_id"),
           "status": ("completed" if status == "completed"
                      else "failed" if status == "failed" else "in_progress")}
    if snapshot.get("error"):
        job["error"] = snapshot["error"]
    if status == "completed":
        job["data"] = snapshot.get("data") or []
    claim = {"scope_id": snapshot["scope_id"],
             "model_id": snapshot.get("model_id"), "modality": "video",
             "progress": []}
    out = {"ok": True, "output_kind": "video", "job": job}
    if terminal:
        with _COMFY_JOBS_LOCK:
            _COMFY_MEDIA_JOBS.pop(job_id, None)
        _MEDIA_SCOPE_JOBS.pop(job_id, None)
        rel = _release_media_lane(target, snapshot["scope_id"])
        note = ("job complete — the lane scope was released and what was "
                "resident at claim time is restored")
        if status == "completed":
            # Honest divergence from the facade: a finished LOCAL video job
            # reports the runner's filesystem paths, not HTTP asset URLs.
            note += (" Each job.data[].url is the runner's local filesystem "
                     "path, NOT an HTTP-fetchable asset URL (the remote "
                     "facade's /assets URLs differ); read it on the runner "
                     "host.")
        out["lane_claim"] = _lane_block(claim, released=rel, note=note)
    else:
        out["lane_claim"] = _lane_block(
            claim,
            note="the job is still running — the lane stays claimed; poll "
                 "again and the scope releases when the job finishes")
    return out


def model_forge_generate_media(args: dict | None = None, **kwargs) -> str:
    """Generate media through a compute target. 3d/music/sfx ride the target's
    media-gateway endpoint; image/video run through the local ComfyUI workflow
    runner when one is available (a local target with ComfyUI detected), else
    image falls back to the gateway route and video is reported as a facade
    job-API lane. 3d and local video submit a job (poll with job_id); image and
    music/sfx run synchronously. The GPU lane is claimed as one workflow scope
    before the job is submitted and released when the job completes or fails; a
    refused claim fails closed with an honest error and submits nothing."""
    args = args or {}
    try:
        import json as _json
        modality = str(args.get("modality", ""))
        target = get_target(args.get("target"))
        if target is None:
            return _dump({"error": f"unknown target: {args.get('target')}"})
        # A LOCAL target whose stack provides a ComfyUI workflow runner serves
        # the image/video lanes directly through the ported templates — inside
        # the same claim/release scope as the gateway path. A remote
        # media-gateway target has no runner and keeps its HTTP path.
        runner = _local_workflow_runner(target)
        if runner is not None and modality in ("image", "video"):
            return _dump(_generate_media_local(args, target, modality, runner))
        if modality == "video":
            return _dump({
                "ok": False, "action": "no_local_runner",
                "detail": "video lanes run through the Model Forge facade "
                          "video job API (POST /v1/videos); point the "
                          "model-forge-video provider at the facade endpoint, "
                          "or run this tool on a local target with ComfyUI "
                          "detected"})
        plan = media_plan(modality, target.base_url)
        if not plan.get("ok"):
            return _dump(plan)
        if modality == "3d":
            if args.get("job_id"):
                job_id = str(args["job_id"])
                if not _JOB_TOKEN_RE.fullmatch(job_id):
                    return _dump({"ok": False, "action": "invalid_job_id",
                                  "error": "job_id must be 1-128 ASCII "
                                           "letters, digits, '_' or '-' "
                                           f"(got {job_id!r})"})
                try:
                    job = _fetch_media_job(target, job_id)
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


def _is_quantized(path: str) -> bool:
    """True for a quantized weights variant (A2). Full-precision files
    (fp16/bf16/fp32, plain .safetensors) are NOT quantized; only conventional
    quant tokens (GGUF q4/q8, int4/int8, fp8/fp4, awq/gptq/exl, quant) mark a
    quant. Used to prefer full weights and label the pick honestly."""
    return bool(_QUANT_RE.search(path.lower()))


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
    # A2: prefer FULL (non-quantized) weights. A quantized variant is picked
    # only when the repo offers no full weights — pass engine_model to request
    # a specific quant explicitly ("quantized variants only on request").
    full = [(p, s) for p, s in candidates if not _is_quantized(p)]
    quants = [(p, s) for p, s in candidates if _is_quantized(p)]
    pool = full or quants
    pick = min(pool,
               key=lambda row: (-row[1], _suffix_rank(row[0]), row[0]))
    quantized = _is_quantized(pick[0])
    detail = ("primary weights file picked from the repo file list"
              + (" (full weights preferred)" if not quantized else
                 " (quantized variant — this repo offers no full weights; "
                 "pass engine_model to name a specific quant)"))
    return {"ok": True, "action": "resolved", "file": pick[0],
            "size_bytes": pick[1], "quantized": quantized,
            "detail": detail}


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
# Unbooked rows label themselves against the booked-ceiling concept — never
# with the booking source token, so no word means two things in one report.
BOOKING_SOURCE_NONE = "no booked ceiling (no VRAM floor declared)"
BOOKING_SOURCE_NOTE = (
    "booked ceilings are the curated catalog's published VRAM floors used as "
    "nominal ceilings — advisory published minimums, not measured peaks; a "
    "measured peak replaces one and is never downgraded. A model that "
    "declares no VRAM floor (engine-managed fit) stays unbooked and the "
    "planner refuses to gate it until a measured peak lands — run bench "
    "after its first load.")
UNBOOKED_HINT = "unbooked — run bench after first load"


def _declared_floor(m: ModelEntry) -> float | None:
    """The entry's published VRAM floor in GB, or None when it declares none
    (engine-managed fit). None means unbooked — never a guessed 0 GB: the
    policy layer gates fit on booked ceilings and never guesses fits."""
    try:
        floor = float(m.vram_gb_min or 0.0)
    except (TypeError, ValueError):
        return None
    return floor if floor > 0.0 else None


def bookings_from_registry(entries=None) -> dict:
    """Curated catalog ModelEntry footprints -> the bookings dict the
    residency layer takes.

    Each entry's published ``vram_gb_min`` floor becomes its nominal booked
    ceiling, honestly labeled ``BOOKING_SOURCE``. Entries that declare no
    VRAM floor (engine-managed fit) and stub entries stay UNBOOKED — a
    guessed 0 GB would be a guessed fit, and the policy layer never guesses
    fits. Unbooked entries surface in broker status with the
    ``UNBOOKED_HINT`` hint, and the planner refuses to gate them until a
    measured peak is recorded (``model_forge_broker`` ``action=bench``).
    Stdlib-only and pure: no I/O."""
    out = {}
    for m in (models_for() if entries is None else entries):
        if m.status == "stub":
            continue
        floor = _declared_floor(m)
        if floor is None:
            continue
        out[m.id] = {
            "vram_gb": floor,
            "source": BOOKING_SOURCE,
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

    WARM_PATH = ("engine-native load on first use (pass-through — the engine "
                 "owns its load and reclaim policy)")

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
_LIFECYCLE_ENGINES = ("ollama", "llama-swap", "llamacpp", "comfyui", "vllm",
                      "maya")
_PASS_THROUGH_ENGINES = ("sglang", "strata", "openai-compatible", "python",
                         "media-gateway")


def _adapter_catalog(target, http=None, runner=None, which=None) -> dict:
    """Engine lifecycle adapters for a compute target. Real control where the
    engine exposes it: Ollama warm/evict calls, llama-swap observation,
    llama.cpp and vLLM process start/stop, ComfyUI readiness plus explicit
    cache-free between generative jobs, Maya engine-native sessions (no
    unload primitive — never faked). Every external seam (HTTP function,
    process runner, binary lookup) is injectable so nothing touches the
    network or spawns a process unless the caller allows it. Remaining
    engines are an honest pass-through. Load/unload calls go to the target's
    user-configured endpoint."""
    root = root_url(target.base_url) or "http://127.0.0.1"
    catalog = {
        "ollama": OllamaLifecycle(base_url=root, http=http),
        "llama-swap": LlamaSwapLifecycle(base_url=root, http=http),
        "llamacpp": LlamaCppServerLifecycle(runner=runner, which=which),
        "comfyui": ComfyUiLifecycle(base_url=root, http=http),
        "vllm": VllmServerLifecycle(runner=runner, which=which),
        "maya": MayaLifecycle(base_url=root, http=http),
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

# A20: the exact restore set the human-approval gate saw for a pending
# release — keyed by scope id, or "*{target_id}" for a media call's inline
# release (its scope id is generated after the hook runs). Written by
# policy_hook's *_would_start_server helpers the moment they inspect a
# restore plan (or find none to inspect); consumed (popped) by the release
# handler, which passes it to Orchestrator.release for re-verification
# against the recomputed plan. Consume-on-read bounds memory and prevents
# replay; a missing entry means no gate ran on this path and the handler
# keeps its pre-A20 behavior.
_APPROVED_RESTORES: dict = {}
_APPROVED_RESTORES_LOCK = threading.Lock()


def _record_approved_restore(key, model_ids) -> None:
    with _APPROVED_RESTORES_LOCK:
        _APPROVED_RESTORES[key] = frozenset(model_ids)


def _consume_approved_restore(key):
    with _APPROVED_RESTORES_LOCK:
        return _APPROVED_RESTORES.pop(key, None)


def _drop_approved_restore(key) -> None:
    with _APPROVED_RESTORES_LOCK:
        _APPROVED_RESTORES.pop(key, None)


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
            "adapters": adapters,
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


# The process-based engine lifecycles (broker.py): these start a server
# process on load; every other engine is HTTP-native or an observational
# pass-through and starts nothing. Their default listening ports are shown in
# the approval prompt.
_PROCESS_ENGINE_PORTS = {"llamacpp": 8080, "vllm": 8000}


def _broker_would_start_server(args) -> dict | None:
    """Whether a model_forge_broker ``claim``/``swap`` call WOULD start an
    engine SERVER process, and the honest detail for the approval prompt —
    else ``None``.

    Mirrors the handler's own resolution (``_claim_entry`` -> ``_claim_engine``
    resolve the model to its engine exactly as the handler does) and the
    planner's own residency gate (``ledger.is_resident``): a claim/swap loads
    the model — and so starts a process — only when it is NOT already resident
    on the target, so a claim that merely takes the lane over a live server
    starts nothing and does not escalate. Only the process-based lifecycles
    (llama.cpp, vLLM) start a server.

    Fail-safe by design: any uncertainty about whether a process would start
    escalates. An extra approval is acceptable; an unapproved server start is
    the bug."""
    model_id = str(args.get("model_id") or "").strip()
    modality = str(args.get("modality") or "").strip()
    target_id = str(args.get("target") or "").strip() or "local"
    try:
        entry, err = _claim_entry(model_id, modality)
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        return {"model_id": model_id or f"(modality {modality!r})",
                "engine": "(unresolved)", "port": "its default",
                "target": target_id}
    # An unresolvable request (unknown model / no curated model for the
    # modality / no engine) is refused by the handler BEFORE anything starts,
    # so no server would start — that is certain, not uncertain.
    if err or entry is None:
        return None
    engine = _claim_engine(entry)
    if engine not in _PROCESS_ENGINE_PORTS:
        return None    # HTTP-native or pass-through engine: no process starts
    try:
        target = get_target(args.get("target"))
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        target = None
    if target is None:
        # An unknown target is refused by the handler before any start.
        return None
    detail = {"model_id": entry.id, "engine": engine,
              "port": _PROCESS_ENGINE_PORTS.get(engine, "its default"),
              "target": target.id}
    try:
        # The same broker stack the handler will use (cached per target) — a
        # construction, never a spawn; the ledger is the planner's own
        # residency gate.
        resident = broker_stack(target)["ledger"].is_resident(target.id,
                                                              entry.id)
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        return detail
    if resident:
        return None    # already resident: the claim/swap starts nothing
    return detail


def _release_would_start_server(args) -> dict | None:
    """Whether releasing a scope WOULD start a process-engine server, and the
    honest detail for the approval prompt — else ``None``.

    A release restores the scope's claim-time residents. The restore plan is
    computed the same way the handler's release does — the orchestrator
    plans without executing (``restore_plan``) and the planner skips models
    that are already resident — so the gate sees exactly what the release
    would load. Only a load step on a process-engine lifecycle (llama.cpp,
    vLLM) starts a server; a release that restores nothing, restores an
    already-resident model, or restores only HTTP-native engines starts
    nothing and does not escalate. An unknown scope is refused by the
    handler before any start, so it does not escalate either.

    Fail-safe: any uncertainty about whether the restore would start a
    process escalates."""
    scope_id = str(args.get("scope_id") or "").strip()
    if not scope_id:
        return None    # the handler refuses: nothing released, nothing starts
    target_id = (str(args.get("target") or "").strip()
                 or _ACTIVE_SCOPES.get(scope_id, {}).get("target_id") or "")
    try:
        target = get_target(target_id or None) or get_target(None)
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        target = None
    if target is None:
        return {"scope_id": scope_id, "target": target_id or "local",
                "engine": "(unresolved)", "model_id": "",
                "port": "its default"}
    try:
        plan = broker_stack(target)["orchestrator"].restore_plan(scope_id)
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        return {"scope_id": scope_id, "target": target.id,
                "engine": "(unresolved)", "model_id": "",
                "port": "its default"}
    if plan is None:
        return None    # unknown scope: release fails closed, starts nothing
    # A20: freeze what this gate inspection SAW. The release handler
    # re-verifies its recomputed plan against exactly this set and fails
    # closed on anything new (a plan that grew in the gate-to-handler
    # window). Recorded for every inspected plan — including an empty one —
    # because growth is the thing being caught.
    _record_approved_restore(scope_id,
                             [s.model_id for s in plan.load_steps])
    steps = [step for step in plan.load_steps
             if step.engine in _PROCESS_ENGINE_PORTS]
    if not steps:
        return None    # nothing process-based would be (re)started
    return {"scope_id": scope_id, "target": target.id,
            "engine": steps[0].engine, "model_id": steps[0].model_id,
            "port": _PROCESS_ENGINE_PORTS.get(steps[0].engine, "its default"),
            "models": [step.model_id for step in steps]}


def _media_would_start_server(args) -> dict | None:
    """Whether a ``model_forge_generate_media`` call WOULD start a
    process-engine server, and the honest detail for the approval prompt —
    else ``None``.

    The media lane claims a workflow scope around the job and releases it
    when the job ends; the release restores what was resident at claim time.
    Two shapes reach the gate:

    - A poll (modality 3d with a ``job_id``). The handler releases the held
      scope ONLY when the job record is terminal (``_media_job_finished``);
      while the job still runs the call performs no release, so a
      still-running poll must NOT escalate — otherwise every poll of a long
      job would prompt. The job record is read here the same way the handler
      reads it; a terminal job computes the restore plan exactly via
      ``restore_plan`` (the same path a ``model_forge_broker`` release uses).
    - A submit that claims the lane now (3d, music, sfx): no scope exists
      yet, so it is fail-safe — the release can only restore a claim-time
      resident, so if a process-engine model is resident on the target the
      claim could evict it and the release would re-start it, and the gate
      escalates (this over-approximation stays). With no process-engine
      resident the release can restore nothing that starts a process
      (certain), and nothing escalates.

    Fail-safe: any uncertainty escalates — including a job record that cannot
    be read at all, which is why the poll path escalates on a fetch failure
    rather than assuming the job is still running."""
    try:
        target = get_target(args.get("target"))
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        target = None
    if target is None:
        return None    # the handler refuses an unknown target before any start

    def unresolved(model_id=""):
        return {"model_id": model_id, "engine": "(unresolved)",
                "port": "its default", "target": target.id}

    modality = str(args.get("modality") or "").strip()
    job_id = str(args.get("job_id") or "").strip()
    held = _MEDIA_SCOPE_JOBS.get(job_id) if job_id else None
    if modality == "3d" and job_id:
        # Poll path — the handler's 3d ``job_id`` branch. It releases the held
        # scope only when the job record is terminal; while the job runs (or
        # if no scope is held here) this call releases nothing and starts
        # nothing. Reading the record decides which case this call is in.
        if held is None:
            return None    # no held scope: this poll releases nothing
        try:
            job = _fetch_media_job(target, job_id)
        except Exception:  # noqa: BLE001 — state unknown -> escalate
            return unresolved(held.get("model_id") or "")
        if not _media_job_finished(job):
            return None    # still running: this poll releases nothing
        try:
            plan = broker_stack(target)["orchestrator"].restore_plan(
                held["scope_id"])
        except Exception:  # noqa: BLE001 — uncertainty -> escalate
            return unresolved(held.get("model_id") or "")
        if plan is None:
            return None
        # A20: freeze what this gate inspection saw (see the release-verb
        # path) — the handler re-verifies the recomputed plan against it.
        _record_approved_restore(held["scope_id"],
                                 [s.model_id for s in plan.load_steps])
        steps = [step for step in plan.load_steps
                 if step.engine in _PROCESS_ENGINE_PORTS]
        if not steps:
            return None
        return {"model_id": steps[0].model_id, "engine": steps[0].engine,
                "port": _PROCESS_ENGINE_PORTS.get(steps[0].engine,
                                                  "its default"),
                "target": target.id,
                "models": [step.model_id for step in steps]}
    # Submit path: predict the claim by what is resident now (fail-safe — a
    # resident process-engine model could be evicted and re-started on the
    # lane's release).
    if not str(args.get("model") or "").strip():
        return None    # needs_args: the handler submits nothing
    try:
        residents = broker_stack(target)["ledger"].residents(target.id)
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        return unresolved(str(args.get("model") or ""))
    process = [rec for rec in residents
               if rec.engine in _PROCESS_ENGINE_PORTS]
    # A20: the gate approves (or, with no process resident, certifies there
    # is nothing to approve) the restore this call's inline release may run.
    # The scope id does not exist yet at hook time, so the manifest is keyed
    # per target for this call's inline release to consume.
    _record_approved_restore(f"*{target.id}",
                             [r.model_id for r in process])
    if not process:
        return None    # release restores nothing that starts a process
    rec = process[0]
    return {"model_id": rec.model_id, "engine": rec.engine,
            "port": _PROCESS_ENGINE_PORTS.get(rec.engine, "its default"),
            "target": target.id,
            "models": [r.model_id for r in process]}


def _external_stop_detail(target, stop_steps) -> dict | None:
    """The first model whose eviction/release would STOP an externally-started
    engine runtime (B8) — else ``None``. ``stop_steps`` are the plan's
    release steps (each carrying ``engine`` and ``model_id``) that the call
    would run. The live adapter is asked directly (``stops_externally_started``)
    so a container the user started is caught from current runtime state, not
    a stale record. Process adapters answer ``False`` (they only stop a handle
    they started), so they never escalate here.

    Fail-safe: any uncertainty about provenance escalates."""
    steps = list(stop_steps or ())
    try:
        stack = broker_stack(target)
        adapters = (stack.get("adapters")
                    or getattr(stack.get("orchestrator"), "_adapters", None))
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        adapters = None
    if adapters is None:
        # Cannot resolve the adapter catalog. With nothing to stop there is
        # nothing to gate; with steps whose provenance is unknowable, fail
        # safe and escalate.
        return ({"target": getattr(target, "id", ""),
                 "engine": steps[0].engine if steps else "(unresolved)",
                 "model_id": steps[0].model_id if steps else "",
                 "external": True} if steps else None)
    for step in steps:
        adapter = adapters.get(step.engine)
        if adapter is None:
            continue
        fn = getattr(adapter, "stops_externally_started", None)
        if not callable(fn):
            # A duck-typed adapter with no provenance capability: it is not
            # modeled as able to stop an externally-started server, so it
            # does not escalate (all real adapters carry the method).
            continue
        try:
            if fn(step.model_id):
                return {"target": target.id, "engine": step.engine,
                        "model_id": step.model_id, "external": True}
        except Exception:  # noqa: BLE001 — real adapter error -> escalate
            return {"target": target.id, "engine": step.engine,
                    "model_id": step.model_id, "external": True}
    return None


def _broker_would_stop_external(args) -> dict | None:
    """Whether a ``model_forge_broker`` claim/swap/release would STOP an
    externally-started server (B8), and the honest detail for the approval
    prompt — else ``None``.

    The eviction/step-down set is computed exactly as the handler computes it:
    a claim/swap's ``plan_load_many(...).release_steps`` (what must be evicted
    to fit the incoming model), and a release's ``restore_plan`` step-down
    (``release_steps``). Only when one of those would ``docker stop`` a
    container Model Forge did NOT start does this escalate — a plugin-owned
    runtime was gated at start and is stopped symmetrically, ungated.

    Fail-safe: any uncertainty escalates. An unresolvable request is refused
    by the handler before any stop, so it does not escalate."""
    action = str(args.get("action") or "").strip()
    model_id = str(args.get("model_id") or "").strip()
    modality = str(args.get("modality") or "").strip()
    try:
        target = get_target(args.get("target"))
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        target = None
    if target is None:
        return None    # the handler refuses an unknown target before any stop
    if action == "release":
        scope_id = str(args.get("scope_id") or "").strip()
        if not scope_id:
            return None    # the handler refuses: nothing released, nothing stops
        try:
            plan = broker_stack(target)["orchestrator"].restore_plan(scope_id)
        except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
            return {"target": target.id, "engine": "(unresolved)",
                    "model_id": "", "external": True}
        if plan is None:
            return None    # unknown scope: release fails closed, stops nothing
        return _external_stop_detail(target, plan.release_steps)
    # claim / swap
    try:
        entry, err = _claim_entry(model_id, modality)
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        return {"target": target.id, "engine": "(unresolved)",
                "model_id": model_id or f"(modality {modality!r})",
                "external": True}
    if err or entry is None:
        return None    # refused by the handler before anything stops
    request, rerr = _claim_request(entry)
    if rerr or request is None:
        return None
    try:
        plan = broker_stack(target)["planner"].plan_load_many(target.id,
                                                              [request])
    except Exception:  # noqa: BLE001 — uncertainty -> escalate (fail-safe)
        return {"target": target.id, "engine": request.engine,
                "model_id": request.model_id, "external": True}
    return _external_stop_detail(target, plan.release_steps)


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


def _seed_remote_pulls(stack: dict, target) -> None:
    """Recorded pulls for a REMOTE target (B10) in the residency summary —
    the persistent pull-records store: engine-observed completions and
    user-confirmed on-target installs. A remote machine cannot be scanned
    over an engine API, so these records are what the swap pre-flight
    trusts; the local target never uses them (its weights are scanned from
    disk)."""
    if target.kind == "local" or stack.get("remote_pulls_seeded"):
        return
    stack["remote_pulls_seeded"] = True
    try:
        from . import pull
        rows = pull.pull_entries(target.id)
    except Exception:  # noqa: BLE001 — the summary degrades, never crashes
        return
    ledger = stack["ledger"]
    for row in rows:
        if row.get("model_id"):
            ledger.record_pull(target.id, str(row["model_id"]),
                               str(row.get("path") or ""),
                               row.get("size_bytes"),
                               source=str(row.get("source") or ""))


def _broker_status(stack: dict, target) -> dict:
    """Honest read-only residency summary: pulled weights with sizes, what is
    resident now, booked ceilings with provenance labels, catalog entries
    that stay unbooked until benched, and current budget usage."""
    ledger = stack["ledger"]
    _seed_local_pulls(stack, target)
    _seed_remote_pulls(stack, target)
    budget = ledger.budget_gb(target.id)
    booked_now = ledger.booked_total_gb(target.id)
    report = ledger.booking_report()
    booked_ids = {row["model_id"] for row in report}
    unbooked = [
        {"model_id": m.id, "modality": m.modality, "vram_gb": None,
         "label": BOOKING_SOURCE_NONE, "hint": UNBOOKED_HINT}
        for m in models_for()
        if (m.status != "stub" and m.id not in _CATALOG_MARKER_IDS
            and _declared_floor(m) is None
            and m.id not in booked_ids)
    ]
    return {
        "ok": True,
        "action": "status",
        "target": target.id,
        "pulled": [{"model_id": r.model_id, "path": r.path,
                    "size_bytes": r.size_bytes, "source": r.source}
                   for r in ledger.pulled(target.id)],
        "resident": [{"model_id": r.model_id, "engine": r.engine,
                      "lane": r.lane} for r in ledger.residents(target.id)],
        "booked": [{"model_id": row["model_id"], "vram_gb": row["vram_gb"],
                    "source": row["source"], "label": row["label"]}
                   for row in report],
        "unbooked": unbooked,
        "budget": {
            "nominal_gb": budget,
            "booked_now_gb": round(booked_now, 3),
            "headroom_gb": (round(budget - booked_now, 3)
                            if budget is not None else None),
        },
        "unbooked_residents": list(ledger.unbooked_residents(target.id)),
        "swap_stats": ledger.swap_stats(),
        "active_scopes": [
            {"scope_id": sid, "model_id": rec.get("model_id"),
             "modality": rec.get("modality"), "kind": rec.get("kind")}
            for sid, rec in sorted(_ACTIVE_SCOPES.items())
            if rec.get("target_id") == target.id],
        "note": ("Read-only report. " + BOOKING_SOURCE_NOTE
                 + " Run action=bench to book a measured peak. Live free "
                   "VRAM, when sampled, only orders which model loads first "
                   "— it never gates fit."),
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
    # A20: consume what the approval gate saw for this release (None = no
    # gate ran on this path; pre-A20 behavior).
    approved = _consume_approved_restore(scope_id)
    result = orchestrator.release(
        scope_id, approved_restore=approved,
        gated_engines=tuple(_PROCESS_ENGINE_PORTS))
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


# --- swap: the one-call weight exchange --------------------------------------

def _swap_release_ids(value) -> list:
    """Explicit outgoing model ids for action=swap: a list of ids or a
    comma-separated string, normalized to clean ids (order kept, duplicates
    dropped)."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    out = []
    for part in parts:
        mid = str(part or "").strip()
        if mid and mid not in out:
            out.append(mid)
    return out


def _swap_dry_run(value) -> bool:
    """The dry_run flag for action=swap: true or 'true' / '1' / 'yes' (any
    case, whitespace tolerated). Default false — the swap executes."""
    if value is True:
        return True
    return isinstance(value, str) \
        and value.strip().lower() in ("true", "1", "yes")


def _broker_swap(stack: dict, target, request, release, dry_run: bool) -> dict:
    """The one-call weight exchange: resident weights step down and the
    incoming model loads in strict measured phases (PREFLIGHT / EVICT / LOAD
    / READY), with measured timings from the injectable clock — never
    estimates. The pre-flight guarantee holds at every layer: nothing is
    evicted before the preflight passes, a failed preflight leaves residency
    untouched, and dry_run=True returns the preflight plan rows and changes
    nothing."""
    _seed_local_pulls(stack, target)
    _seed_remote_pulls(stack, target)
    orchestrator = stack["orchestrator"]
    result = orchestrator.swap(target.id, request, release=release,
                               dry_run=dry_run)
    ok = bool(result.get("ok"))
    out = {
        "ok": ok,
        "action": "swap_failed" if (not ok and not dry_run) else "swap",
        "dry_run": bool(dry_run),
        "target": target.id,
        "model_id": request.model_id,
        "modality": request.modality,
        "evicted": list(result.get("evicted") or []),
        "warm_path": result.get("warm_path"),
        "phases": result.get("phases") or {},
        "timings": dict(result.get("timings") or {}),
        "swap_stats": dict(result.get("swap_stats") or {}),
        "resident_now": list(result.get("resident_now") or []),
        "resident_state": result.get("resident_state", ""),
        "reason": result.get("reason", ""),
    }
    if result.get("note"):
        out["note"] = result["note"]
    if not ok and not dry_run:
        out["error"] = result.get("reason", "the swap did not complete")
    return out


# --- measured footprints: the bench window -----------------------------------

# How many VRAM readings action=bench takes, and how far apart. Fixed, not
# caller-facing knobs (verbs, not knobs) — tests shorten the interval through
# the module constant.
_BENCH_SAMPLES = 6
_BENCH_INTERVAL_S = 0.5
# Same probe style as hardware.py's nvidia-smi CSV query: guarded binary use,
# bounded timeout, tolerant line parsing — read-only memory query.
_NVIDIA_SMI_USED = ("nvidia-smi", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits")


def gpu_vram_used_gb() -> float:
    """Live bench sampler: total GPU memory in use across NVIDIA GPUs, in GB
    (the machine running Model Forge — the same probe scope as the hardware
    report). Raises on failure; the bench window turns a failing sampler into
    an honest row, never a guessed peak."""
    out = subprocess.run(
        _NVIDIA_SMI_USED, capture_output=True, text=True, timeout=10,
        check=True, env=_minimal_env()).stdout
    total_mib = 0.0
    seen = False
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            total_mib += float(line)
        except ValueError:
            continue
        seen = True
    if not seen:
        raise ValueError("nvidia-smi returned no readable memory readings")
    return total_mib / 1024.0


def _live_bench_sampler():
    """(sampler, "") for the live bench window — or (None, honest reason)
    when this machine has no GPU memory the bench can sample. Fails soft:
    the action reports the reason instead of guessing a footprint."""
    hw = probe_hardware()
    if hw.total_vram_gb <= 0:
        return None, ("bench measures real GPU memory and no GPU is present "
                      "on this machine — nothing to measure; run bench "
                      "where the model is loaded")
    if shutil.which("nvidia-smi") is None:
        return None, ("a GPU is present but bench samples GPU memory through "
                      "nvidia-smi, which is not available on this machine — "
                      "nothing to sample")
    return gpu_vram_used_gb, ""


def _broker_bench(stack: dict, target, model_id: str, sampler=None) -> dict:
    """Bench one model's VRAM footprint: measure what runs, book the peak.

    The measurement window samples GPU memory through the VRAM sampler
    (injectable; live it reads real GPU memory via the hardware-probe
    nvidia-smi pattern) while the model is resident or being exercised. The
    observed peak is booked as the model's nominal ceiling through the
    measured-footprint recorder — a measured peak replaces any published
    estimate and is never downgraded by one, and the planner gates fit on it
    from then on. With no GPU to sample the action fails soft with an honest
    reason; with no valid reading in the window it fails closed — a booked
    ceiling must be an observation, never a guess."""
    ledger = stack["ledger"]
    if sampler is None:
        sampler, reason = _live_bench_sampler()
        if sampler is None:
            return {"ok": False, "action": "bench", "target": target.id,
                    "model_id": model_id, "error": reason}
    recorder = MeasuredFootprintRecorder(sampler, model_id, ledger)
    with recorder:
        for _ in range(_BENCH_SAMPLES):
            recorder.sample()
            time.sleep(_BENCH_INTERVAL_S)
    row = recorder.recorded or {}
    if not row.get("ok"):
        return {"ok": False, "action": "bench", "target": target.id,
                "model_id": model_id, "samples": recorder.sample_count,
                "error": row.get("error", "the bench window recorded no "
                                          "peak"),
                "note": ("no measured peak was booked — load the model (or "
                         "exercise it) during the bench window so its VRAM "
                         "footprint is observable")}
    stack["bookings"][model_id] = {"vram_gb": row["vram_gb"],
                                   "source": row["source"], "load_s": None}
    return {
        "ok": True,
        "action": "bench",
        "target": target.id,
        "model_id": model_id,
        "samples": recorder.sample_count,
        "peak_gb": row["vram_gb"],
        "label": row["label"],
        "booking": {"model_id": model_id, "vram_gb": row["vram_gb"],
                    "source": row["source"], "label": row["label"]},
        "gates_fit": True,
        "resident": ledger.is_resident(target.id, model_id),
        "note": (f"measured peak {row['vram_gb']:g} GB booked as this "
                 f"model's nominal ceiling ({row['label']}) — the planner "
                 f"now gates fit on it; measured peaks replace published "
                 f"estimates and are never downgraded by them"),
    }


def model_forge_broker(args: dict | None = None, sampler=None, **kwargs) -> str:
    """Weight residency and workflow scopes on a compute target. Verbs:
    status (read-only residency summary, including measured swap stats),
    claim (take the GPU lane for a modality or model — evict per policy,
    pre-warm, return a scope id with honest progress states), release (end a
    scope and restore what was resident at claim time), plan (dry run of
    claim — nothing executed), swap (exchange the resident weights for one
    incoming model in one choreographed call — PREFLIGHT / EVICT / LOAD /
    READY, every phase measured, never estimated; a failed preflight leaves
    residency untouched; optional release names explicit outgoing ids,
    optional dry_run=true returns the plan rows and changes nothing), bench
    (measure a model's real VRAM peak over a short measurement window and
    book it as the model's nominal ceiling — measured peaks replace
    published estimates and are never downgraded). ``sampler`` is a test
    seam for the bench window's VRAM sampler; the live action samples real
    GPU memory."""
    args = args or {}
    try:
        action = str(args.get("action") or "").strip()
        if action not in ("status", "claim", "release", "plan", "swap",
                          "bench"):
            return _dump({"error": f"unknown action: {action!r}; valid "
                                   f"actions: ['status', 'claim', 'release', "
                                   f"'plan', 'swap', 'bench']"})
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
        if action == "bench":
            model_id = str(args.get("model_id") or "").strip()
            if not model_id:
                return _dump({"error": "model_id required for action=bench: "
                                       "pass the model id to measure (load "
                                       "it first, or exercise it during the "
                                       "bench window)"})
            return _dump(_broker_bench(stack, target, model_id,
                                       sampler=sampler))
        if action == "swap":
            model_id = str(args.get("model_id") or "").strip()
            if not model_id:
                return _dump({"error": "model_id required for action=swap: "
                                       "pass the incoming model id (a "
                                       "catalog model id from "
                                       "model_forge_browse)"})
            entry, err = _claim_entry(model_id,
                                      str(args.get("modality") or ""))
            if err:
                return _dump({"error": err})
            request, err = _claim_request(entry)
            if err:
                return _dump({"error": err})
            return _dump(_broker_swap(
                stack, target, request,
                release=_swap_release_ids(args.get("release")),
                dry_run=_swap_dry_run(args.get("dry_run"))))
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


def _fetch_media_job(target, job_id: str) -> dict:
    """GET one media job record from a target's gateway. Seam: tests replace
    this so no network call happens. Raises on a transport failure — the
    caller turns it into an honest error row (the poll path) or a fail-safe
    escalation (the approval gate), never a raise across a tool boundary."""
    req = urllib.request.Request(
        f"{root_url(target.base_url)}/jobs/{job_id}")
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


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
    # A20: consume what the approval gate saw for this release — the
    # scope-keyed manifest (poll/release-verb paths) or this call's
    # "*{target}" inline manifest (submit path). None means no gate ran on
    # this path (programmatic use) and the pre-A20 behavior applies.
    approved = _consume_approved_restore(scope_id)
    if approved is None:
        approved = _consume_approved_restore(f"*{target.id}")
    result = stack["orchestrator"].release(
        scope_id, approved_restore=approved,
        gated_engines=tuple(_PROCESS_ENGINE_PORTS))
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

# --- resource view: the on-demand memory snapshot -----------------------------

# Design intent (the product owner's): resource visibility for weights and
# memory belongs INSIDE the weight-storage management experience, on demand
# and quiet — never an intrusive always-visible display. Nothing here polls,
# animates, or auto-refreshes; the snapshot is taken only when the user asks
# for it (model_forge_storage action=resources, /model-forge storage
# resources, the pane's collapsed Resource view, the dashboard's collapsed
# Resource view).
RESOURCE_VIEW_NOTE = (
    "On-demand read-only snapshot — shown only when asked for; nothing here "
    "polls or updates on its own (run the action again for a fresh one). "
    "used_gb is live observed usage from the hardware-probe family; "
    "booked_now_gb is the live claims — the booked ceilings of the models "
    "resident right now (nominal catalog ceilings, never a measured usage "
    "figure); free_gb is total minus used. A probe that fails or is not "
    "observable is an honest 'unavailable' row, never a guessed number.")
# One guidance line on the default storage listing, naming the on-demand
# subcommand — the listing itself stays lean (no resource data rides along).
RESOURCES_HINT = (
    "Resource view (on-demand): run /model-forge storage resources — a "
    "quiet memory snapshot (VRAM/RAM usage bars, resident model blocks) "
    "shown only when you ask; nothing polls in the background.")

def _gb_number(value) -> float | None:
    """A finite, non-negative GB number — or None (never a guessed 0)."""
    try:
        gb = float(value)
    except (TypeError, ValueError):
        return None
    return gb if math.isfinite(gb) and gb >= 0.0 else None


def _pct(part, total) -> float | None:
    """part/total as a one-decimal percentage — or None when either side is
    unknown (never a fake 0%)."""
    part = _gb_number(part)
    total = _gb_number(total)
    if part is None or total is None or total <= 0.0:
        return None
    return round(100.0 * part / total, 1)


def _memory_row(kind: str, label: str, total_gb, used_gb, booked_now_gb,
                detail: str = "") -> dict:
    """One memory-kind row for the resource snapshot (pure).

    ``total_gb`` comes from the hardware probe, ``used_gb`` is live observed
    usage (the hardware-probe family), ``booked_now_gb`` is the live claims
    (the booked ceilings of the models resident right now) from the
    residency ledger, and ``free_gb`` is total minus used. A missing or
    failed probe is an honest ``unavailable`` row with a stated reason —
    never a guessed number."""
    total = _gb_number(total_gb)
    used = _gb_number(used_gb)
    booked = _gb_number(booked_now_gb)
    row = {
        "kind": kind,
        "label": label,
        "status": "unavailable",
        "detail": "",
        "total_gb": (round(total, 3)
                     if total is not None and total > 0.0 else None),
        "used_gb": None,
        "booked_now_gb": round(booked, 3) if booked is not None else None,
        "free_gb": None,
        "used_pct": None,
        "booked_now_pct": None,
        "free_pct": None,
    }
    if total is None or total <= 0.0:
        # No total to place usage against: the whole usage picture is
        # unavailable (a used fragment without its total is uninterpretable).
        row["detail"] = (detail or f"no {label} total was reported by the "
                                   "hardware probe")
        return row
    row["booked_now_pct"] = _pct(booked, total)
    if used is None:
        row["detail"] = detail or f"live {label} usage was not observable"
        return row
    row["used_gb"] = round(used, 3)
    free = max(total - used, 0.0)
    row["free_gb"] = round(free, 3)
    row["used_pct"] = _pct(used, total)
    row["free_pct"] = _pct(free, total)
    row["status"] = "ok"
    row["detail"] = detail or ""
    return row


def _meminfo_available_bytes() -> float:
    """Available RAM from /proc/meminfo's MemAvailable (the kernel's own
    estimate of what new allocations can use). Raises when unreadable."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) * 1024.0
    raise ValueError("no MemAvailable reading in /proc/meminfo")


def _ram_used_gb() -> float:
    """Live RAM usage on this machine, in GB (physical RAM minus what is
    available to new allocations). Probe order: /proc/meminfo MemAvailable,
    then the os.sysconf page counts (SC_AVAIL_PAGES where that name exists,
    else SC_AVPHYS_PAGES free pages). Raises when nothing answers; the
    snapshot turns a failing probe into an honest 'unavailable' row, never a
    guessed number."""
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    readers = [_meminfo_available_bytes]
    if "SC_AVAIL_PAGES" in getattr(os, "sysconf_names", {}):
        readers.append(lambda: os.sysconf("SC_AVAIL_PAGES")
                       * os.sysconf("SC_PAGE_SIZE"))
    if "SC_AVPHYS_PAGES" in getattr(os, "sysconf_names", {}):
        readers.append(lambda: os.sysconf("SC_AVPHYS_PAGES")
                       * os.sysconf("SC_PAGE_SIZE"))
    for read_avail in readers:
        try:
            used = total - read_avail()
        except Exception:  # noqa: BLE001 — try the next reader
            continue
        if used >= 0.0:
            return used / (1024 ** 3)
    raise ValueError("no available-RAM reading (tried /proc/meminfo "
                     "MemAvailable and the os.sysconf page counts)")


def _sample_usage(target, kind: str, seam) -> tuple[float | None, str]:
    """(used_gb, "") from a usage probe — or (None, honest reason). Live: the
    hardware-probe family (nvidia-smi GPU memory for VRAM, sysconf pages for
    RAM) and only on the machine running Model Forge — a remote target
    reports its declared totals and the live claims only, never a guessed
    usage number. ``seam`` is the injectable test seam."""
    if seam is None:
        if target.kind != "local":
            return None, (
                "live memory usage is observable only where the hardware "
                "probe runs (this machine); this target reports its declared "
                "totals and the live claims only")
        seam = gpu_vram_used_gb if kind == "vram" else _ram_used_gb
    try:
        used = _gb_number(seam())
    except Exception as exc:  # noqa: BLE001 — an honest unavailable row
        return None, (f"live {kind} usage probe failed: "
                      f"{type(exc).__name__}: {exc}")
    if used is None:
        return None, f"the live {kind} usage probe returned no reading"
    return used, ""


def _queued_load_rows(stack: dict, target_id: str) -> list[dict]:
    """Loads the orchestrator reports as queued or loading right now (the
    open workflow scopes on the target) — rows only while a load is in
    flight, an empty list otherwise."""
    out = []
    orchestrator = (stack or {}).get("orchestrator")
    progress = getattr(orchestrator, "progress", None)
    if not callable(progress):
        return out
    for scope_id, rec in sorted(_ACTIVE_SCOPES.items()):
        if rec.get("target_id") != target_id:
            continue
        row = progress(scope_id)
        if not isinstance(row, dict):
            row = {}
        state = str(row.get("state") or "")
        if state == "queued" or state.startswith("loading"):
            out.append({"scope_id": scope_id,
                        "model_id": rec.get("model_id"),
                        "state": state})
    return out


def resource_snapshot(target, hw=None, stack=None, vram_used=None,
                      ram_used=None) -> dict:
    """One compute target's on-demand resource snapshot (read-only data rows).

    Per memory kind (VRAM, RAM): total from the hardware probe, live
    observed usage from the usage probe, booked_now = the live claims (the
    booked ceilings of the models resident right now) from the residency
    ledger, and free = total minus used — plus the resident model blocks
    (model id, size, engine, lane), any queued loads, and the measured swap
    stats where present. Every seam (hw / stack / vram_used / ram_used) is
    injectable; every failure is an honest 'unavailable' row with a stated
    reason, never a guessed number. Never raises."""
    row = {"target": target.id, "label": target.label, "kind": target.kind}
    hw_detail = ""
    if hw is None:
        try:
            hw = _hw_for_target(target)
        except Exception as exc:  # noqa: BLE001 — honest unavailable rows
            hw = None
            hw_detail = f"hardware probe failed: {type(exc).__name__}: {exc}"
    totals = {"vram": getattr(hw, "total_vram_gb", None) if hw is not None
                     else None,
              "ram": getattr(hw, "ram_gb", None) if hw is not None else None}
    booked_now = None
    resident_rows: list[dict] = []
    try:
        stack = stack if stack is not None else broker_stack(target)
        ledger = stack["ledger"]
        sizes = {p.model_id: p.size_bytes for p in ledger.pulled(target.id)}
        for rec in ledger.residents(target.id):
            booking = ledger.booking(rec.model_id)
            resident_rows.append({
                "model_id": rec.model_id,
                "size_gb": (round(booking.vram_gb, 3)
                            if booking is not None else None),
                "size_label": (booking.label if booking is not None
                               else BOOKING_SOURCE_NONE),
                "size_bytes": sizes.get(rec.model_id),
                "engine": rec.engine,
                "lane": rec.lane,
            })
        booked_now = _gb_number(ledger.booked_total_gb(target.id))
        stats = ledger.swap_stats() or {}
        if stats.get("count"):
            row["swap_stats"] = dict(stats)
    except Exception as exc:  # noqa: BLE001 — the ledger degrades, never crashes
        row["detail"] = (f"residency ledger unavailable: "
                         f"{type(exc).__name__}: {exc}")
    row["residents"] = resident_rows
    row["queued_loads"] = _queued_load_rows(stack if stack is not None else {},
                                           target.id)
    memory = []
    for kind, label, seam in (("vram", "VRAM", vram_used),
                              ("ram", "RAM", ram_used)):
        used, used_detail = _sample_usage(target, kind, seam)
        total = _gb_number(totals.get(kind))
        if hw_detail:
            detail = hw_detail
        elif total is None or total <= 0.0:
            detail = ""   # _memory_row's own honest "no total" reason
        else:
            detail = used_detail
        memory.append(_memory_row(
            kind, label, totals.get(kind), used, booked_now, detail=detail))
    row["memory"] = memory
    return row


def storage_resources(args: dict | None = None) -> dict:
    """The action=resources payload: one on-demand resource snapshot per
    compute target (or one target when ``target`` is passed). Read-only and
    fail-soft; never raises."""
    args = args or {}
    tid = str(args.get("target") or "").strip()
    try:
        if tid:
            one = get_target(tid)
            if one is None:
                return {"error": f"unknown target: {tid}"}
            targets = [one]
        else:
            targets = load_targets()
    except Exception as exc:  # noqa: BLE001 — honest, not fatal
        return {"error": f"compute targets unavailable: "
                         f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "action": "resources",
        "targets": [resource_snapshot(t) for t in targets],
        "note": RESOURCE_VIEW_NOTE,
    }


def cockpit_resources() -> dict:
    """The resource snapshot in the cockpit JSON data-door shape (the
    Desktop pane's and the dashboard's collapsed Resource views read this on
    expand). Read-only — nothing here runs unless the view is opened."""
    return storage_resources()


# --- cockpit data: the /model-forge command and the Desktop pane --------------

# Deletion is deliberately NOT a cockpit verb. A cockpit data read cannot
# ride the human-approval gate (it is not a tool call), so the cockpit and
# the Desktop pane stay read-only and name the delete verb honestly instead.
DELETE_VERB_NOTE = (
    "Read-only listing. To delete an entry, run the model_forge_storage "
    "tool with action=delete and the entry id (or exact path) from this "
    "list — the human-approval gate shows the exact path and size before "
    "anything is removed. This command and the Desktop pane never delete "
    "files.")

_COCKPIT_USAGE = (
    "Model Forge cockpit (read-only). Usage: /model-forge [cockpit|models "
    "[modality]|storage [resources]|resources|help]\n"
    "  cockpit           full snapshot: compute targets + hardware, "
    "declared + detected engines, models with fit data, model storage\n"
    "  models [modality] curated model rows with the model_forge_browse "
    "fit data (advisory)\n"
    "  storage           installed model weights in the "
    "model_forge_storage list shape (sizes, modification times)\n"
    "  storage resources the on-demand resource snapshot as JSON — the "
    "same data the quiet text bars render\n"
    "  resources         the on-demand resource snapshot as JSON (the "
    "pane's Resource-view data door; nothing runs until it is opened)\n"
    "Mutating verbs are tool verbs that ride the human-approval gate "
    "(model_forge_target_add, model_forge_install, model_forge_activate, "
    "model_forge_storage action=delete).")


def cockpit_models(modality: str | None = None) -> dict:
    """Curated model rows with the browse fit data, priced against the local
    target — the model section the Desktop pane reuses. Curated catalog only
    (no live HuggingFace layer): a pane read must stay offline-safe."""
    if modality is not None and modality not in MODALITIES:
        return {"error": f"unknown modality: {modality!r}; valid "
                         f"modalities: {list(MODALITIES)}"}
    targets = load_targets()
    local = next((t for t in targets if t.id == "local"), targets[0])
    hw = _hw_for_target(local)
    available = [e.id for e in _detect("") if e.available]
    entries = models_for(modality=modality or None)
    return {
        "target": local.id,
        "hardware_budget_gb": hw.budget_gb,
        "models": _browse_model_rows(entries, hw, available),
        "note": "Fit verdicts are advisory (the browse fit data, curated "
                "catalog only); engines do authoritative memory sizing at "
                "launch.",
    }


def cockpit_data() -> dict:
    """The read-only cockpit snapshot behind the /model-forge command and
    the Desktop pane: compute targets with hardware, declared + detected
    engines, model rows reusing the browse fit data, and installed model
    storage in the model_forge_storage list shape. Never raises."""
    declared = _settings_engines()
    targets = load_targets()
    out = {
        "ok": True,
        "targets": _target_rows(targets),
        "declared_engines": [dict(row) for row in declared],
        "models": cockpit_models().get("models", []),
        "storage": _storage_listing(),
        "residency": cockpit_residency(),
        "delete_note": DELETE_VERB_NOTE,
    }
    local = next((t for t in targets if t.id == "local"), targets[0])
    invitation = pointing_invitation(targets, _hw_for_target(local),
                                     declared=declared)
    if invitation is not None:
        out["invitation"] = invitation
    return out


def cockpit_residency() -> dict:
    """The broker's read-only weight-residency rows for the local compute
    target — pulled (with sizes + provenance), resident, booked (with honest
    ``measured peak`` / ``published footprint`` labels), unbooked catalog
    entries, and current budget usage. This is the data behind
    ``model_forge_broker action=status``, surfaced for the Desktop cockpit
    pane and the dashboard tab to render (B2). Read-only and bounded: it
    reuses the same honest shape as the broker status verb and never raises
    (a failure is a stated reason)."""
    try:
        targets = load_targets()
        local = next((t for t in targets if t.id == "local"), targets[0])
        stack = broker_stack(local)
        return _broker_status(stack, local)
    except Exception as exc:  # noqa: BLE001 — fail soft: the reason is payload
        return {"ok": False, "target": "local",
                "error": f"residency snapshot unavailable: "
                         f"{type(exc).__name__}: {exc}"}


def cockpit_storage() -> dict:
    """The installed-models listing in the model_forge_storage list shape,
    plus the honest delete-verb note (the cockpit never deletes) and one
    guidance line naming the on-demand resources subcommand."""
    out = dict(_storage_listing())
    out["delete_note"] = DELETE_VERB_NOTE
    out["resources_hint"] = RESOURCES_HINT
    return out


def cockpit_command(raw_args: str = "") -> str:
    """/model-forge — the read-only cockpit data command behind the Desktop
    pane (also handy in chat and the CLI). Subcommands: cockpit (default),
    models [modality], storage, resources (the on-demand memory snapshot —
    also reachable as `storage resources`), help. Returns a JSON string
    (help returns usage text) and never raises. Read-only by design: policy
    decisions and deletions are tool verbs routed through the human-approval
    gate."""
    parts = [p for p in str(raw_args or "").split() if p]
    sub = parts[0].lower() if parts else "cockpit"
    try:
        if sub in ("help", "-h", "--help"):
            return _COCKPIT_USAGE
        if sub == "cockpit":
            return _dump(cockpit_data())
        if sub == "models":
            return _dump(cockpit_models(parts[1] if len(parts) > 1 else None))
        if sub == "storage":
            if len(parts) > 1 and parts[1].lower() == "resources":
                # The on-demand resource snapshot's JSON data door (the same
                # rows the quiet text bars render; nothing runs unless asked).
                return _dump(cockpit_resources())
            return _dump(cockpit_storage())
        if sub == "resources":
            return _dump(cockpit_resources())
        return _dump({"error": f"unknown subcommand: {sub!r}",
                      "usage": _COCKPIT_USAGE})
    except Exception as exc:
        return _err(exc)
