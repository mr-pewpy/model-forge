"""Service assembly + entrypoint for the single-host inference facade.

WHY this module exists: ``facade.py`` (routes), ``broker.py`` (residency policy),
``engines/comfyui.py`` (the workflow runner) and ``registry.py`` (footprints) are
each correct and tested in isolation, but nothing yet *assembles* them into the
single process that replaces a predecessor self-hosted weight broker. That
assembly — and only that — is this file. It binds no policy of its own: the
bookings describe the declared hardware honestly, the lifecycles come from the
runtime declarations, and the orchestrator/planner are the tested broker core.

The reference topology this file encodes (one host, two accelerators, one
container runtime):

* **chat lane** — a tensor-parallel chat server owning BOTH accelerators
  (the ``chat_container``, default ``chat-engine``). It books 0.95 of each
  accelerator's memory (the booking figure).
* **gen lane** — the ComfyUI sidecar pinned to ONE accelerator (the
  ``gen_container``, default ``gen-engine``). Each gen workflow books at
  most one accelerator's memory; a registry footprint is used where the
  registry declares one.

Both lanes contend for the same physical memory, so they share ONE ledger
target (the facade's gen target id). A chat claim brings the chat container
up; a gen claim evicts the chat resident (it cannot fit beside a full-size
chat model) and starts ComfyUI; releasing the gen scope restores the chat
lane. That contention is the whole reason the two lanes are assembled into
one orchestrator rather than two.

Everything external is a seam: the docker CLI runner, the container health
probe, the ComfyUI HTTP client and the clock are all injectable, so the tests
drive the real wiring with fakes and touch no docker and no network. The module
is stdlib-only, like the rest of the package.
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import threading
from http.server import ThreadingHTTPServer

from . import broker
from .broker import (
    EvictionPlanner,
    LoadRequest,
    Orchestrator,
    ResidencyLedger,
    runtime_lifecycle_for,
)
from .engines.comfyui import ComfyWorkflowRunner
from .facade import (
    CHAT_MODEL_ID,
    GEN_MODELS,
    FacadeConfig,
    FacadeService,
    comfy_args_state,
    make_handler,
)
from .registry import ModelEntry, registry_bookings

__all__ = [
    "build_stack",
    "graceful_close",
    "main",
    "preflight_comfy_args",
    "release_scopes",
    "resolve_settings",
]

# ---------------------------------------------------------------------------
# the reference topology
# ---------------------------------------------------------------------------

# One RTX 3090 = 24 GB. The machine has two of them; the chat lane (TP2) owns
# both, the gen lane is pinned to one.
CARD_VRAM_GB = 24.0
MACHINE_VRAM_GB = CARD_VRAM_GB * 2.0          # nominal pool: both cards

# vLLM TP2 books ~22.8 GB/GPU at 0.95 utilization (broker parity). It holds
# both cards, so the chat lane's nominal ceiling is twice that.
CHAT_BOOKED_GB = 22.8 * 2.0                    # 45.6 GB across both cards
GEN_CARD_CEILING_GB = CARD_VRAM_GB             # a gen lane never exceeds one card

# Cold-boot budgets the CALLER passes to the container adapter (never guessed
# by the adapter): vLLM ~8 min cold, the ai-dock ComfyUI image cold-boots for
# minutes. These double as the lane's honest load-duration estimate.
CHAT_BOOT_S = 480.0
GEN_BOOT_S = 600.0

# The registry ``runtime`` entries for the reference topology. They are declared
# here (not in registry.py, which lists *installable* models) because they
# describe THIS host's running containers. ``runtime_lifecycle_for`` turns each
# into a DockerContainerLifecycle; the engine id is the entry's first engine
# ('vllm' / 'comfyui'), which is exactly the engine name the facade's gen
# claim and the chat claim put on their LoadRequests.
#
# The container names / health URLs / boot budgets are read from the resolved
# ``FacadeConfig`` (see ``topology_entries``), so a stub or shadow deployment
# can point the lanes at throwaway containers without touching this file. With
# the config defaults the result is byte-identical to the former hardcoded
# tuple (chat-engine / gen-engine).
def topology_entries(config: "FacadeConfig | None" = None) -> tuple[ModelEntry, ...]:
    """The lane runtime entries, resolved from ``config`` (defaults when
    ``config`` is None)."""
    config = config or FacadeConfig()
    return (
        ModelEntry(
            id=CHAT_MODEL_ID,
            name="Qwen3.8-27B-FP8 vLLM TP2 lane",
            modality="chat",
            engines=("vllm",),
            source=CHAT_MODEL_ID,
            runtime={
                "kind": "container",
                "name": config.chat_container,
                "health_url": config.chat_health_url,
                "boot_budget_s": float(config.chat_boot_s),
                "gpu_count": 2,
            },
        ),
        ModelEntry(
            id="comfyui-gen-lane",
            name="ComfyUI ai-dock gen lane (single card)",
            modality="image",
            engines=("comfyui",),
            source="Comfy-Org/Krea-2",
            runtime={
                "kind": "container",
                "name": config.gen_container,
                # the ai-dock lane; the broker exposed it on 18188. The health
                # URL is its root, as the container adapter probes it.
                "health_url": config.gen_health_url,
                "boot_budget_s": float(config.gen_boot_s),
                "gpu_count": 1,
            },
        ),
    )


# The production-default entries (byte-identical to the former hardcoded
# tuple). Kept as a module constant for importers (tests) that read the
# documented production runtimes.
TOPOLOGY_ENTRIES: tuple[ModelEntry, ...] = topology_entries()

# The bare runner workflow ids the gen lane can claim: the facade advertises
# ``gen/<id>`` and strips the prefix to reach these. Booking every one keeps
# the planner from failing closed ("no booked footprint") on any advertised
# gen model.
GEN_WORKFLOW_IDS: tuple[str, ...] = tuple(
    dict.fromkeys(mid.removeprefix("gen/") for mid in GEN_MODELS))


def topology_bookings(config: "FacadeConfig | None" = None) -> dict[str, dict]:
    """The nominal booked ceilings for the reference topology (ledger booking shape).

    Honest provenance: every figure here is a published/advisory ceiling (the
    broker's utilization-derived vLLM number, registry footprints, the card
    ceiling) — none is a bench-measured peak, so none is labeled measured. A
    measured peak recorded later wins: ``ResidencyLedger.book`` never lets a
    published footprint downgrade a measured one.

    Registry footprints are used where the registry declares one for a bare
    workflow id (e.g. ``qwen-image-2.1``); otherwise a gen workflow books the
    full single-card ceiling. ``load_s`` carries the lane's cold-boot budget as
    the honest load-duration estimate. ``config`` supplies those budgets (and
    the chat id) for a shadow/stub deployment; ``None`` uses the production
    defaults.
    """
    chat_boot = float(config.chat_boot_s) if config is not None else CHAT_BOOT_S
    gen_boot = float(config.gen_boot_s) if config is not None else GEN_BOOT_S
    bookings: dict[str, dict] = {
        CHAT_MODEL_ID: {"vram_gb": CHAT_BOOKED_GB, "source": "published",
                        "load_s": chat_boot},
    }
    reg = registry_bookings()
    for wf in GEN_WORKFLOW_IDS:
        fp = reg.get(wf)
        vram = float(fp["vram_gb"]) if fp else GEN_CARD_CEILING_GB
        bookings[wf] = {
            "vram_gb": min(vram, GEN_CARD_CEILING_GB),
            "source": "published",
            "load_s": gen_boot,
        }
    return bookings


# ---------------------------------------------------------------------------
# the stack assembly
# ---------------------------------------------------------------------------

def build_stack(config: FacadeConfig | None = None, *,
                docker_runner=None, health=None, sleep=None, clock=None,
                http=None, comfy_runner=None) -> dict:
    """Assemble the running facade from the tested parts (no side effects
    beyond installing the optional health seam).

    ``docker_runner`` / ``sleep`` / ``clock`` are the docker CLI seams threaded
    straight to ``runtime_lifecycle_for``; ``health`` installs the container
    health probe (``model_forge.broker._http_healthy`` — a module-level seam the
    container adapter calls, since the registry factory threads no instance
    probe) so callers can wire it in one call; ``http`` is the ComfyUI HTTP
    client seam; ``comfy_runner`` overrides the runner wholesale.

    Returns the wired pieces so callers (the entrypoint, the tests) can inspect
    or drive them: the ledger/planner/orchestrator, the per-engine adapters, the
    runner, the service and the single shared target id.
    """
    config = config or FacadeConfig.from_env()

    entries = topology_entries(config)
    ledger = ResidencyLedger(bookings=topology_bookings(config), clock=clock)
    # The chat lane and the gen lane share ONE physical pool: same target id as
    # the facade's gen claims, so the planner sees the chat resident and evicts
    # it for a gen claim, then restores it on release.
    target_id = config.gen_target_id
    ledger.set_budget(target_id, MACHINE_VRAM_GB)
    planner = EvictionPlanner(ledger, clock=clock)

    if health is not None:
        broker._http_healthy = health

    seams: dict = {}
    if docker_runner is not None:
        seams["runner"] = docker_runner
    if sleep is not None:
        seams["sleep"] = sleep
    if clock is not None:
        seams["clock"] = clock

    adapters: dict = {}
    for entry in entries:
        lifecycle = runtime_lifecycle_for(entry, **seams)
        if lifecycle is not None:                 # all reference lanes are containers
            adapters[lifecycle.engine_id] = lifecycle

    orchestrator = Orchestrator(ledger, planner, adapters, clock=clock)
    runner = comfy_runner or ComfyWorkflowRunner(
        config.comfy_base_url, http=http, output_dir=config.comfy_output_dir)
    service = FacadeService(config, orchestrator=orchestrator, runner=runner)

    return {
        "config": config,
        "target_id": target_id,
        "ledger": ledger,
        "planner": planner,
        "adapters": adapters,
        "orchestrator": orchestrator,
        "runner": runner,
        "service": service,
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# config: env + optional JSON file (file overrides env; explicit wins)
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "0.0.0.0"       # consumers may be LAN-wide
_DEFAULT_PORT = 8080
_DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_COMFYUI_BASE_URL = "http://127.0.0.1:18188"
_DEFAULT_ASSETS_DIR = "/var/lib/model-forge/assets"
_DEFAULT_PUBLIC_BASE_URL = "http://127.0.0.1:8080"

_FACADE_FIELDS = {f.name for f in dataclasses.fields(FacadeConfig)}

# Env-var name -> FacadeConfig field name. The JSON file may use either
# spelling (field name or env name); both normalize to the field.
_ENV_TO_FIELD = {
    "LLM_BASE_URL": "llm_base_url",
    "LLM_SERVED_ID": "llm_served_id",
    "LLM_CANONICAL_ID": "llm_canonical_id",
    "COMFYUI_BASE_URL": "comfy_base_url",
    "ASSETS_DIR": "assets_dir",
    "MODEL_FORGE_ASSETS_DIR": "assets_dir",
    "PUBLIC_BASE_URL": "public_base_url",
    "GEN_TARGET_ID": "gen_target_id",
    "UPSTREAM_TIMEOUT_S": "upstream_timeout_s",
    # -- topology / runtime overrides (shadow + stub deployments) ---------
    # These let a deployment point the two lanes at alternate containers,
    # health URLs, cold-boot budgets and an alternate runner output dir —
    # the same seam a stub-based acceptance run needs. Absent any override,
    # the packaged defaults apply (see FacadeConfig).
    "CHAT_CONTAINER": "chat_container",
    "CHAT_HEALTH_URL": "chat_health_url",
    "GEN_CONTAINER": "gen_container",
    "GEN_HEALTH_URL": "gen_health_url",
    "CHAT_BOOT_S": "chat_boot_s",
    "GEN_BOOT_S": "gen_boot_s",
    "COMFY_OUTPUT_DIR": "comfy_output_dir",
    # The ComfyUI args file the startup preflight checks for --novram (the h3
    # video models require it on a 24 GB card; see facade.comfy_args_state for
    # the 2026-09-22 incident this guards against). Also FACADE_CONFIG
    # settable under either spelling.
    "COMFY_ARGS_FILE": "comfy_args_file",
}

# The fields a file or an explicit override may set: exactly those reachable
# from a documented env var. ``llm_api_key`` is deliberately absent — it is
# env-only (``BROKER_VLLM_API_KEY``, the FacadeConfig field default_factory),
# so neither FACADE_CONFIG JSON nor an explicit kwarg can inject a key over
# the documented contract. Restricting to this whitelist is what keeps the
# key out of code, not merely a docstring promise.
_ALLOWED_FIELDS = frozenset(_ENV_TO_FIELD.values())


def _field_for(name: str) -> str | None:
    """Map an env name or a whitelisted field name to a FacadeConfig field
    name, or None. A field not reachable from a documented env var (e.g.
    ``llm_api_key``) resolves to None, so file/explicit overrides cannot set
    it."""
    if name in _ALLOWED_FIELDS:
        return name
    return _ENV_TO_FIELD.get(name)


def resolve_settings(env: dict | None = None, config_file: str | None = None,
                     *, host: str | None = None, port: int | str | None = None,
                     **explicit) -> tuple[FacadeConfig, str, int]:
    """Resolve host/port + FacadeConfig from env, an optional JSON file and
    explicit overrides.

    Precedence, lowest to highest: environment variables -> the ``FACADE_CONFIG``
    JSON file -> explicit keyword arguments. ``env`` defaults to ``os.environ``;
    ``config_file`` overrides the ``FACADE_CONFIG`` env value. Only fields
    reachable from a documented env var are settable from the file or an
    explicit kwarg (see ``_ALLOWED_FIELDS``) — ``llm_api_key`` is env-only
    (``BROKER_VLLM_API_KEY``, the FacadeConfig field default), so a secret can
    never be injected from JSON or code.
    """
    env = dict(os.environ if env is None else env)

    # serve-owned defaults for the facade fields it names (lowest precedence);
    # env vars, the JSON file and explicit kwargs each override in turn. The
    # defaults are applied explicitly rather than left to FacadeConfig so the
    # deployment's asset volume wins over the package default.
    merged: dict = {
        "llm_base_url": _DEFAULT_LLM_BASE_URL,
        "llm_served_id": CHAT_MODEL_ID,
        "comfy_base_url": _DEFAULT_COMFYUI_BASE_URL,
        "assets_dir": _DEFAULT_ASSETS_DIR,
        "public_base_url": _DEFAULT_PUBLIC_BASE_URL,
        # topology / runtime defaults (default container names + budgets);
        # a stub/shadow deployment overrides these via env or FACADE_CONFIG.
        "chat_container": "chat-engine",
        "chat_health_url": "http://127.0.0.1:8000/health",
        "gen_container": "gen-engine",
        "gen_health_url": "http://127.0.0.1:18188/",
        "chat_boot_s": 480.0,
        "gen_boot_s": 600.0,
        "comfy_output_dir": "/output",
    }
    for env_name, field in _ENV_TO_FIELD.items():
        value = env.get(env_name)
        if value not in (None, ""):
            merged[field] = value

    resolved_host = env.get("FACADE_HOST") or _DEFAULT_HOST
    resolved_port: object = env.get("FACADE_PORT") or _DEFAULT_PORT

    path = config_file if config_file is not None else env.get("FACADE_CONFIG")
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        if not isinstance(doc, dict):
            raise ValueError(
                f"FACADE_CONFIG {path!r} must contain a JSON object, got "
                f"{type(doc).__name__} (fail-closed)")
        for key, value in doc.items():
            if key in ("FACADE_HOST", "host"):
                resolved_host = value
            elif key in ("FACADE_PORT", "port"):
                resolved_port = value
            elif key in ("FACADE_CONFIG",):
                continue
            else:
                field = _field_for(key)
                if field is not None:
                    merged[field] = value

    if host is not None:
        resolved_host = host
    if port is not None:
        resolved_port = port
    for key, value in explicit.items():
        field = _field_for(key)
        if field is not None:
            merged[field] = value

    field_kwargs = {k: v for k, v in merged.items() if k in _FACADE_FIELDS}
    try:
        port_int = int(str(resolved_port))
    except (TypeError, ValueError):
        raise ValueError(
            f"FACADE_PORT must be an integer, got {resolved_port!r} "
            f"(fail-closed)")
    config = FacadeConfig.from_env(**field_kwargs)
    return config, str(resolved_host), port_int


# ---------------------------------------------------------------------------
# graceful shutdown + entrypoint
# ---------------------------------------------------------------------------

def _log(message: str) -> None:
    """Journal-friendly line to stdout (systemd/console read plain lines)."""
    print(f"[model-forge-serve] {message}", flush=True)

def preflight_comfy_args(config: FacadeConfig, *, log=_log) -> str:
    """Startup preflight for the ComfyUI args file (``COMFY_ARGS_FILE``).

    The h3 video models REQUIRE ``--novram`` on smaller cards, and the args
    file has been silently emptied in a real deployment — every h3 run OOMd
    afterwards and nothing surfaced the drift. This check exists so that
    drift is LOUD: a file present without the flag logs an honest WARNING
    naming the file and the consequence (video OOM risk); a named file that is
    missing logs an honest note (the flag can be neither confirmed nor
    denied); a file carrying the flag logs the checked state. WARN, never
    hard-fail — a bad args file must not stop the facade from serving chat and
    image. Returns the ``/health`` ``comfy_args`` verdict (``novram-ok`` /
    ``novram-missing`` / ``unchecked`` — see ``facade.comfy_args_state``).
    """
    path = str(config.comfy_args_file or "")
    state = comfy_args_state(path)
    if not path:
        return state                      # not configured -> unchecked, quiet
    if state == "novram-missing":
        log(f"WARNING: {path} exists but has no --novram flag — the h3 video "
            "models require --novram on a 24 GB card, so every h3 video run "
            "risks a VRAM OOM (comfy_args: novram-missing)")
    elif state == "unchecked":
        log(f"note: COMFY_ARGS_FILE names {path} but the file is missing or "
            "unreadable — --novram cannot be confirmed (comfy_args: unchecked)")
    else:
        log(f"comfy args preflight: --novram present in {path} "
            "(comfy_args: novram-ok)")
    return state


def release_scopes(orchestrator) -> list[str]:
    """Release every still-open scope on ``orchestrator`` (idempotent, honest).

    On shutdown a claim left open holds a lane; releasing it steps the scope's
    models down and restores what was resident at claim time — the same path a
    normal release takes. A release that raises is swallowed (a shutdown must
    never be sunk by one scope) but the scope id is still reported. Reads the
    orchestrator's scope table, which is process-local and only ever mutated
    under its own lock.
    """
    scopes = list((getattr(orchestrator, "_scopes", None) or {}).keys())
    released: list[str] = []
    for scope_id in scopes:
        try:
            orchestrator.release(scope_id)
        except Exception:  # noqa: BLE001 — shutdown must not be sunk by one scope
            pass
        released.append(scope_id)
    return released


def graceful_close(server, stack: dict, *, log=_log) -> list[str]:
    """Stop accepting, finish in-flight requests, then release open scopes.

    ``server.shutdown()`` makes ``serve_forever`` return; ``server_close()``
    joins the request threads (the entrypoint sets ``daemon_threads = False`` so
    that join WAITS for in-flight requests instead of dropping them). Only once
    the socket is closed are open scopes released — a request draining through
    the facade still owns its scope, so releasing earlier would step a lane down
    mid-request.
    """
    released: list[str] = []
    if server is not None:
        log("draining — no new connections accepted, finishing in-flight")
        try:
            server.shutdown()
        except Exception:  # noqa: BLE001 — already shut down is not an error
            pass
        try:
            server.server_close()
        except Exception:  # noqa: BLE001
            pass
    released = release_scopes(stack.get("orchestrator"))
    if released:
        log(f"released {len(released)} open scope(s): {', '.join(released)}")
    return released


def main(argv=None) -> int:
    """Parse config, assemble the stack, serve until SIGTERM/SIGINT.

    Binds a ThreadingHTTPServer and runs it on the calling thread. A signal
    handler sets a stop event and asks the server to stop from a helper thread
    (calling ``shutdown`` from inside ``serve_forever``'s own thread would
    deadlock); ``serve_forever`` then returns and the graceful drain runs. Logs
    plain lines to stdout so journald shows the bind address, the lanes and the
    shutdown reason.
    """
    config, host, port = resolve_settings()
    stack = build_stack(config)
    service = stack["service"]
    # Startup preflight: name the --novram drift loudly before serving (the
    # h3 video lanes OOM without it; warn, never hard-fail).
    preflight_comfy_args(config)

    server = ThreadingHTTPServer((host, port), make_handler(service))
    # Finish in-flight requests on close instead of dropping them.
    server.daemon_threads = False

    def _handle(signum, _frame):
        _log(f"signal {signum} — graceful shutdown requested")
        # shutdown() must run off the serve_forever thread; a daemon helper
        # performs it and the main thread's serve_forever returns.
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):  # not the main thread / unsupported
            pass

    lanes = ", ".join(f"{eid}:{getattr(a, '_container', '?')}"
                      for eid, a in sorted(stack["adapters"].items()))
    _log(f"listening on {host}:{port} — lanes [{lanes}] — target "
         f"{stack['target_id']!r}, budget {MACHINE_VRAM_GB:g} GB")
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        graceful_close(server, stack)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a process
    raise SystemExit(main())
