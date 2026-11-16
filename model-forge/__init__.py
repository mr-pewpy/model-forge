"""Model Forge — local models for every modality.

register(ctx) wires the tool schemas to handlers. Engine probing happens at
tool-call time, never at registration, so loading is fast and offline-safe.

The ``/model-forge`` command surface is defined here: one entry point that
works on every surface commands work on (Desktop chat, Telegram, CLI/TUI).
It renders the read-only cockpit data as plain-text rows that carry the
tools' own state vocabulary (advisory fit verdicts, engine availability with
stated reasons, residency states), and it delegates the historical JSON
subcommands (``cockpit|models|storage``) — the Desktop cockpit pane's data
door — to ``model_forge.tools.cockpit_command`` unchanged, so existing
consumers keep receiving exactly the same JSON. Every command-side read is
bounded by a hard wall-clock timeout and fails soft to an honest reason; the
command never mutates any state (mutating verbs are the tools below, which
ride the human-approval gate through ``policy_hook``).
"""

from __future__ import annotations

import json
import threading

from .model_forge import schemas, tools

# The read-only JSON cockpit implementation (the Desktop pane's slash.exec
# data door). Captured before the extended entry point below is installed as
# tools.cockpit_command, so legacy delegation stays byte-identical and never
# recurses.
_json_cockpit_command = tools.cockpit_command

# Hard wall-clock bound for a command-side data read. Commands run unattended
# (gateway chats, cron, Telegram): a hung engine probe must never hang the
# chat. The reader runs on a daemon thread so a stuck call cannot keep the
# process alive, and the caller gets an honest timeout row instead of
# silence.
COMMAND_TIMEOUT_S = 25.0

# Chat messages cap at 4096 characters on Telegram; cap rendered output a bit
# under that and say so honestly when it happens.
_CHAT_OUTPUT_LIMIT = 3800

# Model rows rendered per browse listing before the honest truncation row.
_BROWSE_ROW_LIMIT = 25


# --- bounded, fail-soft data reads -------------------------------------------

def _bounded_read(label: str, fn, timeout_s=None) -> tuple[str | None, str]:
    """Run one read-only data fetch under a hard timeout.

    Returns ``(value, "")`` on success and ``(None, reason)`` on failure or
    timeout — the reason is written for a human and states that nothing was
    changed. Never raises."""
    deadline = COMMAND_TIMEOUT_S if timeout_s is None else timeout_s
    box: dict = {}

    def _run():
        try:
            box["value"] = fn()
        except Exception as exc:  # fail soft: the reason is the output
            box["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=_run, daemon=True,
                              name=f"model-forge-cmd-{label}")
    worker.start()
    worker.join(deadline)
    if worker.is_alive():
        return None, (f"the {label} read timed out after {deadline:g} s (a "
                      "probe or engine call is not answering); nothing was "
                      "changed — retry later")
    if "error" in box:
        return None, (f"the {label} read failed: {box['error']}; nothing was "
                      "changed")
    return box.get("value"), ""


# --- data seams (JSON strings, exactly the tool output shapes; faked in tests)

def _status_payload() -> str:
    return tools.model_forge_status({})


def _browse_payload(modality: str) -> str:
    args = {"modality": modality} if modality else {}
    return tools.model_forge_browse(args)


def _broker_payload() -> str:
    return tools.model_forge_broker({"action": "status"})


# --- plain-text rendering (chat rows, tool state vocabulary verbatim) --------

def _fmt_bytes(n) -> str:
    try:
        value = float(n or 0)
    except (TypeError, ValueError):
        value = 0.0
    if value >= 1e9:
        return f"{value / 1e9:.1f} GB"
    if value >= 1e6:
        return f"{value / 1e6:.1f} MB"
    if value >= 1e3:
        return f"{value / 1e3:.1f} KB"
    return f"{int(value)} B"


def _num(value, suffix="") -> str:
    return "unknown" if value is None else f"{value}{suffix}"


def _cap_output(text: str) -> str:
    """Keep chat delivery inside the message cap, honestly labeled."""
    if len(text) <= _CHAT_OUTPUT_LIMIT:
        return text
    return (text[:_CHAT_OUTPUT_LIMIT].rstrip()
            + "\n(output truncated for chat delivery; the cockpit|models|"
              "storage subcommands carry the full JSON)")


def _json_error_row(raw: str, label: str) -> str | None:
    """If a tool payload is an error JSON (or unparseable), render an honest
    error row; return None when the payload is usable data."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return f"error: the {label} payload was not valid JSON; nothing was changed"
    if not isinstance(data, dict):
        return (f"error: the {label} payload was not a JSON object; "
                "nothing was changed")
    if data.get("error"):
        return f"error: {data['error']}"
    return None


def _render_invitation(invitation) -> list[str]:
    if not isinstance(invitation, dict):
        return []
    lines = [f"pointing invitation: {invitation.get('reason', '')}"]
    for i, step in enumerate(invitation.get("next_steps") or [], 1):
        lines.append(f"  {i}. {step}")
    return lines


def _render_status(data: dict) -> str:
    """Plain-text status rows: targets + hardware + engines, declared engine
    endpoints, storage summary, pull credentials, pointing invitation."""
    lines = ["Model Forge status (read-only)"]
    targets = data.get("targets") or []
    lines.append(f"targets ({len(targets)}):")
    for t in targets:
        hw = t.get("hardware") or {}
        gpus = hw.get("gpus") or []
        gpu_txt = ", ".join(
            (g.get("name") or "GPU")
            + (f" ({g.get('vram_gb')} GB)" if g.get("vram_gb") else "")
            for g in gpus) or "no GPU detected"
        lines.append(f"  {t.get('id')} [{t.get('kind')}] {t.get('base_url')}"
                     f" — {t.get('label')}")
        lines.append(f"    hardware: {hw.get('platform')} · GPUs: {gpu_txt} · "
                     f"VRAM {_num(hw.get('total_vram_gb'), ' GB')} · "
                     f"RAM {_num(hw.get('ram_gb'), ' GB')}")
        engines = t.get("engines") or []
        if not engines:
            lines.append("    engines: none detected (no engine answered "
                         "the probes)")
        for e in engines:
            loaded = ", ".join(e.get("models_loaded") or []) or "none"
            detail = e.get("detail") or "no detail reported"
            lines.append(
                f"    engine {e.get('id')} ({e.get('name')}): "
                f"available={str(bool(e.get('available'))).lower()} · "
                f"{detail} · models_loaded: {loaded} · "
                f"modalities: {', '.join(e.get('modalities') or []) or 'none'} · "
                f"source={e.get('source') or 'auto-detected'}")
    declared = data.get("declared_engines") or []
    lines.append(f"declared engine endpoints ({len(declared)}):")
    if not declared:
        lines.append("  none — the bring-your-own-engine path is the "
                     "'Declared engine endpoints' setting (engines_json)")
    for row in declared:
        lines.append(f"  {row.get('label')} — {row.get('base_url')} · "
                     f"source={row.get('source') or 'user-declared'}")
    storage = data.get("storage") or {}
    entries = storage.get("entries") or []
    total = sum(int(e.get("size_bytes") or 0) for e in entries)
    roots = storage.get("roots") or []
    skipped = storage.get("roots_skipped") or []
    lines.append(f"storage: {len(entries)} entries · {_fmt_bytes(total)} · "
                 f"roots scanned {len(roots)} · skipped {len(skipped)} "
                 "(model_forge_storage list shape; the 'storage' subcommand "
                 "lists every entry)")
    creds = data.get("pull_credentials") or {}
    if creds:
        lines.append("pull credentials: " + ", ".join(
            f"{k}={str(bool(v)).lower()}" for k, v in sorted(creds.items()))
            + " (values are never shown)")
    lines.extend(_render_invitation(data.get("invitation")))
    return "\n".join(lines)


def _render_browse(data: dict) -> str:
    """Plain-text browse rows: curated models with the model_forge_browse
    fit block (advisory verdicts verbatim)."""
    lines = [f"Model Forge browse — target={data.get('target')} · "
             f"budget={_num(data.get('hardware_budget_gb'), ' GB')} · "
             "fit verdicts are advisory (engines do authoritative memory "
             "sizing at launch)"]
    rows = data.get("models") or []
    lines.append(f"models ({len(rows)}):")
    shown = rows[:_BROWSE_ROW_LIMIT]
    for m in shown:
        fit = m.get("fit") or {}
        parts = [f"  {m.get('name')} "
                 f"[{m.get('modality_label') or m.get('modality')}]"]
        if fit.get("verdict"):
            parts.append(f"fit={fit['verdict']}")
        if m.get("size_gb"):
            parts.append(f"~{m.get('size_gb')} GB")
        engines = ", ".join(m.get("engines") or []) or "none listed"
        parts.append(f"engines: {engines}")
        parts.append(f"status={m.get('status')}")
        lines.append(" · ".join(parts))
        if fit.get("advisory"):
            lines.append(f"    {fit['advisory']}")
        if m.get("note"):
            lines.append(f"    note: {m['note']}")
    if len(rows) > len(shown):
        lines.append(f"  … and {len(rows) - len(shown)} more rows (run "
                     "model_forge_browse for the full JSON)")
    if data.get("note"):
        lines.append(f"note: {data['note']}")
    lines.extend(_render_invitation(data.get("invitation")))
    return "\n".join(lines)


def _render_broker(data: dict) -> str:
    """Plain-text broker rows: pulled weights, residency, bookings with
    provenance labels, unbooked rows, budget usage, active scopes — plus the
    claim/release guidance (this command never mutates)."""
    lines = [f"Broker status — target={data.get('target')} (read-only)"]
    pulled = data.get("pulled") or []
    lines.append(f"pulled ({len(pulled)}): " + ("; ".join(
        f"{p.get('model_id')} ({_fmt_bytes(p.get('size_bytes'))})"
        for p in pulled) or "none"))
    resident = data.get("resident") or []
    lines.append(f"resident ({len(resident)}): " + ("; ".join(
        f"{r.get('model_id')} (engine={r.get('engine')}, lane={r.get('lane')})"
        for r in resident) or "none"))
    booked = data.get("booked") or []
    lines.append(f"booked ({len(booked)}):")
    if not booked:
        lines.append("  none booked yet")
    for row in booked:
        lines.append(f"  {row.get('model_id')} = "
                     f"{_num(row.get('vram_gb'), ' GB')} · "
                     f"{row.get('label')} · source={row.get('source')}")
    unbooked = data.get("unbooked") or []
    if unbooked:
        lines.append(f"unbooked ({len(unbooked)}):")
        for row in unbooked:
            lines.append(f"  {row.get('model_id')} — {row.get('label')}"
                         + (f" · {row.get('hint')}" if row.get("hint") else ""))
    budget = data.get("budget") or {}
    lines.append(f"budget: nominal={_num(budget.get('nominal_gb'), ' GB')} · "
                 f"booked_now={_num(budget.get('booked_now_gb'), ' GB')} · "
                 f"headroom={_num(budget.get('headroom_gb'), ' GB')}")
    for sid in data.get("unbooked_residents") or []:
        lines.append(f"unbooked resident: {sid}")
    scopes = data.get("active_scopes") or []
    lines.append(f"active scopes ({len(scopes)}): " + ("; ".join(
        f"{s.get('scope_id')} {s.get('model_id')} "
        f"({s.get('modality')}, {s.get('kind')})" for s in scopes) or "none"))
    if data.get("note"):
        lines.append(f"note: {data['note']}")
    lines.append("")
    lines.extend(_CLAIM_RELEASE_GUIDANCE)
    return "\n".join(lines)


# --- claim / release guidance (read-only; the verbs are tool verbs) ----------

_APPROVAL_GATE_ROWS = (
    "Approval-gated writes (the human-approval gate; fail closed on denial "
    "or timeout):",
    "  model_forge_target_add — register a compute target",
    "  model_forge_activate with apply=true — write the default-model config",
    "  model_forge_storage action=delete — delete a storage entry "
    "(the gate names the exact path and size first)",
    "  model_forge_install action=engine_install with execute=true — run a "
    "vendor's official engine installer",
)

_CLAIM_RELEASE_GUIDANCE = (
    "Claim / release guidance (tool verbs — this command never mutates "
    "anything):",
    "  claim:   model_forge_broker action=claim model_id=<id> "
    "[modality=<m>] [target=<id>]",
    "           one workflow scope: eviction runs, weights load, and the "
    "scope reports honest states queued -> loading -> ready "
    "(a stated reason on failure, never a silent stall).",
    "  preview: model_forge_broker action=plan ... — a dry run of the claim; "
    "nothing is executed.",
    "  bench:   model_forge_broker action=bench model_id=<id> — measure the "
    "model's real VRAM peak and book it as its nominal ceiling "
    "(measured peaks replace published estimates).",
    "  release: model_forge_broker action=release scope_id=<the scope id "
    "claim returned> — ends the scope and restores what was resident at "
    "claim time.",
    "  status:  /model-forge broker — the read-only residency rows above.",
    "Broker claim/release are not approval-gated: they only move model "
    "weights on compute targets you already control.",
)


def _render_claim_guidance() -> str:
    lines = ["Model Forge claim guidance (read-only — nothing is claimed)"]
    lines.extend(_CLAIM_RELEASE_GUIDANCE)
    lines.append("")
    lines.extend(_APPROVAL_GATE_ROWS)
    return "\n".join(lines)


def _render_release_guidance() -> str:
    lines = ["Model Forge release guidance (read-only — nothing is released)"]
    lines.append("  model_forge_broker action=release scope_id=<the scope id "
                 "claim returned>")
    lines.append("  ends the scope and restores what was resident at claim "
                 "time; states and steps come back in the tool's honest "
                 "progress rows (queued -> loading -> ready on the way in, "
                 "a stated reason on failure).")
    lines.append("  /model-forge broker shows current active scopes and "
                 "residency.")
    lines.append("")
    lines.extend(_APPROVAL_GATE_ROWS)
    return "\n".join(lines)


_USAGE = (
    "Model Forge (read-only command). Usage: /model-forge [subcommand]\n"
    "  cockpit            full JSON snapshot: compute targets + hardware, "
    "declared + detected engines, model fit data, model storage\n"
    "  models [modality]  curated model rows as JSON (the model_forge_browse "
    "fit data)\n"
    "  storage            installed model weights as JSON (the "
    "model_forge_storage list shape: sizes, modification times)\n"
    "  status             plain-text rows: targets, engines, storage, pull "
    "credentials, pointing invitation\n"
    "  browse [modality]  plain-text model rows with advisory fit verdicts\n"
    "  broker [status]    weight-residency rows plus claim/release guidance\n"
    "  claim              claim guidance (read-only; nothing is claimed)\n"
    "  release            release guidance (read-only; nothing is released)\n"
    "  help               this text\n"
    "Mutating verbs are tool verbs that ride the human-approval gate "
    "(model_forge_target_add, model_forge_install, model_forge_activate, "
    "model_forge_storage action=delete). Broker claim/release are tool "
    "verbs too (model_forge_broker)."
)


def cockpit_command(raw_args: str = "") -> str:
    """/model-forge — the Model Forge command surface on every chat surface
    (Desktop, Telegram, CLI/TUI).

    Subcommands: cockpit (default), models [modality], storage — the
    historical read-only JSON data paths, delegated unchanged to
    model_forge.tools.cockpit_command (the Desktop cockpit pane's data
    door) — plus plain-text chat verbs: status, browse [modality],
    broker [status], claim, release, help. Returns text and never raises;
    every data read is bounded by COMMAND_TIMEOUT_S and fails soft to an
    honest reason. Read-only by design: policy decisions and mutations are
    tool verbs routed through the human-approval gate.
    """
    parts = [p for p in str(raw_args or "").split() if p]
    sub = parts[0].lower() if parts else "cockpit"
    try:
        if sub in ("cockpit", "models", "storage"):
            # Historical JSON subcommands: byte-identical delegation so the
            # Desktop pane's slash.exec bridge keeps parsing the same JSON.
            return _json_cockpit_command(raw_args)
        if sub in ("help", "-h", "--help"):
            return _USAGE
        if sub == "status":
            raw, why = _bounded_read("status", _status_payload)
            if why or raw is None:
                return f"error: {why or 'the status read returned no data'}"
            err = _json_error_row(raw, "status")
            if err is not None:
                return err
            return _cap_output(_render_status(json.loads(raw)))
        if sub == "browse":
            modality = parts[1] if len(parts) > 1 else ""
            raw, why = _bounded_read("browse",
                                     lambda: _browse_payload(modality))
            if why or raw is None:
                return f"error: {why or 'the browse read returned no data'}"
            err = _json_error_row(raw, "browse")
            if err is not None:
                return err
            return _cap_output(_render_browse(json.loads(raw)))
        if sub == "broker":
            if len(parts) > 1 and parts[1].lower() not in ("status",):
                if parts[1].lower() == "claim":
                    return _render_claim_guidance()
                if parts[1].lower() == "release":
                    return _render_release_guidance()
                return (f"error: unknown broker view: {parts[1]!r} (valid: "
                        "status)\n" + _USAGE)
            raw, why = _bounded_read("broker", _broker_payload)
            if why or raw is None:
                return f"error: {why or 'the broker read returned no data'}"
            err = _json_error_row(raw, "broker")
            if err is not None:
                return err
            return _cap_output(_render_broker(json.loads(raw)))
        if sub == "claim":
            return _render_claim_guidance()
        if sub == "release":
            return _render_release_guidance()
        return f"error: unknown subcommand: {sub!r}\n{_USAGE}"
    except Exception as exc:  # fail soft: the reason is the output
        return (f"error: the /model-forge {sub} read failed: "
                f"{type(exc).__name__}: {exc}; nothing was changed")


# The command entry point is the extended dispatcher above. The tools module
# keeps the JSON cockpit implementation (_json_cockpit_command) and exposes
# the same entry point under its historical name, so every reference to the
# command handler — the registration below, the Desktop pane bridge, and the
# wiring tests — resolves to exactly one entry point.
tools.cockpit_command = cockpit_command


def register(ctx):
    """Called once at startup by the Hermes plugin loader."""
    ctx.register_tool(
        name="model_forge_status",
        toolset="model_forge",
        schema=schemas.STATUS,
        handler=tools.model_forge_status,
    )
    ctx.register_tool(
        name="model_forge_browse",
        toolset="model_forge",
        schema=schemas.BROWSE,
        handler=tools.model_forge_browse,
    )
    ctx.register_tool(
        name="model_forge_target_add",
        toolset="model_forge",
        schema=schemas.TARGET_ADD,
        handler=tools.model_forge_target_add,
    )
    ctx.register_tool(
        name="model_forge_install",
        toolset="model_forge",
        schema=schemas.INSTALL,
        handler=tools.model_forge_install,
    )
    ctx.register_tool(
        name="model_forge_activate",
        toolset="model_forge",
        schema=schemas.ACTIVATE,
        handler=tools.model_forge_activate,
    )
    ctx.register_tool(
        name="model_forge_generate_media",
        toolset="model_forge",
        schema=schemas.GENERATE_MEDIA,
        handler=tools.model_forge_generate_media,
    )
    ctx.register_tool(
        name="model_forge_storage",
        toolset="model_forge",
        schema=schemas.STORAGE,
        handler=tools.model_forge_storage,
    )
    ctx.register_tool(
        name="model_forge_broker",
        toolset="model_forge",
        schema=schemas.BROKER,
        handler=tools.model_forge_broker,
    )
    # Read-only cockpit data command: the Desktop cockpit pane renders this
    # JSON (through the documented slash.exec door), and it doubles as the
    # chat/CLI command surface on every surface commands work on (Desktop
    # chat, Telegram, CLI/TUI): cockpit|models [modality]|storage return the
    # JSON rows, status|browse [modality]|broker|claim|release render
    # plain-text rows for chat. Read-only by design — mutating verbs are the
    # tools above, which ride the human-approval gate through policy_hook.
    ctx.register_command(
        "model-forge",
        tools.cockpit_command,
        description=("Model Forge cockpit: compute targets, GPU hardware, "
                     "declared + detected engines, model fit data, and model "
                     "storage (read-only); plain-text status, browse and "
                     "broker rows for chat."),
        args_hint=("[cockpit|models [modality]|storage|status|"
                   "browse [modality]|broker|claim|release|help]"),
    )
    # Policy gate: registering a compute target, rewriting the default model
    # config, deleting a model-storage entry, and running a vendor engine
    # installer (model_forge_install action=engine_install, execute=true) are
    # user decisions, not model decisions — tools.policy_hook escalates all
    # four to Hermes' human-approval gate (fail closed on denial or timeout).
    ctx.register_hook("pre_tool_call", tools.policy_hook)
