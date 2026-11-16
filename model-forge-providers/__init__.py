"""Model Forge model providers — local inference endpoints as first-class
Hermes model providers.

Three profiles are registered:

- ``model-forge`` — the Model Forge catalog row: the curated chat/vision
  models plus live engine-served listings when an engine endpoint is
  detectable (``fetch_models``), with ``fallback_models`` for pickers when
  the fetch fails.
- ``strata`` / ``maya`` — the local MoE inference engines, registered as
  their own provider slugs (resolvable identities per
  model-selection-integration.md).

``base_url`` stays EMPTY on every profile on purpose: a loopback ``base_url``
would map every 127.0.0.1 / localhost endpoint to this provider in
URL->provider inference and disable local-server detection for Ollama,
llama.cpp and vLLM (same rule the bundled LM Studio profile follows).
Endpoint resolution at runtime: ``MODEL_FORGE_BASE_URL`` / ``STRATA_BASE_URL``
/ ``MAYA_BASE_URL`` env overrides, else local-server detection.

Slug discipline (model-selection-integration.md): persisted provider slugs
are resolvable identities only — ``custom``, ``custom:<name>``, registered
provider slugs, and the ``llamacpp``/``llama.cpp``/``llama-cpp`` aliases.
Engine codenames are never slugs. ``custom_provider_entry`` builds the
``custom_providers`` config entry (the documented discovery surface) that
makes a served model selectable via ``/model``; the same shape is written by
the Model Forge tools package on ``model_forge_activate`` (tested for
parity), because the two packages deploy separately and cannot import each
other.

Capability facts mirror ``model_forge/registry.py`` (the source of truth);
fields are OMITTED where the registry states nothing — unknown, never
``False``. Tests cross-check the mirror against the registry.

This package is self-contained (stdlib only): it never imports the Model
Forge tools package, which lives in a separate plugin package.
"""

from __future__ import annotations

import json
import os
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile

# --- shared state vocabulary -------------------------------------------------
# The same sentence everywhere a selection is persisted (the activation plan,
# the approval prompt, this setup flow).
NEXT_SESSION_NOTE = "takes effect on your next session"
API_MODE = "chat_completions"


# --- curated chat/vision catalog --------------------------------------------
# Mirrors model_forge/registry.py (source of truth): the curated entries whose
# modality is chat or vision — the models Model Forge can wire as the Hermes
# chat model. Catalog markers (core-gguf-catalog) and stubs are excluded:
# they are not served model ids and must never be offered to a picker.

CURATED_CHAT_VISION = (
    {
        "id": "qwen3.8-flash-next",
        "name": "Qwen3.8 Flash Next (125B MoE)",
        "modality": "chat",
        "engines": ("strata", "maya"),
        "served_ids": ("qwen3.8-flash-next",),
        "note": "Strata streams experts RAM->GPU; 12 GB VRAM cards work as "
                "hot cache.",
    },
    {
        "id": "deepseek-v4.1-flash",
        "name": "DeepSeek-V4.1-Flash-1017 (748B MoE)",
        "modality": "chat",
        "engines": ("strata", "maya"),
        "served_ids": ("deepseek-v4.1-flash", "deepseek-maya-s"),
        "note": "Maya tiers run it from 128 GB machines upward; Strata on "
                "PC/Mac.",
    },
    {
        "id": "minimax-m3.1-flash",
        "name": "MiniMax-M3.1-Flash (428B MoE)",
        "modality": "chat",
        "engines": ("maya",),
        "served_ids": ("minimax-m3.1-flash", "minimax-maya-xs"),
        "note": "Maya-XS fits a 128 GB machine; vision + thinking levels; "
                "MiniMax Community License (check commercial terms).",
    },
    {
        "id": "glm-5.3-flash",
        "name": "GLM-5.3 Flash (321B MoE)",
        "modality": "chat",
        "engines": ("maya",),
        "served_ids": ("glm-5.3-flash", "glm-maya-s"),
        "note": "Maya tiers (XS..L) self-select from your memory pool; "
                "vision capable.",
    },
    {
        "id": "minicpm-v-4.6",
        "name": "MiniCPM-V 4.6",
        "modality": "vision",
        "engines": ("vllm",),
        "served_ids": ("minicpm-v-4.6",),
        "note": "Small vision sidecar; any OpenAI-compatible multimodal "
                "server works.",
    },
)

CURATED_IDS = tuple(row["id"] for row in CURATED_CHAT_VISION)

# Per-model capability declarations (model_overrides schema). Vision flags
# only where model_forge/registry.py states them; a context window only where
# the registry states one (DeepSeek-OCR-2: "Max context 8192"). Everything
# else stays OMITTED — unknown, not False.
MODEL_CAPABILITIES = {
    "qwen3.8-flash-next": {},
    "deepseek-v4.1-flash": {},
    "deepseek-v4.1-flash-1017": {},
    "deepseek-maya-s": {},
    "minimax-m3.1-flash": {"supports_vision": True, "supports_reasoning": True},
    "minimax-maya-xs": {"supports_vision": True, "supports_reasoning": True},
    "glm-5.3-flash": {"supports_vision": True},
    "glm-maya-s": {"supports_vision": True},
    "minicpm-v-4.6": {"supports_vision": True},
    # Not a chat-model candidate (OCR modality) but served on OpenAI-compatible
    # endpoints: the one model whose context window the registry states.
    "deepseek-ocr-2": {"supports_vision": True, "context_window": 8192},
}


# --- slug rules (model-selection-integration.md) -----------------------------

# The provider slugs this package registers — each one a resolvable identity.
REGISTERED_SLUGS = frozenset({"model-forge", "strata", "maya"})


def custom_provider_slug(name, provider_key: str = "") -> str:
    """Stable ``custom:<name>`` identity for a custom_providers entry — the
    same normalization hermes_cli.providers.custom_provider_slug applies to
    legacy entries (identity = the entry's normalized display name)."""
    identity = str(provider_key or "").strip() or str(name or "").strip()
    normalized = identity.lower().replace(" ", "-")
    return normalized if normalized.startswith("custom:") else "custom:" + normalized


def is_resolvable_slug(slug) -> bool:
    """The reference's identity rule: ``custom``, ``custom:<name>``,
    registered provider slugs, and the llamacpp aliases. Engine codenames are
    never resolvable. (The registered-slug set is this package's own
    registrations; the tools package persists only those or the custom
    class.)"""
    s = str(slug or "").strip()
    if s == "custom":
        return True
    if s.startswith("custom:"):
        return len(s) > len("custom:")
    return s in REGISTERED_SLUGS | {"llamacpp", "llama.cpp", "llama-cpp"}


# --- custom_providers registration ------------------------------------------
# The documented discovery surface: a typed model id a custom_providers entry
# DECLARES routes to that entry automatically, so an activated Model Forge
# model is selectable via /model with no manual configuration. This mirrors
# model_forge/tools.py (the packages deploy separately; a parity test keeps
# the two shapes identical).

def custom_provider_entry(model_id: str, *, name: str, base_url: str) -> dict:
    """The ``custom_providers`` entry declaring a Model Forge-served model.
    Declared models merge on update; per-model metadata is written only where
    it is actually known."""
    return {
        "name": name,
        "base_url": base_url,
        "api_mode": API_MODE,
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
    """Register the entry in config.yaml's ``custom_providers`` list and
    save. Raises on config failure — callers report it honestly."""
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    rows, merged = merge_custom_provider_entry(entry, cfg.get("custom_providers"))
    cfg["custom_providers"] = rows
    save_config(cfg)
    return merged


# --- live engine listings (fetch_models) ------------------------------------

def _load_config():
    try:
        from hermes_cli.config import load_config_readonly
        return load_config_readonly() or {}
    except Exception:  # noqa: BLE001 — config read is advisory, fail soft
        return {}


def known_endpoints(config=None) -> list[str]:
    """Engine root URLs Model Forge knows about, read-only and fail-soft: the
    ``MODEL_FORGE_BASE_URL`` user override, the user-declared engine endpoints
    and compute targets (plugin settings), and the ``custom_providers``
    entries Model Forge registers on activation."""
    cfg = config if isinstance(config, dict) else _load_config()
    out: list[str] = []
    env = os.environ.get("MODEL_FORGE_BASE_URL", "").strip()
    if env:
        out.append(env)
    entries = (cfg.get("plugins") or {}).get("entries") or {}
    settings = (entries.get("model-forge") or {}).get("settings") or {}
    for key in ("engines_json", "targets_json"):
        rows = settings.get(key) or []
        if isinstance(rows, list):
            for row in rows:
                url = str(row.get("base_url") or "").strip() \
                    if isinstance(row, dict) else ""
                if url:
                    out.append(url)
    for entry in (cfg.get("custom_providers") or []):
        if isinstance(entry, dict):
            url = str(entry.get("base_url") or entry.get("url") or "").strip()
            if url:
                out.append(url)
    deduped: list[str] = []
    for url in out:
        if url not in deduped:
            deduped.append(url)
    return deduped


def _probe_served_models(root, timeout: float = 8.0) -> list[str] | None:
    """Live served model ids at an engine root (``{root}/v1/models``, then
    ``{root}/models``). ``None`` when nothing answers — never raises."""
    root = str(root or "").strip().rstrip("/")
    if not root:
        return None
    for url in (root + "/v1/models", root + "/models"):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            items = data if isinstance(data, list) \
                else (data.get("data") or data.get("models") or [])
            ids = []
            for item in items:
                mid = item.get("id") if isinstance(item, dict) else item
                if isinstance(mid, str) and mid.strip():
                    ids.append(mid.strip())
            return ids
        except Exception:  # noqa: BLE001 — probe is advisory, fail soft
            continue
    return None


def served_model_ids(endpoints, timeout: float = 8.0) -> tuple[list[str], bool]:
    """(union of live served ids across *endpoints*, whether any endpoint
    answered). Fail-soft: an unreachable engine is just not listed."""
    live: list[str] = []
    answered = False
    for root in endpoints or ():
        ids = _probe_served_models(root, timeout=timeout)
        if ids is None:
            continue
        answered = True
        for mid in ids:
            if mid not in live:
                live.append(mid)
    return live, answered


# --- the `hermes model` setup step ------------------------------------------

def _read_line(prompt: str) -> str:
    """Input seam (monkeypatched in tests)."""
    return input(prompt)


def resolve_served_model(pick: str, config=None):
    """Round-trip identity for a picker input: ``(served id, declaring
    custom_providers entry)`` for a curated catalog id (or one of its
    engine-native served ids) or a typed served id. ``(None, None)`` when no
    registered entry declares any candidate — Model Forge never persists a
    model it cannot confirm is served."""
    candidates = [pick]
    for row in CURATED_CHAT_VISION:
        if row["id"] == pick:
            candidates = [pick] + [s for s in row["served_ids"] if s != pick]
            break
    cfg = config if isinstance(config, dict) else _load_config()
    for entry in (cfg.get("custom_providers") or []):
        if not isinstance(entry, dict):
            continue
        models = entry.get("models")
        if isinstance(models, dict):
            declared = {str(k) for k in models}
        elif isinstance(models, list):
            declared = {str(m) for m in models if isinstance(m, str)}
        else:
            declared = set()
        for cand in candidates:
            if cand in declared:
                return cand, entry
    return None, None


def model_forge_setup_flow(config, current_model: str = "", args=None, *,
                           engine: str = ""):
    """This provider's own ``hermes model`` step (the documented
    ``ProviderProfile.model_setup_flow`` hook).

    It reuses Model Forge's own verbs where model-selection-integration.md
    allows: the catalog shown is ``model_forge_browse``'s curated chat/vision
    set, serving a model is ``model_forge_install``'s job (this flow never
    downloads), and wiring a served model as the default behind the approval
    gate is ``model_forge_activate``'s job. What this flow does itself is the
    documented persist (``hermes_cli.model_setup.persist_model_selection``)
    plus the ``custom_providers`` registration so ``/model`` can switch to
    the pick afterwards. It never builds a plugin-side ``/model`` selector —
    ``/model`` is core's command."""
    rows = [r for r in CURATED_CHAT_VISION
            if not engine or engine in r["engines"]]
    print("Model Forge — chat/vision models served by local inference "
          "engines.")
    if current_model:
        print(f"  Current default: {current_model}")
    print("  Get a model running with the Model Forge tools: "
          "model_forge_browse (catalog + advisory fit), model_forge_install "
          "(pull a model, or action=engine_install for an engine), "
          "model_forge_activate (wires a served model as the default after "
          "your approval).")
    for i, row in enumerate(rows, 1):
        print(f"  {i}. {row['id']} — {row['name']} ({row['modality']}) — "
              f"{row['note']}")
    pick = (_read_line("Pick a number or type a served model id "
                       "(Enter cancels): ") or "").strip()
    if not pick:
        print("No change.")
        return None
    if pick.isdigit() and 1 <= int(pick) <= len(rows):
        pick = rows[int(pick) - 1]["id"]
    served_id, entry = resolve_served_model(pick, config)
    if entry is None or served_id is None:
        print(f"'{pick}' is not declared by any registered custom_providers "
              "entry, so Model Forge cannot confirm an endpoint serves it. "
              "Pull and activate it first: model_forge_install "
              "(model_id=<id>) then model_forge_activate (model_id=<id>, "
              "apply=true) — activation registers the custom_providers "
              "entry after your approval. No change.")
        return None
    name = str(entry.get("name") or "").strip()
    slug = custom_provider_slug(name)
    base = str(entry.get("base_url") or entry.get("url") or "").strip()
    saved = None
    try:
        from hermes_cli.model_setup import persist_model_selection
        saved = persist_model_selection(served_id, slug, base_url=base,
                                        api_mode=API_MODE)
    except Exception as exc:  # noqa: BLE001 — reported honestly below
        print(f"Could not write the model config ({type(exc).__name__}: "
              f"{exc}). No change.")
        return None
    try:
        write_custom_provider_entry(
            custom_provider_entry(served_id, name=name, base_url=base))
    except Exception as exc:  # noqa: BLE001 — the model section did persist
        print(f"Model saved, but the custom_providers registration failed "
              f"({type(exc).__name__}: {exc}); /model may not list it until "
              f"the entry exists.")
    print(f"Saved model {served_id} on {slug}. {NEXT_SESSION_NOTE}; switch "
          "any time with /model.")
    return saved


def _bind_setup_flow(engine: str):
    def flow(config, current_model="", args=None):
        return model_forge_setup_flow(config, current_model, args,
                                      engine=engine)
    return flow


# --- profiles ---------------------------------------------------------------

class ModelForgeCatalogProfile(ProviderProfile):
    """The Model Forge catalog row: curated chat/vision models plus live
    engine-served listings when an engine endpoint is detectable."""

    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
        """Curated-first merged listing when an endpoint answers; ``None``
        when nothing is detectable or every probe fails, so pickers fall
        back to ``fallback_models`` (honest state, never a guessed list)."""
        endpoints: list[str] = []
        if base_url:
            endpoints.append(str(base_url).strip())
        for url in known_endpoints():
            if url not in endpoints:
                endpoints.append(url)
        if not endpoints:
            return None
        live, answered = served_model_ids(endpoints, timeout=timeout)
        if not answered:
            return None
        out = list(CURATED_IDS)
        for mid in live:
            if mid not in out:
                out.append(mid)
        return out


register_provider(ModelForgeCatalogProfile(
    name="model-forge",
    display_name="Model Forge (local)",
    description=("Model Forge — curated chat/vision models served by local "
                 "inference engines"),
    env_vars=("MODEL_FORGE_API_KEY", "MODEL_FORGE_BASE_URL"),
    base_url="",
    supports_health_check=False,
    fallback_models=CURATED_IDS,
    model_capabilities=dict(MODEL_CAPABILITIES),
    model_setup_flow=model_forge_setup_flow,
))

register_provider(ProviderProfile(
    name="strata",
    display_name="Strata (local)",
    description="Strata — local MoE inference engine (OpenAI + Anthropic compatible)",
    supports_model_listing=False,
    supports_health_check=False,
    env_vars=("STRATA_API_KEY", "STRATA_BASE_URL"),
    fallback_models=("qwen3.8-flash-next", "deepseek-v4.1-flash"),
    model_aliases={"flash-next": "qwen3.8-flash-next"},
    model_capabilities=dict(MODEL_CAPABILITIES),
    model_setup_flow=_bind_setup_flow("strata"),
))

register_provider(ProviderProfile(
    name="maya",
    display_name="Project Maya (local)",
    description="Project Maya — GLM-5.3 / DeepSeek-V4.1 / MiniMax-M3.1 MoE engines on your own GPUs",
    supports_model_listing=False,
    supports_health_check=False,
    env_vars=("MAYA_API_KEY", "MAYA_BASE_URL"),
    fallback_models=("glm-5.3-flash", "deepseek-v4.1-flash-1017", "minimax-m3.1-flash"),
    model_aliases={"glm-flash": "glm-5.3-flash"},
    model_capabilities=dict(MODEL_CAPABILITIES),
    model_setup_flow=_bind_setup_flow("maya"),
))
