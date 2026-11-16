"""Exo adapter (github.com/exo-explore/exo).

Exo clusters devices into one distributed inference engine and serves, from
every node, a dashboard and an HTTP API (default http://localhost:52415,
docs/api.md) with several compatible faces: OpenAI Chat Completions
(/v1/chat/completions), the Claude Messages API (/v1/messages), the OpenAI
Responses API (/v1/responses) and an Ollama-compatible API. Because
GET /v1/models is one of those shared faces — and answers from many engines —
it NEVER identifies Exo. "Exo" is claimed only on an Exo-specific
fingerprint from the project's own meta endpoints (docs/api.md §1):

1. GET /node_id — the serving node's identity as a bare JSON string (a
   fresh random id per exo start);
2. fallback: GET /state — the cluster state object (topology, nodes and
   active instances).

Generic routes such as GET /healthcheck (older servers) are deliberately
NOT fingerprints: plenty of software serves them. Without a fingerprint, an
endpoint answering /v1/models is reported as the generic OpenAI-compatible
endpoint it honestly is — engines/openai_compat.py's no-mislabel rule:
the row never claims an engine identity from /v1/models alone.

Lifecycle control goes through the instance API (docs/api.md §2):
POST /place_instance launches a model with server-side placement
(asynchronous — the model downloads and loads engine-natively; clients wait
for readiness via GET /instance/await), and DELETE /instance/{id} stops it
by instance or deployment id — a real unload primitive. broker.ExoLifecycle
drives exactly those two.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import EngineStatus, route_exists, root_url
from . import openai_compat

DEFAULT_ROOT = "http://127.0.0.1:52415"
_TIMEOUT_S = 2.0

def _get_json(url: str):
    """Parsed JSON of any shape (a node id is a bare string), or None."""
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None

def served_models(base_url: str = "") -> list[str]:
    """GET /v1/models — the shared OpenAI listing (ModelList: an OpenAI
    ``{"object": "list", "data": [...]}`` shape in current exo; a bare
    array on some builds). Available models, not necessarily loaded ones."""
    root = root_url(base_url) or DEFAULT_ROOT
    try:
        with urllib.request.urlopen(f"{root}/v1/models",
                                    timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    if isinstance(data, dict):
        rows = data.get("data", [])
    else:
        rows = data if isinstance(data, list) else []
    return [m.get("id", "") for m in rows if isinstance(m, dict)][:8]

def exo_fingerprint(base_url: str = "") -> bool:
    """True only on an Exo-specific surface — never from generic /v1/models.

    Primary: GET /node_id answers the serving node's identity as a bare
    JSON string (docs/api.md §1). Fallback: GET /state answers the cluster
    state object (topology, nodes, active instances). Anything else —
    including any /v1/models list and generic routes like /healthcheck —
    is not proof of Exo."""
    root = root_url(base_url) or DEFAULT_ROOT
    node = _get_json(f"{root}/node_id")
    if isinstance(node, str) and node.strip():
        return True
    state = _get_json(f"{root}/state")
    if isinstance(state, dict) and "instances" in state \
            and ("topology" in state or "nodes" in state
                 or "deployments" in state):
        return True
    return False

def detect(base_url: str = "") -> EngineStatus:
    root = root_url(base_url) or DEFAULT_ROOT
    fingerprinted = exo_fingerprint(base_url)
    models = served_models(base_url)
    if fingerprinted:
        modalities = ["chat", "vision", "ocr"]
        if route_exists(root, "/v1/images/generations"):
            modalities += ["image", "image_edit"]
        return EngineStatus(
            id="exo",
            display_name="Exo",
            available=True,
            endpoint=root,
            detail=(f"models: {', '.join(models)}" if models else "running"),
            modalities=tuple(modalities),
            models_loaded=tuple(m for m in models if m),
        )
    if models:
        # No Exo fingerprint — the honest generic row, byte-identical to
        # openai_compat's own output for this endpoint (same function, same
        # root): /v1/models never identifies the engine, so this row claims
        # no identity and is never selectable as "exo".
        return openai_compat.detect(root)
    return EngineStatus(
        id="exo",
        display_name="Exo",
        available=False,
        detail=(f"no exo answering at {root}" if base_url else
                "not running (default port 52415 — the exo dashboard and "
                "API)"),
        modalities=("chat", "vision", "ocr"),
    )
