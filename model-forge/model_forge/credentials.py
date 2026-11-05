"""Pull credentials (Hugging Face / ModelScope) — read ONLY via Hermes'
secret scope, never from .env or any credential file (catalog rule 11).

Storage names are declared in plugin.yaml's `config_schema` (`type: secret`
fields render as masked inputs in Desktop Settings -> Plugins -> Model Forge;
the Desktop stores values through the same credential route as provider API
keys). This module only ever reports whether a token is SET and attaches it
to requests for its own hosts.
"""

from __future__ import annotations

from urllib.parse import urlparse

ENV_HF = "HF_TOKEN"
ENV_MODELSCOPE = "MODELSCOPE_API_TOKEN"

# token -> the ONLY hosts it may be sent to
_TOKEN_HOSTS = {
    "huggingface.co": ENV_HF,
    "hf.co": ENV_HF,
    "www.modelscope.cn": ENV_MODELSCOPE,
    "modelscope.cn": ENV_MODELSCOPE,
}


def _get_secret(name: str) -> str:
    try:
        from agent.secret_scope import get_secret_str
        return get_secret_str(name, "")
    except Exception:  # outside a Hermes process (tests) — no secrets here
        return ""


def status() -> dict:
    """Which pull credentials are configured — booleans, never values."""
    return {
        ENV_HF: bool(_get_secret(ENV_HF)),
        ENV_MODELSCOPE: bool(_get_secret(ENV_MODELSCOPE)),
    }


def token_env_for(url: str) -> str:
    """Which credential env var (if any) belongs to this URL's host family.
    Pure routing rule — the token itself is never consulted here."""
    host = (urlparse(url).hostname or "").lower()
    env = _TOKEN_HOSTS.get(host)
    if env is None and host.endswith(".huggingface.co"):
        env = ENV_HF
    if env is None and host.endswith(".modelscope.cn"):
        env = ENV_MODELSCOPE
    return env or ""


def auth_headers(url: str) -> dict:
    """Authorization header for a download URL — only for the token's own
    host family, empty dict otherwise (a token never leaves its host)."""
    env = token_env_for(url)
    if not env:
        return {}
    token = _get_secret(env)
    return {"Authorization": f"Bearer {token}"} if token else {}
