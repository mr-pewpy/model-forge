"""Model Forge video generation backend.

Routes the native ``video_generate`` tool to ANY endpoint speaking the OpenAI
video job API (``POST /v1/videos`` -> poll -> content) — a weights broker,
Sora-style engines — configured via Desktop Settings -> Plugins (or the
``MODEL_FORGE_VIDEO_BASE_URL`` env var). ``image_url`` rides through as the
image-to-video input.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.secret_scope import get_secret_str
from agent.video_gen_provider import OpenAICompatibleVideoGenProvider

logger = logging.getLogger(__name__)

_ENV_BASE_URL = "MODEL_FORGE_VIDEO_BASE_URL"


def _base_url() -> str:
    """Settings first (Desktop form), then env through the secret scope."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        settings = (entries.get("model-forge-video") or {}).get("settings") or {}
        url = settings.get("base_url") or ""
    except Exception:
        url = ""
    return (url or get_secret_str(_ENV_BASE_URL)).rstrip("/")


def _default_model() -> str:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        settings = (entries.get("model-forge-video") or {}).get("settings") or {}
        return settings.get("default_model") or ""
    except Exception:
        return ""


def _live_models() -> List[Dict[str, Any]]:
    """GET /v1/models; video workflows first (output_kind filter), else all."""
    base = _base_url()
    if not base:
        return []
    try:
        import httpx
        r = httpx.get(f"{base}/models", timeout=5.0)
        r.raise_for_status()
        rows = r.json().get("data", [])
        video_rows = [m for m in rows if m.get("output_kind") == "video"]
        return video_rows or rows
    except Exception as exc:
        logger.debug("model-forge video catalog fetch failed: %s", exc)
        return []


class ModelForgeVideoGenProvider(OpenAICompatibleVideoGenProvider):
    _env_key = "MODEL_FORGE_VIDEO_API_KEY"   # unused — local service, see _api_key
    _default_base_url = ""                   # no default endpoint; must configure

    @property
    def name(self) -> str:
        return "model-forge"

    @property
    def display_name(self) -> str:
        return "Model Forge (local)"

    def _api_key(self) -> str:
        return "local-endpoint"  # keyless local engines; non-empty for the base class

    def _base_url(self) -> str:
        return _base_url()

    def is_available(self) -> bool:
        return True  # availability is the engine's job; generate() explains setup

    def list_models(self) -> List[Dict[str, Any]]:
        rows = []
        for m in _live_models():
            mid = m.get("id", "")
            if not mid:
                continue
            rows.append({
                "id": mid,
                "display": f"Model Forge {mid}",
                "speed": "local GPU",
                "strengths": "runs on your own hardware, no cloud cost",
                "price": "free (local)",
            })
        return rows

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Model Forge (local)",
            "badge": "local",
            "tag": "any OpenAI-compatible video job API — broker, Sora-style "
                   "engines. Set the endpoint in Settings -> Plugins.",
            "env_vars": [{
                "key": _ENV_BASE_URL,
                "prompt": "Video endpoint base URL (e.g. http://192.168.1.62:8080/v1)",
                "url": "",
            }],
        }

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        if not _base_url():
            from agent.video_gen_provider import error_response
            return error_response(
                error=("no video endpoint configured — set the base URL in "
                       "Desktop Settings -> Plugins -> Model Forge Video, or "
                       f"export {_ENV_BASE_URL}"),
                error_type="missing_credentials", provider=self.name,
                prompt=(prompt or "").strip())
        if not kwargs.get("model"):
            default = _default_model()
            models = _live_models()
            if default:
                kwargs["model"] = default
            elif models:
                kwargs["model"] = models[0].get("id")
        return super().generate(prompt, **kwargs)


def register(ctx) -> None:
    ctx.register_video_gen_provider(ModelForgeVideoGenProvider())
