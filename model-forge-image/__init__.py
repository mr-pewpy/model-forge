"""Model Forge image generation backend.

Routes the native ``image_generate`` tool to ANY OpenAI-compatible images
endpoint — a weights broker, vLLM-Omni (``vllm serve <model> --omni``),
SGLang Diffusion — configured via Desktop Settings -> Plugins (or the
``MODEL_FORGE_IMAGE_BASE_URL`` env var). The endpoint speaks the OpenAI sync
shape (``POST /v1/images/generations`` -> ``{"data": [{"url": ...}]}``);
``image_url`` rides through as the edit input (a local file path becomes a
data: URL first).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)
from agent.secret_scope import get_secret_str

logger = logging.getLogger(__name__)

_ENV_BASE_URL = "MODEL_FORGE_IMAGE_BASE_URL"
_ASPECT_TO_SIZE = {
    "1:1": "1024x1024",
    "16:9": "1024x576",
    "9:16": "576x1024",
    "4:3": "832x1216",
    "3:4": "1216x832",
}


def _base_url() -> str:
    """Settings first (Desktop form), then env through the secret scope."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        settings = (entries.get("model-forge-image") or {}).get("settings") or {}
        url = settings.get("base_url") or ""
    except Exception:
        url = ""
    return (url or get_secret_str(_ENV_BASE_URL)).rstrip("/")


def _default_model() -> str:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        settings = (entries.get("model-forge-image") or {}).get("settings") or {}
        return settings.get("default_model") or ""
    except Exception:
        return ""


def _live_models() -> List[Dict[str, Any]]:
    """GET /v1/models; image workflows first (output_kind filter), else all."""
    base = _base_url()
    if not base:
        return []
    try:
        import httpx
        r = httpx.get(f"{base}/models", timeout=5.0)
        r.raise_for_status()
        rows = r.json().get("data", [])
        image_rows = [m for m in rows if m.get("output_kind") == "image"]
        return image_rows or rows
    except Exception as exc:
        logger.debug("model-forge image catalog fetch failed: %s", exc)
        return []


def _as_data_url(value: str) -> str:
    if value.startswith(("data:", "http://", "https://", "file://")):
        return value
    path = Path(value)
    if not path.exists():
        return value
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _resolve_model(explicit: Optional[str]) -> str:
    """Model id: explicit arg, else settings default, else the live catalog's
    first row. Empty string when nothing names a model (caller reports it)."""
    if explicit:
        return explicit
    default = _default_model()
    if default:
        return default
    models = _live_models()
    return models[0].get("id", "") if models else ""


def _extra_body(image_url: Optional[str],
                reference_image_urls: Optional[list]) -> Dict[str, Any]:
    """Build the OpenAI extra_body: the edit input plus up to 4 reference
    images, local file paths converted to data: URLs first."""
    extra: Dict[str, Any] = {}
    if image_url:
        extra["image_url"] = _as_data_url(str(image_url))
    if reference_image_urls:
        extra["reference_image_urls"] = [
            _as_data_url(str(u)) for u in reference_image_urls[:4]]
    return extra


def _datum_image(datum: Any) -> tuple:
    """(url, b64_json) from a response datum — SDK object or plain dict."""
    if datum is None:
        return "", ""
    image = getattr(datum, "url", None) or ""
    b64 = getattr(datum, "b64_json", None) or ""
    if isinstance(datum, dict):
        image = image or datum.get("url") or ""
        b64 = b64 or datum.get("b64_json") or ""
    return image, b64


class ModelForgeImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "model-forge"

    @property
    def display_name(self) -> str:
        return "Model Forge (local)"

    def is_available(self) -> bool:
        return True  # availability is the engine's job; generate() explains setup

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Model Forge (local)",
            "badge": "local",
            "tag": "any OpenAI-compatible images API — broker, vLLM-Omni, "
                   "SGLang Diffusion. Set the endpoint in Settings -> Plugins.",
            "env_vars": [{
                "key": _ENV_BASE_URL,
                "prompt": "Images endpoint base URL (e.g. http://192.168.1.62:8080/v1)",
                "url": "",
            }],
        }

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

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 4}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument", provider=self.name,
                aspect_ratio=aspect)
        base = _base_url()
        if not base:
            return error_response(
                error=("no images endpoint configured — set the base URL in "
                       "Desktop Settings -> Plugins -> Model Forge Image, or "
                       f"export {_ENV_BASE_URL}"),
                error_type="missing_credentials", provider=self.name,
                prompt=prompt, aspect_ratio=aspect)
        model_id = _resolve_model(kwargs.get("model"))
        if not model_id:
            return error_response(
                error=f"no image model available at {base}/models",
                error_type="no_model", provider=self.name, prompt=prompt,
                aspect_ratio=aspect)

        extra = _extra_body(image_url, reference_image_urls)
        try:
            import openai
            client = openai.OpenAI(api_key="local-endpoint", base_url=base,
                                   timeout=1800.0)
            try:
                resp = client.images.generate(
                    model=model_id, prompt=prompt,
                    size=_ASPECT_TO_SIZE.get(aspect, "1024x1024"),
                    extra_body=extra or None)
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("model-forge image generation failed", exc_info=True)
            return error_response(
                error=f"model-forge image generation failed: {exc}",
                error_type="api_error", provider=self.name, model=model_id,
                prompt=prompt, aspect_ratio=aspect)

        datum = (resp.data or [None])[0]
        image, b64 = _datum_image(datum)
        if not image and b64:
            from agent.image_gen_provider import save_b64_image
            image = str(save_b64_image(str(b64)))
        if not image:
            return error_response(
                error="engine returned neither a URL nor image data",
                error_type="api_error", provider=self.name, model=model_id,
                prompt=prompt, aspect_ratio=aspect)
        return success_response(
            image=str(image), model=model_id, prompt=prompt,
            aspect_ratio=aspect,
            modality="image" if image_url else "text", provider=self.name)


def register(ctx) -> None:
    ctx.register_image_gen_provider(ModelForgeImageGenProvider())
