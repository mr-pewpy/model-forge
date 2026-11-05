"""Local engine providers (Strata, Project Maya) — keyless OpenAI-compatible
endpoints registered as first-class Hermes model providers.

``base_url`` stays EMPTY on purpose: a loopback ``base_url`` would map every
127.0.0.1 / localhost endpoint to this provider in URL->provider inference and
disable local-server detection for Ollama, llama.cpp and vLLM (same rule the
bundled LM Studio profile follows). Endpoint resolution at runtime:
``STRATA_BASE_URL`` / ``MAYA_BASE_URL`` env overrides, else the no-auth
placeholder registry row plus local-server detection.
"""

from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="strata",
    display_name="Strata (local)",
    description="Strata — local MoE inference engine (OpenAI + Anthropic compatible)",
    supports_model_listing=False,
    supports_health_check=False,
    env_vars=("STRATA_API_KEY", "STRATA_BASE_URL"),
    fallback_models=("qwen3.8-flash-next", "deepseek-v4.1-flash"),
    model_aliases={"flash-next": "qwen3.8-flash-next"},
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
))
