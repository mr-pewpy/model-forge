"""Engine adapter shared types."""

from __future__ import annotations

from dataclasses import dataclass


def root_url(base_url: str) -> str:
    """Engine root: strip trailing slash and a trailing /v1 prefix."""
    url = (base_url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


@dataclass(frozen=True)
class EngineStatus:
    id: str
    display_name: str
    available: bool
    endpoint: str = ""
    detail: str = ""
    modalities: tuple[str, ...] = ()
    models_loaded: tuple[str, ...] = ()
