"""Engine adapter shared types."""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass


def root_url(base_url: str) -> str:
    """Engine root: strip trailing slash and a trailing /v1 prefix."""
    url = (base_url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def route_exists(root: str, path: str, timeout: float = 2.0) -> bool:
    """True when the server answers on path with anything but 404 — e.g. 405
    for a POST-only route. A cheap capability probe that runs no generation."""
    try:
        with urllib.request.urlopen(root + path, timeout=timeout):
            return True
    except urllib.error.HTTPError as exc:
        return exc.code != 404
    except (urllib.error.URLError, OSError):
        return False


@dataclass(frozen=True)
class EngineStatus:
    id: str
    display_name: str
    available: bool
    endpoint: str = ""
    detail: str = ""
    modalities: tuple[str, ...] = ()
    models_loaded: tuple[str, ...] = ()
