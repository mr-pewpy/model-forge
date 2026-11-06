"""Engine adapter shared types."""

from __future__ import annotations

import json
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
    """True when the server answers on path with anything but 404.

    Probe chain (the weights broker 404s BOTH wrong-method GETs and
    empty-body POSTs on routes that exist, so status codes alone cannot be
    trusted everywhere):
    1. GET /openapi.json — authoritative route list for FastAPI apps
       (broker, vLLM, SGLang, Maya);
    2. GET the path — 405/other non-404 proves a POST-only route (plain
       FastAPI default);
    3. empty POST — 422/400/405 prove the route on servers that validate
       the body first."""
    probe = urllib.request.Request(root + "/openapi.json")
    try:
        with urllib.request.urlopen(probe, timeout=timeout) as resp:
            spec = json.loads(resp.read().decode("utf-8", "replace"))
        paths = spec.get("paths")
        if isinstance(paths, dict):
            return path in paths or path.rstrip("/") in paths
    except Exception:
        pass
    try:
        with urllib.request.urlopen(root + path, timeout=timeout):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return True
    except (urllib.error.URLError, OSError):
        return False
    req = urllib.request.Request(
        root + path, data=b"{}", method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
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
