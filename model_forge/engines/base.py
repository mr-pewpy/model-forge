"""Engine adapter shared types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineStatus:
    id: str
    display_name: str
    available: bool
    endpoint: str = ""
    detail: str = ""
    modalities: tuple[str, ...] = ()
