"""Model Forge — local models for every modality.

register(ctx) wires the tool schemas to handlers. Engine probing happens at
tool-call time, never at registration, so loading is fast and offline-safe.
"""

from __future__ import annotations

from .model_forge import schemas, tools


def register(ctx):
    """Called once at startup by the Hermes plugin loader."""
    ctx.register_tool(
        name="model_forge_status",
        toolset="model_forge",
        schema=schemas.STATUS,
        handler=tools.model_forge_status,
    )
    ctx.register_tool(
        name="model_forge_browse",
        toolset="model_forge",
        schema=schemas.BROWSE,
        handler=tools.model_forge_browse,
    )
    ctx.register_tool(
        name="model_forge_target_add",
        toolset="model_forge",
        schema=schemas.TARGET_ADD,
        handler=tools.model_forge_target_add,
    )
    ctx.register_tool(
        name="model_forge_install",
        toolset="model_forge",
        schema=schemas.INSTALL,
        handler=tools.model_forge_install,
    )
    ctx.register_tool(
        name="model_forge_activate",
        toolset="model_forge",
        schema=schemas.ACTIVATE,
        handler=tools.model_forge_activate,
    )
    ctx.register_tool(
        name="model_forge_generate_media",
        toolset="model_forge",
        schema=schemas.GENERATE_MEDIA,
        handler=tools.model_forge_generate_media,
    )
