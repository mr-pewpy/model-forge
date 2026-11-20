"""Legacy model-id aliases — the rename seam the facade canonicalizes through.

WHY this exists: model-forge is replacing the production weights broker, but
the broker is NOT the only client with a model id stored on disk. Hermes
providers, the image plugin, the vision cron and OpenViking (.63) all pin ids
that predate the current naming (the broker's "Phase B rename pass"). Those
consumers keep pointing at the facade, so the facade must keep accepting the
old ids. This module is that mapping and nothing else.

The map is deliberately FLAT: the broker's ``aliases:`` section maps every old
id straight to a canonical HF repo id in one hop, so ``resolve`` never chases a
chain. One hop is the whole contract — a ``resolve`` that re-resolved its own
output would be surprising to reason about and would hide accidental cycles.
When a caller overrides ``aliases`` (tests, future config-driven tables) an
unmapped id is returned unchanged: identity is the correct default because the
canonical ids themselves are not in the table.
"""

from __future__ import annotations

# Ported verbatim from the broker config.yaml ``aliases:`` section (the
# production map the live consumers were written against). Keys are the ids
# clients may still send; values are the canonical ids the facade routes on.
#   * chat lane aliases (old advertised / research / deleted VLM ids) all land
#     on the single resident lane Qwen/Qwen3.8-27B-FP8 — the research container
#     and the MiniCPM/DeepSeek-OCR sidecars were purged 2026-11-19 and those
#     requests now ride the resident lane.
#   * gen lane aliases keep the pre-rename krea2/ideogram-4 ids working against
#     the Qwen-Image-2.1 workflows.
DEFAULT_ALIASES: dict[str, str] = {
    # -- chat lane --
    "Qwen/Qwen3.8-27B-W8A8": "Qwen/Qwen3.8-27B-FP8",
    "/root/.cache/huggingface/qwen3.8-obliterated-fp8": "Qwen/Qwen3.8-27B-FP8",
    "research/qwen38-uncensored": "Qwen/Qwen3.8-27B-FP8",
    "OBLITERATUS/Qwen3.8-27B-OBLITERATED": "Qwen/Qwen3.8-27B-FP8",
    # OpenViking (.63) VLM/OCR compat: the small VLM weights are gone; the old
    # ids route to the resident Qwen VL lane.
    "openbmb/MiniCPM-V-4.6": "Qwen/Qwen3.8-27B-FP8",
    "vision/minicpm-v-4.6": "Qwen/Qwen3.8-27B-FP8",
    "ocr/deepseek-ocr-2": "Qwen/Qwen3.8-27B-FP8",
    # -- gen lane --
    "gen/krea2": "gen/qwen-image-2.1",
    "gen/ideogram-4": "gen/qwen-image-2.1",
    "gen/krea2-img2img": "gen/qwen-image-2.1-edit",
}


def resolve(model_id: str, aliases: dict[str, str] | None = None) -> str:
    """Canonicalize a requested model id through the flat alias table.

    Returns the canonical id when ``model_id`` is a known legacy id, otherwise
    ``model_id`` unchanged (identity). One hop only — the output is never fed
    back through the table, matching the broker's flat ``aliases.get`` lookup.
    ``aliases=None`` uses the ported :data:`DEFAULT_ALIASES`; pass a dict to
    override (an empty dict disables aliasing entirely).
    """
    table = DEFAULT_ALIASES if aliases is None else aliases
    return table.get(model_id, model_id)
