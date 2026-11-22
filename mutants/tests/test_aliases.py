"""Alias seam tests (model_forge/aliases.py) — parity + resolve semantics.

The alias map is a contract with live pre-rename clients (Hermes providers, the
image plugin, OpenViking .63), so parity is asserted against the broker's
config.yaml ``aliases:`` entries explicitly, not against whatever the module
happens to contain. Resolve is pinned to the broker's flat one-hop lookup.
"""

from __future__ import annotations

from model_forge.aliases import DEFAULT_ALIASES, resolve

CHAT = "Qwen/Qwen3.8-27B-FP8"
IMG = "gen/qwen-image-2.1"
EDIT = "gen/qwen-image-2.1-edit"

# The exact legacy -> canonical pairs ported from the broker config. Kept as a
# literal table so a silent change to DEFAULT_ALIASES fails here.
EXPECTED = {
    "Qwen/Qwen3.8-27B-W8A8": CHAT,
    "/root/.cache/huggingface/qwen3.8-obliterated-fp8": CHAT,
    "research/qwen38-uncensored": CHAT,
    "OBLITERATUS/Qwen3.8-27B-OBLITERATED": CHAT,
    "openbmb/MiniCPM-V-4.6": CHAT,
    "vision/minicpm-v-4.6": CHAT,
    "ocr/deepseek-ocr-2": CHAT,
    "gen/krea2": IMG,
    "gen/ideogram-4": IMG,
    "gen/krea2-img2img": EDIT,
}


def test_default_aliases_parity_with_broker_config():
    # No missing / no extra keys, and every target matches the stated one.
    assert DEFAULT_ALIASES == EXPECTED


def test_every_listed_key_maps_to_its_stated_target():
    for legacy, canonical in EXPECTED.items():
        assert DEFAULT_ALIASES[legacy] == canonical
        assert resolve(legacy) == canonical


def test_resolve_identity_for_unmapped_id():
    # Canonical ids are not in the table — identity is the correct answer.
    for canonical in (CHAT, IMG, EDIT, "gen/h3-t2v", "some/other-model"):
        assert resolve(canonical) == canonical


def test_resolve_is_one_hop_only():
    # Flat map: resolving a canonical id never chases further. If a target were
    # itself a key, resolve must still return the target unchanged.
    assert resolve(CHAT) == CHAT
    # An explicitly supplied table replaces the default entirely.
    table = {"a": "b", "b": "c"}
    assert resolve("a", table) == "b"   # one hop, not "c"
    assert resolve("b", table) == "c"   # "b" is a key in THIS table
    # Empty table disables aliasing: everything is identity.
    assert resolve("a", {}) == "a"
