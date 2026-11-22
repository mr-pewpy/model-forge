"""Discovery freshness-layer tests: pure fakes, no network, no live config.

Covers the HuggingFace discovery module (ordering, query shape, cache TTL
hit/expiry, fail-soft offline) and the browse integration (curated block
shape unchanged, fresh block present + labeled, unavailable marker when
offline, fresh=false opt-out).
"""

from __future__ import annotations

import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

from model_forge import discovery
from model_forge.registry import MODALITIES, MODELS, ModelEntry

CURATED_ROW_KEYS = {"id", "name", "modality", "modality_label", "engines",
                    "source", "engine_models", "size_gb", "vram_gb_min",
                    "fit", "status", "note"}


def _item(mid, when, downloads=0, tag="image-to-3d"):
    return {"id": mid, "lastModified": when + "T00:00:00.000Z",
            "downloads": downloads, "pipeline_tag": tag}


def _fake_fetch(items_by_tag=None, default=None, exc=None):
    """fetch(url) fake; records call URLs. Maps pipeline_tag -> rows."""
    calls = []

    def fetch(url, timeout=None):
        calls.append(url)
        if exc is not None:
            raise exc
        tag = ""
        for part in url.split("?", 1)[-1].split("&"):
            if part.startswith("pipeline_tag="):
                tag = part.split("=", 1)[1]
        rows = (items_by_tag or {}).get(tag, default if default is not None
                                        else [])
        return json.loads(json.dumps(rows))

    fetch.calls = calls
    return fetch


# --- discovery.py: freshness ordering + row shape ----------------------------

def test_fresh_results_sorted_last_modified_desc(tmp_path):
    fetch = _fake_fetch(default=[
        _item("org/old", "2026-01-05"),
        _item("org/new", "2026-03-01"),
        _item("org/mid", "2026-02-10"),
    ])
    res = discovery.fresh_models("3d", cache_dir=tmp_path, fetch=fetch)
    assert res["live_discovery"] == "ok"
    assert [r["id"] for r in res["models"]] == \
        ["org/new", "org/mid", "org/old"]
    assert [r["last_modified"] for r in res["models"]] == \
        ["2026-03-01", "2026-02-10", "2026-01-05"]


def test_fresh_rows_carry_recency_metadata(tmp_path):
    fetch = _fake_fetch(default=[_item("org/m", "2026-03-01", downloads=42,
                                       tag="text-to-image")])
    res = discovery.fresh_models("image", cache_dir=tmp_path, fetch=fetch)
    row = res["models"][0]
    assert row["id"] == "org/m"
    assert row["last_modified"] == "2026-03-01"
    assert row["downloads"] == 42
    assert row["pipeline_tag"] == "text-to-image"
    assert row["url"] == "https://huggingface.co/org/m"
    assert row["status"] == "fresh"


def test_query_url_shape_per_lane(tmp_path):
    fetch = _fake_fetch(default=[])
    discovery.fresh_models("image", limit=8, cache_dir=tmp_path, fetch=fetch)
    assert len(fetch.calls) == 1
    url = fetch.calls[0]
    assert url.startswith("https://huggingface.co/api/models?")
    assert "pipeline_tag=text-to-image" in url
    assert "sort=lastModified" in url
    assert "direction=-1" in url
    assert "limit=8" in url


def test_multi_lane_modality_queries_each_tag_and_dedupes(tmp_path):
    shared = _item("org/both", "2026-03-01")
    fetch = _fake_fetch(items_by_tag={
        "text-to-3d": [shared, _item("org/t", "2026-02-01", tag="text-to-3d")],
        "image-to-3d": [shared, _item("org/i", "2026-01-01",
                                      tag="image-to-3d")],
    })
    res = discovery.fresh_models("3d", cache_dir=tmp_path, fetch=fetch)
    ids = [r["id"] for r in res["models"]]
    assert ids == ["org/both", "org/t", "org/i"]
    assert len(fetch.calls) == 2  # one call per pipeline lane


def test_limit_cap_keeps_newest(tmp_path):
    rows = [_item(f"org/m{i:02d}", f"2026-03-{i + 1:02d}") for i in range(6)]
    fetch = _fake_fetch(default=rows)
    res = discovery.fresh_models("3d", limit=2, cache_dir=tmp_path, fetch=fetch)
    assert [r["id"] for r in res["models"]] == ["org/m05", "org/m04"]


def test_modality_tags_cover_registry_modalities():
    assert set(discovery.MODALITY_TAGS) == set(MODALITIES)
    for mod, tags in discovery.MODALITY_TAGS.items():
        assert tags and all(isinstance(t, str) and t for t in tags), mod


# --- discovery.py: cache TTL ------------------------------------------------

def test_cache_hit_within_ttl_skips_fetch(tmp_path):
    fetch = _fake_fetch(default=[_item("org/m", "2026-03-01")])
    first = discovery.fresh_models("image", cache_dir=tmp_path, now=1000.0,
                                   fetch=fetch)
    second = discovery.fresh_models("image", cache_dir=tmp_path, now=2000.0,
                                    fetch=fetch)
    assert len(fetch.calls) == 1  # second call served from cache
    assert second["models"] == first["models"]
    assert second["live_discovery"] == "ok"
    assert (tmp_path / "hf_image.json").is_file()


def test_cache_expiry_refetches(tmp_path):
    fetch = _fake_fetch(default=[_item("org/m", "2026-03-01")])
    discovery.fresh_models("image", cache_dir=tmp_path, now=1000.0,
                           ttl=3600.0, fetch=fetch)
    # age exactly at the TTL boundary counts as expired
    res = discovery.fresh_models("image", cache_dir=tmp_path, now=4600.0,
                                 ttl=3600.0, fetch=fetch)
    assert len(fetch.calls) == 2
    assert res["live_discovery"] == "ok"


def test_cache_is_per_modality(tmp_path):
    fetch = _fake_fetch(default=[])
    discovery.fresh_models("image", cache_dir=tmp_path, now=1000.0, fetch=fetch)
    discovery.fresh_models("vision", cache_dir=tmp_path, now=1000.0, fetch=fetch)
    assert len(fetch.calls) == 2  # separate lanes, separate cache entries


# --- discovery.py: fail-soft ------------------------------------------------

def test_offline_failsoft(tmp_path):
    fetch = _fake_fetch(exc=urllib.error.URLError("no route to host"))
    res = discovery.fresh_models("3d", cache_dir=tmp_path, fetch=fetch)
    assert res["live_discovery"] == "unavailable"
    assert res["models"] == []
    assert "URLError" in res["error"]


def test_malformed_payload_failsoft(tmp_path):
    fetch = _fake_fetch(default={"not": "a list"})
    res = discovery.fresh_models("3d", cache_dir=tmp_path, fetch=fetch)
    assert res["live_discovery"] == "unavailable"
    assert res["models"] == []
    assert "ValueError" in res["error"]


def test_empty_lane_is_ok_not_unavailable(tmp_path):
    # world-model lane honestly returns no rows today
    fetch = _fake_fetch(default=[])
    res = discovery.fresh_models("world_model", cache_dir=tmp_path, fetch=fetch)
    assert res["live_discovery"] == "ok"
    assert res["models"] == []
    assert res["error"] == ""


def test_cache_write_failure_is_soft(tmp_path):
    # a read-only cache dir must not break discovery
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    fetch = _fake_fetch(default=[_item("org/m", "2026-03-01")])
    try:
        res = discovery.fresh_models("image", cache_dir=ro / "cache",
                                     fetch=fetch)
        assert res["live_discovery"] == "ok"
        assert len(res["models"]) == 1
    finally:
        ro.chmod(0o700)


# --- registry.py: additive released field -----------------------------------

def test_released_field_backward_compatible():
    # old-style constructors (no released kwarg) still work
    m = ModelEntry(id="x", name="X", modality="chat", engines=("llamacpp",),
                   source="s", status="known", note="n",
                   engine_models={"llamacpp": "x"})
    assert m.released == ""
    # positional construction is unchanged too
    p = ModelEntry("y", "Y", "3d", ("comfyui",), "src")
    assert p.released == ""
    dated = ModelEntry(id="z", name="Z", modality="chat", engines=(),
                       source="", released="2026-01-02")
    assert dated.released == "2026-01-02"
    # the curated catalog itself still constructs (spot check)
    assert all(isinstance(m.released, str) for m in MODELS)


# --- tools.py: browse integration ------------------------------------------

def _stub_plugins(monkeypatch, data_dir: Path):
    plugins = types.ModuleType("plugins")
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_data_dir = lambda name: data_dir / name
    plugins.plugin_storage = storage
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)


def _browse_env(tmp_path, monkeypatch, fetch):
    _stub_plugins(monkeypatch, tmp_path)
    from model_forge import tools
    from model_forge.targets import ComputeTarget, add_target, get_target
    add_target(ComputeTarget(id="gpu-box", label="GPU Box", kind="lan",
                             base_url="http://gpu-box.example.com:8080",
                             vram_gb=48.0, ram_gb=96.0),
               base_dir=tmp_path)
    monkeypatch.setattr(tools, "get_target",
                        lambda tid: get_target(tid, base_dir=tmp_path))
    monkeypatch.setattr(discovery, "fetch_json", fetch)
    return tools


def test_browse_curated_block_unchanged_plus_fresh_block(tmp_path, monkeypatch):
    fetch = _fake_fetch(default=[
        _item("org/fresh-old", "2026-02-01"),
        _item("org/fresh-new", "2026-03-01"),
    ])
    tools = _browse_env(tmp_path, monkeypatch, fetch)
    out = json.loads(tools.model_forge_browse({"target": "gpu-box",
                                               "modality": "3d"}))
    # curated block: unchanged shape
    assert out["target"] == "gpu-box"
    assert out["hardware_budget_gb"] == 136.0
    assert isinstance(out["models"], list) and out["models"]
    for m in out["models"]:
        assert set(m.keys()) == CURATED_ROW_KEYS
    # fresh block: labeled, newest first, dates shown
    assert out["live_discovery"] == "ok"
    assert [r["id"] for r in out["fresh"]] == \
        ["org/fresh-new", "org/fresh-old"]
    assert out["fresh"][0]["last_modified"] == "2026-03-01"
    assert all(r["status"] == "fresh" for r in out["fresh"])
    assert "unvetted" in out["fresh_note"]
    # curated block keys at top level are all still present
    assert {"target", "hardware_budget_gb", "models", "note"} <= set(out)


def test_browse_offline_curated_only_with_unavailable_marker(tmp_path,
                                                             monkeypatch):
    fetch = _fake_fetch(exc=urllib.error.URLError("offline"))
    tools = _browse_env(tmp_path, monkeypatch, fetch)
    out = json.loads(tools.model_forge_browse({"target": "gpu-box",
                                               "modality": "3d"}))
    assert out["live_discovery"] == "unavailable"
    assert out["fresh"] == []
    assert "unavailable" in out["fresh_note"]
    # curated entries survive the outage untouched
    assert isinstance(out["models"], list) and out["models"]
    for m in out["models"]:
        assert set(m.keys()) == CURATED_ROW_KEYS


def test_browse_fresh_false_skips_network(tmp_path, monkeypatch):
    def no_network(url, timeout=None):
        raise AssertionError("network must not be touched when fresh=false")
    tools = _browse_env(tmp_path, monkeypatch, no_network)
    out = json.loads(tools.model_forge_browse({"target": "gpu-box",
                                               "modality": "3d",
                                               "fresh": False}))
    assert out["live_discovery"] == "disabled"
    assert out["fresh"] == []
    assert isinstance(out["models"], list) and out["models"]


def test_browse_fresh_dedupes_against_curated_sources(tmp_path, monkeypatch):
    fetch = _fake_fetch(default=[
        _item("Comfy-Org/Hunyuan-3D-2.0", "2026-03-01"),
        _item("org/novel", "2026-02-01"),
    ])
    tools = _browse_env(tmp_path, monkeypatch, fetch)
    out = json.loads(tools.model_forge_browse({"target": "gpu-box",
                                               "modality": "3d"}))
    assert [r["id"] for r in out["fresh"]] == ["org/novel"]


def test_browse_schema_has_optional_fresh_arg():
    from model_forge import schemas
    props = schemas.BROWSE["parameters"]["properties"]
    assert props["fresh"]["type"] == "boolean"
    assert schemas.BROWSE["parameters"].get("required", []) == []
