"""A19 live-engine fixtures for error-degradation branches.

The remaining coverage gap is mostly error-degradation paths that need a live
engine / filesystem fault to reach (docs/quality-gates.md). These tests inject
those faults — a fake engine HTTP surface that errors, a filesystem that
raises, a config read that fails — so the degradation branches are OBSERVED
rather than left uncovered. No network, no real engine: every seam is a fake.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def _hermes_fakes(monkeypatch, tmp_path, *, config=None, secret=""):
    """Install the Hermes-runtime modules the plugin degrades against."""
    plugins = types.ModuleType("plugins")
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_data_dir = lambda name: tmp_path / name
    plugins.plugin_storage = storage
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)

    cfg = types.ModuleType("hermes_cli.config")
    cfg.load_config_readonly = lambda: (config or {})
    cfg.load_config = lambda: (config or {})
    cfg.save_config = lambda c: None
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg)

    sec = types.ModuleType("agent.secret_scope")
    if secret == "__RAISE__":
        def _get(name, default=""):
            raise RuntimeError("no secret scope")
    else:
        def _get(name, default=""):
            return secret
    sec.get_secret_str = _get
    agent = types.ModuleType("agent")
    agent.secret_scope = sec
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.secret_scope", sec)
    return plugins


# --- credentials: degradation on a missing secret scope ----------------------

def test_credentials_status_degrades_without_secret_scope(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path, secret="__RAISE__")
    from model_forge import credentials
    st = credentials.status()
    # outside a Hermes process there is no secret scope -> both False, no raise
    assert st == {"HF_TOKEN": False, "MODELSCOPE_API_TOKEN": False}


def test_credentials_status_reports_configured_secret(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path, secret="tok")
    from model_forge import credentials
    assert credentials.status()["HF_TOKEN"] is True


def test_credentials_auth_headers_token_never_leaves_host(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path, secret="tok")
    from model_forge import credentials
    # matching host -> Authorization; foreign host -> empty (token never leaves)
    assert credentials.auth_headers("https://huggingface.co/x") == {
        "Authorization": "Bearer tok"}
    assert credentials.auth_headers("https://example.com/x") == {}


# --- discovery: cache + response-shape degradation --------------------------

def test_discovery_cache_none_dir_reads_none():
    from model_forge import discovery
    assert discovery._cache_read(None, "k", 10.0, 0.0) is None
    assert discovery._cache_write(None, "k", [], 0.0) is None  # no raise


def test_discovery_cache_expiry_and_bad_shape(tmp_path):
    from model_forge import discovery
    f = discovery._cache_file(tmp_path, "k")
    f.write_text('{"fetched_at": 0, "results": []}')
    assert discovery._cache_read(tmp_path, "k", 10.0, 100.0) is None  # expired
    f.write_text('{"fetched_at": 0, "results": "not-a-list"}')
    assert discovery._cache_read(tmp_path, "k", 10.0, 1.0) is None
    f.write_text("not json at all")
    assert discovery._cache_read(tmp_path, "k", 10.0, 1.0) is None  # corrupt


def test_discovery_row_handles_bad_downloads():
    from model_forge.discovery import _row
    # non-int downloads degrade to 0, never a crash
    r = _row({"modelId": "org/m", "downloads": "many", "lastModified": "2026"})
    assert r["downloads"] == 0 and r["id"] == "org/m"


def test_discovery_bad_response_shape_is_unavailable(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path)
    from model_forge import discovery
    res = discovery.fresh_models("image", fetch=lambda url: {"not": "a list"})
    assert res["live_discovery"] != "ok"  # honest unavailable, not a fake row


# --- storage: filesystem faults during size + delete ------------------------

def test_storage_tree_size_oserror_is_zero(monkeypatch, tmp_path):
    from model_forge import storage
    monkeypatch.setattr(storage.os, "walk",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert storage._tree_size(tmp_path) == 0


def test_storage_delete_symlink_refused(tmp_path):
    from model_forge import storage
    real = tmp_path / "real.bin"
    real.write_bytes(b"abc")
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    p, err = storage._confine(str(link), [tmp_path])
    assert p is None and "symlink" in err


def test_storage_delete_ref_outside_roots_refused(tmp_path):
    from model_forge import storage
    outside = tmp_path.parent / "outside.bin"
    p, err = storage._confine(str(outside), [tmp_path])
    assert p is None and "storage" in err


def test_storage_delete_unknown_ref_refused(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path)
    from model_forge import storage
    out = storage.delete_entry("no-such-ref", home=tmp_path)
    assert out["ok"] is False and "not an entry" in out["error"]


# --- pull: dest confinement + job write faults -----------------------------

def test_pull_confine_dest_rejects_relative_and_symlink(tmp_path):
    from model_forge.pull import _confine_dest
    p, err = _confine_dest("relative/path", [tmp_path])
    assert p is None and "absolute" in err
    real = tmp_path / "r.bin"
    real.write_bytes(b"x")
    link = tmp_path / "l.bin"
    link.symlink_to(real)
    p, err = _confine_dest(str(link), [tmp_path])
    assert p is None and "symlink" in err


def test_pull_confine_dest_rejects_outside_root(tmp_path):
    from model_forge.pull import _confine_dest
    outside = tmp_path.parent / "outside.bin"
    p, err = _confine_dest(str(outside), [tmp_path])
    assert p is None  # outside the allowed root -> refused


def test_pull_record_missing_ids_refused(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path)
    from model_forge.pull import record_pull_entry
    out = record_pull_entry("", "")
    assert out["ok"] is False and "required" in out["error"]


def test_pull_bad_size_degrades(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path)
    from model_forge.pull import record_pull_entry
    out = record_pull_entry("t", "m", path=str(tmp_path), size_bytes="huge")
    assert out["ok"] is True  # size degrades to None, record still lands


# --- media: route-missing / unknown-modality degradation -------------------

def test_media_unknown_modality_refused():
    from model_forge.media import media_plan
    out = media_plan("not-a-modality", "http://gw:8080")
    assert out["ok"] is False and out["action"] == "unknown_modality"


# --- targets: config read fault degrades to the default local target --------

def test_targets_settings_fault_degrades_to_local(monkeypatch, tmp_path):
    cfg = types.ModuleType("hermes_cli.config")

    def _boom():
        raise RuntimeError("config down")
    cfg.load_config_readonly = _boom
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg)
    from model_forge import targets
    rows = targets.load_targets(tmp_path)
    assert any(t.id == targets.LOCAL_ID for t in rows)  # local always present


def test_targets_settings_rows_merge(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path, config={
        "plugins": {"entries": {"model-forge": {"settings": {
            "targets_json": [{"id": "gpu", "base_url": "http://gpu:8080",
                              "label": "GPU"}]}}}}})
    from model_forge import targets
    rows = targets.load_targets(tmp_path)
    assert any(t.id == "gpu" for t in rows)
    assert any(t.id == targets.LOCAL_ID for t in rows)
