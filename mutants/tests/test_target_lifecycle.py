"""Target lifecycle (W1.1 / W1.2 / W1.3): the guided-pointing preview, the
edit/remove/retest verbs, budget provenance labels, and the approval-gate
coverage for every write. Everything runs against the stubbed plugin data
dir — no network, no real Hermes config (same seams as test_tools.py).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from model_forge.targets import (
    ComputeTarget,
    add_target,
    budget_label,
    get_target,
    load_targets,
    remove_target,
    settings_target_ids,
    update_target,
    validate_target_fields,
)


def _stub_hermes_modules(monkeypatch, data_dir: Path):
    plugins = types.ModuleType("plugins")
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_data_dir = lambda name: data_dir / name
    plugins.plugin_storage = storage
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)
    monkeypatch.setenv("HOME", str(data_dir / "home"))
    (data_dir / "home").mkdir(exist_ok=True)
    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.load_config_readonly = lambda: {}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg_mod)


_REMOTE = ComputeTarget(id="gpu-box", label="GPU Box", kind="lan",
                        base_url="http://gpu-box.example.com:8080",
                        vram_gb=48.0, ram_gb=96.0,
                        vram_source="declared", ram_source="declared")


# --- targets.py: validation + provenance -----------------------------------

def test_validate_target_fields():
    assert validate_target_fields(base_url="http://x:1") == ""
    assert "http(s)" in validate_target_fields(base_url="ftp://x:1")
    assert "kind" in validate_target_fields(kind="toaster")
    assert "vram_gb" in validate_target_fields(vram_gb=-1)
    assert "ram_gb" in validate_target_fields(ram_gb="lots")
    assert validate_target_fields(vram_gb=0, ram_gb=0) == ""


def test_budget_label_provenance():
    assert "detected" in budget_label("detected", 24.0)
    assert "confirmed" in budget_label("suggested", 24.0)
    assert "declared" in budget_label("declared", 24.0)
    assert "declared" in budget_label("", 24.0)   # legacy row
    assert "unset" in budget_label("declared", 0.0)


def test_load_targets_tolerates_unknown_and_missing_keys(tmp_path):
    # a row from a different version (extra keys, no provenance fields) must
    # load, not wipe the list
    (tmp_path / "targets.json").write_text(json.dumps(
        [{"id": "old-box", "label": "Old", "base_url": "http://o:1",
          "kind": "lan", "vram_gb": 8, "ram_gb": 16, "note": "",
          "future_field": "ignored"}]))
    rows = load_targets(base_dir=tmp_path)
    ids = [t.id for t in rows]
    assert "old-box" in ids and "local" in ids
    row = get_target("old-box", base_dir=tmp_path)
    assert row.vram_gb == 8 and row.vram_source == ""


def test_update_target_merges_and_refuses_local(tmp_path):
    add_target(_REMOTE, base_dir=tmp_path)
    updated, err = update_target("gpu-box", {"vram_gb": 64.0,
                                             "vram_source": "suggested"},
                                 base_dir=tmp_path)
    assert err == "" and updated.vram_gb == 64.0
    assert updated.vram_source == "suggested"
    assert updated.label == "GPU Box"        # untouched fields kept
    assert get_target("gpu-box", base_dir=tmp_path).base_url == \
        _REMOTE.base_url

    # local is permanent
    updated, err = update_target("local", {"label": "x"}, base_dir=tmp_path)
    assert updated is None and "permanent" in err

    # unknown id and invalid fields fail with honest reasons
    updated, err = update_target("nope", {"label": "x"}, base_dir=tmp_path)
    assert updated is None and "unknown target" in err
    updated, err = update_target("gpu-box", {"base_url": "ftp://x"},
                                 base_dir=tmp_path)
    assert updated is None and "http(s)" in err


def test_update_remove_refuse_settings_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("model_forge.targets.settings_target_ids",
                        lambda: frozenset({"cfg-box"}))
    updated, err = update_target("cfg-box", {"label": "x"}, base_dir=tmp_path)
    assert updated is None and "Desktop settings" in err
    ok, err = remove_target("cfg-box", base_dir=tmp_path)
    assert not ok and "Desktop settings" in err


def test_remove_target_only_forgets_registration(tmp_path):
    add_target(_REMOTE, base_dir=tmp_path)
    ok, err = remove_target("gpu-box", base_dir=tmp_path)
    assert ok and err == ""
    assert get_target("gpu-box", base_dir=tmp_path) is None
    ok, err = remove_target("local", base_dir=tmp_path)
    assert not ok and "permanent" in err
    ok, err = remove_target("nope", base_dir=tmp_path)
    assert not ok and "unknown target" in err


# --- tools.py: the lifecycle verbs -----------------------------------------

def _engine_row(available=True, **kw):
    from model_forge.engines import base
    kw.setdefault("id", "comfyui")
    kw.setdefault("display_name", "ComfyUI")
    kw.setdefault("endpoint", "http://gpu-box.example.com:8080")
    return base.EngineStatus(available=available, **kw)


def test_target_preview_is_read_only_with_probe_and_suggestion(
        tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "_detect",
                        lambda base_url="": [_engine_row(available=True),
                                             _engine_row(
                                                 available=False,
                                                 id="vllm",
                                                 display_name="vLLM")])
    monkeypatch.setattr(
        tools, "_fetch_url",
        lambda url, timeout=0: json.dumps({
            "devices": [{"vram_total": 24 * 1024 ** 3}],
            "system": {"ram_total": 64 * 1024 ** 3}}).encode())
    out = json.loads(tools.model_forge_target_add(
        {"action": "preview", "id": "gpu-box",
         "base_url": "http://gpu-box.example.com:8080", "kind": "lan"}))
    assert out["ok"] is True and out["writes"] is False
    assert out["probe"]["engines_answered"] == ["comfyui"]
    assert out["suggested_budgets"]["vram_gb"] == 24.0
    assert out["suggested_budgets"]["ram_gb"] == 64.0
    assert "confirm" in out["suggested_budgets"]["note"]
    # the suggestion is NEVER written: nothing is registered yet
    assert get_target("gpu-box") is None
    assert "action='add'" in out["next"]


def test_target_preview_no_engine_reports_memory(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    monkeypatch.setattr(tools, "_detect", lambda base_url="": [])
    monkeypatch.setattr(
        tools, "_fetch_url",
        lambda url, timeout=0: (_ for _ in ()).throw(OSError("down")))
    out = json.loads(tools.model_forge_target_add(
        {"action": "preview", "id": "gpu-box",
         "base_url": "http://gpu-box.example.com:8080"}))
    assert out["ok"] is True
    assert out["suggested_budgets"]["suggested"] is False
    assert "declare" in out["suggested_budgets"]["note"]
    assert "no known engine answered" in out["probe"]["detail"]


def test_target_preview_validation(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    for args in ({"id": "local", "base_url": "http://x:1"},
                 {"id": "x", "base_url": "ftp://x"},
                 {"id": "x", "base_url": "http://x", "kind": "toaster"}):
        out = json.loads(tools.model_forge_target_add(
            {"action": "preview", **args}))
        assert out["ok"] is False and out["error"]


def test_target_retest_reprobes_registered_target(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    add_target(_REMOTE)
    monkeypatch.setattr(tools, "_detect",
                        lambda base_url="": [_engine_row(available=True)])
    out = json.loads(tools.model_forge_target_add(
        {"action": "retest", "id": "gpu-box"}))
    assert out["ok"] is True and out["writes"] is False
    assert out["target"]["id"] == "gpu-box"
    assert out["engines_answered"] == ["comfyui"]
    assert "declared by the user" in out["target"]["vram_source"]
    # read-only: the registration is unchanged
    assert get_target("gpu-box") == _REMOTE

    err = json.loads(tools.model_forge_target_add(
        {"action": "retest", "id": "nope"}))
    assert err["ok"] is False and "unknown target" in err["error"]


def test_target_edit_updates_and_labels_provenance(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    add_target(_REMOTE)
    out = json.loads(tools.model_forge_target_add(
        {"action": "edit", "id": "gpu-box", "vram_gb": 64,
         "base_url": "http://gpu-box.example.com:9090"}))
    assert out["ok"] is True and out["edited"] == "gpu-box"
    assert out["before"]["vram_gb"] == 48.0
    assert out["after"]["vram_gb"] == 64.0
    assert out["after"]["base_url"] == "http://gpu-box.example.com:9090"
    assert "declared by the user" in out["after"]["vram_source"]
    assert out["after"]["label"] == "GPU Box"

    # a confirmed suggestion keeps its provenance label
    out = json.loads(tools.model_forge_target_add(
        {"action": "edit", "id": "gpu-box", "vram_gb": 64,
         "vram_source": "suggested"}))
    assert "confirmed by the user" in out["after"]["vram_source"]

    err = json.loads(tools.model_forge_target_add(
        {"action": "edit", "id": "local", "label": "x"}))
    assert err["ok"] is False and "permanent" in err["error"]
    err = json.loads(tools.model_forge_target_add(
        {"action": "edit", "id": "nope", "label": "x"}))
    assert err["ok"] is False and "unknown target" in err["error"]
    err = json.loads(tools.model_forge_target_add({"action": "edit"}))
    assert err["ok"] is False and err["error"] == "id is required"


def test_target_remove_drops_registration_and_stack(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    add_target(_REMOTE)
    out = json.loads(tools.model_forge_target_add(
        {"action": "remove", "id": "gpu-box"}))
    assert out["ok"] is True and out["removed"] == "gpu-box"
    assert all(t["id"] != "gpu-box" for t in out["targets"])
    assert get_target("gpu-box") is None

    err = json.loads(tools.model_forge_target_add(
        {"action": "remove", "id": "local"}))
    assert err["ok"] is False and "permanent" in err["error"]


def test_target_add_budget_provenance_recorded(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    out = json.loads(tools.model_forge_target_add(
        {"id": "gpu-box", "base_url": "http://gpu-box.example.com:8080",
         "vram_gb": 24, "vram_source": "suggested"}))
    assert out["added"] == "gpu-box"
    assert "confirmed by the user" in out["target"]["vram_source"]
    assert "unset" in out["target"]["ram_source"]     # nothing declared

    err = json.loads(tools.model_forge_target_add(
        {"id": "bad", "base_url": "http://x:1", "vram_gb": -4}))
    assert "error" in err

    err = json.loads(tools.model_forge_target_add({"action": "wat"}))
    assert "error" in err and "valid actions" in err["error"]


# --- policy_hook: gate coverage for every write -----------------------------

def test_policy_hook_target_lifecycle_rule_keys(tmp_path, monkeypatch):
    _stub_hermes_modules(monkeypatch, tmp_path)
    from model_forge import tools

    add_target(_REMOTE)

    # add (default action) — as before
    d = tools.policy_hook("model_forge_target_add",
                          {"id": "x", "base_url": "http://x:1"})
    assert d["rule_key"] == "model-forge:target_add"

    # edit and remove ride their own gates, with the honest detail
    d = tools.policy_hook("model_forge_target_add",
                          {"action": "edit", "id": "gpu-box",
                           "vram_gb": 64})
    assert d["rule_key"] == "model-forge:target_edit"
    assert "gpu-box" in d["message"] and "vram_gb" in d["message"]
    d = tools.policy_hook("model_forge_target_add",
                          {"action": "remove", "id": "gpu-box"})
    assert d["rule_key"] == "model-forge:target_remove"
    assert "forgets the registration" in d["message"]

    # preview and retest are read-only: no gate
    assert tools.policy_hook("model_forge_target_add",
                             {"action": "preview", "id": "x",
                              "base_url": "http://x:1"}) is None
    assert tools.policy_hook("model_forge_target_add",
                             {"action": "retest", "id": "x"}) is None