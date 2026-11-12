"""Storage inventory and user-approved delete tests (model_forge/storage.py).

The storage tool walks the plugin downloads dir plus the engine-managed
model directories under the user home. Tests fake those roots under tmp_path
(the ``plugins.plugin_storage`` stub pattern from tests/test_tools.py, with
HOME pointed at a temp dir) so nothing touches the real home directory, and
they exercise the delete safety rules: listed entries only, containment
re-verification, symlink refusal, freed-bytes reporting, and the fail-closed
human-approval gate.
"""

from __future__ import annotations

import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from model_forge import storage
from model_forge.storage import StorageRoot, delete_entry, list_storage


# --- shared stubs -----------------------------------------------------------

def _stub_plugins(monkeypatch, data_dir: Path):
    """Point the lazy plugins.plugin_storage import at a fake data dir."""
    plugins = types.ModuleType("plugins")
    store = types.ModuleType("plugins.plugin_storage")
    store.plugin_data_dir = lambda name: data_dir / name
    plugins.plugin_storage = store
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", store)


def _fake_home(monkeypatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    return home


# --- list -------------------------------------------------------------------

def test_list_shape_with_faked_roots(tmp_path):
    downloads = tmp_path / "downloads"
    ollama = tmp_path / "ollama-models"
    for d in (downloads, ollama):
        d.mkdir()
    model_dir = downloads / "model-a"
    model_dir.mkdir()
    (model_dir / "w1.safetensors").write_bytes(b"1" * 5)
    (model_dir / "w2.bin").write_bytes(b"2" * 3)
    single = downloads / "single.gguf"
    single.write_bytes(b"3" * 4)
    os.utime(single, (1_600_000_000, 1_600_000_000))
    (ollama / "weights.bin").write_bytes(b"4" * 2)

    out = list_storage(roots=[StorageRoot("plugin-downloads", "dl", downloads),
                              StorageRoot("ollama-models", "ol", ollama)])
    assert out["ok"] is True
    assert "usage is not tracked" in out["note"]
    by_name = {e["name"]: e for e in out["entries"]}
    assert set(by_name) == {"model-a", "single.gguf", "weights.bin"}
    required = {"id", "name", "path", "root", "type", "size_bytes",
                "last_modified"}
    for entry in out["entries"]:
        assert required <= set(entry)
        assert os.path.isabs(entry["path"])
        assert isinstance(entry["size_bytes"], int)
        datetime.fromisoformat(entry["last_modified"])  # parses
    # a folder entry adds up every file inside; a file is its own size
    assert by_name["model-a"]["type"] == "dir"
    assert by_name["model-a"]["size_bytes"] == 8
    assert by_name["single.gguf"]["type"] == "file"
    assert by_name["single.gguf"]["size_bytes"] == 4
    # last_modified is the filesystem modification time, labeled honestly
    expected = datetime.fromtimestamp(1_600_000_000, timezone.utc).isoformat(
        timespec="seconds")
    assert by_name["single.gguf"]["last_modified"] == expected
    # every entry records which root it came from
    assert by_name["model-a"]["root"] == "plugin-downloads"
    assert by_name["weights.bin"]["root"] == "ollama-models"
    assert by_name["model-a"]["id"] == "plugin-downloads:model-a"
    assert by_name["weights.bin"]["id"] == "ollama-models:weights.bin"


def test_list_default_roots_use_plugin_dir_and_home(tmp_path, monkeypatch):
    _stub_plugins(monkeypatch, tmp_path)
    home = _fake_home(monkeypatch, tmp_path)
    downloads = tmp_path / "model-forge" / "downloads"
    ollama = home / ".ollama" / "models"
    hub = home / ".cache" / "huggingface" / "hub"
    for d in (downloads, ollama, hub):
        d.mkdir(parents=True)
        (d / "one.bin").write_bytes(b"1")

    roots = storage.storage_roots()
    assert [r.key for r in roots] == ["plugin-downloads", "ollama-models",
                                     "huggingface-hub"]
    assert [r.path for r in roots] == [downloads, ollama, hub]

    out = list_storage()
    assert {e["root"] for e in out["entries"]} == {
        "plugin-downloads", "ollama-models", "huggingface-hub"}
    assert out["roots_skipped"] == []
    assert len(out["entries"]) == 3


def test_missing_and_unreadable_roots_fail_soft(tmp_path, monkeypatch):
    good = tmp_path / "good"
    good.mkdir()
    (good / "kept.bin").write_bytes(b"12")
    missing = tmp_path / "never-created"
    not_a_dir = tmp_path / "a-file"
    not_a_dir.write_bytes(b"x")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.bin").write_bytes(b"y")

    real_iterdir = Path.iterdir

    def _iterdir(self):
        if self == locked:
            raise PermissionError(13, "Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _iterdir)
    out = list_storage(roots=[
        StorageRoot("r-good", "good", good),
        StorageRoot("r-missing", "missing", missing),
        StorageRoot("r-file", "not-a-dir", not_a_dir),
        StorageRoot("r-locked", "locked", locked),
    ])
    # fail-soft: the healthy root is listed, the bad ones are skipped
    assert out["ok"] is True
    assert [e["name"] for e in out["entries"]] == ["kept.bin"]
    skipped = {s["root"]: s["reason"] for s in out["roots_skipped"]}
    assert skipped["r-missing"] == "directory not present"
    assert skipped["r-locked"] == "directory not readable"
    assert "not usable" in skipped["r-file"]


# --- delete: only listed entries -------------------------------------------

def test_delete_accepts_only_listed_entries(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    kept = downloads / "kept.bin"
    kept.write_bytes(b"keep")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    roots = [StorageRoot("plugin-downloads", "dl", downloads)]

    # an existing but unlisted path is refused — never an arbitrary path
    out = delete_entry(str(outside), roots=roots)
    assert out["ok"] is False
    assert "not an entry" in out["error"]
    # a traversal string never matches a listed path exactly
    out = delete_entry(f"{downloads}/../outside.bin", roots=roots)
    assert out["ok"] is False and "not an entry" in out["error"]
    # a crafted id is refused too
    out = delete_entry("plugin-downloads:../outside.bin", roots=roots)
    assert out["ok"] is False and "not an entry" in out["error"]
    assert outside.read_bytes() == b"outside"
    assert kept.read_bytes() == b"keep"


def test_delete_by_id_and_exact_path_reports_freed_bytes(tmp_path):
    downloads = tmp_path / "downloads"
    model_dir = downloads / "model-a"
    model_dir.mkdir(parents=True)
    (model_dir / "w1.safetensors").write_bytes(b"1" * 5)
    (model_dir / "w2.bin").write_bytes(b"2" * 3)
    single = downloads / "single.gguf"
    single.write_bytes(b"3" * 4)
    roots = [StorageRoot("plugin-downloads", "dl", downloads)]

    # a download-dir model folder is removed as a folder, bytes freed = 8
    out = delete_entry("plugin-downloads:model-a", roots=roots)
    assert out["ok"] is True
    assert out["freed_bytes"] == 8
    assert out["name"] == "model-a" and out["type"] == "dir"
    assert out["root"] == "plugin-downloads"
    assert not model_dir.exists()
    # a file entry is removed as a file; the exact listed path also works
    out = delete_entry(str(single), roots=roots)
    assert out["ok"] is True
    assert out["freed_bytes"] == 4
    assert out["type"] == "file"
    assert not single.exists()


# --- delete: safety rules ----------------------------------------------------

def test_delete_refuses_symlink_entries(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"do not touch")
    link = downloads / "weights.bin"
    link.symlink_to(outside)
    roots = [StorageRoot("plugin-downloads", "dl", downloads)]

    out = list_storage(roots=roots)
    entry = out["entries"][0]
    assert entry["type"] == "symlink"  # listed honestly, but ...
    for ref in (entry["id"], entry["path"]):
        got = delete_entry(ref, roots=roots)
        assert got["ok"] is False, ref
        assert "symlink" in got["error"], ref
    # ... nothing behind the link is removed
    assert outside.read_bytes() == b"do not touch"
    assert link.is_symlink()


def test_delete_reverifies_containment_for_forged_listing(tmp_path,
                                                          monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"keep")
    forged = {"ok": True, "entries": [{
        "id": "plugin-downloads:outside.bin", "name": "outside.bin",
        "path": str(outside), "root": "plugin-downloads",
        "root_label": "dl", "type": "file", "size_bytes": 4,
        "last_modified": "2020-01-01T00:00:00+00:00"}],
        "roots": [], "roots_skipped": [], "note": ""}
    monkeypatch.setattr(storage, "list_storage",
                        lambda roots=None, home=None: forged)

    out = delete_entry("plugin-downloads:outside.bin",
                       roots=[StorageRoot("plugin-downloads", "dl", root)])
    # containment is re-verified even though the ref matched the listing
    assert out["ok"] is False
    assert "strictly inside" in out["error"]
    assert outside.read_bytes() == b"keep"


def test_confine_rules(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "f.bin"
    inside.write_bytes(b"x")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"y")
    link = root / "link.bin"
    link.symlink_to(outside)

    assert storage._confine(str(inside), [root])[0] == inside.resolve()
    got, err = storage._confine(str(outside), [root])
    assert got is None and "strictly inside" in err
    got, err = storage._confine(str(link), [root])
    assert got is None and "symlink" in err
    got, err = storage._confine("relative/path", [root])
    assert got is None and "absolute" in err
    # the root itself is not deletable through an entry
    got, err = storage._confine(str(root), [root])
    assert got is None and "strictly inside" in err
    # a symlinked parent escapes on resolve and is refused
    escape = tmp_path / "escape"
    escape.mkdir()
    (escape / "f.bin").write_bytes(b"z")
    alias = root / "alias"
    alias.symlink_to(escape, target_is_directory=True)
    got, err = storage._confine(str(alias / "f.bin"), [root])
    assert got is None and "strictly inside" in err


# --- delete: human-approval gate (fail closed) ------------------------------

def test_delete_fail_closed_when_gate_denies_or_times_out(tmp_path, monkeypatch):
    """The approve directive holds the call at Hermes' human-approval gate;
    a gate that denies, times out, or errors blocks the call before the
    handler runs — the dispatch contract in hermes_cli.plugins
    ._resolve_block_from_details ("an approve whose gate errors, denies, or
    times out is blocked"). The simulation below mirrors that contract:
    policy_hook must escalate (otherwise the gate would never see the call),
    and only an approved answer reaches model_forge_storage — so a denial
    deletes nothing."""
    _stub_plugins(monkeypatch, tmp_path)
    _fake_home(monkeypatch, tmp_path)
    downloads = tmp_path / "model-forge" / "downloads"
    downloads.mkdir(parents=True)
    target = downloads / "weights.bin"
    target.write_bytes(b"123456")
    from model_forge import tools

    def _dispatch(gate_answer, args):
        directive = tools.policy_hook("model_forge_storage", args)
        if directive is not None and directive.get("action") == "approve":
            if gate_answer != "approved":
                return "blocked"
        return tools.model_forge_storage(args)

    ref = "plugin-downloads:weights.bin"
    for answer in ("denied", "timeout", "gate-error"):
        assert _dispatch(answer, {"action": "delete",
                                  "entry": ref}) == "blocked", answer
        assert target.exists(), answer
    out = json.loads(_dispatch("approved", {"action": "delete", "entry": ref}))
    assert out["ok"] is True and out["freed_bytes"] == 6
    assert not target.exists()


# --- handler wiring ---------------------------------------------------------

def test_handler_actions_surface_json(tmp_path, monkeypatch):
    _stub_plugins(monkeypatch, tmp_path)
    _fake_home(monkeypatch, tmp_path)
    downloads = tmp_path / "model-forge" / "downloads"
    downloads.mkdir(parents=True)
    (downloads / "a.bin").write_bytes(b"1234")
    from model_forge import tools

    # list is the default action; absent home roots are skipped fail-soft
    out = json.loads(tools.model_forge_storage({}))
    assert out["ok"] is True
    assert {e["name"] for e in out["entries"]} == {"a.bin"}
    out = json.loads(tools.model_forge_storage({"action": "bogus"}))
    assert "unknown action" in out["error"]
    out = json.loads(tools.model_forge_storage({"action": "delete"}))
    assert "entry required" in out["error"]
    out = json.loads(tools.model_forge_storage(
        {"action": "delete", "entry": "nope"}))
    assert out["ok"] is False and "not an entry" in out["error"]
    out = json.loads(tools.model_forge_storage(
        {"action": "delete", "entry": "plugin-downloads:a.bin"}))
    assert out["ok"] is True and out["freed_bytes"] == 4
    assert not (downloads / "a.bin").exists()
