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


def test_non_model_artifacts_are_never_listed(tmp_path):
    # .locks (HF hub cache) and CACHEDIR.TAG are cache bookkeeping, never
    # model weights: filtered at the inventory source so counts and pulled
    # lists contain only model artifacts
    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "CACHEDIR.TAG").write_bytes(b"x" * 191)
    (hub / ".locks").mkdir()
    model = hub / "models--org--m"
    model.mkdir()
    (model / "w.bin").write_bytes(b"123")
    only = tmp_path / "only-artifacts"
    only.mkdir()
    (only / "CACHEDIR.TAG").write_bytes(b"x")
    (only / ".locks").mkdir()

    out = list_storage(roots=[StorageRoot("huggingface-hub", "hub", hub),
                              StorageRoot("artifacts-only", "only", only)])
    # only the model artifact is listed and counted
    assert [e["name"] for e in out["entries"]] == ["models--org--m"]
    by_root = {r["root"]: r for r in out["roots"]}
    assert "note" not in by_root["huggingface-hub"]
    # a root holding ONLY such artifacts says so instead of listing them
    assert by_root["artifacts-only"]["note"] == "no model artifacts"
    # and they are not deletable through the tool either (delete accepts
    # only entries the inventory returned)
    got = delete_entry(str(hub / "CACHEDIR.TAG"),
                       roots=[StorageRoot("huggingface-hub", "hub", hub)])
    assert got["ok"] is False and "not an entry" in got["error"]


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


# --- storage list shape in the status/pane data path -------------------------

def test_status_and_cockpit_carry_storage_list_shape(tmp_path, monkeypatch):
    # The one storage data path serves model_forge_status, the /model-forge
    # cockpit command (the Desktop pane's data door), and model_forge_storage
    # — all in the same list shape (sizes and modification times).
    _stub_plugins(monkeypatch, tmp_path)
    _fake_home(monkeypatch, tmp_path)
    downloads = tmp_path / "model-forge" / "downloads"
    model_dir = downloads / "model-a"
    model_dir.mkdir(parents=True)
    (model_dir / "w.bin").write_bytes(b"1" * 5)

    from model_forge import tools
    from model_forge.registry import HardwareProfile
    monkeypatch.setattr(tools, "probe_hardware",
                        lambda: HardwareProfile(gpus=(), total_vram_gb=0.0,
                                                ram_gb=0.0, platform="linux"))
    monkeypatch.setattr(tools, "detect_engines", lambda base_url="", **kw: [])

    ref = list_storage()
    out = json.loads(tools.model_forge_status())
    assert out["storage"]["entries"] == ref["entries"]
    required = {"id", "name", "path", "root", "type", "size_bytes",
                "last_modified"}
    for entry in out["storage"]["entries"]:
        assert required <= set(entry)
        assert isinstance(entry["size_bytes"], int)
        datetime.fromisoformat(entry["last_modified"])  # parses
    assert {e["name"] for e in out["storage"]["entries"]} == {"model-a"}

    # the pane's cockpit command serves the same shape plus the honest
    # delete-verb note (the pane renders it read-only)
    cmd = json.loads(tools.cockpit_command("storage"))
    assert cmd["entries"] == ref["entries"]
    assert "model_forge_storage" in cmd["delete_note"]
    assert "action=delete" in cmd["delete_note"]
    # and the full cockpit snapshot carries the storage block too
    snap = json.loads(tools.cockpit_command("cockpit"))
    assert snap["storage"]["entries"] == ref["entries"]


# --- resource view (on-demand): model_forge_storage action=resources ----------

# Design intent: the resource snapshot runs ONLY when asked (never along with
# action=list), reports honest numbers from the hardware probe and the
# residency ledger, and never fabricates a value for a failed probe.

class _ProgressOrchestrator:
    """Fake orchestrator reporting fixed progress rows (queued-load rows)."""

    def __init__(self, rows=None):
        self._rows = dict(rows or {})

    def progress(self, key):
        return self._rows.get(key)


def _residency_ledger():
    """A real ledger: one booked resident with an on-disk pull record."""
    from model_forge.broker import ResidencyLedger
    ledger = ResidencyLedger()
    ledger.book("chat-a", {"vram_gb": 6.0, "source": "catalog-vram-floor",
                           "load_s": None})
    ledger.record_resident("local", "chat-a", engine="ollama", lane="chat")
    ledger.record_pull("local", "chat-a", "/x/chat-a", 5_000_000_000)
    return ledger


def _resource_seams(monkeypatch, *, hw_probe=None, stack=None,
                    vram_used=None, ram_used=None):
    """Point every live seam of the resource snapshot at fakes: one local
    compute target, a fake hardware probe (12 GB VRAM / 64 GB RAM), the
    residency stack, and fake usage probes (6.2 GB VRAM / 16 GB RAM used).
    Returns (tools, ledger, stack)."""
    from model_forge import tools
    from model_forge.registry import HardwareProfile
    from model_forge.targets import ComputeTarget
    target = ComputeTarget(id="local", label="This machine", kind="local",
                           base_url="http://127.0.0.1:8080")
    if hw_probe is None:
        def _fake_hw_probe(t):
            return HardwareProfile(
                gpus=({"name": "Fake RTX", "vram_gb": 12.0},),
                total_vram_gb=12.0, ram_gb=64.0, platform="linux")
        hw_probe = _fake_hw_probe
    ledger = _residency_ledger()
    if stack is None:
        stack = {"target_id": "local", "ledger": ledger,
                 "orchestrator": None, "planner": None,
                 "bookings": {}, "budget_gb": 12.0}
    monkeypatch.setattr(tools, "load_targets", lambda: [target])
    monkeypatch.setattr(tools, "get_target",
                        lambda tid=None: target
                        if tid in (None, "", "local") else None)
    monkeypatch.setattr(tools, "_hw_for_target", hw_probe)
    monkeypatch.setattr(tools, "broker_stack", lambda t, **kw: stack)
    monkeypatch.setattr(tools, "gpu_vram_used_gb",
                        vram_used or (lambda: 6.2))
    monkeypatch.setattr(tools, "_ram_used_gb", ram_used or (lambda: 16.0))
    monkeypatch.setattr(tools, "_ACTIVE_SCOPES", {})
    return tools, ledger, stack


def test_resources_action_row_shapes_and_bars_math(tmp_path, monkeypatch):
    tools, _ledger, _stack = _resource_seams(monkeypatch)
    out = json.loads(tools.model_forge_storage({"action": "resources"}))
    assert out["ok"] is True and out["action"] == "resources"
    assert "shown only when asked" in out["note"]
    (target,) = out["targets"]
    assert target["target"] == "local" and target["kind"] == "local"
    rows = {r["kind"]: r for r in target["memory"]}
    assert set(rows) == {"vram", "ram"}
    vram = rows["vram"]
    assert vram["label"] == "VRAM" and vram["status"] == "ok"
    assert vram["total_gb"] == 12.0
    assert vram["used_gb"] == 6.2
    # booked_now = the live claims: the booked ceiling of the one resident
    assert vram["booked_now_gb"] == 6.0
    assert vram["free_gb"] == 5.8
    # bars math: used/total percentages from the fakes
    assert vram["used_pct"] == 51.7
    assert vram["booked_now_pct"] == 50.0
    assert vram["free_pct"] == 48.3
    ram = rows["ram"]
    assert ram["label"] == "RAM" and ram["status"] == "ok"
    assert ram["total_gb"] == 64.0 and ram["used_gb"] == 16.0
    assert ram["free_gb"] == 48.0
    assert ram["used_pct"] == 25.0
    assert ram["booked_now_pct"] == 9.4
    assert ram["free_pct"] == 75.0
    # resident model blocks: model id, size, engine, lane
    (res,) = target["residents"]
    assert res == {"model_id": "chat-a", "size_gb": 6.0,
                   "size_label": "published footprint (advisory)",
                   "size_bytes": 5_000_000_000,
                   "engine": "ollama", "lane": "chat"}
    assert target["queued_loads"] == []
    # swap stats are present only where measured swaps exist
    assert "swap_stats" not in target
    # vocabulary: booked_now (live claims) never carries a booking label
    assert "published" not in str(vram["booked_now_gb"])


def test_resources_queued_loads_and_swap_stats_where_present(tmp_path,
                                                             monkeypatch):
    tools, ledger, stack = _resource_seams(monkeypatch)
    ledger.record_swap({"model_id": "krea-2-turbo", "evicted": ["chat-a"],
                        "total_seconds": 12.5, "ok": True})
    stack["orchestrator"] = _ProgressOrchestrator({
        "shot-1": {"state": "loading"},
        "shot-2": {"state": "ready"},
    })
    monkeypatch.setattr(tools, "_ACTIVE_SCOPES", {
        "shot-1": {"target_id": "local", "model_id": "krea-2-turbo",
                   "modality": "image", "kind": "workflow"},
        "shot-2": {"target_id": "local", "model_id": "chat-a",
                   "modality": "chat", "kind": "workflow"},
    })
    out = json.loads(tools.model_forge_storage({"action": "resources"}))
    target = out["targets"][0]
    # queued loads: in-flight scopes only — a ready scope is not a queued load
    assert target["queued_loads"] == [
        {"scope_id": "shot-1", "model_id": "krea-2-turbo", "state": "loading"}]
    assert target["swap_stats"] == {"count": 1, "last_total_seconds": 12.5,
                                    "average_total_seconds": 12.5}


def test_resources_probe_failure_is_an_honest_unavailable_row(tmp_path,
                                                              monkeypatch):
    def _boom(t):
        raise RuntimeError("nvidia-smi exploded")

    tools, _ledger, _stack = _resource_seams(monkeypatch, hw_probe=_boom)
    out = json.loads(tools.model_forge_storage({"action": "resources"}))
    target = out["targets"][0]
    assert len(target["memory"]) == 2
    for row in target["memory"]:
        assert row["status"] == "unavailable"
        assert ("hardware probe failed: RuntimeError: nvidia-smi exploded"
                in row["detail"])
        # never fake numbers: no total, no usage, no free, no percentages
        assert row["total_gb"] is None
        assert row["used_gb"] is None
        assert row["free_gb"] is None
        assert row["used_pct"] is None
        assert row["free_pct"] is None
        # the ledger's live claims stay honest and labeled
        assert row["booked_now_gb"] == 6.0


def test_resources_usage_probe_failure_keeps_honest_totals(tmp_path,
                                                           monkeypatch):
    def _boom():
        raise OSError("sysconf gone")

    tools, _ledger, _stack = _resource_seams(monkeypatch, ram_used=_boom)
    out = json.loads(tools.model_forge_storage({"action": "resources"}))
    rows = {r["kind"]: r for r in out["targets"][0]["memory"]}
    # VRAM (its probe answered) stays a full ok row
    assert rows["vram"]["status"] == "ok" and rows["vram"]["used_gb"] == 6.2
    ram = rows["ram"]
    assert ram["status"] == "unavailable"
    assert "live ram usage probe failed: OSError: sysconf gone" in ram["detail"]
    assert ram["total_gb"] == 64.0        # the honest probe total stays
    assert ram["booked_now_gb"] == 6.0
    assert ram["used_gb"] is None and ram["free_gb"] is None
    assert ram["used_pct"] is None and ram["free_pct"] is None


def test_resources_zero_total_is_an_unavailable_row_not_a_fake_zero(
        tmp_path, monkeypatch):
    from model_forge.registry import HardwareProfile

    def _no_gpu(t):
        return HardwareProfile(gpus=(), total_vram_gb=0.0, ram_gb=64.0,
                               platform="linux")

    tools, _ledger, _stack = _resource_seams(monkeypatch, hw_probe=_no_gpu)
    out = json.loads(tools.model_forge_storage({"action": "resources"}))
    rows = {r["kind"]: r for r in out["targets"][0]["memory"]}
    vram = rows["vram"]
    # a probe that reports no VRAM total -> an honest unavailable row
    assert vram["status"] == "unavailable"
    assert "no VRAM total was reported by the hardware probe" in vram["detail"]
    assert vram["total_gb"] is None and vram["used_gb"] is None
    # RAM on the same machine is unaffected
    assert rows["ram"]["status"] == "ok" and rows["ram"]["total_gb"] == 64.0


def test_resources_action_targets_one_compute_target(tmp_path, monkeypatch):
    tools, _ledger, _stack = _resource_seams(monkeypatch)
    out = json.loads(tools.model_forge_storage({"action": "resources",
                                                "target": "local"}))
    assert [t["target"] for t in out["targets"]] == ["local"]
    out = json.loads(tools.model_forge_storage({"action": "resources",
                                                "target": "nope"}))
    assert "unknown target" in out["error"]


def test_resources_action_is_read_only_and_list_stays_lean(tmp_path,
                                                           monkeypatch):
    _stub_plugins(monkeypatch, tmp_path)
    _fake_home(monkeypatch, tmp_path)
    downloads = tmp_path / "model-forge" / "downloads"
    downloads.mkdir(parents=True)
    (downloads / "a.bin").write_bytes(b"1234")
    from model_forge import tools
    # the resources action never rides the approval gate (read-only)
    assert tools.policy_hook("model_forge_storage",
                             {"action": "resources"}) is None
    assert tools.policy_hook("model_forge_storage", {}) is None  # list
    # non-intrusion at the data layer: action=list stays exactly the lean
    # inventory — no resource block, no heavier read runs
    out = json.loads(tools.model_forge_storage({"action": "list"}))
    assert set(out) == {"ok", "entries", "roots", "roots_skipped", "note"}
    raw = tools.model_forge_storage({})
    assert '"resources"' not in raw
    assert '"memory"' not in raw
    assert '"queued_loads"' not in raw
    assert '"swap_stats"' not in raw
    assert "resources_hint" not in out


def test_storage_output_has_one_resources_guidance_line(tmp_path, monkeypatch):
    _stub_plugins(monkeypatch, tmp_path)
    _fake_home(monkeypatch, tmp_path)
    downloads = tmp_path / "model-forge" / "downloads"
    downloads.mkdir(parents=True)
    (downloads / "a.bin").write_bytes(b"1234")
    from model_forge import tools
    ref = list_storage()
    cmd = json.loads(tools.cockpit_command("storage"))
    # the listing itself is unchanged; one guidance line names the
    # resources subcommand
    assert cmd["entries"] == ref["entries"]
    assert "storage resources" in cmd["resources_hint"]
    assert "on-demand" in cmd["resources_hint"]


def test_resources_json_data_door_returns_the_snapshot(tmp_path, monkeypatch):
    tools, _ledger, _stack = _resource_seams(monkeypatch)
    for args in ("resources", "storage resources"):
        out = json.loads(tools.cockpit_command(args))
        assert out["ok"] is True and out["action"] == "resources", args
        assert out["targets"][0]["memory"], args
