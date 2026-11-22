"""Surface-parity tests (Phase C): the /model-forge command surface, the
dashboard extension, and the capability-matrix documentation.

Everything runs against fakes — the command's payload seams return canned
tool-shaped JSON (no probes, no network), the dashboard manifest is
shape-checked as JSON, and the bundle and backend API module are checked as
text and AST. Stdlib + pytest only: nothing here imports FastAPI, the
desktop SDK, or the live Hermes config.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
try:
    from _repo import repo_root as _repo_root
    REPO = _repo_root()
except Exception:  # noqa: BLE001 — keep the literal path as the fallback
    pass
PLUGIN_DIR = REPO / "model-forge"
DESKTOP_PLUGIN_PATH = REPO / "model-forge-desktop" / "desktop" / "plugin.js"
DASH_DIR = PLUGIN_DIR / "dashboard"
MANIFEST_PATH = DASH_DIR / "manifest.json"
BUNDLE_PATH = DASH_DIR / "dist" / "index.js"
API_PATH = DASH_DIR / "plugin_api.py"
MATRIX_PATH = REPO / "docs" / "surface-parity.md"

_ALL_VERBS = ("cockpit", "models", "storage", "resources", "status", "browse",
              "broker", "claim", "release", "swap", "help")


def _load_plugin_module():
    """Import model-forge/__init__.py as a package module (the directory name
    is not importable), the same way the plugin loader and the wiring tests
    do. Loaded once per session; payloads are faked per test."""
    name = "model_forge_surfaces_pkg"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    init_path = PLUGIN_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name, init_path, submodule_search_locations=[str(PLUGIN_DIR)])
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def plugin():
    return _load_plugin_module()


class _Ctx:
    """Minimal PluginContext stand-in (the documented register_* surface)."""

    def __init__(self):
        self.tools = {}
        self.hooks = []
        self.commands = {}

    def register_tool(self, name, toolset=None, schema=None, handler=None):
        self.tools[name] = (toolset, schema, handler)

    def register_hook(self, name, handler):
        self.hooks.append((name, handler))

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = (handler, description, args_hint)


# --- canned tool-shaped payloads (the exact output shapes of tools.py) -------

def _status_json() -> str:
    return json.dumps({
        "targets": [{
            "id": "local", "label": "This machine", "kind": "local",
            "base_url": "http://127.0.0.1:8080",
            "hardware": {
                "platform": "linux",
                "gpus": [{"name": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0}],
                "total_vram_gb": 24.0, "ram_gb": 64.0,
            },
            "engines": [{
                "id": "llamacpp", "name": "llama.cpp (llama-server)",
                "available": True, "endpoint": "http://127.0.0.1:8080",
                "detail": "serving qwen", "modalities": ["chat"],
                "models_loaded": ["qwen"], "source": "auto-detected",
            }],
        }],
        "declared_engines": [{
            "label": "Workstation vLLM",
            "base_url": "http://gpu-box.example.com:8000",
            "source": "user-declared",
        }],
        "storage": {
            "ok": True,
            "entries": [{
                "id": "downloads:qwen", "name": "qwen", "path": "/x/qwen",
                "root": "downloads", "root_label": "Plugin downloads",
                "type": "dir", "size_bytes": 5_000_000_000,
                "last_modified": "2026-01-01T00:00:00+00:00",
            }],
            "roots": [{"root": "downloads", "label": "Plugin downloads",
                       "path": "/x"}],
            "roots_skipped": [],
            "note": "size_bytes is on-disk size; last_modified is the "
                    "filesystem modification time.",
        },
        "pull_credentials": {"HF_TOKEN": False, "MODELSCOPE_API_TOKEN": True},
        "invitation": {
            "reason": "No GPU was detected on this machine and no compute "
                      "targets are registered.",
            "next_steps": ["Add a cloud GPU or another machine as a compute "
                           "target with model_forge_target_add.",
                           "Or declare an engine endpoint in Settings."],
        },
    })


def _browse_json() -> str:
    return json.dumps({
        "target": "local", "hardware_budget_gb": 80.0,
        "models": [
            {
                "id": "flux-schnell", "name": "Flux Image",
                "modality": "image", "modality_label": "Image generation",
                "engines": ["comfyui"],
                "source": "black-forest-labs/FLUX.1-schnell",
                "engine_models": {}, "size_gb": 23.0, "vram_gb_min": 16.0,
                "fit": {"verdict": "fits_gpu", "engine_available": True,
                        "engine": "comfyui", "ram_gb_min": 0, "advisory": ""},
                "status": "validated", "note": "",
            },
            {
                "id": "hunyuan-video", "name": "Hunyuan Video",
                "modality": "video",
                "modality_label": "Video generation (t2v / i2v)",
                "engines": ["comfyui"], "source": "tencent/HunyuanVideo",
                "engine_models": {}, "size_gb": 50.0, "vram_gb_min": 40.0,
                "fit": {"verdict": "engine_unavailable",
                        "engine_available": False, "engine": "",
                        "ram_gb_min": 0,
                        "advisory": "no engine that can serve this model "
                                    "was detected on the target (needs: "
                                    "comfyui)"},
                "status": "known", "note": "text-to-video",
            },
        ],
        "note": "Fit verdicts are advisory and cross-check the engines "
                "detected on the target.",
        "live_discovery": "ok",
        "fresh": [
            {"id": "org/fresh-new", "last_modified": "2026-03-01",
             "downloads": 1200, "pipeline_tag": "image-classification",
             "url": "https://huggingface.co/org/fresh-new",
             "status": "fresh"},
            {"id": "org/fresh-old", "last_modified": "2026-02-01",
             "downloads": 5, "pipeline_tag": "",
             "url": "https://huggingface.co/org/fresh-old",
             "status": "fresh"},
        ],
        "fresh_note": "Fresh = live HuggingFace results, newest first "
                      "(unvetted: no fit estimate, no engine mapping). "
                      "Curated entries above are hand-maintained and "
                      "fit-priced.",
    })


def _broker_json() -> str:
    return json.dumps({
        "ok": True, "action": "status", "target": "local",
        "pulled": [{"model_id": "qwen", "path": "/x/qwen",
                    "size_bytes": 5_000_000_000}],
        "resident": [{"model_id": "qwen", "engine": "llamacpp",
                      "lane": "chat"}],
        "booked": [{"model_id": "qwen", "vram_gb": 16.0, "source": "bench",
                    "label": "measured peak"}],
        "unbooked": [{"model_id": "hunyuan-video", "modality": "video",
                      "vram_gb": None,
                      "label": "no booked ceiling (no VRAM floor declared)",
                      "hint": "run bench after first load"}],
        "budget": {"nominal_gb": 24.0, "booked_now_gb": 16.0,
                   "headroom_gb": 8.0},
        "unbooked_residents": [],
        "swap_stats": {"count": 2, "last_total_seconds": 12.5,
                       "average_total_seconds": 11.0},
        "active_scopes": [{"scope_id": "scope-1", "model_id": "qwen",
                           "modality": "chat", "kind": "claim"}],
        "note": "Read-only report. Booked ceilings carry provenance labels.",
    })


def _resources_json() -> str:
    return json.dumps({
        "ok": True, "action": "resources",
        "targets": [{
            "target": "local", "label": "This machine", "kind": "local",
            "memory": [
                {"kind": "vram", "label": "VRAM", "status": "ok", "detail": "",
                 "total_gb": 12.0, "used_gb": 6.2, "booked_now_gb": 6.0,
                 "free_gb": 5.8, "used_pct": 51.7, "booked_now_pct": 50.0,
                 "free_pct": 48.3},
                {"kind": "ram", "label": "RAM", "status": "unavailable",
                 "detail": "live ram usage probe failed: OSError: no sysconf",
                 "total_gb": 64.0, "used_gb": None, "booked_now_gb": 6.0,
                 "free_gb": None, "used_pct": None, "booked_now_pct": 9.4,
                 "free_pct": None},
            ],
            "residents": [{"model_id": "chat-a", "size_gb": 6.0,
                           "size_label": "published footprint (advisory)",
                           "size_bytes": 5_000_000_000, "engine": "ollama",
                           "lane": "chat"}],
            "queued_loads": [{"scope_id": "shot-1",
                              "model_id": "krea-2-turbo",
                              "state": "loading"}],
            "swap_stats": {"count": 2, "last_total_seconds": 12.5,
                           "average_total_seconds": 11.0},
        }],
        "note": ("On-demand read-only snapshot — shown only when asked for; "
                 "nothing here polls or updates on its own."),
    })


# --- command registration shape ----------------------------------------------

def test_command_registration_shape(plugin):
    ctx = _Ctx()
    plugin.register(ctx)
    # the eight read-only tools and the policy hook are untouched
    assert set(ctx.tools) == {
        "model_forge_status", "model_forge_browse", "model_forge_target_add",
        "model_forge_install", "model_forge_activate",
        "model_forge_generate_media", "model_forge_storage",
        "model_forge_broker",
    }
    assert all(toolset == "model_forge"
               for toolset, _schema, _handler in ctx.tools.values())
    assert all(callable(handler)
               for _toolset, _schema, handler in ctx.tools.values())
    assert ctx.hooks == [("pre_tool_call", plugin.model_forge.tools.policy_hook)]
    # /model-forge is registered on every surface commands work on, as one
    # entry point exposed under the tools module's historical name
    cmd = ctx.commands.get("model-forge")
    assert cmd is not None
    handler, description, args_hint = cmd
    assert handler is plugin.cockpit_command
    assert handler is plugin.model_forge.tools.cockpit_command
    assert "read-only" in description
    for verb in _ALL_VERBS:
        assert verb in args_hint


def test_legacy_json_subcommands_delegate_unchanged(plugin, monkeypatch):
    # cockpit|models|storage (and the default) are the Desktop pane's JSON
    # data door: they must delegate to the historical implementation
    # byte-identically, argument string included.
    calls = []

    def fake(raw):
        calls.append(raw)
        return '{"delegated": true}'

    monkeypatch.setattr(plugin, "_json_cockpit_command", fake)
    for args in ("cockpit", "models image", "storage", "models", ""):
        assert plugin.cockpit_command(args) == '{"delegated": true}'
        assert calls[-1] == args
    # help is extended for chat and never delegated
    calls.clear()
    out = plugin.cockpit_command("help")
    assert calls == []
    assert "browse" in out and "broker" in out


# --- plain-text chat rows (the handlers, run with fakes) ----------------------

def test_status_renders_plain_text_rows(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_status_payload", _status_json)
    out = plugin.cockpit_command("status")
    assert out.startswith("Model Forge status (read-only)")
    assert not out.lstrip().startswith("{")  # plain-text rows, not JSON
    assert "local [local] http://127.0.0.1:8080" in out
    assert "hardware: linux · GPUs: NVIDIA GeForce RTX 4090 (24.0 GB)" in out
    assert "VRAM 24.0 GB · RAM 64.0 GB" in out
    assert "engine llamacpp (llama.cpp (llama-server)): available=true" in out
    assert "models_loaded: qwen" in out
    assert "source=auto-detected" in out
    assert ("Workstation vLLM — http://gpu-box.example.com:8000 · "
            "source=user-declared") in out
    assert "storage: 1 entries · 5.0 GB · roots scanned 1 · skipped 0" in out
    assert "HF token: not set" in out
    assert "ModelScope API token: set (value never shown)" in out
    # never KEY=true/false: that shape collides with env-assignment redaction
    # and can make an unset token display as if set-and-hidden
    assert "TOKEN=" not in out
    assert "pointing invitation: No GPU was detected" in out
    assert "1. Add a cloud GPU" in out


def test_status_credentials_use_words_when_both_unset(plugin, monkeypatch):
    payload = json.loads(_status_json())
    payload["pull_credentials"] = {"HF_TOKEN": False,
                                   "MODELSCOPE_API_TOKEN": False}
    monkeypatch.setattr(plugin, "_status_payload", lambda: json.dumps(payload))
    out = plugin.cockpit_command("status")
    assert ("pull credentials: HF token: not set, "
            "ModelScope API token: not set") in out
    assert "TOKEN=" not in out
    # nothing renders as set-and-hidden when nothing is set
    assert "value never shown" not in out


def test_status_error_json_renders_error_row(plugin, monkeypatch):
    monkeypatch.setattr(
        plugin, "_status_payload",
        lambda: json.dumps({"error": "unknown target: 'nope'"}))
    assert plugin.cockpit_command("status") == "error: unknown target: 'nope'"


def test_status_storage_says_no_model_artifacts_when_empty(plugin, monkeypatch):
    # a root holding only cache bookkeeping says 'no model artifacts' rather
    # than listing it as model weights
    payload = json.loads(_status_json())
    payload["storage"]["entries"] = []
    payload["storage"]["roots"] = [{
        "root": "huggingface-hub", "label": "Hugging Face hub cache",
        "path": "/x/hub", "note": "no model artifacts"}]
    monkeypatch.setattr(plugin, "_status_payload", lambda: json.dumps(payload))
    out = plugin.cockpit_command("status")
    assert "storage: no model artifacts" in out
    assert "0 entries" not in out


def test_browse_renders_fit_verdict_rows(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_browse_payload", lambda modality: _browse_json())
    out = plugin.cockpit_command("browse image")
    assert "Model Forge browse — target=local · budget=80.0 GB" in out
    assert "fit verdicts are advisory" in out
    assert ("Flux Image [Image generation] · fit=fits_gpu · ~23.0 GB · "
            "engines: comfyui · status=validated") in out
    assert "fit=engine_unavailable" in out
    assert ("no engine that can serve this model was detected on the "
            "target (needs: comfyui)") in out
    assert "note: text-to-video" in out
    assert "Fit verdicts are advisory and cross-check the engines" in out
    # the live freshness block renders: compact newest-first rows plus the
    # honest 'fresh = live results, unvetted' note — never silently dropped
    assert "fresh (2) · live discovery: ok" in out
    assert "fresh = live results, unvetted" in out
    assert ("org/fresh-new · 2026-03-01 · 1200 downloads · "
            "image-classification") in out
    assert out.index("org/fresh-new") < out.index("org/fresh-old")
    assert "fresh note: Fresh = live HuggingFace results" in out


def test_browse_fresh_block_states_why_when_rows_are_empty(plugin, monkeypatch):
    # fresh=false / offline: zero rows with the honest reason attached
    payload = json.loads(_browse_json())
    payload["live_discovery"] = "disabled"
    payload["fresh"] = []
    payload["fresh_note"] = ("Live discovery skipped (fresh=false); "
                             "curated catalog only.")
    monkeypatch.setattr(plugin, "_browse_payload",
                        lambda modality: json.dumps(payload))
    out = plugin.cockpit_command("browse")
    assert "fresh (0) · live discovery: disabled" in out
    assert "Live discovery skipped (fresh=false); curated catalog only." in out


def test_browse_says_so_when_fresh_block_is_absent(plugin, monkeypatch):
    # a payload carrying no live-discovery block is never silently dropped
    bare = json.loads(_browse_json())
    del bare["live_discovery"], bare["fresh"], bare["fresh_note"]
    monkeypatch.setattr(plugin, "_browse_payload",
                        lambda modality: json.dumps(bare))
    out = plugin.cockpit_command("browse")
    assert "fresh: omitted" in out
    assert "no live-discovery block" in out


def test_browse_modality_reaches_payload_and_caps_rows(plugin, monkeypatch):
    seen = []

    def fake(modality):
        seen.append(modality)
        return _browse_json()

    monkeypatch.setattr(plugin, "_browse_payload", fake)
    plugin.cockpit_command("browse music")
    assert seen == ["music"]
    # rows beyond the chat row cap leave an honest remainder row (short rows
    # here so the character cap cannot cut the remainder line first)
    short = {
        "target": "local", "hardware_budget_gb": 80.0,
        "models": [{
            "id": f"m{i}", "name": f"M{i}", "modality": "chat",
            "modality_label": "Chat", "engines": ["llamacpp"],
            "source": "x/m", "engine_models": {}, "size_gb": 1.0,
            "vram_gb_min": 1.0,
            "fit": {"verdict": "fits_gpu", "engine_available": True,
                    "engine": "llamacpp", "ram_gb_min": 0, "advisory": ""},
            "status": "known", "note": "",
        } for i in range(40)],
        "note": "",
    }
    monkeypatch.setattr(plugin, "_browse_payload", lambda modality: json.dumps(short))
    out = plugin.cockpit_command("browse")
    assert "models (40):" in out
    assert "… and 15 more rows" in out
    assert "model_forge_browse for the full JSON" in out


def test_broker_renders_residency_rows_and_guidance(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_broker_payload", _broker_json)
    out = plugin.cockpit_command("broker")
    assert "Broker status — target=local (read-only)" in out
    assert "pulled (1): qwen (5.0 GB)" in out
    assert "resident (1): qwen (engine=llamacpp, lane=chat)" in out
    assert "qwen = 16.0 GB · measured peak · source=bench" in out
    # 'booked ceilings' = catalog nominal ceilings; the budget line's
    # 'booked_now' = live claims — one meaning per word
    assert "booked ceilings (1):" in out
    assert "\nbooked (1):\n" not in out  # bare old header, line-anchored
    assert "unbooked (1):" in out
    assert ("hunyuan-video — no booked ceiling (no VRAM floor declared) · "
            "run bench after first load") in out
    assert ("budget: nominal=24.0 GB · booked_now=16.0 GB · "
            "headroom=8.0 GB") in out
    assert "active scopes (1): scope-1 qwen (chat, claim)" in out
    # the claim/release guidance rides along, in the tool's state vocabulary
    assert "action=claim model_id=<id>" in out
    assert "action=release scope_id=" in out
    assert "queued -> loading -> ready" in out
    assert "this command never mutates" in out


def test_broker_view_dispatch(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_broker_payload", _broker_json)
    assert "Broker status" in plugin.cockpit_command("broker status")
    assert "claim guidance" in plugin.cockpit_command("broker claim").lower()
    assert "release guidance" in plugin.cockpit_command("broker release").lower()
    bad = plugin.cockpit_command("broker warp")
    assert "unknown broker view: 'warp'" in bad


# --- claim / release guidance is read-only output ----------------------------

def test_claim_and_release_are_guidance_only(plugin, monkeypatch):
    def explode():
        raise AssertionError("claim/release guidance must not read data")

    monkeypatch.setattr(plugin, "_broker_payload", explode)
    monkeypatch.setattr(plugin, "_status_payload", explode)
    monkeypatch.setattr(plugin, "_browse_payload", explode)
    claim = plugin.cockpit_command("claim")
    release = plugin.cockpit_command("release")
    for out in (claim, release):
        assert "model_forge_broker action=" in out
        assert "read-only" in out
    assert "action=claim model_id=<id>" in claim
    assert "scope_id=" in claim
    assert "action=plan" in claim  # dry-run guidance
    assert "action=release scope_id=" in release
    # the residency state vocabulary is named honestly in the guidance
    assert "queued -> loading -> ready" in claim
    assert "queued -> loading -> ready" in release


def test_swap_guidance_is_read_only_and_names_measured_timings(plugin,
                                                               monkeypatch):
    def explode():
        raise AssertionError("swap guidance must not read data")

    monkeypatch.setattr(plugin, "_broker_payload", explode)
    for args in ("swap", "broker swap"):
        out = plugin.cockpit_command(args)
        assert "swap guidance" in out.lower()
        assert "nothing is swapped" in out.lower()
        assert "action=swap model_id=" in out
        # measured, never estimated — and the phase names ride along
        assert "every one measured" in out and "never estimated" in out
        for phase in ("PREFLIGHT", "EVICT", "LOAD", "READY"):
            assert phase in out
        # the pre-flight guarantee is stated in the guidance itself
        assert "Pre-flight guarantee" in out
        assert "nothing is evicted before the preflight passes" in out
        assert "nothing was changed" in out
        # non-destructive, same class as claim/release — stated verbatim
        assert "not approval-gated" in out
        assert "same class as claim/release" in out


def test_broker_claim_release_and_help_guidance_name_swap(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_broker_payload", _broker_json)
    broker = plugin.cockpit_command("broker")
    claim = plugin.cockpit_command("claim")
    release = plugin.cockpit_command("release")
    help_text = plugin.cockpit_command("help")
    for out in (broker, claim, release):
        # swap is named as the one-call exchange verb with measured timings
        assert "action=swap" in out
        assert "measured" in out
    assert "nothing is changed" in claim or "changes nothing" in claim
    # the broker swap view is the same read-only swap guidance
    assert plugin.cockpit_command("broker swap") == \
        plugin.cockpit_command("swap")
    # the broker rows carry the measured swap stats too
    assert "swap stats (measured): count=2" in broker
    assert "last=12.5s" in broker and "average=11.0s" in broker
    # help lists the verb and names the non-gated class honestly
    assert "swap" in help_text
    assert "Broker claim/release/swap are tool verbs too" in help_text


def test_approval_gate_inventory_is_stated(plugin):
    out = plugin.cockpit_command("claim")
    assert "human-approval gate" in out
    assert "fail closed on denial or timeout" in out
    for verb in ("model_forge_target_add", "model_forge_activate",
                 "model_forge_storage", "model_forge_install"):
        assert verb in out
    assert "action=delete" in out


# --- fail-soft and timeout behavior -----------------------------------------

def test_command_times_out_cleanly(plugin, monkeypatch):
    def slow():
        time.sleep(0.5)
        return _status_json()

    monkeypatch.setattr(plugin, "_status_payload", slow)
    monkeypatch.setattr(plugin, "COMMAND_TIMEOUT_S", 0.05)
    start = time.monotonic()
    out = plugin.cockpit_command("status")
    elapsed = time.monotonic() - start
    assert elapsed < 2.0  # unattended commands never hang
    assert out.startswith("error:")
    assert "timed out after 0.05 s" in out
    assert "nothing was changed" in out


def test_command_never_raises_on_payload_failure(plugin, monkeypatch):
    def failing_probe():
        raise RuntimeError("engine probe exploded")

    monkeypatch.setattr(plugin, "_status_payload", failing_probe)
    monkeypatch.setattr(plugin, "_browse_payload", lambda modality: "not json{")
    monkeypatch.setattr(plugin, "_broker_payload",
                        lambda: json.dumps(["not", "an", "object"]))
    status = plugin.cockpit_command("status")
    assert status.startswith("error:")
    assert "RuntimeError: engine probe exploded" in status
    assert "nothing was changed" in status
    assert "not valid JSON" in plugin.cockpit_command("browse image")
    assert "not a JSON object" in plugin.cockpit_command("broker")


# --- dashboard manifest and bundle shape ------------------------------------

def test_dashboard_manifest_shape():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for field in ("name", "label", "tab", "entry"):
        assert field in manifest, f"manifest missing required field {field!r}"
    assert manifest["name"] == "model-forge"
    assert re.fullmatch(r"[a-z0-9-]+", manifest["name"])
    assert manifest["label"].strip()
    assert manifest.get("description", "").strip()
    assert manifest.get("icon", "").strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest.get("version", ""))
    tab = manifest["tab"]
    assert tab["path"].startswith("/")
    position = tab.get("position", "end")
    assert position == "end" or re.fullmatch(r"(after|before):[a-z0-9_-]+",
                                             position)
    assert isinstance(tab.get("hidden", False), bool)
    # entry (and api, when declared) resolve inside the dashboard folder
    assert (DASH_DIR / manifest["entry"]).is_file()
    if manifest.get("api"):
        assert (DASH_DIR / manifest["api"]).is_file()


def test_dashboard_bundle_registers_manifest_name():
    manifest = json.loads(MANIFEST_PATH.read_text())
    source = BUNDLE_PATH.read_text()
    assert "window.__HERMES_PLUGIN_SDK__" in source
    assert "window.__HERMES_PLUGINS__" in source
    # register() is called with exactly the manifest name
    assert f'register("{manifest["name"]}"' in source
    # the documented plugin API route is the bundle's only data door
    assert f"/api/plugins/{manifest['name']}/cockpit" in source
    # honest states: loading, failure reason, read-only delete note
    assert "Loading the Model Forge cockpit" in source
    assert "could not load" in source
    assert "never deletes files" in source


def test_dashboard_bundle_syntax():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed on this host")
    assert node is not None
    result = subprocess.run([node, "--check", str(BUNDLE_PATH)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dashboard_api_module_shape():
    source = API_PATH.read_text()
    tree = ast.parse(source)  # syntax validity without importing FastAPI
    assigned = {
        target.id
        for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    assert "router" in assigned  # the export name the dashboard host loads
    assert "router.get(" in source
    assert "cockpit_data" in source  # the one shared cockpit data path
    # bounded and fail-soft: a hard timeout and a stated reason on failure
    assert "asyncio" in source and "timeout" in source
    assert "nothing was changed" in source


# --- capability matrix documentation -----------------------------------------

def test_capability_matrix_names_every_surface():
    doc = MATRIX_PATH.read_text()
    lower = doc.lower()
    for surface in ("desktop", "telegram", "cli", "tui", "dashboard",
                    "mobile"):
        assert surface in lower, f"capability matrix omits surface {surface!r}"
    # every capability the matrix must cover is a named row
    for capability in ("model browsing", "fit verdicts", "model selection",
                       "install", "claim", "release", "swap", "storage",
                       "settings"):
        assert capability in lower
    # the honest gaps are named, not hidden
    assert "rides the approval-gate verb" in doc
    assert "untested in ci" in lower
    assert "on-device" in lower
    assert "future lane" in lower
    # the shared state vocabulary (W5.1) is documented
    for token in ("fits_gpu", "engine_unavailable", "queued", "loading",
                  "ready", "unbooked"):
        assert token in doc


def test_readme_surface_parity_sections():
    for path in (REPO / "README.md", PLUGIN_DIR / "README.md"):
        text = path.read_text()
        assert "## Surface parity" in text
        assert "docs/surface-parity.md" in text
        for surface in ("Desktop", "Telegram", "CLI / TUI", "dashboard"):
            assert surface in text
        assert "/model-forge" in text


def test_surface_parity_broker_server_start_rows_document_the_gate():
    # The approval table documents that a claim / release / swap / media lane
    # that would start or re-start a process-engine server rides the gate, and
    # that a call which starts nothing needs none — verified against the
    # handlers in tests/test_tools.py.
    doc = MATRIX_PATH.read_text()
    table = doc[doc.index("## Approval coverage"):]

    def row(marker):
        return next(line for line in table.splitlines() if marker in line)

    claim = row("`model_forge_broker` `claim`")
    release = row("`model_forge_broker` `release`")
    swap = row("`model_forge_broker` `swap`")
    media = row("`model_forge_generate_media`")
    for line in (claim, release, swap, media):
        assert "model-forge:broker_server_start" in line
        assert "starts a server process" in line or "re-start" in line
    assert "already-resident model starts nothing" in claim
    assert "restores nothing" in release and "already-resident" in release
    assert "dry_run=true" in swap
    # no row still claims an unconditional "No gate" reason for a server start
    assert "No gate — documented reason" not in claim
    assert "No gate — documented reason" not in release
    assert "No gate — documented reason" not in swap


def test_readme_names_hiccup_free_swaps():
    for path in (REPO / "README.md", PLUGIN_DIR / "README.md"):
        text = path.read_text()
        assert "Hiccup-free swaps" in text
        assert "action=swap" in text
        assert "pre-flight guarantee" in text
        assert "measured timings, never estimates" in text
        assert "nothing was changed" in text
        # the command surface docs list the verb
        assert "`broker`, `claim`, `release`,\n`swap`, and `help`" in text


def test_readme_disclosure_names_every_network_surface():
    # Disclosure completeness: every network/config surface the plugin gained
    # (dashboard API route, /model-forge command reads, user-declared endpoint
    # probing, the approved config + custom_providers write) is named in the
    # Disclosure section of both READMEs.
    for path in (REPO / "README.md", PLUGIN_DIR / "README.md"):
        text = path.read_text()
        start = text.index("## Disclosure")
        section = text[start:text.index("## Requirements")]
        assert "/api/plugins/model-forge/cockpit" in section
        assert "engines_json" in section
        assert "/model-forge" in section
        assert "custom_providers" in section
        assert "human-approval gate" in section
        assert "**Telemetry:** none" in section


# --- the on-demand resource view (quiet by design) ----------------------------

def test_storage_resources_renders_quiet_text_bars(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_resources_payload", _resources_json)
    out = plugin.cockpit_command("storage resources")
    assert not out.lstrip().startswith("{")  # plain-text rows, not JSON
    assert out.startswith("Model Forge resource view (on-demand, read-only)")
    # one quiet bar line per memory kind: used/total GB and the percentage
    # (6.2/12.0 -> 51.7% -> '52%'; ceil(0.517 * 14) = 8 filled cells)
    assert "vram  [████████░░░░░░] 6.2/12.0 GB (52%)" in out
    assert "· booked_now 6.0 GB (50%)" in out
    assert "· free 5.8 GB" in out
    # an unobservable kind says 'unavailable' with its reason, never a bar
    assert "ram   unavailable — live ram usage probe failed: OSError: no sysconf" in out
    assert "64.0 GB (" not in out  # no percentage bar for an unavailable row
    # then the resident model blocks
    assert "residents (1):" in out
    assert ("chat-a · size 6.0 GB (published footprint (advisory)) · "
            "engine=ollama · lane=chat · on disk 5.0 GB") in out
    # queued loads (only when any) and swap stats (only where present)
    assert "queued loads (1):" in out
    assert "shot-1 · krea-2-turbo · loading" in out
    assert ("swap stats (measured): count=2 · last=12.5s · average=11.0s"
            in out)
    # the quiet note: on-demand, nothing polls
    assert "shown only when asked" in out
    assert "nothing here" in out and "polls" in out


def test_storage_resources_render_says_nothing_when_lean(plugin, monkeypatch):
    # no queued loads / no measured swaps -> those lines stay away entirely
    payload = json.loads(_resources_json())
    del payload["targets"][0]["swap_stats"]
    payload["targets"][0]["queued_loads"] = []
    monkeypatch.setattr(plugin, "_resources_payload",
                        lambda: json.dumps(payload))
    out = plugin.cockpit_command("storage resources")
    assert "queued loads" not in out
    assert "swap stats" not in out
    assert "vram  [" in out


def test_resource_view_reads_run_only_when_typed(plugin, monkeypatch):
    # non-intrusion at the command layer: no other subcommand touches the
    # resource snapshot — it runs only when the user asks for it
    def explode():
        raise AssertionError("the resource snapshot must not run unasked")

    monkeypatch.setattr(plugin, "_resources_payload", explode)
    monkeypatch.setattr(plugin, "_status_payload", _status_json)
    monkeypatch.setattr(plugin, "_browse_payload", lambda modality: _browse_json())
    monkeypatch.setattr(plugin, "_broker_payload", _broker_json)
    monkeypatch.setattr(plugin, "_json_cockpit_command",
                        lambda raw: '{"delegated": true}')
    for args in ("", "cockpit", "models image", "storage", "status",
                 "browse", "broker", "claim", "release", "swap", "help"):
        plugin.cockpit_command(args)
    # the JSON data door delegates to the historical implementation ...
    seen = []
    monkeypatch.setattr(plugin, "_json_cockpit_command",
                        lambda raw: seen.append(raw) or '{"delegated": true}')
    assert plugin.cockpit_command("resources") == '{"delegated": true}'
    assert seen == ["resources"]
    # ... while the typed chat subcommand renders text from its own read
    monkeypatch.setattr(plugin, "_resources_payload", _resources_json)
    assert plugin.cockpit_command("storage resources").startswith(
        "Model Forge resource view")


def test_pane_resource_view_is_quiet_collapsed_and_manual(plugin):
    src = DESKTOP_PLUGIN_PATH.read_text()
    # collapsed by default; the toggle is the only door to the data
    assert "Resource view (on-demand)" in src
    assert "useState(false)" in src
    assert 'if (!open && state.phase === "idle") load();' in src
    # one snapshot through the existing read-only command bridge ...
    assert "slash.exec" in src
    assert "/model-forge resources" in src
    # ... and a small manual refresh control (the only other load trigger)
    assert "Refresh (manual" in src
    # quiet by design: no timers, no polling, no animation machinery at all
    for token in ("setInterval", "setTimeout", "requestAnimationFrame",
                  "WebGL", "canvas"):
        assert token not in src, token
    # honest failure state instead of silent defaults
    assert "Resource snapshot unavailable" in src
    assert "Nothing was changed" in src


def test_dashboard_resource_view_is_quiet_collapsed_and_manual():
    src = BUNDLE_PATH.read_text()
    assert "Resource view (on-demand)" in src
    assert "useState(false)" in src
    assert '/api/plugins/model-forge/resources' in src
    assert "Refresh (manual" in src
    for token in ("setInterval", "setTimeout", "requestAnimationFrame"):
        assert token not in src, token
    # the backend route is read-only, bounded, and only serves the view
    api = API_PATH.read_text()
    assert '@router.get("/resources")' in api
    assert "cockpit_resources" in api
    assert "nothing polls this route" in api


def test_desktop_plugin_bundle_syntax():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed on this host")
    assert node is not None
    result = subprocess.run([node, "--check", str(DESKTOP_PLUGIN_PATH)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_docs_state_resource_view_design_intent():
    # both READMEs carry the owner's design intent in the owner's terms
    for path in (REPO / "README.md", PLUGIN_DIR / "README.md"):
        text = path.read_text()
        assert "### Resource view (on-demand)" in text
        assert "belongs with weight management" in text
        assert "never a persistent display" in text
        assert "/model-forge storage resources" in text
        assert "no timers and no auto-refresh" in text
        assert "stays lean" in text
    # the capability matrix names the resource rows for command/pane/dashboard
    doc = MATRIX_PATH.read_text()
    row = next(line for line in doc.splitlines()
               if "**Storage resources**" in line)
    assert "/model-forge storage resources" in row
    assert "Resource view" in row
    assert "manual refresh" in row
    assert "no polling" in row
    # the shared vocabulary names the disambiguated quantities
    assert "booked_now" in doc
    assert "live claims" in doc
