"""Media wiring tests (P5b-B2): the ComfyUI workflow-runner seam.

Three seams are covered here, all with fakes (no network, no ports):

* ``model_forge.aliases.workflow_id_for`` — the pinned facade->runner id seam
  (legacy id -> canonical -> bare workflow_id, idempotent);
* ``model_forge.media.runner_plan`` — the pure local-lane planner, including
  pre-rename ids landing on the right lane and a video/image kind mismatch;
* ``model_forge.tools.model_forge_generate_media`` running image/video lanes
  through a fake ``ComfyWorkflowRunner`` inside the existing claim/release
  scope, returning the OpenAI-ish shapes callers consume.

Plus the model-forge-image / model-forge-video provider plugins against the
OpenAI facade shapes (legacy ids resolved before the request leaves).
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from model_forge import tools
from model_forge.aliases import workflow_id_for
from model_forge.broker import (
    EngineLifecycle, EvictionPlanner, Orchestrator, ResidencyLedger,
)
from model_forge.media import runner_plan
from model_forge.targets import ComputeTarget, get_target
from model_forge.workflows import load_templates

_REPO = Path(__file__).resolve().parent.parent
_IMAGE_PKG = _REPO / "model-forge-image" / "__init__.py"
_VIDEO_PKG = _REPO / "model-forge-video" / "__init__.py"

_LOCAL = ComputeTarget(id="local", label="This machine",
                       base_url="http://127.0.0.1:8080", kind="local")


# --- aliases.workflow_id_for ------------------------------------------------

def test_workflow_id_for_strips_gen_prefix():
    assert workflow_id_for("gen/qwen-image-2.1") == "qwen-image-2.1"
    assert workflow_id_for("gen/h3-t2v") == "h3-t2v"


def test_workflow_id_for_resolves_legacy_then_strips():
    # legacy gen id -> canonical -> bare runner lane
    assert workflow_id_for("gen/krea2") == "qwen-image-2.1"
    assert workflow_id_for("gen/ideogram-4") == "qwen-image-2.1"
    assert workflow_id_for("gen/krea2-img2img") == "qwen-image-2.1-edit"


def test_workflow_id_for_is_idempotent_and_bare_passthrough():
    for bare in ("qwen-image-2.1", "qwen-image-2.1-edit", "moss-image",
                 "h3-t2v", "h3-i2v", "h3-r2v"):
        assert workflow_id_for(bare) == bare
        assert workflow_id_for(workflow_id_for(bare)) == bare
        assert workflow_id_for(workflow_id_for(workflow_id_for(bare))) == bare


def test_workflow_id_for_only_unwraps_gen_namespace():
    # an unrelated prefix is not this seam's contract — left intact
    assert workflow_id_for("chat/whatever") == "chat/whatever"
    # a custom alias table is honoured (and still stripped one hop)
    assert workflow_id_for("old/x", {"old/x": "gen/moss-image"}) == "moss-image"


# --- media.runner_plan ------------------------------------------------------

def _lanes():
    return load_templates()


def test_runner_plan_no_runner_is_honest():
    plan = runner_plan("image", "gen/krea2", None)
    assert plan["ok"] is False and plan["action"] == "no_runner"


def test_runner_plan_resolves_legacy_id_to_lane():
    plan = runner_plan("image", "gen/krea2", _FakeRunner(_lanes()))
    assert plan["ok"] is True and plan["action"] == "run_local"
    assert plan["workflow_id"] == "qwen-image-2.1"
    assert plan["output_kind"] == "image"


def test_runner_plan_accepts_bare_and_facade_ids():
    runner = _FakeRunner(_lanes())
    assert runner_plan("video", "h3-t2v", runner)["workflow_id"] == "h3-t2v"
    assert runner_plan("image", "gen/moss-image",
                       runner)["workflow_id"] == "moss-image"


def test_runner_plan_kind_mismatch_is_refused():
    # gen/krea2 aliases to an image lane; a video request must be refused
    plan = runner_plan("video", "gen/krea2", _FakeRunner(_lanes()))
    assert plan["ok"] is False and plan["action"] == "kind_mismatch"
    assert plan["output_kind"] == "image"


def test_runner_plan_unknown_lane_is_refused():
    plan = runner_plan("image", "gen/does-not-exist", _FakeRunner(_lanes()))
    assert plan["ok"] is False and plan["action"] == "unknown_lane"
    assert plan["workflow_id"] == "does-not-exist"


def test_media_asset_url_falls_back_to_the_view_url():
    # B1: a connected/remote ComfyUI output lives on its host (not readable
    # here) -> the fetchable /view URL is used; never a dead local path.
    from model_forge import tools
    url = tools._media_asset_url("/output/remote-only.png",
                                 fallback="http://comfy:8188/view?filename=x")
    assert url == "http://comfy:8188/view?filename=x"
    # no fallback -> the honest raw path is kept (never invented)
    assert tools._media_asset_url("/output/remote-only.png") == \
        "/output/remote-only.png"


def test_local_workflow_runner_wires_a_connected_comfyui(monkeypatch):
    # B1: a connected/remote ComfyUI endpoint gets a runner, not just a local
    # one — so a connected ComfyUI serves generation, not only downloads.
    from model_forge import tools
    from model_forge.engines import comfyui as comfy_mod

    class _St:
        available = True
        id = "comfyui"
        endpoint = "http://comfy:8188"

    remote = ComputeTarget(id="comfy-box", label="Comfy", kind="lan",
                           base_url="http://comfy:8188")
    monkeypatch.setattr(comfy_mod, "detect", lambda base_url="": _St())
    runner = tools._local_workflow_runner(remote)
    assert runner is not None
    assert runner.base == "http://comfy:8188"


def test_local_workflow_runner_none_when_not_comfyui(monkeypatch):
    # a remote target that is NOT a ComfyUI keeps its gateway path (no runner)
    from model_forge import tools
    from model_forge.engines import comfyui as comfy_mod

    class _St:
        available = False
        id = "other"
        endpoint = ""

    remote = ComputeTarget(id="gw", label="GW", kind="lan",
                           base_url="http://gw:8000")
    monkeypatch.setattr(comfy_mod, "detect", lambda base_url="": _St())
    assert tools._local_workflow_runner(remote) is None


# --- fakes for the generate_media local flow --------------------------------

class _RecordingAdapter(EngineLifecycle):
    def __init__(self, engine_id, calls):
        self.engine_id = engine_id
        self.calls = calls

    def ensure_loaded(self, model_id, **kwargs):
        self.calls.append(("ensure_loaded", self.engine_id, model_id))
        return {"ok": True, "reason": "warmed"}

    def release(self, model_id):
        self.calls.append(("release", self.engine_id, model_id))
        return {"ok": True, "reason": "evicted"}

    def probe(self):
        return {"ok": True, "state": "observed", "resident": []}


class _FakeRunner:
    """Stand-in for ComfyWorkflowRunner: a real lane registry + a scripted
    run() so the whole claim/submit/release flow runs in-process."""

    def __init__(self, workflows, *, ok=True, files=None, reason=""):
        self.workflows = workflows
        self.result = {"ok": ok, "output_kind": "", "files": files or [],
                       "reason": reason, "meta": {}}
        self.calls = []

    def run(self, workflow_id, params, *, assets_dir, job_id):
        self.calls.append({"workflow_id": workflow_id, "params": params,
                           "assets_dir": assets_dir, "job_id": job_id})
        return dict(self.result)


def _stack(calls, budget=48.0):
    bookings = tools.bookings_from_registry()
    ledger = ResidencyLedger(bookings=bookings)
    ledger.set_budget(_LOCAL.id, budget)
    adapters = {engine: _RecordingAdapter(engine, calls)
                for engine in ("ollama", "comfyui", "media-gateway")}
    planner = EvictionPlanner(ledger)
    orchestrator = Orchestrator(ledger, planner, adapters)
    return {"target_id": _LOCAL.id, "ledger": ledger, "planner": planner,
            "orchestrator": orchestrator, "bookings": bookings,
            "budget_gb": budget}


def _env(monkeypatch, tmp_path, runner):
    """Local target, isolated scope state, fake runner, residency stack."""
    plugins = types.ModuleType("plugins")
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_data_dir = lambda name: tmp_path / name
    plugins.plugin_storage = storage
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)
    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.load_config_readonly = lambda: {}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg_mod)

    monkeypatch.setattr(tools, "get_target", lambda tid: _LOCAL)
    monkeypatch.setattr(tools, "_local_workflow_runner",
                        lambda target, **kw: runner)
    calls: list = []
    stack = _stack(calls)
    monkeypatch.setattr(tools, "broker_stack", lambda target, **kw: stack)
    monkeypatch.setattr(tools, "_ACTIVE_SCOPES", {})
    monkeypatch.setattr(tools, "_MEDIA_SCOPE_JOBS", {})
    monkeypatch.setattr(tools, "_COMFY_MEDIA_JOBS", {})
    return calls, stack


def _png(path: Path):
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    return path


def test_generate_media_local_image_shape_and_scope(tmp_path, monkeypatch):
    img = _png(tmp_path / "out.png")
    runner = _FakeRunner(load_templates(), ok=True, files=[str(img)])
    calls, _ = _env(monkeypatch, tmp_path, runner)

    out = json.loads(tools.model_forge_generate_media(
        {"modality": "image", "target": "local", "model": "gen/krea2",
         "prompt": "a cat"}))

    assert out["ok"] is True and out["action"] == "done"
    assert out["output_kind"] == "image"
    # the pre-rename id landed on the right lane
    assert runner.calls[0]["workflow_id"] == "qwen-image-2.1"
    # OpenAI images shape: one output, a data: URL of the rendered file
    assert len(out["data"]) == 1
    url = out["data"][0]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == img.read_bytes()
    # the lane scope was claimed (load) then released (release) around the job
    assert calls == [("ensure_loaded", "media-gateway", "gen/krea2"),
                     ("release", "media-gateway", "gen/krea2")]
    assert out["lane_claim"]["state"] == "released"


def test_generate_media_local_image_releases_on_failure(tmp_path, monkeypatch):
    runner = _FakeRunner(load_templates(), ok=False, reason="comfyui boom")
    calls, _ = _env(monkeypatch, tmp_path, runner)

    out = json.loads(tools.model_forge_generate_media(
        {"modality": "image", "target": "local", "model": "gen/krea2",
         "prompt": "x"}))

    assert out["ok"] is False and out["action"] == "submit_failed"
    assert "comfyui boom" in out["error"]
    assert out["lane_claim"]["state"] == "released"
    assert calls == [("ensure_loaded", "media-gateway", "gen/krea2"),
                     ("release", "media-gateway", "gen/krea2")]


def test_generate_media_local_kind_mismatch_refused(tmp_path, monkeypatch):
    runner = _FakeRunner(load_templates())
    calls, _ = _env(monkeypatch, tmp_path, runner)

    out = json.loads(tools.model_forge_generate_media(
        {"modality": "video", "target": "local", "model": "gen/krea2",
         "prompt": "x"}))

    assert out["ok"] is False and out["action"] == "kind_mismatch"
    assert runner.calls == []          # nothing submitted
    assert calls == []                 # no lane claimed


def _wait_terminal(job_id, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with tools._COMFY_JOBS_LOCK:
            rec = tools._COMFY_MEDIA_JOBS.get(job_id)
            if rec is not None and rec["status"] in ("completed", "failed"):
                return
        time.sleep(0.01)


def test_generate_media_local_video_job_poll_contract(tmp_path, monkeypatch):
    clip = str(tmp_path / "clip.mp4")
    Path(clip).write_bytes(b"\x00\x00\x00\x18ftypmp42")
    runner = _FakeRunner(load_templates(), ok=True, files=[clip])
    calls, _ = _env(monkeypatch, tmp_path, runner)

    # submit -> OpenAI video-job shape {id, status}
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "video", "target": "local", "model": "gen/h3-t2v",
         "prompt": "a bird"}))
    assert out["ok"] is True and out["action"] == "submitted"
    assert out["output_kind"] == "video"
    jid = out["submitted"]["id"]
    assert jid.startswith("vid-") and out["submitted"]["status"] == "queued"
    assert out["lane_claim"]["state"] == "held"

    _wait_terminal(jid)
    assert runner.calls[0]["workflow_id"] == "h3-t2v"

    # poll -> {status: completed, data: [{url}]} and the scope releases
    polled = json.loads(tools.model_forge_generate_media(
        {"modality": "video", "target": "local", "job_id": jid}))
    assert polled["ok"] is True
    assert polled["job"]["status"] == "completed"
    assert polled["job"]["data"] == [{"url": clip}]
    assert polled["lane_claim"]["state"] == "released"
    assert jid not in tools._COMFY_MEDIA_JOBS
    assert calls == [("ensure_loaded", "media-gateway", "gen/h3-t2v"),
                     ("release", "media-gateway", "gen/h3-t2v")]


def test_generate_media_local_video_unknown_job(tmp_path, monkeypatch):
    runner = _FakeRunner(load_templates())
    _env(monkeypatch, tmp_path, runner)
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "video", "target": "local", "job_id": "nope-123"}))
    assert out["ok"] is False and out["action"] == "unknown_job"


def test_generate_media_remote_video_is_facade_hint(tmp_path, monkeypatch):
    # a remote target has no local runner: video is reported as a facade lane
    monkeypatch.setattr(tools, "get_target", lambda tid: ComputeTarget(
        id="gpu-box", label="GPU Box", kind="lan",
        base_url="http://gpu-box.example.com:8080"))
    out = json.loads(tools.model_forge_generate_media(
        {"modality": "video", "target": "gpu-box", "model": "gen/h3-t2v",
         "prompt": "x"}))
    assert out["ok"] is False and out["action"] == "no_local_runner"
    assert "/v1/videos" in out["detail"]


def test_local_workflow_runner_none_for_remote_target():
    remote = ComputeTarget(id="r", label="R", kind="lan",
                           base_url="http://r:8080")
    assert tools._local_workflow_runner(remote) is None


# --- provider plugins: OpenAI facade shapes + alias resolution --------------

_DEFAULT_ASPECT = "landscape"
_VALID_ASPECTS = {"landscape", "portrait", "square"}


def _error_response(*, error, error_type="provider_error", provider="",
                    model="", prompt="", aspect_ratio=_DEFAULT_ASPECT):
    return {"success": False, "image": None, "error": error,
            "error_type": error_type, "model": model, "prompt": prompt,
            "aspect_ratio": aspect_ratio, "provider": provider}


def _success_response(*, image, model, prompt, aspect_ratio, provider,
                      modality="text", extra=None):
    return {"success": True, "image": image, "model": model, "prompt": prompt,
            "aspect_ratio": aspect_ratio, "modality": modality,
            "provider": provider}


def _resolve_aspect(value):
    v = value.strip().lower() if isinstance(value, str) else ""
    return v if v in _VALID_ASPECTS else _DEFAULT_ASPECT


class _VideoBase:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return {"success": True, "video": "saved.mp4",
                "model": kwargs.get("model") or "base-model",
                "prompt": prompt, "provider": "model-forge"}


def _install_agent_fakes(monkeypatch):
    secret = types.ModuleType("agent.secret_scope")
    secret.get_secret_str = lambda name, default="": ""

    img = types.ModuleType("agent.image_gen_provider")
    img.DEFAULT_ASPECT_RATIO = _DEFAULT_ASPECT
    img.resolve_aspect_ratio = _resolve_aspect
    img.error_response = _error_response
    img.success_response = _success_response
    img.save_b64_image = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("save_b64_image not stubbed"))

    class _ImageProvider:
        pass

    img.ImageGenProvider = _ImageProvider

    vid = types.ModuleType("agent.video_gen_provider")
    vid.OpenAICompatibleVideoGenProvider = _VideoBase
    vid.error_response = _error_response

    agent_pkg = types.ModuleType("agent")
    agent_pkg.image_gen_provider = img
    agent_pkg.video_gen_provider = vid
    agent_pkg.secret_scope = secret

    fs = types.ModuleType("agent.file_safety")
    fs.raise_if_read_blocked = lambda path: None
    agent_pkg.file_safety = fs

    openai_fake = types.ModuleType("openai")
    openai_fake.OpenAI = object
    httpx_fake = types.ModuleType("httpx")
    httpx_fake.get = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("httpx.get not stubbed"))

    for name, mod in (("agent", agent_pkg),
                      ("agent.secret_scope", secret),
                      ("agent.image_gen_provider", img),
                      ("agent.video_gen_provider", vid),
                      ("agent.file_safety", fs),
                      ("openai", openai_fake),
                      ("httpx", httpx_fake)):
        monkeypatch.setitem(sys.modules, name, mod)


def _load(path, name, monkeypatch):
    _install_agent_fakes(monkeypatch)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cfg_images(base_url="http://facade:8080/v1", default_model=""):
    return types.SimpleNamespace(load_config_readonly=lambda: {
        "plugins": {"entries": {"model-forge-image": {
            "settings": {"base_url": base_url, "default_model": default_model}}}}})


def _cfg_videos(base_url="http://facade:8080/v1", default_model=""):
    return types.SimpleNamespace(load_config_readonly=lambda: {
        "plugins": {"entries": {"model-forge-video": {
            "settings": {"base_url": base_url, "default_model": default_model}}}}})


def test_image_provider_resolves_legacy_id_and_sync_shape(monkeypatch):
    pkg = _load(_IMAGE_PKG, "wiring_provider_image", monkeypatch)
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        _cfg_images(default_model="gen/krea2"))

    seen = {}

    class _Datum:
        url = "http://facade:8080/assets/j1/out.png"
        b64_json = None

    class _Resp:
        data = [_Datum()]

    class _Images:
        def generate(self, **kw):
            seen.update(kw)
            return _Resp()

    class _Client:
        def __init__(self, **kw):
            self.images = _Images()
            self.kwargs = kw

        def close(self):
            pass

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Client)

    # default model (gen/krea2) resolves to the canonical image lane
    out = pkg.ModelForgeImageGenProvider().generate("a robot")
    assert out["success"] is True and out["image"] == _Datum.url
    assert seen["model"] == "gen/qwen-image-2.1"
    assert seen["prompt"] == "a robot"

    # an explicit legacy id resolves too
    pkg.ModelForgeImageGenProvider().generate("x", model="gen/ideogram-4")
    assert seen["model"] == "gen/qwen-image-2.1"


def test_image_provider_live_catalog_fallback_resolves_alias(monkeypatch):
    pkg = _load(_IMAGE_PKG, "wiring_provider_image_catalog", monkeypatch)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", _cfg_images())
    # no explicit model and no settings default -> the live catalog's first row
    monkeypatch.setattr(
        pkg, "_live_models",
        lambda: [{"id": "gen/krea2", "output_kind": "image"}])
    # the catalog id is resolved through the same alias table the explicit and
    # settings-default branches use (gen/krea2 -> gen/qwen-image-2.1)
    assert pkg._resolve_model(None) == "gen/qwen-image-2.1"


def test_video_provider_resolves_legacy_id(monkeypatch):
    pkg = _load(_VIDEO_PKG, "wiring_provider_video", monkeypatch)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", _cfg_videos())

    provider = pkg.ModelForgeVideoGenProvider()
    base = provider  # OpenAICompatibleVideoGenProvider == _VideoBase instance methods
    out = provider.generate("a clip", model="gen/krea2")
    # the provider delegated to the base with the resolved model id
    assert isinstance(base, _VideoBase)
    assert out["success"] is True
    assert base.calls[-1]["kwargs"]["model"] == "gen/qwen-image-2.1"


def test_video_provider_identity_for_bare_model(monkeypatch):
    pkg = _load(_VIDEO_PKG, "wiring_provider_video_bare", monkeypatch)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", _cfg_videos())
    p = pkg.ModelForgeVideoGenProvider()
    p.generate("clip", model="gen/h3-i2v")
    assert p.calls[-1]["kwargs"]["model"] == "gen/h3-i2v"
