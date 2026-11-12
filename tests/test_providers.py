"""Provider tests: model-forge-image and model-forge-video.

These two are Hermes plugin packages with hyphenated directory names, so they
cannot be imported normally. We load each ``__init__.py`` by file path with
``importlib``. The providers import Hermes ``agent.*`` modules at import time
(``agent.image_gen_provider``, ``agent.video_gen_provider``,
``agent.secret_scope``); those are not in the test venv, so we install minimal
fakes matching the real contracts (the response builders, aspect-ratio
resolver, and the video base class) into ``sys.modules`` before loading. That
isolates the provider's own logic — the code under test — from the Hermes
runtime.

Network is never touched: the image provider's ``openai`` client and the
``httpx`` model-catalog fetch are monkeypatched.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_IMAGE_PKG = _REPO / "model-forge-image" / "__init__.py"
_VIDEO_PKG = _REPO / "model-forge-video" / "__init__.py"


# --- fakes for the agent.* contracts the providers import -------------------

_DEFAULT_ASPECT = "landscape"
_VALID_ASPECTS = {"landscape", "portrait", "square"}


def _error_response(*, error, error_type="provider_error", provider="",
                    model="", prompt="", aspect_ratio=_DEFAULT_ASPECT):
    return {"success": False, "image": None, "error": error,
            "error_type": error_type, "model": model, "prompt": prompt,
            "aspect_ratio": aspect_ratio, "provider": provider}


def _success_response(*, image, model, prompt, aspect_ratio, provider,
                      modality="text", extra=None):
    payload = {"success": True, "image": image, "model": model,
               "prompt": prompt, "aspect_ratio": aspect_ratio,
               "modality": modality, "provider": provider}
    for k, v in (extra or {}).items():
        payload.setdefault(k, v)
    return payload


def _resolve_aspect(value):
    v = value.strip().lower() if isinstance(value, str) else ""
    return v if v in _VALID_ASPECTS else _DEFAULT_ASPECT


class _VideoBase:
    """Stand-in for OpenAICompatibleVideoGenProvider — records the call so a
    test can assert the provider delegated correctly after its own logic."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return _success_video(prompt=prompt,
                              model=kwargs.get("model") or "base-model")


def _success_video(*, video="", model="", prompt="", provider=""):
    return {"success": True, "video": video, "model": model,
            "prompt": prompt, "provider": provider}


def _install_agent_fakes(monkeypatch):
    secret = types.ModuleType("agent.secret_scope")
    secret.get_secret_str = lambda name, default="": ""

    img = types.ModuleType("agent.image_gen_provider")
    img.DEFAULT_ASPECT_RATIO = _DEFAULT_ASPECT
    img.resolve_aspect_ratio = _resolve_aspect
    img.error_response = _error_response
    img.success_response = _success_response

    def _save_b64(b64_data, **kw):
        raise RuntimeError("save_b64_image not stubbed in this test")

    img.save_b64_image = _save_b64

    class _ImageProvider:  # base for ModelForgeImageGenProvider
        def name(self):
            raise NotImplementedError

    img.ImageGenProvider = _ImageProvider

    vid = types.ModuleType("agent.video_gen_provider")
    vid.OpenAICompatibleVideoGenProvider = _VideoBase
    vid.error_response = _error_response
    vid.success_response = _success_video

    agent_pkg = types.ModuleType("agent")
    agent_pkg.image_gen_provider = img
    agent_pkg.video_gen_provider = vid
    agent_pkg.secret_scope = secret

    # the read-block chokepoint: permissive by default, monkeypatch to raise
    fs_mod = types.ModuleType("agent.file_safety")
    fs_mod.raise_if_read_blocked = lambda path: None
    agent_pkg.file_safety = fs_mod

    # the providers lazily `import openai` / `import httpx` inside functions —
    # provide module fakes so the import resolves (tests then monkeypatch the
    # specific attributes they exercise).
    openai_fake = types.ModuleType("openai")
    openai_fake.OpenAI = object  # replaced per-test via monkeypatch

    httpx_fake = types.ModuleType("httpx")
    httpx_fake.get = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("httpx.get not stubbed in this test"))

    monkeypatch.setitem(sys.modules, "agent", agent_pkg)
    monkeypatch.setitem(sys.modules, "agent.secret_scope", secret)
    monkeypatch.setitem(sys.modules, "agent.image_gen_provider", img)
    monkeypatch.setitem(sys.modules, "agent.video_gen_provider", vid)
    monkeypatch.setitem(sys.modules, "agent.file_safety", fs_mod)
    monkeypatch.setitem(sys.modules, "openai", openai_fake)
    monkeypatch.setitem(sys.modules, "httpx", httpx_fake)

    # the providers also lazily read the Hermes config for settings/env
    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.load_config_readonly = lambda: {}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg_mod)


def _load(path: Path, name: str, monkeypatch):
    _install_agent_fakes(monkeypatch)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def image_pkg(monkeypatch):
    return _load(_IMAGE_PKG, "provider_image", monkeypatch)


@pytest.fixture
def video_pkg(monkeypatch):
    return _load(_VIDEO_PKG, "provider_video", monkeypatch)


# --- image provider: pure/module-level --------------------------------------

def test_image_base_url_empty_when_unconfigured(image_pkg):
    assert image_pkg._base_url() == ""
    assert image_pkg._default_model() == ""


def test_image_live_models_empty_when_unconfigured(image_pkg):
    assert image_pkg._live_models() == []


def test_image_base_url_from_settings(image_pkg, monkeypatch):
    cfg = {"plugins": {"entries": {"model-forge-image": {
        "settings": {"base_url": "http://192.168.1.62:8080/v1/"}}}}}
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(
                            load_config_readonly=lambda: cfg))
    # settings win and are rstripped of trailing slash
    assert image_pkg._base_url() == "http://192.168.1.62:8080/v1"
    assert image_pkg._default_model() == ""


def test_image_live_models_prefers_image_kind(image_pkg, monkeypatch):
    import httpx

    rows = [
        {"id": "chat-1", "output_kind": "chat"},
        {"id": "img-1", "output_kind": "image"},
        {"id": "img-2", "output_kind": "image"},
    ]

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": rows}

    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1"}}}}}))
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _Resp())
    got = image_pkg._live_models()
    assert [m["id"] for m in got] == ["img-1", "img-2"]


def test_image_live_models_falls_back_to_all(image_pkg, monkeypatch):
    import httpx

    rows = [{"id": "chat-1"}, {"id": "vid-1"}]

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": rows}

    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1"}}}}}))
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _Resp())
    got = image_pkg._live_models()
    assert [m["id"] for m in got] == ["chat-1", "vid-1"]


def test_image_live_models_network_error(image_pkg, monkeypatch):
    import httpx

    class _Resp:
        def raise_for_status(self):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1"}}}}}))
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _Resp())
    assert image_pkg._live_models() == []


def test_image_as_data_url(image_pkg, tmp_path, monkeypatch):
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    got = image_pkg._as_data_url(str(p))
    assert got.startswith("data:image/png;base64,")
    assert base64.b64decode(got.split(",", 1)[1]) == p.read_bytes()
    # passthroughs: URLs and data: ride through untouched
    assert image_pkg._as_data_url("https://x/a.png") == "https://x/a.png"
    assert image_pkg._as_data_url("data:image/png;base64,AA") \
        == "data:image/png;base64,AA"
    # jpeg / webp mime by real magic
    j = tmp_path / "a.jpg"
    j.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 8)
    assert image_pkg._as_data_url(str(j)).startswith("data:image/jpeg;base64,")
    w = tmp_path / "a.webp"
    w.write_bytes(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4)
    assert image_pkg._as_data_url(str(w)).startswith("data:image/webp;base64,")


def test_image_as_data_url_rejects_raw_paths(image_pkg, tmp_path, monkeypatch):
    # the old code passed raw strings through — that let any file be named
    with pytest.raises(ValueError):
        image_pkg._as_data_url("/nope/a.png")
    # blocked reads propagate (core's read-block check)
    import agent.file_safety as fs

    def blocked(path):
        raise ValueError("read blocked: secrets area")
    monkeypatch.setattr(fs, "raise_if_read_blocked", blocked)
    with pytest.raises(ValueError, match="read blocked"):
        image_pkg._as_data_url(str(tmp_path / "auth.json"))
    # suffix and magic bytes are enforced
    monkeypatch.setattr(fs, "raise_if_read_blocked", lambda path: None)
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"#!/bin/sh\nrm -rf ~\n")
    with pytest.raises(ValueError, match="magic bytes"):
        image_pkg._as_data_url(str(fake))
    txt = tmp_path / "a.txt"
    txt.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="unsupported image type"):
        image_pkg._as_data_url(str(txt))


def test_image_generate_invalid_image_arg(image_pkg, monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1",
                                             "default_model": "img-1"}}}}}))
    out = image_pkg.ModelForgeImageGenProvider().generate(
        "make it wink", image_url="/etc/shadow")
    assert out["success"] is False
    assert out["error_type"] == "invalid_argument"


# --- image provider: class surface ------------------------------------------

def test_image_provider_identity(image_pkg):
    p = image_pkg.ModelForgeImageGenProvider()
    assert p.name == "model-forge"
    assert p.display_name == "Model Forge (local)"
    assert p.is_available() is True
    assert p.capabilities() == {"modalities": ["text", "image"],
                                "max_reference_images": 4}


def test_image_provider_setup_schema(image_pkg):
    p = image_pkg.ModelForgeImageGenProvider()
    s = p.get_setup_schema()
    assert s["badge"] == "local"
    assert s["env_vars"][0]["key"] == "MODEL_FORGE_IMAGE_BASE_URL"


def test_image_provider_list_models(image_pkg, monkeypatch):
    import httpx

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "img-1"}, {"id": ""}]}

    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1"}}}}}))
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _Resp())
    rows = image_pkg.ModelForgeImageGenProvider().list_models()
    assert len(rows) == 1  # empty id dropped
    assert rows[0]["id"] == "img-1"
    assert rows[0]["price"] == "free (local)"


# --- image provider: generate() ---------------------------------------------

def test_image_generate_requires_prompt(image_pkg):
    p = image_pkg.ModelForgeImageGenProvider()
    out = p.generate("")
    assert out["success"] is False
    assert out["error_type"] == "invalid_argument"


def test_image_generate_no_endpoint(image_pkg):
    p = image_pkg.ModelForgeImageGenProvider()
    out = p.generate("a robot")
    assert out["success"] is False
    assert out["error_type"] == "missing_credentials"


def test_image_generate_no_model(image_pkg, monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1"}}}}}))
    import httpx

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _Resp())
    out = image_pkg.ModelForgeImageGenProvider().generate("a robot")
    assert out["success"] is False
    assert out["error_type"] == "no_model"


def _fake_openai(monkeypatch, url="http://x:1/v1/img.png", b64=None):
    """A fake openai client whose images.generate returns a canned datum."""

    class _Datum:
        def __init__(self):
            self.url = url
            self.b64_json = b64

    class _Resp:
        data = [_Datum()]

    class _Images:
        def generate(self, **kw):
            self.last = kw
            return _Resp()

    class _Client:
        def __init__(self, **kw):
            self.images = _Images()
            self.kwargs = kw

        def close(self):
            self.closed = True

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Client)


def test_image_generate_success_url(image_pkg, monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1",
                                             "default_model": "img-1"}}}}}))
    _fake_openai(monkeypatch)
    out = image_pkg.ModelForgeImageGenProvider().generate("a robot")
    assert out["success"] is True
    assert out["image"] == "http://x:1/v1/img.png"
    assert out["model"] == "img-1"
    assert out["modality"] == "text"
    assert out["aspect_ratio"] == "landscape"


def test_image_generate_edit_modality(image_pkg, monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1",
                                             "default_model": "img-1"}}}}}))
    _fake_openai(monkeypatch)
    out = image_pkg.ModelForgeImageGenProvider().generate(
        "make it wink", image_url="https://x:1/a.png")
    assert out["success"] is True
    assert out["modality"] == "image"


def test_image_generate_b64_fallback(image_pkg, monkeypatch, tmp_path):
    from agent.image_gen_provider import save_b64_image

    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1",
                                             "default_model": "img-1"}}}}}))
    b64 = base64.b64encode(b"\x89PNG").decode()
    _fake_openai(monkeypatch, url="", b64=b64)
    saved = tmp_path / "saved.png"

    def _fake_save(b64_data, **kw):
        saved.write_bytes(base64.b64decode(b64_data))
        return saved

    monkeypatch.setattr("agent.image_gen_provider.save_b64_image", _fake_save)
    out = image_pkg.ModelForgeImageGenProvider().generate("a robot")
    assert out["success"] is True
    assert out["image"] == str(saved)


def test_image_generate_api_error(image_pkg, monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1",
                                             "default_model": "img-1"}}}}}))

    import openai

    class _Boom:
        def __init__(self, **kw):
            raise RuntimeError("endpoint down")

    monkeypatch.setattr(openai, "OpenAI", _Boom)
    out = image_pkg.ModelForgeImageGenProvider().generate("a robot")
    assert out["success"] is False
    assert out["error_type"] == "api_error"


def test_image_generate_no_image_returned(image_pkg, monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-image": {
                                "settings": {"base_url": "http://x:1/v1",
                                             "default_model": "img-1"}}}}}))

    class _Datum:
        url = ""
        b64_json = ""

    class _Resp:
        data = [_Datum()]

    class _Images:
        def generate(self, **kw):
            return _Resp()

    class _Client:
        def __init__(self, **kw):
            self.images = _Images()

        def close(self):
            pass

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Client)
    out = image_pkg.ModelForgeImageGenProvider().generate("a robot")
    assert out["success"] is False
    assert "neither a URL nor image data" in out["error"]


def test_image_register(image_pkg):
    captured = {}

    class _Ctx:
        def register_image_gen_provider(self, p):
            captured["p"] = p

    image_pkg.register(_Ctx())
    assert isinstance(captured["p"], image_pkg.ModelForgeImageGenProvider)


# --- video provider: pure/module-level --------------------------------------

def test_video_base_url_empty_when_unconfigured(video_pkg):
    assert video_pkg._base_url() == ""
    assert video_pkg._default_model() == ""


def test_video_live_models_prefers_video_kind(video_pkg, monkeypatch):
    import httpx

    rows = [
        {"id": "vid-1", "output_kind": "video"},
        {"id": "chat-1", "output_kind": "chat"},
    ]

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": rows}

    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-video": {
                                "settings": {"base_url": "http://x:1/v1"}}}}}))
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _Resp())
    assert [m["id"] for m in video_pkg._live_models()] == ["vid-1"]


def test_video_live_models_error(video_pkg, monkeypatch):
    import httpx

    class _Resp:
        def raise_for_status(self):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-video": {
                                "settings": {"base_url": "http://x:1/v1"}}}}}))
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _Resp())
    assert video_pkg._live_models() == []


# --- video provider: class surface ------------------------------------------

def test_video_provider_identity(video_pkg):
    p = video_pkg.ModelForgeVideoGenProvider()
    assert p.name == "model-forge"
    assert p.display_name == "Model Forge (local)"
    assert p.is_available() is True
    assert p._api_key() == "local-endpoint"


def test_video_provider_setup_schema(video_pkg):
    s = video_pkg.ModelForgeVideoGenProvider().get_setup_schema()
    assert s["badge"] == "local"
    assert s["env_vars"][0]["key"] == "MODEL_FORGE_VIDEO_BASE_URL"


def test_video_provider_list_models(video_pkg, monkeypatch):
    import httpx

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "vid-1"}, {"id": ""}]}

    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-video": {
                                "settings": {"base_url": "http://x:1/v1"}}}}}))
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _Resp())
    rows = video_pkg.ModelForgeVideoGenProvider().list_models()
    assert len(rows) == 1 and rows[0]["id"] == "vid-1"


# --- video provider: generate() ---------------------------------------------

def test_video_generate_no_endpoint(video_pkg):
    p = video_pkg.ModelForgeVideoGenProvider()
    out = p.generate("a robot")
    assert out["success"] is False
    assert out["error_type"] == "missing_credentials"


def test_video_generate_delegates_with_default_model(video_pkg, monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-video": {
                                "settings": {"base_url": "http://x:1/v1",
                                             "default_model": "vid-1"}}}}}))
    p = video_pkg.ModelForgeVideoGenProvider()
    out = p.generate("a robot")
    # resolved the default model, then delegated to the base class
    assert p.calls[-1]["kwargs"]["model"] == "vid-1"
    assert p.calls[-1]["prompt"] == "a robot"
    assert out["success"] is True


def test_video_generate_explicit_model_passthrough(video_pkg, monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli.config",
                        types.SimpleNamespace(load_config_readonly=lambda: {
                            "plugins": {"entries": {"model-forge-video": {
                                "settings": {"base_url": "http://x:1/v1"}}}}}))
    p = video_pkg.ModelForgeVideoGenProvider()
    p.generate("a robot", model="vid-9")
    assert p.calls[-1]["kwargs"]["model"] == "vid-9"


def test_video_register(video_pkg):
    captured = {}

    class _Ctx:
        def register_video_gen_provider(self, p):
            captured["p"] = p

    video_pkg.register(_Ctx())
    assert isinstance(captured["p"], video_pkg.ModelForgeVideoGenProvider)