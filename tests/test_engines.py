"""Engine-adapter tests (llama.cpp/Ollama, Strata, Maya, ComfyUI, vLLM, SGLang,
openai-compatible, plus the base helpers and the detect_engines aggregator).

Every adapter probes local engines over stdlib urllib (no auth assumed), so a
single fake ``urllib.request.urlopen`` that dispatches on the requested path
covers the whole surface. Nothing here touches the network: every probe is
routed through the fake, and the "not running" branches are exercised by
having the fake raise (as a dead endpoint would).
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from model_forge.engines import (
    comfyui,
    llamacpp,
    maya,
    openai_compat,
    sglang,
    strata,
    vllm,
)
from model_forge.engines import detect_engines
from model_forge.engines.base import EngineStatus, root_url, route_exists


# --- base helpers -----------------------------------------------------------

def test_root_url_strips_v1_and_slash():
    assert root_url("http://192.168.1.62:8080/v1") == "http://192.168.1.62:8080"
    assert root_url("http://192.168.1.62:8080/") == "http://192.168.1.62:8080"
    assert root_url("") == ""
    assert root_url(None) == ""


class _Resp(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200):
        super().__init__(data)
        self.status = status

    def read(self, n=-1):
        return super().read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _install_urlopen(monkeypatch, routes: dict):
    """routes: {path_suffix: response} or {path_suffix: Exception}. A path not
    in routes raises URLError (a dead endpoint)."""

    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        for suffix, resp in routes.items():
            if url.endswith(suffix):
                if isinstance(resp, Exception):
                    raise resp
                if isinstance(resp, int):
                    raise urllib.error.HTTPError(url, resp, "e", {}, None)
                return _Resp(resp)
        raise urllib.error.URLError("no route in test")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return _fake


# --- base.route_exists ------------------------------------------------------

def test_route_exists_via_openapi(monkeypatch):
    spec = json.dumps({"paths": {"/v1/images/generations": {"post": {}}}})
    _install_urlopen(monkeypatch, {"/openapi.json": spec.encode()})
    assert route_exists("http://x:1", "/v1/images/generations") is True
    assert route_exists("http://x:1", "/v1/videos") is False


def test_route_exists_via_non404(monkeypatch):
    # no openapi; GET the path -> 405 proves a POST-only route
    _install_urlopen(monkeypatch, {
        "/openapi.json": 404,
        "/v1/videos": 405,
    })
    assert route_exists("http://x:1", "/v1/videos") is True


def test_route_exists_via_empty_post(monkeypatch):
    # no openapi; GET 404s; empty POST -> 422 proves the route
    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        method = getattr(req, "method", "GET")
        if url.endswith("/openapi.json"):
            raise urllib.error.HTTPError(url, 404, "e", {}, None)
        if method == "POST":
            raise urllib.error.HTTPError(url, 422, "e", {}, None)
        raise urllib.error.HTTPError(url, 404, "e", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    assert route_exists("http://x:1", "/v1/audio/music") is True


def test_route_exists_dead(monkeypatch):
    _install_urlopen(monkeypatch, {})
    assert route_exists("http://dead:1", "/v1/videos") is False


# --- detect_engines aggregator ----------------------------------------------

def test_detect_engines_drops_generic_when_named_present(monkeypatch):
    # local probe: strata (named) and openai-compatible both find /v1/models
    # at the default root -> the generic row for that endpoint is suppressed
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps(
            {"data": [{"id": "m1"}]}).encode(),
    })
    monkeypatch.setattr(strata.shutil, "which", lambda _: "/usr/bin/strata")
    out = detect_engines("")
    # strata reports available with the default root as its endpoint
    assert any(e.id == "strata" and e.available for e in out)
    # no generic openai-compatible row claims the SAME (default) endpoint
    for e in out:
        if e.id == "openai-compatible" and e.available:
            assert e.endpoint != strata.DEFAULT_ROOT


def test_detect_engines_catches_probe_errors(monkeypatch):
    # patch the bound detector the aggregator already captured in _DETECTORS
    import model_forge.engines as eng
    orig = eng._DETECTORS
    eng._DETECTORS = tuple(
        (lambda base_url="": (_ for _ in ()).throw(
            RuntimeError("adapter blew up"))) if d is llamacpp.detect else d
        for d in orig)
    try:
        out = detect_engines("http://x:1")
    finally:
        eng._DETECTORS = orig
    assert any(e.id == "unknown" and "adapter blew up" in e.detail
               for e in out)


# --- strata -----------------------------------------------------------------

def test_strata_launch_cmd():
    assert strata.launch_cmd("deepseek41", "ist-das") == [
        "strata", "launch", "--family", "deepseek41", "--model", "ist-das"]
    assert strata.launch_cmd("f", "m", plan_only=True)[-1] == "--plan"
    with pytest.raises(ValueError):
        strata.launch_cmd("", "m")


def test_strata_served_models(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"data": [{"id": "a"}, {"id": "b"}]}).encode()})
    assert strata.served_models("http://x:1") == ["a", "b"]


def test_strata_detect_remote_is_neutral(monkeypatch):
    monkeypatch.setattr(strata.shutil, "which", lambda _: None)
    st = strata.detect("http://x:1")
    assert st.available is False and st.id == "strata"


def test_strata_detect_local_serving(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"data": [{"id": "glm"}]}).encode()})
    monkeypatch.setattr(strata.shutil, "which", lambda _: "/usr/bin/strata")
    st = strata.detect()
    assert st.available is True
    assert st.models_loaded == ("glm",)
    assert "serving glm" in st.detail


def test_strata_detect_local_cli_only(monkeypatch):
    _install_urlopen(monkeypatch, {})  # no models
    monkeypatch.setattr(strata.shutil, "which", lambda _: "/usr/bin/strata")
    st = strata.detect()
    assert st.available is True and "CLI found" in st.detail


def test_strata_detect_not_installed(monkeypatch):
    _install_urlopen(monkeypatch, {})
    monkeypatch.setattr(strata.shutil, "which", lambda _: None)
    st = strata.detect()
    assert st.available is False and "not installed" in st.detail


# --- llamacpp / ollama ------------------------------------------------------

def test_llamacpp_detect_llama_server(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/props": json.dumps({"build_info": "x", "model_path": "m.gguf"}).encode(),
    })
    st = llamacpp.detect("http://x:1")
    assert st.id == "llamacpp" and st.available is True
    assert st.models_loaded == ("m.gguf",)


def test_llamacpp_detect_ollama(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/props": 404,
        "/api/tags": json.dumps(
            {"models": [{"name": "llama3"}, {"name": "mistral"}]}).encode(),
    })
    st = llamacpp.detect("http://x:1")
    assert st.id == "ollama" and st.available is True
    assert st.models_loaded == ("llama3", "mistral")


def test_llamacpp_detect_dead(monkeypatch):
    _install_urlopen(monkeypatch, {})
    st = llamacpp.detect("http://x:1")
    assert st.available is False and st.id == "llamacpp"


def test_llamacpp_detect_local_port_scan(monkeypatch):
    # 8080 dead, 8000 serves props -> found on 8000
    routes = {
        "8080/props": 404,
        "8080/api/tags": 404,
        "8081/props": 404,
        "8081/api/tags": 404,
        "8000/props": json.dumps({"model_path": "s.gguf"}).encode(),
    }
    _install_urlopen(monkeypatch, routes)
    st = llamacpp.detect()
    assert st.available is True and st.endpoint == "http://127.0.0.1:8000"


def test_llamacpp_detect_local_all_dead(monkeypatch):
    _install_urlopen(monkeypatch, {})
    st = llamacpp.detect()
    assert st.available is False and "8080/8081/8000/8001" in st.detail


# --- openai_compat ----------------------------------------------------------

def test_openai_compat_detect_with_models(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"data": [{"id": "m1"}]}).encode(),
        "/openapi.json": json.dumps(
            {"paths": {"/v1/images/generations": {}}}).encode(),
    })
    st = openai_compat.detect("http://x:1")
    assert st.id == "openai-compatible" and st.available is True
    assert "image" in st.modalities and "image_edit" in st.modalities


def test_openai_compat_detect_no_image_route(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"data": [{"id": "m1"}]}).encode(),
        "/openapi.json": json.dumps({"paths": {}}).encode(),
    })
    st = openai_compat.detect("http://x:1")
    assert st.available is True
    assert "image" not in st.modalities and "video" not in st.modalities


def test_openai_compat_detect_dead(monkeypatch):
    _install_urlopen(monkeypatch, {})
    st = openai_compat.detect("http://x:1")
    assert st.available is False


def test_openai_compat_detect_no_base(monkeypatch):
    st = openai_compat.detect("")
    assert st.available is False and "no base_url" in st.detail


# --- vllm -------------------------------------------------------------------

def test_vllm_detect_running(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/version": json.dumps({"version": "0.6.0"}).encode(),
        "/v1/models": json.dumps({"data": [{"id": "llama"}]}).encode(),
        "/openapi.json": json.dumps({"paths": {}}).encode(),
    })
    st = vllm.detect("http://x:1")
    assert st.available is True and "vllm 0.6.0" in st.detail
    assert "Omni" not in st.display_name
    assert st.models_loaded == ("llama",)


def test_vllm_detect_omni(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/version": json.dumps({"version": "0.6.0"}).encode(),
        "/v1/models": json.dumps({"data": [{"id": "q"}]}).encode(),
        "/openapi.json": json.dumps(
            {"paths": {"/v1/images/generations": {}}}).encode(),
    })
    st = vllm.detect("http://x:1")
    assert "Omni" in st.display_name
    assert "image" in st.modalities and "image_edit" in st.modalities


def test_vllm_detect_dead(monkeypatch):
    _install_urlopen(monkeypatch, {})
    st = vllm.detect("http://x:1")
    assert st.available is False


# --- sglang -----------------------------------------------------------------

def test_sglang_detect_llm(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/get_model_info": json.dumps({"model_path": "deepseek"}).encode(),
        "/openapi.json": json.dumps({"paths": {}}).encode(),
    })
    st = sglang.detect("http://x:1")
    assert st.available is True and "Diffusion" not in st.display_name
    assert st.models_loaded == ("deepseek",)
    assert "image" not in st.modalities


def test_sglang_detect_diffusion(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/get_model_info": json.dumps({"model_path": "flux"}).encode(),
        "/openapi.json": json.dumps(
            {"paths": {"/v1/images/generations": {}}}).encode(),
    })
    st = sglang.detect("http://x:1")
    assert "Diffusion" in st.display_name
    assert "image" in st.modalities and "image_edit" in st.modalities


def test_sglang_detect_dead(monkeypatch):
    _install_urlopen(monkeypatch, {})
    st = sglang.detect("http://x:1")
    assert st.available is False


# --- comfyui ----------------------------------------------------------------

def test_comfyui_detect_running(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/system_stats": json.dumps({
            "system": {"comfyui-version": "0.3.0"},
            "devices": [{"name": "RTX 3090"}, {"name": "RTX 4090"}],
        }).encode(),
    })
    st = comfyui.detect("http://x:1")
    assert st.available is True and "comfyui 0.3.0" in st.detail
    assert "RTX 3090" in st.detail


def test_comfyui_detect_dead(monkeypatch):
    _install_urlopen(monkeypatch, {})
    st = comfyui.detect("http://x:1")
    assert st.available is False and "no ComfyUI" in st.detail


def test_comfyui_detect_local_default(monkeypatch):
    _install_urlopen(monkeypatch, {})
    st = comfyui.detect()
    assert st.available is False and "8188" in st.detail


# --- maya -------------------------------------------------------------------

def _maya_routes(monkeypatch, with_catalog=True):
    routes = {
        "/api/catalog": json.dumps(
            {"models": [{"name": "glm"}, {"name": "deepseek"}]}).encode()
        if with_catalog else 404,
        "/api/ps": json.dumps(
            {"models": [{"name": "glm"}]}).encode(),
        "/api/pull": json.dumps({"ok": True, "job": "p1"}).encode(),
        "/api/pull/status": json.dumps({"state": "running"}).encode(),
        "/api/pull/cancel": json.dumps({"ok": True}).encode(),
        "/api/show": json.dumps({"name": "glm"}).encode(),
        "/api/delete": json.dumps({"ok": True}).encode(),
    }
    _install_urlopen(monkeypatch, routes)


def test_maya_detect_running(monkeypatch):
    _maya_routes(monkeypatch)
    st = maya.detect("http://x:1")
    assert st.id == "maya" and st.available is True
    assert st.models_loaded == ("glm",)
    assert "catalog: glm, deepseek" in st.detail


def test_maya_detect_dead(monkeypatch):
    _install_urlopen(monkeypatch, {})
    st = maya.detect("http://x:1")
    assert st.available is False and "no Maya" in st.detail


def test_maya_api_functions(monkeypatch):
    _maya_routes(monkeypatch)
    assert maya.catalog("http://x:1")["models"][0]["name"] == "glm"
    assert maya.loaded("http://x:1") == ["glm"]
    assert maya.pull("glm", base_url="http://x:1") == {"ok": True, "job": "p1"}
    assert maya.pull_status("http://x:1")["state"] == "running"
    assert maya.pull_cancel("http://x:1") == {"ok": True}
    assert maya.show("glm", "http://x:1")["name"] == "glm"
    assert maya.delete("glm", "http://x:1") == {"ok": True}


def test_maya_api_dead_returns_none(monkeypatch):
    _install_urlopen(monkeypatch, {})
    assert maya.catalog("http://x:1") is None
    assert maya.loaded("http://x:1") == []
    assert maya.pull("glm", "http://x:1") is None