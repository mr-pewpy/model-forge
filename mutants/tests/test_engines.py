"""Engine-adapter tests (llama.cpp/Ollama, Strata, Maya, ComfyUI, vLLM, SGLang,
Exo, openai-compatible, plus the base helpers, the detect_engines aggregator
and its user-declared extra_roots seam).

Every adapter probes local engines over stdlib urllib (no auth assumed), so a
single fake ``urllib.request.urlopen`` that dispatches on the requested path
covers the whole surface. Nothing here touches the network: every probe is
routed through the fake, and the "not running" branches are exercised by
having the fake raise (as a dead endpoint would).
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error

import pytest

from model_forge.engines import (
    comfyui,
    exo,
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
    assert root_url("http://gpu-box.example.com:8080/v1") == "http://gpu-box.example.com:8080"
    assert root_url("http://gpu-box.example.com:8080/") == "http://gpu-box.example.com:8080"
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
    # a llama-server on the default port answers both /props (its distinctive
    # fingerprint) and /v1/models (generic): the named row wins and the
    # generic row for that endpoint is suppressed
    _install_urlopen(monkeypatch, {
        "/props": json.dumps(
            {"build_info": "x", "model_path": "m.gguf"}).encode(),
        "/v1/models": json.dumps(
            {"data": [{"id": "m1"}]}).encode(),
    })
    out = detect_engines("")
    assert any(e.id == "llamacpp" and e.available for e in out)
    # no generic openai-compatible row claims the SAME (default) endpoint
    for e in out:
        if e.id == "openai-compatible" and e.available:
            assert e.endpoint != strata.DEFAULT_ROOT
    # and the llama-server is never double-labeled as Strata
    assert not any(e.id == "strata" and e.available for e in out)


def test_detect_engines_generic_models_not_mislabeled(monkeypatch):
    # generic /v1/models on the default port with NO engine fingerprint:
    # one honest "identity unknown" row, and never a selectable Strata row
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"data": [{"id": "m1"}]}).encode(),
    })
    monkeypatch.setattr(strata.shutil, "which", lambda _: None)
    out = detect_engines("")
    assert not any(e.id == "strata" and e.available for e in out)
    generic = [e for e in out
               if e.id == "openai-compatible" and e.available
               and e.endpoint == strata.DEFAULT_ROOT]
    assert len(generic) == 1
    assert "identity unknown" in generic[0].detail


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


# --- detect_engines parallelism: dead targets must not stall ------------------
def test_detect_engines_probes_detectors_in_parallel(monkeypatch):
    # every call burns a stand-in for a dead-connection timeout; eight
    # sequential detectors would sum to >= 3.0 s here, while the concurrent
    # aggregator must finish near the slowest single detector chain (~0.9 s)
    delay = 0.3
    lock = threading.Lock()
    calls = []

    def _fake(req, timeout=None):
        with lock:
            calls.append(1)
        time.sleep(delay)
        raise urllib.error.URLError("no route in test")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    start = time.monotonic()
    out = detect_engines("http://x:1")
    elapsed = time.monotonic() - start
    assert len(calls) == 10         # every adapter probed exactly once-chain
    assert len(out) == 8            # and every detector still yields a row
    assert elapsed < 1.5            # sequential would be >= 3.0 s


def test_detect_engines_order_stable_when_probes_finish_out_of_order(monkeypatch):
    # the first detector is the slowest one; if results were collected in
    # completion order the fast SGLang row would jump ahead of it
    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.endswith("/props"):
            time.sleep(0.4)
            raise urllib.error.HTTPError(url, 404, "e", {}, None)
        if url.endswith("/get_model_info"):
            return _Resp(json.dumps({"model_path": "flux"}).encode())
        raise urllib.error.URLError("no route in test")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    out = detect_engines("http://x:1")
    ids = [e.id for e in out]
    assert ids == ["llamacpp", "strata", "maya", "comfyui", "vllm", "sglang",
                   "exo", "openai-compatible"]
    assert ids.index("llamacpp") < ids.index("sglang")


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


def test_strata_detect_generic_models_not_mislabeled(monkeypatch):
    # generic /v1/models must NEVER yield a selectable 'Strata available' row
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"data": [{"id": "glm"}]}).encode()})
    monkeypatch.setattr(strata.shutil, "which", lambda _: None)
    st = strata.detect()
    assert not (st.id == "strata" and st.available)
    assert st.id == "openai-compatible" and st.available is True
    assert "generic OpenAI-compatible endpoint on port 8080" in st.detail
    assert "identity unknown" in st.detail


def test_strata_detect_generic_models_with_cli_still_not_mislabeled(monkeypatch):
    # even with the strata CLI on PATH, a generic /v1/models answer is not
    # proof of a Strata server: still no selectable 'Strata available' row
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"data": [{"id": "glm"}]}).encode()})
    monkeypatch.setattr(strata.shutil, "which", lambda _: "/usr/bin/strata")
    st = strata.detect()
    assert not (st.id == "strata" and st.available)
    assert st.id == "openai-compatible"
    assert "identity unknown" in st.detail
    assert "strata status" in st.detail   # how to confirm, if it is Strata


def test_strata_fingerprint_never_from_models(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"data": [{"id": "glm"}]}).encode()})
    assert strata.strata_fingerprint("") is False


def test_strata_detect_serving_confirmed_by_telemetry(monkeypatch):
    # only a Strata-specific surface (the telemetry snapshot behind
    # `strata status`) lets detect claim 'Strata serving <models>'
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"data": [{"id": "glm"}]}).encode(),
        "/v1/telemetry": json.dumps({
            "sampled_at": 1, "model": {"id": "glm", "loaded": True},
            "instance": {"id": "i1", "state": "ready"},
            "work": {}, "quality": {}}).encode(),
    })
    monkeypatch.setattr(strata.shutil, "which", lambda _: "/usr/bin/strata")
    st = strata.detect()
    assert st.available is True and st.id == "strata"
    assert st.models_loaded == ("glm",)
    assert "serving glm" in st.detail


def test_strata_detect_serving_confirmed_by_status_fallback(monkeypatch):
    # older servers: /v1/status exposes `loaded` and `auto_load`
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"data": [{"id": "glm"}]}).encode(),
        "/v1/telemetry": 404,
        "/v1/status": json.dumps({"loaded": True, "auto_load": True,
                                  "model": "glm"}).encode(),
    })
    monkeypatch.setattr(strata.shutil, "which", lambda _: None)
    st = strata.detect()
    assert st.available is True and st.id == "strata"
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


def test_llamacpp_detect_ollama_default_port(monkeypatch):
    # stock Ollama listens on 11434 — the local scan must cover it
    assert 11434 in llamacpp.PROBE_PORTS
    routes = {
        "11434/props": 404,
        "11434/api/tags": json.dumps({"models": [{"name": "llama3"}]}).encode(),
    }
    _install_urlopen(monkeypatch, routes)
    st = llamacpp.detect()
    assert st.id == "ollama" and st.available is True
    assert st.endpoint == "http://127.0.0.1:11434"
    assert st.models_loaded == ("llama3",)


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
    assert st.available is False and "8080/8081/8000/8001/11434" in st.detail


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
            "devices": [{"name": "NVIDIA A100"}, {"name": "RTX 4090"}],
        }).encode(),
    })
    st = comfyui.detect("http://x:1")
    assert st.available is True and "comfyui 0.3.0" in st.detail
    assert "NVIDIA A100" in st.detail


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


# --- exo ---------------------------------------------------------------------
def test_exo_detect_fingerprint_node_id(monkeypatch):
    # the documented exo meta endpoint (docs/api.md §1): GET /node_id answers
    # the node identity as a bare JSON string — with it, "exo" is claimed
    _install_urlopen(monkeypatch, {
        "/node_id": json.dumps("3f9c2a7e1b5d4c8e9a0f6b2d7c4e1a3b").encode(),
        "/v1/models": json.dumps({"object": "list", "data": [
            {"id": "mlx-community/Llama-3.2-1B-Instruct-4bit"}]}).encode(),
        "/openapi.json": json.dumps(
            {"paths": {"/v1/images/generations": {}}}).encode(),
    })
    st = exo.detect("http://x:1")
    assert st.id == "exo" and st.available is True
    assert st.endpoint == "http://x:1"
    assert st.models_loaded == ("mlx-community/Llama-3.2-1B-Instruct-4bit",)
    assert "image" in st.modalities and "image_edit" in st.modalities


def test_exo_detect_fingerprint_state_fallback(monkeypatch):
    # /node_id gone (404): the cluster state object is the fallback fingerprint
    _install_urlopen(monkeypatch, {
        "/node_id": 404,
        "/state": json.dumps({"topology": {"nodes": []}, "instances": {},
                              "deployments": {}}).encode(),
        "/v1/models": json.dumps({"object": "list", "data": []}).encode(),
    })
    st = exo.detect("http://x:1")
    assert st.id == "exo" and st.available is True


def test_exo_fingerprint_never_from_models_or_healthcheck(monkeypatch):
    # a generic /v1/models list — and a generic /healthcheck — are NOT proof
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"object": "list",
                                  "data": [{"id": "m1"}]}).encode(),
        "/healthcheck": json.dumps({"status": "ok"}).encode(),
    })
    assert exo.exo_fingerprint("http://x:1") is False


def test_exo_detect_generic_models_not_mislabeled(monkeypatch):
    # /v1/models alone: never an 'exo' row — the honest generic row instead
    _install_urlopen(monkeypatch, {
        "/v1/models": json.dumps({"object": "list",
                                  "data": [{"id": "m1"}]}).encode(),
    })
    st = exo.detect("http://x:1")
    assert not (st.id == "exo" and st.available)
    assert st.id == "openai-compatible" and st.available is True
    assert "identity unknown" in st.detail


def test_exo_detect_dead(monkeypatch):
    _install_urlopen(monkeypatch, {})
    st = exo.detect("http://x:1")
    assert st.id == "exo" and st.available is False
    assert "no exo answering" in st.detail


def test_exo_detect_local_default_root(monkeypatch):
    # stock exo serves the dashboard and API on :52415 — the local probe
    # covers that default endpoint
    assert exo.DEFAULT_ROOT == "http://127.0.0.1:52415"
    _install_urlopen(monkeypatch, {
        "52415/node_id": json.dumps("abc123").encode(),
        "52415/v1/models": json.dumps({"object": "list",
                                       "data": [{"id": "m"}]}).encode(),
    })
    st = exo.detect()
    assert st.id == "exo" and st.available is True
    assert st.endpoint == "http://127.0.0.1:52415"


def test_detect_engines_exo_named_suppresses_generic(monkeypatch):
    # an exo fingerprint at the endpoint drops the generic openai-compatible
    # row for that same endpoint (same server)
    _install_urlopen(monkeypatch, {
        "52415/node_id": json.dumps("abc123").encode(),
        "52415/v1/models": json.dumps({"object": "list",
                                       "data": [{"id": "m"}]}).encode(),
    })
    monkeypatch.setattr(strata.shutil, "which", lambda _: None)
    out = detect_engines("")
    exo_rows = [e for e in out if e.id == "exo" and e.available]
    assert len(exo_rows) == 1
    for e in out:
        if e.id == "openai-compatible" and e.available:
            assert e.endpoint != "http://127.0.0.1:52415"


# --- detect_engines: user-declared roots (extra_roots) -----------------------
def test_extra_roots_probes_declared_roots_and_labels_rows(monkeypatch):
    seen = []
    lock = threading.Lock()

    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        with lock:
            seen.append(url)
        raise urllib.error.URLError("no route in test")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    out = detect_engines("http://auto:1", extra_roots=["http://decl:9000"])
    # the declared root was actually probed, alongside the base endpoint
    assert any(u.startswith("http://decl:9000/") for u in seen)
    assert any(u.startswith("http://auto:1/") for u in seen)
    # every declared-root row carries the provenance label; base rows do not
    declared = [e for e in out if e.source == "user-declared"]
    base = [e for e in out if e.source != "user-declared"]
    assert len(declared) == 8 and len(base) == 8
    assert not any(e.source == "user-declared" for e in base)


def test_extra_roots_fingerprint_first_and_generic_dropped(monkeypatch):
    # a declared root with an exo fingerprint: the named row is reported as
    # user-declared, and the generic row for that endpoint is dropped
    _install_urlopen(monkeypatch, {
        "decl:9000/node_id": json.dumps("abc123").encode(),
        "decl:9000/v1/models": json.dumps({"object": "list",
                                           "data": [{"id": "m1"}]}).encode(),
    })
    out = detect_engines("http://auto:1", extra_roots=["http://decl:9000"])
    declared = [e for e in out if e.source == "user-declared"]
    named = [e for e in declared if e.id == "exo" and e.available]
    assert len(named) == 1
    assert named[0].endpoint == "http://decl:9000"
    assert named[0].models_loaded == ("m1",)
    assert not any(e.id == "openai-compatible" and e.available
                   and e.endpoint == "http://decl:9000" for e in declared)


def test_extra_roots_generic_never_mislabeled(monkeypatch):
    # a declared root serving only generic /v1/models: one honest generic
    # row labeled user-declared — never an 'exo' claim
    _install_urlopen(monkeypatch, {
        "decl:9000/v1/models": json.dumps({"object": "list",
                                           "data": [{"id": "m1"}]}).encode(),
    })
    out = detect_engines("http://auto:1", extra_roots=["http://decl:9000"])
    declared = [e for e in out if e.source == "user-declared"]
    assert not any(e.id == "exo" and e.available for e in declared)
    generic = [e for e in declared if e.id == "openai-compatible"
               and e.available and e.endpoint == "http://decl:9000"]
    assert len(generic) == 1
    assert "identity unknown" in generic[0].detail


def test_extra_roots_dedupes_repeated_and_skips_empty(monkeypatch):
    seen = []

    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        seen.append(url)
        raise urllib.error.URLError("no route in test")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    out = detect_engines("http://auto:1",
                         extra_roots=["http://decl:9000", "", "http://decl:9000/"])
    declared = [e for e in out if e.source == "user-declared"]
    # repeated and empty declarations probe the root once and yield one group
    assert len(declared) == 8
    assert sum(1 for u in seen if u.startswith("http://decl:9000/")) == 10


def test_extra_roots_absent_behavior_is_unchanged(monkeypatch):
    # with the parameter absent (or empty), no declared root is probed and
    # the rows are exactly the base detection result
    seen = []
    lock = threading.Lock()

    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        with lock:
            seen.append(url)
        if url.endswith("/props"):
            return _Resp(json.dumps(
                {"build_info": "x", "model_path": "m.gguf"}).encode())
        if url.endswith("/v1/models"):
            return _Resp(json.dumps({"data": [{"id": "m1"}]}).encode())
        raise urllib.error.URLError("no route in test")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    plain = detect_engines("http://x:1")
    explicit_none = detect_engines("http://x:1", extra_roots=None)
    explicit_empty = detect_engines("http://x:1", extra_roots=[])
    assert plain == explicit_none == explicit_empty
    assert not any(e.source == "user-declared" for e in plain)
    # no probe ever left the base endpoint
    assert all(u.startswith("http://x:1/") for u in seen)
