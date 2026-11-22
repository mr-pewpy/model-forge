"""A1 live-entry vetting tests (tools.vet_live_entry + model_forge_install).

A1's unblock is per-row vetting of a LIVE (non-curated) entry — file
resolution + honest fit + engine candidates — before it becomes installable.
The vetting OBSERVES (repo file list, HF pipeline_tag) and never guesses: a
repo that is not one installable file refuses, the VRAM floor stays unknown
(fit is size-advisory), and an unfetchable id refuses fail-closed.
"""

from __future__ import annotations

import json

from model_forge.tools import (
    _modality_for_pipeline_tag, _vet_engines_for_modality, vet_live_entry,
)

_GB = 1024 ** 3


def _fake_hf(siblings=None, payload=None, exc=None):
    """fetch fake for vet_live_entry; records call URLs."""
    calls = []

    def fetch(url):
        calls.append(url)
        if exc is not None:
            raise exc
        if payload is not None:
            return payload
        return {"id": "org/repo", "pipeline_tag": "text-to-image",
                "siblings": siblings or []}

    fetch.calls = calls
    return fetch


# --- pipeline_tag -> modality / engine candidates (data, not guesses) -------

def test_modality_for_pipeline_tag_maps_known_tags():
    assert _modality_for_pipeline_tag("text-to-image") == "image"
    assert _modality_for_pipeline_tag("text-generation") == "chat"
    assert _modality_for_pipeline_tag("text-to-video") == "video"


def test_modality_for_pipeline_tag_unknown_or_absent():
    assert _modality_for_pipeline_tag("") == ""
    assert _modality_for_pipeline_tag("not-a-lane") == ""


def test_vet_engines_narrow_to_file_pull_for_modality():
    # image models install via comfyui (a file-pull engine) — never a phantom.
    engines = _vet_engines_for_modality("image")
    assert "comfyui" in engines
    assert "python" not in engines
    # unknown modality -> the full file-pull set, caller names the engine
    assert set(_vet_engines_for_modality("")) == {"comfyui", "vllm", "sglang"}


# --- vetting: file resolution is verified, fit is honest --------------------

def test_vet_live_resolves_file_and_honest_fit():
    fetch = _fake_hf([
        {"rfilename": "README.md", "size": 1000},
        {"rfilename": "model.safetensors", "size": 5 * _GB},
    ])
    res = vet_live_entry("org/repo", fetch=fetch)
    assert res["ok"] is True and res["action"] == "vetted"
    entry = res["entry"]
    assert entry.id == "org/repo" and entry.source == "org/repo"
    # verified file resolution
    assert res["resolution"]["file"] == "model.safetensors"
    # honest fit: real size, VRAM floor UNKNOWN (0) — never a guessed ceiling
    assert entry.size_gb == round(5 * _GB / 1e9, 3)
    assert entry.vram_gb_min == 0.0
    # engine candidates from the real pipeline_tag (text-to-image -> comfyui)
    assert res["pipeline_tag"] == "text-to-image"
    assert res["modality"] == "image"
    assert "comfyui" in res["engine_candidates"]
    # one fetch (model-info), no re-fetch
    assert len(fetch.calls) == 1


def test_vet_live_refuses_multi_file_repo():
    # a diffusers pipeline (model_index.json) is not one installable file
    fetch = _fake_hf([
        {"rfilename": "model_index.json", "size": 100},
        {"rfilename": "unet/diffusion_model.safetensors", "size": 5 * _GB},
    ])
    res = vet_live_entry("org/repo", fetch=fetch)
    assert res["ok"] is False
    assert res["action"] == "multi_file"
    assert "not installable as a single file" in res["detail"]


def test_vet_live_refuses_no_weights():
    fetch = _fake_hf([{"rfilename": "README.md", "size": 10}])
    res = vet_live_entry("org/repo", fetch=fetch)
    assert res["ok"] is False
    assert res["action"] == "no_weights_found"


def test_vet_live_refuses_unfetchable_fail_closed():
    # network error / not a real repo -> fail-closed, never a guessed entry
    fetch = _fake_hf(exc=RuntimeError("boom"))
    res = vet_live_entry("org/repo", fetch=fetch)
    assert res["ok"] is False
    assert res["action"] == "vet_failed"
    assert "fail-closed" in res["detail"]


def test_vet_live_refuses_non_dict_payload():
    fetch = _fake_hf(payload=["not", "a", "dict"])
    res = vet_live_entry("org/repo", fetch=fetch)
    assert res["ok"] is False and res["action"] == "vet_failed"


# --- install wiring: a vetted live row becomes installable ------------------

def _install_env(monkeypatch, vet_fetch, tmp_path=None):
    """Wire model_forge_install's seams at fakes: target + engine detect +
    the vet fetch + the Hermes plugin-storage modules (the download dest root).
    Returns the tools module."""
    import sys
    import types

    from model_forge import tools
    from model_forge.targets import ComputeTarget

    plugins = types.ModuleType("plugins")
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_data_dir = lambda name: (tmp_path or __import__(
        "pathlib").Path("/tmp")) / name
    plugins.plugin_storage = storage
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)
    cfg_mod = types.ModuleType("hermes_cli.config")
    cfg_mod.load_config_readonly = lambda: {}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg_mod)

    target = ComputeTarget(id="local", label="This machine", kind="local",
                           base_url="http://127.0.0.1:8080")
    monkeypatch.setattr(tools, "get_target", lambda tid=None: target)
    monkeypatch.setattr(tools, "_detect",
                        lambda base="": [type("E", (), {"id": "comfyui",
                                                        "available": True})()])
    monkeypatch.setattr(tools, "_fetch_hf_json", vet_fetch)
    return tools


def test_install_vetted_live_row_is_installable(monkeypatch, tmp_path):
    fetch = _fake_hf([{"rfilename": "model.safetensors", "size": 5 * _GB}])
    tools = _install_env(monkeypatch, fetch, tmp_path)
    out = json.loads(tools.model_forge_install(
        {"model_id": "org/repo", "target": "local",
         "dest": str(tmp_path / "model.safetensors")}))
    assert out.get("vetted") is True and out.get("live_entry") is True
    assert out["vet"]["resolved_file"] == "model.safetensors"
    assert out["vet"]["vram_gb_min"] is None  # unknown, never guessed
    assert "size-advisory" in out["vet"]["fit"]
    # the vet's resolved file drives the install — the download URL is built
    # from it (not a guessed path)
    assert out["url"].endswith("/resolve/main/model.safetensors")


def test_install_refuses_unvettable_live_row(monkeypatch):
    fetch = _fake_hf([{"rfilename": "model_index.json", "size": 100}])
    tools = _install_env(monkeypatch, fetch)
    out = json.loads(tools.model_forge_install(
        {"model_id": "org/repo", "target": "local"}))
    assert out.get("vetted") is False
    assert out.get("action") == "install_refused"
    assert "not installable as a single file" in out["error"]


def test_install_curated_id_unchanged(monkeypatch):
    # a curated id still resolves through the registry with no vetting flag
    fetch = _fake_hf(exc=AssertionError("must not fetch for a curated id"))
    tools = _install_env(monkeypatch, fetch)
    out = json.loads(tools.model_forge_install(
        {"model_id": "glm-5.3-flash", "target": "local"}))
    assert out.get("vetted") is None
    assert fetch.calls == []
