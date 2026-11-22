"""A19 tools.py error-degradation branch fixtures.

The largest remaining coverage gap is tools.py's refusal / degradation paths —
the honest "this input is wrong / this target is missing / this engine cannot
serve it" branches. These drive those branches directly (pure functions and
tool handlers with fakes) so the refusals are OBSERVED, not left uncovered.
"""

from __future__ import annotations

import sys
import types


def _hermes_fakes(monkeypatch, tmp_path):
    plugins = types.ModuleType("plugins")
    storage = types.ModuleType("plugins.plugin_storage")
    storage.plugin_data_dir = lambda name: tmp_path / name
    plugins.plugin_storage = storage
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.plugin_storage", storage)
    cfg = types.ModuleType("hermes_cli.config")
    cfg.load_config_readonly = lambda: {}
    cfg.load_config = lambda: {}
    cfg.save_config = lambda c: None
    monkeypatch.setitem(sys.modules, "hermes_cli.config", cfg)


# --- target lifecycle refusals (W1.2) --------------------------------------

def test_target_remove_requires_id():
    from model_forge.tools import _target_remove
    out = _target_remove({})
    assert out["ok"] is False and out["error"] == "id is required"


def test_target_remove_local_is_permanent():
    from model_forge.tools import _target_remove
    out = _target_remove({"id": "local"})
    assert out["ok"] is False and "permanent" in out["error"].lower()


# --- install refusals ------------------------------------------------------

def test_install_unknown_target_refused(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path)
    from model_forge import tools
    monkeypatch.setattr(tools, "get_target", lambda tid=None: None)
    out = tools.model_forge_install({"model_id": "glm-5.3-flash",
                                     "target": "nope"})
    assert "unknown target" in out


def test_install_requires_model_id(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path)
    from model_forge import tools
    out = tools.model_forge_install({})
    assert "model_id required" in out


def test_install_unknown_action_refused(monkeypatch, tmp_path):
    _hermes_fakes(monkeypatch, tmp_path)
    from model_forge import tools
    out = tools.model_forge_install({"action": "bogus"})
    assert "unknown action" in out


# --- install_plan engine-specific refusals --------------------------------

def _entry(mid):
    from model_forge.registry import models_for
    return next(m for m in models_for() if m.id == mid)


def _target(kind="local", base="http://x:8080"):
    from model_forge.targets import ComputeTarget
    return ComputeTarget(id="t", label="T", kind=kind, base_url=base)


def test_install_plan_engine_mismatch():
    from model_forge.tools import install_plan
    plan = install_plan(_entry("glm-5.3-flash"), _target(),
                        engine="comfyui", available_engines=["comfyui"])
    assert plan["ok"] is False and plan["action"] == "engine_mismatch"


def test_install_plan_maya_resolves_native_id():
    from model_forge.tools import install_plan
    # glm-5.3-flash carries a Maya native id (glm-maya-s) -> resolves, no refusal
    plan = install_plan(_entry("glm-5.3-flash"), _target(kind="lan"),
                        available_engines=["maya"])
    assert plan["ok"] is True and plan["model"] == "glm-maya-s"


def test_install_plan_maya_needs_engine_model_when_no_native_id():
    from model_forge.tools import install_plan
    from model_forge.registry import ModelEntry
    e = ModelEntry(id="m", name="M", modality="chat", engines=("maya",),
                   source="org/m")  # no engine_models -> needs a Maya setup id
    plan = install_plan(e, _target(kind="lan"), available_engines=["maya"])
    assert plan["ok"] is False and plan["action"] == "needs_engine_model"


def test_install_plan_strata_needs_family_pack():
    from model_forge.tools import install_plan
    plan = install_plan(_entry("qwen3.8-flash-next"), _target(),
                        engine_model="nosplit", available_engines=["strata"])
    assert plan["ok"] is False and plan["action"] == "needs_engine_model"


def test_install_plan_phantom_python_unmanaged():
    from model_forge.tools import install_plan
    from model_forge.registry import ModelEntry
    e = ModelEntry(id="x", name="X", modality="music", engines=("python",),
                   source="org/x")
    plan = install_plan(e, _target(), available_engines=["python"])
    assert plan["ok"] is False and plan["action"] == "custom_env_unmanaged"


def test_install_plan_core_gguf_redirects():
    from model_forge.tools import install_plan
    plan = install_plan(_entry("core-gguf-catalog"), _target(),
                        available_engines=["llamacpp"])
    assert plan["action"] == "use_core_local_models"


# --- activation refusals ---------------------------------------------------

def test_activation_not_a_chat_model():
    from model_forge.tools import activation_plan
    plan = activation_plan(_entry("hunyuan3d-2"), _target(),
                           available_engines=["comfyui"])
    assert plan["ok"] is False and plan["action"] == "not_a_chat_model"


def test_activation_stub_refused():
    from model_forge.tools import activation_plan
    from model_forge.registry import ModelEntry
    e = ModelEntry(id="x", name="X", modality="chat", engines=("maya",),
                   source="s", status="stub")
    plan = activation_plan(e, _target(), available_engines=["maya"])
    assert plan["ok"] is False and plan["action"] == "placeholder_not_activatable"


def test_activation_engine_not_running():
    from model_forge.tools import activation_plan
    plan = activation_plan(_entry("glm-5.3-flash"), _target(),
                           available_engines=["llamacpp"])
    assert plan["ok"] is False and plan["action"] == "engine_not_running"
