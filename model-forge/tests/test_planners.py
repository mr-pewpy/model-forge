"""Planner + targets tests (pure; persistence to a temp dir)."""

import tempfile
from pathlib import Path

from model_forge.registry import models_for
from model_forge.targets import ComputeTarget, add_target, get_target, load_targets
from model_forge.tools import activation_plan, install_plan

_LOCAL = ComputeTarget(id="local", label="This machine",
                       base_url="http://127.0.0.1:8080", kind="local")
_LXC = ComputeTarget(id="ai-lxc", label="AI LXC", kind="lan",
                     base_url="http://192.168.1.62:8080",
                     vram_gb=48.0, ram_gb=96.0)


def _entry(mid):
    return next(m for m in models_for() if m.id == mid)


def test_targets_round_trip():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        assert [t.id for t in load_targets(base)] == ["local"]
        add_target(_LXC, base_dir=base)
        ids = [t.id for t in load_targets(base)]
        assert ids == ["local", "ai-lxc"]
        got = get_target("ai-lxc", base_dir=base)
        assert got is not None and got.vram_gb == 48.0
        assert get_target("nope", base_dir=base) is None
        # replace by id, local always stays
        add_target(ComputeTarget(id="ai-lxc", label="renamed",
                                 base_url="http://192.168.1.62:8080"), base_dir=base)
        assert len(load_targets(base)) == 2
        assert get_target("ai-lxc", base_dir=base).label == "renamed"


def test_install_plan_maya():
    plan = install_plan(_entry("glm-5.3-flash"), _LXC)
    assert plan["ok"] is True and plan["action"] == "maya_pull"
    assert plan["model"] == "glm-maya-s"          # verified engine-native id
    assert plan["endpoint"] == "http://192.168.1.62:8080/api/pull"
    # explicit override wins
    plan = install_plan(_entry("glm-5.3-flash"), _LXC, engine_model="glm-maya-xs")
    assert plan["model"] == "glm-maya-xs"


def test_install_plan_strata():
    plan = install_plan(_entry("qwen3.8-flash-next"), _LOCAL,
                        engine_model="deepseek41:ist-das-2.00bpw")
    assert plan["ok"] is True and plan["action"] == "run_cli_on_target"
    assert plan["command"] == ["strata", "launch", "--family", "deepseek41",
                               "--model", "ist-das-2.00bpw"]
    plan = install_plan(_entry("qwen3.8-flash-next"), _LOCAL)
    assert plan["ok"] is False and plan["action"] == "needs_engine_model"


def test_install_plan_edges():
    assert install_plan(_entry("core-gguf-catalog"), _LOCAL)["action"] \
        == "use_core_local_models"
    assert install_plan(_entry("world-model-slot"), _LOCAL)["action"] \
        == "unavailable"
    pending = install_plan(_entry("hunyuan3d-2"), _LXC)
    assert pending["ok"] is False and pending["action"] == "adapter_pending"


def test_activation_plan():
    plan = activation_plan(_entry("glm-5.3-flash"), _LXC, ["maya"])
    assert plan["ok"] is True
    assert plan["provider"] == "maya" and plan["model"] == "glm-maya-s"
    assert plan["base_url"] == "http://192.168.1.62:8080/v1"
    # engine not running
    plan = activation_plan(_entry("glm-5.3-flash"), _LXC, ["llamacpp"])
    assert plan["ok"] is False and plan["action"] == "engine_not_running"
    # non-chat modality refuses
    plan = activation_plan(_entry("hunyuan3d-2"), _LXC, ["comfyui"])
    assert plan["ok"] is False and plan["action"] == "not_a_chat_model"
    # explicit served id override
    plan = activation_plan(_entry("glm-5.3-flash"), _LXC, ["maya"],
                           engine_model="glm-maya-l")
    assert plan["model"] == "glm-maya-l"
    # generic endpoint covers unknown engines (remote Strata, vLLM, broker)
    plan = activation_plan(_entry("qwen3.8-flash-next"), _LXC,
                           ["openai-compatible"])
    assert plan["ok"] is True and plan["provider"] == "custom"
    assert plan["base_url"] == "http://192.168.1.62:8080/v1"
