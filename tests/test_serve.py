"""Service-assembly tests (model_forge/serve.py) — fakes only, no docker/net.

``build_stack`` is the one place the tested parts are wired together, so these
tests assert that wiring directly: the registry ``runtime`` entries become the
right container lifecycles, the bookings cover both lanes, env + JSON config
resolves with the documented precedence, and the graceful path releases scopes.
The wiring smoke drives the REAL Orchestrator + EvictionPlanner + the REAL
DockerContainerLifecycle adapters against a fake docker CLI and a fake health
probe, so a chat claim, a gen claim (which must evict the chat lane) and a
release (which restores it) exercise the true code paths and touch no docker and
no network.
"""

from __future__ import annotations

import json

import pytest

import model_forge.broker as broker
from model_forge.broker import LoadRequest
from model_forge.facade import CHAT_MODEL_ID
from model_forge.registry import registry_bookings
from model_forge.serve import (
    CHAT_BOOKED_GB,
    GEN_CARD_CEILING_GB,
    GEN_WORKFLOW_IDS,
    MACHINE_VRAM_GB,
    TOPOLOGY_ENTRIES,
    build_stack,
    graceful_close,
    release_scopes,
    resolve_settings,
    topology_bookings,
)


# --- fakes ------------------------------------------------------------------

class FakeDockerRunner:
    """Injected docker CLI seam: records the subcommand and returns an opaque
    handle (so the adapter's probe falls back to its tracked state). Never
    touches real docker."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, args):
        self.calls.append([str(a) for a in args])
        return object()


class FakeHealth:
    """Injected container health probe. Records every URL polled; ``healthy``
    controls the answer (default: every container answers healthy)."""

    def __init__(self, healthy: bool = True):
        self.healthy = healthy
        self.urls: list[str] = []

    def __call__(self, url, timeout=2.0) -> bool:
        self.urls.append(url)
        return self.healthy


@pytest.fixture
def restore_health():
    """build_stack installs broker._http_healthy when ``health`` is passed;
    snapshot/restore the module seam so no test leaks it."""
    original = broker._http_healthy
    yield
    broker._http_healthy = original


# --- build_stack: lifecycle wiring ------------------------------------------

def test_build_stack_wires_container_lifecycles_per_runtime_entries():
    runner = FakeDockerRunner()
    stack = build_stack(docker_runner=runner, health=FakeHealth(),
                        sleep=lambda _s: None)

    adapters = stack["adapters"]
    assert set(adapters) == {"vllm", "comfyui"}

    vllm = adapters["vllm"]
    assert vllm._container == "chat-engine"
    assert vllm._health_url == "http://127.0.0.1:8000/health"
    assert vllm._boot_budget_s == 480.0
    assert vllm._gpu_count == 2                     # TP2: both 3090s

    comfy = adapters["comfyui"]
    assert comfy._container == "gen-engine"
    assert comfy._health_url == "http://127.0.0.1:18188/"
    assert comfy._boot_budget_s == 600.0
    assert comfy._gpu_count == 1                    # single card

    # the orchestrator holds exactly those adapters under the engine ids the
    # facade puts on its LoadRequests.
    assert stack["orchestrator"]._adapters == adapters


def test_topology_entries_are_the_documented_runtimes():
    by_engine = {e.engines[0]: e.runtime for e in TOPOLOGY_ENTRIES}
    assert by_engine["vllm"]["name"] == "chat-engine"
    assert by_engine["vllm"]["gpu_count"] == 2
    assert by_engine["comfyui"]["name"] == "gen-engine"
    assert by_engine["comfyui"]["gpu_count"] == 1


# --- build_stack: bookings + budget -----------------------------------------

def test_bookings_cover_chat_lane_and_gen_lanes(restore_health):
    stack = build_stack(docker_runner=FakeDockerRunner(), health=FakeHealth(),
                        sleep=lambda _s: None)
    ledger = stack["ledger"]

    # chat lane: TP2 across both cards (22.8 GB/card).
    assert ledger.booked_vram_gb(CHAT_MODEL_ID) == pytest.approx(CHAT_BOOKED_GB)
    chat_booking = ledger.booking(CHAT_MODEL_ID)
    assert chat_booking.source == "published"           # advisory, not measured
    assert "advisory" in chat_booking.label
    assert chat_booking.load_s == 480.0

    # every advertised gen workflow is booked, at most the single-card ceiling.
    registry = registry_bookings()
    for wf in GEN_WORKFLOW_IDS:
        gb = ledger.booked_vram_gb(wf)
        assert gb is not None, wf
        assert gb <= GEN_CARD_CEILING_GB, wf
        if wf in registry:                          # registry footprint used
            assert gb == pytest.approx(registry[wf]["vram_gb"]), wf


def test_budget_declared_on_the_shared_target(restore_health):
    stack = build_stack(docker_runner=FakeDockerRunner(), health=FakeHealth(),
                        sleep=lambda _s: None)
    assert stack["ledger"].budget_gb(stack["target_id"]) == pytest.approx(
        MACHINE_VRAM_GB)
    # the chat booking fits the declared pool (else the lane could never load).
    assert CHAT_BOOKED_GB <= MACHINE_VRAM_GB


def test_topology_bookings_shape_is_ledger_consumable():
    bookings = topology_bookings()
    for model_id, spec in bookings.items():
        assert set(spec) == {"vram_gb", "source", "load_s"}, model_id
        assert spec["source"] == "published"


# --- config resolution ------------------------------------------------------

def test_resolve_settings_defaults():
    config, host, port = resolve_settings(env={})
    assert host == "0.0.0.0" and port == 8080
    assert config.llm_base_url == "http://127.0.0.1:8000/v1"
    assert config.llm_served_id == CHAT_MODEL_ID
    assert config.comfy_base_url == "http://127.0.0.1:18188"
    assert config.assets_dir == "/var/lib/model-forge/assets"


def test_resolve_settings_reads_env():
    env = {
        "FACADE_HOST": "127.0.0.1",
        "FACADE_PORT": "9099",
        "LLM_BASE_URL": "http://vllm:8000/v1",
        "LLM_SERVED_ID": "served-id",
        "COMFYUI_BASE_URL": "http://comfy:18188",
        "ASSETS_DIR": "/srv/assets",
        "PUBLIC_BASE_URL": "http://62.lan:8080",
    }
    config, host, port = resolve_settings(env=env)
    assert (host, port) == ("127.0.0.1", 9099)
    assert config.llm_base_url == "http://vllm:8000/v1"
    assert config.llm_served_id == "served-id"
    assert config.comfy_base_url == "http://comfy:18188"
    assert config.assets_dir == "/srv/assets"
    assert config.public_base_url == "http://62.lan:8080"


def test_facade_config_file_overrides_env_and_explicit_wins(tmp_path):
    cfg_path = tmp_path / "facade.json"
    cfg_path.write_text(json.dumps({
        "llm_served_id": "file-served",
        "COMFYUI_BASE_URL": "http://file-comfy:18188",   # env-name spelling
        "FACADE_PORT": 7000,
        "host": "198.51.100.7",
    }))

    env = {
        "LLM_SERVED_ID": "env-served",                  # overridden by file
        "COMFYUI_BASE_URL": "http://env-comfy:18188",   # overridden by file
        "FACADE_PORT": "8080",
        "FACADE_HOST": "0.0.0.0",
    }
    config, host, port = resolve_settings(env=env, config_file=str(cfg_path))
    assert config.llm_served_id == "file-served"        # file over env
    assert config.comfy_base_url == "http://file-comfy:18188"
    assert (host, port) == ("198.51.100.7", 7000)

    # explicit keyword arguments win over the file.
    config2, host2, port2 = resolve_settings(
        env=env, config_file=str(cfg_path),
        llm_served_id="explicit", host="192.0.2.10", port=1234)
    assert config2.llm_served_id == "explicit"
    assert (host2, port2) == ("192.0.2.10", 1234)


def test_resolve_settings_facade_config_env_path(monkeypatch, tmp_path):
    cfg_path = tmp_path / "fromenv.json"
    cfg_path.write_text(json.dumps({"llm_base_url": "http://x/v1"}))
    monkeypatch.setenv("FACADE_CONFIG", str(cfg_path))
    config, _, _ = resolve_settings()                   # env=None -> os.environ
    assert config.llm_base_url == "http://x/v1"


def test_resolve_settings_bad_port_fails_closed():
    with pytest.raises(ValueError, match="FACADE_PORT"):
        resolve_settings(env={"FACADE_PORT": "not-a-port"})


def test_llm_api_key_is_env_only_never_injectable_from_file_or_explicit(
        tmp_path, monkeypatch):
    # The documented contract: BROKER_VLLM_API_KEY is the only source of
    # llm_api_key (the FacadeConfig field default reads os.environ). A
    # FACADE_CONFIG file — or an explicit kwarg — must NOT be able to inject a
    # key over that contract.
    cfg_path = tmp_path / "facade.json"
    cfg_path.write_text(json.dumps({"llm_api_key": "file-key",
                                    "llm_served_id": "file-served"}))
    monkeypatch.setenv("BROKER_VLLM_API_KEY", "env-key")

    config, _, _ = resolve_settings(env={}, config_file=str(cfg_path))
    assert config.llm_served_id == "file-served"     # whitelisted field applies
    assert config.llm_api_key == "env-key"           # file key ignored

    # an explicit llm_api_key kwarg is likewise refused (whitelist only)
    config2, _, _ = resolve_settings(env={}, llm_api_key="explicit-key")
    assert config2.llm_api_key == "env-key"

    # with no env key at all, no source can set one
    monkeypatch.delenv("BROKER_VLLM_API_KEY", raising=False)
    config3, _, _ = resolve_settings(env={}, llm_api_key="explicit-key")
    assert config3.llm_api_key == ""
    config4, _, _ = resolve_settings(env={}, config_file=str(cfg_path))
    assert config4.llm_api_key == ""


# --- graceful shutdown ------------------------------------------------------

class FakeServer:
    def __init__(self):
        self.shutdown_calls = 0
        self.close_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1

    def server_close(self):
        self.close_calls += 1


class FakeOrchestrator:
    """Records release calls; a stand-in for the scope table under shutdown."""

    def __init__(self, scopes):
        self._scopes = {s: {} for s in scopes}
        self.released: list[str] = []

    def release(self, scope_id):
        self.released.append(scope_id)
        self._scopes.pop(scope_id, None)
        return {"ok": True}


def test_release_scopes_releases_every_open_scope():
    orch = FakeOrchestrator(["a", "b", "c"])
    released = release_scopes(orch)
    assert sorted(released) == ["a", "b", "c"]
    assert sorted(orch.released) == ["a", "b", "c"]
    # idempotent: nothing left to release
    assert release_scopes(orch) == []


def test_graceful_close_stops_server_then_releases_scopes():
    order: list[str] = []

    class _Server(FakeServer):
        def shutdown(self):
            order.append("shutdown")
            super().shutdown()

        def server_close(self):
            order.append("close")
            super().server_close()

    class _Orch(FakeOrchestrator):
        def release(self, scope_id):
            order.append(f"release:{scope_id}")
            return super().release(scope_id)

    server = _Server()
    stack = {"orchestrator": _Orch(["s1"])}
    logs: list[str] = []
    released = graceful_close(server, stack, log=logs.append)

    assert released == ["s1"]
    assert server.shutdown_calls == 1 and server.close_calls == 1
    # socket closed before any scope is released (a draining request still owns
    # its scope), and the shutdown was logged honestly.
    assert order == ["shutdown", "close", "release:s1"]
    assert any("drain" in line for line in logs)


def test_graceful_close_tolerates_a_missing_server():
    orch = FakeOrchestrator(["only"])
    assert graceful_close(None, {"orchestrator": orch}) == ["only"]


# --- wiring smoke: chat claim + gen claim eviction + release restore --------

def test_wiring_smoke_chat_gen_evict_and_release_restore(restore_health):
    runner = FakeDockerRunner()
    health = FakeHealth()
    stack = build_stack(docker_runner=runner, health=health,
                        sleep=lambda _s: None)
    orch = stack["orchestrator"]
    target = stack["target_id"]

    # -- chat claim: brings the vLLM container up ---------------------------
    chat_req = LoadRequest(model_id=CHAT_MODEL_ID, engine="vllm",
                           lane="chat", modality="chat")
    chat_claim = orch.claim("chat-1", target, chat_req)
    assert chat_claim["ok"] is True
    assert ["start", "chat-engine"] in runner.calls
    assert stack["ledger"].is_resident(target, CHAT_MODEL_ID)
    assert "http://127.0.0.1:8000/health" in health.urls

    # -- gen claim: evicts the chat lane, then brings ComfyUI up ------------
    gen_req = LoadRequest(model_id="qwen-image-2.1", engine="comfyui",
                          lane="image", modality="image")
    gen_claim = orch.claim("img-1", target, gen_req)
    assert gen_claim["ok"] is True
    # the chat lane was evicted (docker stop) BEFORE ComfyUI started.
    assert ["stop", "chat-engine"] in runner.calls
    assert ["start", "gen-engine"] in runner.calls
    assert runner.calls.index(["stop", "chat-engine"]) < \
        runner.calls.index(["start", "gen-engine"])
    assert not stack["ledger"].is_resident(target, CHAT_MODEL_ID)
    assert stack["ledger"].is_resident(target, "qwen-image-2.1")
    assert "http://127.0.0.1:18188/" in health.urls

    # -- release: steps ComfyUI down and restores the chat lane -------------
    rel = orch.release("img-1")
    assert rel["ok"] is True
    assert ["stop", "gen-engine"] in runner.calls
    assert stack["ledger"].is_resident(target, CHAT_MODEL_ID)
    assert not stack["ledger"].is_resident(target, "qwen-image-2.1")
    # vLLM was started again to restore the lane (second chat boot).
    assert runner.calls.count(["start", "chat-engine"]) == 2


def test_wiring_smoke_video_claim_also_evicts_chat(restore_health):
    runner = FakeDockerRunner()
    stack = build_stack(docker_runner=runner, health=FakeHealth(),
                        sleep=lambda _s: None)
    orch = stack["orchestrator"]
    target = stack["target_id"]

    orch.claim("chat-1", target,
               LoadRequest(model_id=CHAT_MODEL_ID, engine="vllm",
                           lane="chat", modality="chat"))
    video_claim = orch.claim("vid-1", target,
                             LoadRequest(model_id="h3-t2v", engine="comfyui",
                                         lane="video", modality="video"))
    assert video_claim["ok"] is True
    # a video lane does not fit beside the TP2 chat model -> chat is evicted.
    assert ["stop", "chat-engine"] in runner.calls
    assert ["start", "gen-engine"] in runner.calls


# --- wiring smoke: the FACADE chat path wakes the vLLM lane and leaves it up -

class _WfCfg:
    def __init__(self, output_kind):
        self.output_kind = output_kind


class FakeComfyRunner:
    """Records runner calls; returns a fixed file list; counts free()."""

    def __init__(self, files=None):
        self.workflows = {"qwen-image-2.1": _WfCfg("image")}
        self.files = list(files or [])
        self.run_calls = []
        self.free_calls = 0

    def run(self, workflow_id, params, *, assets_dir, job_id, timeout_s=None):
        self.run_calls.append((workflow_id, job_id))
        return {"ok": True, "output_kind": "image", "files": list(self.files),
                "reason": "", "meta": {}}

    def free(self):
        self.free_calls += 1
        return {"ok": True}


def test_wiring_smoke_facade_chat_wakes_vllm_and_gen_evicts_then_restores(
        restore_health):
    """The acceptance shape driven through the FACADE's real routes:
    claim(chat) -> vLLM up (and left UP after the chat request); then
    claim(image) -> chat evicted + ComfyUI up -> job -> release -> chat
    restored."""
    runner = FakeDockerRunner()
    comfy = FakeComfyRunner()
    stack = build_stack(docker_runner=runner, health=FakeHealth(),
                        sleep=lambda _s: None, comfy_runner=comfy)
    service = stack["service"]
    service._dispatch = lambda fn: fn()          # run gen jobs inline
    service._upstream = lambda raw, headers: (200, b'{"ok":true}')
    target = stack["target_id"]

    # -- chat via the facade: wakes the vLLM lane, leaves it UP -------------
    status, _ = service.handle_chat(
        json.dumps({"model": CHAT_MODEL_ID}).encode())
    assert status == 200
    assert ["start", "chat-engine"] in runner.calls
    assert stack["ledger"].is_resident(target, CHAT_MODEL_ID)   # lane stays up
    # the chat scope was released (restore=False) — no open scope remains.
    assert stack["orchestrator"]._scopes == {}

    # -- gen via the facade: evicts chat, brings ComfyUI up, then restores ---
    status, _body = service.handle_image_generations(
        json.dumps({"model": "gen/qwen-image-2.1"}).encode())
    assert status == 200
    assert ["stop", "chat-engine"] in runner.calls              # chat evicted
    assert ["start", "gen-engine"] in runner.calls
    assert runner.calls.index(["stop", "chat-engine"]) < \
        runner.calls.index(["start", "gen-engine"])
    # the job ran; its scope released (restore=True) restored the chat lane.
    assert comfy.run_calls and comfy.run_calls[0][0] == "qwen-image-2.1"
    assert stack["ledger"].is_resident(target, CHAT_MODEL_ID)
    assert not stack["ledger"].is_resident(target, "qwen-image-2.1")
    assert runner.calls.count(["start", "chat-engine"]) == 2    # restored


# --- COMFY_ARGS_FILE preflight (P5 follow-up: the silently-emptied --novram) -
#
# The h3 video models REQUIRE --novram on smaller cards; the args file
# (/data/broker/comfy_args.conf, bind-mounted at /etc/comfyui_args.conf) was
# silently emptied 2026-09-22 and every h3 run OOMd after. The startup
# preflight must WARN (never hard-fail) when the named file lacks the flag,
# note a missing file, and surface its verdict on GET /health.

from model_forge.facade import FacadeConfig, FacadeService  # noqa: E402
from model_forge.serve import preflight_comfy_args  # noqa: E402


def test_comfy_args_preflight_warns_when_novram_absent(tmp_path):
    f = tmp_path / "comfy_args.conf"
    f.write_text("--lowvram --preview-method auto")     # no --novram
    lines = []
    state = preflight_comfy_args(FacadeConfig(comfy_args_file=str(f)),
                                 log=lines.append)
    assert state == "novram-missing"
    (line,) = lines
    # honest WARNING naming the file AND the consequence
    assert "WARNING" in line and str(f) in line
    assert "--novram" in line and "OOM" in line


def test_comfy_args_preflight_notes_a_missing_file(tmp_path):
    f = tmp_path / "gone.conf"
    lines = []
    state = preflight_comfy_args(FacadeConfig(comfy_args_file=str(f)),
                                 log=lines.append)
    assert state == "unchecked"        # flag neither confirmed nor denied
    (line,) = lines
    assert "note" in line.lower() and str(f) in line   # honest note, no lie


def test_comfy_args_preflight_ok_state_and_unset_are_quiet(tmp_path):
    f = tmp_path / "comfy_args.conf"
    f.write_text("--novram")
    lines = []
    assert preflight_comfy_args(FacadeConfig(comfy_args_file=str(f)),
                                log=lines.append) == "novram-ok"
    assert lines and str(f) in lines[0]
    # nothing configured -> nothing checked, nothing logged
    lines = []
    assert preflight_comfy_args(FacadeConfig(), log=lines.append) == "unchecked"
    assert lines == []


def test_health_surfaces_comfy_args(tmp_path):
    f = tmp_path / "comfy_args.conf"
    f.write_text("--novram")
    probes = {"llm": lambda: True, "comfy": lambda: True, "mode": lambda: "gen"}
    svc = FacadeService(FacadeConfig(comfy_args_file=str(f)), probes=probes)
    assert svc.health()["comfy_args"] == "novram-ok"
    f.write_text("--lowvram")
    assert svc.health()["comfy_args"] == "novram-missing"   # live read
    svc = FacadeService(FacadeConfig(), probes=probes)
    assert svc.health()["comfy_args"] == "unchecked"


def test_resolve_settings_maps_comfy_args_file(tmp_path):
    config, _host, _port = resolve_settings(
        env={"COMFY_ARGS_FILE": "/data/broker/comfy_args.conf"})
    assert config.comfy_args_file == "/data/broker/comfy_args.conf"
    cfg = tmp_path / "facade.json"
    cfg.write_text(json.dumps({"COMFY_ARGS_FILE": "/etc/comfyui_args.conf"}))
    config, _host, _port = resolve_settings(env={}, config_file=str(cfg))
    assert config.comfy_args_file == "/etc/comfyui_args.conf"
    config, _host, _port = resolve_settings(env={})
    assert config.comfy_args_file == ""          # unset -> unchecked
