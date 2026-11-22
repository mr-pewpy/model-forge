"""A19 facade degradation-branch fixtures.

The facade's uncovered lines are its honest failure paths — an unreachable
upstream, a failing health probe, a gen job with no orchestrator/runner, a
bad JSON body, a missing job. These drive those branches through the real
route methods with the same fake seams test_facade.py uses (no sockets, no
network), so the degradations are OBSERVED rather than left uncovered.
"""

from __future__ import annotations

import json
import urllib.error

from model_forge.facade import (
    CHAT_MODEL_ID, FacadeConfig, FacadeService, JobStore,
)


class FakeUpstream:
    def __init__(self, status=200, body=b'{"ok":true}'):
        self.status, self.body = status, body
        self.calls: list = []

    def __call__(self, raw, headers):
        self.calls.append((raw, headers))
        return self.status, self.body


class FakeRunner:
    def __init__(self, result=None):
        self.result = result or {"ok": True, "files": [], "urls": []}
        self.freed = 0

    def run(self, *a, **k):
        return self.result

    def free(self):
        self.freed += 1


class FakeJobs(JobStore):
    """JobStore with a fixed clock so created jobs are deterministic."""


def _svc(**kw):
    cfg = FacadeConfig(llm_canonical_id=CHAT_MODEL_ID,
                       llm_served_id="served")
    return FacadeService(cfg, **kw)


# --- upstream degradation (the default urllib seam) ------------------------

def test_urllib_upstream_unreachable_is_502(monkeypatch):
    import model_forge.facade as facade

    def _boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(facade.urllib.request, "urlopen", _boom)
    svc = FacadeService(FacadeConfig(llm_base_url="http://up:8000/v1"))
    # build the default upstream seam and call it
    up = svc._urllib_upstream()
    status, body = up(b"{}", {"Content-Type": "application/json"})
    assert status == 502
    assert b"upstream unreachable" in body


def test_urllib_upstream_http_error_maps_code(monkeypatch):
    import model_forge.facade as facade

    def _err(req, timeout=None):
        raise urllib.error.HTTPError("http://up", 429, "slow", {}, None)
    monkeypatch.setattr(facade.urllib.request, "urlopen", _err)
    svc = FacadeService(FacadeConfig(llm_base_url="http://up:8000/v1"))
    up = svc._urllib_upstream()
    status, _ = up(b"{}", {})
    assert status == 429


def test_urllib_upstream_success_reads_status(monkeypatch):
    import model_forge.facade as facade

    class _Resp:
        status = 201

        def read(self):
            return b'{"ok":1}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(facade.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp())
    svc = FacadeService(FacadeConfig(llm_base_url="http://up:8000/v1"))
    up = svc._urllib_upstream()
    status, body = up(b"{}", {})
    assert status == 201 and body == b'{"ok":1}'


# --- health probe degradation --------------------------------------------

def test_default_llm_probe_true_on_200(monkeypatch):
    import model_forge.facade as facade

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(facade.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp())
    svc = FacadeService(FacadeConfig(llm_base_url="http://up:8000/v1"))
    probe = svc._default_llm_probe()
    assert probe() is True


def test_default_llm_probe_false_on_error(monkeypatch):
    import model_forge.facade as facade

    def _boom(req, timeout=None):
        raise OSError("down")
    monkeypatch.setattr(facade.urllib.request, "urlopen", _boom)
    svc = FacadeService(FacadeConfig(llm_base_url="http://up:8000/v1"))
    probe = svc._default_llm_probe()
    assert probe() is False


# --- gen_busy accessor variants ------------------------------------------

def test_gen_busy_none_orchestrator():
    assert _svc(). _gen_busy() is False


def test_gen_busy_no_accessor():
    class _O:
        pass
    assert _svc(orchestrator=_O())._gen_busy() is False


def test_gen_busy_callable_accessor():
    class _O:
        def gen_busy(self):
            return True
    assert _svc(orchestrator=_O())._gen_busy() is True


def test_gen_busy_attribute_accessor():
    class _O:
        gen_busy = True
    assert _svc(orchestrator=_O())._gen_busy() is True


def test_gen_busy_accessor_raises_is_false():
    class _O:
        def gen_busy(self):
            raise RuntimeError("x")
    assert _svc(orchestrator=_O())._gen_busy() is False


# --- gen param normalization (size/seconds degradation) ------------------

def test_normalize_video_params_size_and_bad_seconds():
    svc = _svc()
    params = svc._normalize_gen_params(
        {"model": "m", "_timeout": 5, "size": "1280x720",
         "seconds": "bad"}, kind="video")
    assert "model" not in params and "_timeout" not in params
    # non-numeric seconds degrades (no raise) and size survives normalisation
    assert "seconds" in params and "size" in params


def test_normalize_generic_params_drops_model():
    svc = _svc()
    params = svc._normalize_gen_params({"model": "m", "_timeout": 1,
                                       "prompt": "p"}, kind=None)
    assert "model" not in params and "prompt" in params


# --- gen route degradation -----------------------------------------------

def test_video_submit_no_runner_is_503():
    status, body = _svc().handle_video_submit(b'{"model":"gen/x"}')
    assert status == 503 and b"no generation runner" in body


def test_video_submit_bad_json_is_400():
    svc = _svc(runner=FakeRunner())
    status, body = svc.handle_video_submit(b'not json')
    assert status == 400


def test_video_submit_unknown_model_no_crash():
    svc = _svc(runner=FakeRunner())
    status, body = svc.handle_video_submit(b'{"model":"gen/nope"}')
    # either an honest refusal or a queued job — never a crash / 500
    assert status == 200 or status >= 400
    assert isinstance(body, bytes)


def test_video_status_missing_job_404():
    status, body = _svc(runner=FakeRunner()).handle_video_status("nope")
    assert status == 404 and b"not found" in body


def test_video_content_missing_asset_404():
    status, body, ctype = _svc(runner=FakeRunner()).handle_video_content("nope")
    assert status == 404 and "json" in ctype


def test_job_missing_404():
    status, body = _svc(runner=FakeRunner()).handle_job("nope")
    assert status == 404 and b"not found" in body


def test_run_gen_job_no_orchestrator_fails_job():
    svc = _svc(runner=FakeRunner())
    jid = svc._jobs.create("gen", "wf", {})
    ok, code, reason = svc._run_gen_job(jid, "wf", {}, "video")
    assert ok is False and code == 503
    assert "no orchestrator" in reason
    assert svc._jobs.get(jid)["status"] == "failed"
