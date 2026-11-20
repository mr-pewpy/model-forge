"""ComfyUI workflow-runner tests (model_forge.engines.comfyui).

The runner ports the production weights broker's ComfyUI gen lane. Nothing here
touches the network: the runner's injectable ``http`` seam is replaced by a
scripted fake (and ``sleep`` by a recorder), so the whole submit/poll/collect
flow — including the 405 cold-boot retry, the history timeout and the upload
normalisation — runs deterministically in-process.

Covered, per the port contract:
  * the 7-lane registry + verbatim template copies;
  * param injection (explicit map wins; class fallback for seed/steps/prompt
    only; size -> width/height; unknown keys ignored; nested upload resolves);
  * 405 retry loop then success, and 405 exhaustion -> ok False;
  * upload normalisation writing the nested ``{"upload": {"path": ...}}`` shape;
  * history polling collecting image / video / 3d outputs;
  * free() posting /free; alive() fail-soft.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error

import pytest

from model_forge.engines import comfyui
from model_forge.engines.comfyui import (
    ComfyWorkflowRunner,
    WorkflowCfg,
    inject_params,
)
from model_forge.workflows import TEMPLATE_DIR, WORKFLOWS, load_templates

# sha256 of the broker's templates at port time — a byte-identical copy is the
# whole point (the graphs encode model wiring that must not drift).
_VERBATIM = {
    "h3_t2v.json": "81d3cac9d2be206a3a94e7cb8ebd74c0c0b89a417777eab45e26ee0dccc2dc44",
    "h3_i2v.json": "447dfdd80c3f6b98b820c50f48e864336b42bafc1b67d4cabe7f8a19f9fdfbae",
    "h3_r2v.json": "0462099a846a9b029a92d2dd5f317ebc86a21b11bc79fbe314872b90d1ef1019",
    "qwen21_t2i.json": "c6a7cd5f952a7ad025cf3ba3c04ac8f7a0a0dac53d14b1d83315e2144091eabf",
    "qwen21_edit.json": "f7be9e401edd5c2f7b2379b83546f1f73fc6e49c8cd8dda608a774715cd71d27",
    "qwen21_edit_nope.json": "ce4b6dc18e29ea4a5ff04f3b49729b1aa604552874510e94d543bf33bc782a32",
    "moss_image_t2i.json": "b45a4ef5af70863bdf7be0fcbf68505fa507d91308624cda762af90aa4f91b35",
}


# --- fakes -----------------------------------------------------------------

class FakeSleep:
    def __init__(self):
        self.sleeps: list[float] = []

    def __call__(self, seconds):
        self.sleeps.append(seconds)


class FakeHTTP:
    """Scripted ComfyUI. Dispatches on the URL path; records every call.

    ``prompt_script`` is consumed per ``/prompt`` call: each item is
    ``("405", None)`` (raise HTTPError 405), ``("exc", <Exception>)``,
    ``("body405", None)`` (200 with a 405-in-body), or ``("ok", {...})``. When
    the script runs out, ``/prompt`` answers ``{"prompt_id": "pid-default"}``.
    ``history`` may be a dict (returned for every history GET), a list (popped
    per call) or a callable(pid).
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.system_stats = {"system": {}, "devices": []}
        self.raise_stats = False
        self.prompt_script: list = []
        self.history = {}
        self.upload = {"name": "uploaded.png", "subfolder": "", "type": "input"}
        self.raise_upload = None
        self.free_ok = True
        self.raw_bytes = b"\x89PNG\r\n\x1a\n"

    # helpers for assertions
    def calls_to(self, suffix):
        return [c for c in self.calls if c["url"].endswith(suffix)]

    def paths(self):
        return [c["url"] for c in self.calls]

    def __call__(self, method, url, body=None, *, files=None, raw=False,
                 timeout=10.0):
        self.calls.append({"method": method, "url": url, "body": body,
                           "files": files, "raw": raw, "timeout": timeout})
        if raw:
            return self.raw_bytes
        if url.endswith("/system_stats"):
            if self.raise_stats:
                raise urllib.error.URLError("connection refused")
            return self.system_stats
        if url.endswith("/free"):
            if not self.free_ok:
                raise urllib.error.URLError("no /free")
            return {}
        if url.endswith("/upload/image"):
            if self.raise_upload:
                raise self.raise_upload
            return self.upload
        if "/history/" in url:
            pid = url.rsplit("/", 1)[1]
            h = self.history
            if callable(h):
                return h(pid)
            if isinstance(h, list):
                h = h.pop(0) if h else {}
            if not isinstance(h, dict):
                return {}
            # a full {pid: entry} response passes through; a bare entry (or {})
            # is wrapped under the requested pid, like real ComfyUI.
            return h if pid in h else {pid: h}
        if url.endswith("/prompt"):
            if not self.prompt_script:
                return {"prompt_id": "pid-default"}
            kind, val = self.prompt_script.pop(0)
            if kind == "405":
                raise urllib.error.HTTPError(url, 405, "Method Not Allowed",
                                             {}, None)
            if kind == "exc":
                raise val
            if kind == "body405":
                return {"error": "405 not ready"}
            return val
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)


def make_runner(http, sleep=None, **kw):
    return ComfyWorkflowRunner("http://comfy:18188", http=http,
                               sleep=sleep or FakeSleep(), **kw)


def _image_entry(filename="out.png", subfolder="", bucket="images", node="9"):
    return {"status": {"status_str": "success", "messages": []},
            "outputs": {node: {bucket: [
                {"filename": filename, "subfolder": subfolder,
                 "type": "output"}]}}}


# --- registry --------------------------------------------------------------

def test_registry_has_seven_lanes_with_expected_kinds():
    assert set(WORKFLOWS) == {
        "h3-t2v", "h3-i2v", "h3-r2v", "qwen-image-2.1",
        "qwen-image-2.1-edit", "qwen-image-2.1-cutout", "moss-image",
    }
    video = {"h3-t2v", "h3-i2v", "h3-r2v"}
    for wid, cfg in WORKFLOWS.items():
        assert isinstance(cfg, WorkflowCfg)
        assert cfg.workflow_id == wid
        assert cfg.output_kind == ("video" if wid in video else "image")


def test_load_templates_resolves_bundled_files():
    reg = load_templates()
    assert set(reg) == set(WORKFLOWS)
    for cfg in reg.values():
        assert os.path.isabs(cfg.template)
        assert os.path.isfile(cfg.template)


def test_templates_are_verbatim_copies():
    for name, digest in _VERBATIM.items():
        path = os.path.join(TEMPLATE_DIR, name)
        with open(path, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        assert got == digest, f"{name} drifted from the broker copy"


def test_load_templates_accepts_a_mapping_verbatim():
    mine = {"x": WorkflowCfg("x", "/tmp/t.json", "3d", {})}
    assert load_templates(mine) == mine
    assert load_templates(mine) is not mine  # a copy, not the same object


# --- param injection -------------------------------------------------------

def _graph(*nodes):
    return {nid: {"class_type": ct, "inputs": inputs}
            for nid, ct, inputs in nodes}


def test_inject_explicit_map_wins_over_class_fallback():
    # two nodes both expose a `seed` input; the explicit map must pick node
    # "8" while the class fallback would have picked the first one ("7").
    graph = _graph(
        ("7", "KSampler", {"seed": 1, "steps": 20}),
        ("8", "KSampler", {"seed": 1, "steps": 20}),
    )
    inject_params(graph, {"seed": {"node": "8", "field": "seed"}}, {"seed": 99})
    assert graph["8"]["inputs"]["seed"] == 99
    assert graph["7"]["inputs"]["seed"] == 1


def test_inject_class_fallback_only_for_seed_steps_prompt():
    graph = _graph(
        ("1", "CLIPTextEncode", {"text": "x"}),
        ("2", "KSampler", {"seed": 0, "steps": 0, "cfg": 7.0}),
    )
    inject_params(graph, {}, {"prompt": "a cat", "seed": 5, "steps": 30,
                              "cfg": 9.0, "mystery": "ignore me"})
    assert graph["1"]["inputs"]["text"] == "a cat"
    assert graph["2"]["inputs"]["seed"] == 5
    assert graph["2"]["inputs"]["steps"] == 30
    # cfg is NOT one of the three fallback params and is unmapped -> untouched
    assert graph["2"]["inputs"]["cfg"] == 7.0
    assert "mystery" not in graph["2"]["inputs"]


def test_inject_nested_upload_resolves_to_nested_path():
    # the injected value for an uploaded image is the NESTED dict; the field
    # must receive the bare filename (reading val["path"] was the prod bug).
    graph = _graph(("4", "LoadImage", {"image": ""}))
    inject_params(graph, {"image": {"node": "4", "field": "image"}},
                  {"image": {"upload": {"path": "abc_edit.png"}}})
    assert graph["4"]["inputs"]["image"] == "abc_edit.png"


def test_inject_size_maps_to_width_height(tmp_path):
    # real lane: qwen-image-2.1 maps width/height onto node 7
    r = make_runner(FakeHTTP())
    prepared = r._prepare_params({"size": "1024x576"}, str(tmp_path))
    graph = json.loads(json.dumps(comfyui.load_template_json(r.workflows["qwen-image-2.1"])))
    inject_params(graph, r.workflows["qwen-image-2.1"].param_map, prepared)
    assert graph["7"]["inputs"]["width"] == 1024
    assert graph["7"]["inputs"]["height"] == 576
    assert prepared["aspect_ratio"] == "16:9"


def test_prepare_params_drops_control_keys_and_maps_image_url(tmp_path):
    r = make_runner(FakeHTTP())
    # image_url is a bare http ref -> uploaded; model/n/_timeout dropped.
    http = r._http
    http.raw_bytes = b"jpgbytes"
    out = r._prepare_params(
        {"model": "x", "n": 1, "_timeout": 5, "prompt": "p",
         "image_url": "http://example/img.jpg"}, str(tmp_path))
    assert "model" not in out and "n" not in out and "_timeout" not in out
    assert "image_url" not in out
    assert out["image"] == {"upload": {"path": "uploaded.png"}}


# --- 405 cold-boot retry ---------------------------------------------------

def test_submit_retries_on_405_then_succeeds():
    http = FakeHTTP()
    http.prompt_script = [("405", None), ("405", None), ("ok", {"prompt_id": "p1"})]
    http.history = _image_entry(node="9")
    sleep = FakeSleep()
    r = make_runner(http, sleep, retry_sleep_s=10.0)
    res = r.run("moss-image", {"prompt": "x"}, assets_dir="/tmp/a", job_id="j1")
    assert res["ok"] is True and res["files"] == ["/output/out.png"]
    assert sleep.sleeps[:2] == [10.0, 10.0]      # two 405 retries slept 10 s
    assert len(http.calls_to("/prompt")) == 3


def test_submit_405_exhaustion_is_ok_false():
    http = FakeHTTP()
    http.prompt_script = [("405", None)] * 6     # default submit_retries == 6
    sleep = FakeSleep()
    r = make_runner(http, sleep)
    res = r.run("moss-image", {"prompt": "x"}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is False and res["files"] == []
    assert "405" in res["reason"] and "6 attempts" in res["reason"]
    assert len(sleep.sleeps) == 5                # six attempts -> five sleeps
    assert len(http.calls_to("/prompt")) == 6


def test_connection_refused_is_honest_no_retry():
    http = FakeHTTP()
    http.prompt_script = [("exc", urllib.error.URLError("connection refused"))]
    sleep = FakeSleep()
    r = make_runner(http, sleep)
    res = r.run("moss-image", {"prompt": "x"}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is False
    assert "container down" in res["reason"] and "unreachable" in res["reason"]
    assert sleep.sleeps == []                    # down != booting: no retry
    assert len(http.calls_to("/prompt")) == 1


# --- upload normalisation --------------------------------------------------

def test_upload_writes_nested_upload_path_shape(tmp_path):
    http = FakeHTTP()
    http.upload = {"name": "seed_edit.png", "subfolder": "", "type": "input"}
    r = make_runner(http)
    b64 = base64.b64encode(b"PNGDATA").decode()
    out = r._prepare_params({"image": f"data:image/png;base64,{b64}"},
                            str(tmp_path))
    # the contract every later wave reads: the NESTED shape.
    assert out["image"] == {"upload": {"path": "seed_edit.png"}}
    up = http.calls_to("/upload/image")
    assert len(up) == 1 and up[0]["files"]           # multipart file part sent
    assert up[0]["body"]["type"] == "input"


def test_upload_nested_shape_reaches_the_injected_graph(tmp_path):
    http = FakeHTTP()
    http.upload = {"name": "comfy_name.png", "subfolder": "", "type": "input"}
    http.history = _image_entry()
    r = make_runner(http)
    b64 = base64.b64encode(b"PNG").decode()
    res = r.run("qwen-image-2.1-edit",
                {"prompt": "edit", "image": f"data:image/png;base64,{b64}"},
                assets_dir=str(tmp_path), job_id="j")
    assert res["ok"] is True
    prompt_body = http.calls_to("/prompt")[0]["body"]
    # node 4 is the LoadImage node for that lane; it must carry the filename,
    # never the {"upload": {...}} dict.
    assert prompt_body["prompt"]["4"]["inputs"]["image"] == "comfy_name.png"
    # the source image was stashed under assets_dir/uploads
    assert os.path.isdir(str(tmp_path / "uploads"))


def test_remote_image_ref_is_fetched_and_uploaded(tmp_path):
    http = FakeHTTP()
    http.raw_bytes = b"JPEGBYTES"
    http.upload = {"name": "remote.jpeg", "subfolder": "", "type": "input"}
    r = make_runner(http)
    out = r._prepare_params({"image": "https://ex.test/a/photo.jpg?sig=1"},
                            str(tmp_path))
    assert out["image"] == {"upload": {"path": "remote.jpeg"}}
    # the fetch rode the seam with raw=True
    assert any(c["raw"] for c in http.calls)
    fname = http.calls_to("/upload/image")[0]["files"][0][1]
    assert fname.endswith(".jpg")                    # ext from the URL path


# --- history polling / output collection -----------------------------------

def test_history_poll_collects_image_output():
    http = FakeHTTP()
    http.history = _image_entry(filename="pic.png", subfolder="sub", node="10")
    r = make_runner(http)
    res = r.run("moss-image", {"prompt": "x"}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is True
    assert res["output_kind"] == "image"
    assert res["files"] == ["/output/sub/pic.png"]
    assert res["meta"]["outputs"][0]["node"] == "10"
    assert res["meta"]["prompt_id"] == "pid-default"


def test_history_poll_collects_video_output():
    http = FakeHTTP()
    http.history = _image_entry(filename="clip.mp4", subfolder="video",
                                bucket="videos", node="92")
    r = make_runner(http)
    res = r.run("h3-t2v", {"prompt": "x"}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is True and res["output_kind"] == "video"
    assert res["files"] == ["/output/video/clip.mp4"]


def test_history_poll_collects_video_legacy_results_bucket():
    http = FakeHTTP()
    http.history = _image_entry(filename="legacy.mp4", bucket="results",
                                node="92")
    r = make_runner(http)
    res = r.run("h3-t2v", {}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is True and res["files"] == ["/output/legacy.mp4"]


def test_history_poll_collects_3d_output():
    # no shipped lane uses output_kind "3d" yet; support it so a future lane
    # (SaveGLB -> ui "3d") needs no runner change.
    lanes = {"tri3d": WorkflowCfg(
        "tri3d", os.path.join(TEMPLATE_DIR, "moss_image_t2i.json"), "3d", {})}
    http = FakeHTTP()
    http.history = _image_entry(filename="mesh.glb", bucket="3d", node="10")
    r = ComfyWorkflowRunner("http://comfy:18188", http=http, sleep=FakeSleep(),
                            workflows=lanes)
    res = r.run("tri3d", {}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is True and res["output_kind"] == "3d"
    assert res["files"] == ["/output/mesh.glb"]


def test_history_poll_waits_until_outputs_appear():
    http = FakeHTTP()
    # first two polls: running (empty history); third: the output.
    http.history = [{}, {}, _image_entry(filename="late.png")]
    sleep = FakeSleep()
    r = make_runner(http, sleep, poll_interval_s=2.0)
    res = r.run("moss-image", {}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is True and res["files"] == ["/output/late.png"]
    assert len(http.calls_to("/history/pid-default")) == 3


def test_history_timeout_is_ok_false():
    http = FakeHTTP()
    http.history = {}                     # never completes
    sleep = FakeSleep()
    r = make_runner(http, sleep, poll_interval_s=1.0, history_timeout_s=3.0)
    res = r.run("moss-image", {}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is False and res["files"] == []
    assert "timed out" in res["reason"]
    assert len(http.calls_to("/history/pid-default")) == 3
    assert sleep.sleeps.count(1.0) == 3    # poll_interval each of the 3 polls


def test_run_request_timeout_clamps_below_history_budget():
    # a small request budget (the OpenAI _timeout) must cap the poll loop
    # WITHOUT waiting the instance's own history budget.
    http = FakeHTTP()
    http.history = {}                     # never completes
    sleep = FakeSleep()
    r = make_runner(http, sleep, poll_interval_s=1.0, history_timeout_s=3600.0)
    res = r.run("moss-image", {}, assets_dir="/tmp/a", job_id="j",
                timeout_s=3.0)
    assert res["ok"] is False and res["files"] == []
    assert "timed out" in res["reason"]
    # 3 polls at 1 s, not 3600 — the request budget won.
    assert len(http.calls_to("/history/pid-default")) == 3
    assert sleep.sleeps.count(1.0) == 3


def test_run_request_timeout_cannot_exceed_history_budget():
    # a budget larger than the instance ceiling is clamped to the ceiling.
    http = FakeHTTP()
    http.history = {}                     # never completes
    sleep = FakeSleep()
    r = make_runner(http, sleep, poll_interval_s=1.0, history_timeout_s=2.0)
    res = r.run("moss-image", {}, assets_dir="/tmp/a", job_id="j",
                timeout_s=9999.0)
    assert res["ok"] is False and "timed out" in res["reason"]
    assert len(http.calls_to("/history/pid-default")) == 2


def test_run_without_timeout_uses_history_budget():
    # timeout_s defaults to None -> the historical history_timeout_s budget is
    # unchanged (backward compatible).
    http = FakeHTTP()
    http.history = {}
    sleep = FakeSleep()
    r = make_runner(http, sleep, poll_interval_s=1.0, history_timeout_s=4.0)
    res = r.run("moss-image", {}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is False and "timed out" in res["reason"]
    assert len(http.calls_to("/history/pid-default")) == 4


def test_history_error_status_is_ok_false():
    http = FakeHTTP()
    http.history = {"pid-default": {"status": {
        "status_str": "error", "messages": ["node boom"]}}}
    r = make_runner(http)
    res = r.run("moss-image", {}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is False and "execution error" in res["reason"]


def test_unknown_workflow_is_ok_false():
    r = make_runner(FakeHTTP())
    res = r.run("nope", {}, assets_dir="/tmp/a", job_id="j")
    assert res["ok"] is False and "unknown workflow_id" in res["reason"]


# --- free / alive ----------------------------------------------------------

def test_free_posts_the_free_body():
    http = FakeHTTP()
    r = make_runner(http)
    out = r.free()
    assert out["ok"] is True and out["cache_freed"] is True
    call = http.calls_to("/free")[0]
    assert call["method"] == "POST"
    assert call["body"] == {"unload_models": True, "free_memory": True}


def test_run_frees_before_submitting():
    http = FakeHTTP()
    http.history = _image_entry()
    r = make_runner(http)
    r.run("moss-image", {}, assets_dir="/tmp/a", job_id="j")
    paths = http.paths()
    assert paths.index("http://comfy:18188/free") < \
        paths.index("http://comfy:18188/prompt")


def test_free_failure_is_soft():
    http = FakeHTTP()
    http.free_ok = False
    r = make_runner(http)
    out = r.free()
    assert out["ok"] is False and out["cache_freed"] is False
    assert "failed" in out["reason"]


def test_alive_true_on_stats():
    http = FakeHTTP()
    http.system_stats = {"system": {"comfyui-version": "0.3"}, "devices": []}
    assert make_runner(http).alive() is True


def test_alive_fail_soft_on_transport_error():
    http = FakeHTTP()
    http.raise_stats = True
    assert make_runner(http).alive() is False


def test_alive_false_on_unexpected_body():
    http = FakeHTTP()
    http.system_stats = {"hello": "world"}
    assert make_runner(http).alive() is False


# --- seam sanity -----------------------------------------------------------

def test_default_http_is_used_when_none_injected():
    r = ComfyWorkflowRunner("http://comfy:18188", sleep=FakeSleep())
    assert r._http is comfyui._default_http


def test_multipart_encoder_shapes_a_body():
    ctype, body = comfyui._encode_multipart(
        {"type": "input"}, [("image", "a.png", b"DATA", "image/png")])
    assert ctype.startswith("multipart/form-data; boundary=")
    assert b'name="image"; filename="a.png"' in body
    assert b"DATA" in body and b'name="type"' in body


# --- reviewer fixes (P5b-B1) -----------------------------------------------

def _lane(wid, template, kind, param_map):
    return WorkflowCfg(wid, os.path.join(TEMPLATE_DIR, template), kind, param_map)


def test_inject_coerces_numeric_string_seed_and_steps():
    # numeric strings are cleaned to ints before injection (seed/steps).
    r = make_runner(FakeHTTP())
    graph = json.loads(json.dumps(
        comfyui.load_template_json(r.workflows["moss-image"])))
    inject_params(graph, r.workflows["moss-image"].param_map,
                  {"seed": "123", "steps": "7"})
    assert graph["7"]["inputs"]["seed"] == 123
    assert graph["7"]["inputs"]["steps"] == 7


def test_run_never_raises_on_string_seed(tmp_path):
    # unmapped seed -> class fallback does int(value); a non-numeric string
    # raises ValueError, which run() must convert into an honest ok=False row.
    lane = _lane("seedlane", "moss_image_t2i.json", "image", {})
    http = FakeHTTP()
    r = ComfyWorkflowRunner("http://comfy:18188", http=http, sleep=FakeSleep(),
                            workflows={"seedlane": lane})
    res = r.run("seedlane", {"seed": "not-a-number"},
                assets_dir=str(tmp_path), job_id="j")
    assert res["ok"] is False and res["files"] == []
    assert "seedlane" in res["reason"] and "ValueError" in res["reason"]
    # the failure is pre-submit: no /prompt was ever sent.
    assert http.calls_to("/prompt") == []


def test_run_never_raises_on_param_map_missing_node(tmp_path):
    # a param_map naming an absent node raises KeyError inside injection.
    lane = _lane("badmap", "moss_image_t2i.json", "image",
                 {"prompt": {"node": "99999", "field": "text"}})
    http = FakeHTTP()
    r = ComfyWorkflowRunner("http://comfy:18188", http=http, sleep=FakeSleep(),
                            workflows={"badmap": lane})
    res = r.run("badmap", {"prompt": "hi"}, assets_dir=str(tmp_path), job_id="j")
    assert res["ok"] is False and res["files"] == []
    assert "badmap" in res["reason"] and "KeyError" in res["reason"]
    assert http.calls_to("/prompt") == []


def test_submit_masked_405_exhaustion_is_ok_false(tmp_path):
    # a 200 body carrying the 405 (proxy-masked) must exhaust to the same
    # "405 cold-boot: service not ready" reason as the HTTPError 405 branch.
    http = FakeHTTP()
    http.prompt_script = [("body405", None)] * 6   # default submit_retries == 6
    sleep = FakeSleep()
    r = make_runner(http, sleep)
    res = r.run("moss-image", {"prompt": "x"}, assets_dir=str(tmp_path), job_id="j")
    assert res["ok"] is False and res["files"] == []
    assert "405" in res["reason"] and "6 attempts" in res["reason"]
    assert "cold-boot" in res["reason"]
    assert len(sleep.sleeps) == 5                  # six attempts -> five sleeps
    assert len(http.calls_to("/prompt")) == 6


def test_collect_outputs_skips_items_missing_filename():
    r = make_runner(FakeHTTP())
    entry = {"outputs": {"9": {"images": [
        {"subfolder": "", "type": "output"},        # no filename -> not an output
        {"filename": "ok.png", "subfolder": "s", "type": "output"}]}}}
    items = r._collect_outputs(entry, "image")
    assert [i["filename"] for i in items] == ["ok.png"]
    assert items[0]["path"] == "/output/s/ok.png"


def test_history_item_without_filename_does_not_count_as_done(tmp_path):
    http = FakeHTTP()
    # first poll: a bucket item with no filename (must NOT read as done).
    # second poll: the real output.
    http.history = [{"status": {"status_str": "success", "messages": []},
                     "outputs": {"9": {"images": [
                         {"subfolder": "", "type": "output"}]}}},
                    _image_entry(filename="real.png")]
    r = make_runner(http, poll_interval_s=1.0, history_timeout_s=5.0)
    res = r.run("moss-image", {}, assets_dir=str(tmp_path), job_id="j")
    assert res["ok"] is True and res["files"] == ["/output/real.png"]


# --- video-param parity (P5b-B2) -------------------------------------------

def test_prepare_params_video_coerces_seconds_and_maps_size(tmp_path):
    # _openai_video_params parity: seconds -> float, size -> resolution, and
    # crucially NOT the image expansion (no width/height/aspect_ratio).
    r = make_runner(FakeHTTP())
    out = r._prepare_params(
        {"model": "h3-t2v", "seconds": "5", "size": "1024x576",
         "prompt": "p"}, str(tmp_path), "video")
    assert out["seconds"] == 5.0 and isinstance(out["seconds"], float)
    assert out["resolution"] == "1024x576"
    assert "width" not in out and "height" not in out
    assert "aspect_ratio" not in out
    assert "model" not in out
    assert out["prompt"] == "p"


def test_prepare_params_video_keeps_n_and_timeout(tmp_path):
    # video prep drops ONLY model (broker parity); n/_timeout ride through and
    # are simply unmapped by injection.
    r = make_runner(FakeHTTP())
    out = r._prepare_params({"model": "x", "n": 1, "_timeout": 5},
                            str(tmp_path), "video")
    assert "model" not in out and out.get("n") == 1 and out.get("_timeout") == 5


def test_prepare_params_image_default_kind_still_expands_size(tmp_path):
    # the default output_kind stays "image" — existing callers unchanged.
    r = make_runner(FakeHTTP())
    out = r._prepare_params({"size": "1024x576"}, str(tmp_path))
    assert out["aspect_ratio"] == "16:9"
    assert out["width"] == 1024 and out["height"] == 576


def test_run_video_lane_injects_seconds_onto_value_node(tmp_path):
    http = FakeHTTP()
    http.history = _image_entry(filename="clip.mp4", bucket="videos", node="92")
    r = make_runner(http)
    res = r.run("h3-t2v", {"prompt": "x", "seconds": "4", "size": "1024x576"},
                assets_dir=str(tmp_path), job_id="j")
    assert res["ok"] is True and res["output_kind"] == "video"
    graph = http.calls_to("/prompt")[0]["body"]["prompt"]
    # seconds -> node 111 value (the lane's PrimitiveFloat), prompt -> node 104
    assert graph["111"]["inputs"]["value"] == 4.0
    assert graph["104"]["inputs"]["prompt"] == "x"
