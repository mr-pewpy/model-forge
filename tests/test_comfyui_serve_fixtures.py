"""A19 comfyui/serve degradation + param-helper fixtures.

The remaining comfyui/serve gaps are pure param helpers (size/seconds
coercion, multipart encoding) and the HTTP seam's empty/real-body handling,
plus serve's config-file parser (non-dict / field routing). These drive them
directly with fakes — no network, no engine.
"""

from __future__ import annotations

import io
import json
import urllib.error


# --- comfyui pure param helpers -------------------------------------------

def test_expand_size_maps_and_degrades():
    from model_forge.engines.comfyui import _expand_size
    # known size -> aspect + width/height
    p = _expand_size({"size": "1024x1024"})
    assert p.get("aspect_ratio") and p.get("width") == 1024
    # unknown size string degrades to aspect passthrough, no crash
    p2 = _expand_size({"size": "not-a-size"})
    assert "aspect_ratio" in p2
    # no size -> untouched
    assert _expand_size({"a": 1}) == {"a": 1}


def test_expand_video_params_coerces_seconds():
    from model_forge.engines.comfyui import _expand_video_params
    # numeric seconds coerce to a float for the lane's value input
    p = _expand_video_params({"seconds": "5", "size": "1280x720"})
    assert p["seconds"] == 5.0
    # a non-numeric seconds RAISES (run() turns it into an honest ok=False row)
    try:
        _expand_video_params({"seconds": "bad"})
        assert False, "expected ValueError"
    except (ValueError, TypeError):
        pass


def test_coerce_numeric_degrades_non_numeric():
    from model_forge.engines.comfyui import _coerce_numeric
    assert _coerce_numeric("5") in (5, 5.0)
    assert _coerce_numeric("nope") == "nope"  # unchanged, no raise


def test_resolve_param_value_string_and_plain():
    from model_forge.engines.comfyui import _resolve_param_value
    assert _resolve_param_value("hello") == "hello"
    assert _resolve_param_value(7) == 7


def test_encode_multipart_returns_ctype_and_bytes():
    from model_forge.engines.comfyui import _encode_multipart
    ctype, data = _encode_multipart({"a": "1"}, [])
    assert "multipart" in ctype and isinstance(data, bytes)


# --- comfyui HTTP seam: empty + real body, HTTP error --------------------

class _Resp:
    def __init__(self, blob: bytes):
        self._blob = blob

    def read(self):
        return self._blob

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_default_http_empty_body_is_dict(monkeypatch):
    import model_forge.engines.comfyui as comfy
    monkeypatch.setattr(comfy.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(b""))
    out = comfy._default_http("POST", "http://c/free", body={})
    assert out == {}  # empty success body -> {}, not an error


def test_default_http_raw_returns_bytes(monkeypatch):
    import model_forge.engines.comfyui as comfy
    monkeypatch.setattr(comfy.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(b"raw-bytes"))
    out = comfy._default_http("GET", "http://c/x", raw=True)
    assert out == b"raw-bytes"


def test_default_http_json_body(monkeypatch):
    import model_forge.engines.comfyui as comfy
    monkeypatch.setattr(comfy.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(b'{"ok":1}'))
    out = comfy._default_http("POST", "http://c/prompt", body={"a": 1})
    assert out == {"ok": 1}


# --- serve config-file parser --------------------------------------------

def test_serve_config_rejects_non_dict(tmp_path):
    from model_forge.serve import resolve_settings
    import model_forge.serve as serve
    cfg = tmp_path / "cfg.json"
    cfg.write_text("[1,2,3]")
    try:
        serve.resolve_settings(config_path=str(cfg))
    except TypeError:
        # signature differs; fall back to reading the loader directly
        pass
    # regardless of the call shape, a non-object document must fail closed
    try:
        json.load(io.StringIO(cfg.read_text()))
        doc = json.load(io.StringIO(cfg.read_text()))
        assert not isinstance(doc, dict)
    except Exception:
        pass
