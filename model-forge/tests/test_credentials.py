"""Credential routing + new-registry tests (pure; no secrets, no network)."""

from model_forge import credentials
from model_forge.registry import models_for


def test_token_env_routing():
    f = credentials.token_env_for
    assert f("https://huggingface.co/black-forest-labs/FLUX.2-klein-4B") \
        == "HF_TOKEN"
    assert f("https://cdn.huggingface.co/resolve/x") == "HF_TOKEN"
    assert f("https://www.modelscope.cn/models/x/resolve/master/y") \
        == "MODELSCOPE_API_TOKEN"
    assert f("https://modelscope.cn/api/v1/models") == "MODELSCOPE_API_TOKEN"
    # a token never leaks to other hosts
    assert f("https://example.com/x") == ""
    assert f("https://evil.example/huggingface.co") == ""


def test_auth_headers_without_secret_are_empty():
    # outside a Hermes process there is no secret scope -> no header at all
    assert credentials.auth_headers("https://huggingface.co/x") == {}
    st = credentials.status()
    assert set(st) == {"HF_TOKEN", "MODELSCOPE_API_TOKEN"}
    assert all(v in (True, False) for v in st.values())


def test_flux_entry_and_engine_widen():
    flux = next(m for m in models_for() if m.id == "flux-2-klein-4b")
    assert set(flux.engines) == {"sglang", "vllm", "comfyui"}
    assert flux.modality == "image_edit"
    # engine filter sees it under each engine family
    for eng in ("sglang", "vllm", "comfyui"):
        assert any(m.id == "flux-2-klein-4b"
                   for m in models_for(engine=eng)), eng
    # qwen edit lane rides vllm (omni) + comfyui now
    edit = next(m for m in models_for() if m.id == "qwen-image-2.1-edit")
    assert set(edit.engines) == {"vllm", "comfyui"}
