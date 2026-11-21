"""Bench VRAM-sampler tests (B3: non-NVIDIA samplers).

``action=bench`` books a MEASURED peak, so the sampler must observe real GPU
memory and never guess. These tests drive the vendor samplers with fakes (no
GPU, no CLI) and pin the vendor dispatch and the honest fail-soft reasons —
the AMD path (rocm-smi, falling back to the amdgpu sysfs) and the Apple
Silicon unified-memory honesty.
"""

from __future__ import annotations

import pytest

from model_forge import tools


# --- NVIDIA path -------------------------------------------------------------

def test_nvidia_vram_used_gb_sums_gpus(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(tools.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": "1024\n2048"})())
    assert tools.gpu_vram_used_gb() == 3072 / 1024.0


def test_nvidia_vram_used_gb_raises_on_no_readings(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(tools.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": ""})())
    with pytest.raises(ValueError):
        tools.gpu_vram_used_gb()


# --- AMD path (rocm-smi) -----------------------------------------------------

def test_amd_vram_used_gb_rocm_smi(monkeypatch):
    # rocm-smi present -> parse "VRAM Used Memory (B)" per GPU.
    monkeypatch.setattr(tools.shutil, "which",
                        lambda n: "/usr/bin/rocm-smi" if n == "rocm-smi" else None)
    out = ("GPU[0]\t\t: VRAM Total Memory (B): 8589934592\n"
           "GPU[0]\t\t: VRAM Used Memory (B): 2147483648\n"
           "GPU[1]\t\t: VRAM Used Memory (B): 1073741824\n")
    monkeypatch.setattr(tools.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": out})())
    # (2 GiB + 1 GiB) = 3 GiB
    assert tools._amd_vram_used_gb() == pytest.approx(3.0)


def test_amd_vram_used_gb_sysfs_fallback(monkeypatch, tmp_path):
    # No rocm-smi -> read mem_info_vram_used per AMD card; skip non-AMD cards.
    monkeypatch.setattr(tools.shutil, "which", lambda n: None)
    card0 = tmp_path / "card0" / "device"
    card1 = tmp_path / "card1" / "device"
    card0.mkdir(parents=True)
    card1.mkdir(parents=True)
    (card0 / "vendor").write_text("0x1002")
    (card0 / "mem_info_vram_used").write_text(str(3 * 1024 ** 3))
    (card1 / "vendor").write_text("0x10de")  # NVIDIA — skipped
    (card1 / "mem_info_vram_used").write_text(str(9 * 1024 ** 3))
    monkeypatch.setattr(tools, "_SYS_DRM", str(tmp_path))
    assert tools._amd_vram_used_gb() == pytest.approx(3.0)


def test_amd_vram_used_gb_raises_when_nothing_readable(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.shutil, "which", lambda n: None)
    monkeypatch.setattr(tools, "_SYS_DRM", str(tmp_path))
    with pytest.raises(ValueError):
        tools._amd_vram_used_gb()


# --- dispatch + honest reasons ----------------------------------------------

def test_gpu_vram_used_gb_dispatches_to_amd_when_no_nvidia(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda n: None)
    monkeypatch.setattr(tools, "_amd_vram_used_gb", lambda: 4.5)
    assert tools.gpu_vram_used_gb() == 4.5


def test_bench_sampler_available_nvidia_amd(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which",
                        lambda n: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None)
    assert tools._bench_sampler_available() is True
    monkeypatch.setattr(tools.shutil, "which",
                        lambda n: "/usr/bin/rocm-smi" if n == "rocm-smi" else None)
    assert tools._bench_sampler_available() is True


def test_bench_sampler_available_sysfs_card(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.shutil, "which", lambda n: None)
    (tmp_path / "card0").mkdir()
    monkeypatch.setattr(tools, "_SYS_DRM", str(tmp_path))
    assert tools._bench_sampler_available() is True


def test_bench_sampler_unavailable_apple_unified(monkeypatch, tmp_path):
    # No CLI and no drm cards -> Apple Silicon unified memory: honestly no
    # discrete VRAM-used counter, so the sampler is unavailable.
    monkeypatch.setattr(tools.shutil, "which", lambda n: None)
    monkeypatch.setattr(tools, "_SYS_DRM", str(tmp_path))
    assert tools._bench_sampler_available() is False


def test_live_bench_sampler_reason_names_vendors(monkeypatch):
    class _HW:
        total_vram_gb = 16.0

    monkeypatch.setattr(tools, "probe_hardware", lambda: _HW())
    monkeypatch.setattr(tools, "_bench_sampler_available", lambda: False)
    sampler, reason = tools._live_bench_sampler()
    assert sampler is None
    assert "nvidia-smi" in reason and "amdgpu" in reason
    assert "unified memory" in reason
