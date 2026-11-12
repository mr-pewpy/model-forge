"""Hardware probe tests: no network, no real GPUs — subprocess/sysfs/platform
are monkeypatched so results are deterministic on any host."""

import dataclasses
import subprocess

from model_forge import hardware
from model_forge.registry import HardwareProfile, ModelEntry, fit_verdict

_PAGE = 4096
_GIB = 1024**3


# --- helpers ---------------------------------------------------------------

def _patch_platform(monkeypatch, system="Linux", machine="x86_64"):
    monkeypatch.setattr(hardware.platform, "system", lambda: system)
    monkeypatch.setattr(hardware.platform, "machine", lambda: machine)


def _patch_ram(monkeypatch, ram_gb):
    pages = int(ram_gb * _GIB) // _PAGE

    def sysconf(name):
        return {"SC_PHYS_PAGES": pages, "SC_PAGE_SIZE": _PAGE}[name]

    monkeypatch.setattr(hardware.os, "sysconf", sysconf)


def _patch_drm(monkeypatch, tmp_path):
    monkeypatch.setattr(hardware, "_SYS_DRM", str(tmp_path))


def _patch_which(monkeypatch, **available):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: available.get(name))


def _patch_run(monkeypatch, stdout="", returncode=0, raise_exc=None):
    def run(*args, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        return subprocess.CompletedProcess(
            args=args[0] if args else (), stdout=stdout,
            stderr="", returncode=returncode)

    monkeypatch.setattr(hardware.subprocess, "run", run)


def _make_card(drm_root, card, vendor, vram_bytes):
    dev = drm_root / card / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor + "\n")
    if vram_bytes is not None:
        (dev / "mem_info_vram_total").write_text(str(vram_bytes) + "\n")


class _FakeKernel32:
    """Stand-in for ctypes.windll.kernel32.GlobalMemoryStatusEx."""

    def __init__(self, total_bytes, ok=True):
        self._total_bytes = total_bytes
        self._ok = ok
        self.calls = 0

    def GlobalMemoryStatusEx(self, ref):
        self.calls += 1
        stat = ref._obj  # ctypes.byref exposes the struct as _obj
        assert stat.dwLength == hardware.ctypes.sizeof(hardware._MemoryStatusEx)
        if self._ok:
            stat.ullTotalPhys = self._total_bytes
        return 1 if self._ok else 0


def _patch_windll(monkeypatch, kernel32):
    fake = type("FakeWindll", (), {"kernel32": kernel32})()
    monkeypatch.setattr(hardware.ctypes, "windll", fake, raising=False)


# --- all-probes-absent degradation ----------------------------------------

def test_probe_hardware_all_probes_absent(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path)  # empty drm dir
    prof = hardware.probe_hardware()
    assert prof.gpus == ()
    assert prof.total_vram_gb == 0.0
    assert prof.platform == "linux"
    assert prof.ram_gb >= 0.0


def test_probe_hardware_broken_everything_still_returns(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch, **{"nvidia-smi": "/usr/bin/nvidia-smi",
                                 "rocm-smi": "/usr/bin/rocm-smi"})
    _patch_drm(monkeypatch, tmp_path / "does-not-exist")
    _patch_run(monkeypatch, raise_exc=OSError("exec failed"))
    prof = hardware.probe_hardware()
    assert isinstance(prof, HardwareProfile)
    assert prof.gpus == ()
    assert prof.total_vram_gb == 0.0


def test_return_shape_preserved(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path)
    prof = hardware.probe_hardware()
    names = {f.name for f in dataclasses.fields(HardwareProfile)}
    # strictly additive is allowed; the original fields must all remain
    assert {"gpus", "total_vram_gb", "ram_gb", "platform"} <= names
    for field in ("gpus", "total_vram_gb", "ram_gb", "platform"):
        assert getattr(prof, field) is not None


# --- NVIDIA (existing behavior) -------------------------------------------

def test_probe_hardware_parses_nvidia_multi_gpu(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch, **{"nvidia-smi": "/usr/bin/nvidia-smi"})
    _patch_drm(monkeypatch, tmp_path)
    _patch_run(monkeypatch, stdout="NVIDIA A100, 40960\nNVIDIA RTX 4090, 24576\n")
    prof = hardware.probe_hardware()
    assert [g["name"] for g in prof.gpus] == ["NVIDIA A100", "NVIDIA RTX 4090"]
    assert [g["vram_gb"] for g in prof.gpus] == [40.0, 24.0]
    assert prof.total_vram_gb == 64.0


def test_probe_hardware_bad_nvidia_lines(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch, **{"nvidia-smi": "/usr/bin/nvidia-smi"})
    _patch_drm(monkeypatch, tmp_path)
    _patch_run(monkeypatch, stdout="NotA,GPU,xx\nbad line without comma\n")
    prof = hardware.probe_hardware()
    assert prof.gpus == ()


def test_probe_hardware_nvidia_error(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch, **{"nvidia-smi": "/usr/bin/nvidia-smi"})
    _patch_drm(monkeypatch, tmp_path)
    _patch_run(monkeypatch, raise_exc=subprocess.SubprocessError("nvidia-smi died"))
    prof = hardware.probe_hardware()
    assert prof.gpus == ()


# --- AMD via rocm-smi ------------------------------------------------------

_ROCM_OUT = (
    "    GPU[0]          : VRAM Total Memory (B): 17179869184\n"
    "    GPU[0]          : VRAM Total Used Memory (B): 130715648\n"
    "    GPU[1]          : VRAM Total Memory (B): 34359738368\n"
    "    GPU[1]          : VRAM Total Used Memory (B): 23068672\n"
)


def test_probe_amd_rocm_parse(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch, **{"rocm-smi": "/opt/rocm/bin/rocm-smi"})
    _patch_drm(monkeypatch, tmp_path)
    _patch_run(monkeypatch, stdout=_ROCM_OUT)
    prof = hardware.probe_hardware()
    assert [g["name"] for g in prof.gpus] == ["AMD GPU 0 (ROCm)", "AMD GPU 1 (ROCm)"]
    assert [g["vram_gb"] for g in prof.gpus] == [16.0, 32.0]
    assert prof.total_vram_gb == 48.0


def test_probe_amd_rocm_error_degrades(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch, **{"rocm-smi": "/opt/rocm/bin/rocm-smi"})
    _patch_drm(monkeypatch, tmp_path)
    _patch_run(monkeypatch, raise_exc=subprocess.SubprocessError("rocm-smi died"))
    prof = hardware.probe_hardware()
    assert prof.gpus == ()
    assert prof.total_vram_gb == 0.0


# --- AMD via amdgpu sysfs --------------------------------------------------

def test_probe_amd_sysfs_parse(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch)  # no CLIs at all
    _make_card(tmp_path, "card0", "0x1002", 16 * _GIB)      # AMD
    _make_card(tmp_path, "card1", "0x10de", 8 * _GIB)       # non-AMD vendor
    _make_card(tmp_path, "card2", "0x1002", None)           # no VRAM attr
    (tmp_path / "card0-Virtual-1").mkdir()                  # connector, not a card
    _patch_drm(monkeypatch, tmp_path)
    prof = hardware.probe_hardware()
    assert prof.gpus == ({"name": "AMD GPU (card0)", "vram_gb": 16.0},)
    assert prof.total_vram_gb == 16.0


def test_probe_amd_sysfs_unreadable_degrades(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path / "not-mounted")
    prof = hardware.probe_hardware()
    assert prof.gpus == ()


def test_probe_hardware_nvidia_and_amd_order(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch, **{"nvidia-smi": "/usr/bin/nvidia-smi",
                                 "rocm-smi": "/opt/rocm/bin/rocm-smi"})
    _make_card(tmp_path, "card0", "0x1002", 4 * _GIB)
    _patch_drm(monkeypatch, tmp_path)

    def run(cmd, **kwargs):
        if cmd == hardware._NVIDIA_SMI:
            stdout = "NVIDIA A100, 40960\n"
        else:
            stdout = _ROCM_OUT
        return subprocess.CompletedProcess(args=cmd, stdout=stdout,
                                           stderr="", returncode=0)

    monkeypatch.setattr(hardware.subprocess, "run", run)
    prof = hardware.probe_hardware()
    names = [g["name"] for g in prof.gpus]
    assert names[0] == "NVIDIA A100"        # NVIDIA first (existing behavior)
    assert names[1:] == ["AMD GPU 0 (ROCm)", "AMD GPU 1 (ROCm)",
                         "AMD GPU (card0)"]
    assert prof.total_vram_gb == 40.0 + 48.0 + 4.0


# --- Apple Silicon unified memory -----------------------------------------

def test_apple_silicon_unified_memory(monkeypatch, tmp_path):
    _patch_platform(monkeypatch, system="Darwin", machine="arm64")
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path)
    _patch_ram(monkeypatch, 32.0)
    prof = hardware.probe_hardware()
    assert prof.platform == "darwin-arm64"
    assert prof.gpus == ({"name": "Apple Silicon (unified memory)",
                          "vram_gb": 32.0},)
    # memory fields stay coherent: VRAM budget IS the unified RAM pool,
    # labeled as such — never a fake discrete-VRAM number
    assert prof.total_vram_gb == 32.0
    assert prof.ram_gb == 32.0


def test_apple_silicon_downstream_not_needs_gpu(monkeypatch, tmp_path):
    _patch_platform(monkeypatch, system="Darwin", machine="arm64")
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path)
    _patch_ram(monkeypatch, 32.0)
    prof = hardware.probe_hardware()
    # registry.fit_verdict reads total_vram_gb == 0 for needs_gpu; a unified
    # Mac must never trip that branch (registry.py is not ours to edit)
    comfy = ModelEntry(id="probe-test", name="probe-test", modality="3d",
                       engines=(), source="probe-test",
                       size_gb=1.0, vram_gb_min=8.0)
    assert fit_verdict(comfy, prof) == "fits_gpu"
    huge = ModelEntry(id="probe-test-2", name="probe-test-2", modality="3d",
                      engines=(), source="probe-test-2",
                      size_gb=1.0, vram_gb_min=64.0)
    assert fit_verdict(huge, prof) != "needs_gpu"


def test_apple_intel_mac_no_unified_gpu(monkeypatch, tmp_path):
    _patch_platform(monkeypatch, system="Darwin", machine="x86_64")
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path)
    _patch_ram(monkeypatch, 16.0)
    prof = hardware.probe_hardware()
    assert prof.platform == "macos"
    assert prof.gpus == ()
    assert prof.total_vram_gb == 0.0


# --- RAM probing -----------------------------------------------------------

def test_ram_sysconf_normal(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path)
    _patch_ram(monkeypatch, 64.0)
    assert hardware.probe_hardware().ram_gb == 64.0


def test_ram_windows_fallback_attribute_error(monkeypatch, tmp_path):
    # Windows: os.sysconf does not exist at all -> AttributeError
    _patch_platform(monkeypatch, system="Windows", machine="AMD64")
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path)

    def no_sysconf(name):
        raise AttributeError("os has no attribute sysconf")

    monkeypatch.setattr(hardware.os, "sysconf", no_sysconf)
    kernel32 = _FakeKernel32(24 * _GIB)
    _patch_windll(monkeypatch, kernel32)
    prof = hardware.probe_hardware()
    assert kernel32.calls == 1
    assert prof.ram_gb == 24.0
    assert prof.platform == "windows"


def test_ram_windows_fallback_value_error(monkeypatch, tmp_path):
    # Windows: os.sysconf exists but the name is unknown -> ValueError
    _patch_platform(monkeypatch, system="Windows", machine="AMD64")
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path)

    def bad_sysconf(name):
        raise ValueError("unrecognized configuration variable")

    monkeypatch.setattr(hardware.os, "sysconf", bad_sysconf)
    kernel32 = _FakeKernel32(16 * _GIB)
    _patch_windll(monkeypatch, kernel32)
    assert hardware.probe_hardware().ram_gb == 16.0


def test_ram_both_probes_fail_zero(monkeypatch, tmp_path):
    _patch_platform(monkeypatch)
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path)

    def no_sysconf(name):
        raise AttributeError("no sysconf")

    monkeypatch.setattr(hardware.os, "sysconf", no_sysconf)

    def no_windows():
        raise AttributeError("no windll")

    monkeypatch.setattr(hardware, "_windows_total_ram_gb", no_windows)
    prof = hardware.probe_hardware()
    assert prof.ram_gb == 0.0  # honest 0.0, and it still returns a profile


def test_windows_memory_status_failure_zero(monkeypatch, tmp_path):
    _patch_platform(monkeypatch, system="Windows", machine="AMD64")
    _patch_which(monkeypatch)
    _patch_drm(monkeypatch, tmp_path)

    def bad_sysconf(name):
        raise ValueError("no sysconf")

    monkeypatch.setattr(hardware.os, "sysconf", bad_sysconf)
    _patch_windll(monkeypatch, _FakeKernel32(0, ok=False))
    assert hardware.probe_hardware().ram_gb == 0.0


def test_memory_status_layout():
    # 2 x uint32 + 7 x uint64 = 64 bytes, on every OS thanks to fixed widths
    assert hardware.ctypes.sizeof(hardware._MemoryStatusEx) == 64
    assert [f[0] for f in hardware._MemoryStatusEx._fields_][0] == "dwLength"
