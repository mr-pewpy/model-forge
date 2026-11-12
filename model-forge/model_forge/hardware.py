"""Hardware probe: what accelerators and memory does this machine have.

Stdlib plus optional nvidia-smi / rocm-smi. Probes are advisory — engines do
their own authoritative sizing at launch. No probe raises into a caller.

Vendor coverage:
- NVIDIA: nvidia-smi CSV query (discrete VRAM per GPU).
- AMD: rocm-smi VRAM totals, falling back to amdgpu sysfs
  (/sys/class/drm/cardN/device/mem_info_vram_total, AMD vendor id 0x1002).
- Apple Silicon: unified memory — the GPU addresses the same pool as the CPU,
  so the honest VRAM budget is total RAM; the GPU is labeled as unified and
  platform reads "darwin-arm64" so consumers can tell this is not discrete VRAM.
- RAM: os.sysconf (POSIX) with a ctypes GlobalMemoryStatusEx fallback on
  Windows; 0.0 only when both fail.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess

from .registry import HardwareProfile

_NVIDIA_SMI = (
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader,nounits",
)
_ROCM_SMI_VRAM = ("rocm-smi", "--showmeminfo", "vram")
_SYS_DRM = "/sys/class/drm"  # amdgpu exposes mem_info_vram_total per card
_AMD_PCI_VENDOR = "0x1002"

_ROCM_TOTAL_RE = re.compile(r"GPU\[(\d+)\]\s*:.*VRAM Total Memory \(B\):\s*(\d+)")


def _probe_nvidia() -> tuple[dict, ...]:
    if shutil.which("nvidia-smi") is None:
        return ()
    try:
        out = subprocess.run(
            _NVIDIA_SMI, capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return ()
    gpus = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, mem = line.rpartition(",")
        try:
            mib = float(mem.strip())
        except ValueError:
            continue
        gpus.append({"name": name.strip(), "vram_gb": round(mib / 1024, 1)})
    return tuple(gpus)


def _probe_amd_rocm() -> tuple[dict, ...]:
    """AMD GPUs via rocm-smi; degrades silently when absent or unparseable."""
    if shutil.which("rocm-smi") is None:
        return ()
    try:
        out = subprocess.run(
            _ROCM_SMI_VRAM, capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return ()
    totals: dict[int, float] = {}
    for match in _ROCM_TOTAL_RE.finditer(out):
        idx, raw = int(match.group(1)), int(match.group(2))
        totals[idx] = round(raw / (1024**3), 1)
    return tuple(
        {"name": f"AMD GPU {idx} (ROCm)", "vram_gb": totals[idx]}
        for idx in sorted(totals)
    )


def _probe_amd_sysfs() -> tuple[dict, ...]:
    """AMD GPUs via amdgpu sysfs (no CLI needed); silent when not readable."""
    try:
        cards = sorted(
            (e for e in os.listdir(_SYS_DRM) if e.startswith("card") and e[4:].isdigit()),
            key=lambda e: int(e[4:]),
        )
    except OSError:
        return ()
    gpus = []
    for card in cards:
        dev = os.path.join(_SYS_DRM, card, "device")
        try:
            with open(os.path.join(dev, "vendor")) as fh:
                vendor = fh.read().strip().lower()
            with open(os.path.join(dev, "mem_info_vram_total")) as fh:
                raw = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        if vendor != _AMD_PCI_VENDOR:
            continue
        gpus.append({"name": f"AMD GPU ({card})", "vram_gb": round(raw / (1024**3), 1)})
    return tuple(gpus)


class _MemoryStatusEx(ctypes.Structure):
    """MEMORYSTATUSEX — fixed-width ints so the layout is right on every OS."""

    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _windows_total_ram_gb() -> float:
    """Total physical RAM via kernel32 GlobalMemoryStatusEx; raises on failure."""
    stat = _MemoryStatusEx()
    stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
    windll = getattr(ctypes, "windll")  # exists on Windows only; else raises
    if not windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise OSError("GlobalMemoryStatusEx failed")
    return round(stat.ullTotalPhys / (1024**3), 1)


def _ram_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 1)
    except (ValueError, OSError, AttributeError, NotImplementedError):
        pass
    try:
        return _windows_total_ram_gb()
    except (OSError, ValueError, AttributeError):
        return 0.0


def probe_hardware() -> HardwareProfile:
    system = platform.system().lower()
    machine = platform.machine().lower()
    apple_silicon = system == "darwin" and machine in ("arm64", "aarch64")
    if system == "darwin":
        system = "macos"
    probed = _probe_nvidia() + _probe_amd_rocm() + _probe_amd_sysfs()
    ram = _ram_gb()
    if apple_silicon:
        # Unified memory: the GPU addresses the same pool as the CPU, so the
        # honest VRAM budget is total RAM. The label says "unified" and the
        # platform is "darwin-arm64" so this is never mistaken for discrete VRAM.
        system = "darwin-arm64"
        gpus = ({"name": "Apple Silicon (unified memory)", "vram_gb": ram},) + probed
        total_vram = round(ram + sum(g.get("vram_gb", 0.0) for g in probed), 1)
    else:
        gpus = probed
        total_vram = round(sum(g.get("vram_gb", 0.0) for g in gpus), 1)
    return HardwareProfile(
        gpus=tuple(gpus),
        total_vram_gb=total_vram,
        ram_gb=ram,
        platform=system,
    )
