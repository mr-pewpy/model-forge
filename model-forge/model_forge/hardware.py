"""Hardware probe: what accelerators and memory does this machine have.

Stdlib plus optional nvidia-smi. Probes are advisory — engines do their own
authoritative sizing at launch. No probe raises into a caller.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

from .registry import HardwareProfile

_NVIDIA_SMI = (
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader,nounits",
)


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


def _ram_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def probe_hardware() -> HardwareProfile:
    gpus = _probe_nvidia()
    system = platform.system().lower()
    if system == "darwin":
        system = "macos"
    return HardwareProfile(
        gpus=gpus,
        total_vram_gb=round(sum(g.get("vram_gb", 0.0) for g in gpus), 1),
        ram_gb=_ram_gb(),
        platform=system,
    )
