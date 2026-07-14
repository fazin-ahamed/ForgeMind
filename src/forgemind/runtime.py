from __future__ import annotations

import ctypes
import os
import subprocess
from collections.abc import Callable

from forgemind.domain import HardwareProfile


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def parse_nvidia_smi(text: str, ram_mib: int) -> HardwareProfile:
    fields = [field.strip() for field in text.strip().split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"unexpected nvidia-smi output: {text!r}")
    return HardwareProfile(fields[0], int(fields[1].removesuffix(" MiB")), fields[2], ram_mib)


def physical_ram_mib() -> int:
    if os.name != "nt":
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1_048_576)
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.total_physical / 1_048_576)


def probe_hardware(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> HardwareProfile:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return parse_nvidia_smi(completed.stdout.splitlines()[0], physical_ram_mib())
