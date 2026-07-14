from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    gpu_name: str
    vram_mib: int
    driver_version: str
    ram_mib: int

