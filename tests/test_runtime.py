import subprocess

from forgemind.runtime import parse_nvidia_smi, physical_ram_mib, probe_hardware


def test_parse_nvidia_smi_returns_typed_hardware_profile() -> None:
    profile = parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288 MiB, 610.74\n", 32_563)

    assert profile.gpu_name == "NVIDIA GeForce RTX 3060"
    assert profile.vram_mib == 12_288
    assert profile.driver_version == "610.74"
    assert profile.ram_mib == 32_563


def test_physical_ram_probe_is_positive() -> None:
    assert physical_ram_mib() > 0


def test_probe_hardware_calls_nvidia_smi() -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "GPU, 1024 MiB, 1.0\n", "")

    profile = probe_hardware(run)

    assert calls[0][0] == "nvidia-smi"
    assert profile.gpu_name == "GPU"
