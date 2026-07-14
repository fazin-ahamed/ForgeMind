import subprocess
from pathlib import Path

import pytest

from forgemind.config import RuntimeConfig
from forgemind.runtime import LlamaServer, start_with_single_fallback
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


def test_llama_server_command_pins_context_and_local_host(tmp_path: Path) -> None:
    config = RuntimeConfig(tmp_path / "llama-server.exe", tmp_path / "model.gguf")

    command = LlamaServer(config).command()

    assert command[:3] == [str(config.llama_server), "-m", str(config.model)]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("-c") + 1] == "16384"
    assert "--metrics" in command
    assert "--cache-prompt" in command


def test_allocation_failure_retries_once_with_safe_profile(tmp_path: Path) -> None:
    attempts: list[tuple[int, int]] = []

    class FakeServer:
        def __init__(self, config: RuntimeConfig) -> None:
            self.config = config

        def start(self) -> None:
            attempts.append((self.config.context_tokens, self.config.batch_tokens))
            if len(attempts) == 1:
                raise RuntimeError("CUDA out of memory")

        def stop(self) -> None:
            return None

    config = RuntimeConfig(tmp_path / "server", tmp_path / "model")
    server = start_with_single_fallback(config, factory=FakeServer)

    assert server.config.context_tokens == 8_192
    assert attempts == [(16_384, 512), (8_192, 256)]


def test_non_allocation_failure_is_not_hidden_by_retry(tmp_path: Path) -> None:
    attempts = 0

    class FakeServer:
        def __init__(self, config: RuntimeConfig) -> None:
            self.config = config

        def start(self) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("model file is corrupt")

        def stop(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="corrupt"):
        start_with_single_fallback(
            RuntimeConfig(tmp_path / "server", tmp_path / "model"), factory=FakeServer
        )
    assert attempts == 1
