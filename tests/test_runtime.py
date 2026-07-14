import subprocess
from pathlib import Path

import pytest

from forgemind.config import RuntimeConfig
from forgemind.runtime import LlamaClient, LlamaServer, start_with_single_fallback
from forgemind.runtime import (
    parse_chat_response,
    parse_nvidia_smi,
    parse_tokenize_response,
    parse_used_vram_mib,
    physical_ram_mib,
    probe_hardware,
    windows_creation_flags,
)


def test_parse_nvidia_smi_returns_typed_hardware_profile() -> None:
    profile = parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288 MiB, 610.74\n", 32_563)

    assert profile.gpu_name == "NVIDIA GeForce RTX 3060"
    assert profile.vram_mib == 12_288
    assert profile.driver_version == "610.74"
    assert profile.ram_mib == 32_563


def test_physical_ram_probe_is_positive() -> None:
    assert physical_ram_mib() > 0


def test_creation_flags_are_platform_safe() -> None:
    class FakeSubprocess:
        CREATE_NO_WINDOW = 123

    assert windows_creation_flags("posix", FakeSubprocess()) == 0
    assert windows_creation_flags("nt", FakeSubprocess()) == 123
    assert windows_creation_flags("nt", object()) == 0


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


def test_parse_chat_response_preserves_usage_and_timings() -> None:
    result = parse_chat_response(
        {
            "choices": [{"message": {"content": "Root cause."}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 8},
            "timings": {"prompt_ms": 240.0, "predicted_ms": 160.0},
        }
    )

    assert result.text == "Root cause."
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 8
    assert result.total_ms == 400.0


def test_parse_tokenize_response_counts_tokens() -> None:
    assert parse_tokenize_response({"tokens": [1, 2, 3, 4]}) == 4


def test_parse_used_vram_uses_largest_visible_gpu() -> None:
    assert parse_used_vram_mib("7421\n128\n") == 7421


def test_llama_client_uses_tokenize_endpoint(tmp_path: Path, monkeypatch) -> None:
    sent: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"tokens": [10, 20, 30]}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            sent["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, json: dict[str, object]) -> FakeResponse:
            sent.update({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr("forgemind.runtime.httpx.Client", FakeClient)
    client = LlamaClient(RuntimeConfig(tmp_path / "server", tmp_path / "model"))

    assert client.count_tokens("hello") == 3
    assert sent["url"] == "http://127.0.0.1:8080/tokenize"
    assert sent["json"] == {"content": "hello"}


def test_llama_client_posts_deterministic_chat_request(tmp_path: Path, monkeypatch) -> None:
    sent: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            sent["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, json: dict[str, object]) -> FakeResponse:
            sent.update({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr("forgemind.runtime.httpx.Client", FakeClient)
    config = RuntimeConfig(tmp_path / "server", tmp_path / "model")

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    result = LlamaClient(config).complete(
        [{"role": "user", "content": "hi"}], max_tokens=7, json_schema=schema
    )

    assert result.text == "ok"
    assert sent["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert sent["json"] == {
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
        "max_tokens": 7,
        "cache_prompt": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "forgemind_response",
                "schema": schema,
                "strict": True,
            },
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }
