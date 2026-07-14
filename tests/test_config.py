from pathlib import Path

import pytest

from forgemind.config import RuntimeConfig


def test_runtime_config_reads_required_paths_and_pins_local_defaults(tmp_path: Path) -> None:
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "qwen3-4b-q4_k_m.gguf"
    server.write_bytes(b"exe")
    model.write_bytes(b"gguf")

    config = RuntimeConfig.from_env(
        {"FORGEMIND_LLAMA_SERVER": str(server), "FORGEMIND_MODEL": str(model)}
    )

    assert config.host == "127.0.0.1"
    assert config.context_tokens == 16_384
    assert config.server_url == "http://127.0.0.1:8080"


def test_runtime_config_rejects_missing_model(tmp_path: Path) -> None:
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"exe")

    with pytest.raises(ValueError, match="FORGEMIND_MODEL"):
        RuntimeConfig.from_env({"FORGEMIND_LLAMA_SERVER": str(server)})


def test_runtime_config_rejects_nonlocal_bind(tmp_path: Path) -> None:
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    server.write_bytes(b"exe")
    model.write_bytes(b"gguf")

    with pytest.raises(ValueError, match="local host"):
        RuntimeConfig.from_env(
            {
                "FORGEMIND_LLAMA_SERVER": str(server),
                "FORGEMIND_MODEL": str(model),
                "FORGEMIND_HOST": "0.0.0.0",
            }
        )
