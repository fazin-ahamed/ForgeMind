import json
from dataclasses import replace
from pathlib import Path

import forgemind.cli as cli
from forgemind.cli import build_parser, main
from forgemind.domain import GenerationResult, HardwareProfile


def test_cli_parses_raw_question_and_context() -> None:
    args = build_parser().parse_args(
        ["ask-raw", "Why did auth fail?", "--context", "notes.txt"]
    )

    assert args.command == "ask-raw"
    assert args.question == "Why did auth fail?"
    assert args.context == "notes.txt"


def test_doctor_prints_hardware_and_runtime(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    server = tmp_path / "server.exe"
    model = tmp_path / "model.gguf"
    server.write_bytes(b"exe")
    model.write_bytes(b"model")
    monkeypatch.setenv("FORGEMIND_LLAMA_SERVER", str(server))
    monkeypatch.setenv("FORGEMIND_MODEL", str(model))
    monkeypatch.setattr(
        "forgemind.cli.probe_hardware",
        lambda: HardwareProfile("RTX 3060", 12_288, "610.74", 32_000),
    )

    assert main(["doctor"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["hardware"]["vram_mib"] == 12_288
    assert output["runtime"]["context_tokens"] == 16_384


def test_raw_ask_uses_effective_fallback_configuration(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    server_path = tmp_path / "server.exe"
    model = tmp_path / "model.gguf"
    context = tmp_path / "context.txt"
    server_path.write_bytes(b"exe")
    model.write_bytes(b"model")
    context.write_text("IDs changed to UUID.", encoding="utf-8")
    monkeypatch.setenv("FORGEMIND_LLAMA_SERVER", str(server_path))
    monkeypatch.setenv("FORGEMIND_MODEL", str(model))
    used: dict[str, object] = {}

    class FakeServer:
        def __init__(self, config) -> None:
            self.config = replace(config, port=9090, context_tokens=8_192)

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeClient:
        def __init__(self, config) -> None:
            used["port"] = config.port

        def complete(self, messages, max_tokens=None) -> GenerationResult:
            used["messages"] = messages
            return GenerationResult("UUID migration", 10, 2, 3.0, 4.0)

    monkeypatch.setattr(cli, "start_with_single_fallback", FakeServer, raising=False)
    monkeypatch.setattr(cli, "LlamaClient", FakeClient, raising=False)

    assert main(["ask-raw", "What changed?", "--context", str(context)]) == 0
    assert used["port"] == 9090
    assert json.loads(capsys.readouterr().out)["text"] == "UUID migration"
