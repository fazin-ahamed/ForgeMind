import json
from dataclasses import replace
from pathlib import Path

import forgemind.cli as cli
from forgemind.cli import build_parser, main
from forgemind.domain import GenerationResult, HardwareProfile
from forgemind.domain import VerifiedAnswer
from forgemind.eval import EvalCase, RunRecord


def test_cli_parses_raw_question_and_context() -> None:
    args = build_parser().parse_args(
        ["ask-raw", "Why did auth fail?", "--context", "notes.txt"]
    )

    assert args.command == "ask-raw"
    assert args.question == "Why did auth fail?"
    assert args.context == "notes.txt"


def test_cli_parses_controlled_evaluation() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "cases.jsonl",
            "--db",
            "archive.sqlite",
            "--systems",
            "raw,vector,hybrid,forgemind",
            "--freeze",
            "benchmark-results",
        ]
    )

    assert args.command == "evaluate"
    assert args.systems == "raw,vector,hybrid,forgemind"
    assert args.freeze == "benchmark-results"


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


def test_archive_commands_do_not_require_llama_environment(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    database = tmp_path / "forge.sqlite"

    class FakeEmbedder:
        dimensions = 3

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 0.0, 1.0] for text in texts]

    monkeypatch.delenv("FORGEMIND_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("FORGEMIND_MODEL", raising=False)
    monkeypatch.setattr("forgemind.retrieval.Embedder", FakeEmbedder)

    assert main(["ingest", str(root), "--db", str(database)]) == 0
    assert json.loads(capsys.readouterr().out)["sources"] == 1
    assert main(["search", "run", "--db", str(database)]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["path"] == "app.py"


def test_offline_reasoning_smoke_command_needs_no_model_environment(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("FORGEMIND_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("FORGEMIND_MODEL", raising=False)

    assert main(["smoke", "--runs", "10", "--offline"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["completed"] == 10
    assert result["empty_evidence_abstained"] is True


def test_ask_command_returns_only_verified_service_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    server_path = tmp_path / "server.exe"
    model = tmp_path / "model.gguf"
    server_path.write_bytes(b"exe")
    model.write_bytes(b"model")
    monkeypatch.setenv("FORGEMIND_LLAMA_SERVER", str(server_path))
    monkeypatch.setenv("FORGEMIND_MODEL", str(model))

    class FakeServer:
        def __init__(self, config) -> None:
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeService:
        def ask(self, question: str, mode: str) -> VerifiedAnswer:
            return VerifiedAnswer(
                summary="Verified",
                claims=[],
                unresolved=[],
                cycles=1,
                status="supported",
            )

    monkeypatch.setattr(cli, "start_with_single_fallback", FakeServer)
    monkeypatch.setattr(cli, "_build_service", lambda config, db: FakeService(), raising=False)

    assert main(["ask", "why", "--db", str(tmp_path / "db.sqlite"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"] == "Verified"


def test_evaluate_freezes_every_requested_system(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    server_path = tmp_path / "server.exe"
    model = tmp_path / "model.gguf"
    server_path.write_bytes(b"exe")
    model.write_bytes(b"model")
    monkeypatch.setenv("FORGEMIND_LLAMA_SERVER", str(server_path))
    monkeypatch.setenv("FORGEMIND_MODEL", str(model))
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        EvalCase(id="c1", question="q", evidence_paths=[], facts=[]).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    class FakeServer:
        def __init__(self, config) -> None:
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class Systems:
        def raw(self, case: EvalCase) -> RunRecord:
            return RunRecord(
                system="raw",
                case_id=case.id,
                claims=[],
                cited_claims=[],
                retrieved_paths=[],
                abstained=True,
                active_tokens=1,
                latency_ms=2,
                peak_vram_mib=3,
            )

    monkeypatch.setattr(cli, "start_with_single_fallback", FakeServer)
    monkeypatch.setattr(
        cli, "_build_evaluation_systems", lambda config, db: Systems(), raising=False
    )
    freeze = tmp_path / "freeze"

    assert (
        main(
            [
                "evaluate",
                str(cases),
                "--db",
                str(tmp_path / "db.sqlite"),
                "--systems",
                "raw",
                "--freeze",
                str(freeze),
            ]
        )
        == 0
    )

    assert (freeze / "runs.jsonl").is_file()
    assert "raw" in json.loads(capsys.readouterr().out)["systems"]
