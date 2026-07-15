import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import forgemind.cli as cli
import pytest
from forgemind.benchmark import (
    BAND_LIMITS,
    AnswerSpec,
    BenchmarkRun,
    GoldCase,
    RuntimeCase,
    sha256_path,
)
from forgemind.cli import build_parser, main
from forgemind.domain import GenerationResult, HardwareProfile
from forgemind.domain import VerifiedAnswer


def test_benchmark_provenance_uses_supplied_cloud_source_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGEMIND_SOURCE_REVISION", "modal-revision")
    monkeypatch.setenv("FORGEMIND_DIRTY_WORKTREE", "true")
    monkeypatch.setenv("FORGEMIND_RUNTIME_SHA256", "frozen-runtime-hash")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="git-revision\n"),
    )
    runtime = tmp_path / "runtime.jsonl"
    runtime.write_text("{}\n", encoding="utf-8")
    database_root = tmp_path / "databases"
    database_root.mkdir()
    config = cli.RuntimeConfig(
        llama_server=tmp_path / "llama-server",
        model=tmp_path / "model.gguf",
    )

    provenance = cli._benchmark_provenance(
        config,
        "model-hash",
        "config-hash",
        database_root,
        runtime,
        [],
    )

    assert provenance["source_revision"] == "modal-revision"
    assert provenance["dirty_worktree"] is True
    assert provenance["runtime_sha256"] == "frozen-runtime-hash"
    assert provenance["execution_runtime_sha256"] == sha256_path(runtime)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )


def _benchmark_matrix(tmp_path: Path) -> tuple[Path, Path, list[RuntimeCase]]:
    runtime: list[RuntimeCase] = []
    gold: list[GoldCase] = []
    for capability in (
        "repository",
        "memory",
        "effective-context",
        "adversarial",
    ):
        for band, (low, _high) in BAND_LIMITS.items():
            case_id = f"{capability}-{band}"
            archive = tmp_path / "archives" / case_id
            archive.mkdir(parents=True)
            (archive / "evidence.txt").write_text(case_id, encoding="utf-8")
            runtime.append(
                RuntimeCase(
                    id=case_id,
                    question="q",
                    capability=capability,
                    archive_band=band,
                    archive_id=case_id,
                    archive_path=str(archive),
                    archive_sha256=sha256_path(archive),
                    archived_tokens=low,
                )
            )
            gold.append(
                GoldCase(
                    case_id=case_id,
                    answer=AnswerSpec(kind="exact", accepted=["answer"]),
                    source="fixture",
                    source_revision="1",
                )
            )
    runtime_path = tmp_path / "runtime.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(runtime_path, runtime)
    _write_jsonl(gold_path, gold)
    return runtime_path, gold_path, runtime


def test_cli_parses_raw_question_and_context() -> None:
    args = build_parser().parse_args(
        ["ask-raw", "Why did auth fail?", "--context", "notes.txt"]
    )

    assert args.command == "ask-raw"
    assert args.question == "Why did auth fail?"
    assert args.context == "notes.txt"


def test_cli_parses_benchmark_workflow() -> None:
    validate = build_parser().parse_args(
        [
            "benchmark-validate",
            "runtime.jsonl",
            "gold.jsonl",
            "--expected-per-cell",
            "10",
            "--freeze",
            "manifest.json",
        ]
    )
    prepare = build_parser().parse_args(
        [
            "benchmark-prepare",
            "runtime.jsonl",
            "--db-root",
            "databases",
        ]
    )
    evaluate = build_parser().parse_args(
        [
            "evaluate",
            "runtime.jsonl",
            "--db-root",
            "databases",
            "--runs",
            "runs",
            "--run-group",
            "final-2026-08-03",
        ]
    )
    report = build_parser().parse_args(
        [
            "benchmark-report",
            "runtime.jsonl",
            "gold.jsonl",
            "--runs",
            "runs/runs.jsonl",
            "--output",
            "summary.json",
        ]
    )

    assert validate.command == "benchmark-validate"
    assert prepare.command == "benchmark-prepare"
    assert evaluate.run_group == "final-2026-08-03"
    assert report.command == "benchmark-report"


def test_benchmark_validation_freezes_exact_archive_hashes(tmp_path: Path) -> None:
    runtime_path, gold_path, _runtime = _benchmark_matrix(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    freeze = tmp_path / "benchmark-manifest.json"

    payload = cli.validate_benchmark(
        runtime_path,
        gold_path,
        expected_per_cell=1,
        freeze=freeze,
        model=model,
    )

    assert payload["archives"] == 16
    assert payload["model_sha256"] == sha256_path(model)
    assert freeze.is_file()


def test_installed_cli_validates_benchmark_outside_repository(tmp_path: Path) -> None:
    runtime_path, gold_path, _runtime = _benchmark_matrix(tmp_path)
    server = tmp_path / "server.exe"
    model = tmp_path / "model.gguf"
    server.write_bytes(b"server")
    model.write_bytes(b"model")
    environment = os.environ | {
        "FORGEMIND_LLAMA_SERVER": str(server),
        "FORGEMIND_MODEL": str(model),
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "forgemind",
            "benchmark-validate",
            str(runtime_path),
            str(gold_path),
            "--expected-per-cell",
            "1",
            "--freeze",
            str(tmp_path / "subprocess-manifest.json"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_benchmark_prepare_indexes_shared_archive_once_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    runtime_path, _gold_path, runtime = _benchmark_matrix(tmp_path)
    shared = runtime[0]
    repeated = shared.model_copy(update={"id": "second-question"})
    _write_jsonl(runtime_path, [shared, repeated])
    calls: list[Path] = []
    database_existed: list[bool] = []

    class FakeEmbedder:
        dimensions = 3

    class FakeStore:
        def __init__(self, database: Path) -> None:
            database_existed.append(database.exists())
            database.parent.mkdir(parents=True, exist_ok=True)
            database.write_bytes(b"sqlite")

        def enable_vectors(self, dimensions: int) -> None:
            assert dimensions == 3

        def close(self) -> None:
            return None

    def fake_ingest(root: Path, store, embedder) -> dict[str, int]:
        calls.append(root)
        return {"sources": 1, "chunks": 2, "events": 0}

    monkeypatch.setattr("forgemind.retrieval.Embedder", FakeEmbedder)
    monkeypatch.setattr("forgemind.store.ForgeStore", FakeStore)
    monkeypatch.setattr("forgemind.ingest.ingest_project", fake_ingest)
    databases = tmp_path / "databases"

    first = cli.prepare_benchmark(runtime_path, databases)
    second = cli.prepare_benchmark(runtime_path, databases)
    (databases / f"{shared.archive_id}.sqlite").write_bytes(b"damaged")
    third = cli.prepare_benchmark(runtime_path, databases)

    assert len(calls) == 2
    assert database_existed == [False, False]
    assert first[0]["status"] == "built"
    assert second[0]["status"] == "reused"
    assert third[0]["status"] == "built"


def test_prepared_database_validation_rejects_tampering(tmp_path: Path) -> None:
    _runtime_path, _gold_path, runtime = _benchmark_matrix(tmp_path)
    case = runtime[0]
    databases = tmp_path / "databases"
    databases.mkdir()
    database = databases / f"{case.archive_id}.sqlite"
    database.write_bytes(b"sqlite")
    (databases / f"{case.archive_id}.json").write_text(
        json.dumps(
            {
                "archive_sha256": case.archive_sha256,
                "embedder_revision": "BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                "database_sha256": sha256_path(database),
            }
        ),
        encoding="utf-8",
    )

    assert cli._prepared_database(case, databases) == database
    database.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="database hash"):
        cli._prepared_database(case, databases)


def test_benchmark_report_includes_index_costs(
    tmp_path: Path, monkeypatch
) -> None:
    runtime_path, gold_path, _runtime = _benchmark_matrix(tmp_path)
    run_root = tmp_path / "runs"
    run_root.mkdir()
    runs = run_root / "runs.jsonl"
    runs.write_text("", encoding="utf-8")
    (run_root / "run-manifest.json").write_text(
        json.dumps(
            {
                "runs_sha256": sha256_path(runs),
                "provenance": {
                    "runtime_sha256": sha256_path(runtime_path),
                    "indexing": [
                        {"ingest_seconds": 1.0, "database_bytes": 100},
                        {"ingest_seconds": 2.0, "database_bytes": 200},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "forgemind.benchmark.summarize_benchmark",
        lambda runtime, gold, records: {"complete": True},
    )
    monkeypatch.setattr(
        "forgemind.benchmark.success_gates",
        lambda summary: {"proof": True},
    )
    output = tmp_path / "summary.json"

    report = cli.report_benchmark(runtime_path, gold_path, runs, output)

    assert report["indexing"] == {
        "archives": 2,
        "total_seconds": 3.0,
        "median_seconds": 1.5,
        "database_bytes": 300,
    }
    assert report["gates"] == {"proof": True}


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
    archive = tmp_path / "archives" / "repository-32k"
    archive.mkdir(parents=True)
    (archive / "evidence.txt").write_text("evidence", encoding="utf-8")
    archive_sha256 = sha256_path(archive)
    second_archive = tmp_path / "archives" / "memory-32k"
    second_archive.mkdir(parents=True)
    (second_archive / "evidence.txt").write_text("memory", encoding="utf-8")
    second_archive_sha256 = sha256_path(second_archive)
    runtime_cases = [
        RuntimeCase(
            id="c1",
            question="q",
            capability="repository",
            archive_band="32k",
            archive_id="repository-32k",
            archive_path=str(archive),
            archive_sha256=archive_sha256,
            archived_tokens=32_000,
        ),
        RuntimeCase(
            id="c2",
            question="q2",
            capability="memory",
            archive_band="32k",
            archive_id="memory-32k",
            archive_path=str(second_archive),
            archive_sha256=second_archive_sha256,
            archived_tokens=32_000,
        ),
    ]
    cases.write_text(
        "".join(case.model_dump_json() + "\n" for case in runtime_cases),
        encoding="utf-8",
    )

    server_starts = 0

    class FakeServer:
        def __init__(self, config) -> None:
            nonlocal server_starts
            server_starts += 1
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class Systems:
        def raw(self, case: RuntimeCase) -> BenchmarkRun:
            return BenchmarkRun(
                run_id=f"r-{case.id}",
                run_group_id="g1",
                system="raw",
                case_id=case.id,
                answer=None,
                raw_outputs=[],
                citations=[],
                retrieved=[],
                retrieved_by_cycle=[],
                abstained=True,
                invalid_citations=0,
                prompt_tokens=1,
                cumulative_prompt_tokens=1,
                completion_tokens=0,
                retrieval_cycles=1,
                latency_ms=2,
                peak_vram_mib=3,
                model_sha256="b" * 64,
                config_sha256="c" * 64,
                started_at="2026-07-14T00:00:00+00:00",
                finished_at="2026-07-14T00:00:01+00:00",
            )

        def error_record(
            self, system: str, case: RuntimeCase, error: Exception
        ) -> BenchmarkRun:
            raise AssertionError(f"unexpected error: {system} {case.id} {error}")

    monkeypatch.setattr(cli, "start_with_single_fallback", FakeServer)
    monkeypatch.setattr(
        cli,
        "_build_evaluation_systems",
        lambda config, db, run_group, *hashes: Systems(),
        raising=False,
    )
    databases = tmp_path / "databases"
    databases.mkdir()
    database = databases / "repository-32k.sqlite"
    database.write_bytes(b"sqlite")
    (databases / "repository-32k.json").write_text(
        json.dumps(
            {
                "archive_sha256": archive_sha256,
                "embedder_revision": "BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                "database_sha256": sha256_path(database),
            }
        ),
        encoding="utf-8",
    )
    second_database = databases / "memory-32k.sqlite"
    second_database.write_bytes(b"sqlite-memory")
    (databases / "memory-32k.json").write_text(
        json.dumps(
            {
                "archive_sha256": second_archive_sha256,
                "embedder_revision": "BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                "database_sha256": sha256_path(second_database),
            }
        ),
        encoding="utf-8",
    )
    freeze = tmp_path / "freeze"
    monkeypatch.setattr(cli, "_benchmark_provenance", lambda *args: {})

    assert (
        main(
            [
                "evaluate",
                str(cases),
                "--db-root",
                str(databases),
                "--systems",
                "raw",
                "--runs",
                str(freeze),
                "--run-group",
                "g1",
            ]
        )
        == 0
    )

    assert (freeze / "runs.jsonl").is_file()
    assert (freeze / "run-manifest.json").is_file()
    assert server_starts == 2
    assert "raw" in json.loads(capsys.readouterr().out)["systems"]


def test_raw32_requires_explicit_32k_band(tmp_path: Path) -> None:
    server = tmp_path / "server.exe"
    model = tmp_path / "model.gguf"
    server.write_bytes(b"server")
    model.write_bytes(b"model")
    config = cli.RuntimeConfig(server, model)

    with pytest.raises(ValueError, match="raw32 requires"):
        cli.evaluate_benchmark(
            tmp_path / "runtime.jsonl",
            tmp_path / "databases",
            "raw32",
            tmp_path / "runs",
            "raw32-check",
            archive_band=None,
            config=config,
        )
