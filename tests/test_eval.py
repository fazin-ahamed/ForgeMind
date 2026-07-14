from pathlib import Path

import pytest

from forgemind.benchmark import BenchmarkRun, RuntimeCase
from forgemind.eval import (
    ControlledSystems,
    EvaluationRunner,
    freeze_results,
    load_cases,
    load_runs,
    parse_system_names,
    write_run,
)
from forgemind.context import assemble_evidence
from forgemind.domain import (
    AnswerDraft,
    Claim,
    GenerationResult,
    ReasoningLedger,
    SearchHit,
    SourceRecord,
)
from forgemind.store import ForgeStore


def proof_case() -> RuntimeCase:
    return RuntimeCase(
        id="c1",
        question="Why?",
        capability="repository",
        archive_band="32k",
        archive_id="repository-32k",
        archive_path="archives/repository-32k",
        archive_sha256="a" * 64,
        archived_tokens=32_000,
    )


def proof_run(
    system: str = "raw",
    case_id: str = "c1",
    error: str | None = None,
) -> BenchmarkRun:
    return BenchmarkRun(
        run_id=f"run-{system}",
        run_group_id="g1",
        system=system,
        case_id=case_id,
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
        error=error,
    )


def test_run_records_append_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    first = proof_run()
    second = first.model_copy(update={"system": "forgemind"})

    write_run(path, first)
    write_run(path, second)

    assert load_runs(path) == [first, second]


def test_runner_keeps_failed_runs_and_fixed_order() -> None:
    case = proof_case()

    def good(item: RuntimeCase) -> BenchmarkRun:
        return proof_run("good", item.id)

    def bad(item: RuntimeCase) -> BenchmarkRun:
        raise RuntimeError("boom")

    runner = EvaluationRunner(
        {"good": good, "bad": bad},
        error_factory=lambda system, item, error: proof_run(
            system, item.id, str(error)
        ),
    )
    runs = runner.run([case], ["bad", "good"])

    assert [run.system for run in runs] == ["bad", "good"]
    assert runs[0].error == "boom"


def test_runner_skips_completed_pairs_and_persists_each_new_run() -> None:
    case = proof_case()
    completed = proof_run("raw")
    vector = proof_run("vector")
    persisted: list[BenchmarkRun] = []
    runner = EvaluationRunner(
        {"raw": lambda item: completed, "vector": lambda item: vector},
        error_factory=lambda system, item, error: proof_run(
            system, item.id, str(error)
        ),
    )

    new = runner.run(
        [case],
        ["raw", "vector"],
        existing=[completed],
        on_run=persisted.append,
    )

    assert [item.system for item in new] == ["vector"]
    assert persisted == new


def test_vector_adapter_uses_runtime_evidence_without_gold_manifest(
    tmp_path: Path,
) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    source = SourceRecord.from_text("session.py", "UUID migration", 1)
    store.upsert_source(source)
    hit = SearchHit(
        "c1",
        source.id,
        source.sha256,
        source.path,
        1,
        1,
        source.text,
        1.0,
        ("semantic",),
    )

    class Retriever:
        def search_vector(self, query: str, limit: int = 20) -> list[SearchHit]:
            return [hit]

        def search(self, query: str, limit: int = 20) -> list[SearchHit]:
            return [hit]

    class Client:
        def complete(self, messages, max_tokens=None, json_schema=None) -> GenerationResult:
            assert "gold-secret" not in str(messages)
            return GenerationResult(
                '{"summary":"Migration","claims":[{"text":"UUID migration","evidence_ids":["c1"]}],"unresolved":[]}',
                10,
                5,
                1.0,
                2.0,
            )

    systems = ControlledSystems(
        store,
        Retriever(),
        controller=object(),
        client=Client(),
        count_tokens=lambda text: len(text.split()),
        vram_mib=lambda: 123,
        run_group_id="g1",
        model_sha256="b" * 64,
        config_sha256="c" * 64,
    )
    case = proof_case()

    run = systems.vector(case)

    assert run.system == "vector"
    assert run.answer == "Migration"
    assert [item.path for item in run.retrieved] == ["session.py"]
    assert [item.path for item in run.citations] == ["session.py"]
    assert systems.raw(case).system == "raw"
    assert systems.hybrid(case).system == "hybrid"


def test_proof_system_record_contains_exact_citations_and_usage(
    tmp_path: Path,
) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    source = SourceRecord.from_text("session.py", "UUID migration", 1)
    store.upsert_source(source)
    hit = SearchHit(
        "c1",
        source.id,
        source.sha256,
        source.path,
        1,
        1,
        source.text,
        1.0,
        ("semantic",),
    )

    class Retriever:
        def search_vector(self, query: str, limit: int = 20) -> list[SearchHit]:
            return [hit]

        def search(self, query: str, limit: int = 20) -> list[SearchHit]:
            return [hit]

    class Client:
        def complete(self, messages, max_tokens=None, json_schema=None) -> GenerationResult:
            return GenerationResult(
                '{"summary":"Migration","claims":[{"text":"UUID migration","evidence_ids":["c1"]}],"unresolved":[]}',
                10,
                5,
                1.0,
                2.0,
            )

    systems = ControlledSystems(
        store,
        Retriever(),
        controller=object(),
        client=Client(),
        count_tokens=lambda text: len(text.split()),
        vram_mib=lambda: 123,
        run_group_id="g1",
        model_sha256="b" * 64,
        config_sha256="c" * 64,
    )

    run = systems.vector(proof_case())

    assert run.answer == "Migration"
    assert run.citations[0].path == "session.py"
    assert run.citations[0].start_line == 1
    assert run.prompt_tokens == 10
    assert run.cumulative_prompt_tokens == 10
    assert run.completion_tokens == 5
    assert run.retrieval_cycles == 1


def test_forgemind_adapter_records_verified_controller_result(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    source = SourceRecord.from_text("session.py", "UUID migration", 1)
    store.upsert_source(source)
    hit = SearchHit(
        "c1",
        source.id,
        source.sha256,
        source.path,
        1,
        1,
        source.text,
        1.0,
        ("semantic",),
    )
    pack = assemble_evidence("Why?", [hit], lambda text: len(text.split()))

    class Controller:
        def investigate(self, question: str, mode: str):
            assert mode == "investigate"
            return (
                AnswerDraft(
                    summary="Migration",
                    claims=[Claim(text="UUID migration", evidence_ids=["c1"])],
                ),
                ReasoningLedger(goal=question, cycle=1, evidence_ids=["c1"]),
                [pack],
                [GenerationResult("{}", 11, 4, 1.0, 2.0)],
            )

    systems = ControlledSystems(
        store,
        retriever=object(),
        controller=Controller(),
        client=object(),
        count_tokens=lambda text: len(text.split()),
        vram_mib=lambda: 123,
        run_group_id="g1",
        model_sha256="b" * 64,
        config_sha256="c" * 64,
    )

    run = systems.forgemind(proof_case())

    assert run.system == "forgemind"
    assert run.answer == "Migration"
    assert run.prompt_tokens == 11


def test_freeze_requires_one_run_per_case_and_system(tmp_path: Path) -> None:
    case = proof_case()
    run = proof_run()
    freeze = tmp_path / "frozen"

    freeze_results(freeze, [case], [run], ["raw"])

    assert load_runs(freeze / "runs.jsonl") == [run]
    assert (freeze / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        freeze_results(freeze, [case], [run], ["raw"])


def test_case_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    case = proof_case().model_copy(update={"id": "same"})
    path.write_text(
        case.model_dump_json() + "\n" + case.model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_parse_system_names_rejects_unknown_and_duplicate_systems() -> None:
    assert parse_system_names("raw, vector,forgemind") == [
        "raw",
        "vector",
        "forgemind",
    ]
    with pytest.raises(ValueError, match="unknown"):
        parse_system_names("raw,magic")
    with pytest.raises(ValueError, match="duplicate"):
        parse_system_names("raw,raw")
