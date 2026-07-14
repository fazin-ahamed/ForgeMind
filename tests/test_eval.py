from pathlib import Path

import pytest

from forgemind.eval import (
    ControlledSystems,
    EvaluationRunner,
    EvalCase,
    GoldFact,
    RunRecord,
    load_runs,
    freeze_results,
    load_cases,
    parse_system_names,
    score_case,
    summarize,
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


def test_scoring_rewards_gold_facts_evidence_and_valid_citations() -> None:
    case = EvalCase(
        id="auth-1",
        question="Why did sessions fail?",
        evidence_paths=["migration.sql", "session.py"],
        facts=[GoldFact(id="f1", any_of=[["uuid", "parseint"]])],
    )
    run = RunRecord(
        system="forgemind",
        case_id="auth-1",
        claims=["UUID values were passed through parseInt."],
        cited_claims=[True],
        retrieved_paths=["migration.sql", "session.py"],
        abstained=False,
        active_tokens=8_000,
        latency_ms=20_000,
        peak_vram_mib=7_500,
    )

    metrics = score_case(case, run)

    assert metrics.factual_f1 == 1.0
    assert metrics.evidence_recall == 1.0
    assert metrics.citation_precision == 1.0


def test_answer_absent_case_rewards_abstention() -> None:
    case = EvalCase(
        id="absent-1",
        question="Unknown?",
        evidence_paths=[],
        facts=[],
        answer_absent=True,
    )
    run = RunRecord(
        system="forgemind",
        case_id="absent-1",
        claims=[],
        cited_claims=[],
        retrieved_paths=[],
        abstained=True,
        active_tokens=100,
        latency_ms=10,
        peak_vram_mib=100,
    )

    assert score_case(case, run).correct_abstention == 1.0


def test_summary_bootstrap_is_seeded() -> None:
    case = EvalCase(
        id="c1",
        question="q",
        evidence_paths=["a"],
        facts=[GoldFact(id="f", any_of=[["uuid"]])],
    )
    run = RunRecord(
        system="forgemind",
        case_id="c1",
        claims=["uuid"],
        cited_claims=[True],
        retrieved_paths=["a"],
        abstained=False,
        active_tokens=10,
        latency_ms=20,
        peak_vram_mib=30,
    )

    assert summarize([case], [run]) == summarize([case], [run])


def test_run_records_append_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    first = RunRecord(
        system="raw",
        case_id="c1",
        claims=[],
        cited_claims=[],
        retrieved_paths=[],
        abstained=True,
        active_tokens=1,
        latency_ms=2,
        peak_vram_mib=3,
    )
    second = first.model_copy(update={"system": "forgemind"})

    write_run(path, first)
    write_run(path, second)

    assert load_runs(path) == [first, second]


def test_runner_keeps_failed_runs_and_fixed_order() -> None:
    case = EvalCase(id="c1", question="q", evidence_paths=[], facts=[])

    def good(item: EvalCase) -> RunRecord:
        return RunRecord(
            system="good",
            case_id=item.id,
            claims=[],
            cited_claims=[],
            retrieved_paths=[],
            abstained=True,
            active_tokens=1,
            latency_ms=1,
            peak_vram_mib=1,
        )

    def bad(item: EvalCase) -> RunRecord:
        raise RuntimeError("boom")

    runs = EvaluationRunner({"good": good, "bad": bad}).run([case], ["bad", "good"])

    assert [run.system for run in runs] == ["bad", "good"]
    assert runs[0].error == "boom"


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
    )
    case = EvalCase(
        id="c1",
        question="Why?",
        evidence_paths=["gold-secret"],
        facts=[],
    )

    run = systems.vector(case)

    assert run.system == "vector"
    assert run.claims == ["UUID migration"]
    assert run.retrieved_paths == ["session.py"]
    assert run.cited_claims == [True]
    assert systems.raw(case).system == "raw"
    assert systems.hybrid(case).system == "hybrid"


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
            )

    systems = ControlledSystems(
        store,
        retriever=object(),
        controller=Controller(),
        client=object(),
        count_tokens=lambda text: len(text.split()),
        vram_mib=lambda: 123,
    )

    run = systems.forgemind(
        EvalCase(id="c1", question="Why?", evidence_paths=[], facts=[])
    )

    assert run.system == "forgemind"
    assert run.claims == ["UUID migration"]
    assert run.active_tokens == pack.active_tokens


def test_freeze_requires_one_run_per_case_and_system(tmp_path: Path) -> None:
    case = EvalCase(id="c1", question="q", evidence_paths=[], facts=[])
    run = RunRecord(
        system="raw",
        case_id="c1",
        claims=[],
        cited_claims=[],
        retrieved_paths=[],
        abstained=True,
        active_tokens=1,
        latency_ms=2,
        peak_vram_mib=3,
    )
    freeze = tmp_path / "frozen"

    freeze_results(freeze, [case], [run], ["raw"])

    assert load_runs(freeze / "runs.jsonl") == [run]
    assert (freeze / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        freeze_results(freeze, [case], [run], ["raw"])


def test_case_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    case = EvalCase(id="same", question="q", evidence_paths=[], facts=[])
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
