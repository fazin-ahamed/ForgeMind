from pathlib import Path

from forgemind.eval import (
    EvaluationRunner,
    EvalCase,
    GoldFact,
    RunRecord,
    load_runs,
    score_case,
    summarize,
    write_run,
)


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
