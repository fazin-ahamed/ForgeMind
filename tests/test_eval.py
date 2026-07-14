from forgemind.eval import EvalCase, GoldFact, RunRecord, score_case


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
