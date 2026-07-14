import pytest

from forgemind.benchmark import (
    AnswerSpec,
    BenchmarkRun,
    CitationSpan,
    GoldCase,
    RuntimeCase,
    validate_bundle,
)


def runtime_case(case_id: str = "repo-32k-00") -> RuntimeCase:
    return RuntimeCase(
        id=case_id,
        question="Which function validates a session identifier?",
        capability="repository",
        archive_band="32k",
        archive_id="repository-32k",
        archive_path="archives/repository-32k",
        archive_sha256="a" * 64,
        archived_tokens=32_000,
    )


def gold_case(case_id: str = "repo-32k-00") -> GoldCase:
    return GoldCase(
        case_id=case_id,
        answer=AnswerSpec(kind="exact", accepted=["validate_session"]),
        evidence=[
            CitationSpan(
                source_id="s1",
                source_sha256="b" * 64,
                path="src/session.py",
                start_line=10,
                end_line=14,
            )
        ],
        source="unit-fixture",
        source_revision="fixture-v1",
    )


def test_runtime_record_cannot_contain_gold_fields() -> None:
    dumped = runtime_case().model_dump()

    assert "answer" not in dumped
    assert "evidence" not in dumped


def test_bundle_requires_matching_unique_ids() -> None:
    with pytest.raises(ValueError, match="runtime and gold IDs differ"):
        validate_bundle([runtime_case()], [gold_case("other")])


def test_bundle_checks_archive_band_token_count() -> None:
    bad = runtime_case().model_copy(update={"archived_tokens": 41_000})

    with pytest.raises(ValueError, match="outside 32k band"):
        validate_bundle([bad], [gold_case()])


def test_benchmark_run_retains_reproducibility_fields() -> None:
    run = BenchmarkRun(
        run_id="r1",
        run_group_id="g1",
        system="raw",
        case_id="repo-32k-00",
        answer=None,
        raw_outputs=[],
        citations=[],
        retrieved=[],
        retrieved_by_cycle=[],
        abstained=True,
        invalid_citations=0,
        prompt_tokens=15_616,
        cumulative_prompt_tokens=15_616,
        completion_tokens=0,
        retrieval_cycles=1,
        latency_ms=1,
        peak_vram_mib=1,
        model_sha256="c" * 64,
        config_sha256="d" * 64,
        started_at="2026-07-14T00:00:00+00:00",
        finished_at="2026-07-14T00:00:01+00:00",
    )

    assert run.prompt_tokens == 15_616
