import hashlib

import pytest

import forgemind.benchmark as benchmark
from forgemind.benchmark import (
    AnswerSpec,
    BenchmarkRun,
    CitationSpan,
    GoldCase,
    RuntimeCase,
    validate_bundle,
)


def test_directory_hash_order_is_platform_independent(
    tmp_path, monkeypatch
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "a.txt").write_bytes(b"lower")
    (archive / "Z.txt").write_bytes(b"upper")
    monkeypatch.setattr(
        type(tmp_path),
        "__lt__",
        lambda self, other: str(self) < str(other),
    )
    expected = hashlib.sha256()
    expected.update(b"a.txt")
    expected.update(b"lower")
    expected.update(b"Z.txt")
    expected.update(b"upper")

    assert benchmark.sha256_path(archive) == expected.hexdigest()


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


def benchmark_run(
    answer: str | list[str] | None,
    citations: list[CitationSpan],
) -> BenchmarkRun:
    return BenchmarkRun(
        run_id="r1",
        run_group_id="g1",
        system="forgemind",
        case_id="repo-32k-00",
        answer=answer,
        raw_outputs=[],
        citations=citations,
        retrieved=citations,
        retrieved_by_cycle=[citations],
        abstained=False,
        invalid_citations=0,
        prompt_tokens=100,
        cumulative_prompt_tokens=100,
        completion_tokens=10,
        retrieval_cycles=1,
        latency_ms=20,
        peak_vram_mib=100,
        model_sha256="c" * 64,
        config_sha256="d" * 64,
        started_at="2026-07-14T00:00:00+00:00",
        finished_at="2026-07-14T00:00:01+00:00",
    )


def test_exact_answer_normalizes_unicode_and_case() -> None:
    metrics = benchmark.score_case(
        gold_case(), benchmark_run("VALIDATE_SESSION", gold_case().evidence)
    )

    assert metrics.answer_f1 == 1.0


def test_exact_answer_accepts_identifier_inside_explanation() -> None:
    metrics = benchmark.score_case(
        gold_case(),
        benchmark_run(
            "The matching function is `validate_session`.",
            gold_case().evidence,
        ),
    )

    assert metrics.answer_f1 == 1.0


def test_citation_overlap_uses_path_and_lines_across_hash_namespaces() -> None:
    runtime_span = gold_case().evidence[0].model_copy(
        update={
            "source_id": "c" * 64,
            "source_sha256": "d" * 64,
        }
    )

    metrics = benchmark.score_case(
        gold_case(), benchmark_run("validate_session", [runtime_span])
    )

    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 1.0
    assert metrics.retrieval_recall20 == 1.0


def test_whole_file_citation_loses_precision() -> None:
    broad = gold_case().evidence[0].model_copy(
        update={"start_line": 1, "end_line": 100}
    )

    metrics = benchmark.score_case(
        gold_case(), benchmark_run("validate_session", [broad])
    )

    assert metrics.citation_recall == 1.0
    assert metrics.citation_precision == pytest.approx(5 / 100)


def test_absent_answer_requires_abstention() -> None:
    gold = gold_case().model_copy(
        update={"answer": None, "answer_absent": True, "evidence": []}
    )
    run = benchmark_run(None, []).model_copy(update={"abstained": True})

    metrics = benchmark.score_case(gold, run)

    assert metrics.answer_f1 == 1.0
    assert metrics.correct_abstention == 1.0


def test_set_answer_parses_concise_text_and_fact_recall() -> None:
    gold = gold_case().model_copy(
        update={
            "answer": AnswerSpec(kind="set", accepted=["VALUE-1", "VALUE-2"]),
            "required_facts": ["VALUE-1", "VALUE-2"],
        }
    )

    metrics = benchmark.score_case(
        gold,
        benchmark_run("VALUE-1, VALUE-2", gold.evidence),
    )

    assert metrics.answer_f1 == 1.0
    assert metrics.fact_recall == 1.0


def test_set_answer_finds_values_inside_key_value_explanation() -> None:
    gold = gold_case().model_copy(
        update={
            "answer": AnswerSpec(kind="set", accepted=["VALUE-1", "VALUE-2"]),
        }
    )

    metrics = benchmark.score_case(
        gold,
        benchmark_run(
            "KEY-1 maps to VALUE-1, and KEY-2 maps to VALUE-2.",
            gold.evidence,
        ),
    )

    assert metrics.answer_f1 == 1.0


def test_summary_is_complete_and_paired() -> None:
    case = runtime_case()
    gold = gold_case()
    runs = [
        benchmark_run("wrong", []).model_copy(
            update={"run_id": f"r-{system}", "system": system}
        )
        for system in benchmark.SYSTEMS[:-1]
    ] + [benchmark_run("validate_session", gold.evidence)]

    summary = benchmark.summarize_benchmark([case], [gold], runs)

    assert summary["complete"] is True
    assert summary["systems"]["forgemind"]["answer_f1"] == 1.0
    assert summary["paired_intervals"]["hybrid"] == (1.0, 1.0)


def test_success_gates_require_five_points_and_positive_intervals() -> None:
    summary = {
        "systems": {
            "raw": {"answer_f1": 0.60, "abstention_f1": 0.70},
            "vector": {"answer_f1": 0.64, "abstention_f1": 0.75},
            "hybrid": {"answer_f1": 0.65, "abstention_f1": 0.80},
            "forgemind": {
                "answer_f1": 0.71,
                "citation_precision": 0.92,
                "citation_recall": 0.84,
                "citation_validity": 1.0,
                "abstention_f1": 0.80,
                "max_prompt_tokens": 15_000,
            },
        },
        "paired_intervals": {
            "raw": (0.05, 0.08),
            "vector": (0.02, 0.06),
            "hybrid": (0.01, 0.05),
        },
        "capability_wins": 3,
        "complete": True,
    }

    assert all(benchmark.success_gates(summary).values())


def test_finalize_rejects_missing_pairs(tmp_path) -> None:
    directory = tmp_path / "group"
    directory.mkdir()
    (directory / "runs.jsonl").write_text(
        benchmark_run(None, []).model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing or duplicate"):
        benchmark.finalize_run_group(
            directory,
            [runtime_case()],
            list(benchmark.SYSTEMS),
            provenance={},
        )


def test_finalize_hashes_complete_group_and_refuses_overwrite(tmp_path) -> None:
    directory = tmp_path / "group"
    directory.mkdir()
    run = benchmark_run(None, []).model_copy(update={"system": "raw"})
    (directory / "runs.jsonl").write_text(
        run.model_dump_json() + "\n", encoding="utf-8"
    )

    manifest = benchmark.finalize_run_group(
        directory,
        [runtime_case()],
        ["raw"],
        provenance={"source_revision": "abc"},
    )

    assert manifest["runs"] == 1
    assert len(manifest["runs_sha256"]) == 64
    with pytest.raises(FileExistsError):
        benchmark.finalize_run_group(
            directory,
            [runtime_case()],
            ["raw"],
            provenance={},
        )
