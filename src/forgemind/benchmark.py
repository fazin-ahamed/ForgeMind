from __future__ import annotations

import hashlib
import json
import re
import stat
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from forgemind.domain import StrictModel


Capability = Literal["repository", "memory", "effective-context", "adversarial"]
ArchiveBand = Literal["32k", "100k", "250k", "1m"]
AnswerKind = Literal["exact", "set", "text"]
SYSTEMS = ("raw", "vector", "hybrid", "forgemind")
REPOQA_CODE_REVISION = "ae876deb1365dbf5a15b0533723c8ed123eee586"
REPOQA_DATA_VERSION = "2024-06-23"
LONGMEMEVAL_CODE_REVISION = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
LONGMEMEVAL_DATA_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
RULER_REVISION = "e8bbff677ca2c239640dc90f93310dcf32408c93"
BAND_LIMITS = {
    "32k": (28_000, 40_000),
    "100k": (90_000, 120_000),
    "250k": (225_000, 280_000),
    "1m": (900_000, 1_100_000),
}


class CitationSpan(StrictModel):
    source_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered_lines(self) -> CitationSpan:
        if self.end_line < self.start_line:
            raise ValueError("citation end_line precedes start_line")
        return self


class RuntimeCase(StrictModel):
    id: str
    question: str
    capability: Capability
    archive_band: ArchiveBand
    archive_id: str
    archive_path: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archived_tokens: int = Field(gt=0)
    input_budget: int = Field(default=15_616, ge=1, le=15_616)
    output_schema_version: Literal["1"] = "1"


class AnswerSpec(StrictModel):
    kind: AnswerKind
    accepted: list[str] = Field(min_length=1)
    case_sensitive: bool = False


class GoldCase(StrictModel):
    case_id: str
    answer: AnswerSpec | None = None
    evidence: list[CitationSpan] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    answer_absent: bool = False
    source: str
    source_revision: str
    audited: bool = False

    @model_validator(mode="after")
    def answer_matches_absence(self) -> GoldCase:
        if self.answer_absent == (self.answer is not None):
            raise ValueError(
                "answer must be present exactly when answer_absent is false"
            )
        return self


class BenchmarkRun(StrictModel):
    run_id: str
    run_group_id: str
    system: str
    case_id: str
    answer: str | list[str] | None
    raw_outputs: list[str]
    citations: list[CitationSpan]
    retrieved: list[CitationSpan]
    retrieved_by_cycle: list[list[CitationSpan]]
    abstained: bool
    invalid_citations: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0, le=32_000)
    cumulative_prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    retrieval_cycles: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    peak_vram_mib: int = Field(ge=0)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: str
    finished_at: str
    error: str | None = None


def load_runtime_cases(path: Path) -> list[RuntimeCase]:
    cases = [
        RuntimeCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError(f"manifest is empty: {path}")
    return cases


def load_gold_cases(path: Path) -> list[GoldCase]:
    cases = [
        GoldCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError(f"manifest is empty: {path}")
    return cases


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    children = (item for item in path.rglob("*") if item.is_file())
    for child in sorted(
        children,
        key=lambda item: (
            item.relative_to(path).as_posix().casefold(),
            item.relative_to(path).as_posix(),
        ),
    ):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(child.read_bytes())
    return digest.hexdigest()


def validate_bundle(
    runtime: list[RuntimeCase],
    gold: list[GoldCase],
    expected_per_cell: int | None = None,
) -> None:
    runtime_ids = [case.id for case in runtime]
    gold_ids = [case.case_id for case in gold]
    if len(runtime_ids) != len(set(runtime_ids)) or len(gold_ids) != len(
        set(gold_ids)
    ):
        raise ValueError("benchmark IDs must be unique")
    if set(runtime_ids) != set(gold_ids):
        raise ValueError("runtime and gold IDs differ")
    for case in runtime:
        low, high = BAND_LIMITS[case.archive_band]
        if not low <= case.archived_tokens <= high:
            raise ValueError(f"{case.id} is outside {case.archive_band} band")
    if expected_per_cell is None:
        return
    counts = Counter((case.capability, case.archive_band) for case in runtime)
    expected = Counter(
        {
            (capability, band): expected_per_cell
            for capability in (
                "repository",
                "memory",
                "effective-context",
                "adversarial",
            )
            for band in BAND_LIMITS
        }
    )
    if counts != expected:
        raise ValueError(f"benchmark matrix mismatch: {dict(counts)}")


class CaseMetrics(StrictModel):
    answer_f1: float
    fact_recall: float
    citation_precision: float
    citation_recall: float
    citation_validity: float
    retrieval_recall20: float
    correct_abstention: float
    unsupported_answer: float


def _normalized(text: str, case_sensitive: bool = False) -> str:
    value = unicodedata.normalize("NFKC", text)
    if not case_sensitive:
        value = value.casefold()
    return " ".join(re.findall(r"[\w./:-]+", value, flags=re.UNICODE))


def _token_f1(prediction: str, reference: str) -> float:
    predicted = _normalized(prediction).split()
    expected = _normalized(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _contains_answer(
    prediction: str, expected: str, case_sensitive: bool
) -> bool:
    predicted = [
        token.strip(".,;:")
        for token in _normalized(prediction, case_sensitive).split()
    ]
    target = [
        token.strip(".,;:")
        for token in _normalized(expected, case_sensitive).split()
    ]
    return bool(target) and any(
        predicted[index : index + len(target)] == target
        for index in range(len(predicted) - len(target) + 1)
    )


def _answer_f1(answer: AnswerSpec, prediction: str | list[str] | None) -> float:
    if prediction is None:
        return 0.0
    if answer.kind == "set":
        if isinstance(prediction, str):
            prediction = [
                item
                for item in answer.accepted
                if _contains_answer(
                    prediction, item, answer.case_sensitive
                )
            ]
        expected = {
            _normalized(item, answer.case_sensitive) for item in answer.accepted
        }
        actual = {
            _normalized(item, answer.case_sensitive) for item in prediction
        }
        if not expected or not actual:
            return float(expected == actual)
        overlap = len(expected & actual)
        if not overlap:
            return 0.0
        precision = overlap / len(actual)
        recall = overlap / len(expected)
        return 2 * precision * recall / (precision + recall)
    if isinstance(prediction, list):
        return 0.0
    if answer.kind == "exact":
        return float(
            any(
                _contains_answer(
                    prediction, item, answer.case_sensitive
                )
                for item in answer.accepted
            )
        )
    return max(_token_f1(prediction, reference) for reference in answer.accepted)


def _lines(spans: list[CitationSpan]) -> set[tuple[str, int]]:
    return {
        (span.path, line)
        for span in spans
        for line in range(span.start_line, span.end_line + 1)
    }


def score_case(gold: GoldCase, run: BenchmarkRun) -> CaseMetrics:
    answer_f1 = (
        float(run.abstained)
        if gold.answer_absent
        else _answer_f1(gold.answer, run.answer)
        if gold.answer is not None
        else 0.0
    )
    cited = _lines(run.citations)
    expected = _lines(gold.evidence)
    citation_precision = (
        len(cited & expected) / len(cited) if cited else float(not expected)
    )
    citation_recall = (
        len(cited & expected) / len(expected) if expected else float(not cited)
    )
    retrieved = _lines(run.retrieved[:20])
    retrieval_recall = (
        len(retrieved & expected) / len(expected) if expected else 1.0
    )
    supported = bool(cited & expected) or gold.answer_absent
    answer_text = (
        " ".join(run.answer)
        if isinstance(run.answer, list)
        else run.answer or ""
    )
    fact_recall = (
        sum(
            _normalized(fact) in _normalized(answer_text)
            for fact in gold.required_facts
        )
        / len(gold.required_facts)
        if gold.required_facts
        else 1.0
    )
    return CaseMetrics(
        answer_f1=answer_f1,
        fact_recall=fact_recall,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        citation_validity=float(run.invalid_citations == 0),
        retrieval_recall20=retrieval_recall,
        correct_abstention=float(gold.answer_absent and run.abstained),
        unsupported_answer=float(not run.abstained and not supported),
    )


def paired_interval(
    forge: list[float], baseline: list[float], seed: int = 20_260_714
) -> tuple[float, float]:
    if len(forge) != len(baseline) or not forge:
        raise ValueError("paired intervals require equal non-empty samples")
    differences = np.asarray(forge, dtype=float) - np.asarray(
        baseline, dtype=float
    )
    rng = np.random.default_rng(seed)
    means = rng.choice(
        differences, (10_000, len(differences)), replace=True
    ).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _abstention_f1(gold: list[GoldCase], runs: list[BenchmarkRun]) -> float:
    by_id = {item.case_id: item for item in gold}
    true_positive = sum(
        by_id[run.case_id].answer_absent and run.abstained for run in runs
    )
    false_positive = sum(
        not by_id[run.case_id].answer_absent and run.abstained for run in runs
    )
    false_negative = sum(
        by_id[run.case_id].answer_absent and not run.abstained for run in runs
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def summarize_benchmark(
    runtime: list[RuntimeCase],
    gold: list[GoldCase],
    runs: list[BenchmarkRun],
) -> dict[str, Any]:
    validate_bundle(runtime, gold)
    expected = {(case.id, system) for case in runtime for system in SYSTEMS}
    actual = [
        (run.case_id, run.system) for run in runs if run.system in SYSTEMS
    ]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("benchmark has missing or duplicate primary runs")
    gold_by_id = {case.case_id: case for case in gold}
    run_by_pair = {
        (run.case_id, run.system): run
        for run in runs
        if run.system in SYSTEMS
    }
    score_by_pair = {
        pair: score_case(gold_by_id[pair[0]], run)
        for pair, run in run_by_pair.items()
    }
    systems: dict[str, dict[str, float | int]] = {}
    ordered_ids = [case.id for case in runtime]
    for system in SYSTEMS:
        system_runs = [run_by_pair[(case_id, system)] for case_id in ordered_ids]
        metrics = [score_by_pair[(case_id, system)] for case_id in ordered_ids]
        systems[system] = {
            "answer_f1": float(np.mean([item.answer_f1 for item in metrics])),
            "fact_recall": float(np.mean([item.fact_recall for item in metrics])),
            "citation_precision": float(
                np.mean([item.citation_precision for item in metrics])
            ),
            "citation_recall": float(
                np.mean([item.citation_recall for item in metrics])
            ),
            "citation_validity": float(
                np.mean([item.citation_validity for item in metrics])
            ),
            "retrieval_recall20": float(
                np.mean([item.retrieval_recall20 for item in metrics])
            ),
            "abstention_f1": _abstention_f1(gold, system_runs),
            "unsupported_answer_rate": float(
                np.mean([item.unsupported_answer for item in metrics])
            ),
            "median_latency_ms": float(
                np.median([item.latency_ms for item in system_runs])
            ),
            "p95_latency_ms": float(
                np.quantile([item.latency_ms for item in system_runs], 0.95)
            ),
            "max_prompt_tokens": max(
                item.prompt_tokens for item in system_runs
            ),
            "median_cumulative_prompt_tokens": float(
                np.median(
                    [item.cumulative_prompt_tokens for item in system_runs]
                )
            ),
            "peak_vram_mib": max(item.peak_vram_mib for item in system_runs),
            "errors": sum(item.error is not None for item in system_runs),
            "malformed_outputs": sum(
                item.error is not None
                and "validation" in item.error.casefold()
                for item in system_runs
            ),
        }
    forge_values = [
        score_by_pair[(case_id, "forgemind")].answer_f1
        for case_id in ordered_ids
    ]
    intervals = {
        baseline: paired_interval(
            forge_values,
            [
                score_by_pair[(case_id, baseline)].answer_f1
                for case_id in ordered_ids
            ],
        )
        for baseline in SYSTEMS[:-1]
    }
    capabilities: dict[str, dict[str, float]] = {}
    capability_wins = 0
    for capability in dict.fromkeys(case.capability for case in runtime):
        ids = [case.id for case in runtime if case.capability == capability]
        row = {
            system: float(
                np.mean(
                    [
                        score_by_pair[(case_id, system)].answer_f1
                        for case_id in ids
                    ]
                )
            )
            for system in SYSTEMS
        }
        capabilities[capability] = row
        capability_wins += int(
            row["forgemind"] > max(row[name] for name in SYSTEMS[:-1])
        )
    return {
        "systems": systems,
        "paired_intervals": intervals,
        "capabilities": capabilities,
        "capability_wins": capability_wins,
        "complete": True,
        "cases": len(runtime),
        "runs": len(actual),
    }


def success_gates(summary: dict[str, Any]) -> dict[str, bool]:
    systems = summary["systems"]
    forge = systems["forgemind"]
    best_baseline = max(
        systems[name]["answer_f1"] for name in SYSTEMS[:-1]
    )
    intervals = summary["paired_intervals"]
    best_baseline_abstention = max(
        systems[name]["abstention_f1"] for name in SYSTEMS[:-1]
    )
    return {
        "answer_gain": forge["answer_f1"] - best_baseline >= 0.05,
        "positive_intervals": all(
            intervals[name][0] > 0 for name in SYSTEMS[:-1]
        ),
        "capability_wins": summary["capability_wins"] >= 3,
        "citation_precision": forge["citation_precision"] >= 0.90,
        "citation_recall": forge["citation_recall"] >= 0.80,
        "citation_validity": forge["citation_validity"] == 1.0,
        "abstention": forge["abstention_f1"] >= best_baseline_abstention,
        "context": forge["max_prompt_tokens"] <= 15_616,
        "complete": bool(summary["complete"]),
    }


def finalize_run_group(
    directory: Path,
    cases: list[RuntimeCase],
    systems: list[str],
    provenance: dict[str, object],
) -> dict[str, object]:
    manifest_path = directory / "run-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"run group is already frozen: {directory}")
    run_path = directory / "runs.jsonl"
    runs = [
        BenchmarkRun.model_validate_json(line)
        for line in run_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {(case.id, system) for case in cases for system in systems}
    actual = [(run.case_id, run.system) for run in runs]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("run group has missing or duplicate case/system pairs")
    run_groups = {run.run_group_id for run in runs}
    if len(run_groups) != 1:
        raise ValueError("run group contains inconsistent identifiers")
    payload: dict[str, object] = {
        "run_group_id": runs[0].run_group_id,
        "runs": len(runs),
        "runs_sha256": sha256_path(run_path),
        "systems": systems,
        "provenance": provenance,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(stat.S_IREAD)
    return payload
