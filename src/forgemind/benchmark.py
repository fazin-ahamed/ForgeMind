from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from forgemind.domain import StrictModel


Capability = Literal["repository", "memory", "effective-context", "adversarial"]
ArchiveBand = Literal["32k", "100k", "250k", "1m"]
AnswerKind = Literal["exact", "set", "text"]
SYSTEMS = ("raw", "vector", "hybrid", "forgemind")
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
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
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
