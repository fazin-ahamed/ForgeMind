from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import Field, model_validator

from forgemind.context import assemble_evidence
from forgemind.domain import (
    AnswerDraft,
    EvidencePack,
    GenerationResult,
    ReasoningLedger,
    SearchHit,
    StrictModel,
    VerifiedAnswer,
)
from forgemind.store import ForgeStore
from forgemind.verification import verify_answer


_SYSTEM_NAMES = ("raw", "vector", "hybrid", "forgemind")


class EvaluationRetriever(Protocol):
    def search(self, query: str, limit: int = 20) -> list[SearchHit]: ...

    def search_vector(self, query: str, limit: int = 20) -> list[SearchHit]: ...


class EvaluationController(Protocol):
    def investigate(
        self, question: str, mode: str = "reason"
    ) -> tuple[AnswerDraft, ReasoningLedger, list[EvidencePack]]: ...


class EvaluationClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> GenerationResult: ...


class GoldFact(StrictModel):
    id: str
    any_of: list[list[str]]


class EvalCase(StrictModel):
    id: str
    question: str
    evidence_paths: list[str]
    facts: list[GoldFact]
    answer_absent: bool = False
    category: str = "direct"
    archive: str = "100k"


class RunRecord(StrictModel):
    system: str
    case_id: str
    claims: list[str]
    cited_claims: list[bool]
    retrieved_paths: list[str]
    abstained: bool
    active_tokens: int = Field(ge=0, le=16_384)
    latency_ms: float = Field(ge=0)
    peak_vram_mib: int = Field(ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def citations_align_with_claims(self) -> "RunRecord":
        if len(self.cited_claims) != len(self.claims):
            raise ValueError("cited claims must align with claims")
        return self


class CaseMetrics(StrictModel):
    factual_precision: float
    factual_recall: float
    factual_f1: float
    evidence_recall: float
    citation_precision: float
    correct_abstention: float


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", text.lower()))


def _matches(fact: GoldFact, claim: str) -> bool:
    normalized = _normalize(claim)
    return any(
        all(_normalize(term) in normalized for term in group) for group in fact.any_of
    )


def score_case(case: EvalCase, run: RunRecord) -> CaseMetrics:
    matched_facts = {
        fact.id
        for fact in case.facts
        if any(_matches(fact, claim) for claim in run.claims)
    }
    matched_claims = sum(
        any(_matches(fact, claim) for fact in case.facts) for claim in run.claims
    )
    precision = matched_claims / len(run.claims) if run.claims else 0.0
    recall = len(matched_facts) / len(case.facts) if case.facts else 0.0
    factual_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    gold_paths = set(case.evidence_paths)
    evidence_recall = (
        len(gold_paths & set(run.retrieved_paths)) / len(gold_paths)
        if gold_paths
        else 0.0
    )
    citation_precision = (
        sum(run.cited_claims) / len(run.cited_claims)
        if run.cited_claims
        else float(not run.claims)
    )
    return CaseMetrics(
        factual_precision=precision,
        factual_recall=recall,
        factual_f1=factual_f1,
        evidence_recall=evidence_recall,
        citation_precision=citation_precision,
        correct_abstention=float(case.answer_absent and run.abstained),
    )


def write_run(path: Path, run: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(run.model_dump_json() + "\n")


def load_runs(path: Path) -> list[RunRecord]:
    return [
        RunRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def load_cases(path: Path) -> list[EvalCase]:
    cases = [
        EvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation cases contain duplicate IDs")
    if not cases:
        raise ValueError("evaluation requires at least one case")
    return cases


def parse_system_names(text: str) -> list[str]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in _SYSTEM_NAMES]
    if unknown:
        raise ValueError(f"unknown evaluation systems: {', '.join(unknown)}")
    if len(names) != len(set(names)):
        raise ValueError("evaluation systems contain duplicates")
    if not names:
        raise ValueError("evaluation requires at least one system")
    return names


def _interval(
    values: list[float], rng: np.random.Generator
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    means = np.asarray(
        [rng.choice(array, len(array), replace=True).mean() for _ in range(2_000)]
    )
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def summarize(
    cases: list[EvalCase], runs: list[RunRecord], seed: int = 20_260_714
) -> dict[str, object]:
    case_by_id = {case.id: case for case in cases}
    grouped: dict[str, list[CaseMetrics]] = defaultdict(list)
    for run in runs:
        grouped[run.system].append(score_case(case_by_id[run.case_id], run))
    rng = np.random.default_rng(seed)
    summary: dict[str, object] = {}
    for system, metrics in sorted(grouped.items()):
        summary[system] = {
            name: dict(
                zip(
                    ("mean", "ci_low", "ci_high"),
                    _interval([getattr(metric, name) for metric in metrics], rng),
                    strict=True,
                )
            )
            for name in (
                "factual_f1",
                "evidence_recall",
                "citation_precision",
                "correct_abstention",
            )
        }
    return summary


def freeze_results(
    directory: Path,
    cases: list[EvalCase],
    runs: list[RunRecord],
    systems: list[str],
) -> None:
    run_path = directory / "runs.jsonl"
    summary_path = directory / "summary.json"
    if run_path.exists() or summary_path.exists():
        raise FileExistsError(f"frozen evaluation already exists: {directory}")
    expected = {(case.id, system) for case in cases for system in systems}
    actual = [(run.case_id, run.system) for run in runs]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("evaluation is missing or duplicates case/system runs")
    directory.mkdir(parents=True, exist_ok=True)
    for run in runs:
        write_run(run_path, run)
    summary_path.write_text(
        json.dumps(summarize(cases, runs), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class EvaluationRunner:
    def __init__(
        self, systems: dict[str, Callable[[EvalCase], RunRecord]]
    ) -> None:
        self.systems = systems

    def run(self, cases: list[EvalCase], order: list[str]) -> list[RunRecord]:
        runs: list[RunRecord] = []
        for case in sorted(cases, key=lambda item: item.id):
            for name in order:
                try:
                    run = self.systems[name](case)
                    if run.system != name or run.case_id != case.id:
                        raise ValueError("system returned a mismatched run record")
                    runs.append(run)
                except Exception as error:
                    runs.append(
                        RunRecord(
                            system=name,
                            case_id=case.id,
                            claims=[],
                            cited_claims=[],
                            retrieved_paths=[],
                            abstained=True,
                            active_tokens=0,
                            latency_ms=0,
                            peak_vram_mib=0,
                            error=str(error),
                        )
                    )
        return runs


class ControlledSystems:
    def __init__(
        self,
        store: ForgeStore,
        retriever: EvaluationRetriever,
        controller: EvaluationController,
        client: EvaluationClient,
        count_tokens: Callable[[str], int],
        vram_mib: Callable[[], int],
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.controller = controller
        self.client = client
        self.count_tokens = count_tokens
        self.vram_mib = vram_mib

    def vector(self, case: EvalCase) -> RunRecord:
        return self._one_shot(
            "vector", case, self.retriever.search_vector(case.question, 20)
        )

    def raw(self, case: EvalCase) -> RunRecord:
        return self._one_shot("raw", case, self.store.active_hits())

    def hybrid(self, case: EvalCase) -> RunRecord:
        return self._one_shot(
            "hybrid", case, self.retriever.search(case.question, 20)
        )

    def forgemind(self, case: EvalCase) -> RunRecord:
        started = time.perf_counter()
        vram_before = self.vram_mib()
        draft, ledger, packs = self.controller.investigate(
            case.question, "investigate"
        )
        answer = verify_answer(draft, ledger, packs, self.store)
        return self._record(
            "forgemind", case, answer, packs, started, vram_before
        )

    def _one_shot(
        self, system: str, case: EvalCase, hits: list[SearchHit]
    ) -> RunRecord:
        started = time.perf_counter()
        vram_before = self.vram_mib()
        pack = assemble_evidence(case.question, hits, self.count_tokens)
        result = self.client.complete(
            [
                {
                    "role": "system",
                    "content": "Answer only from supplied evidence. Cite evidence IDs. /no_think",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": case.question, "evidence": pack.model_payload()},
                        ensure_ascii=False,
                    ),
                },
            ],
            json_schema=AnswerDraft.model_json_schema(),
        )
        draft = AnswerDraft.model_validate_json(result.text)
        ledger = ReasoningLedger(
            goal=case.question,
            cycle=1,
            retrieval_queries=[case.question],
            evidence_ids=[item.id for item in pack.items],
        )
        answer = verify_answer(draft, ledger, [pack], self.store)
        return self._record(system, case, answer, [pack], started, vram_before)

    def _record(
        self,
        system: str,
        case: EvalCase,
        answer: VerifiedAnswer,
        packs: list[EvidencePack],
        started: float,
        vram_before: int,
    ) -> RunRecord:
        paths = list(
            dict.fromkeys(item.path for pack in packs for item in pack.items)
        )
        return RunRecord(
            system=system,
            case_id=case.id,
            claims=[claim.text for claim in answer.claims],
            cited_claims=[True for _claim in answer.claims],
            retrieved_paths=paths,
            abstained=answer.status == "abstained",
            active_tokens=max((pack.active_tokens for pack in packs), default=0),
            latency_ms=(time.perf_counter() - started) * 1_000,
            # ponytail: model memory is stable in the local server; sample endpoints unless transient peaks become material.
            peak_vram_mib=max(vram_before, self.vram_mib()),
        )
