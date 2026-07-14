from __future__ import annotations

import re

from pydantic import Field, model_validator

from forgemind.domain import StrictModel


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
