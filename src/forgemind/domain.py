import hashlib
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    gpu_name: str
    vram_mib: int
    driver_version: str
    ram_mib: int


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    prompt_ms: float
    generation_ms: float

    @property
    def total_ms(self) -> float:
        return self.prompt_ms + self.generation_ms


@dataclass(frozen=True, slots=True)
class SourceRecord:
    id: str
    path: str
    sha256: str
    modified_ns: int
    text: str

    @classmethod
    def from_text(cls, path: str, text: str, modified_ns: int) -> "SourceRecord":
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_id = hashlib.sha256(f"{path}\0{digest}".encode("utf-8")).hexdigest()
        return cls(source_id, path, digest, modified_ns, text)


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    id: str
    source_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectEvent:
    id: str
    commit: str
    occurred_at: str
    summary: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    source_id: str
    source_sha256: str
    path: str
    start_line: int
    end_line: int
    text: str
    score: float
    channels: tuple[str, ...]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceItem(StrictModel):
    id: str
    source_id: str
    source_sha256: str
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str
    model_text: str
    aliases: dict[str, str] = Field(default_factory=dict)
    channels: tuple[str, ...] = ()


class EvidencePack(StrictModel):
    query: str
    items: list[EvidenceItem]
    archived_tokens: int = Field(ge=0)
    active_tokens: int = Field(ge=0, le=16_384)

    def model_payload(self) -> dict[str, object]:
        return {
            "query": self.query,
            "items": [
                {
                    "id": item.id,
                    "path": item.path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "text": item.model_text,
                }
                for item in self.items
            ],
        }


class Hypothesis(StrictModel):
    claim: str
    status: Literal["testing", "supported", "rejected"] = "testing"


class VerifiedFact(StrictModel):
    fact: str
    evidence_ids: list[str]


class ReasoningLedger(StrictModel):
    goal: str
    retrieval_queries: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    verified_facts: list[VerifiedFact] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    rejected_hypotheses: list[str] = Field(default_factory=list)
    next_query: str | None = None
    cycle: int = Field(default=0, ge=0, le=6)


class Claim(StrictModel):
    text: str
    evidence_ids: list[str]


class AnswerDraft(StrictModel):
    summary: str
    claims: list[Claim]
    unresolved: list[str] = Field(default_factory=list)


class ControllerDecision(StrictModel):
    action: Literal["retrieve", "answer"]
    ledger: ReasoningLedger
    query: str | None = None
    answer: AnswerDraft | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "ControllerDecision":
        if self.answer is not None:
            self.action = "answer"
            self.query = None
            return self
        if self.action == "retrieve" and not self.query:
            raise ValueError("retrieve action requires query")
        if self.action == "answer":
            raise ValueError("answer action requires answer")
        return self


class VerifiedAnswer(StrictModel):
    summary: str
    claims: list[Claim]
    unresolved: list[str]
    cycles: int = Field(ge=0, le=6)
    status: Literal["supported", "partial", "abstained"]
    retrieval_queries: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
