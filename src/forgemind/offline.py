from __future__ import annotations

import json
from pathlib import Path

from forgemind.domain import GenerationResult, SearchHit, SourceRecord
from forgemind.reasoning import InvestigationService, ReasoningController
from forgemind.store import ForgeStore


class _Retriever:
    def __init__(self, hit: SearchHit) -> None:
        self.hit = hit

    def search(self, query: str, limit: int = 20) -> list[SearchHit]:
        return [] if query.startswith("__missing__") else [self.hit]


class _Client:
    def complete(self, messages, max_tokens=None, json_schema=None) -> GenerationResult:
        request = json.loads(messages[-1]["content"])
        payload = {
            "action": "answer",
            "ledger": {"goal": request["question"]},
            "answer": {
                "summary": "The exact source supports the answer.",
                "claims": [{"text": "The source contains the fact.", "evidence_ids": ["c1"]}],
            },
        }
        return GenerationResult(json.dumps(payload), 1, 1, 1.0, 1.0)


def run_offline_smoke(runs: int) -> dict[str, object]:
    if runs < 1:
        raise ValueError("runs must be positive")
    store = ForgeStore(Path(":memory:"))
    source = SourceRecord.from_text("auth.py", "safe fact\n", 1)
    store.upsert_source(source)
    hit = SearchHit(
        "c1",
        source.id,
        source.sha256,
        source.path,
        1,
        1,
        "safe fact",
        1.0,
        ("offline",),
    )
    service = InvestigationService(
        ReasoningController(_Retriever(hit), _Client(), lambda text: len(text.split())),
        store,
    )
    answers = [
        service.ask("__missing__" if index == runs - 1 else f"case {index}", "reason")
        for index in range(runs)
    ]
    return {
        "runs": runs,
        "completed": len(answers),
        "citations_valid": all(
            store.validate_evidence(item) for answer in answers for item in answer.evidence
        ),
        "empty_evidence_abstained": answers[-1].status == "abstained",
    }
