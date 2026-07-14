from __future__ import annotations

import json
import time
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
    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> GenerationResult:
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


def run_offline_smoke(
    runs: int, jsonl_path: Path | None = None
) -> dict[str, object]:
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
    answers = []
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")
    for index in range(runs):
        started = time.perf_counter()
        try:
            answer = service.ask(
                "__missing__" if index == runs - 1 else f"case {index}",
                "reason",
            )
            answers.append(answer)
            record = {
                "exit_code": 0,
                "uncited_material_claims": sum(
                    not claim.evidence_ids for claim in answer.claims
                ),
                "active_tokens": sum(
                    len(item.model_text.split()) for item in answer.evidence
                ),
                "peak_vram_mib": 0,
                "latency_ms": (time.perf_counter() - started) * 1_000,
                "answer": answer.summary,
            }
        except Exception:
            record = {
                "exit_code": 1,
                "uncited_material_claims": 0,
                "active_tokens": 0,
                "peak_vram_mib": 0,
                "latency_ms": (time.perf_counter() - started) * 1_000,
                "answer": "",
            }
        if jsonl_path is not None:
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
    return {
        "runs": runs,
        "completed": len(answers),
        "citations_valid": all(
            store.validate_evidence(item) for answer in answers for item in answer.evidence
        ),
        "empty_evidence_abstained": bool(answers)
        and answers[-1].status == "abstained",
    }
