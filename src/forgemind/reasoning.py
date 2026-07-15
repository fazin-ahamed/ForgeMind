from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Protocol

from forgemind.context import assemble_evidence
from forgemind.domain import (
    AnswerDraft,
    ControllerDecision,
    EvidencePack,
    GenerationResult,
    ReasoningLedger,
    SearchHit,
    VerifiedAnswer,
    model_schema_with_evidence_ids,
)
from forgemind.store import ForgeStore
from forgemind.verification import verify_answer


MODE_CYCLES = {"retrieve": 1, "reason": 3, "investigate": 6}
CONTROLLER_MAX_TOKENS = 2_048
CONTROLLER_EVIDENCE_TOKENS = 8_000
STOPPED = ["Investigation stopped without sufficient evidence."]


class SearchRetriever(Protocol):
    def search(self, query: str, limit: int = 20) -> list[SearchHit]: ...


class CompletionClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> GenerationResult: ...


def _json_text(text: str) -> str:
    stripped = text.strip()
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1] == "```":
        return "\n".join(lines[1:-1])
    return stripped


class ReasoningController:
    def __init__(
        self,
        retriever: SearchRetriever,
        client: CompletionClient,
        count_tokens: Callable[[str], int],
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.count_tokens = count_tokens
        self.system_prompt = (
            Path(__file__).parent / "prompts" / "reasoning-system.txt"
        ).read_text(encoding="utf-8")

    def investigate(
        self, question: str, mode: str = "reason"
    ) -> tuple[
        AnswerDraft,
        ReasoningLedger,
        list[EvidencePack],
        list[GenerationResult],
    ]:
        if mode not in MODE_CYCLES:
            raise ValueError(f"unsupported mode: {mode}")
        ledger = ReasoningLedger(goal=question)
        query = question
        packs: list[EvidencePack] = []
        queries: list[str] = []
        evidence_ids: list[str] = []
        seen_evidence: set[str] = set()
        broadened = False
        generations: list[GenerationResult] = []

        for cycle in range(1, MODE_CYCLES[mode] + 1):
            queries.append(query)
            pack = assemble_evidence(
                query,
                self.retriever.search(query, 20),
                self.count_tokens,
                budget=CONTROLLER_EVIDENCE_TOKENS,
            )
            ledger = ledger.model_copy(
                update={"cycle": cycle, "retrieval_queries": queries}
            )
            if not pack.items:
                packs.append(pack)
                if broadened:
                    break
                broadened = True
                query = f"{question} source code project history configuration"
                continue

            new_ids = [item.id for item in pack.items if item.id not in seen_evidence]
            if not new_ids:
                break
            packs.append(pack)
            evidence_ids.extend(new_ids)
            seen_evidence.update(new_ids)
            ledger = ledger.model_copy(update={"evidence_ids": list(evidence_ids)})
            schema = model_schema_with_evidence_ids(
                ControllerDecision, evidence_ids
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "ledger": ledger.model_dump(),
                            "evidence": pack.model_payload(),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            result = self.client.complete(
                messages,
                max_tokens=CONTROLLER_MAX_TOKENS,
                json_schema=schema,
            )
            generations.append(result)
            try:
                decision = ControllerDecision.model_validate_json(_json_text(result.text))
            except ValueError:
                repair = self.client.complete(
                    [
                        {
                            "role": "system",
                            "content": "Return valid JSON matching the provided schema. /no_think",
                        },
                        {"role": "user", "content": result.text},
                    ],
                    max_tokens=CONTROLLER_MAX_TOKENS,
                    json_schema=schema,
                )
                generations.append(repair)
                try:
                    decision = ControllerDecision.model_validate_json(
                        _json_text(repair.text)
                    )
                except ValueError:
                    break

            ledger = decision.ledger.model_copy(
                update={
                    "goal": question,
                    "cycle": cycle,
                    "retrieval_queries": list(queries),
                    "evidence_ids": list(evidence_ids),
                }
            )
            if decision.action == "answer":
                assert decision.answer is not None
                return decision.answer, ledger, packs, generations
            query = decision.query or ""
            if not query or query in queries:
                break

        return (
            AnswerDraft(summary="Insufficient evidence", claims=[], unresolved=STOPPED),
            ledger,
            packs,
            generations,
        )


class InvestigationService:
    def __init__(self, controller: ReasoningController, store: ForgeStore) -> None:
        self.controller = controller
        self.store = store
        # ponytail: one local investigation at a time; use per-request stores if multi-user throughput matters.
        self._lock = Lock()

    def ask(self, question: str, mode: str = "reason") -> VerifiedAnswer:
        with self._lock:
            draft, ledger, packs, _generations = self.controller.investigate(
                question, mode
            )
            return verify_answer(draft, ledger, packs, self.store)
