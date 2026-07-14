import json

import pytest
from pydantic import ValidationError

from forgemind.domain import (
    AnswerDraft,
    Claim,
    ControllerDecision,
    GenerationResult,
    ReasoningLedger,
    SearchHit,
)
from forgemind.reasoning import ReasoningController


def test_controller_decision_requires_query_for_retrieve_action() -> None:
    ledger = ReasoningLedger(goal="Find the auth failure")

    with pytest.raises(ValidationError):
        ControllerDecision(action="retrieve", ledger=ledger)


def test_answer_payload_takes_precedence_over_conflicting_action() -> None:
    decision = ControllerDecision(
        action="retrieve",
        ledger=ReasoningLedger(goal="why"),
        query="why",
        answer=AnswerDraft(
            summary="Supported",
            claims=[Claim(text="Fact", evidence_ids=["c1"])],
        ),
    )

    assert decision.action == "answer"
    assert decision.query is None


def test_ledger_starts_compact_and_bounded() -> None:
    ledger = ReasoningLedger(goal="Find the auth failure")

    assert ledger.cycle == 0
    assert ledger.hypotheses == []
    assert ledger.verified_facts == []
    assert ledger.retrieval_queries == []
    assert ledger.evidence_ids == []


class FakeRetriever:
    def search(self, query: str, limit: int = 20) -> list[SearchHit]:
        return [
            SearchHit(
                "c1",
                "s1",
                "hash",
                "auth.py",
                1,
                1,
                "parseInt(user.id)",
                1.0,
                ("lexical",),
            )
        ]


class RepeatingClient:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_evidence_ids: list[str] = []

    def complete(self, messages, max_tokens=None, json_schema=None) -> GenerationResult:
        self.calls += 1
        self.seen_evidence_ids = json.loads(messages[-1]["content"])["ledger"][
            "evidence_ids"
        ]
        payload = {
            "action": "retrieve",
            "ledger": {"goal": "why"},
            "query": "parseInt user id",
        }
        return GenerationResult(json.dumps(payload), 10, 10, 1.0, 1.0)


def test_controller_stops_when_retrieval_adds_no_evidence() -> None:
    client = RepeatingClient()
    controller = ReasoningController(
        FakeRetriever(), client, lambda text: len(text.split())
    )

    draft, ledger, packs = controller.investigate("why", mode="reason")

    assert client.calls == 1
    assert client.seen_evidence_ids == ["c1"]
    assert ledger.retrieval_queries == ["why", "parseInt user id"]
    assert ledger.evidence_ids == ["c1"]
    assert len(packs) == 1
    assert draft.unresolved == ["Investigation stopped without sufficient evidence."]


class RepairingClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, max_tokens=None, json_schema=None) -> GenerationResult:
        self.calls += 1
        if self.calls == 1:
            return GenerationResult("not json", 1, 1, 1.0, 1.0)
        payload = {
            "action": "answer",
            "ledger": {"goal": "why"},
            "answer": {
                "summary": "Integer parsing caused the failure.",
                "claims": [{"text": "ID was parsed as an integer.", "evidence_ids": ["c1"]}],
            },
        }
        return GenerationResult(f"```json\n{json.dumps(payload)}\n```", 1, 1, 1.0, 1.0)


def test_controller_repairs_invalid_model_json_once() -> None:
    client = RepairingClient()
    controller = ReasoningController(
        FakeRetriever(), client, lambda text: len(text.split())
    )

    draft, ledger, _packs = controller.investigate("why")

    assert client.calls == 2
    assert draft.claims[0].evidence_ids == ["c1"]
    assert ledger.cycle == 1


class EmptyRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, limit: int = 20) -> list[SearchHit]:
        self.queries.append(query)
        return []


def test_empty_evidence_broadens_once_then_abstains() -> None:
    retriever = EmptyRetriever()
    controller = ReasoningController(
        retriever,
        object(),
        lambda text: len(text.split()),
    )

    draft, ledger, packs = controller.investigate("why did auth fail")

    assert len(retriever.queries) == 2
    assert retriever.queries[0] == "why did auth fail"
    assert ledger.retrieval_queries == retriever.queries
    assert len(packs) == 2
    assert draft.claims == []
