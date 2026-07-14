import pytest
from pydantic import ValidationError

from forgemind.domain import ControllerDecision, ReasoningLedger


def test_controller_decision_requires_query_for_retrieve_action() -> None:
    ledger = ReasoningLedger(goal="Find the auth failure")

    with pytest.raises(ValidationError):
        ControllerDecision(action="retrieve", ledger=ledger)


def test_ledger_starts_compact_and_bounded() -> None:
    ledger = ReasoningLedger(goal="Find the auth failure")

    assert ledger.cycle == 0
    assert ledger.hypotheses == []
    assert ledger.verified_facts == []
