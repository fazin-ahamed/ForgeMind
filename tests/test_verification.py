from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from forgemind.domain import (
    AnswerDraft,
    Claim,
    EvidenceItem,
    EvidencePack,
    ReasoningLedger,
    SourceRecord,
)
from forgemind.reasoning import InvestigationService
from forgemind.store import ForgeStore
from forgemind.verification import verify_answer


def test_invalid_evidence_id_is_removed_before_display(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    store.upsert_source(
        SourceRecord.from_text("auth.py", "user_id = parseInt(raw)\n", 1)
    )
    draft = AnswerDraft(
        summary="Bad parse",
        claims=[Claim(text="UUID is parsed as int", evidence_ids=["missing"])],
    )
    pack = EvidencePack(
        query="why", items=[], archived_tokens=1_000_000, active_tokens=0
    )

    ledger = ReasoningLedger(
        goal="why",
        cycle=1,
        retrieval_queries=["why", "parseInt UUID"],
        evidence_ids=["missing"],
    )
    answer = verify_answer(draft, ledger, [pack], store)

    assert answer.claims == []
    assert answer.status == "partial"
    assert answer.retrieval_queries == ["why", "parseInt UUID"]
    assert answer.evidence_ids == ["missing"]
    assert "Unsupported claim removed: UUID is parsed as int" in answer.unresolved


def test_evidence_must_match_immutable_source_hash_and_exact_lines(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    source = SourceRecord.from_text("auth.py", "first\nuser_id = parseInt(raw)\n", 1)
    store.upsert_source(source)
    item = EvidenceItem(
        id="c1",
        source_id=source.id,
        source_sha256=source.sha256,
        path=source.path,
        start_line=2,
        end_line=2,
        text="user_id = parseInt(raw)",
        model_text="user_id = parseInt(raw)",
    )

    assert store.validate_evidence(item)
    assert not store.validate_evidence(item.model_copy(update={"text": "tampered"}))
    assert not store.validate_evidence(item.model_copy(update={"path": "other.py"}))


def test_verified_answer_carries_displayable_source_spans(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    source = SourceRecord.from_text("auth.py", "user_id = parseInt(raw)\n", 1)
    store.upsert_source(source)
    item = EvidenceItem(
        id="c1",
        source_id=source.id,
        source_sha256=source.sha256,
        path=source.path,
        start_line=1,
        end_line=1,
        text="user_id = parseInt(raw)",
        model_text="user_id = parseInt(raw)",
    )
    draft = AnswerDraft(
        summary="Bad parse",
        claims=[Claim(text="ID is parsed as an integer.", evidence_ids=["c1"])],
    )
    pack = EvidencePack(query="why", items=[item], archived_tokens=1, active_tokens=1)
    ledger = ReasoningLedger(goal="why", cycle=1, evidence_ids=["c1"])

    answer = verify_answer(draft, ledger, [pack], store)

    assert answer.status == "supported"
    assert answer.evidence == [item]


def test_investigation_service_always_verifies_controller_output(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    source = SourceRecord.from_text("auth.py", "safe fact\n", 1)
    store.upsert_source(source)
    item = EvidenceItem(
        id="c1",
        source_id=source.id,
        source_sha256=source.sha256,
        path=source.path,
        start_line=1,
        end_line=1,
        text="safe fact",
        model_text="safe fact",
    )

    class FixedController:
        def investigate(self, question: str, mode: str):
            return (
                AnswerDraft(
                    summary="Safe",
                    claims=[Claim(text="Fact is safe.", evidence_ids=["c1"])],
                ),
                    ReasoningLedger(goal=question, cycle=1, evidence_ids=["c1"]),
                    [EvidencePack(query=question, items=[item], archived_tokens=1, active_tokens=1)],
                    [],
                )

    service = InvestigationService(FixedController(), store)
    with ThreadPoolExecutor(max_workers=1) as executor:
        answer = executor.submit(service.ask, "why", "reason").result()

    assert answer.status == "supported"
    assert answer.evidence == [item]
