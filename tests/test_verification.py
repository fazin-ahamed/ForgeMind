from pathlib import Path

from forgemind.domain import (
    AnswerDraft,
    Claim,
    EvidenceItem,
    EvidencePack,
    ReasoningLedger,
    SourceRecord,
)
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

    answer = verify_answer(draft, ReasoningLedger(goal="why", cycle=1), [pack], store)

    assert answer.claims == []
    assert answer.status == "partial"
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
    )

    assert store.validate_evidence(item)
    assert not store.validate_evidence(item.model_copy(update={"text": "tampered"}))
    assert not store.validate_evidence(item.model_copy(update={"path": "other.py"}))
