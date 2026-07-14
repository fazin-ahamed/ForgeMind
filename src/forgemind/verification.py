from __future__ import annotations

from forgemind.domain import AnswerDraft, EvidencePack, ReasoningLedger, VerifiedAnswer
from forgemind.store import ForgeStore


def verify_answer(
    draft: AnswerDraft,
    ledger: ReasoningLedger,
    packs: list[EvidencePack],
    store: ForgeStore,
) -> VerifiedAnswer:
    evidence = {
        item.id: item
        for pack in packs
        for item in pack.items
        if store.validate_evidence(item)
    }
    claims = []
    unresolved = list(draft.unresolved)
    for claim in draft.claims:
        if claim.evidence_ids and all(item in evidence for item in claim.evidence_ids):
            claims.append(claim)
        else:
            unresolved.append(f"Unsupported claim removed: {claim.text}")
    status = "supported" if claims and not unresolved else "partial" if claims or unresolved else "abstained"
    return VerifiedAnswer(
        summary=draft.summary if claims else "Insufficient verified evidence",
        claims=claims,
        unresolved=unresolved,
        cycles=ledger.cycle,
        status=status,
    )
