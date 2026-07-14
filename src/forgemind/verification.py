from __future__ import annotations

from typing import Literal

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
    cited = {item for claim in claims for item in claim.evidence_ids}
    if claims and not unresolved:
        status: Literal["supported", "partial", "abstained"] = "supported"
    elif claims or any(item.startswith("Unsupported claim removed:") for item in unresolved):
        status = "partial"
    else:
        status = "abstained"
    return VerifiedAnswer(
        summary=draft.summary if claims else "Insufficient verified evidence",
        claims=claims,
        unresolved=unresolved,
        cycles=ledger.cycle,
        status=status,
        retrieval_queries=ledger.retrieval_queries,
        evidence_ids=ledger.evidence_ids,
        evidence=[item for item_id, item in evidence.items() if item_id in cited],
    )
