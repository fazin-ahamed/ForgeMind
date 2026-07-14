from __future__ import annotations

from collections.abc import Callable

from forgemind.domain import EvidenceItem, EvidencePack, SearchHit


def assemble_evidence(
    query: str,
    hits: list[SearchHit],
    count_tokens: Callable[[str], int],
    budget: int = 12_000,
    archived_tokens: int = 0,
) -> EvidencePack:
    if not 0 <= budget <= 16_384:
        raise ValueError("active evidence budget must be between 0 and 16,384 tokens")
    items: list[EvidenceItem] = []
    used = 0
    seen_spans: set[tuple[str, int, int]] = set()
    for hit in hits:
        span = (hit.source_id, hit.start_line, hit.end_line)
        tokens = count_tokens(hit.text)
        if span in seen_spans or used + tokens > budget:
            continue
        seen_spans.add(span)
        used += tokens
        items.append(
            EvidenceItem(
                id=hit.chunk_id,
                source_id=hit.source_id,
                source_sha256=hit.source_sha256,
                path=hit.path,
                start_line=hit.start_line,
                end_line=hit.end_line,
                text=hit.text,
                channels=hit.channels,
            )
        )
    return EvidencePack(
        query=query,
        items=items,
        archived_tokens=archived_tokens,
        active_tokens=used,
    )
