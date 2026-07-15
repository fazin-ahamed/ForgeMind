from __future__ import annotations

from collections.abc import Callable

from forgemind.domain import EvidenceItem, EvidencePack, SearchHit
from forgemind.tokenforge import TokenForge


def assemble_evidence(
    query: str,
    hits: list[SearchHit],
    count_tokens: Callable[[str], int],
    budget: int = 10_000,
    archived_tokens: int = 0,
) -> EvidencePack:
    if not 0 <= budget <= 32_768:
        raise ValueError("active evidence budget must be between 0 and 32,768 tokens")
    items: list[EvidenceItem] = []
    used = 0
    seen_spans: set[tuple[str, int, int]] = set()
    tokenforge = TokenForge()
    for hit in hits:
        span = (hit.source_id, hit.start_line, hit.end_line)
        compressed = tokenforge.compress(hit.text)
        try:
            restored = tokenforge.restore(compressed)
            model_text = compressed.text if restored == hit.text else hit.text
            aliases = compressed.aliases if model_text != hit.text else {}
        except ValueError:
            model_text = hit.text
            aliases = {}
        tokens = count_tokens(model_text)
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
                model_text=model_text,
                aliases=aliases,
                channels=hit.channels,
            )
        )
    return EvidencePack(
        query=query,
        items=items,
        archived_tokens=archived_tokens,
        active_tokens=used,
    )
