import pytest

from forgemind.context import assemble_evidence
from forgemind.domain import SearchHit


def test_evidence_pack_never_exceeds_budget() -> None:
    hits = [
        SearchHit(
            "c1",
            "s1",
            "hash1",
            "a.py",
            1,
            2,
            "one two three",
            1.0,
            ("lexical",),
        ),
        SearchHit(
            "c2",
            "s2",
            "hash2",
            "b.py",
            1,
            2,
            "four five six",
            0.9,
            ("semantic",),
        ),
    ]

    pack = assemble_evidence("why", hits, lambda text: len(text.split()), budget=5)

    assert pack.active_tokens <= 5
    assert [item.id for item in pack.items] == ["c1"]


def test_evidence_pack_rejects_oversized_active_budget() -> None:
    with pytest.raises(ValueError, match="16,384"):
        assemble_evidence("why", [], lambda text: len(text.split()), budget=16_385)
