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


def test_default_budget_reserves_prompt_envelope() -> None:
    hit = SearchHit(
        "c1", "s1", "hash", "large.txt", 1, 1, "x", 1.0, ("lexical",)
    )

    pack = assemble_evidence("why", [hit], lambda _text: 10_001)

    assert pack.items == []


def test_evidence_pack_rejects_oversized_active_budget() -> None:
    with pytest.raises(ValueError, match="32,768"):
        assemble_evidence("why", [], lambda text: len(text.split()), budget=32_769)


def test_evidence_pack_allows_isolated_raw_32k_budget() -> None:
    pack = assemble_evidence(
        "why",
        [],
        lambda text: len(text.split()),
        budget=32_000,
    )

    assert pack.active_tokens == 0


def test_model_payload_uses_compressed_text_but_retains_exact_source() -> None:
    repeated = "customer_identifier customer_identifier customer_identifier"
    hit = SearchHit(
        "c1",
        "s1",
        "hash",
        "a.py",
        1,
        1,
        repeated,
        1.0,
        ("lexical",),
    )

    pack = assemble_evidence("why", [hit], lambda text: len(text.split()), budget=20)
    payload = pack.model_payload()

    assert pack.items[0].text == repeated
    assert payload["items"][0]["text"] != repeated
    assert "source_sha256" not in payload["items"][0]
