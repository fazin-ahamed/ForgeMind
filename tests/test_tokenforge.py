from forgemind.tokenforge import TokenForge


def test_repeated_long_paths_round_trip_byte_for_byte() -> None:
    original = "/src/features/authentication/session/decoder.ts\n" * 3
    compressed = TokenForge().compress(original)

    assert len(compressed.text) < len(original)
    assert TokenForge().restore(compressed) == original


def test_existing_alias_like_text_is_not_reused() -> None:
    identifier = "customer_authentication_identifier"
    original = f"¤1 {identifier} {identifier}"
    compressed = TokenForge().compress(original)

    assert TokenForge().restore(compressed) == original
    assert compressed.aliases == {"¤2": identifier}
