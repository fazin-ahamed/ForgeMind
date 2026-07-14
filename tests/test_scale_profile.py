from pathlib import Path

import pytest

from benchmarks.generate_archive import generate_archive, generate_distractors
from forgemind.cli import profile_scale
from forgemind.tokenforge import TokenForge


def test_scale_archive_is_deterministic(tmp_path: Path) -> None:
    first = generate_archive(tmp_path / "a", target_words=1_000, seed=42)
    second = generate_archive(tmp_path / "b", target_words=1_000, seed=42)

    assert first == second
    assert (
        sum(
            len(path.read_text(encoding="utf-8").split())
            for path in (tmp_path / "a").glob("*.log")
        )
        >= 1_000
    )
    sample = next((tmp_path / "a").glob("*.log")).read_text(encoding="utf-8")
    assert TokenForge().compress(sample).aliases


def test_distractor_corpus_is_seeded_and_substantial() -> None:
    first = generate_distractors(seed=8, documents=3, lines_per_document=16)
    second = generate_distractors(seed=8, documents=3, lines_per_document=16)

    assert first == second
    assert len(first) == 3
    assert all(len(document.splitlines()) == 16 for document in first)


def test_scale_profile_rejects_active_context_above_hard_limit(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="hard limit"):
        profile_scale(tmp_path, tmp_path / "profile.sqlite", 16_385)
