import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.build_submission import build_bundle, validate_members


def test_submission_policy_accepts_required_files() -> None:
    members = [
        "forgemind-paper.pdf",
        "summary.json",
        "demo-script.md",
        "SOURCE_REVISION",
    ]
    assert validate_members(members) == []


def test_submission_policy_rejects_sensitive_assets() -> None:
    errors = validate_members(
        [
            "forgemind-paper.pdf",
            "summary.json",
            "demo-script.md",
            "SOURCE_REVISION",
            "models/model.gguf",
            ".env",
            "data/private/corpus.txt",
        ]
    )
    assert len(errors) == 3


def test_submission_policy_rejects_traversal_and_absolute_paths() -> None:
    required = [
        "forgemind-paper.pdf",
        "summary.json",
        "demo-script.md",
        "SOURCE_REVISION",
    ]
    errors = validate_members(required + ["../secret.txt", "C:/secret.txt"])
    assert len(errors) == 2


def test_bundle_is_deterministic_and_contains_checksums(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    fixtures = {
        "forgemind-paper.pdf": b"%PDF-fixture",
        "summary.json": b"{}\n",
        "demo-script.md": b"# Demo\n",
        "SOURCE_REVISION": b"a" * 40 + b"\n",
    }
    for name, payload in fixtures.items():
        (staging / name).write_bytes(payload)

    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    build_bundle(staging, first)
    build_bundle(staging, second)

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "SHA256SUMS.json",
            "SOURCE_REVISION",
            "demo-script.md",
            "forgemind-paper.pdf",
            "summary.json",
        ]


def test_bundle_refuses_symlinked_members(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    for name in (
        "forgemind-paper.pdf",
        "summary.json",
        "demo-script.md",
        "SOURCE_REVISION",
    ):
        (staging / name).write_text("fixture", encoding="utf-8")
    target = tmp_path / "outside.txt"
    target.write_text("secret", encoding="utf-8")
    link = staging / "linked.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symlink"):
        build_bundle(staging, tmp_path / "bundle.zip")
