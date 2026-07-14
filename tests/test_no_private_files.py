import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_git_index_contains_no_private_artifacts() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    forbidden = (
        ".forgemind-private/",
        "planning/",
        "research/",
        "reports/",
        "models/",
        "data/private/",
        "artifacts/",
        "benchmark-results/",
    )

    assert not [path for path in tracked if path.startswith(forbidden)]


def test_clean_checkout_automation_is_public() -> None:
    assert (ROOT / ".github/workflows/ci.yml").is_file()
    assert (ROOT / "scripts/clean_checkout_test.ps1").is_file()
