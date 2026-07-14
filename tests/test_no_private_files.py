import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_git_index_contains_no_private_artifacts() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    forbidden = (
        ".forgemind-private/",
        ".forgemind-private/benchmarks/",
        ".forgemind-private/runs/",
        "docs/superpowers/",
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


def test_clean_checkout_automation_preserves_imports_and_native_failures() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    script = (ROOT / "scripts/clean_checkout_test.ps1").read_text(encoding="utf-8")

    assert "uv run python -m pytest" in workflow
    assert "uv run python -m pytest" in script
    assert "$LASTEXITCODE" in script
