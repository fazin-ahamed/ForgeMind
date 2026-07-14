from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_public_files_exist_and_define_bounded_claim() -> None:
    for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".env.example"):
        assert (ROOT / name).is_file(), name

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "16,384" in readme
    assert "Qwen3-4B" in readme
    assert "million-token information space" in readme
    assert "does not directly attend to one million tokens" in readme


def test_example_environment_is_local_and_contains_no_secret_values() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "FORGEMIND_HOST=127.0.0.1" in text
    assert "FORGEMIND_CONTEXT=16384" in text
    assert "api_key=" not in text.lower()
    assert "token=" not in text.lower()


def test_private_directories_remain_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (
        "/.forgemind-private/",
        "/reports/",
        "/benchmark-results/",
        "/models/",
    ):
        assert entry in gitignore


def test_paper_exporter_targets_only_the_private_workspace() -> None:
    exporter = ROOT / "scripts" / "export_paper.ps1"
    assert exporter.is_file()

    text = exporter.read_text(encoding="utf-8")
    assert ".forgemind-private" in text
    assert "paper.md" in text
    assert "pandoc" in text
    assert "xelatex" in text


def test_public_docs_name_proof_commands_and_non_claims() -> None:
    benchmark = (ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for command in (
        "benchmark-validate",
        "benchmark-prepare",
        "benchmark-report",
    ):
        assert command in benchmark
    assert "does not directly attend to one million tokens" in readme
    assert "derived" in benchmark
