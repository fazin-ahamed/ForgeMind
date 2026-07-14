import json
from pathlib import Path

from fastapi.testclient import TestClient

from forgemind.web import create_app


ROOT = Path(__file__).resolve().parents[1]


class FakeService:
    def ask(self, question: str, mode: str) -> object:
        raise AssertionError("not called by page contract")


def test_demo_page_has_accessible_landmarks_and_fallback(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"forgemind": {}}), encoding="utf-8")
    client = TestClient(create_app(FakeService(), summary_path=summary))

    page = client.get("/")

    assert page.status_code == 200
    for marker in (
        '<main id="main-content"',
        '<nav aria-label="Demo navigation"',
        'aria-live="polite"',
        'id="question"',
        'for="question"',
        'class="skip-link"',
        "Previously frozen run",
        "Live model timed out",
    ):
        assert marker in page.text

    fallback = client.get("/api/results")
    assert fallback.status_code == 200
    assert fallback.headers["cache-control"] == "no-store"


def test_demo_launcher_runs_local_preflight_and_frozen_fallback() -> None:
    launcher = (ROOT / "scripts" / "demo.ps1").read_text(encoding="utf-8")

    assert "verify_assets.py" in launcher
    assert "forgemind doctor" in launcher
    assert '"127.0.0.1"' in launcher
    assert "--summary" in launcher
    assert "Start-Process" in launcher
    assert "-WindowStyle Hidden" in launcher
