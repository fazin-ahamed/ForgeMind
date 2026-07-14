import json
from pathlib import Path

from fastapi.testclient import TestClient

from forgemind.web import create_app


class FakeService:
    def ask(self, question: str, mode: str) -> object:
        raise AssertionError("not called")


def test_results_route_reads_frozen_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"forgemind": {"factual_f1": {"mean": 0.8}}}),
        encoding="utf-8",
    )
    client = TestClient(create_app(FakeService(), summary_path=summary))

    response = client.get("/api/results")

    assert response.json()["forgemind"]["factual_f1"]["mean"] == 0.8
    assert response.headers["cache-control"] == "no-store"
