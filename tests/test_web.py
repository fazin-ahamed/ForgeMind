from fastapi.testclient import TestClient

from forgemind.domain import VerifiedAnswer
from forgemind.web import create_app


class FakeService:
    def ask(self, question: str, mode: str) -> VerifiedAnswer:
        return VerifiedAnswer(
            summary="UUID parse",
            claims=[],
            unresolved=[],
            cycles=2,
            status="supported",
        )


def test_local_api_returns_verified_answer() -> None:
    client = TestClient(create_app(FakeService()))

    assert client.get("/").status_code == 200
    response = client.post(
        "/api/ask", json={"question": "why", "mode": "investigate"}
    )

    assert response.status_code == 200
    assert response.json()["summary"] == "UUID parse"


def test_api_rejects_unknown_reasoning_mode() -> None:
    response = TestClient(create_app(FakeService())).post(
        "/api/ask", json={"question": "why", "mode": "unbounded"}
    )

    assert response.status_code == 422


def test_api_converts_model_failure_to_generic_unavailable_response() -> None:
    class FailingService:
        def ask(self, question: str, mode: str) -> VerifiedAnswer:
            raise RuntimeError("private upstream detail")

    response = TestClient(create_app(FailingService())).post(
        "/api/ask", json={"question": "why", "mode": "investigate"}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Live model unavailable"}
    assert "private upstream detail" not in response.text
