import json
from pathlib import Path

import scripts.runtime_smoke as smoke
from forgemind.domain import GenerationResult, HardwareProfile
from forgemind.offline import run_offline_smoke


def test_runtime_smoke_writes_ten_run_record(tmp_path: Path, monkeypatch) -> None:
    server_path = tmp_path / "server.exe"
    model = tmp_path / "model.gguf"
    server_path.write_bytes(b"exe")
    model.write_bytes(b"model")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORGEMIND_LLAMA_SERVER", str(server_path))
    monkeypatch.setenv("FORGEMIND_MODEL", str(model))

    class FakeServer:
        def __init__(self, config) -> None:
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeClient:
        def __init__(self, config) -> None:
            return None

        def complete(self, messages, max_tokens=None) -> GenerationResult:
            assert messages[0]["content"].endswith("/no_think")
            return GenerationResult("ForgeMind ready", 3, 2, 1.0, 2.0)

    monkeypatch.setattr(smoke, "start_with_single_fallback", FakeServer)
    monkeypatch.setattr(smoke, "LlamaClient", FakeClient)
    monkeypatch.setattr(
        smoke,
        "probe_hardware",
        lambda: HardwareProfile("RTX 3060", 12_288, "610.74", 32_000),
    )

    assert smoke.main() == 0
    output = tmp_path / ".forgemind-private/results/m1-hardware-profile.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 10
    assert payload["hardware"]["vram_mib"] == 12_288


def test_offline_reasoning_smoke_verifies_citations_and_abstention() -> None:
    result = run_offline_smoke(10)

    assert result == {
        "runs": 10,
        "completed": 10,
        "citations_valid": True,
        "empty_evidence_abstained": True,
    }


def test_offline_smoke_writes_one_release_record_per_run(tmp_path: Path) -> None:
    output = tmp_path / "smoke.jsonl"

    result = run_offline_smoke(3, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert result["completed"] == 3
    assert len(rows) == 3
    assert all(
        set(row)
        == {
            "exit_code",
            "uncited_material_claims",
            "active_tokens",
            "peak_vram_mib",
            "latency_ms",
            "answer",
        }
        for row in rows
    )
