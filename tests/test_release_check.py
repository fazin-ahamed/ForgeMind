from scripts.release_check import validate_run


def test_healthy_run_passes() -> None:
    record = {
        "exit_code": 0,
        "uncited_material_claims": 0,
        "active_tokens": 12_000,
        "peak_vram_mib": 8_200,
        "latency_ms": 41_000,
        "answer": "Supported answer",
    }

    assert validate_run(record) == []


def test_unhealthy_run_reports_every_gate() -> None:
    record = {
        "exit_code": 1,
        "uncited_material_claims": 2,
        "active_tokens": 17_000,
        "peak_vram_mib": 10_000,
        "latency_ms": 130_000,
        "answer": "",
    }

    errors = validate_run(record)

    assert len(errors) == 6
