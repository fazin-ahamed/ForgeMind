from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def validate_model_free_benchmark() -> list[str]:
    errors: list[str] = []
    try:
        benchmark = importlib.import_module("forgemind.benchmark")
    except Exception as error:
        return [f"benchmark contract import failed: {error}"]
    if tuple(benchmark.SYSTEMS) != ("raw", "vector", "hybrid", "forgemind"):
        errors.append("primary benchmark system contract changed")
    if set(benchmark.BAND_LIMITS) != {"32k", "100k", "250k", "1m"}:
        errors.append("benchmark archive-band contract changed")
    return errors


def validate_run(record: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if int(record.get("exit_code", 1)) != 0:
        errors.append("nonzero run exit")
    if int(record.get("uncited_material_claims", 1)) != 0:
        errors.append("uncited material claim")
    if int(record.get("active_tokens", 16_385)) > 16_384:
        errors.append("active-token budget exceeded")
    if float(record.get("peak_vram_mib", 9_501)) > 9_500:
        errors.append("VRAM target exceeded")
    if float(record.get("latency_ms", 120_001)) > 120_000:
        errors.append("p95 latency ceiling exceeded")
    if not str(record.get("answer", "")).strip():
        errors.append("empty answer")
    return errors


def run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ForgeMind release gates.")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--skip-static", action="store_true")
    args = parser.parse_args(argv)
    benchmark_errors = validate_model_free_benchmark()
    if benchmark_errors:
        for error in benchmark_errors:
            print(error)
        return 1
    if not args.skip_static:
        run_checked([sys.executable, "-m", "ruff", "check", "."])
        run_checked([sys.executable, "-m", "mypy", "src"])
        run_checked(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-m",
                "not model and not benchmark",
            ]
        )
    with tempfile.TemporaryDirectory() as temp:
        records = Path(temp) / "smoke.jsonl"
        command = [
            sys.executable,
            "-m",
            "forgemind",
            "smoke",
            "--runs",
            str(args.runs),
            "--jsonl",
            str(records),
        ]
        if args.offline:
            command.append("--offline")
        run_checked(command)
        rows = [
            json.loads(line)
            for line in records.read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != args.runs:
            print(f"expected {args.runs} runs, received {len(rows)}")
            return 1
        failures = [
            (index, validate_run(row)) for index, row in enumerate(rows, 1)
        ]
        failures = [(index, errors) for index, errors in failures if errors]
        for index, errors in failures:
            print(f"run {index}: {', '.join(errors)}")
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
