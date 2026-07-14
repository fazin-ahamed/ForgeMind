import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import benchmarks.build_forgebench as builder
from benchmarks.build_forgebench import (
    build_adversarial_cases,
    build_ruler_case,
    build_suite,
    llama_token_count,
    rebase_span,
    scale_archive,
    select_cell,
    write_jsonl,
)
from forgemind.benchmark import CitationSpan


def test_ruler_case_is_seeded_and_answer_is_not_in_runtime_record(
    tmp_path: Path,
) -> None:
    first_runtime, first_gold = build_ruler_case(tmp_path / "a", seed=7, pairs=4)
    second_runtime, second_gold = build_ruler_case(tmp_path / "b", seed=7, pairs=4)

    assert first_runtime.question == second_runtime.question
    assert first_gold.answer == second_gold.answer
    assert "accepted" not in first_runtime.model_dump_json()


def test_scale_archive_stops_inside_requested_band(tmp_path: Path) -> None:
    source = tmp_path / "archive"
    source.mkdir()
    evidence = source / "evidence.txt"
    evidence.write_text("needle fact\n", encoding="utf-8")
    original = evidence.read_bytes()

    def count(text: str) -> int:
        return len(text.split())

    total = scale_archive(
        source,
        "32k",
        ["distractor " * 4_096],
        count,
        seed=3,
    )

    assert 28_000 <= total <= 40_000
    assert evidence.read_bytes() == original


def test_adversarial_set_includes_absence_and_superseded_facts(
    tmp_path: Path,
) -> None:
    pairs = build_adversarial_cases(tmp_path, count=40, seed=11)

    assert len(pairs) == 40
    assert sum(gold.answer_absent for _, gold in pairs) >= 12
    assert sum(gold.source.endswith("/superseded") for _, gold in pairs) >= 12
    assert all("accepted" not in runtime.model_dump_json() for runtime, _ in pairs)


def test_rebase_span_recomputes_source_identity() -> None:
    span = CitationSpan(
        source_id="a" * 64,
        source_sha256="b" * 64,
        path="src/auth.py",
        start_line=3,
        end_line=5,
    )

    rebased = rebase_span(span, "case-7")

    expected_path = "case-7/src/auth.py"
    assert rebased.path == expected_path
    assert rebased.source_id == hashlib.sha256(
        f"{expected_path}\0{span.source_sha256}".encode("utf-8")
    ).hexdigest()


def test_select_cell_is_seeded_and_bands_do_not_overlap(tmp_path: Path) -> None:
    candidates = [
        build_ruler_case(tmp_path / str(index), seed=index)
        for index in range(8)
    ]

    first = select_cell(candidates, "effective-context", "32k", 2, seed=5)
    repeated = select_cell(candidates, "effective-context", "32k", 2, seed=5)
    next_band = select_cell(candidates, "effective-context", "100k", 2, seed=5)

    assert [runtime.id for runtime, _ in first] == [
        runtime.id for runtime, _ in repeated
    ]
    assert {runtime.id for runtime, _ in first}.isdisjoint(
        runtime.id for runtime, _ in next_band
    )


def test_repository_cell_uses_one_shared_repository_per_band(
    tmp_path: Path,
) -> None:
    candidates = []
    for group in range(4):
        runtime, gold = build_ruler_case(tmp_path / f"repo-{group}", seed=group)
        for question in range(2):
            case_id = f"repo-{group}-question-{question}"
            candidates.append(
                (
                    runtime.model_copy(
                        update={"id": case_id, "capability": "repository"}
                    ),
                    gold.model_copy(update={"case_id": case_id}),
                )
            )

    first = select_cell(candidates, "repository", "32k", 2, seed=5)
    second = select_cell(candidates, "repository", "100k", 2, seed=5)

    assert len({runtime.archive_path for runtime, _ in first}) == 1
    assert {runtime.archive_path for runtime, _ in first}.isdisjoint(
        runtime.archive_path for runtime, _ in second
    )


def test_write_jsonl_keeps_one_record_per_line(tmp_path: Path) -> None:
    runtime, _ = build_ruler_case(tmp_path / "case", seed=9)
    output = tmp_path / "runtime.jsonl"

    write_jsonl(output, [runtime])

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"id":"ruler-9"' in lines[0]


def test_documented_builder_module_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.build_forgebench", "--help"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_llama_token_count_uses_tokenize_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return io.BytesIO(b'{"tokens":[1,2,3]}')

    monkeypatch.setattr(builder.urllib.request, "urlopen", urlopen)

    assert llama_token_count("http://localhost:8080/", "hello") == 3
    assert captured == {
        "url": "http://localhost:8080/tokenize",
        "payload": {"content": "hello"},
        "timeout": 120,
    }


def test_build_suite_creates_sixteen_shared_archives(
    tmp_path: Path, monkeypatch
) -> None:
    candidates = []
    for capability in builder.CAPABILITIES:
        for index in range(8):
            source_index = index // 2 if capability == "repository" else index
            runtime, gold = build_ruler_case(
                tmp_path / "candidates" / capability / str(source_index),
                seed=100 * builder.CAPABILITIES.index(capability) + source_index,
            )
            case_id = f"{capability}-{index}"
            candidates.append(
                (
                    runtime.model_copy(
                        update={"id": case_id, "capability": capability}
                    ),
                    gold.model_copy(update={"case_id": case_id}),
                )
            )

    def fake_scale(root, band, distractors, count_tokens, seed):
        (root / "distractor-0000.txt").write_text(
            f"{band} distractor\n", encoding="utf-8"
        )
        return builder.BAND_LIMITS[band][0]

    monkeypatch.setattr(builder, "scale_archive", fake_scale)
    output = tmp_path / "suite"

    runtime, gold = build_suite(
        output,
        candidates,
        split="dev",
        count_tokens=lambda text: len(text.split()),
        seed=17,
    )

    assert len(runtime) == len(gold) == 32
    assert len({case.archive_id for case in runtime}) == 16
    assert all("accepted" not in case.model_dump_json() for case in runtime)
    assert (output / "runtime.jsonl").is_file()
    assert (output / "gold.jsonl").is_file()
