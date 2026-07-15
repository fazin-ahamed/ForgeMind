from __future__ import annotations

import argparse
import gzip
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from forgemind.benchmark import (
    AnswerSpec,
    CitationSpan,
    GoldCase,
    LONGMEMEVAL_CODE_REVISION,
    LONGMEMEVAL_DATA_REVISION,
    REPOQA_CODE_REVISION,
    REPOQA_DATA_VERSION,
    RuntimeCase,
    sha256_path,
)
from forgemind.domain import SourceRecord


LONG_BENCH_REVISION = "2b48e49"
SWEBENCH_REVISION = "fd80552"
REPOQA_URL = (
    "https://github.com/evalplus/repoqa_release/releases/download/"
    f"{REPOQA_DATA_VERSION}/repoqa-{REPOQA_DATA_VERSION}.json.gz"
)
LONGMEMEVAL_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    f"{LONGMEMEVAL_DATA_REVISION}/longmemeval_s_cleaned.json"
)


def patch_paths(patch: str) -> list[str]:
    return sorted(set(re.findall(r"^diff --git a/(.+?) b/", patch, flags=re.MULTILINE)))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def _contains_answer(text: str, answer: str) -> bool:
    tokens = [
        token.strip(".,;:")
        for token in re.findall(r"[\w./:-]+", text.casefold())
    ]
    target = [
        token.strip(".,;:")
        for token in re.findall(r"[\w./:-]+", answer.casefold())
    ]
    return bool(target) and any(
        tokens[index : index + len(target)] == target
        for index in range(len(tokens) - len(target) + 1)
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def import_repoqa(output: Path, limit: int = 0) -> None:
    compressed = output.with_suffix(output.suffix + ".gz")
    download(REPOQA_URL, compressed)
    with gzip.open(compressed, "rt", encoding="utf-8") as handle:
        source = json.load(handle)

    if limit:
        remaining = limit
        limited: dict[str, object] = {}
        for language, repositories in sorted(source.items()):
            selected = list(repositories)[:remaining]
            limited[language] = selected
            remaining -= len(selected)
            if remaining <= 0:
                break
        source = limited

    _write_json(output, source)


def import_longmemeval(output: Path, limit: int = 0) -> None:
    download(LONGMEMEVAL_URL, output)
    if not limit:
        return

    source = json.loads(output.read_text(encoding="utf-8"))
    selected = sorted(source, key=lambda item: item["question_id"])[:limit]
    _write_json(output, selected)


def repoqa_candidates(
    source: dict[str, list[dict[str, Any]]], root: Path
) -> tuple[list[RuntimeCase], list[GoldCase]]:
    runtime: list[RuntimeCase] = []
    gold: list[GoldCase] = []
    for language, raw_repositories in sorted(source.items()):
        repositories = list(raw_repositories)
        for repository in sorted(repositories, key=lambda item: item["repo"]):
            contents = dict(repository["content"])
            needles = sorted(repository.get("needles", []), key=lambda item: item["name"])
            repository_id = _slug(
                f"repoqa-{language}-{repository['repo']}-{repository['commit_sha']}"
            )
            case_root = root / repository_id
            for relative, text in sorted(contents.items()):
                path = case_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(text), encoding="utf-8")
            archive_sha256 = sha256_path(case_root)
            for needle in needles:
                case_id = _slug(
                    f"repoqa-{language}-{repository['repo']}-{needle['name']}"
                )
                target_path = str(needle["path"])
                target_text = str(contents[target_path])
                target = SourceRecord.from_text(target_path, target_text, 0)
                question = (
                    "Which function matches this behavior? "
                    + str(needle["description"]).strip()
                    + " Return only its function name."
                )
                runtime_case = RuntimeCase(
                    id=case_id,
                    question=question,
                    capability="repository",
                    archive_band="32k",
                    archive_id=f"candidate-{case_id}",
                    archive_path=str(case_root),
                    archive_sha256=archive_sha256,
                    archived_tokens=32_000,
                )
                runtime.append(runtime_case)
                gold.append(
                    GoldCase(
                        case_id=case_id,
                        answer=AnswerSpec(
                            kind="exact",
                            accepted=[str(needle["name"])],
                            case_sensitive=True,
                        ),
                        evidence=[
                            CitationSpan(
                                source_id=target.id,
                                source_sha256=target.sha256,
                                path=target_path,
                                start_line=int(needle["start_line"]) + 1,
                                end_line=int(needle["end_line"]),
                            )
                        ],
                        source="evalplus/repoqa",
                        source_revision=(
                            f"{REPOQA_CODE_REVISION}:{REPOQA_DATA_VERSION}:"
                            f"{repository['commit_sha']}"
                        ),
                    )
                )
    return runtime, gold


def longmemeval_candidates(
    source: list[dict[str, Any]], root: Path
) -> tuple[list[RuntimeCase], list[GoldCase]]:
    runtime: list[RuntimeCase] = []
    gold: list[GoldCase] = []
    for row in sorted(source, key=lambda item: str(item["question_id"])):
        question_id = str(row["question_id"])
        case_id = _slug(f"longmemeval-{question_id}")
        case_root = root / case_id
        evidence: list[CitationSpan] = []
        answer_absent = question_id.endswith("_abs")
        raw_answer = row["answer"]
        accepted = (
            [str(item) for item in raw_answer]
            if isinstance(raw_answer, list)
            else [str(raw_answer)]
        )
        sessions = list(row["haystack_sessions"])
        session_ids = list(row["haystack_session_ids"])
        dates = list(row["haystack_dates"])
        answer_session_ids = {
            str(item) for item in row.get("answer_session_ids", [])
        }
        evidence_sessions = {
            index
            for index, (session_id, raw_session) in enumerate(
                zip(session_ids, sessions, strict=True)
            )
            if str(session_id) in answer_session_ids
            or any(bool(turn.get("has_answer")) for turn in raw_session)
        }
        retained_sessions = evidence_sessions or {max(0, len(sessions) - 1)}
        for index, (session_id, date, raw_session) in enumerate(
            zip(session_ids, dates, sessions, strict=True)
        ):
            if index not in retained_sessions:
                continue
            path = case_root / f"session-{index:04d}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            turns = list(raw_session)
            lines = [
                f"# Session {session_id}",
                "",
                f"Session date: {date}",
                f"Question date: {row['question_date']}",
                "",
            ]
            matching_lines: list[int] = []
            marked_lines: list[int] = []
            for turn in turns:
                lines.append(
                    "- "
                    + json.dumps(
                        {
                            "role": turn["role"],
                            "content": turn["content"],
                        },
                        ensure_ascii=False,
                    )
                )
                if any(
                    _contains_answer(str(turn["content"]), answer)
                    for answer in accepted
                ):
                    matching_lines.append(len(lines))
                if bool(turn.get("has_answer")):
                    marked_lines.append(len(lines))
            answer_lines = (
                [] if answer_absent else matching_lines or marked_lines
            )
            text = "\n".join(lines) + "\n"
            path.write_text(text, encoding="utf-8")
            relative = path.relative_to(case_root).as_posix()
            record = SourceRecord.from_text(relative, text, 0)
            evidence.extend(
                CitationSpan(
                    source_id=record.id,
                    source_sha256=record.sha256,
                    path=relative,
                    start_line=line_number,
                    end_line=line_number,
                )
                for line_number in answer_lines
            )
        runtime.append(
            RuntimeCase(
                id=case_id,
                question=str(row["question"]),
                capability="memory",
                archive_band="32k",
                archive_id=f"candidate-{case_id}",
                archive_path=str(case_root),
                archive_sha256=sha256_path(case_root),
                archived_tokens=32_000,
            )
        )
        gold.append(
            GoldCase(
                case_id=case_id,
                answer=(
                    None
                    if answer_absent
                    else AnswerSpec(kind="text", accepted=accepted)
                ),
                evidence=evidence,
                answer_absent=answer_absent,
                source="xiaowu0162/longmemeval-cleaned",
                source_revision=(
                    f"{LONGMEMEVAL_CODE_REVISION}:{LONGMEMEVAL_DATA_REVISION}"
                ),
            )
        )
    return runtime, gold


def import_longbench_v2(output: Path, limit: int = 25) -> None:
    from datasets import load_dataset  # type: ignore[import-untyped]

    dataset = load_dataset(
        "THUDM/LongBench-v2", split="train", revision=LONG_BENCH_REVISION
    )
    selected = sorted(
        (
            row
            for row in dataset
            if "code" in f"{row['domain']} {row['sub_domain']}".lower()
        ),
        key=lambda row: row["_id"],
    )
    if limit:
        selected = selected[:limit]
    rows = [
        {
            "id": row["_id"],
            "question": row["question"],
            "choices": [
                row["choice_A"],
                row["choice_B"],
                row["choice_C"],
                row["choice_D"],
            ],
            "answer": row["answer"],
            "context": row["context"],
            "source": "THUDM/LongBench-v2",
            "revision": LONG_BENCH_REVISION,
        }
        for row in selected
    ]
    _write(output, rows)


def import_swebench_verified(output: Path, limit: int = 25) -> None:
    from datasets import load_dataset

    dataset = load_dataset(
        "SWE-bench/SWE-bench_Verified",
        split="test",
        revision=SWEBENCH_REVISION,
    )
    selected = sorted(dataset, key=lambda row: row["instance_id"])
    if limit:
        selected = selected[:limit]
    rows = [
        {
            "id": row["instance_id"],
            "question": row["problem_statement"],
            "evidence_paths": patch_paths(row["patch"]),
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "source": "SWE-bench/SWE-bench_Verified",
            "revision": SWEBENCH_REVISION,
        }
        for row in selected
    ]
    _write(output, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=(
            "repoqa",
            "longmemeval",
            "longbench-v2",
            "swebench-verified",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    functions = {
        "repoqa": import_repoqa,
        "longmemeval": import_longmemeval,
        "longbench-v2": import_longbench_v2,
        "swebench-verified": import_swebench_verified,
    }
    functions[args.dataset](args.output, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
