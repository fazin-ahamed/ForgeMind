from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LONG_BENCH_REVISION = "2b48e49"
SWEBENCH_REVISION = "fd80552"


def patch_paths(patch: str) -> list[str]:
    return sorted(set(re.findall(r"^diff --git a/(.+?) b/", patch, flags=re.MULTILINE)))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def import_longbench_v2(output: Path, limit: int = 25) -> None:
    from datasets import load_dataset

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
    )[:limit]
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
    selected = sorted(dataset, key=lambda row: row["instance_id"])[:limit]
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
    parser.add_argument("dataset", choices=("longbench-v2", "swebench-verified"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    function = (
        import_longbench_v2
        if args.dataset == "longbench-v2"
        else import_swebench_verified
    )
    function(args.output, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
