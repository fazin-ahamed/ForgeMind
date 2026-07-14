from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path


TEMPLATES = (
    "2026-04-18 migration user_identifier changed from INTEGER to UUID request_id={request_id} source=/src/features/authentication/session/decoder.ts",
    "2026-04-20 session decoder called parseInt for user_identifier request_id={request_id} source=/src/features/authentication/session/decoder.ts",
    "2026-04-21 unrelated billing reconciliation completed request_id={request_id} source=/src/features/authentication/session/decoder.ts",
)

DISTRACTOR_TEMPLATES = (
    "Routine audit event {event:09d} completed without changing authority.",
    "Background worker {event:09d} retried an unrelated cache refresh.",
    "Historical proposal {event:09d} was discussed but never approved.",
    "Telemetry sample {event:09d} contains no evidence for the active query.",
)


def generate_distractors(
    seed: int,
    documents: int = 8,
    lines_per_document: int = 128,
) -> list[str]:
    if documents < 1 or lines_per_document < 1:
        raise ValueError("distractor dimensions must be positive")
    randomizer = random.Random(seed)
    return [
        "\n".join(
            randomizer.choice(DISTRACTOR_TEMPLATES).format(
                event=randomizer.randrange(10**9)
            )
            for _ in range(lines_per_document)
        )
        for _ in range(documents)
    ]


def generate_archive(root: Path, target_words: int, seed: int) -> str:
    if target_words < 1:
        raise ValueError("target words must be positive")
    randomizer = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    words = 0
    file_index = 0
    digest = hashlib.sha256()
    while words < target_words:
        lines: list[str] = []
        for _ in range(500):
            line = randomizer.choice(TEMPLATES).format(
                request_id=randomizer.randrange(1_000_000)
            )
            lines.append(line)
            words += len(line.split())
        text = "\n".join(lines) + "\n"
        path = root / f"history-{file_index:04d}.log"
        path.write_text(text, encoding="utf-8")
        digest.update(path.name.encode("utf-8"))
        digest.update(text.encode("utf-8"))
        file_index += 1
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/private/forgebench-1m"))
    parser.add_argument("--target-words", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(generate_archive(args.root, args.target_words, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
