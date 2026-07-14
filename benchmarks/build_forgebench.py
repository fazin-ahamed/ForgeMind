from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Literal, cast

from benchmarks.generate_archive import generate_distractors
from forgemind.benchmark import (
    BAND_LIMITS,
    AnswerSpec,
    ArchiveBand,
    Capability,
    CitationSpan,
    GoldCase,
    RuntimeCase,
    sha256_path,
    validate_bundle,
)
from forgemind.domain import SourceRecord, StrictModel


RULER_REVISION = "e8bbff677ca2c239640dc90f93310dcf32408c93"
CAPABILITIES: tuple[Capability, ...] = (
    "repository",
    "memory",
    "effective-context",
    "adversarial",
)


def build_ruler_case(
    root: Path,
    seed: int,
    pairs: int = 8,
) -> tuple[RuntimeCase, GoldCase]:
    if pairs < 3:
        raise ValueError("RULER-derived cases require at least three key-value pairs")
    rng = random.Random(seed)
    keys = [f"KEY-{rng.randrange(10**9):09d}" for _ in range(pairs)]
    values = [f"VALUE-{rng.randrange(10**9):09d}" for _ in range(pairs)]
    lines = [
        f"Registry entry {key} maps to {value}."
        for key, value in zip(keys, values, strict=True)
    ]
    root.mkdir(parents=True, exist_ok=True)
    source = root / "registry.txt"
    text = "\n".join(lines) + "\n"
    source.write_text(text, encoding="utf-8")
    record = SourceRecord.from_text("registry.txt", text, 0)
    selected = (1, pairs // 2, pairs - 1)
    question = "Return the values for " + ", ".join(
        keys[index] for index in selected
    ) + "."
    evidence = [
        CitationSpan(
            source_id=record.id,
            source_sha256=record.sha256,
            path=record.path,
            start_line=index + 1,
            end_line=index + 1,
        )
        for index in selected
    ]
    runtime = RuntimeCase(
        id=f"ruler-{seed}",
        question=question,
        capability="effective-context",
        archive_band="32k",
        archive_id=f"candidate-ruler-{seed}",
        archive_path=str(root),
        archive_sha256=sha256_path(root),
        archived_tokens=max(1, len(text.split())),
    )
    gold = GoldCase(
        case_id=runtime.id,
        answer=AnswerSpec(
            kind="set",
            accepted=[values[index] for index in selected],
            case_sensitive=True,
        ),
        evidence=evidence,
        required_facts=[values[index] for index in selected],
        source="NVIDIA/RULER-derived",
        source_revision=RULER_REVISION,
    )
    return runtime, gold


def _adversarial_case(
    root: Path,
    seed: int,
    index: int,
) -> tuple[RuntimeCase, GoldCase]:
    case_id = f"adversarial-{seed}-{index:03d}"
    case_root = root / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    variant = index % 3
    if variant == 0:
        question = f"What is the verified checksum for release ticket R-{seed}-{index}?"
        lines = [
            f"Ticket R-{seed}-{index} was opened for a release.",
            "A draft checksum was proposed but explicitly marked unverified.",
            "The verification record is missing from this archive.",
        ]
        answer = None
        answer_absent = True
        evidence_lines: tuple[int, ...] = ()
        source_kind = "missing-evidence"
    elif variant == 1:
        old_value = f"REGION-OLD-{seed:03d}-{index:03d}"
        current_value = f"REGION-NEW-{seed:03d}-{index:03d}"
        question = f"Which region currently owns service S-{seed}-{index}?"
        lines = [
            f"Service S-{seed}-{index} was originally assigned to {old_value}.",
            "A migration review approved changing the service owner.",
            "The original assignment is superseded and must not be used.",
            f"The current owner for service S-{seed}-{index} is {current_value}.",
        ]
        answer = AnswerSpec(kind="exact", accepted=[current_value], case_sensitive=True)
        answer_absent = False
        evidence_lines = (3, 4)
        source_kind = "superseded"
    else:
        style = (index // 3) % 4
        answer_value = f"TARGET-{seed:03d}-{index:03d}"
        if style == 0:
            question = f"What final target follows incident I-{seed}-{index}?"
            hops = 3 + index % 6
            lines = [f"Incident I-{seed}-{index} points to STEP-1."]
            lines.extend(
                f"STEP-{hop} points to STEP-{hop + 1}." for hop in range(1, hops)
            )
            lines.append(f"STEP-{hops} resolves to {answer_value}.")
            evidence_lines = tuple(range(1, len(lines) + 1))
            source_kind = "causal-chain"
        elif style == 1:
            question = f"Which identifier belongs to the engineer named Jordan-{index}?"
            lines = [
                f"Jordan-{index}, the analyst, uses identifier DECOY-{seed}-{index}.",
                f"Jordan-{index}, the engineer, uses identifier {answer_value}.",
            ]
            evidence_lines = (2,)
            source_kind = "same-name"
        elif style == 2:
            question = f"Which target handles delayed jobs for queue Q-{seed}-{index}?"
            lines = [
                f"When Q-{seed}-{index} cannot dispatch work promptly, its deferred-task sink is {answer_value}."
            ]
            evidence_lines = (1,)
            source_kind = "weak-overlap"
        else:
            question = f"What is the approved target for change C-{seed}-{index}?"
            lines = [
                f"Candidate DECOY-A-{index} was plausible but rejected.",
                f"Candidate DECOY-B-{index} was tested but not approved.",
                f"Change C-{seed}-{index} was approved for {answer_value}.",
            ]
            evidence_lines = (3,)
            source_kind = "plausible-distractor"
        answer = AnswerSpec(kind="exact", accepted=[answer_value], case_sensitive=True)
        answer_absent = False

    path = case_root / "facts.md"
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    record = SourceRecord.from_text("facts.md", text, 0)
    evidence = [
        CitationSpan(
            source_id=record.id,
            source_sha256=record.sha256,
            path=record.path,
            start_line=line,
            end_line=line,
        )
        for line in evidence_lines
    ]
    runtime = RuntimeCase(
        id=case_id,
        question=question,
        capability="adversarial",
        archive_band="32k",
        archive_id=f"candidate-{case_id}",
        archive_path=str(case_root),
        archive_sha256=sha256_path(case_root),
        archived_tokens=max(1, len(text.split())),
    )
    gold = GoldCase(
        case_id=case_id,
        answer=answer,
        evidence=evidence,
        required_facts=[] if answer is None else answer.accepted,
        answer_absent=answer_absent,
        source=f"ForgeBench/adversarial/{source_kind}",
        source_revision=f"generator-v1:{seed}",
    )
    return runtime, gold


def build_adversarial_cases(
    root: Path,
    count: int,
    seed: int,
) -> list[tuple[RuntimeCase, GoldCase]]:
    return [_adversarial_case(root, seed, index) for index in range(count)]


def _archive_text(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def scale_archive(
    root: Path,
    band: ArchiveBand,
    distractors: list[str],
    count_tokens: Callable[[str], int],
    seed: int,
) -> int:
    if not distractors:
        raise ValueError("at least one distractor template is required")
    low, high = BAND_LIMITS[band]
    exact = count_tokens(_archive_text(root))
    if exact > high:
        raise ValueError(f"archive already exceeds {band} upper bound")
    if exact >= low:
        return exact

    rng = random.Random(seed)
    for index in range(2_000):
        text = distractors[rng.randrange(len(distractors))]
        (root / f"distractor-{index:04d}.txt").write_text(
            text.rstrip() + "\n",
            encoding="utf-8",
        )
        if index >= 10 and index % 5:
            continue
        exact = count_tokens(_archive_text(root))
        if low <= exact <= high:
            return exact
        if exact > high:
            raise ValueError(f"archive exceeded {band} upper bound")
    raise RuntimeError(f"archive did not reach {band} token band")


def rebase_span(span: CitationSpan, case_id: str) -> CitationSpan:
    path = f"{case_id}/{span.path}"
    source_id = hashlib.sha256(
        f"{path}\0{span.source_sha256}".encode("utf-8")
    ).hexdigest()
    return span.model_copy(update={"path": path, "source_id": source_id})


def select_cell(
    candidates: list[tuple[RuntimeCase, GoldCase]],
    capability: Capability,
    band: ArchiveBand,
    count: int,
    seed: int,
) -> list[tuple[RuntimeCase, GoldCase]]:
    eligible = sorted(
        (pair for pair in candidates if pair[0].capability == capability),
        key=lambda pair: pair[0].id,
    )
    rng = random.Random(f"{seed}:{capability}")
    rng.shuffle(eligible)
    band_index = list(BAND_LIMITS).index(band)
    start = band_index * count
    selected = eligible[start : start + count]
    if len(selected) != count:
        raise ValueError(f"not enough {capability} candidates for {band}")
    return selected


def write_jsonl(path: Path, rows: Sequence[StrictModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )


def llama_token_count(base_url: str, text: str) -> int:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/tokenize",
        data=json.dumps({"content": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("llama.cpp tokenize response omitted tokens")
    return len(tokens)


def _cell_seed(seed: int, capability: Capability, band: ArchiveBand) -> int:
    digest = hashlib.sha256(f"{seed}:{capability}:{band}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _copy_candidate(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"candidate archive is not a directory: {source}")
    shutil.copytree(source, destination)


def build_suite(
    output: Path,
    candidates: list[tuple[RuntimeCase, GoldCase]],
    split: Literal["dev", "final"],
    count_tokens: Callable[[str], int],
    seed: int,
) -> tuple[list[RuntimeCase], list[GoldCase]]:
    expected_per_cell = 2 if split == "dev" else 10
    runtime_rows: list[RuntimeCase] = []
    gold_rows: list[GoldCase] = []
    archives_root = output / "archives"
    output.mkdir(parents=True, exist_ok=True)

    for capability in CAPABILITIES:
        for raw_band in BAND_LIMITS:
            band = cast(ArchiveBand, raw_band)
            selected = select_cell(
                candidates,
                capability,
                band,
                expected_per_cell,
                seed,
            )
            archive_id = f"{split}-{capability}-{band}"
            archive_root = archives_root / archive_id
            if archive_root.exists():
                raise FileExistsError(f"archive already exists: {archive_root}")
            archive_root.mkdir(parents=True)
            cell_runtime: list[RuntimeCase] = []
            cell_gold: list[GoldCase] = []
            for runtime, gold in selected:
                _copy_candidate(Path(runtime.archive_path), archive_root / runtime.id)
                cell_runtime.append(runtime)
                cell_gold.append(
                    gold.model_copy(
                        update={
                            "evidence": [
                                rebase_span(span, runtime.id)
                                for span in gold.evidence
                            ]
                        }
                    )
                )

            token_count = scale_archive(
                archive_root,
                band,
                generate_distractors(_cell_seed(seed, capability, band)),
                count_tokens,
                seed=_cell_seed(seed, capability, band),
            )
            archive_sha256 = sha256_path(archive_root)
            runtime_rows.extend(
                runtime.model_copy(
                    update={
                        "archive_band": band,
                        "archive_id": archive_id,
                        "archive_path": str(archive_root),
                        "archive_sha256": archive_sha256,
                        "archived_tokens": token_count,
                    }
                )
                for runtime in cell_runtime
            )
            gold_rows.extend(cell_gold)

    validate_bundle(runtime_rows, gold_rows, expected_per_cell=expected_per_cell)
    write_jsonl(output / "runtime.jsonl", runtime_rows)
    write_jsonl(output / "gold.jsonl", gold_rows)
    return runtime_rows, gold_rows


def main() -> int:
    from benchmarks.import_external import (
        longmemeval_candidates,
        repoqa_candidates,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repoqa", type=Path, required=True)
    parser.add_argument("--longmemeval", type=Path, required=True)
    parser.add_argument("--tokenizer-url", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    candidate_root = args.output / "candidates"
    repoqa_source = json.loads(args.repoqa.read_text(encoding="utf-8"))
    longmemeval_source = json.loads(args.longmemeval.read_text(encoding="utf-8"))
    candidates: list[tuple[RuntimeCase, GoldCase]] = []
    repository, repository_gold = repoqa_candidates(
        repoqa_source,
        candidate_root / "repository",
    )
    candidates.extend(zip(repository, repository_gold, strict=True))
    memory, memory_gold = longmemeval_candidates(
        longmemeval_source,
        candidate_root / "memory",
    )
    candidates.extend(zip(memory, memory_gold, strict=True))

    per_capability = 8 if args.split == "dev" else 40
    candidates.extend(
        build_ruler_case(
            candidate_root / "effective-context" / f"case-{index:03d}",
            seed=args.seed * 1_000 + index,
        )
        for index in range(per_capability)
    )
    candidates.extend(
        build_adversarial_cases(
            candidate_root / "adversarial",
            count=per_capability,
            seed=args.seed,
        )
    )
    build_suite(
        args.output,
        candidates,
        split=args.split,
        count_tokens=partial(llama_token_count, args.tokenizer_url),
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
