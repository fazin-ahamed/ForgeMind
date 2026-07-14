from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from forgemind.config import RuntimeConfig
from forgemind.runtime import LlamaClient, probe_hardware, start_with_single_fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forgemind")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    ask = subparsers.add_parser("ask-raw")
    ask.add_argument("question")
    ask.add_argument("--context", required=True)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("root")
    ingest.add_argument("--db", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument("--db", required=True)
    search.add_argument("--limit", type=int, default=20)
    evaluate = subparsers.add_parser("eval-retrieval")
    evaluate.add_argument("cases")
    evaluate.add_argument("--db", required=True)
    evaluate.add_argument("--min-recall20", type=float, default=0.70)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"ingest", "search", "eval-retrieval"}:
        from forgemind.ingest import ingest_project
        from forgemind.retrieval import Embedder, Retriever
        from forgemind.store import ForgeStore

        embedder = Embedder()
        store = ForgeStore(Path(args.db))
        store.enable_vectors(embedder.dimensions)
        if args.command == "ingest":
            print(json.dumps(ingest_project(Path(args.root), store, embedder), indent=2))
            return 0

        retriever = Retriever(store, embedder)
        if args.command == "search":
            hits = retriever.search(args.query, args.limit)
            print(json.dumps([asdict(hit) for hit in hits], indent=2))
            return 0

        cases = [
            json.loads(line)
            for line in Path(args.cases).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not cases:
            raise ValueError("retrieval evaluation requires at least one case")
        recalls: list[float] = []
        for case in cases:
            found = {hit.path for hit in retriever.search(case["query"], 20)}
            gold = set(case["evidence_paths"])
            recalls.append(len(found & gold) / len(gold))
        recall20 = sum(recalls) / len(recalls)
        print(json.dumps({"recall20": recall20, "cases": len(cases)}, indent=2))
        return 0 if recall20 >= args.min_recall20 else 1

    config = RuntimeConfig.from_env(os.environ)
    if args.command == "doctor":
        print(
            json.dumps(
                {"hardware": asdict(probe_hardware()), "runtime": config.as_dict()}, indent=2
            )
        )
        return 0
    context = Path(args.context).read_text(encoding="utf-8")
    with start_with_single_fallback(config) as server:
        result = LlamaClient(server.config).complete(
            [
                {"role": "system", "content": "Answer only from the supplied context."},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {args.question}",
                },
            ]
        )
    print(json.dumps(asdict(result) | {"total_ms": result.total_ms}, indent=2))
    return 0
