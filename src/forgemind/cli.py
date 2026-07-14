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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
