from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from forgemind.config import RuntimeConfig
from forgemind.runtime import (
    LlamaClient,
    probe_hardware,
    start_with_single_fallback,
    used_vram_mib,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forgemind")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    raw = subparsers.add_parser("ask-raw")
    raw.add_argument("question")
    raw.add_argument("--context", required=True)
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
    benchmark = subparsers.add_parser("evaluate")
    benchmark.add_argument("cases")
    benchmark.add_argument("--db", required=True)
    benchmark.add_argument(
        "--systems", default="raw,vector,hybrid,forgemind"
    )
    benchmark.add_argument("--freeze", required=True)
    ask = subparsers.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--db", required=True)
    ask.add_argument(
        "--mode", choices=("retrieve", "reason", "investigate"), default="reason"
    )
    ask.add_argument("--json", action="store_true", dest="as_json")
    web = subparsers.add_parser("web")
    web.add_argument("--db", required=True)
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--summary")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--runs", type=int, default=10)
    smoke.add_argument("--offline", action="store_true")
    profile = subparsers.add_parser("profile-scale")
    profile.add_argument("root")
    profile.add_argument("--db", required=True)
    profile.add_argument("--max-active-tokens", type=int, default=16_384)
    return parser


def profile_scale(root: Path, db: Path, max_active_tokens: int) -> dict[str, object]:
    import time

    if not 0 <= max_active_tokens <= 16_384:
        raise ValueError("max active tokens exceeds ForgeMind hard limit")

    from forgemind.ingest import ingest_project
    from forgemind.retrieval import Embedder
    from forgemind.store import ForgeStore

    embedder = Embedder()
    store = ForgeStore(db)
    store.enable_vectors(embedder.dimensions)
    started = time.perf_counter()
    counts = ingest_project(root, store, embedder)
    elapsed = time.perf_counter() - started
    return {
        "sources": counts["sources"],
        "chunks": counts["chunks"],
        "events": counts["events"],
        "ingest_seconds": elapsed,
        "max_active_tokens": max_active_tokens,
    }


def _build_stack(config: RuntimeConfig, database: Path):
    from forgemind.reasoning import ReasoningController
    from forgemind.retrieval import Embedder, Retriever
    from forgemind.store import ForgeStore

    if not database.is_file():
        raise ValueError(f"archive database not found: {database}")
    embedder = Embedder()
    store = ForgeStore(database)
    store.enable_vectors(embedder.dimensions)
    client = LlamaClient(config)
    retriever = Retriever(store, embedder)
    controller = ReasoningController(
        retriever,
        client,
        client.count_tokens,
    )
    return store, retriever, client, controller


def _build_service(config: RuntimeConfig, database: Path):
    from forgemind.reasoning import InvestigationService

    store, _retriever, _client, controller = _build_stack(config, database)
    return InvestigationService(controller, store)


def _build_evaluation_systems(config: RuntimeConfig, database: Path):
    from forgemind.eval import ControlledSystems

    store, retriever, client, controller = _build_stack(config, database)
    return ControlledSystems(
        store,
        retriever,
        controller,
        client,
        client.count_tokens,
        used_vram_mib,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "profile-scale":
        result = profile_scale(Path(args.root), Path(args.db), args.max_active_tokens)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "smoke":
        if not args.offline:
            raise ValueError("only the deterministic offline smoke is available")
        from forgemind.offline import run_offline_smoke

        result = run_offline_smoke(args.runs)
        print(json.dumps(result, indent=2))
        return 0 if result["completed"] == args.runs else 1

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
    if args.command == "evaluate":
        from forgemind.eval import (
            EvaluationRunner,
            freeze_results,
            load_cases,
            parse_system_names,
        )

        names = parse_system_names(args.systems)
        cases = load_cases(Path(args.cases))
        with start_with_single_fallback(config) as server:
            systems = _build_evaluation_systems(server.config, Path(args.db))
            runners = {name: getattr(systems, name) for name in names}
            runs = EvaluationRunner(runners).run(cases, names)
        freeze_results(Path(args.freeze), cases, runs, names)
        errors = sum(run.error is not None for run in runs)
        print(
            json.dumps(
                {
                    "cases": len(cases),
                    "systems": names,
                    "runs": len(runs),
                    "errors": errors,
                },
                indent=2,
            )
        )
        return 1 if errors else 0
    if args.command == "doctor":
        print(
            json.dumps(
                {"hardware": asdict(probe_hardware()), "runtime": config.as_dict()}, indent=2
            )
        )
        return 0
    if args.command in {"ask", "web"}:
        with start_with_single_fallback(config) as server:
            service = _build_service(server.config, Path(args.db))
            if args.command == "web":
                import uvicorn

                from forgemind.web import create_app

                summary = Path(args.summary) if args.summary else None
                uvicorn.run(
                    create_app(service, summary_path=summary),
                    host="127.0.0.1",
                    port=args.port,
                )
                return 0
            answer = service.ask(args.question, args.mode)
        if args.as_json:
            print(answer.model_dump_json(indent=2))
        else:
            print(answer.summary)
            for claim in answer.claims:
                print(f"- {claim.text} [{', '.join(claim.evidence_ids)}]")
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
