from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

from forgemind.benchmark import RuntimeCase
from forgemind.config import RuntimeConfig
from forgemind.runtime import (
    LlamaClient,
    probe_hardware,
    start_with_single_fallback,
    used_vram_mib,
)

if TYPE_CHECKING:
    from forgemind.eval import ControlledSystems
    from forgemind.reasoning import InvestigationService, ReasoningController
    from forgemind.retrieval import Retriever
    from forgemind.store import ForgeStore


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
    validate = subparsers.add_parser("benchmark-validate")
    validate.add_argument("runtime")
    validate.add_argument("gold")
    validate.add_argument("--expected-per-cell", type=int, required=True)
    validate.add_argument("--freeze", required=True)
    prepare = subparsers.add_parser("benchmark-prepare")
    prepare.add_argument("runtime")
    prepare.add_argument("--db-root", required=True)
    benchmark = subparsers.add_parser("evaluate")
    benchmark.add_argument("runtime")
    benchmark.add_argument("--db-root", required=True)
    benchmark.add_argument(
        "--systems", default="raw,vector,hybrid,forgemind"
    )
    benchmark.add_argument("--runs", required=True)
    benchmark.add_argument("--run-group", required=True)
    benchmark.add_argument(
        "--archive-band", choices=("32k", "100k", "250k", "1m")
    )
    report = subparsers.add_parser("benchmark-report")
    report.add_argument("runtime")
    report.add_argument("gold")
    report.add_argument("--runs", required=True)
    report.add_argument("--output", required=True)
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
    smoke.add_argument("--jsonl")
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


def _archive_groups(
    cases: list[RuntimeCase],
) -> dict[str, list[RuntimeCase]]:
    groups: defaultdict[str, list[RuntimeCase]] = defaultdict(list)
    for case in cases:
        if re.fullmatch(r"[A-Za-z0-9._-]+", case.archive_id) is None:
            raise ValueError(f"unsafe archive ID: {case.archive_id}")
        groups[case.archive_id].append(case)
    for archive_id, grouped in groups.items():
        signatures = {
            (
                case.archive_path,
                case.archive_sha256,
                case.archived_tokens,
                case.archive_band,
            )
            for case in grouped
        }
        if len(signatures) != 1:
            raise ValueError(f"archive metadata differs within {archive_id}")
    return dict(sorted(groups.items()))


def validate_benchmark(
    runtime_path: Path,
    gold_path: Path,
    expected_per_cell: int,
    freeze: Path,
    model: Path,
) -> dict[str, object]:
    from forgemind.benchmark import (
        LONGMEMEVAL_CODE_REVISION,
        LONGMEMEVAL_DATA_REVISION,
        REPOQA_CODE_REVISION,
        REPOQA_DATA_VERSION,
        RULER_REVISION,
        load_gold_cases,
        load_runtime_cases,
        sha256_path,
        validate_bundle,
    )

    if freeze.exists():
        raise FileExistsError(f"benchmark manifest already exists: {freeze}")
    runtime = load_runtime_cases(runtime_path)
    gold = load_gold_cases(gold_path)
    validate_bundle(runtime, gold, expected_per_cell=expected_per_cell)
    groups = _archive_groups(runtime)
    archive_hashes: dict[str, str] = {}
    for archive_id, cases in groups.items():
        archive = Path(cases[0].archive_path)
        actual = sha256_path(archive)
        if actual != cases[0].archive_sha256:
            raise ValueError(f"archive hash mismatch: {archive_id}")
        archive_hashes[archive_id] = actual
    counts = Counter(
        f"{case.capability}/{case.archive_band}" for case in runtime
    )
    payload: dict[str, object] = {
        "runtime_sha256": sha256_path(runtime_path),
        "gold_sha256": sha256_path(gold_path),
        "model_sha256": sha256_path(model),
        "archives": len(groups),
        "archive_sha256": archive_hashes,
        "cases": len(runtime),
        "matrix": dict(sorted(counts.items())),
        "source_pins": {
            "repoqa_code": REPOQA_CODE_REVISION,
            "repoqa_data": REPOQA_DATA_VERSION,
            "longmemeval_code": LONGMEMEVAL_CODE_REVISION,
            "longmemeval_data": LONGMEMEVAL_DATA_REVISION,
            "ruler": RULER_REVISION,
        },
    }
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def prepare_benchmark(
    runtime_path: Path,
    database_root: Path,
) -> list[dict[str, object]]:
    from forgemind.benchmark import load_runtime_cases, sha256_path
    from forgemind.ingest import ingest_project
    from forgemind.retrieval import EMBEDDER_MODEL, EMBEDDER_REVISION, Embedder
    from forgemind.store import ForgeStore

    groups = _archive_groups(load_runtime_cases(runtime_path))
    database_root.mkdir(parents=True, exist_ok=True)
    embedder = Embedder()
    embedder_identity = f"{EMBEDDER_MODEL}@{EMBEDDER_REVISION}"
    results: list[dict[str, object]] = []
    for archive_id, cases in groups.items():
        case = cases[0]
        archive = Path(case.archive_path)
        if sha256_path(archive) != case.archive_sha256:
            raise ValueError(f"archive hash mismatch: {archive_id}")
        database = database_root / f"{archive_id}.sqlite"
        sidecar_path = database_root / f"{archive_id}.json"
        if database.is_file() and sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if (
                sidecar.get("archive_sha256") == case.archive_sha256
                and sidecar.get("embedder_revision") == embedder_identity
                and sidecar.get("database_sha256") == sha256_path(database)
            ):
                results.append(sidecar | {"status": "reused"})
                continue
        for artifact in (
            database,
            Path(f"{database}-wal"),
            Path(f"{database}-shm"),
        ):
            if artifact.exists():
                artifact.unlink()
        store = ForgeStore(database)
        store.enable_vectors(embedder.dimensions)
        started = time.perf_counter()
        try:
            counts = ingest_project(archive, store, embedder)
        finally:
            store.close()
        sidecar = {
            "archive_id": archive_id,
            "archive_sha256": case.archive_sha256,
            "embedder_revision": embedder_identity,
            "ingest_seconds": time.perf_counter() - started,
            "database_bytes": database.stat().st_size,
            "database_sha256": sha256_path(database),
            **counts,
        }
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(sidecar | {"status": "built"})
    return results


def _prepared_database(case: RuntimeCase, database_root: Path) -> Path:
    from forgemind.benchmark import sha256_path
    from forgemind.retrieval import EMBEDDER_MODEL, EMBEDDER_REVISION

    if sha256_path(Path(case.archive_path)) != case.archive_sha256:
        raise ValueError(f"archive hash mismatch: {case.archive_id}")
    database = database_root / f"{case.archive_id}.sqlite"
    sidecar_path = database_root / f"{case.archive_id}.json"
    if not database.is_file() or not sidecar_path.is_file():
        raise ValueError(f"prepared database is missing: {case.archive_id}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("archive_sha256") != case.archive_sha256:
        raise ValueError(f"prepared archive hash mismatch: {case.archive_id}")
    identity = f"{EMBEDDER_MODEL}@{EMBEDDER_REVISION}"
    if sidecar.get("embedder_revision") != identity:
        raise ValueError(f"prepared embedder mismatch: {case.archive_id}")
    if sidecar.get("database_sha256") != sha256_path(database):
        raise ValueError(f"prepared database hash mismatch: {case.archive_id}")
    return database


def report_benchmark(
    runtime_path: Path,
    gold_path: Path,
    runs_path: Path,
    output: Path,
) -> dict[str, object]:
    from forgemind.benchmark import (
        load_gold_cases,
        load_runtime_cases,
        sha256_path,
        success_gates,
        summarize_benchmark,
    )
    from forgemind.eval import load_runs

    if output.exists():
        raise FileExistsError(f"benchmark report already exists: {output}")
    manifest_path = runs_path.parent / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("runs_sha256") != sha256_path(runs_path):
        raise ValueError("frozen run hash does not match run manifest")
    provenance = manifest.get("provenance", {})
    if (
        isinstance(provenance, dict)
        and provenance.get("runtime_sha256") is not None
        and provenance.get("runtime_sha256") != sha256_path(runtime_path)
    ):
        raise ValueError("runtime hash does not match run provenance")
    summary = summarize_benchmark(
        load_runtime_cases(runtime_path),
        load_gold_cases(gold_path),
        load_runs(runs_path),
    )
    gates = success_gates(summary)
    indexing = provenance.get("indexing", []) if isinstance(provenance, dict) else []
    seconds = [float(item["ingest_seconds"]) for item in indexing]
    database_bytes = sum(int(item["database_bytes"]) for item in indexing)
    index_summary = {
        "archives": len(indexing),
        "total_seconds": sum(seconds),
        "median_seconds": statistics.median(seconds) if seconds else 0.0,
        "database_bytes": database_bytes,
    }
    payload: dict[str, object] = {
        "summary": summary,
        "gates": gates,
        "indexing": index_summary,
        "runtime_sha256": sha256_path(runtime_path),
        "gold_sha256": sha256_path(gold_path),
        "run_manifest": manifest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _benchmark_provenance(
    config: RuntimeConfig,
    model_sha256: str,
    config_sha256: str,
    database_root: Path,
    runtime_path: Path,
    cases: list[RuntimeCase],
) -> dict[str, object]:
    from forgemind.benchmark import sha256_path

    source_revision = os.environ.get("FORGEMIND_SOURCE_REVISION")
    dirty_value = os.environ.get("FORGEMIND_DIRTY_WORKTREE")
    if source_revision is not None:
        if dirty_value not in {"true", "false"}:
            raise ValueError(
                "FORGEMIND_DIRTY_WORKTREE must be true or false when "
                "FORGEMIND_SOURCE_REVISION is set"
            )
        dirty_worktree = dirty_value == "true"
    else:
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_worktree = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    execution_runtime_sha256 = sha256_path(runtime_path)
    frozen_runtime_sha256 = os.environ.get(
        "FORGEMIND_RUNTIME_SHA256", execution_runtime_sha256
    )
    provenance: dict[str, object] = {
        "source_revision": source_revision,
        "dirty_worktree": dirty_worktree,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": asdict(probe_hardware()),
        "runtime": config.as_dict(),
        "model_sha256": model_sha256,
        "config_sha256": config_sha256,
        "runtime_sha256": frozen_runtime_sha256,
        "archive_sha256": {
            case.archive_id: case.archive_sha256
            for case in cases
        },
        "benchmark_seed": 20_260_714,
        "indexing": [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(database_root.glob("*.json"))
        ],
    }
    if frozen_runtime_sha256 != execution_runtime_sha256:
        provenance["execution_runtime_sha256"] = execution_runtime_sha256
    return provenance


def evaluate_benchmark(
    runtime_path: Path,
    database_root: Path,
    systems_text: str,
    runs_directory: Path,
    run_group_id: str,
    archive_band: str | None,
    config: RuntimeConfig,
) -> dict[str, object]:
    from forgemind.benchmark import finalize_run_group, sha256_path
    from forgemind.eval import (
        EvaluationRunner,
        load_cases,
        load_runs,
        parse_system_names,
        write_run,
    )

    names = parse_system_names(systems_text)
    if "raw32" in names and archive_band != "32k":
        raise ValueError("raw32 requires --archive-band 32k")
    cases = load_cases(runtime_path)
    if archive_band is not None:
        cases = [case for case in cases if case.archive_band == archive_band]
    if not cases:
        raise ValueError("no benchmark cases match the requested archive band")
    benchmark_config = (
        replace(config, context_tokens=32_768)
        if "raw32" in names
        else config
    )
    runs_path = runs_directory / "runs.jsonl"
    if (runs_directory / "run-manifest.json").exists():
        raise FileExistsError(f"run group is already frozen: {runs_directory}")
    existing = load_runs(runs_path) if runs_path.is_file() else []
    if existing and {run.run_group_id for run in existing} != {run_group_id}:
        raise ValueError("existing runs belong to a different run group")
    runs_directory.mkdir(parents=True, exist_ok=True)
    groups = _archive_groups(cases)
    databases = {
        archive_id: _prepared_database(grouped[0], database_root)
        for archive_id, grouped in groups.items()
    }
    resolved_config: RuntimeConfig | None = None
    model_sha256: str | None = None
    config_sha256: str | None = None
    for archive_id, grouped in groups.items():
        with start_with_single_fallback(resolved_config or benchmark_config) as server:
            if "raw32" in names and server.config.context_tokens != 32_768:
                raise RuntimeError("raw32 server did not retain 32,768-token context")
            if resolved_config is None:
                resolved_config = server.config
                model_sha256 = sha256_path(server.config.model)
                config_sha256 = hashlib.sha256(
                    json.dumps(server.config.as_dict(), sort_keys=True).encode("utf-8")
                ).hexdigest()
            elif server.config != resolved_config:
                raise RuntimeError("benchmark server profile changed between archives")
            assert model_sha256 is not None
            assert config_sha256 is not None
            systems = _build_evaluation_systems(
                server.config,
                databases[archive_id],
                run_group_id,
                model_sha256,
                config_sha256,
            )
            functions = {name: getattr(systems, name) for name in names}
            new_runs = EvaluationRunner(functions, systems.error_record).run(
                grouped,
                names,
                existing=existing,
                on_run=partial(write_run, runs_path),
            )
            existing.extend(new_runs)
    assert resolved_config is not None
    assert model_sha256 is not None
    assert config_sha256 is not None
    provenance = _benchmark_provenance(
        resolved_config,
        model_sha256,
        config_sha256,
        database_root,
        runtime_path,
        cases,
    )
    finalize_run_group(runs_directory, cases, names, provenance)
    return {
        "cases": len(cases),
        "systems": names,
        "runs": len(existing),
        "errors": sum(run.error is not None for run in existing),
        "run_group": run_group_id,
    }


def _build_stack(
    config: RuntimeConfig, database: Path
) -> tuple[ForgeStore, Retriever, LlamaClient, ReasoningController]:
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


def _build_service(
    config: RuntimeConfig, database: Path
) -> InvestigationService:
    from forgemind.reasoning import InvestigationService

    store, _retriever, _client, controller = _build_stack(config, database)
    return InvestigationService(controller, store)


def _build_evaluation_systems(
    config: RuntimeConfig,
    database: Path,
    run_group_id: str,
    model_sha256: str | None = None,
    config_sha256: str | None = None,
) -> ControlledSystems:
    from forgemind.benchmark import sha256_path
    from forgemind.eval import ControlledSystems

    store, retriever, client, controller = _build_stack(config, database)
    return ControlledSystems(
        store,
        retriever,
        controller,
        client,
        client.count_tokens,
        used_vram_mib,
        run_group_id,
        model_sha256 or sha256_path(config.model),
        config_sha256
        or hashlib.sha256(
            json.dumps(config.as_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark-validate":
        config = RuntimeConfig.from_env(os.environ)
        validation = validate_benchmark(
            Path(args.runtime),
            Path(args.gold),
            args.expected_per_cell,
            Path(args.freeze),
            config.model,
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-prepare":
        prepared = prepare_benchmark(Path(args.runtime), Path(args.db_root))
        print(json.dumps(prepared, indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-report":
        report = report_benchmark(
            Path(args.runtime),
            Path(args.gold),
            Path(args.runs),
            Path(args.output),
        )
        gates = cast(dict[str, bool], report["gates"])
        print(json.dumps(gates, indent=2, sort_keys=True))
        return 0 if all(gates.values()) else 1
    if args.command == "profile-scale":
        profile = profile_scale(Path(args.root), Path(args.db), args.max_active_tokens)
        print(json.dumps(profile, indent=2))
        return 0
    if args.command == "smoke":
        if not args.offline:
            raise ValueError("only the deterministic offline smoke is available")
        from forgemind.offline import run_offline_smoke

        jsonl_path = Path(args.jsonl) if args.jsonl else None
        smoke_result = run_offline_smoke(args.runs, jsonl_path)
        print(json.dumps(smoke_result, indent=2))
        return 0 if smoke_result["completed"] == args.runs else 1

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
        evaluation = evaluate_benchmark(
            Path(args.runtime),
            Path(args.db_root),
            args.systems,
            Path(args.runs),
            args.run_group,
            args.archive_band,
            config,
        )
        print(json.dumps(evaluation, indent=2, sort_keys=True))
        return 1 if cast(int, evaluation["errors"]) else 0
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
        generation = LlamaClient(server.config).complete(
            [
                {"role": "system", "content": "Answer only from the supplied context."},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {args.question}",
                },
            ]
        )
    print(
        json.dumps(
            asdict(generation) | {"total_ms": generation.total_ms}, indent=2
        )
    )
    return 0
