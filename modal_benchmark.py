from __future__ import annotations

import json
import os
import subprocess
import zipfile
from pathlib import Path

import modal


app = modal.App("forgemind-dev-benchmark")
volume = modal.Volume.from_name("forgemind-benchmark", create_if_missing=True)
image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda12-b9994",
        add_python="3.11",
    )
    .entrypoint([])
    .pip_install(
        "httpx>=0.28,<1",
        "numpy>=2.3,<3",
        "pydantic>=2.11,<3",
        "sentence-transformers>=5,<6",
        "sqlite-vec>=0.1.9,<0.2",
        "tree-sitter-language-pack>=0.9,<1",
    )
    .env({"LD_LIBRARY_PATH": "/app"})
    .add_local_dir("src/forgemind", "/root/forgemind")
)


@app.function(image=image, gpu="L4", timeout=4 * 60 * 60, volumes={"/data": volume})
def evaluate(
    run_group: str, source_revision: str, dirty_worktree: bool
) -> dict[str, object]:
    from forgemind.cli import evaluate_benchmark
    from forgemind.benchmark import RuntimeCase, sha256_path
    from forgemind.config import RuntimeConfig

    os.environ["FORGEMIND_SOURCE_REVISION"] = source_revision
    os.environ["FORGEMIND_DIRTY_WORKTREE"] = str(dirty_worktree).lower()

    benchmark_root = Path("/tmp/benchmarks/dev")
    with zipfile.ZipFile("/data/uploads/dev-modal.zip") as archive:
        archive.extractall(benchmark_root)
    source = benchmark_root / "runtime.jsonl"
    os.environ["FORGEMIND_RUNTIME_SHA256"] = sha256_path(source)
    runtime = Path("/tmp/runtime-modal.jsonl")
    cases: list[RuntimeCase] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        case = RuntimeCase.model_validate_json(line)
        cases.append(
            case.model_copy(
                update={
                    "archive_path": str(
                        benchmark_root / "archives" / case.archive_id
                    )
                }
            )
        )
    runtime.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases),
        encoding="utf-8",
    )
    config = RuntimeConfig(
        llama_server=Path("/app/llama-server"),
        model=Path("/data/models/Qwen3-4B-Q4_K_M.gguf"),
    )
    result = evaluate_benchmark(
        runtime,
        benchmark_root / "databases",
        "raw,vector,hybrid,forgemind",
        Path("/data/runs") / run_group,
        run_group,
        None,
        config,
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main(run_group: str = "dev-modal-20260715") -> None:
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
    result = evaluate.remote(run_group, source_revision, dirty_worktree)
    print(json.dumps(result, indent=2, sort_keys=True))
