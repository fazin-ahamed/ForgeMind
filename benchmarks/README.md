# ForgeBench reproduction

ForgeBench tests whether one unchanged Qwen3-4B model benefits from ForgeMind's context organization. The primary comparison runs raw archive, vector retrieval, hybrid retrieval, and ForgeMind with the same model file and runtime configuration. Gold answers are stored separately and are never accepted by the evaluation command.

RepoQA, LongMemEval, and RULER cases in ForgeBench are deterministic **derived** slices. They are not official leaderboard submissions. External benchmark scores must retain their official task name, source revision, and native metric and must not be merged into ForgeBench scores.

## Pinned sources

| Source | Pin | Use |
|---|---|---|
| [RepoQA](https://github.com/evalplus/repoqa) | code `ae876deb1365dbf5a15b0533723c8ed123eee586`; release `2024-06-23` | repository navigation |
| [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | code `9e0b455f4ef0e2ab8f2e582289761153549043fc`; data `98d7416c24c778c2fee6e6f3006e7a073259d48f` | temporal and updated memory |
| [RULER](https://github.com/NVIDIA/RULER) | `e8bbff677ca2c239640dc90f93310dcf32408c93` | derived multi-key retrieval |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | CPU embeddings |

Downloaded datasets, generated archives, gold manifests, databases, runs, and reports stay below `.forgemind-private/`, which is ignored by Git.

## Development workflow

Install both development and dataset-import dependencies:

```powershell
uv sync --extra dev --extra eval
```

Set `FORGEMIND_LLAMA_SERVER` and `FORGEMIND_MODEL` as shown in the root README. Import the pinned sources:

```powershell
uv run python benchmarks/import_external.py repoqa --output .forgemind-private/benchmarks/sources/repoqa.json
uv run python benchmarks/import_external.py longmemeval --output .forgemind-private/benchmarks/sources/longmemeval.json
```

Start the local llama.cpp server, then construct the 32-case development matrix using its exact `/tokenize` endpoint:

```powershell
uv run python -m benchmarks.build_forgebench --split dev --output .forgemind-private/benchmarks/dev --repoqa .forgemind-private/benchmarks/sources/repoqa.json --longmemeval .forgemind-private/benchmarks/sources/longmemeval.json --tokenizer-url http://127.0.0.1:8080 --seed 20260714
```

Validate and freeze the runtime/gold boundary, prepare one resumable database per shared archive, evaluate all four systems, and generate the deterministic report:

```powershell
uv run forgemind benchmark-validate .forgemind-private/benchmarks/dev/runtime.jsonl .forgemind-private/benchmarks/dev/gold.jsonl --expected-per-cell 2 --freeze .forgemind-private/benchmarks/dev/manifest.json
uv run forgemind benchmark-prepare .forgemind-private/benchmarks/dev/runtime.jsonl --db-root .forgemind-private/benchmarks/dev/databases
uv run forgemind evaluate .forgemind-private/benchmarks/dev/runtime.jsonl --db-root .forgemind-private/benchmarks/dev/databases --runs .forgemind-private/runs/dev --run-group dev-20260714
uv run forgemind benchmark-report .forgemind-private/benchmarks/dev/runtime.jsonl .forgemind-private/benchmarks/dev/gold.jsonl --runs .forgemind-private/runs/dev/runs.jsonl --output .forgemind-private/results/dev-summary.json
```

`evaluate` appends every terminal record immediately. Repeating the identical command resumes missing case/system pairs. It freezes `run-manifest.json` only after the requested matrix is complete. `benchmark-report` still writes its report when a success gate fails, then exits with status one.

The isolated raw-32K sanity check is secondary and never enters the primary gates:

```powershell
uv run forgemind evaluate .forgemind-private/benchmarks/final/runtime.jsonl --db-root .forgemind-private/benchmarks/final/databases --archive-band 32k --systems raw32 --runs .forgemind-private/runs/final-raw32 --run-group final-raw32-20260804
```

## What is measured

The frozen report includes answer F1, required-fact recall, citation precision/recall/validity, retrieval recall@20, abstention F1, unsupported-answer rate, paired 10,000-resample confidence intervals, latency, prompt and cumulative tokens, peak sampled VRAM, indexing time and size, malformed outputs, and terminal errors.

The headline gate requires at least a five-point ForgeMind answer-F1 gain over the best primary baseline, positive paired intervals against every baseline, wins in at least three capabilities, 90% citation precision, 80% citation recall, 100% citation validity, non-inferior abstention, and no primary prompt above 15,616 tokens. Failed gates narrow the public claim; cases or baselines are never removed after inspection.

ForgeMind reasons over million-token information spaces through indexing, retrieval, exact-source rehydration, and bounded reasoning cycles. It does not directly attend to one million tokens.

## Separate external checks

- LongBench v2: `uv run python benchmarks/import_external.py longbench-v2 --output .forgemind-private/results/external/longbench-v2-code.jsonl --limit 25`
- SWE-bench Verified: `uv run python benchmarks/import_external.py swebench-verified --output .forgemind-private/results/external/swebench-verified.jsonl --limit 25`
- RULER: run the pinned official pipeline and copy its native result JSON into `.forgemind-private/results/external/ruler/`.

These checks remain separate from ForgeBench and retain their upstream licenses and evaluation rules.
