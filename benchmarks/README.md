# External checks

Install import support with `uv sync --extra eval`.

- LongBench v2: `uv run python benchmarks/import_external.py longbench-v2 --output data/private/external/longbench-v2-code.jsonl --limit 25`
- SWE-bench Verified: `uv run python benchmarks/import_external.py swebench-verified --output data/private/external/swebench-verified.jsonl --limit 25`
- RULER: clone `https://github.com/NVIDIA/RULER`, record the evaluated commit, use its current `rulerv1-ns` or `rulerv2-ns` pipeline, and copy official result JSON into `.forgemind-private/results/external/ruler/`.

Do not merge external scores into ForgeBench. Report every external benchmark with its official task name, source revision, and native metric. Import outputs remain under ignored private-data paths.
