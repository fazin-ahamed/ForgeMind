# ForgeMind

ForgeMind is a local, evidence-constrained investigation runtime for reasoning over software archives. It combines hybrid retrieval, reversible TokenForge compression, bounded evidence packs, a persistent reasoning ledger, exact-source verification, and a local web interface.

The current vertical runs Qwen3-4B through `llama.cpp`, answers from a SQLite archive, and removes claims whose cited source span or source hash cannot be verified.

## Quickstart

Requirements: Python 3.11+, `uv`, a `llama-server` executable, and a compatible GGUF model.

```powershell
uv sync --extra dev
$env:FORGEMIND_LLAMA_SERVER = "C:\path\to\llama-server.exe"
$env:FORGEMIND_MODEL = "C:\path\to\model.gguf"

uv run forgemind ingest examples/showcase-repository --db artifacts/showcase.sqlite
uv run forgemind ask "Why did sessions fail after the April migration?" --db artifacts/showcase.sqlite --mode investigate
uv run forgemind web --db artifacts/showcase.sqlite
```

The web interface binds to `http://127.0.0.1:8000` by default. Models, indexes, benchmark outputs, private planning, research, and reports are intentionally excluded from version control.

## Verification

```powershell
uv run pytest -q
uv run forgemind smoke --runs 10 --offline
```

The synthetic archive under `examples/showcase-repository` provides a reproducible SQL, TypeScript, and log investigation without publishing private data.

To reproduce the ignored large-archive profile:

```powershell
uv run python benchmarks/generate_archive.py --root data/private/forgebench-1m --target-words 1000000 --seed 42
uv run forgemind profile-scale data/private/forgebench-1m --db artifacts/forgebench-1m.sqlite --max-active-tokens 16384
```
