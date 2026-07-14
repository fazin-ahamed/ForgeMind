# Third-party notices

ForgeMind's MIT license applies only to this repository's original source. It does not relicense model weights, runtime binaries, parsers, imported repositories, or benchmark data. Users must obtain those assets from their upstream publishers and comply with the applicable terms.

## Tested runtime assets

| Asset | Tested identity | Canonical source | Upstream license |
|---|---|---|---|
| Qwen3-4B Q4_K_M GGUF | Qwen upload commit `a9a60d0`; SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF/blob/a9a60d009fa7ff9606305047c2bf77ac25dbec49/Qwen3-4B-Q4_K_M.gguf) | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/LICENSE) |
| llama.cpp | release `b9994`, commit `14d3ba4`; tested `llama-server.exe` SHA-256 `3c31f53feeff4d43dbb62b307023a44946074948cf925034d120089c95e20b8c` | [ggml-org/llama.cpp b9994](https://github.com/ggml-org/llama.cpp/releases/tag/b9994) | [MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) |
| BGE small English v1.5 | model revision `5c38ec7` | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) | MIT |
| sqlite-vec | Python package `0.1.9` | [asg017/sqlite-vec](https://github.com/asg017/sqlite-vec/tree/v0.1.9) | [MIT or Apache-2.0](https://github.com/asg017/sqlite-vec) |
| Tree-sitter language pack | Python package `0.13.0` | [xberg-io/tree-sitter-language-pack v0.13.0](https://github.com/xberg-io/tree-sitter-language-pack/tree/v0.13.0) | MIT for the pack; Apache-2.0 notice retained from the original package; individual grammars retain their own licenses |
| Tree-sitter Python runtime | Python package `0.26.0` | [tree-sitter/py-tree-sitter](https://github.com/tree-sitter/py-tree-sitter) | MIT |

The language pack bundles many independently maintained grammars. Review the pack's generated license metadata before redistributing its wheels or grammar binaries.

## Benchmarks and example data

| Asset | Pinned identity | Canonical source | License / handling |
|---|---|---|---|
| ForgeBench generator and showcase repository | ForgeMind repository revision used for the run; seed recorded with each generated archive | `benchmarks/generate_archive.py` and `examples/showcase-repository/` | ForgeMind MIT license; generated private archives are not distributed |
| LongBench v2 | dataset revision `2b48e49` | [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2/tree/2b48e49) | Apache-2.0 |
| SWE-bench Verified | importer revision `fd80552` | [SWE-bench/SWE-bench_Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) | SWE-bench tooling is MIT; issue text and patches originate from third-party repositories and retain their original licenses. Imported rows are private evaluation inputs and are not redistributed here. |
| RULER | record the evaluated commit and pipeline branch with every result | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) | Apache-2.0; downloaded auxiliary QA/text datasets retain their own terms |

External benchmark scores are reported under their native task names and metrics. ForgeMind does not merge imported benchmark material into ForgeBench or commit imported rows.

## Python dependencies

Python dependencies and their exact resolved versions are recorded in `uv.lock`. Each package retains its upstream license. This notice is a practical inventory, not legal advice; inspect the resolved distribution metadata for redistribution decisions.
