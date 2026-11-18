# Benchmark results and commands

These files hold the measurements behind the [README tables](../../README.md#relevance-and-speed).

| File | Contents |
|---|---|
| [results.csv](results.csv) | Per dataset and system: nDCG@10, Recall@100, MRR@10, MAP@100, p50 and p95 latency, latency boundary, and index build time. |
| [means.json](means.json) | Means for the tuning and not-tuned dataset groups. |
| [matrix.json](matrix.json) | The IR Bench suite configuration. |
| [inprocess_adapters.py](inprocess_adapters.py) | IR Bench adapters for bm25s and for Sift through the C API. |
| [siftbench](siftbench/src/main.rs) | The Rust program that times `Engine::search`. |
| [run_rust_bench.py](run_rust_bench.py), [summarize.py](summarize.py) | Run `siftbench` on each suite index, then write the CSV and JSON files. |

## Systems

| System in results.csv | What it measures |
|---|---|
| `sift` | Sift through `sift serve`. Latency is one local HTTP request, including JSON. |
| `sift-ffi` | Sift through the C API, called with ctypes from Python. Same index build as `sift`. |
| `sift-rust` | Wall time of one `Engine::search` call in Rust. Latency only. Its rankings match `sift` for every query. |
| `bm25s` | bm25s 0.3.16 in Python. k1=1.5, b=0.75, Lucene scoring, English stopwords, PyStemmer English stemmer. |
| `bm25` | Terrier 5.11 BM25 through PyTerrier 1.1.2, Java 17. |

All searches return 100 results, one query at a time, on one thread, with the Sift result cache off.
Each system ran one warmup pass and two measured passes over all judged queries.
Relevance uses the first measured pass. Matching query and document IDs are removed, as in the BEIR evaluator.
ArguAna keeps its five judged documents that are absent from the corpus.

## Environment

- Apple M1 Ultra, 20 CPU cores (16 performance, 4 efficiency), 128 GB memory, macOS 26.1. CPU only.
- Other processes ran during the test. The load average was about 5.
- Sift source at commit `994848c`, built with `cargo build --release --locked` and Rust 1.100.0.
- IR Bench at commit `6b92b4b`, Python 3.12.12, ir-datasets 0.6.3, ir-measures 0.4.3.
- Model `minishlab/potion-base-8M` at revision `bf8b056651a2c21b8d2565580b8569da283cab23`.

## Commands

Run these from an IR Bench checkout. Replace `SIFT_CHECKOUT` in `matrix.json` with the path of this repository,
and set `SIFT` to the same path in the shell.

```sh
# Sift binaries
(cd "$SIFT" && cargo build --release --locked && cargo build -p sift-ffi --release --locked)

# IR Bench with Terrier and bm25s
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[terrier]' bm25s PyStemmer numpy
uv run --no-project --with 'huggingface_hub[cli]' hf download minishlab/potion-base-8M \
  --revision bf8b056651a2c21b8d2565580b8569da283cab23 tokenizer.json model.safetensors \
  --local-dir work/model

# Relevance and Python-side latency for all four Python-driven systems
PYTHONPATH="$SIFT/docs/benchmarks" .venv/bin/ir-bench \
  suite "$SIFT/docs/benchmarks/matrix.json" --output work/suite-main

# In-process Rust latency on the same indexes
CARGO_TARGET_DIR=work/siftbench cargo build --release \
  --manifest-path "$SIFT/docs/benchmarks/siftbench/Cargo.toml"
PYTHONPATH="$SIFT/docs/benchmarks" .venv/bin/python "$SIFT/docs/benchmarks/run_rust_bench.py" \
  "$SIFT/docs/benchmarks/matrix.json" work/suite-main work work/siftbench/release/siftbench work/rust-out

# CSV and JSON summaries
.venv/bin/python "$SIFT/docs/benchmarks/summarize.py" work/suite-main work/rust-out work/summary
```

## Limits

- These are seven small English BEIR datasets, up to 57,638 documents. They are not a full BEIR evaluation.
- SciFact, NFCorpus, and FiQA informed Sift's default settings.
- The Python-side comparison measures library calls with different overheads, not two engines in one language.
- No native BM25 engine was measured in process. The Rust timing has no matching BM25 baseline.
- Each number comes from one run on a shared machine. Run-to-run variation was not measured.
