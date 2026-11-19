# Benchmark results and commands

These files hold the measurements behind the [README tables](../../README.md#relevance-and-speed).

| File | Contents |
|---|---|
| [quality.csv](quality.csv) | Per dataset and system: nDCG@10, Recall@100, MRR@10, and MAP@100. |
| [latency.csv](latency.csv) | Per dataset and system: median and p95 query latency, passes, pass-to-pass spread, and core use by other processes. |
| [means.json](means.json) | Mean nDCG@10 and Recall@100 for the tuning and not-tuned dataset groups. |
| [matrix.json](matrix.json) | The IR Bench suite configuration for Sift, Terrier, and bm25s. |
| [inprocess_adapters.py](inprocess_adapters.py) | IR Bench adapters for bm25s and for Sift through the C API. |
| [tantivybench](tantivybench/src/main.rs) | The Rust program that builds Tantivy indexes and times Tantivy search. |
| [siftbench](siftbench/src/main.rs) | The Rust program that times Sift's `Engine::search`. |
| [tantivy_quality.py](tantivy_quality.py) | Builds the Tantivy indexes, exports TREC runs, and scores them with the IR Bench evaluator. |
| [latency.py](latency.py) | Runs the latency rounds and computes the latency summary. |
| [summarize.py](summarize.py) | Writes the CSV and JSON files. |

## Systems

| System | Configuration |
|---|---|
| `sift` | Sift with default settings. The quality rows come from `sift serve`. The latency rows use the same index. |
| `tantivy` | Tantivy 0.27.0 BM25 (k1=1.2, b=0.75), `en_stem` tokenizer, one segment. The query is the OR of its analyzed terms. |
| `terrier-bm25` | Terrier 5.11 BM25 through PyTerrier 1.1.2, Java 17, default English pipeline. |
| `bm25s` | bm25s 0.3.16. k1=1.5, b=0.75, Lucene scoring, English stopwords, PyStemmer English stemmer. |

All systems index the document title and text joined by a newline.
All searches return 100 results. Matching query and document IDs are removed, as in the BEIR evaluator.
ArguAna keeps its five judged documents that are absent from the corpus.
Every quality number comes from the IR Bench evaluator (ir-measures 0.4.3).

## Latency method

| Column in latency.csv | What it times |
|---|---|
| `sift-rust` | One `Engine::search` call in Rust, cache off, no payload. |
| `tantivy` | Query analysis, one `Searcher::search` with `TopDocs`, and the lookup of 100 document IDs, in Rust. |
| `sift-ffi` | One call to the Sift C API through ctypes in Python, including JSON request and response handling. |
| `bm25s` | One bm25s call in Python, including query tokenization and stemming. |

Both Rust programs and the Python loop call one query at a time on one thread.
A round runs each system and dataset cell once, in a new random order of cells.
Each cell is a new process, pinned to one Cortex-X925 core with `taskset`.
The process opens the index and warms up for at least 2 seconds.
Then it runs one measured pass of at least 1 second.
Warmup and the pass repeat all judged queries, each time in a new random order.
After the pass, one more iteration collects rankings. They must be identical to the quality run.
Before each cell, `latency.py` waits until the measurement core is less than 5% busy over 2 seconds.
While the cell runs, it compares the core's busy time with the measured process's own CPU time.
If other processes used 5% or more of the core, the cell runs again.
Each cell has five measured passes in five rounds. The tables report the median of the five pass
medians and the median of the five pass p95 values.
The pass-to-pass spread is the range of the five pass medians divided by their median.
The README latency table lists the datasets where every system's spread is below 5%.
In this run, that is every dataset except NFCorpus, where Sift's Rust spread was 7.8%.
latency.csv lists all 28 cells.

## Environment

Quality was measured on an Apple M1 Ultra (macOS 26.1). The rankings do not depend on the machine.
On the DGX, every latency cell returned the same rankings as the quality run.

- Sift indexes built at commit `994848c`. Searches ran at the commit that adds this file, with identical rankings.
- Tantivy 0.27.0. IR Bench at commit `6b92b4b`, Python 3.12, ir-datasets 0.6.3, ir-measures 0.4.3.
- Model `minishlab/potion-base-8M` at revision `bf8b056651a2c21b8d2565580b8569da283cab23`.

Latency was measured on an NVIDIA DGX Spark, CPU only.

- NVIDIA GB10 Arm CPU: 10 Cortex-X925 cores at up to 3.9 GHz and 10 Cortex-A725 cores at up to 2.8 GHz.
- 128 GB unified memory, Ubuntu 24.04, Linux 7.0, CPU frequency governor `performance`.
- Rust 1.100.0 (`--release`), Python 3.12.3, bm25s 0.3.16, PyStemmer 3.1.0, NumPy 2.5.5.
- The indexes are the same files as the quality runs, copied to the DGX.

## Commands

Run these from an IR Bench checkout at commit `6b92b4b`. Replace `SIFT_CHECKOUT` in `matrix.json`
with the path of this repository, and set `SIFT` to the same path in the shell.

```sh
# Sift binaries
(cd "$SIFT" && cargo build --release --locked && cargo build -p sift-ffi --release --locked)

# IR Bench with Terrier and bm25s
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[terrier]' bm25s PyStemmer numpy
uv run --no-project --with 'huggingface_hub[cli]' hf download minishlab/potion-base-8M \
  --revision bf8b056651a2c21b8d2565580b8569da283cab23 tokenizer.json model.safetensors \
  --local-dir work/model

# Quality for Sift, Terrier, and bm25s
PYTHONPATH="$SIFT/docs/benchmarks" .venv/bin/ir-bench \
  suite "$SIFT/docs/benchmarks/matrix.json" --output work/suite-main

# Rust bench programs
for bench in siftbench tantivybench; do
  CARGO_TARGET_DIR=work/bench cargo build --release \
    --manifest-path "$SIFT/docs/benchmarks/$bench/Cargo.toml"
done

# Tantivy indexes and quality
.venv/bin/python "$SIFT/docs/benchmarks/tantivy_quality.py" \
  "$SIFT/docs/benchmarks/matrix.json" work work/bench/release/tantivybench work/tantivy-quality

# Latency, Linux only: put siftbench, tantivybench, sift, and libsift_ffi.so in one folder.
# --cpu selects the core for taskset. Use a fast core that no other process is using.
mkdir -p work/tools
cp work/bench/release/siftbench work/bench/release/tantivybench work/tools/
cp "$SIFT/target/release/sift" "$SIFT/target/release/libsift_ffi.so" work/tools/
PYTHONPATH="$SIFT/docs/benchmarks" .venv/bin/python "$SIFT/docs/benchmarks/latency.py" run \
  "$SIFT/docs/benchmarks/matrix.json" work/suite-main work work/tools work/model work/latency 5 \
  --cpu=7

# CSV and JSON summaries
PYTHONPATH="$SIFT/docs/benchmarks" .venv/bin/python "$SIFT/docs/benchmarks/summarize.py" \
  work/suite-main work/tantivy-quality work/summary work/latency
```

To measure some cells again, run `latency.py run` with a new output folder and the cells as
`SYSTEM:DATASET` arguments. Then pass both folders to `summarize.py`. The later folder replaces
the earlier one for those cells.
