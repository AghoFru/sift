# Sift

**A small, embeddable search engine for CPUs.**

Sift is a sparse keyword index in one Rust binary or library, with filters, facets, and document updates.
At index time, a small Model2Vec model adds related terms to each document.
Search uses only the index. No model runs at query time.

## Get started

Requires a stable Rust toolchain. From this checkout:

```sh
cargo build --release --locked
./target/release/sift build --input examples/animals.jsonl --out artifacts/animals.sift \
  --threshold 0.5 --k-expand 20 --stop-df 1.0
./target/release/sift search artifacts/animals.sift cat
```

The first index build downloads a model.
To search your own documents, replace the input with a JSONL file:

```json
{"id":"1","text":"A document to search","title":"Optional title"}
```

## Add it to your application

- [Embed with Rust or C, including an Android example](docs/EMBEDDING.md).
- [Run an HTTP service, update documents, and configure search](docs/REFERENCE.md).

## Relevance and speed

Relevance is close to BM25. Sift is not a dense or neural retriever.
Sift's defaults were chosen on SciFact, NFCorpus, and FiQA. The other four datasets did not inform them.

nDCG@10 / Recall@100 on BEIR test judgments, 100 results per query. Higher is better.

| Dataset | Documents | Sift | Tantivy | BM25 (Terrier) | BM25 (bm25s) |
|---|---:|---:|---:|---:|---:|
| SciFact (tuning) | 5,183 | 0.696 / 0.926 | 0.685 / 0.921 | 0.684 / 0.926 | 0.686 / 0.913 |
| NFCorpus (tuning) | 3,633 | 0.332 / 0.253 | 0.323 / 0.247 | 0.328 / 0.248 | 0.321 / 0.247 |
| FiQA (tuning) | 57,638 | 0.262 / 0.563 | 0.251 / 0.552 | 0.253 / 0.559 | 0.251 / 0.559 |
| **Mean, tuning** | | **0.430 / 0.581** | **0.419 / 0.574** | **0.422 / 0.578** | **0.420 / 0.573** |
| ArguAna | 8,674 | 0.449 / 0.945 | 0.462 / 0.951 | 0.491 / 0.970 | 0.493 / 0.966 |
| SciDocs | 25,657 | 0.160 / 0.364 | 0.154 / 0.358 | 0.158 / 0.362 | 0.158 / 0.365 |
| CQADupStack Android | 22,998 | 0.406 / 0.703 | 0.402 / 0.713 | 0.373 / 0.704 | 0.397 / 0.707 |
| CQADupStack Webmasters | 17,405 | 0.333 / 0.614 | 0.319 / 0.598 | 0.290 / 0.582 | 0.313 / 0.601 |
| **Mean, not tuned** | | **0.337 / 0.656** | **0.334 / 0.655** | **0.328 / 0.655** | **0.340 / 0.660** |

Tantivy is Tantivy 0.27 BM25 with its English stemming tokenizer.

Query latency in microseconds, median / p95. One query at a time on one CPU core. Lower is better.

| Dataset | Sift, Rust | Tantivy, Rust | Sift, Python C API | bm25s, Python |
|---|---:|---:|---:|---:|
| SciFact | 120 / 208 | 155 / 499 | 202 / 291 | 185 / 216 |
| FiQA | 311 / 554 | 958 / 2,847 | 387 / 620 | 501 / 666 |
| ArguAna | 944 / 1,754 | 14,600 / 49,783 | 1,034 / 1,831 | 619 / 946 |
| SciDocs | 169 / 289 | 306 / 713 | 253 / 373 | 383 / 468 |
| CQADupStack Android | 147 / 256 | 405 / 1,211 | 223 / 332 | 283 / 366 |
| CQADupStack Webmasters | 147 / 250 | 312 / 1,022 | 222 / 320 | 254 / 318 |

- **Sift, Rust**: one `Engine::search` call.
- **Tantivy, Rust**: query analysis, one Tantivy search for the top 100, and the lookup of 100 document IDs.
- **Sift, Python C API**: one call to the Sift C API from Python, including JSON request and response handling.
- **bm25s, Python**: one bm25s call, including query tokenization and stemming.

In Rust, Sift's median is lower than Tantivy's on every dataset: 1.3 times lower on SciFact,
1.8 to 3.1 times lower on FiQA, SciDocs, and CQADupStack, and 15 times lower on ArguAna.
From Python, Sift's median is lower than bm25s on FiQA, SciDocs, and CQADupStack, and higher on SciFact and ArguAna.

Hardware for latency: NVIDIA GB10 Arm CPU, 10 Cortex-X925 and 10 Cortex-A725 cores. Each measurement ran on one Cortex-X925 core.
Each value is the median of five passes, with less than 5% difference between passes.

See the [raw results, commands, and benchmark code](docs/benchmarks/).

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache-2.0](LICENSE)
