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
On the four datasets that did not inform Sift's defaults, mean nDCG@10 is
0.337 for Sift, 0.328 for BM25 (Terrier), and 0.340 for BM25 (bm25s).

Sift's defaults were chosen on SciFact, NFCorpus, and FiQA, so those results are optimistic.
SciDocs and the two CQADupStack subsets were measured for the first time in this run.
ArguAna appeared in earlier reports but was not in the tuning grid.

nDCG@10 / Recall@100 on BEIR test judgments, 100 results per query. Higher is better.

| Dataset | Documents | Sift | BM25 (Terrier) | BM25 (bm25s) |
|---|---:|---:|---:|---:|
| SciFact (tuning) | 5,183 | 0.696 / 0.926 | 0.684 / 0.926 | 0.686 / 0.913 |
| NFCorpus (tuning) | 3,633 | 0.332 / 0.253 | 0.328 / 0.248 | 0.321 / 0.247 |
| FiQA (tuning) | 57,638 | 0.262 / 0.563 | 0.253 / 0.559 | 0.251 / 0.559 |
| **Mean, tuning** | | **0.430 / 0.581** | **0.422 / 0.578** | **0.420 / 0.573** |
| ArguAna | 8,674 | 0.449 / 0.945 | 0.491 / 0.970 | 0.493 / 0.966 |
| SciDocs | 25,657 | 0.160 / 0.364 | 0.158 / 0.362 | 0.158 / 0.365 |
| CQADupStack Android | 22,998 | 0.406 / 0.703 | 0.373 / 0.704 | 0.397 / 0.707 |
| CQADupStack Webmasters | 17,405 | 0.333 / 0.614 | 0.290 / 0.582 | 0.313 / 0.601 |
| **Mean, not tuned** | | **0.337 / 0.656** | **0.328 / 0.655** | **0.340 / 0.660** |

Against BM25 (Terrier), Sift is 0.042 nDCG@10 lower on ArguAna and 0.033 to 0.043 higher on CQADupStack.

Query latency in milliseconds, median / p95 over all judged queries.
One query at a time on one thread, after one warmup pass, over two measured passes.
Lower is better.

| Dataset | Sift, Rust | Sift, Python C API | bm25s, Python | Terrier, PyTerrier | Sift, HTTP |
|---|---:|---:|---:|---:|---:|
| SciFact | 0.16 / 0.27 | 0.31 / 0.46 | 0.35 / 0.52 | 3.94 / 4.62 | 0.64 / 0.85 |
| NFCorpus | 0.04 / 0.09 | 0.19 / 0.30 | 0.27 / 0.43 | 3.35 / 4.35 | 0.47 / 0.66 |
| FiQA | 0.41 / 0.73 | 0.56 / 0.91 | 0.80 / 1.09 | 4.50 / 6.14 | 1.03 / 1.41 |
| ArguAna | 1.17 / 2.31 | 1.36 / 2.72 | 1.81 / 9.92 | 7.43 / 10.75 | 1.93 / 3.20 |
| SciDocs | 0.21 / 0.37 | 0.38 / 0.56 | 0.63 / 0.83 | 4.41 / 5.56 | 0.82 / 1.07 |
| CQADupStack Android | 0.19 / 0.33 | 0.33 / 0.54 | 0.52 / 0.73 | 4.25 / 5.40 | 0.83 / 1.09 |
| CQADupStack Webmasters | 0.19 / 0.33 | 0.33 / 0.49 | 0.44 / 0.63 | 4.08 / 4.95 | 0.82 / 1.07 |

- **Sift, Rust**: in-process engine time. Wall time of one `Engine::search` call.
- **Sift, Python C API** and **bm25s, Python**: in-process library time from the same Python loop.
  Sift's time includes JSON request and response handling. The bm25s time includes query tokenization and stemming.
- **Terrier, PyTerrier**: in-process PyTerrier call, including the Python to Java boundary.
- **Sift, HTTP**: end-to-end local HTTP request to `sift serve`, including JSON encoding.

From the same Python loop, Sift's median was 1.1 to 1.6 times lower than bm25s on every dataset.
This compares two library calls, not two engines in the same language.
There is no native Rust BM25 baseline here.

Hardware: Apple M1 Ultra, 20 CPU cores, CPU only. Other processes ran on the machine during the test.
Sift built every index with default settings. bm25s used k1=1.5, b=0.75, English stopwords, and Snowball stemming.
Terrier used BM25 with its default English pipeline.

An [earlier IR Bench run](https://github.com/AghoFru/ir-bench/tree/main/results/beir-cpu) on SciFact, NFCorpus,
and ArguAna also measured neural retrievers, with encoders limited to 4 of 20 CPU threads.
Its mean nDCG@10 was 0.553 for BGE-small, 0.523 for SPLADE, 0.514 for Weaviate hybrid search,
and 0.501 for BM25 (Terrier).
Those systems are more relevant than Sift on those datasets. They run a model at query time.

See the [raw results, commands, and benchmark code](docs/benchmarks/).

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache-2.0](LICENSE)
