# Sift

**Fast, CPU-only hybrid search.**

Sift combines exact keyword matching with semantic term expansion in one index.
Search runs offline, with no model inference at query time.

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

## Performance

BEIR subset: SciFact, NFCorpus, and ArguAna. Apple M1 Ultra, CPU only.
Higher nDCG@10 means better relevance. Lower times are better.

| System | Median nDCG@10 | Median query latency | Median ingestion |
|---|---:|---:|---:|
| Sift | 0.449 | 0.83 ms | 2.72 s |
| BM25 (Terrier) | 0.491 | 4.05 ms | 1.48 s |
| BGE-small | 0.603 | 14.09 ms | 175.42 s |
| SPLADE | 0.508 | 36.48 ms | 379.67 s |
| Weaviate hybrid (E5 + BM25) | 0.478 | 20.77 ms | 181.27 s |

Each dataset has equal weight. nDCG@10 is the median of dataset means.
Query latency is the median of dataset medians, including encoding and adapter overhead.
Ingestion is the median full index build time, with files already local.
See [datasets, test settings, and full results](https://github.com/AghoFru/ir-bench/tree/main/results/beir-cpu).

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache-2.0](LICENSE)
