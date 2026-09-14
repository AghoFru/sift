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

SciFact: 5,183 documents, 300 queries, Apple M1 Ultra, CPU only.
Higher nDCG@10 means better relevance. Lower latency is better.

| System | nDCG@10 | Median query latency |
|---|---:|---:|
| Sift | 0.696 | 0.75 ms |
| BM25 (Terrier) | 0.684 | 4.20 ms |
| BGE-small | 0.713 | 15.17 ms |
| SPLADE | 0.708 | 37.41 ms |
| Weaviate hybrid (E5 + BM25) | 0.723 | 20.82 ms |

Top-100 query latency includes encoding and adapter overhead, including HTTP where used.
These results cover one query at a time on small English datasets.
See [full comparisons, test settings, and reproduction steps](https://github.com/AghoFru/ir-bench/tree/main/results/sift-cpu).

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache-2.0](LICENSE)
