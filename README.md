# Sift

**Fast, CPU-only semantic search.**

Sift finds related terms as well as exact matches, with no model inference or
network connection at query time.

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
- [Compare retrieval quality with IR Bench](https://github.com/AghoFru/ir-bench).

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache-2.0](LICENSE)
