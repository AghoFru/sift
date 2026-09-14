# Sift

Embeddable search with semantic matching and offline queries.
Sift adds related terms when building the index, then searches it on the CPU.
Standard searches need no model or network connection.

## Try it

From this checkout, with a stable Rust toolchain:

```sh
cargo build --release --locked
./target/release/sift build --input examples/animals.jsonl --out artifacts/animals.sift \
  --threshold 0.5 --k-expand 20 --stop-df 1.0
./target/release/sift search artifacts/animals.sift cat
```

The first build downloads a model. Use `--model /path/to/model` for a local copy.
Use `--semantic-weight 0` for exact matching. The default is `0.5`.

Your own corpus uses one JSON document per line:

```json
{"id":"1","text":"A document to search","title":"Optional title"}
```

## Use it

- **Embed:** [Rust and C APIs, Android example](docs/EMBEDDING.md).
- **Serve:** `./target/release/sift serve --artifacts ./artifacts`.
- **Configure:** [HTTP requests, updates, and search options](docs/REFERENCE.md).
- **Evaluate:** [IR Bench](https://github.com/AghoFru/ir-bench).

An index is a directory. Sift is single-node and does not provide database
transactions. Semantic expansion can introduce irrelevant matches, so evaluate
it on your own documents and queries.

[Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache-2.0](LICENSE)
