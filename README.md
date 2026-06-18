# sift

**Semantic recall with BM25-shaped operations.**

sift is a CPU-first search engine that expands documents with weighted semantic
neighbors when the index is built, then serves queries from one mmap-backed
sparse index. There is no embedding model, GPU, vector search, or second
database in the query path.

```text
build time:  text -> tokens -> semantic neighbors -> weighted posting lists
query time:  query tokens -> posting lists -> BM25-style scoring -> top-k
```

Use sift when exact lexical search is fast and trustworthy but misses vocabulary
mismatches such as `cat` versus `kitten`, translations, or morphological
variants.

## See the difference in one minute

The repository includes a tiny animal corpus:

```bash
cargo build --release
./target/release/sift build \
  --input examples/animals.jsonl \
  --out /tmp/animals.sift \
  --threshold 0.5 --k-expand 20 --stop-df 1.0
```

Exact BM25 only finds the literal term:

```console
$ sift search /tmp/animals.sift cat --semantic-weight 0
# 1 hits
cat     A domestic cat sleeps on the windowsill.
```

The same artifact with semantic expansion also finds the vocabulary mismatch:

```console
$ sift search /tmp/animals.sift cat --semantic-weight 0.5
# 2 hits
cat     A domestic cat sleeps on the windowsill.
kitten  A playful kitten chases a piece of string.
```

`--semantic-weight 0` is exact BM25, `1` is the fully expanded index, and
the default `0.5` keeps literal matches strong while allowing expansion-only
matches to surface.

## Why one sparse index

A conventional hybrid stack maintains a lexical index and a dense-vector index,
then fuses two result sets. Inserts, deletes, snapshots, replication, and
recovery must keep both stores consistent.

sift writes exact and semantically expanded postings into one versioned artifact
directory. Search is ordinary sparse retrieval over memory-mapped files.

- No model or network call at query time
- No vector index to operate
- Exact and semantic scores remain separately controllable
- A snapshot or replica is an ordinary directory copy
- The standard binary is Rust and CPU-only

This is **expansion-augmented sparse retrieval**, not a dense retriever. Static
term neighbors improve recall, but they do not understand phrases, negation, or
full document context as well as a contextual model.

## Measured performance

### Same-machine BM25 comparison

Measured on an Apple M1 Ultra with 5,183 SciFact documents. Both variants were
built by the same release binary with the same tokenizer. Build order alternated
over five runs; query results cover 600 uncached searches per engine through the
same server. Engine latency excludes HTTP and JSON overhead.

| engine | mean build | artifact | mean query | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| exact BM25 | 2.96 s | 24.0 MB | 11.1 µs | 10 µs | 22 µs |
| sift semantic | 2.81 s | 29.6 MB | 14.1 µs | 12 µs | 32 µs |

At this corpus size, build times are effectively in the same range; semantic
postings add about 24% to the artifact and roughly 3 µs to mean engine latency.
Run the benchmark on your hardware:

```bash
python3 benchmarks/compare.py artifacts/scifact.sift --runs 5 --query-repeat 30
```

The benchmark exports the original JSONL from a payload-bearing artifact,
rebuilds exact and semantic variants, alternates build order to reduce cache
bias, disables the result cache, and reports machine-readable JSON.

### Retrieval behavior

Expansion helps when relevant documents use related terms that are absent from
the query. It can also introduce weak associations. Use `--semantic-weight` to
control that tradeoff and evaluate it against relevance judgments from your own
corpus.

## Install and build

Requirements: Rust 1.75 or newer.

```bash
git clone https://github.com/AghoFru/sift
cd sift
cargo build --release
```

Build a JSONL corpus:

```json
{"id":"1","text":"A document to index"}
{"id":"2","text":"Another document","title":"Optional title"}
```

```bash
sift build --input corpus.jsonl --out artifacts/docs.sift
sift search artifacts/docs.sift "search terms" -k 10
```

For BEIR-shaped input, use `--format beir` with rows containing
`{_id, title, text}`.

The first build resolves the static embedding table used to create semantic
edges. Serving and searching the finished artifact do not load that table.

## Serve

```bash
sift serve --artifacts ./artifacts --bind 127.0.0.1:8080
```

```bash
curl -s http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"index":"docs","q":"search terms","k":10}'
```

The response contains ranked document IDs, scores, snippets, measured engine
latency, and optional stored payloads.

## Operational surface

The core path stays small, while optional sidecars and commands add production
features without changing the one-index model:

- Incremental segments, tombstone deletes, and compaction
- Consistent filesystem or HTTP replication
- Filtering, facets, numeric ranking fields, and pagination
- Phrase filtering and proximity or bigram bonuses
- Spell correction and query suggestions
- Exact/semantic score blending and explain output
- WAND and Block-Max WAND query execution
- Optional deduplication, PRF, MMR, and reranking

Detailed commands, HTTP fields, artifact layout, and feature tradeoffs live in
[the reference guide](docs/REFERENCE.md).

## When not to use sift

Use a dense retriever or reranker when contextual meaning dominates lexical
evidence. sift's static term expansion does not fully model:

- word order, negation, or compositional meaning
- long-query intent
- domain meanings absent from the embedding table
- contextual similarity between whole passages

A contextual reranker over sift's candidates is often the right compromise.
The reranker changes ordering, not retrieval storage.

sift is currently single-node. Immutable segments replicate cleanly, but there
is no distributed query fan-out or consensus layer.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
[RELEASE.md](RELEASE.md).

## License

Apache License 2.0.
