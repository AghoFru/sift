# sift

**A local search engine with SQLite-shaped operations.**

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

## BM25, BM25 plus HNSW, and Sift

These systems solve different parts of the search problem:

| | BM25 | BM25 plus HNSW | Sift |
|---|---|---|---|
| Main signal | Exact term matching | Lexical and vector matching | Exact and build-time semantic postings |
| Query path | Posting lists | Query embedding, HNSW search, score fusion | Posting lists and BM25-style scoring |
| Storage | One lexical index | Usually lexical and vector indexes | One portable sparse artifact |
| Query-time model | No | Usually required | No |
| Updates | One index | Keep lexical and vector data consistent | Append segments, then compact |
| Best fit | Precise lexical search | Strong semantic retrieval with extra infrastructure | Local, CPU-first semantic recall |

HNSW is a good choice when vector similarity is the primary retrieval signal.
Sift is a good choice when lexical behavior, simple operations, and predictable
CPU cost matter more. Sift does not replace a full contextual reranker or a
vector database for every workload.

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

For order-sensitive queries, build an optional composition sidecar and enable
its rerank weight only when the target corpus has labels that reward this mode:

```bash
sift build --input corpus.jsonl --out artifacts/docs.sift --compositional
sift search artifacts/docs.sift "cat eats mouse" \
  --semantic-weight 0.5 --composition-weight 0.7
```

The HTTP equivalent is `"composition_weight": 0.7`. The sidecar combines
mean, alternating-position, and position-weighted pools of the active static
term vectors. This preserves the fast sparse candidate search and adds a
bounded order-aware rerank. The default value is `0`. Local BEIR evaluation
showed that positive composition weights reduced ranking quality, even though
candidate recall stayed unchanged. Composition reranking requires a single
segment. Run `sift compact` after incremental updates.

For full query-document context, build the optional cross-encoder feature and
start the server with the included MiniLM model:

```bash
cargo build --release --features cross-encoder
sift serve --artifacts ./artifacts \
  --cross-encoder ./reranker/ce-minilm-l6
```

Set `"contextual_weight"` in a search request to blend the cross-encoder score
with the sparse score. Use `1.0` for the contextual score within the rerank
window. This mode reads the query and document together, so it can distinguish
word order and phrases. It adds model inference cost and requires a
single-segment index.

For a CPU-only learned reranker, build the artifacts with `--compositional` so
the model can use qexp and composition evidence as candidate features. Then
train a LightGBM LambdaMART model and load it when serving:

```bash
uv run --no-project --with lightgbm --with requests --with numpy \
  reranker/train_lgbm.py --sift http://127.0.0.1:8080 \
  --train scifact fiqa --ood nfcorpus --out reranker/reranker.lgb.json
sift serve --artifacts ./artifacts --reranker reranker/reranker.lgb.json
```

When loaded, the native tree reranker is on by default for basic searches.
Set `"rerank": false` to compare the underlying Sift ordering. The model
combines Sift's exact, semantic, blended, qexp, composition, coverage, length,
and rank signals. The direct qexp and composition weights can stay at `0`
because the tree learns when those features help. It adds no query-time text
encoder and cannot recover documents missing from the candidate window.

Sift also recognizes common negative scopes such as `animal not cat`,
`animal other than cats`, `animal without cats`, and equivalent cues in several
common languages. It removes the negative phrase from the positive query and
applies an exact exclusion before ranking. It also maps common plural forms to
their singular token. This keeps the rule CPU-only and bounded. Use explicit
`-term` syntax when the query needs an exact exclusion.

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
term neighbors improve recall, while the optional composition sidecar adds
limited order sensitivity for selected corpora. Neither mode understands
natural-language negation or full document context as well as a contextual
model.

## Measured performance

### Same-machine BM25 comparison

Measured on an Apple M1 Ultra with 5,183 SciFact documents. Both variants were
built by the same release binary with the same tokenizer. Build order alternated
over five runs. Query results cover 600 uncached searches per engine through the
same server. Engine latency excludes HTTP and JSON overhead.

| engine | mean build | artifact | mean query | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| exact BM25 | 2.96 s | 24.0 MB | 11.1 µs | 10 µs | 22 µs |
| sift semantic | 2.81 s | 29.6 MB | 14.1 µs | 12 µs | 32 µs |

At this corpus size, build times are effectively in the same range. Semantic
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

### Same-suite retrieval quality

The local BEIR suite contains SciFact, NFCorpus, and FiQA, with 1,271 queries
in total. These numbers use the same Sift release binary and candidate depth.
The BM25 row disables semantic expansion and order bonuses. The HNSW hybrid
configuration was not benchmarked in this repository, so no unsupported HNSW
number is shown.

| method | nDCG@10 | MRR@10 | Recall@100 |
|---|---:|---:|---:|
| Exact BM25 | 0.4648 | 0.4857 | 0.5723 |
| Sift 2 sparse | 0.4879 | 0.5096 | 0.5812 |
| Sift 2 integrated tree | 0.4952 | 0.5177 | 0.5812 |
| Sift 2 contextual | 0.5096 | 0.5334 | 0.5812 |

Against exact BM25, the measured Sift 2 sparse path adds `+0.0231` nDCG@10,
`+0.0239` MRR@10, and `+0.0089` Recall@100. The integrated tree uses qexp and
composition as learned features. The contextual row requires a model and is
corpus-dependent.

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

## The SQLite for search idea

Sift is designed to have the operational shape of SQLite for search:

- One local artifact that can be copied, backed up, or memory-mapped
- An embedded Rust core and a single CPU-first binary
- Deterministic search without a query-time model or network dependency
- Incremental writes, deletes, compaction, filtering, facets, and pagination
- A CLI for local use and HTTP for applications that need a process boundary

Sift is not an SQLite extension today. It does not provide SQL or transactions
through the search API. The natural next integration is a small C ABI and an
SQLite virtual table that maps rows to a Sift artifact while leaving normal
SQLite queries and transactions in control of application data.

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
evidence. Sift's static modes do not fully model:

- natural-language negation in every domain
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
