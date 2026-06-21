# sift reference

Complete command, API, and artifact reference. For installation and a short
example, start with the [README](../README.md).

```bash
sift build --input corpus.jsonl --out my.sift --format beir
sift search my.sift "your query" -k 10
```

## What it is

sift is a sparse search index packaged as a portable artifact directory.
It scores like BM25, but the index is expanded at build time with semantic
neighbours from a static embedding lookup,
so it also matches morphological variants, synonyms, and (with a multilingual
model) across languages. You mmap the artifact and serve; there is no
transformer in the hot path.

**Category: expansion-augmented sparse.** It sits between classic sparse
(BM25 / Lucene / Tantivy / FTS5) and learned sparse (SPLADE / ELSER): it gets
the semantic-expansion benefit of learned sparse, but from a pretrained
embedding table applied once at index time: nothing trained, nothing run per
query. Query execution remains sparse and CPU-only.

## Why not a vector DB

Standard hybrid retrieval runs two physical stores (a lexical index and a
vector index) fused at query time. Every insert, update, and delete has to hit
both atomically: ghost results when one side is stale, partial-failure recovery
you own, coordinated migrations, dual-snapshot backups, sharding strategies that
have to agree.

sift stores exact and expanded postings in the same artifact. A backup is a
snapshot of that artifact (see Replication below), rather than a coordinated
dump of lexical and vector indices.

## Performance

The [benchmark results](../benchmarks/RESULTS.md) compare exact and expanded
retrieval from the same binary on SciFact. The
[benchmark harness](../benchmarks/compare.py) rebuilds both variants and emits
JSON for results collected on other machines or corpora.

## Controlling expansion

`--threshold` and `--k-expand` control which neighboring terms are written at
build time. `blend_alpha` controls how much their scores contribute at query
time. Lower thresholds add more neighbors and can improve recall at the cost of
more weak associations.

```bash
sift build --threshold 0.5 --k-expand 10 --input corpus.jsonl --out my.sift
# query: score = (1 - alpha)*exact + alpha*combined, alpha in [0,1], default 0.5
curl -sX POST localhost:8080/search -d '{"index":"my","q":"...","blend_alpha":0.4}'
```

Set `blend_alpha` to `0` for exact scoring and `1` for the fully expanded score.

## Quick start

```bash
cargo build --release

# index a JSONL corpus
./target/release/sift build --input corpus.jsonl --out artifacts/my.sift \
    --format beir

# query from the CLI
./target/release/sift search artifacts/my.sift "your query" -k 5

# or serve over HTTP
SIFT_ARTIFACTS=./artifacts SIFT_BIND=0.0.0.0:8080 ./target/release/sift serve
```

Posting weights are written as f16 by default (`data_f16.bin`,
`exact_data_f16.bin`). Pass `--f16-postings=false` to store f32 weights instead.
Older f32 artifacts load unchanged.

For edge / battery deployments, a smaller single-threaded build:

```bash
cargo build --release --no-default-features --features edge
# or cap threads at runtime: sift build --threads 1 ...
```

## Incremental updates

An index can grow without a full rebuild. Internally it becomes a set of
immutable **segments** plus a tombstone set; a query scores every segment and
merges the results, so adds and deletes are cheap and never touch existing data.

```bash
# append a new batch of documents as a fresh segment
./target/release/sift add --index artifacts/my.sift --input new_docs.jsonl

# delete by external id (tombstoned; filtered at query time)
./target/release/sift delete --index artifacts/my.sift --id doc-42 --id doc-99

# rebuild back into one segment, dropping tombstoned docs and reclaiming space
./target/release/sift compact --index artifacts/my.sift
```

A plain `sift build` artifact is a single-segment index; the first `add`/`delete`
migrates it into the segment layout transparently.

The cross-segment query path covers ranked search (BM25, WAND/Block-Max WAND,
blended synonym scoring) and highlighting, and scores every segment on
collection-wide IDF so a live index ranks like a compacted one. The richer
single-segment features (facets, rank tiers, MMR, PRF, filters, dedup,
pagination) run after a `compact`.

### Replication and backup

This is the same data model Lucene uses: immutable segment directories plus a
`manifest.json` "commit pointer" that is swapped atomically (temp file, fsync,
rename, directory fsync). Segments are never modified after they are written, so
copying a live index is consistent as long as you copy in the right order:
segment dirs first (they don't change), the manifest last. Any segment the new
manifest references is already fully on disk (it was fsync'd before the manifest
committed); extra segments the manifest doesn't reference yet are harmless and
get cleaned up on the next open. So:

- **Snapshot**: `POST /snapshot` tars a consistent point-in-time view of one
  index; copy or restore the tarball anywhere.
- **Replication**: `sift replicate` does the consistent copy for you.

```bash
# one-shot: bring a replica current with the source
./target/release/sift replicate --from /src/my.sift --to /replica/my.sift

# continuous: re-sync every 5s, copying only what changed
./target/release/sift replicate --from /src/my.sift --to /replica/my.sift --watch 5
```

It copies the segments the destination is missing first (immutable, so present
ones are skipped and only new segments transfer), mirrors the tombstone set,
then commits the manifest last; it is a no-op when the replica already matches
the source generation, and prunes segments a source-side `compact` replaced.
This is the commit-point file-copy scheme Solr's master/slave and Lucene's
`replicator` module formalize; segment immutability is what makes it safe.

`--from` is either a filesystem path (local disk or any mounted FS) or an
HTTP(S) URL served by another sift instance:

```bash
# pull over HTTP from a serve --enable-replication source
./target/release/sift serve --artifacts ./artifacts --enable-replication   # source
./target/release/sift replicate \
    --from https://search.internal/replicate/my --to /replica/my.sift --watch 5
# (add --token <key> when the source requires auth)
```

The source mounts two read-only endpoints, `/replicate/{index}/listing` (a
generation + file inventory) and `/replicate/{index}/file` (streams one file,
path-confined to the index directory). They are off unless you pass
`--enable-replication` (they serve the full index, document text included) and
sit behind the same auth as the other protected routes. The HTTP path uses the
exact same ordering as the filesystem path: missing segments stream into a temp
dir and rename in, then the manifest is adopted last.

What is *not* built in is per-operation log shipping (the document-forwarding +
translog approach Elasticsearch uses for sub-second replica lag); `replicate`
is commit-granular (a replica is current as of the last source commit it
pulled). See Limits.

### Live writes over HTTP

A running server can take writes directly, no rebuild and no reload. The
embedding model is loaded once and kept resident, each write builds a segment
and swaps it in atomically (immediately queryable), writes are serialized by a
single writer, and the server auto-compacts in the background once an index
crosses `--compact-threshold` segments.

```bash
# add documents to (or create) an index; docs carry arbitrary fields
curl -sX POST localhost:8080/add -d '{"index":"docs","docs":[
  {"id":"a1","text":"first document","brand":"acme","price":12},
  {"id":"a2","text":"second document","brand":"globex","price":40}]}'

# update an existing id (fully supersedes the old version)
curl -sX POST localhost:8080/add -d '{"index":"docs","upsert":true,"docs":[
  {"id":"a1","text":"first document, revised","brand":"acme","price":15}]}'

# delete by id; compact on demand
curl -sX POST localhost:8080/delete  -d '{"index":"docs","ids":["a1"]}'
curl -sX POST localhost:8080/compact -d '{"index":"docs"}'
```

Writes are durable: a freshly built segment is fsync'd before the manifest that
references it is committed (atomic rename + directory fsync), so once `/add` or
`/delete` returns, the change survives a crash. A crash mid-build leaves an
uncommitted segment that the server discards on the next startup.

**Read-only mode.** Start with `--read-only` (or `SIFT_READONLY=1`) and
the write and admin routes are not mounted at all (they return 404); only the
search API is reachable.

## Metadata payloads and filtering

Each document is stored as a JSON payload in the same artifact, so search
returns the source and you can filter on any field, no schema declared up front.

```bash
curl -sX POST localhost:8080/search -d '{
  "index":"docs", "q":"laptop", "k":10,
  "filter":[
    {"field":"brand","in":["dell","hp"]},
    {"field":"price","lt":1500},
    {"field":"in_stock","eq":true},
    {"field":"discount","exists":true}
  ]
}'
```

Predicates: `eq`/`neq` (string, number, bool), `in` (keyword set), `lt`/`lte`/
`gt`/`gte` (numeric range), `exists`. All clauses must pass (logical AND).
Filtering works on a live, multi-segment index too. The response carries each
hit's `payload` (set `"with_payload": false` to omit it) and a `total` count for
pagination (`offset` + `k`). `total` is exact within the retrieval candidate
window and may be capped for very broad queries. Pass `--no-payload` at build
to skip payload storage.

## HTTP API

The server discovers every `*.sift` directory under `SIFT_ARTIFACTS`:

- `POST /search`: `{"index","q","k", ...filters/facets/blend_alpha}`
- `POST /explain`: per-term score breakdown for a query
- `POST /suggest`: spell-corrected query suggestions
- `GET  /datasets`: loaded indices and their stats
- `GET  /stats`: per-index rolling-window latency
- `GET  /metrics`: Prometheus metrics
- `GET  /healthz`, `GET /readyz`, `GET /version`
- `GET  /`: version and loaded-index metadata

Write/admin routes (omitted in `--read-only` mode):

- `POST /add`: add documents to (or create) an index
- `POST /delete`: tombstone documents by id
- `POST /compact`: merge segments back into one
- `POST /reload`, `POST /snapshot`, `POST /alias`: index lifecycle

Replication routes (read-only, mounted only with `--enable-replication`):

- `GET /replicate/{index}/listing`: generation + file inventory
- `GET /replicate/{index}/file?path=<rel>`: stream one index file (path-confined)

```bash
curl -sX POST localhost:8080/search \
  -H 'content-type: application/json' \
  -d '{"index":"scifact","q":"cancer mortality","k":3}' | jq
```

```json
{
  "index": "scifact",
  "matched_terms": 2,
  "total": 18,
  "latency_us": 47,
  "hits": [
    { "doc_id": "21009874", "score": 10.5, "snippet": "Overall and cancer related mortality..." },
    { "doc_id": "52188256", "score": 10.4, "snippet": "Global cancer statistics 2018..." }
  ]
}
```

## Capabilities

- BM25 ranking with index-time semantic expansion (one static embedding lookup)
- Block-Max WAND for top-k at multi-million-doc scale
- Incremental add / delete / upsert via immutable segments, with `compact`
- Live writes over HTTP (`/add`, `/delete`) with resident model, crash-durable
  commits, and background auto-compaction; `--read-only` mode serves search only
- JSON payload per doc, returned in results; rich filtering (eq/in/range/exists)
- `total` match count and `offset`/`k` pagination
- Facets, field-weighted ranking, highlighting, dedup, MMR, PRF
- Spell correction and query suggestions
- Blended exact/expansion scoring for synonym recall (`blend_alpha`)
- BM25F-style title weighting (`--title-weight`) and subword damping (`--subword-weight`)
- Query-side expansion sidecar (`qexp_weight`) and term-proximity bonus (`proximity_weight`)
- Collection-wide IDF across segments: live multi-segment indices rank like a compacted one
- Corpus-fitted PPMI expansion (`--corpus-expand-weight`, opt-in, recall-oriented)
- Optional GBDT / ONNX cross-encoder reranking over the top-k (off by default)
- Multilingual / cross-lingual via `--model`
- f16 postings (default), u24 doc-id packing, i8 build-time embedding quantization
- Single binary, mmap'd artifact, no runtime dependencies

## Layout

```
sift/
├── pyproject.toml      Python reference indexer (parity oracle)
├── Cargo.toml          Rust workspace
├── sift_build/         Python reference impl, same artifact shape
├── crates/
│   ├── sift-core/      library: mmap an artifact, score it
│   └── sift/           binary: build, serve, search
└── artifacts/          built *.sift directories (gitignored)
```

A legacy Python reference indexer ships in `sift_build/`; `tests/parity_test.sh`
builds both implementations with an explicitly compatible configuration and
verifies retrieval matches on a canonical query set. The Rust builder is the
production implementation and source of truth for defaults.

## Optional reranking

A reranker reorders candidates after retrieval. Two implementations are
available and disabled by default:

- **ONNX cross-encoder** (`serve --cross-encoder <dir>`, built with
  `--features cross-encoder`). Scores `(query, document)` pairs jointly. The
  `reranker/` tooling exports an ONNX model and can distill one from a teacher.
- **GBDT reranker** (`serve --reranker model.json`). Native LightGBM-tree
  evaluation over per-hit features without an ONNX runtime.

Both honor `"rerank": false` per request, and both require the exact-CSR
sidecar. Evaluate ranking quality and latency on the target corpus before
enabling either implementation.

For recall rather than top-k precision, `--corpus-expand-weight` adds
corpus-fitted expansion edges from term co-occurrence (PPMI) on top of the
static-embedding neighbours: where the embedding table gives generic
substitutability, PPMI gives domain association (insulin <-> diabetes) learned
from the corpus itself, still consumed entirely at build time. It is off by
default because topical associations can add unrelated results.

## Limits

- **Semantic recall has bounds**: phrase-level and compositional meaning still
  benefit from a reranker over the top-k (which keeps the single index; it
  reorders results, it is not a second store).
- **Cross-segment statistics**: a multi-segment (live-written) index scores every
  segment on collection-wide IDF and average document length, recomputed at open,
  so its ranking matches a compacted single-segment index (verified exactly on a
  skewed two-segment corpus). The one approximation: tombstoned / upsert-superseded
  docs still count toward document frequency until a `compact` rebuilds the
  collection.
- **Single writer**: writes are serialized by one lock per server; the model is
  built for moderate write rates with background compaction, not high-QPS ingest.
- **Single-node**: `sift replicate` gives consistent, incremental, commit-granular
  replication over a filesystem path or HTTP from another sift instance (and
  `/snapshot` a point-in-time tarball), but there is no per-operation log shipping
  for sub-second replica lag and no distributed sharding or cross-node query
  fan-out.

## License

Apache-2.0. See LICENSE.
