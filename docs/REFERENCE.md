# Sift reference

Use `sift <command> --help` for all flags and defaults.
See [embedding](EMBEDDING.md) for library use and storage rules.

## Build and update

These commands assume the built binary is on `PATH`:

```sh
sift build --input corpus.jsonl --out artifacts/docs.sift --model /path/to/model
sift add --index artifacts/docs.sift --input new-documents.jsonl
sift delete --index artifacts/docs.sift --id doc-42
sift compact --index artifacts/docs.sift
```

JSONL documents require `id` and `text`. Use `--format beir` for `_id`, `title`,
and `text` input. Writes and compaction need the recorded build model.

| Build option | Purpose |
|---|---|
| `--threshold`, `--k-expand` | Select semantic neighbors. |
| `--positions` | Enable phrase filtering and proximity scoring. |
| `--rank-fields price,rating` | Store numeric fields for ranking and facets. |
| `--compositional` | Enable the optional composition reranker. |
| `--no-payload` | Omit stored payloads from search results. |

Adds create immutable segments. Compaction merges them and removes deleted
content. Deleted documents affect term statistics until compaction.

## HTTP search

```sh
sift serve --artifacts ./artifacts --bind 127.0.0.1:8080
```

From another terminal:

```sh
curl -s http://127.0.0.1:8080/search \
  -H 'content-type: application/json' \
  -d '{"index":"docs","q":"laptop","k":10,"filter":[{"field":"price","lt":1500}]}'
```

The response includes `hits`, `total`, and `latency_us`. Each hit contains
`doc_id`, `score`, `snippet`, and an optional `payload`.

| Request field | Purpose |
|---|---|
| `index`, `q` | Index name and query text. |
| `k`, `offset` | Page size and offset. Keep the index unchanged between pages. |
| `blend_alpha` | Semantic weight from 0 to 1. Default 0.5, equivalent to CLI `--semantic-weight`. |
| `filter` | AND predicates: `eq`, `neq`, `in`, `lt`, `lte`, `gt`, `gte`, `exists`. |
| `facets` | Numeric field names to count. |
| `highlight` | Include HTML-escaped `snippet_html` with matched terms marked. |
| `with_payload` | Include stored documents. Default true. |
| `cache` | Use the result cache. Set false for benchmarks. |
| `rerank` | Set false to bypass a configured reranker. |

`total` can be capped by the retrieval candidate window. Phrase queries need
`--positions`. Use `-term` for explicit exclusions. The full request schema is
[SearchOptions](../crates/sift/src/query/request.rs).

Some options require one segment, including facets, ranking tiers, MMR, PRF,
deduplication, composition, and contextual reranking. Run `sift compact` when
an option reports this restriction.

## HTTP writes

```sh
curl -s http://127.0.0.1:8080/add -H 'content-type: application/json' \
  -d '{"index":"docs","upsert":true,"docs":[{"id":"1","text":"Updated document"}]}'
curl -s http://127.0.0.1:8080/delete -H 'content-type: application/json' \
  -d '{"index":"docs","ids":["1"]}'
curl -s http://127.0.0.1:8080/compact -H 'content-type: application/json' \
  -d '{"index":"docs"}'
```

`--read-only` disables write and admin routes. `--api-keys FILE` enables bearer
authentication. See `sift serve --help` for server settings.

Other routes: `POST /explain`, `/suggest`, `/reload`, `/snapshot`, `/alias`, and
`GET /datasets`, `/stats`, `/metrics`, `/healthz`, `/readyz`, `/version`.

## Replication

```sh
sift replicate --from /source/docs.sift --to /replica/docs.sift
```

Use `--watch 5` to repeat. HTTP replication requires `serve --enable-replication`
and a source URL such as `https://host/replicate/docs`. These endpoints expose
index contents, including documents. Protect them with authentication.
Replication copies committed index state.

## Optional reranking

| Mode | Setup |
|---|---|
| Composition | Build with `--compositional`, then set `composition_weight` from 0 to 1. |
| Tree model | Start with `serve --reranker model.json`. |
| Cross-encoder | Build with `--features cross-encoder`, then use `serve --cross-encoder MODEL_DIR`. |

Rerankers only reorder retrieved candidates. Cross-encoders add query-time model
inference. Positive `contextual_weight` requires a basic single-segment query.
