## SciFact comparison, 2026-07-17

Command:

```bash
python3 benchmarks/compare.py artifacts/scifact.sift --runs 5 --query-repeat 30
```

Machine: Apple M1 Ultra, Darwin 25.1.0. Corpus: 5,183 documents.
Build means use five alternating-order runs. Query metrics use 600 uncached
requests per engine and report engine time separately from HTTP/JSON overhead.

```json
{
  "bm25": {
    "build_mean_seconds": 2.9633,
    "artifact_bytes": 23958650,
    "engine_mean_us": 11.0517,
    "engine_p50_us": 10.0,
    "engine_p95_us": 22.0,
    "http_mean_us": 393.9886
  },
  "sift": {
    "build_mean_seconds": 2.8132,
    "artifact_bytes": 29603640,
    "engine_mean_us": 14.12,
    "engine_p50_us": 12.0,
    "engine_p95_us": 32.0,
    "http_mean_us": 423.6197
  }
}
```
