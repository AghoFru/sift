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

## Sift 1 and Sift 2 BEIR comparison

The comparison used the local SciFact, NFCorpus, and FiQA test sets. Sift 1
means the checked-in revision before the Sift 2 changes. Sift 2 default uses
the tuned sparse path with `blend_alpha=0.5`, `bigram_weight=0.4`, and query
expansion and composition disabled. Sift 2 contextual adds the bounded int8
query-document rerank over the sparse candidate window.

Command shape:

```bash
python tests/beir_eval.py --datasets scifact,nfcorpus,fiqa
python tests/beir_eval.py \
  --serve-args="--cross-encoder reranker/ce-minilm-l6" \
  --search-params '{"contextual_weight":0.9,"rerank":true}'
```

| Variant | nDCG@10 | MRR@10 | Recall@100 |
| --- | ---: | ---: | ---: |
| Exact BM25 | 0.4648 | 0.4857 | 0.5723 |
| Sift 1 | 0.4859 | 0.5078 | 0.5811 |
| Sift 2 default | 0.4879 | 0.5096 | 0.5812 |
| Sift 2 contextual | 0.5096 | 0.5334 | 0.5812 |
| Sift 2 static composition | 0.4449 | 0.4661 | 0.5812 |

The default row uses the strongest measured sparse configuration in the local
grid. The static composition row uses the best tested positive composition
weight after the candidate-window fix. It preserved recall but did not improve
ranking quality, so composition remains opt-in.

The exact BM25 row uses `blend_alpha=0`, `bigram_weight=0`, `qexp_weight=0`,
and `composition_weight=0`. The HNSW hybrid alternative was not benchmarked in
this repository, so this table does not claim a measured HNSW comparison.

Contextual deltas against Sift 1 were `+0.0237` nDCG@10, `+0.0256` MRR@10,
and `+0.0001` Recall@100. The gains came from NFCorpus and FiQA. SciFact
dropped by `0.0325` nDCG@10, so the contextual mode should be evaluated per
corpus rather than enabled blindly.

On 40-query latency samples per dataset, contextual reranking added between
`11` and `20` microseconds of reported engine time on this machine. The
candidate depth and model thread count remain configurable because this cost
depends on corpus size and hardware.

### Sift-native tree reranking

The native LightGBM LambdaMART path was trained on SciFact and FiQA candidate
labels, then loaded by the Rust server and evaluated on the same three local
sets. It uses fourteen relative Sift-native features and the candidate set from
normal Sift scoring.

| Variant | nDCG@10 | MRR@10 | Recall@100 |
| --- | ---: | ---: | ---: |
| Sift 2 native tree, 12 features | 0.4905 | 0.5118 | 0.5812 |
| Sift 2 integrated tree, 14 features | 0.4952 | 0.5177 | 0.5812 |

The earlier 12-feature tree improved nDCG@10 by `+0.0042` and MRR@10 by
`+0.0035` while preserving candidate recall. The integrated 14-feature tree
adds qexp and composition evidence to the learned path and reaches `0.4952`
nDCG@10 and `0.5177` MRR@10. The direct global weights remain `0` in the
sparse default. A production model should use labels that cover the target
corpus mixture.
