"""Generate reranker training candidates from sift (stdlib only).

For N msmarco *train* queries (labels untouched; we only need query text),
retrieve top-K with features from a running sift server, then join the
candidate doc texts from corpus.jsonl in one streaming pass.

Output: JSONL rows {"qid", "query", "cands": [{"doc_id", "f": [8 floats]}]}
plus a doc-text sidecar {"doc_id", "text"} for the teacher.

Usage:
  python3 reranker/gen_candidates.py --sift http://127.0.0.1:18097 \
      --data /path/to/beir/msmarco \
      --n-queries 4000 --k 50 --out /tmp/rr-cands.jsonl --texts /tmp/rr-texts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request


def search(url: str, index: str, q: str, k: int) -> dict:
    body = json.dumps({
        "index": index, "q": q, "k": k, "features": True,
        "cache": False, "with_payload": False,
    }).encode()
    req = request.Request(f"{url}/search", data=body,
                          headers={"content-type": "application/json"})
    with request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sift", default="http://127.0.0.1:18097")
    ap.add_argument("--index", default="msmarco")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-queries", type=int, default=4000)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--texts", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    # Train-split query ids only.
    train_qids: set[str] = set()
    with open(args.data / "qrels" / "train.tsv") as f:
        next(f)
        for line in f:
            parts = line.split("\t")
            if parts:
                train_qids.add(parts[0])

    queries: dict[str, str] = {}
    with open(args.data / "queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["_id"] in train_qids:
                queries[d["_id"]] = d["text"]

    rng = random.Random(args.seed)
    qids = rng.sample(sorted(queries), min(args.n_queries, len(queries)))
    print(f"{len(train_qids)} train qids, sampling {len(qids)}", file=sys.stderr)

    def fetch(qid: str):
        try:
            r = search(args.sift, args.index, queries[qid], args.k)
        except Exception as e:
            print(f"  query {qid} failed: {e}", file=sys.stderr)
            return None
        cands = []
        for rank, h in enumerate(r.get("hits", [])):
            ft = h.get("features")
            if not ft:
                return None  # artifact lacks exact CSR; abort signal
            cands.append({
                "doc_id": h["doc_id"],
                "f": [
                    ft["bm25_combined"], ft["bm25_exact"], ft["bm25_semantic"],
                    ft["coverage"], ft["doc_len"],
                    __import__("math").log(1.0 + ft["doc_len"]),
                    ft["bm25_semantic"] / max(ft["bm25_combined"], 1e-6),
                    1.0 / (rank + 1.0),
                ],
            })
        return {"qid": qid, "query": queries[qid], "cands": cands}

    needed_docs: set[str] = set()
    n_rows = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex, open(args.out, "w") as out:
        for row in ex.map(fetch, qids):
            if row is None or not row["cands"]:
                continue
            out.write(json.dumps(row) + "\n")
            n_rows += 1
            for c in row["cands"]:
                needed_docs.add(c["doc_id"])
            if n_rows % 500 == 0:
                print(f"  {n_rows} queries done", file=sys.stderr)
    print(f"{n_rows} queries, {len(needed_docs)} unique docs", file=sys.stderr)

    # One streaming pass over the corpus for the needed texts.
    n_txt = 0
    with open(args.data / "corpus.jsonl") as f, open(args.texts, "w") as out:
        for line in f:
            d = json.loads(line)
            did = d.get("_id") or d.get("id")
            if did in needed_docs:
                text = ((d.get("title") or "") + " " + (d.get("text") or "")).strip()
                out.write(json.dumps({"doc_id": did, "text": text[:2000]}) + "\n")
                n_txt += 1
    print(f"{n_txt} doc texts written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
