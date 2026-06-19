"""Train a LightGBM lambdarank reranker on top of sift.

Pipeline:
  1. Start a sift server pointing at artifacts-reranker/ (must include exact CSR).
  2. For each training dataset (scifact, fiqa), walk BEIR qrels.
  3. For each query, fetch top-K candidates from sift WITH features.
  4. Join hits with qrels: label = relevance if labeled else 0.
  5. Train LGBMRanker(objective="lambdarank").
  6. Evaluate NDCG@10 on:
       - Held-out queries from training datasets
       - OOD: nfcorpus (never seen during training)

Usage:
  python train_lgbm.py --sift http://127.0.0.1:8090 \
                        --train scifact fiqa \
                        --ood   nfcorpus \
                        --out   reranker.lgb.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import requests

DATA_DIR = Path(os.environ.get("SIFT_REGRESSION_DATA", "tests/data"))
FEATURE_NAMES = [
    "bm25_combined",
    "bm25_exact",
    "bm25_semantic",
    "coverage",
    "doc_len",
    "log_doc_len",
    "semantic_ratio",
    "rank_inv",
]


# ─── data loading ──────────────────────────────────────────────────────

def load_qrels(dataset: str) -> dict[str, dict[str, int]]:
    """BEIR qrels/test.tsv → {qid: {doc_id: rel}}."""
    p = DATA_DIR / dataset / "qrels" / "test.tsv"
    qrels: dict[str, dict[str, int]] = {}
    with open(p) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q, d, r = parts[0], parts[1], int(parts[2])
            qrels.setdefault(q, {})[d] = r
    return qrels


def load_queries(dataset: str) -> dict[str, str]:
    """BEIR queries.jsonl → {qid: text}."""
    p = DATA_DIR / dataset / "queries.jsonl"
    out: dict[str, str] = {}
    with open(p) as f:
        for line in f:
            obj = json.loads(line)
            out[obj["_id"]] = obj["text"]
    return out


# ─── feature extraction via sift HTTP ──────────────────────────────────

def fetch_candidates(sift_url: str, dataset: str, query: str, k: int) -> list[dict]:
    r = requests.post(
        f"{sift_url}/search",
        json={"index": dataset, "q": query, "k": k, "features": True},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("hits", [])


def derive_features(hit: dict, rank: int) -> dict[str, float]:
    f = hit.get("features") or {}
    bm25_c = float(f.get("bm25_combined", 0.0))
    bm25_e = float(f.get("bm25_exact", 0.0))
    bm25_s = float(f.get("bm25_semantic", 0.0))
    dl = float(f.get("doc_len", 0.0))
    return {
        "bm25_combined": bm25_c,
        "bm25_exact": bm25_e,
        "bm25_semantic": bm25_s,
        "coverage": float(f.get("coverage", 0.0)),
        "doc_len": dl,
        "log_doc_len": float(np.log1p(dl)),
        "semantic_ratio": bm25_s / max(bm25_c, 1e-6),
        "rank_inv": 1.0 / (rank + 1),
    }


def build_rows(sift_url: str, dataset: str, k: int, max_q: int | None = None) -> list[dict]:
    qrels = load_qrels(dataset)
    queries = load_queries(dataset)
    qids = [q for q in qrels.keys() if q in queries]
    if max_q:
        qids = qids[:max_q]
    rows: list[dict] = []
    t0 = time.time()
    for i, qid in enumerate(qids):
        hits = fetch_candidates(sift_url, dataset, queries[qid], k)
        rel_for_q = qrels[qid]
        for rank, h in enumerate(hits):
            row = derive_features(h, rank)
            row["qid"] = f"{dataset}::{qid}"
            row["doc_id"] = h["doc_id"]
            row["label"] = rel_for_q.get(h["doc_id"], 0)
            rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {dataset}: {i+1}/{len(qids)} queries  ({time.time()-t0:.1f}s)")
    print(f"  {dataset}: {len(qids)} queries → {len(rows):,} rows in {time.time()-t0:.1f}s")
    return rows


# ─── lgb assembly ──────────────────────────────────────────────────────

def to_lgb_inputs(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int], list[str], list[str]]:
    """Sort by qid, return X / y / group_sizes / qids / doc_ids."""
    rows = sorted(rows, key=lambda r: r["qid"])
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows], dtype=np.int32)
    qids = [r["qid"] for r in rows]
    doc_ids = [r["doc_id"] for r in rows]
    groups: list[int] = []
    cur, n = qids[0], 1
    for q in qids[1:]:
        if q == cur:
            n += 1
        else:
            groups.append(n)
            cur, n = q, 1
    groups.append(n)
    return X, y, groups, qids, doc_ids


def split_train_val(rows: list[dict], val_frac: float = 0.2, seed: int = 0):
    rng = random.Random(seed)
    uniq_qids = sorted({r["qid"] for r in rows})
    rng.shuffle(uniq_qids)
    n_val = max(1, int(len(uniq_qids) * val_frac))
    val_qids = set(uniq_qids[:n_val])
    train = [r for r in rows if r["qid"] not in val_qids]
    val = [r for r in rows if r["qid"] in val_qids]
    return train, val


# ─── eval ──────────────────────────────────────────────────────────────

def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores)
    ranked = labels[order]
    gains = (2.0 ** ranked - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
    dcg = (gains[:k] * discounts[:k]).sum()
    ideal = np.sort(labels)[::-1]
    igains = (2.0 ** ideal - 1.0)
    idcg = (igains[:k] * discounts[:k]).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def per_query_ndcg(X, y, groups, scores, k=10) -> float:
    offsets = np.cumsum([0] + groups)
    vals = []
    for i in range(len(groups)):
        s, e = offsets[i], offsets[i + 1]
        if y[s:e].sum() == 0:
            continue  # no relevant docs for this query in candidate pool
        vals.append(ndcg_at_k(y[s:e], scores[s:e], k))
    return float(np.mean(vals)) if vals else 0.0


def baseline_ndcg(X, y, groups, score_col: int = 0, k: int = 10) -> float:
    """NDCG@k using a single feature column as the score (e.g. raw bm25_combined)."""
    return per_query_ndcg(X, y, groups, X[:, score_col], k)


# ─── main ──────────────────────────────────────────────────────────────

def main():
    global DATA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--sift", default="http://127.0.0.1:8090",
                    help="URL of a running sift server with features support")
    ap.add_argument("--train", nargs="+", default=["scifact", "fiqa"])
    ap.add_argument("--ood", nargs="+", default=["nfcorpus"])
    ap.add_argument("--k", type=int, default=100, help="candidates per query from sift")
    ap.add_argument("--max-q-per-ds", type=int, default=None,
                    help="cap queries per dataset (for fast iteration)")
    ap.add_argument("--data", type=Path, default=DATA_DIR,
                    help="BEIR dataset root (default: SIFT_REGRESSION_DATA or tests/data)")
    ap.add_argument("--out", default="reranker/reranker.lgb.json")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    DATA_DIR = args.data

    print("[1/4] fetching candidates + features from sift")
    train_rows: list[dict] = []
    for ds in args.train:
        print(f"  {ds}:")
        train_rows.extend(build_rows(args.sift, ds, args.k, args.max_q_per_ds))

    print("\n[2/4] splitting train / held-out (by qid)")
    train_split, val_split = split_train_val(train_rows, val_frac=0.2, seed=args.seed)
    print(f"  train rows={len(train_split):,}  held-out rows={len(val_split):,}")

    Xtr, ytr, gtr, _, _ = to_lgb_inputs(train_split)
    Xv,  yv,  gv,  _, _ = to_lgb_inputs(val_split)

    print("\n[3/4] training LGBMRanker (lambdarank)")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        n_estimators=args.rounds,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=10,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=args.seed,
        verbose=-1,
    )
    model.fit(
        Xtr, ytr,
        group=gtr,
        eval_set=[(Xv, yv)],
        eval_group=[gv],
        eval_at=[10],
        callbacks=[lgb.early_stopping(20)],
    )

    print("\n[4/4] evaluation")
    def show(label, X, y, groups):
        bm = baseline_ndcg(X, y, groups, score_col=FEATURE_NAMES.index("bm25_combined"))
        scores = model.predict(X)
        rerank = per_query_ndcg(X, y, groups, scores)
        delta = rerank - bm
        sign = "+" if delta >= 0 else ""
        print(f"  {label:>12s}  sift-only NDCG@10 = {bm:.4f}   reranked NDCG@10 = {rerank:.4f}   ({sign}{delta:.4f})")

    show("train", Xtr, ytr, gtr)
    show("held-out", Xv, yv, gv)
    for ds in args.ood:
        print(f"  {ds} (OOD): fetching...")
        rows = build_rows(args.sift, ds, args.k, args.max_q_per_ds)
        X, y, g, _, _ = to_lgb_inputs(rows)
        show(ds, X, y, g)

    # Save model as LightGBM JSON for Rust inference
    model.booster_.save_model(args.out, num_iteration=model.best_iteration_)
    print(f"\nsaved model → {args.out}")
    print(f"feature order: {FEATURE_NAMES}")
    print(f"best iteration: {model.best_iteration_}")


if __name__ == "__main__":
    main()
