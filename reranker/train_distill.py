"""Train a LightGBM lambdarank reranker distilled from a cross-encoder teacher.

Inputs: candidate features (gen_candidates.py) + teacher scores
(teacher_score.py). Labels are per-query relevance grades derived from the
teacher's ranking (no human labels touched), so the model learns to imitate
the cross-encoder ordering from sift's cheap features.

Run inside a venv with lightgbm + numpy:
  python reranker/train_distill.py --cands /tmp/rr-cands.jsonl \
      --teacher /tmp/rr-teacher.jsonl --out reranker/model.json

The output is LightGBM dump_model() JSON, served natively by
`sift serve --reranker reranker/model.json`.
"""

from __future__ import annotations

import argparse
import json
import sys

import lightgbm as lgb
import numpy as np


def teacher_grades(scores: np.ndarray) -> np.ndarray:
    """Rank-based grades from teacher scores within one query's candidates:
    top-2 -> 4, top-5 -> 3, top-10 -> 2, top-20 -> 1, rest -> 0."""
    order = np.argsort(-scores)
    grades = np.zeros(len(scores), dtype=np.int32)
    for pos, idx in enumerate(order):
        if pos < 2:
            grades[idx] = 4
        elif pos < 5:
            grades[idx] = 3
        elif pos < 10:
            grades[idx] = 2
        elif pos < 20:
            grades[idx] = 1
    return grades


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--trees", type=int, default=400)
    args = ap.parse_args()

    tmap: dict[tuple[str, str], float] = {}
    with open(args.teacher) as f:
        for line in f:
            d = json.loads(line)
            tmap[(d["qid"], d["doc_id"])] = d["t"]

    feats, labels, groups = [], [], []
    with open(args.cands) as f:
        for line in f:
            d = json.loads(line)
            rows = [(c["f"], tmap.get((d["qid"], c["doc_id"]))) for c in d["cands"]]
            rows = [(f_, t) for f_, t in rows if t is not None]
            if len(rows) < 5:
                continue
            scores = np.array([t for _, t in rows], dtype=np.float64)
            g = teacher_grades(scores)
            for (f_, _), grade in zip(rows, g):
                feats.append(f_)
                labels.append(grade)
            groups.append(len(rows))

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    g = np.asarray(groups, dtype=np.int32)
    print(f"{len(g)} queries, {len(y)} rows", file=sys.stderr)

    n_val = max(1, int(len(g) * args.val_frac))
    gv, gt = g[:n_val], g[n_val:]
    split = int(gv.sum())
    Xv, yv = X[:split], y[:split]
    Xt, yt = X[split:], y[split:]

    train = lgb.Dataset(Xt, label=yt, group=gt)
    val = lgb.Dataset(Xv, label=yv, group=gv, reference=train)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "lambdarank_truncation_level": 20,
        "verbosity": -1,
    }
    booster = lgb.train(
        params, train, num_boost_round=args.trees,
        valid_sets=[val], valid_names=["val"],
        callbacks=[lgb.early_stopping(40, verbose=True), lgb.log_evaluation(50)],
    )
    with open(args.out, "w") as f:
        json.dump(booster.dump_model(), f)
    print(f"wrote {args.out} ({booster.num_trees()} trees)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
