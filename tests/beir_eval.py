"""BEIR quality harness: build (or reuse) artifacts, eval nDCG@10 / MRR@10 /
Recall@100 across a configurable dataset suite.

This is the tuning workbench; `regression.py` stays the fast 3-set CI gate.

Usage:
    python tests/beir_eval.py --bin /tmp/sift-target/release/sift
    python tests/beir_eval.py --datasets scifact,nfcorpus --search-params '{"blend_alpha":0.4}'
    python tests/beir_eval.py --build-args "--threshold 0.5 --k-expand 10" --tag t05k10

Artifacts are cached under --work-dir keyed on (dataset, build args), so
sweeping query-time parameters never rebuilds. Results are appended as JSON
lines to --log so sweeps are auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = REPO_ROOT / "target" / "release" / "sift"
DEFAULT_DATA = Path(os.environ.get(
    "SIFT_REGRESSION_DATA",
    str(REPO_ROOT / "tests" / "data"),
))
DEFAULT_WORK = Path("/tmp/sift-beir-eval")
DEFAULT_SETS = "scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid,webis-touche2020,quora"
PORT = 18098
K_EVAL = 100  # retrieve depth; nDCG/MRR cut at 10, recall at 100


def load_qrels(data_dir: Path, ds: str) -> dict[str, dict[str, int]]:
    q: dict[str, dict[str, int]] = {}
    with open(data_dir / ds / "qrels" / "test.tsv") as f:
        next(f)
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) >= 3:
                q.setdefault(parts[0], {})[parts[1]] = int(parts[2])
    return q


def load_queries(data_dir: Path, ds: str) -> dict[str, str]:
    o: dict[str, str] = {}
    with open(data_dir / ds / "queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            o[d["_id"]] = d["text"]
    return o


def ndcg_at_k(rels: list[int], k: int = 10) -> float:
    g = [(2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k])]
    ig = [(2 ** r - 1) / math.log2(i + 2)
          for i, r in enumerate(sorted(rels, reverse=True)[:k])]
    s, idl = sum(g), sum(ig)
    return s / idl if idl > 0 else 0.0


def mrr_at_k(rels: list[int], k: int = 10) -> float:
    for i, r in enumerate(rels[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(rels: list[int], n_rel: int, k: int = 100) -> float:
    if n_rel == 0:
        return 0.0
    return sum(1 for r in rels[:k] if r > 0) / n_rel


def build_artifact(bin_path: Path, data_dir: Path, ds: str, out: Path,
                   build_args: list[str]) -> None:
    corpus = data_dir / ds / "corpus.jsonl"
    if not corpus.exists():
        raise FileNotFoundError(f"missing corpus for {ds} at {corpus}")
    cmd = [str(bin_path), "build", "--input", str(corpus),
           "--out", str(out), "--format", "beir"] + build_args
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def ensure_artifact(bin_path: Path, data_dir: Path, ds: str, work: Path,
                    build_args: list[str], force: bool) -> Path:
    key = hashlib.sha256(" ".join(build_args).encode()).hexdigest()[:12]
    art_dir = work / f"build-{key}"
    art_dir.mkdir(parents=True, exist_ok=True)
    out = art_dir / f"{ds}.sift"
    stamp = out / ".build-stamp"
    want = f"{ds} {' '.join(build_args)}"
    if not force and stamp.exists() and stamp.read_text() == want:
        return art_dir
    t0 = time.time()
    build_artifact(bin_path, data_dir, ds, out, build_args)
    stamp.write_text(want)
    print(f"  built {ds} in {time.time() - t0:.1f}s", file=sys.stderr)
    return art_dir


def start_server(bin_path: Path, artifact_dir: Path, port: int,
                 serve_args: list[str] | None = None) -> subprocess.Popen:
    log = open("/tmp/sift_beir_eval_server.log", "w")
    proc = subprocess.Popen(
        [str(bin_path), "serve", "--artifacts", str(artifact_dir),
         "--bind", f"127.0.0.1:{port}"] + (serve_args or []),
        stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(100):
        try:
            request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.5).read()
            return proc
        except Exception:
            time.sleep(0.15)
    proc.terminate()
    raise RuntimeError("server failed to start within 15s")


def search(port: int, ds: str, q: str, params: dict) -> dict:
    body = {"index": ds, "q": q, "k": K_EVAL, "cache": False,
            "with_payload": False}
    body.update(params)
    req = request.Request(
        f"http://127.0.0.1:{port}/search",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def eval_dataset(port: int, data_dir: Path, ds: str, params: dict,
                 jobs: int, max_queries: int | None) -> dict:
    qrels = load_qrels(data_dir, ds)
    queries = load_queries(data_dir, ds)
    items = [(qid, rel) for qid, rel in qrels.items() if qid in queries and rel]
    if max_queries:
        items = items[:max_queries]

    def one(item):
        qid, rel_map = item
        try:
            r = search(port, ds, queries[qid], params)
        except Exception:
            return None
        hits = r.get("hits", [])
        rels = [rel_map.get(h["doc_id"], 0) for h in hits]
        n_rel = sum(1 for v in rel_map.values() if v > 0)
        return (ndcg_at_k(rels), mrr_at_k(rels), recall_at_k(rels, n_rel))

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        rows = [r for r in ex.map(one, items) if r is not None]
    if not rows:
        return {"ndcg_at_10": 0.0, "mrr_at_10": 0.0, "recall_at_100": 0.0, "n_queries": 0}
    return {
        "ndcg_at_10": round(statistics.mean(r[0] for r in rows), 4),
        "mrr_at_10": round(statistics.mean(r[1] for r in rows), 4),
        "recall_at_100": round(statistics.mean(r[2] for r in rows), 4),
        "n_queries": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--datasets", default=DEFAULT_SETS)
    ap.add_argument("--build-args", default="",
                    help="extra `sift build` flags, e.g. '--threshold 0.5'")
    ap.add_argument("--search-params", default="{}",
                    help="JSON merged into each /search body")
    ap.add_argument("--serve-args", default="",
                    help="extra `sift serve` flags, e.g. '--reranker model.json'")
    ap.add_argument("--tag", default="", help="label written to the log line")
    ap.add_argument("--log", type=Path, default=DEFAULT_WORK / "results.jsonl")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--max-queries", type=int, default=0,
                    help="cap queries per dataset (0 = all)")
    ap.add_argument("--force-build", action="store_true")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    build_args = shlex.split(args.build_args)
    params = json.loads(args.search_params)
    max_q = args.max_queries or None

    if not args.bin.exists():
        print(f"sift binary not found at {args.bin}", file=sys.stderr)
        return 2

    args.work_dir.mkdir(parents=True, exist_ok=True)
    per_ds: dict[str, dict] = {}
    for ds in datasets:
        art_dir = ensure_artifact(args.bin, args.data, ds, args.work_dir,
                                  build_args, args.force_build)
        proc = start_server(args.bin, art_dir, PORT, shlex.split(args.serve_args))
        try:
            per_ds[ds] = eval_dataset(PORT, args.data, ds, params, args.jobs, max_q)
        finally:
            proc.terminate()
            proc.wait()
        m = per_ds[ds]
        print(f"  {ds:<18} ndcg={m['ndcg_at_10']:.4f}  mrr={m['mrr_at_10']:.4f}  "
              f"r@100={m['recall_at_100']:.4f}  ({m['n_queries']} q)")

    mean = {
        k: round(statistics.mean(per_ds[d][k] for d in datasets), 4)
        for k in ("ndcg_at_10", "mrr_at_10", "recall_at_100")
    }
    print(f"  {'MEAN':<18} ndcg={mean['ndcg_at_10']:.4f}  mrr={mean['mrr_at_10']:.4f}  "
          f"r@100={mean['recall_at_100']:.4f}")

    record = {
        "tag": args.tag,
        "build_args": args.build_args,
        "search_params": params,
        "datasets": per_ds,
        "mean": mean,
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a") as f:
        f.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
