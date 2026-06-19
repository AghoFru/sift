"""Regression test: build fresh artifacts, eval nDCG + latency, diff against pinned baseline.

Usage:
    python tests/regression.py [--bin PATH] [--data PATH] [--update]

    --bin    path to the sift binary (default: target/release/sift relative to repo)
    --data   directory containing BEIR corpora (default: $SIFT_REGRESSION_DATA or hard-coded)
    --update overwrite tests/regression_baseline.json with the measured values.
             Use only after a deliberate, reviewed change.

Exits 0 on pass, 1 on regression, 2 on infra failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import request

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = REPO_ROOT / "target" / "release" / "sift"
DEFAULT_DATA = Path(os.environ.get(
    "SIFT_REGRESSION_DATA",
    str(REPO_ROOT / "tests" / "data"),
))
BASELINE = REPO_ROOT / "tests" / "regression_baseline.json"
PORT = 18099
K = 100
LATENCY_SAMPLES = 60


# ─── color helpers ──────────────────────────────────────────────────────

def green(s): return f"\033[32m{s}\033[0m"
def red(s):   return f"\033[31m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def dim(s):   return f"\033[2m{s}\033[0m"


# ─── BEIR data ──────────────────────────────────────────────────────────

def load_qrels(data_dir: Path, ds: str):
    q: dict[str, dict[str, int]] = {}
    with open(data_dir / ds / "qrels" / "test.tsv") as f:
        next(f)
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) >= 3:
                q.setdefault(parts[0], {})[parts[1]] = int(parts[2])
    return q


def load_queries(data_dir: Path, ds: str):
    o: dict[str, str] = {}
    with open(data_dir / ds / "queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            o[d["_id"]] = d["text"]
    return o


# ─── ndcg ──────────────────────────────────────────────────────────────

def ndcg_at_k(rels: list[int], k: int = 10) -> float:
    g = [(2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k])]
    ig = [(2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(sorted(rels, reverse=True)[:k])]
    s, idl = sum(g), sum(ig)
    return s / idl if idl > 0 else 0.0


# ─── sift server lifecycle ─────────────────────────────────────────────

def start_server(bin_path: Path, artifact_dir: Path, port: int) -> subprocess.Popen:
    log = open("/tmp/sift_regression.log", "w")
    proc = subprocess.Popen(
        [str(bin_path), "serve", "--artifacts", str(artifact_dir), "--bind", f"127.0.0.1:{port}"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    # poll until listening
    for _ in range(50):
        try:
            request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.5).read()
            return proc
        except Exception:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("server failed to start within 5s")


def search(port: int, ds: str, q: str) -> dict:
    body = json.dumps({"index": ds, "q": q, "k": K}).encode()
    req = request.Request(
        f"http://127.0.0.1:{port}/search",
        data=body,
        headers={"content-type": "application/json"},
    )
    with request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


# ─── eval ──────────────────────────────────────────────────────────────

def eval_quality(port: int, data_dir: Path, ds: str) -> float:
    qrels = load_qrels(data_dir, ds)
    queries = load_queries(data_dir, ds)
    scores = []
    for qid, rel_map in qrels.items():
        if qid not in queries or not rel_map:
            continue
        try:
            r = search(port, ds, queries[qid])
        except Exception:
            continue
        rels = [rel_map.get(h["doc_id"], 0) for h in r.get("hits", [])]
        scores.append(ndcg_at_k(rels))
    return statistics.mean(scores) if scores else 0.0


def eval_latency(port: int, data_dir: Path, ds: str, n: int = LATENCY_SAMPLES) -> tuple[float, float]:
    queries = list(load_queries(data_dir, ds).values())[:n]
    # warmup
    for q in queries[:5]:
        try:
            search(port, ds, q)
        except Exception:
            pass
    samples = []
    for q in queries:
        try:
            r = search(port, ds, q)
            samples.append(r["latency_us"])
        except Exception:
            continue
    if not samples:
        return float("inf"), float("inf")
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)]
    return float(p50), float(p95)


# ─── main ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--update", action="store_true",
                    help="overwrite the baseline JSON with the measured values")
    ap.add_argument("--keep-artifacts", action="store_true",
                    help="don't delete the tmp build dir at the end (for debugging)")
    args = ap.parse_args()

    if not args.bin.exists():
        print(red(f"sift binary not found at {args.bin}; run `cargo build --release` first."))
        return 2

    with open(BASELINE) as f:
        baseline = json.load(f)
    tol = baseline["_tolerance"]
    datasets = baseline["datasets"]

    # Build fresh artifacts to a tmp dir so we don't disturb production indices.
    with tempfile.TemporaryDirectory(prefix="sift_regress_", dir="/tmp") as tmpdir:
        tmp = Path(tmpdir)
        if args.keep_artifacts:
            tmp = Path("/tmp/sift_regression_artifacts")
            tmp.mkdir(exist_ok=True)

        print(dim(f"building fresh artifacts into {tmp}"))
        build_times: dict[str, float] = {}
        artifact_sizes: dict[str, int] = {}
        for ds in datasets:
            corpus = args.data / ds / "corpus.jsonl"
            if not corpus.exists():
                print(red(f"  missing corpus for {ds} at {corpus}"))
                return 2
            t0 = time.time()
            built = subprocess.run(
                [str(args.bin), "build",
                 "--input", str(corpus),
                 "--out", str(tmp / f"{ds}.sift"),
                 "--format", "beir"],
                capture_output=True, text=True,
            )
            if built.returncode != 0:
                print(red(f"  build failed for {ds}"))
                if built.stdout.strip():
                    print(built.stdout.rstrip())
                if built.stderr.strip():
                    print(built.stderr.rstrip(), file=sys.stderr)
                return 2
            build_times[ds] = time.time() - t0
            size = sum(p.stat().st_size for p in (tmp / f"{ds}.sift").rglob("*") if p.is_file())
            artifact_sizes[ds] = size
            print(dim(f"  built {ds} in {build_times[ds]:.1f}s ({size/1e6:.1f} MB on disk)"))

        proc = start_server(args.bin, tmp, PORT)
        try:
            measured = {}
            for ds in datasets:
                ndcg = eval_quality(PORT, args.data, ds)
                p50, p95 = eval_latency(PORT, args.data, ds)
                p50_int = int(p50) if p50 != float("inf") else -1
                p95_int = int(p95) if p95 != float("inf") else -1
                measured[ds] = {
                    "ndcg_at_10": round(ndcg, 4),
                    "latency_p50_us": p50_int,
                    "latency_p95_us": p95_int,
                    "build_seconds": round(build_times[ds], 2),
                    "artifact_bytes": artifact_sizes[ds],
                }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()

    if args.update:
        baseline["datasets"] = measured
        with open(BASELINE, "w") as f:
            json.dump(baseline, f, indent=2)
            f.write("\n")
        print(green(f"\nbaseline updated with measured values"))
        for ds, vals in measured.items():
            print(f"  {ds:>10s}  ndcg={vals['ndcg_at_10']:.4f}  p50={vals['latency_p50_us']}us  p95={vals['latency_p95_us']}us")
        return 0

    # Compare measured vs baseline.
    fail_lines: list[str] = []
    for ds, ref in datasets.items():
        m = measured.get(ds, {})
        ref_ndcg = ref["ndcg_at_10"]
        m_ndcg = m.get("ndcg_at_10", 0.0)
        ref_p50 = ref["latency_p50_us"]
        m_p50 = m.get("latency_p50_us", float("inf"))
        ref_p95 = ref["latency_p95_us"]
        m_p95 = m.get("latency_p95_us", float("inf"))
        ref_build = ref.get("build_seconds")
        m_build = m.get("build_seconds")
        ref_size = ref.get("artifact_bytes")
        m_size = m.get("artifact_bytes")

        ndcg_ok = (m_ndcg + tol["ndcg_at_10"]) >= ref_ndcg
        p50_ok = m_p50 <= ref_p50 * tol["latency_p50_us_factor"]
        p95_ok = m_p95 <= ref_p95 * tol["latency_p95_us_factor"]
        build_ok = ref_build is None or m_build <= ref_build * tol.get("build_seconds_factor", 1.5)
        size_ok = ref_size is None or m_size <= ref_size * tol.get("artifact_bytes_factor", 1.25)
        ok = ndcg_ok and p50_ok and p95_ok and build_ok and size_ok

        verdict = green("PASS") if ok else red("FAIL")
        print(
            f"  {ds:<10s}  ndcg={m_ndcg:.4f} (ref {ref_ndcg:.4f})  "
            f"p50={m_p50}us  p95={m_p95}us  "
            f"build={m_build}s  size={m_size/1e6:.1f}MB  {verdict}"
        )
        if not ok:
            details = []
            if not ndcg_ok:
                details.append(f"ndcg {m_ndcg:.4f} < {ref_ndcg:.4f} - {tol['ndcg_at_10']}")
            if not p50_ok:
                details.append(f"p50 {m_p50}us > {ref_p50}us * {tol['latency_p50_us_factor']}")
            if not p95_ok:
                details.append(f"p95 {m_p95}us > {ref_p95}us * {tol['latency_p95_us_factor']}")
            if not build_ok:
                details.append(f"build {m_build}s > {ref_build}s * {tol.get('build_seconds_factor', 1.5)}")
            if not size_ok:
                details.append(f"size {m_size} > {ref_size} * {tol.get('artifact_bytes_factor', 1.25)}")
            fail_lines.append(f"{ds}: " + "; ".join(details))

    if fail_lines:
        print()
        for line in fail_lines:
            print(red("  ✗ " + line))
        return 1
    print(green("\nall datasets within tolerance"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
