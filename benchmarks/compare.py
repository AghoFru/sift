"""Reproduce a same-machine BM25-versus-sift build and query comparison.

The input is an existing payload-bearing `.sift` artifact. The script exports
its original JSONL, builds an exact-only artifact and a semantic artifact with
the same binary and tokenizer, then measures both through the HTTP API.

Example:
    python3 benchmarks/compare.py artifacts/scifact.sift --runs 3
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from urllib import request


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / "target" / "release" / "sift"
DEFAULT_QUERIES = ROOT / "benchmarks" / "scifact_queries.txt"


def export_payload(artifact: Path, output: Path) -> int:
    payload = (artifact / "payload_text.bin").read_bytes()
    raw_offsets = (artifact / "payload_off.bin").read_bytes()
    if len(raw_offsets) % 8:
        raise ValueError("payload_off.bin is not an array of u64 offsets")
    offsets = struct.unpack(f"<{len(raw_offsets) // 8}Q", raw_offsets)
    if len(offsets) < 2 or offsets[-1] != len(payload):
        raise ValueError("payload offsets do not match payload bytes")
    with output.open("wb") as stream:
        for lo, hi in zip(offsets, offsets[1:]):
            row = payload[lo:hi]
            json.loads(row)
            stream.write(row)
            stream.write(b"\n")
    return len(offsets) - 1


def build_once(binary: Path, corpus: Path, output: Path, semantic: bool) -> float:
    command = [
        str(binary),
        "build",
        "--input",
        str(corpus),
        "--out",
        str(output),
        "--format",
        "beir",
    ]
    if not semantic:
        command.extend(["--k-expand", "0"])
    started = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return time.perf_counter() - started


def artifact_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def wait_for_server(port: int) -> None:
    for _ in range(100):
        try:
            request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.2).read()
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("sift server did not start")


def search(port: int, index: str, query: str) -> tuple[int, float]:
    body = json.dumps(
        {
            "index": index,
            "q": query,
            "k": 10,
            "cache": False,
            "with_payload": False,
        }
    ).encode()
    req = request.Request(
        f"http://127.0.0.1:{port}/search",
        data=body,
        headers={"content-type": "application/json"},
    )
    started = time.perf_counter_ns()
    with request.urlopen(req, timeout=10) as response:
        result = json.loads(response.read())
    end_to_end_us = (time.perf_counter_ns() - started) / 1_000
    return int(result["latency_us"]), end_to_end_us


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def query_benchmark(
    binary: Path,
    artifacts: Path,
    queries: list[str],
    repeat: int,
    port: int,
) -> dict[str, dict[str, float]]:
    log = tempfile.TemporaryFile(mode="w+")
    server = subprocess.Popen(
        [str(binary), "serve", "--artifacts", str(artifacts), "--bind", f"127.0.0.1:{port}"],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        try:
            wait_for_server(port)
        except Exception:
            log.seek(0)
            raise RuntimeError(f"sift server did not start:\n{log.read()}")
        results: dict[str, dict[str, float]] = {}
        for index in ("bm25", "sift"):
            for query in queries[:5]:
                search(port, index, query)
            engine: list[float] = []
            end_to_end: list[float] = []
            for _ in range(repeat):
                for query in queries:
                    inside, total = search(port, index, query)
                    engine.append(float(inside))
                    end_to_end.append(total)
            results[index] = {
                "engine_mean_us": statistics.mean(engine),
                "engine_p50_us": statistics.median(engine),
                "engine_p95_us": percentile(engine, 0.95),
                "http_mean_us": statistics.mean(end_to_end),
            }
        return results
    finally:
        server.terminate()
        server.wait(timeout=5)
        log.close()


def machine() -> str:
    chip = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            chip = result.stdout.strip()
    return f"{platform.system()} {platform.release()}, {chip}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--query-repeat", type=int, default=20)
    parser.add_argument("--port", type=int, default=18101)
    args = parser.parse_args()

    if args.runs < 1 or args.query_repeat < 1:
        parser.error("runs and query-repeat must be positive")
    queries = [line.strip() for line in args.queries.read_text().splitlines() if line.strip()]
    if not queries:
        parser.error("query file is empty")

    subprocess.run(["cargo", "build", "--release", "--locked"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="sift-compare-") as raw_tmp:
        tmp = Path(raw_tmp)
        corpus = tmp / "corpus.jsonl"
        docs = export_payload(args.artifact, corpus)
        builds: dict[str, list[float]] = {"bm25": [], "sift": []}
        # Alternate order so filesystem/model cache warming does not always
        # favor the same engine.
        for run in range(args.runs):
            order = (("bm25", False), ("sift", True))
            if run % 2:
                order = tuple(reversed(order))
            for name, semantic in order:
                output = tmp / f"{name}-{run}.sift"
                builds[name].append(build_once(args.bin, corpus, output, semantic))
                previous = tmp / f"{name}-{run - 1}.sift"
                if previous.exists():
                    shutil.rmtree(previous)
        for name in builds:
            (tmp / f"{name}-{args.runs - 1}.sift").rename(tmp / f"{name}.sift")

        queries_result = query_benchmark(
            args.bin, tmp, queries, args.query_repeat, args.port
        )
        report = {
            "machine": machine(),
            "documents": docs,
            "queries": len(queries) * args.query_repeat,
            "build_runs": args.runs,
            "bm25": {
                "build_mean_seconds": statistics.mean(builds["bm25"]),
                "artifact_bytes": artifact_bytes(tmp / "bm25.sift"),
                **queries_result["bm25"],
            },
            "sift": {
                "build_mean_seconds": statistics.mean(builds["sift"]),
                "artifact_bytes": artifact_bytes(tmp / "sift.sift"),
                **queries_result["sift"],
            },
        }
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
