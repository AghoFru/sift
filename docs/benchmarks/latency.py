"""Measure per-query search latency for four in-process systems on a quiet machine.

Each round runs every system and dataset cell once, in a new random order. A cell is a new
process that opens the index, warms up for at least 2 seconds, then runs one measured pass.
Warmup and the measured pass repeat all judged queries, each time in a new random order, until
at least the minimum time has passed (2 seconds for warmup, 1 second for the pass).
Every cell runs on one core, pinned with taskset. Before each cell, the script waits until that
core is less than 5% busy over 2 seconds. If other processes used 5% or more of the core while
the cell ran, the cell runs again. Linux only, because the gate reads /proc/stat.

  latency.py run MATRIX SUITE_DIR WORK TOOLS_DIR MODEL_DIR OUT_DIR ROUNDS --cpu=N
      [SYSTEM:DATASET ...]
  latency.py summarize OUT_DIR [OUT_DIR ...]

TOOLS_DIR holds siftbench, tantivybench, sift, and the Sift C library (FFI_LIBRARY).
--cpu=N selects the core. Choose a fast core that other busy processes are not using.
"""

import json
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# A cell starts only when its core is nearly idle, and is repeated if other work used the core.
CORE_BUSY_LIMIT = 0.05
CORE_SAMPLE_SECONDS = 2.0
POLL_SECONDS = 10
MAX_POLLS = 6 * 60 * 24
CELL_TIMEOUT_SECONDS = 1800
MAX_CELL_ATTEMPTS = 50
SYSTEMS = ("sift-rust", "tantivy", "sift-ffi", "bm25s")
DEPTH = 100
# Match WARMUP_MIN, PASS_MIN, and MAX_ITERATIONS in siftbench and tantivybench.
WARMUP_MIN_SECONDS = 2.0
PASS_MIN_SECONDS = 1.0
MAX_ITERATIONS = 100_000
FFI_LIBRARY = "libsift_ffi.so" if sys.platform == "linux" else "libsift_ffi.dylib"


def core_ticks(cpu):
    """Returns (busy, total) clock ticks of one CPU from /proc/stat."""
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields[0] == f"cpu{cpu}":
            values = [int(value) for value in fields[1:]]
            idle = values[3] + values[4]
            return sum(values) - idle, sum(values)
    raise SystemExit(f"CPU {cpu} is not in /proc/stat.")


def wait_for_idle_core(cpu):
    for _ in range(MAX_POLLS):
        busy_start, total_start = core_ticks(cpu)
        time.sleep(CORE_SAMPLE_SECONDS)
        busy_end, total_end = core_ticks(cpu)
        busy = (busy_end - busy_start) / max(total_end - total_start, 1)
        if busy < CORE_BUSY_LIMIT:
            return busy
        print(f"cpu{cpu} {busy:.0%} busy, waiting", flush=True)
        time.sleep(POLL_SECONDS)
    raise SystemExit(f"CPU {cpu} stayed busy for {MAX_POLLS} polls.")


def run_on_core(command, cpu):
    """Runs a pinned command. Returns the share of the core that other processes used."""
    tick_seconds = 1 / os.sysconf("SC_CLK_TCK")
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    busy_start, total_start = core_ticks(cpu)
    subprocess.run(["taskset", "-c", str(cpu), *command], check=True, timeout=CELL_TIMEOUT_SECONDS)
    busy_end, total_end = core_ticks(cpu)
    child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    own_seconds = (child_end.ru_utime - child_start.ru_utime) + (
        child_end.ru_stime - child_start.ru_stime
    )
    other_seconds = (busy_end - busy_start) * tick_seconds - own_seconds
    return max(other_seconds, 0.0) / ((total_end - total_start) * tick_seconds)


def percentiles(samples):
    ordered = sorted(samples)
    return statistics.median(ordered), ordered[math.ceil(len(ordered) * 0.95) - 1]


def python_cell(system, artifact, dataset, out, seed, tools, model):
    """Times the adapter call the same way IR Bench does, with perf_counter_ns."""
    sys.path.insert(0, str(Path(__file__).parent))
    from inprocess_adapters import BM25S, SiftFFI
    from ir_bench.dataset import dataset_paths, load_qrels, load_queries

    paths = dataset_paths(dataset)
    queries = load_queries(paths["queries"])
    order = sorted({row.query_id for row in load_qrels(paths["qrels"])})
    if system == "bm25s":
        engine = BM25S({})
    else:
        engine = SiftFFI(
            {
                "library": str(tools / FFI_LIBRARY),
                "binary": str(tools / "sift"),
                "model": str(model),
            }
        )
    randomizer = random.Random(seed)

    def timed_iterations(search, min_seconds):
        # Same loop as timed_iterations in siftbench and tantivybench.
        started, samples, iterations = time.monotonic(), [], 0
        while iterations < MAX_ITERATIONS and (
            iterations == 0 or time.monotonic() - started < min_seconds
        ):
            randomizer.shuffle(order)
            for query_id in order:
                begun = time.perf_counter_ns()
                search(queries[query_id], DEPTH)
                samples.append((time.perf_counter_ns() - begun) / 1000)
            iterations += 1
        return samples, iterations

    with engine.open(artifact) as search:
        _, warmup_iterations = timed_iterations(search, WARMUP_MIN_SECONDS)
        samples, iterations = timed_iterations(search, PASS_MIN_SECONDS)
        rankings = {query_id: search(queries[query_id], DEPTH) for query_id in order}
    passes = [{"samples_us": samples, "iterations": iterations}]
    report = {
        "system": system,
        "seed": seed,
        "warmup_iterations": warmup_iterations,
        "passes": passes,
        "rankings": rankings,
    }
    Path(out).write_text(json.dumps(report))


def cell_command(system, name, context, out, seed):
    tools, artifact, dataset = context["tools"], context["artifacts"], context["datasets"][name]
    if system == "sift-rust":
        index = artifact[("sift-ffi", name)] / "docs.sift"
        return [str(tools / "siftbench"), str(index), str(dataset), str(out), str(seed), "1"]
    if system == "tantivy":
        index = context["work"] / "tantivy" / name
        trec = out.with_suffix(".trec")
        return [
            str(tools / "tantivybench"),
            "search",
            str(index),
            str(dataset),
            str(out),
            str(seed),
            "1",
            str(trec),
        ]
    return [
        sys.executable,
        __file__,
        "cell",
        system,
        str(artifact[(system, name)]),
        str(dataset),
        str(out),
        str(seed),
        str(tools),
        str(context["model"]),
    ]


def reference_rankings(context, system, name):
    if system == "tantivy":
        path = context["work"] / "tantivy-quality" / name / "tantivy.json"
    else:
        engine = "sift-ffi" if system == "sift-rust" else system
        path = context["suite"] / name / f"{engine}.json"
    return json.loads(path.read_text())["rankings"]


def run_cell(system, name, context, raw_dir, seed):
    from ir_bench.metrics import remove_self_matches

    out = raw_dir / f"{system}--{name}.json"
    command = cell_command(system, name, context, out, seed)
    cpu = context["cpu"]
    for _ in range(MAX_CELL_ATTEMPTS):
        busy_before = wait_for_idle_core(cpu)
        other_during = run_on_core(command, cpu)
        if other_during < CORE_BUSY_LIMIT:
            break
        print(
            f"{system} {name}: other work used {other_during:.0%} of cpu{cpu}, repeating",
            flush=True,
        )
    else:
        raise SystemExit(f"{system} {name}: other work used the core during every attempt.")
    report = json.loads(out.read_text())
    # A TREC run cannot list a query with no results, so compare nonempty rankings only.
    rankings = {q: h for q, h in remove_self_matches(report.pop("rankings")).items() if h}
    reference = {q: h for q, h in reference_rankings(context, system, name).items() if h}
    if rankings != reference:
        raise SystemExit(f"{system} {name}: rankings differ from the quality run.")
    out.write_text(json.dumps(report))
    out.with_suffix(".trec").unlink(missing_ok=True)
    p50, p95 = percentiles(report["passes"][0]["samples_us"])
    return {
        "system": system,
        "dataset": name,
        "seed": seed,
        "samples": len(report["passes"][0]["samples_us"]),
        "p50_us": p50,
        "p95_us": p95,
        "core_busy_before": busy_before,
        "core_other_during": other_during,
    }


def load_context(matrix, suite, work, tools, model):
    from ir_bench.catalog import prepare

    config = json.loads(matrix.read_text())
    names = [entry["name"] for entry in config["datasets"]]
    datasets = {entry["name"]: prepare(entry["source"], work) for entry in config["datasets"]}
    artifacts = {}
    for name in names:
        for engine in ("sift-ffi", "bm25s"):
            report = json.loads((suite / name / f"{engine}.json").read_text())
            artifacts[(engine, name)] = work / "cache" / report["build"]["key"] / "artifact"
    return {
        "names": names,
        "datasets": datasets,
        "artifacts": artifacts,
        "work": work,
        "suite": suite,
        "tools": tools,
        "model": model,
    }


def write_atomic(path, value):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=1)
    os.replace(temporary, path)


def run(matrix, suite, work, tools, model, out_dir, rounds, only, cpu):
    context = {**load_context(matrix, suite, work, tools, model), "cpu": cpu}
    cells = [(s, n) for s in SYSTEMS for n in context["names"]]
    if only:
        cells = [cell for cell in cells if f"{cell[0]}:{cell[1]}" in only]
        if len(cells) != len(only):
            raise SystemExit("Unknown SYSTEM:DATASET cell.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for number in range(rounds):
        path = out_dir / f"round-{number}.json"
        if path.is_file():
            continue
        raw_dir = out_dir / "raw" / f"round-{number}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        order = cells.copy()
        random.Random(1000 + number).shuffle(order)
        results = [run_cell(system, name, context, raw_dir, number) for system, name in order]
        write_atomic(path, {"round": number, "cells": results})
        print(f"round {number} done", flush=True)


def summarize(out_dirs):
    # A later directory re-measures a cell and replaces the earlier session for that cell.
    sessions = {}
    for out_dir in out_dirs:
        found = {}
        for path in sorted(out_dir.glob("round-*.json")):
            for row in json.loads(path.read_text())["cells"]:
                found.setdefault((row["system"], row["dataset"]), []).append(row)
        sessions.update({cell: (out_dir.name, rows) for cell, rows in found.items()})
    summary = []
    for (system, name), (session, rows) in sorted(sessions.items()):
        p50s = [row["p50_us"] for row in rows]
        median = statistics.median(p50s)
        summary.append(
            {
                "system": system,
                "dataset": name,
                "session": session,
                "passes": len(rows),
                "samples_per_pass": rows[0]["samples"],
                "p50_us": median,
                "p95_us": statistics.median(row["p95_us"] for row in rows),
                "p50_spread": (max(p50s) - min(p50s)) / median,
                "max_core_busy_before": max(row["core_busy_before"] for row in rows),
                "max_core_other_during": max(row["core_other_during"] for row in rows),
            }
        )
    return summary


def main():
    command, args = sys.argv[1], sys.argv[2:]
    if command == "cell":
        system, artifact, dataset, out, seed, tools, model = args
        python_cell(system, Path(artifact), Path(dataset), out, int(seed), Path(tools), model)
    elif command == "run":
        matrix, suite, work, tools, model, out_dir = map(Path, args[:6])
        options = [arg for arg in args[7:] if arg.startswith("--cpu=")]
        if len(options) != 1:
            raise SystemExit("Give exactly one --cpu=N.")
        cpu = int(options[0].removeprefix("--cpu="))
        cells = {arg for arg in args[7:] if not arg.startswith("--cpu=")}
        run(matrix, suite, work, tools, model, out_dir, int(args[6]), cells, cpu)
    elif command == "summarize":
        print(json.dumps(summarize([Path(p) for p in args]), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
