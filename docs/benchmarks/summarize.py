"""Collect suite reports and Rust bench outputs into small CSV and JSON files."""

import csv
import json
import statistics
import sys
from pathlib import Path

from ir_bench.metrics import remove_self_matches

suite_dir, bench_dir, out_dir = map(Path, sys.argv[1:4])
out_dir.mkdir(parents=True, exist_ok=True)
suite = json.loads((suite_dir / "suite.json").read_text())
rows = []
for row in suite["results"]:
    if row["status"] != "success":
        raise SystemExit(f"failed row: {row}")
    report = json.loads((suite_dir / row["report"]).read_text())
    m, lat = report["metrics"], report["latency"]
    build = report["build"]
    rows.append(
        {
            "dataset": row["dataset"],
            "system": row["engine"],
            "documents": report["dataset"]["documents"],
            "queries": len(report["rankings"]),
            "nDCG@10": m["nDCG@10"],
            "R@100": m["R@100"],
            "RR@10": m["RR@10"],
            "AP@100": m["AP@100"],
            "p50_us": lat["p50_us"],
            "p95_us": lat["p95_us"],
            "latency_boundary": lat["boundary"],
            "build_seconds": build["build_seconds"],
        }
    )
    if row["engine"] == "sift":
        rust = json.loads((bench_dir / f"{row['dataset']}.json").read_text())
        rust_rankings = remove_self_matches(rust["rankings"])
        same = sum(
            rust_rankings.get(q) == hits for q, hits in report["rankings"].items()
        )
        rows.append(
            {
                "dataset": row["dataset"],
                "system": "sift-rust",
                "documents": report["dataset"]["documents"],
                "queries": rust["samples"] // rust["measured_passes"],
                "nDCG@10": "",
                "R@100": "",
                "RR@10": "",
                "AP@100": "",
                "p50_us": rust["p50_us"],
                "p95_us": rust["p95_us"],
                "latency_boundary": rust["boundary"]
                + f"; rankings identical to HTTP for {same}/{len(report['rankings'])} queries",
                "build_seconds": "",
            }
        )
with (out_dir / "results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
groups = {
    "used_in_tuning": ["scifact", "nfcorpus", "fiqa"],
    "not_used_in_tuning": ["arguana", "scidocs", "cqadupstack-android", "cqadupstack-webmasters"],
    "new_held_out": ["scidocs", "cqadupstack-android", "cqadupstack-webmasters"],
}
means = {}
for group, names in groups.items():
    means[group] = {}
    for system in sorted({r["system"] for r in rows}):
        sel = [r for r in rows if r["system"] == system and r["dataset"] in names]
        entry = {"datasets": names}
        for key in ("nDCG@10", "R@100"):
            if sel and sel[0][key] != "":
                entry[key] = statistics.mean(r[key] for r in sel)
        entry["median_of_p50_us"] = statistics.median(r["p50_us"] for r in sel)
        means[group][system] = entry
(out_dir / "means.json").write_text(json.dumps(means, indent=1) + "\n")
for r in []:
    print(r["dataset"], r["system"], r["nDCG@10"] and round(r["nDCG@10"], 3),
          r["R@100"] and round(r["R@100"], 3), round(r["p50_us"], 1), round(r["p95_us"], 1),
          r["latency_boundary"][-60:])
for g, systems in means.items():
    for name, e in systems.items():
        print(g, name, *(f"{k}={round(v, 3)}" for k, v in e.items() if k != "datasets"))
