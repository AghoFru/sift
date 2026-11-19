"""Collect suite reports, Tantivy reports, and latency rounds into small CSV and JSON files.

summarize.py SUITE_DIR TANTIVY_DIR OUT_DIR LATENCY_DIR [LATENCY_DIR ...]
"""

import csv
import json
import statistics
import sys
from pathlib import Path

from latency import summarize as latency_summary

QUALITY_SYSTEMS = {"sift": "sift", "bm25": "terrier-bm25", "bm25s": "bm25s"}
GROUPS = {
    "used_in_tuning": ["scifact", "nfcorpus", "fiqa"],
    "not_used_in_tuning": ["arguana", "scidocs", "cqadupstack-android", "cqadupstack-webmasters"],
}
MEASURES = ("nDCG@10", "R@100", "RR@10", "AP@100")


def quality_rows(suite_dir, tantivy_dir):
    suite = json.loads((suite_dir / "suite.json").read_text())
    rows = []
    for row in suite["results"]:
        if row["status"] != "success":
            raise SystemExit(f"failed row: {row}")
        if row["engine"] not in QUALITY_SYSTEMS:
            continue
        report = json.loads((suite_dir / row["report"]).read_text())
        rows.append((row["dataset"], QUALITY_SYSTEMS[row["engine"]], report))
        if row["engine"] == "sift":
            tantivy = json.loads((tantivy_dir / row["dataset"] / "tantivy.json").read_text())
            rows.append((row["dataset"], "tantivy", {**tantivy, "dataset": report["dataset"]}))
    return [
        {
            "dataset": dataset,
            "system": system,
            "documents": report["dataset"]["documents"],
            "queries": report["query_count"],
            **{measure: report["metrics"][measure] for measure in MEASURES},
        }
        for dataset, system, report in rows
    ]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    suite_dir, tantivy_dir, out_dir = map(Path, sys.argv[1:4])
    latency_dirs = [Path(path) for path in sys.argv[4:]]
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = quality_rows(suite_dir, tantivy_dir)
    write_csv(out_dir / "quality.csv", quality)
    latency = latency_summary(latency_dirs)
    write_csv(out_dir / "latency.csv", latency)
    means = {}
    for group, names in GROUPS.items():
        means[group] = {}
        for system in sorted({row["system"] for row in quality}):
            selected = [r for r in quality if r["system"] == system and r["dataset"] in names]
            means[group][system] = {
                "datasets": names,
                **{key: statistics.mean(r[key] for r in selected) for key in ("nDCG@10", "R@100")},
            }
    (out_dir / "means.json").write_text(json.dumps(means, indent=1) + "\n")
    worst = max(latency, key=lambda row: row["p50_spread"])
    print(json.dumps(means, indent=1))
    print("largest pass-to-pass spread:", worst["system"], worst["dataset"], worst["p50_spread"])


if __name__ == "__main__":
    main()
