"""Build a Tantivy index per suite dataset, export a TREC run, and score it with IR Bench."""

import json
import subprocess
import sys
from pathlib import Path

from ir_bench.catalog import prepare
from ir_bench.run import evaluate_file, write_report

TIMEOUT_SECONDS = 3600

matrix, work, binary, out_dir = map(Path, sys.argv[1:5])
config = json.loads(matrix.read_text())
for entry in config["datasets"]:
    name = entry["name"]
    dataset = prepare(entry["source"], work)
    index = work / "tantivy" / name
    if not (index / "meta.json").is_file():
        corpus = dataset / "corpus.jsonl"
        subprocess.run(
            [str(binary), "build", str(corpus), str(index)], check=True, timeout=TIMEOUT_SECONDS
        )
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    run = target / "tantivy.trec"
    subprocess.run(
        [
            str(binary),
            "search",
            str(index),
            str(dataset),
            str(target / "tantivy-pass.json"),
            "0",
            "1",
            str(run),
        ],
        check=True,
        timeout=TIMEOUT_SECONDS,
    )
    report = evaluate_file(
        run,
        dataset / "qrels.txt",
        config["depth"],
        config["measures"],
        exclude_self_matches=config["exclude_self_matches"],
    )
    write_report(target / "tantivy.json", report)
    metrics = report["metrics"]
    print(name, round(metrics["nDCG@10"], 3), round(metrics["R@100"], 3), flush=True)
