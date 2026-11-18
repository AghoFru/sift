"""Run the Rust in-process Sift bench on each Sift index built by the suite."""

import json
import subprocess
import sys
from pathlib import Path

from ir_bench.catalog import prepare

matrix, suite_dir, work, binary, out_dir = map(Path, sys.argv[1:6])
out_dir.mkdir(parents=True, exist_ok=True)
config = json.loads(matrix.read_text())
for entry in config["datasets"]:
    report = json.loads((suite_dir / entry["name"] / "sift.json").read_text())
    index = work / "cache" / report["build"]["key"] / "artifact" / "docs.sift"
    dataset = prepare(entry["source"], work)
    print(entry["name"], end=" ", flush=True)
    subprocess.run(
        [str(binary), str(index), str(dataset), str(out_dir / f"{entry['name']}.json")],
        check=True,
        timeout=3600,
    )
