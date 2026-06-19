"""`sift` CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .build import build_artifact, BuildConfig


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj.get("id", obj.get("_id")), obj.get("text", obj.get("body", ""))


def _iter_beir(corpus_jsonl: Path):
    """BEIR corpus.jsonl: {_id, title, text}. Title + text concatenated."""
    with open(corpus_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            yield obj["_id"], text


def cmd_build(args):
    cfg = BuildConfig(
        model_name=args.model,
        k_expand=args.k,
        sim_threshold=args.threshold,
        stop_df_ratio=args.stop_df,
    )
    inp = Path(args.input)
    if args.format == "beir":
        corpus = _iter_beir(inp)
    else:
        corpus = _iter_jsonl(inp)
    meta = build_artifact(corpus, args.out, cfg=cfg, verbose=not args.quiet)
    print(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser(prog="sift")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="build a .sift artifact from a corpus")
    b.add_argument("--input", required=True, help="jsonl corpus file")
    b.add_argument("--out", required=True, help="output artifact directory")
    b.add_argument("--format", choices=["jsonl", "beir"], default="jsonl",
                   help="input format (default: jsonl with id+text)")
    b.add_argument("--model", default="minishlab/potion-base-8M")
    b.add_argument("-k", type=int, default=10, help="HNSW neighbors per term")
    b.add_argument("--threshold", type=float, default=0.65, help="min cosine sim for expansion")
    b.add_argument("--stop-df", type=float, default=0.4, help="stopword df/N cutoff")
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(func=cmd_build)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
