"""Score (query, doc) candidate pairs with a cross-encoder teacher on GPU.

Run with a Python environment containing torch, transformers, and CUDA support:
  python reranker/teacher_score.py \
      --cands /tmp/rr-cands.jsonl --texts /tmp/rr-texts.jsonl \
      --out /tmp/rr-teacher.jsonl

Output rows: {"qid", "doc_id", "t": teacher_logit}
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()

    texts: dict[str, str] = {}
    with open(args.texts) as f:
        for line in f:
            d = json.loads(line)
            texts[d["doc_id"]] = d["text"]
    print(f"{len(texts)} doc texts loaded", file=sys.stderr)

    pairs: list[tuple[str, str, str, str]] = []  # qid, doc_id, query, text
    with open(args.cands) as f:
        for line in f:
            d = json.loads(line)
            for c in d["cands"]:
                t = texts.get(c["doc_id"])
                if t is not None:
                    pairs.append((d["qid"], c["doc_id"], d["query"], t))
    print(f"{len(pairs)} pairs to score", file=sys.stderr)

    out = open(args.out, "w")
    bs = args.batch
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            enc = tok(
                [p[2] for p in chunk],
                [p[3] for p in chunk],
                padding=True, truncation=True, max_length=256,
                return_tensors="pt",
            ).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.squeeze(-1).float().cpu()
            for (qid, did, _, _), s in zip(chunk, logits.tolist()):
                out.write(json.dumps({"qid": qid, "doc_id": did, "t": s}) + "\n")
            if (i // bs) % 50 == 0:
                print(f"  {i + len(chunk)}/{len(pairs)}", file=sys.stderr)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
