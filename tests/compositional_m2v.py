#!/usr/bin/env python3
"""Test Model2Vec pooling and the Sift composition retrieval path.

The experiment uses a small corpus with synonym, order, and negation cases.
It compares current Sift retrieval with pooled Model2Vec expansion and the
production composition sidecar.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from model2vec import StaticModel


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target" / "release" / "sift"
MODEL_NAME = os.environ.get(
    "SIFT_TEST_MODEL", "minishlab/potion-multilingual-128M"
)

CORPUS = [
    ("feline_consumes_rodent", "A feline consumes a rodent."),
    ("canine_consumes_rodent", "A canine consumes a rodent."),
    ("rodent_consumes_feline", "A rodent consumes a feline."),
    ("feline_watches_bird", "A feline watches a bird."),
    ("cat_eats_mouse", "A cat eats a mouse."),
    ("mouse_eats_cat", "A mouse eats a cat."),
    ("cat_is_animal", "A cat is an animal."),
    ("dog_is_animal", "A dog is an animal."),
    ("dog_not_cat", "A dog is not a cat."),
]


def write_corpus(path: Path) -> None:
    rows = [{"id": doc_id, "text": text} for doc_id, text in CORPUS]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(artifact_root: Path) -> tuple[subprocess.Popen[str], str]:
    port = free_port()
    proc = subprocess.Popen(
        [
            str(BINARY),
            "serve",
            "--artifacts",
            str(artifact_root),
            "--bind",
            f"127.0.0.1:{port}",
            "--read-only",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        if proc.poll() is not None:
            raise RuntimeError("Sift server exited before becoming ready")
        try:
            with urlopen(f"{base}/readyz", timeout=0.25) as response:
                if response.status == 200:
                    return proc, base
        except (OSError, URLError):
            pass
        time.sleep(0.05)
    proc.terminate()
    proc.wait(timeout=5)
    raise RuntimeError("Sift server did not become ready")


def search(
    base: str,
    query: str,
    *,
    qexp_weight: float,
    blend_alpha: float,
    composition_weight: float = 0.0,
) -> list[str]:
    body = json.dumps(
        {
            "index": "compositional",
            "q": query,
            "k": len(CORPUS),
            "cache": False,
            "rerank": False,
            "qexp_weight": qexp_weight,
            "blend_alpha": blend_alpha,
            "composition_weight": composition_weight,
            "with_payload": False,
        }
    ).encode()
    request = Request(
        f"{base}/search",
        data=body,
        headers={"content-type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        result: dict[str, Any] = json.load(response)
    return [hit["doc_id"] for hit in result["hits"]]


def search_features(base: str, query: str) -> list[dict[str, Any]]:
    body = json.dumps(
        {
            "index": "compositional",
            "q": query,
            "k": len(CORPUS),
            "cache": False,
            "features": True,
            "with_payload": False,
        }
    ).encode()
    request = Request(
        f"{base}/search",
        data=body,
        headers={"content-type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        result: dict[str, Any] = json.load(response)
    return [hit["features"] for hit in result["hits"]]


def encode_tokens(model: StaticModel, text: str) -> list[int]:
    return [int(tid) for tid in model.tokenize([text])[0]]


def standalone_terms(model: StaticModel, texts: list[str]) -> dict[int, str]:
    active = sorted({tid for text in texts for tid in encode_tokens(model, text)})
    terms: dict[int, str] = {}
    for tid in active:
        token = model.tokenizer.id_to_token(tid) or ""
        surface = token.removeprefix("▁").removeprefix("##")
        if not surface.isalnum() or len(surface) < 2:
            continue
        encoded = model.tokenizer.encode(surface, add_special_tokens=False)
        if encoded.ids == [tid]:
            terms[tid] = surface
    return terms


def pooled_expansion(
    model: StaticModel,
    query: str,
    term_ids: list[int],
    term_names: dict[int, str],
    limit: int = 8,
) -> list[str]:
    query_vector = model.encode([query], use_multiprocessing=False)[0].astype(
        np.float32
    )
    query_vector /= max(float(np.linalg.norm(query_vector)), 1e-9)
    vectors = model.embedding[np.asarray(term_ids)].astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    scores = vectors @ query_vector
    query_terms = set(query.lower().split())
    order = np.argsort(-scores)
    result: list[str] = []
    for pos in order:
        name = term_names[term_ids[int(pos)]]
        if name.lower() in query_terms or name in result:
            continue
        result.append(name)
        if len(result) == limit:
            break
    return result


def dense_rank(model: StaticModel, query: str, texts: list[str]) -> list[str]:
    vectors = model.encode([query, *texts], use_multiprocessing=False).astype(
        np.float32
    )
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    order = np.argsort(-(vectors[1:] @ vectors[0]))
    return [CORPUS[int(pos)][0] for pos in order]


def rank_of(ranking: list[str], doc_id: str) -> int | None:
    try:
        return ranking.index(doc_id) + 1
    except ValueError:
        return None


def main() -> int:
    if not BINARY.exists():
        raise SystemExit(f"missing {BINARY}. Run cargo build --release first")

    texts = [text for _, text in CORPUS]
    model = StaticModel.from_pretrained(MODEL_NAME)
    term_names = standalone_terms(model, texts)
    term_ids = sorted(term_names)

    with tempfile.TemporaryDirectory(prefix="compositional-", dir=ROOT / "tests") as work:
        work_dir = Path(work)
        corpus_path = work_dir / "corpus.jsonl"
        artifact = work_dir / "compositional.sift"
        write_corpus(corpus_path)
        subprocess.run(
            [
                str(BINARY),
                "build",
                "--input",
                str(corpus_path),
                "--out",
                str(artifact),
                "--model",
                MODEL_NAME,
                "--k-expand",
                "20",
                "--threshold",
                "0.45",
                "--stop-df",
                "1.0",
                "--no-payload",
                "--compositional",
            ],
            cwd=ROOT,
            check=True,
        )

        server, base = start_server(work_dir)
        try:
            print(f"model={MODEL_NAME}")
            print(f"standalone_active_terms={len(term_ids)}")

            forward = model.encode(
                ["cat eats mouse", "mouse eats cat"],
                use_multiprocessing=False,
            ).astype(np.float32)
            forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)
            print(
                "pooled_order_cosine="
                f"{float(forward[0] @ forward[1]):.6f} "
                "(1.0 means pooling cannot distinguish word order)"
            )
            feature_rows = search_features(base, "cat eats mouse")
            assert feature_rows
            assert max(row["qexp_score"] for row in feature_rows) > 0.0
            assert max(row["composition_similarity"] for row in feature_rows) > 0.0
            print(
                "integrated_features="
                f"qexp_max={max(row['qexp_score'] for row in feature_rows):.4f} "
                "composition_max="
                f"{max(row['composition_similarity'] for row in feature_rows):.4f}"
            )
            cases = [
                ("cat eats mouse", "feline_consumes_rodent"),
                ("mouse eats cat", "rodent_consumes_feline"),
                ("animals other than cats", "dog_is_animal"),
            ]
            for query, expected in cases:
                current = search(base, query, qexp_weight=0.5, blend_alpha=0.5)
                expansions = pooled_expansion(
                    model, query, term_ids, term_names
                )
                rewritten = " ".join([query, *expansions])
                pooled_sift = search(
                    base,
                    rewritten,
                    qexp_weight=0.0,
                    blend_alpha=0.0,
                )
                compositional_sift = search(
                    base,
                    query,
                    qexp_weight=0.5,
                    blend_alpha=0.5,
                    composition_weight=0.7,
                )
                pooled_dense = dense_rank(model, query, texts)
                print(f"\nquery={query!r} expected={expected}")
                print(f"  current_sift={current[:5]}")
                print(f"  pooled_terms={expansions}")
                print(f"  pooled_sift={pooled_sift[:5]}")
                print(f"  pooled_dense={pooled_dense[:5]}")
                print(f"  compositional_sift={compositional_sift[:5]}")
                print(
                    "  ranks="
                    f"current:{rank_of(current, expected)} "
                    f"pooled_sift:{rank_of(pooled_sift, expected)} "
                    f"pooled_dense:{rank_of(pooled_dense, expected)} "
                    f"compositional:{rank_of(compositional_sift, expected)}"
                )

            forward_composition = search(
                base,
                "cat eats mouse",
                qexp_weight=0.5,
                blend_alpha=0.5,
                composition_weight=0.7,
            )
            reverse_composition = search(
                base,
                "mouse eats cat",
                qexp_weight=0.5,
                blend_alpha=0.5,
                composition_weight=0.7,
            )
            assert forward_composition[0] == "cat_eats_mouse"
            assert reverse_composition[0] == "mouse_eats_cat"

            natural_not = search(
                base, "animal not cat", qexp_weight=0.5, blend_alpha=0.5
            )
            other_than = search(
                base, "animal other than cats", qexp_weight=0.5, blend_alpha=0.5
            )
            explicit_not = search(
                base, "animal -cat", qexp_weight=0.0, blend_alpha=0.0
            )
            print("\nnegation")
            print(f"  natural_animal_not_cat={natural_not[:5]}")
            print(f"  natural_animal_other_than_cats={other_than[:5]}")
            print(f"  explicit_animal_minus_cat={explicit_not[:5]}")
            assert "cat_is_animal" not in natural_not
            assert "cat_is_animal" not in other_than
            assert explicit_not[0] == "dog_is_animal"
            assert "cat_is_animal" not in explicit_not

            pure_negative = search(
                base, "not a cat", qexp_weight=0.0, blend_alpha=0.0
            )
            print(f"  pure_not_a_cat={pure_negative[:5]}")
            assert pure_negative
            assert "cat_eats_mouse" not in pure_negative
            assert "cat_is_animal" not in pure_negative
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
