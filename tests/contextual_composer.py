#!/usr/bin/env python3
"""Test contextual query-document reranking with the bundled ONNX model."""

from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target" / "release" / "sift"
CROSS_ENCODER = ROOT / "reranker" / "ce-minilm-l6"
MODEL = "minishlab/potion-multilingual-128M"
CORPUS = [
    ("feline_consumes_rodent", "A feline consumes a rodent."),
    ("rodent_consumes_feline", "A rodent consumes a feline."),
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
    process = subprocess.Popen(
        [
            str(BINARY),
            "serve",
            "--artifacts",
            str(artifact_root),
            "--bind",
            f"127.0.0.1:{port}",
            "--cross-encoder",
            str(CROSS_ENCODER),
            "--read-only",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        if process.poll() is not None:
            raise RuntimeError("Sift server exited before becoming ready")
        try:
            with urlopen(f"{base}/readyz", timeout=0.25) as response:
                if response.status == 200:
                    return process, base
        except (OSError, URLError):
            pass
        time.sleep(0.05)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError("Sift server did not become ready")


def search(base: str, query: str, contextual_weight: float) -> list[str]:
    body = json.dumps(
        {
            "index": "contextual",
            "q": query,
            "k": len(CORPUS),
            "cache": False,
            "rerank": contextual_weight > 0.0,
            "contextual_weight": contextual_weight,
            "with_payload": False,
        }
    ).encode()
    request = Request(
        f"{base}/search",
        data=body,
        headers={"content-type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        result: dict[str, Any] = json.load(response)
    return [hit["doc_id"] for hit in result["hits"]]


def average_search_ms(
    base: str, query: str, contextual_weight: float, repeats: int = 3
) -> float:
    search(base, query, contextual_weight)
    started = time.perf_counter()
    for _ in range(repeats):
        search(base, query, contextual_weight)
    return (time.perf_counter() - started) * 1000.0 / repeats


def main() -> int:
    if not BINARY.exists():
        raise SystemExit(f"missing {BINARY}. Build with --features cross-encoder")
    if not CROSS_ENCODER.joinpath("model-int8.onnx").exists():
        raise SystemExit(f"missing contextual model under {CROSS_ENCODER}")

    with tempfile.TemporaryDirectory(prefix="contextual-", dir=ROOT / "tests") as work:
        work_dir = Path(work)
        corpus = work_dir / "corpus.jsonl"
        artifact = work_dir / "contextual.sift"
        write_corpus(corpus)
        subprocess.run(
            [
                str(BINARY),
                "build",
                "--input",
                str(corpus),
                "--out",
                str(artifact),
                "--model",
                MODEL,
                "--k-expand",
                "20",
                "--threshold",
                "0.45",
                "--stop-df",
                "1.0",
            ],
            cwd=ROOT,
            check=True,
        )
        server, base = start_server(work_dir)
        try:
            baseline_forward = search(base, "cat eats mouse", 0.0)
            baseline_reverse = search(base, "mouse eats cat", 0.0)
            contextual_forward = search(base, "cat eats mouse", 0.9)
            contextual_reverse = search(base, "mouse eats cat", 0.9)
            contextual_negation = search(base, "animal not cat", 0.9)
            explicit_negation = search(base, "animal -cat", 0.0)
            sparse_ms = average_search_ms(base, "cat eats mouse", 0.0)
            contextual_ms = average_search_ms(base, "cat eats mouse", 0.9)
            print(f"baseline_forward={baseline_forward[:5]}")
            print(f"baseline_reverse={baseline_reverse[:5]}")
            print(f"contextual_forward={contextual_forward[:5]}")
            print(f"contextual_reverse={contextual_reverse[:5]}")
            print(f"contextual_negation={contextual_negation[:5]}")
            print(f"explicit_negation={explicit_negation[:5]}")
            print(f"average_latency_ms sparse={sparse_ms:.2f} contextual={contextual_ms:.2f}")
            assert contextual_forward[0] == "cat_eats_mouse"
            assert contextual_reverse[0] == "mouse_eats_cat"
            assert "cat_is_animal" not in contextual_negation
            assert explicit_negation[0] == "dog_is_animal"
            assert "cat_is_animal" not in explicit_negation
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
