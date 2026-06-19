"""sift artifact builder.

Single function ``build_artifact(corpus, out_dir, cfg)``. Implements the v4
recipe: WordPiece tokenize (via m2v), HNSW over m2v subword embeddings,
index-time expansion to top-K HNSW neighbors with sim >= threshold,
weighted-CSR storage.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import hnswlib
from model2vec import StaticModel
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm


SCHEMA_VERSION = 1
SPECIAL = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}


@dataclass
class BuildConfig:
    """All knobs the offline build cares about."""
    model_name: str = "minishlab/potion-base-8M"
    k_expand: int = 10
    sim_threshold: float = 0.65
    stop_df_ratio: float = 0.4
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # If True, normalise embeddings before HNSW (recommended).
    normalize_embeddings: bool = True


def _make_keep_mask(tokenizer) -> np.ndarray:
    """1 = content token (alphanumeric, not special, not pure punct)."""
    vocab = tokenizer.get_vocab()
    V = len(vocab)
    keep = np.zeros(V, dtype=np.uint8)
    for tok_str, tid in vocab.items():
        if tok_str in SPECIAL:
            continue
        clean = tok_str[2:] if tok_str.startswith("##") else tok_str
        if not clean:
            continue
        if not any(c.isalnum() for c in clean):
            continue
        if len(clean) == 1 and not clean.isalnum():
            continue
        keep[tid] = 1
    return keep


def _tokenize(text: str, tokenizer, keep_mask: np.ndarray) -> list[int]:
    enc = tokenizer.encode(text)
    return [tid for tid in enc.ids if keep_mask[tid]]


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def build_artifact(
    corpus: Iterable[tuple[str, str]],
    out_dir: str | Path,
    cfg: BuildConfig | None = None,
    verbose: bool = True,
    snippet_chars: int = 320,
) -> dict:
    """Build a .sift artifact directory from ``(doc_id, text)`` pairs.

    Returns a stats dict (nnz, build times, sizes).
    """
    cfg = cfg or BuildConfig()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    log = print if verbose else (lambda *a, **k: None)
    t_total = time.time()

    # ── 1. Load model + tokenizer
    log(f"[1/6] loading {cfg.model_name}")
    model = StaticModel.from_pretrained(cfg.model_name)
    tokenizer = model.tokenizer
    keep_mask = _make_keep_mask(tokenizer)
    V = len(tokenizer.get_vocab())
    log(f"      vocab={V:,} content-tokens={int(keep_mask.sum()):,}")

    # ── 2. Tokenize corpus, collect df, doc_lens
    log("[2/6] tokenizing corpus")
    t0 = time.time()
    doc_ids: list[str] = []
    doc_snips: list[str] = []
    tokenized: list[list[int]] = []
    df = np.zeros(V, dtype=np.int64)
    for doc_id, text in corpus:
        toks = _tokenize(text, tokenizer, keep_mask)
        doc_ids.append(doc_id)
        snip = (text or "")[:snippet_chars]
        # collapse whitespace so the snippet stays one line
        doc_snips.append(" ".join(snip.split()))
        tokenized.append(toks)
        for t in set(toks):
            df[t] += 1
    N = len(doc_ids)
    log(f"      {N:,} docs in {time.time()-t0:.1f}s")

    # ── 3. Stopword cut (purely on df ratio, language-agnostic)
    stop_mask = (df > (cfg.stop_df_ratio * N)).astype(np.uint8)
    n_stops = int(stop_mask.sum())
    log(f"[3/6] stopwords (df/N > {cfg.stop_df_ratio}): {n_stops}")
    # apply stopword filter
    if n_stops:
        stop_set = set(np.where(stop_mask)[0].tolist())
        for i, toks in enumerate(tokenized):
            tokenized[i] = [t for t in toks if t not in stop_set]
        # recompute df on kept tokens
        df = np.zeros(V, dtype=np.int64)
        for toks in tokenized:
            for t in set(toks):
                df[t] += 1

    doc_lens = np.array([len(d) for d in tokenized], dtype=np.float32)
    avgdl = float(doc_lens.mean()) if N else 1.0

    # IDF (Robertson-Spärck Jones, smoothed) - only defined where df > 0
    idf = np.zeros(V, dtype=np.float32)
    mask = df > 0
    idf[mask] = np.log((N - df[mask] + 0.5) / (df[mask] + 0.5) + 1.0).astype(np.float32)

    # active = appears in any kept doc
    active = mask.copy()
    active_ids = np.where(active)[0]
    log(f"      active vocab: {active.sum():,}")

    # ── 4. Build HNSW over active subwords (m2v embedding table)
    log("[4/6] building HNSW")
    t0 = time.time()
    embs_all = model.embedding.astype(np.float32)
    if cfg.normalize_embeddings:
        embs_all = _normalize(embs_all)
    active_embs = embs_all[active_ids]
    dim = embs_all.shape[1]
    hnsw = hnswlib.Index(space="cosine", dim=dim)
    hnsw.init_index(max_elements=len(active_ids), ef_construction=200, M=32)
    hnsw.add_items(active_embs, active_ids.tolist())  # labels = vocab IDs
    hnsw.set_ef(100)
    log(f"      hnsw {time.time()-t0:.2f}s")

    # ── 5. Index-time expansion → COO triplets → CSR
    log(f"[5/6] expanding (k={cfg.k_expand}, sim>={cfg.sim_threshold})")
    t0 = time.time()
    inverted: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for di, toks in enumerate(tokenized):
        for t, tf in Counter(toks).items():
            inverted[t].append((di, tf))

    # batched HNSW for expansion sources
    labels, dists = hnsw.knn_query(active_embs, k=min(cfg.k_expand + 1, len(active_ids)))
    sims = 1.0 - dists

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rows_a, cols_a, data_a = rows.append, cols.append, data.append

    # exact postings
    for term_id, postings in inverted.items():
        for doc_idx, tf in postings:
            rows_a(term_id); cols_a(doc_idx); data_a(float(tf))

    # expansion
    for i, w in enumerate(tqdm(active_ids, desc="      expansion", disable=not verbose)):
        w_int = int(w)
        post = inverted.get(w_int)
        if not post:
            continue
        for j in range(labels.shape[1]):
            n = int(labels[i, j])
            sim = float(sims[i, j])
            if n == w_int or sim < cfg.sim_threshold:
                continue
            for doc_idx, tf in post:
                rows_a(n); cols_a(doc_idx); data_a(tf * sim)
    accum_t = time.time() - t0

    mat = coo_matrix(
        (np.array(data, dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(V, N),
    ).tocsr()
    mat.sum_duplicates()
    log(f"      accum {accum_t:.2f}s  nnz={mat.nnz:,}")

    # ── 6. Write artifact files
    log(f"[6/6] writing artifact to {out_dir}")
    indptr = mat.indptr.astype(np.uint64)
    indices = mat.indices.astype(np.uint32)
    csr_data = mat.data.astype(np.float32)
    (out_dir / "indptr.bin").write_bytes(indptr.tobytes())
    (out_dir / "indices.bin").write_bytes(indices.tobytes())
    (out_dir / "data.bin").write_bytes(csr_data.tobytes())
    (out_dir / "idf.bin").write_bytes(idf.tobytes())
    (out_dir / "vocab_keep.bin").write_bytes(keep_mask.tobytes())
    (out_dir / "doc_lens.bin").write_bytes(doc_lens.tobytes())

    # packed doc ids
    text_bytes = bytearray()
    offsets = np.zeros(N + 1, dtype=np.uint64)
    for i, did in enumerate(doc_ids):
        b = did.encode("utf-8")
        offsets[i + 1] = offsets[i] + len(b)
        text_bytes.extend(b)
    (out_dir / "doc_ids_text.bin").write_bytes(bytes(text_bytes))
    (out_dir / "doc_ids_off.bin").write_bytes(offsets.tobytes())

    # packed doc snippets (same packed-strings format)
    snip_bytes = bytearray()
    snip_off = np.zeros(N + 1, dtype=np.uint64)
    for i, s in enumerate(doc_snips):
        b = s.encode("utf-8")
        snip_off[i + 1] = snip_off[i] + len(b)
        snip_bytes.extend(b)
    (out_dir / "doc_snips_text.bin").write_bytes(bytes(snip_bytes))
    (out_dir / "doc_snips_off.bin").write_bytes(snip_off.tobytes())

    # tokenizer - save as HF format so the Rust `tokenizers` crate can load it
    tokenizer.save(str(out_dir / "tokenizer.json"))

    meta = {
        "schema_version": SCHEMA_VERSION,
        "model_name": cfg.model_name,
        "vocab_size": int(V),
        "n_docs": int(N),
        "n_nonzero": int(mat.nnz),
        "n_active_terms": int(active.sum()),
        "n_stopwords": int(n_stops),
        "avgdl": avgdl,
        "bm25_k1": cfg.bm25_k1,
        "bm25_b": cfg.bm25_b,
        "k_expand": cfg.k_expand,
        "sim_threshold": cfg.sim_threshold,
        "build_seconds": time.time() - t_total,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log(f"      wrote {N:,} docs, {mat.nnz:,} postings, total {time.time()-t_total:.1f}s")
    return meta
