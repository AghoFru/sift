"""In-process adapters for ir-bench: bm25s (Python) and Sift (C API through ctypes)."""

import ctypes
import importlib.metadata
import json
from contextlib import contextmanager
from pathlib import Path

from ir_bench.cache import digest
from ir_bench.dataset import documents, identifier


class BM25S:
    boundary = (
        "bm25s call in the current process: query tokenization, stemming, scoring, top-k, "
        "one thread"
    )

    def __init__(self, config):
        if set(config) - {"k1", "b", "method"}:
            raise ValueError("Unknown bm25s configuration field.")
        self.config = {"k1": 1.5, "b": 0.75, "method": "lucene", **config}

    def identity(self):
        return {
            "engine": "bm25s",
            "bm25s": importlib.metadata.version("bm25s"),
            "pystemmer": importlib.metadata.version("PyStemmer"),
            "numpy": importlib.metadata.version("numpy"),
            "configuration": self.config,
            "tokenization": "bm25s.tokenize, stopwords=en, english Snowball stemmer",
        }

    def build(self, corpus, artifact):
        import bm25s
        import Stemmer

        ids, texts = [], []
        for doc_id, title, text in documents(corpus):
            ids.append(doc_id)
            texts.append(title + "\n" + text)
        stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
        retriever = bm25s.BM25(k1=self.config["k1"], b=self.config["b"],
                               method=self.config["method"])
        retriever.index(tokens, show_progress=False)
        retriever.save(str(artifact / "index"))
        (artifact / "ids.json").write_text(json.dumps(ids))

    @contextmanager
    def open(self, artifact):
        import bm25s
        import numpy as np
        import Stemmer

        retriever = bm25s.BM25.load(str(artifact / "index"))
        ids = json.loads((artifact / "ids.json").read_text())
        vocab = retriever.vocab_dict
        stemmer = Stemmer.Stemmer("english")

        def search(query, depth):
            tokens = bm25s.tokenize(
                [query], stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False
            )[0]
            known = [token for token in tokens if token in vocab]
            if not known:
                return []
            k = min(depth, len(ids))
            found, scores = retriever.retrieve([known], k=k, show_progress=False, n_threads=1)
            found, scores = found[0], scores[0]
            # Keep only matching documents, ordered like the Terrier adapter (score, then ID).
            keep = scores > 0
            rows = sorted(
                zip(scores[keep].tolist(), (ids[i] for i in np.asarray(found)[keep])),
                key=lambda row: (row[0], row[1]),
                reverse=True,
            )
            return [identifier(doc_id) for _, doc_id in rows[:depth]]

        yield search


class SiftFFI:
    boundary = (
        "Sift C API call through ctypes in the current process: JSON request, search, "
        "JSON response parse, one thread"
    )

    def __init__(self, config):
        if set(config) - {"library", "binary", "model"} or not {"library", "binary", "model"} <= (
            config.keys()
        ):
            raise ValueError("SiftFFI requires library, binary, and model paths.")
        self.library = Path(config["library"]).resolve(strict=True)
        self.binary = Path(config["binary"]).resolve(strict=True)
        self.model = Path(config["model"]).resolve(strict=True)

    def identity(self):
        return {**self.build_identity(), "library_sha256": digest(self.library)}

    def build_identity(self):
        # Matches the HTTP Sift adapter's build so the index is built the same way.
        return {
            "engine": "sift-ffi",
            "binary_sha256": digest(self.binary),
            "model": {
                name: digest(self.model / name) for name in ("tokenizer.json", "model.safetensors")
            },
        }

    def build(self, corpus, artifact):
        from ir_bench.adapters import Sift

        Sift({"binary": str(self.binary), "model": str(self.model)}).build(corpus, artifact)

    @contextmanager
    def open(self, artifact):
        lib = ctypes.CDLL(str(self.library))

        class SiftString(ctypes.Structure):
            _fields_ = [("data", ctypes.c_char_p), ("len", ctypes.c_size_t)]

        lib.sift_open.argtypes = [SiftString, ctypes.POINTER(ctypes.c_void_p)]
        lib.sift_open.restype = ctypes.c_int32
        lib.sift_search.argtypes = [
            ctypes.c_void_p,
            SiftString,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.sift_search.restype = ctypes.c_int32
        lib.sift_string_free.argtypes = [ctypes.c_void_p]
        lib.sift_close.argtypes = [ctypes.c_void_p]
        lib.sift_last_error.restype = ctypes.c_char_p

        def text(value):
            raw = value.encode()
            return SiftString(raw, len(raw))

        handle = ctypes.c_void_p()
        if lib.sift_open(text(str(artifact / "docs.sift")), ctypes.byref(handle)) != 0:
            raise RuntimeError(lib.sift_last_error().decode())

        def search(query, depth):
            body = json.dumps(
                {"q": query, "k": depth, "cache": False, "with_payload": False}
            )
            output = ctypes.c_void_p()
            if lib.sift_search(handle, text(body), ctypes.byref(output)) != 0:
                raise RuntimeError(lib.sift_last_error().decode())
            try:
                raw = ctypes.string_at(output.value)
            finally:
                lib.sift_string_free(output)
            return [hit["doc_id"] for hit in json.loads(raw)["hits"]]

        try:
            yield search
        finally:
            lib.sift_close(handle)


