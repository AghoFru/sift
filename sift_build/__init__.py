"""sift - offline index builder.

The single public entry point is ``sift_build.build.build_artifact``. The
``sift`` CLI is a thin wrapper around it.

Artifact layout (directory):
    meta.json          schema, config, counts
    tokenizer.json     HuggingFace tokenizer (same one the Rust server loads)
    vocab_keep.bin     u8[vocab_size]      1 = content token, 0 = special/punct/stop
    idf.bin            f32[vocab_size]     BM25 IDF per token id (0 for inactive)
    doc_lens.bin       f32[N]              document lengths (in kept-token count)
    doc_ids_text.bin   utf8 bytes          concatenated doc-id strings
    doc_ids_off.bin    u64[N+1]            byte offsets into doc_ids_text.bin
    indptr.bin         u64[V+1]            CSR row pointer
    indices.bin        u32[nnz]            CSR column indices (doc ids 0..N)
    data.bin           f32[nnz]            CSR weighted TF values
"""

from .build import build_artifact, BuildConfig  # noqa: F401
