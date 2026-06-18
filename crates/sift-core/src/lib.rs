//! sift-core - load a `.sift` artifact and score BM25 queries against it.
//!
//! The artifact is a directory of plain little-endian binary files, each a
//! flat array of one of: `u8`, `u32`, `u64`, `f32`. We mmap them and reinterpret
//! the bytes - the layout is defined by `sift_build` and is single-version.
//!
//! Scoring is the standard BM25 sum, but executed over the **already-expanded**
//! sparse `[V × N]` matrix. Each query token's row already contains the
//! weighted-TF contributions from its semantic neighbors (computed at index time).

use anyhow::{anyhow, Result};
use memmap2::Mmap;
use serde::Deserialize;
use std::path::PathBuf;
use tokenizers::Tokenizer;

mod open;
/// Query scoring methods (`score`, `score_wand`, `score_block_max_wand`,
/// `score_with_*`, `facet_counts`) - split into their own file for size.
mod retrieval;
mod retrieval_variants;
mod spell;

/// Multi-segment indices: incremental add / delete via immutable segments.
pub mod index_set;
pub use index_set::{
    add_tombstones, fsync_dir, fsync_dir_contents, is_tombstoned, next_segment_name,
    parse_seg_ordinal, read_tombstones, recover_orphans, IndexSet, Manifest, MergedHit,
    MergedResults, ScoreMode, SegmentRef, MANIFEST_FORMAT, TOMBSTONE_ALL,
};

#[derive(Debug, Deserialize, Clone)]
pub struct Meta {
    pub schema_version: u32,
    pub model_name: String,
    pub vocab_size: u64,
    pub n_docs: u64,
    pub n_nonzero: u64,
    #[serde(default)]
    pub n_nonzero_exact: u64,
    pub n_active_terms: u64,
    pub n_stopwords: u64,
    pub avgdl: f32,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    /// Lower-bound term frequency offset. When > 0 sift uses BM25+ (the
    /// score per matching term gains `idf * delta`, mitigating BM25's tendency
    /// to under-reward rare terms in long documents). When 0 it's classic BM25.
    #[serde(default)]
    pub bm25_delta: f32,
    pub k_expand: u32,
    pub sim_threshold: f32,
    /// On-disk packing of the doc-id arrays in indices.bin / exact_indices.bin.
    /// "u32" (the default) = plain little-endian 4-byte ints. "u24" = packed
    /// 3-byte ints, decoded into an owned Vec<u32> at open. Saves ~25% on
    /// the largest files of the artifact.
    #[serde(default = "default_indices_packing")]
    pub indices_packing: String,
    /// idf multiplier applied to WordPiece continuation (`##`) tokens at
    /// build (the `--subword-weight` damping, baked into idf.bin). Recorded
    /// so the merged path can re-apply the same damping when it recomputes
    /// collection-wide idf. Defaults to 1.0 (no damping) for older artifacts.
    #[serde(default = "default_subword_weight")]
    pub subword_weight: f32,
}

fn default_subword_weight() -> f32 {
    1.0
}

fn default_indices_packing() -> String {
    "u32".to_string()
}

/// Posting-weight storage. Weights ship on disk as either f32 (`data.bin`)
/// or f16 (`data_f16.bin`, the build default); both are mmap'd and scored
/// in place - f16 postings convert per access (one hardware instruction on
/// aarch64) and halve the memory bandwidth of the score inner loop.
#[derive(Clone, Copy)]
pub enum DataView {
    F32(&'static [f32]),
    F16(&'static [half::f16]),
}

impl DataView {
    #[inline(always)]
    pub fn len(&self) -> usize {
        match self {
            DataView::F32(s) => s.len(),
            DataView::F16(s) => s.len(),
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Weight at position `i`, widened to f32.
    #[inline(always)]
    pub fn get(&self, i: usize) -> f32 {
        match self {
            DataView::F32(s) => s[i],
            DataView::F16(s) => s[i].to_f32(),
        }
    }

    /// Sub-view over `[lo, hi)`, same dtype.
    #[inline(always)]
    pub fn slice(&self, lo: usize, hi: usize) -> DataView {
        match self {
            DataView::F32(s) => DataView::F32(&s[lo..hi]),
            DataView::F16(s) => DataView::F16(&s[lo..hi]),
        }
    }
}

/// A loaded artifact. All large arrays are mmap-backed.
pub struct Index {
    pub meta: Meta,
    pub path: PathBuf,
    pub tokenizer: Tokenizer,

    // mmaps kept alive for the lifetime of the index
    _mm_indptr: Mmap,
    /// `None` when indices.bin was packed as u24 - the bytes are decoded
    /// into `_owned_indices` at open and the mmap is released so we
    /// don't pay for it twice in RAM.
    _mm_indices: Option<Mmap>,
    /// Owned decode of a u24-packed indices.bin. Held in the struct (not
    /// leaked) so reload/compact swaps release the buffer with the Index.
    _owned_indices: Option<Box<[u32]>>,
    /// Backs `data` (either data.bin or data_f16.bin).
    _mm_data: Mmap,
    _mm_exact_indptr: Option<Mmap>,
    _mm_exact_indices: Option<Mmap>,
    _mm_exact_data: Option<Mmap>,
    _mm_bg_keys: Option<Mmap>,
    _mm_bg_indptr: Option<Mmap>,
    _mm_bg_indices: Option<Mmap>,
    _mm_bg_idf: Option<Mmap>,
    _mm_spell_words: Option<Mmap>,
    _mm_spell_word_offs: Option<Mmap>,
    _mm_spell_word_df: Option<Mmap>,
    _mm_spell_del_keys: Option<Mmap>,
    _mm_spell_del_offs: Option<Mmap>,
    _mm_spell_del_ptr: Option<Mmap>,
    _mm_spell_del_posts: Option<Mmap>,
    _mm_fwd_indptr: Option<Mmap>,
    _mm_fwd_indices: Option<Mmap>,
    _mm_fwd_data: Option<Mmap>,
    _mm_pos_indptr: Option<Mmap>,
    _mm_pos_terms: Option<Mmap>,
    _mm_qexp_indptr: Option<Mmap>,
    _mm_qexp_terms: Option<Mmap>,
    _mm_qexp_sims: Option<Mmap>,
    _mm_dedup: Option<Mmap>,
    _mm_block_max_indptr: Option<Mmap>,
    _mm_block_max: Option<Mmap>,
    _mm_rank: Vec<Mmap>,
    _mm_idf: Mmap,
    _mm_keep: Mmap,
    _mm_doc_lens: Mmap,
    _mm_doc_ids_text: Mmap,
    _mm_doc_ids_off: Mmap,
    _mm_doc_snips_text: Mmap,
    _mm_doc_snips_off: Mmap,
    _mm_payload_text: Option<Mmap>,
    _mm_payload_off: Option<Mmap>,

    // typed views (lifetime-tied to the mmaps above via &'static, see Drop note)
    pub(crate) indptr: &'static [u64],
    pub(crate) indices: &'static [u32],
    pub(crate) data: DataView,
    pub(crate) exact_indptr: Option<&'static [u64]>,
    pub(crate) exact_indices: Option<&'static [u32]>,
    pub(crate) exact_data: Option<DataView>,
    pub(crate) bigram_keys: Option<&'static [u64]>,
    pub(crate) bigram_indptr: Option<&'static [u64]>,
    pub(crate) bigram_indices: Option<&'static [u32]>,
    pub(crate) bigram_idf: Option<&'static [f32]>,
    pub(crate) spell_words: Option<&'static [u8]>,
    pub(crate) spell_word_offs: Option<&'static [u64]>,
    pub(crate) spell_word_df: Option<&'static [u32]>,
    pub(crate) spell_del_keys: Option<&'static [u8]>,
    pub(crate) spell_del_offs: Option<&'static [u64]>,
    pub(crate) spell_del_ptr: Option<&'static [u64]>,
    pub(crate) spell_del_posts: Option<&'static [u32]>,
    pub(crate) fwd_indptr: Option<&'static [u64]>,
    pub(crate) fwd_indices: Option<&'static [u32]>,
    pub(crate) fwd_data: Option<&'static [f32]>,
    pub(crate) pos_indptr: Option<&'static [u64]>,
    pub(crate) pos_terms: Option<&'static [u32]>,
    /// Query-expansion sidecar: term -> top-K (neighbor term, cosine sim).
    /// None on artifacts built before the sidecar existed.
    pub(crate) qexp_indptr: Option<&'static [u64]>,
    pub(crate) qexp_terms: Option<&'static [u32]>,
    pub(crate) qexp_sims: Option<&'static [f32]>,
    /// Per-doc canonical mask (1 = canonical, 0 = exact duplicate of an
    /// earlier doc). None when the artifact wasn't built with --dedup.
    pub(crate) dedup_canonical: Option<&'static [u8]>,
    /// Per-term offsets into the block_max array. None when the artifact
    /// wasn't built with --block-max.
    pub(crate) block_max_indptr: Option<&'static [u64]>,
    /// Per-block max BM25 contribution (`idf * stf_max_in_block`). Block
    /// size is the build-time constant BLOCK_SIZE (128).
    pub(crate) block_max: Option<&'static [f32]>,
    /// Per-doc numeric ranking attributes loaded from rank_<name>.bin
    /// sidecars. Empty when no rank fields were declared at build.
    pub(crate) rank_attrs: std::collections::HashMap<String, &'static [f32]>,
    pub(crate) idf: &'static [f32],
    pub(crate) keep: &'static [u8],
    pub(crate) doc_lens: &'static [f32],
    pub(crate) doc_ids_text: &'static [u8],
    pub(crate) doc_ids_off: &'static [u64],
    pub(crate) doc_snips_text: &'static [u8],
    pub(crate) doc_snips_off: &'static [u64],
    /// Optional per-doc raw JSON payload (the stored source document), packed
    /// like the snippet table. Present when the artifact was built with payload
    /// storage enabled. Enables returning the source and filtering on arbitrary
    /// fields.
    pub(crate) payload_text: Option<&'static [u8]>,
    pub(crate) payload_off: Option<&'static [u64]>,
}

// SAFETY: the `'static` views are slices into mmaps owned by the same struct.
// Index never escapes those slices outside its own methods; we drop the slices
// before the mmaps (Rust field-drop order is declaration order). Index is
// `!Send + !Sync` for those raw views, but since the underlying mmaps are
// `Send + Sync` and we only do read access, we override that:
unsafe impl Send for Index {}
unsafe impl Sync for Index {}

/// Touch every 4 KB page of an mmap so it's resident in the page cache.
/// `read_volatile` prevents the compiler from optimising the read away.
pub(crate) fn prewarm(mm: &Mmap) {
    const PAGE: usize = 4096;
    let bytes = &mm[..];
    let mut sum: u8 = 0;
    let mut i = 0;
    while i < bytes.len() {
        // SAFETY: in-bounds read of a single byte from the mmap slice.
        sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(&bytes[i]) });
        i += PAGE;
    }
    // touch the last byte too (last page might be short)
    if !bytes.is_empty() {
        sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(&bytes[bytes.len() - 1]) });
    }
    // sink the side-effect: an opaque write guarantees `sum` is observed.
    std::hint::black_box(sum);
}

/// The schema version of the `.sift` artifact format that this crate
/// understands. Bumped when an on-disk file layout changes incompatibly.
pub const SCHEMA_VERSION: u32 = 1;

/// Largest doc-id representable in the u24 indices packing.
pub const U24_MAX: u32 = 0x00FF_FFFF;

/// Pack doc-ids as little-endian 3-byte integers. Caller guarantees every
/// value is ≤ `U24_MAX` (the build only selects u24 packing when
/// `n_docs <= U24_MAX + 1`); the high byte is dropped.
pub fn pack_u24(vals: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 3);
    for &v in vals {
        out.push((v & 0xff) as u8);
        out.push(((v >> 8) & 0xff) as u8);
        out.push(((v >> 16) & 0xff) as u8);
    }
    out
}

/// Inverse of `pack_u24`. Errors if the byte length isn't a multiple of 3.
pub fn unpack_u24(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.len() % 3 != 0 {
        return Err(anyhow!(
            "u24 byte length {} not divisible by 3",
            bytes.len()
        ));
    }
    let n = bytes.len() / 3;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b0 = bytes[i * 3] as u32;
        let b1 = bytes[i * 3 + 1] as u32;
        let b2 = bytes[i * 3 + 2] as u32;
        out.push(b0 | (b1 << 8) | (b2 << 16));
    }
    Ok(out)
}

/// Posting-block size for the Block-Max WAND sidecar. Must match the
/// build-time block size used to compute `block_max.bin` (sift build also
/// hard-codes 128).
pub(crate) const BLOCK_MAX_BLOCK_SIZE: usize = 128;

/// Pre-tokenization normalization applied identically at index time and query
/// time. Goals:
///
///   - Collapse `401(k)`, `401 (k)` → `401k`
///   - Collapse `U.S.A.` → `USA`
///   - Collapse `1,000,000` → `1000000`
///   - Drop enclosing punctuation around identifiers (parens, brackets, quotes)
///   - Leave hyphenated words alone; WordPiece will split + filter the hyphen
pub fn normalize_text(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(s.len());
    let strip_set = "()[]{}\"'`";

    for i in 0..n {
        let c = chars[i];
        if strip_set.contains(c) {
            continue;
        }
        // Drop dots between alphanumerics (handles U.S.A. → USA)
        if c == '.' && i > 0 && i + 1 < n {
            let prev = chars[i - 1];
            let next = chars[i + 1];
            if prev.is_alphanumeric() && next.is_alphanumeric() {
                continue;
            }
        }
        // Drop commas between digits (handles 1,000,000 → 1000000)
        if c == ',' && i > 0 && i + 1 < n {
            let prev = chars[i - 1];
            let next = chars[i + 1];
            if prev.is_ascii_digit() && next.is_ascii_digit() {
                continue;
            }
        }
        out.push(c);
    }
    out
}

/// Reinterpret an mmap's bytes as a `&'static [T]` slice. Centralizes the
/// two unsafe steps every artifact field needs: a `cast_slice` reinterpret
/// plus a lifetime extension to `'static`.
///
/// SAFETY of the lifetime extension: the returned slice points into an mmap
/// owned by the `Index` for its whole lifetime; `Index` never lets the slice
/// outlive the mmap (field-drop order releases slices before mmaps), and we
/// only ever read. See the `unsafe impl Send/Sync for Index` note.
pub(crate) fn static_cast<T: Copy>(bytes: &[u8]) -> Result<&'static [T]> {
    let s = cast_slice::<T>(bytes)?;
    Ok(unsafe { std::mem::transmute::<&[T], &'static [T]>(s) })
}

/// Lifetime-extend an mmap's raw bytes to `&'static [u8]`. Same safety
/// rationale as `static_cast`.
pub(crate) fn static_bytes(bytes: &[u8]) -> &'static [u8] {
    unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) }
}

fn cast_slice<T: Copy>(bytes: &[u8]) -> Result<&[T]> {
    let elem = std::mem::size_of::<T>();
    if bytes.len() % elem != 0 {
        return Err(anyhow!(
            "byte length {} not a multiple of {}",
            bytes.len(),
            elem
        ));
    }
    let ptr = bytes.as_ptr() as *const T;
    // SAFETY: little-endian, flat-array layout written by the build pipeline;
    // alignment guaranteed by mmap base alignment (page-aligned).
    Ok(unsafe { std::slice::from_raw_parts(ptr, bytes.len() / elem) })
}

impl Index {
    pub fn n_docs(&self) -> usize {
        self.meta.n_docs as usize
    }

    /// mlock the big mmap'd posting files so the OS can't page them out
    /// under memory pressure. u24-packed indices live in an owned buffer
    /// (anonymous memory) and are skipped. Best-effort: ignores
    /// RLIMIT_MEMLOCK failures and just logs.
    pub fn mlock_postings(&self) {
        let mm = &self._mm_data;
        let rc = unsafe { libc::mlock(mm.as_ptr() as *const libc::c_void, mm.len()) };
        if rc != 0 {
            eprintln!(
                "sift: mlock(data, {} bytes) failed: {}",
                mm.len(),
                std::io::Error::last_os_error()
            );
        }
        if let Some(mm) = self._mm_indices.as_ref() {
            let rc = unsafe { libc::mlock(mm.as_ptr() as *const libc::c_void, mm.len()) };
            if rc != 0 {
                eprintln!(
                    "sift: mlock(indices.bin, {} bytes) failed: {}",
                    mm.len(),
                    std::io::Error::last_os_error()
                );
            }
        }
    }

    /// Touch every page of the big posting-list files (`indices.bin`,
    /// `data*.bin`) so they're resident in the page cache. Cheap if the
    /// kernel has already prefetched them via the `WillNeed` hint we set at
    /// open(). Safe to call from any thread; designed to be invoked from a
    /// background task.
    pub fn prewarm_postings(&self) {
        if let Some(mm) = self._mm_indices.as_ref() {
            prewarm(mm);
        }
        prewarm(&self._mm_data);
        prewarm(&self._mm_doc_ids_text);
        prewarm(&self._mm_doc_snips_text);
    }

    pub fn doc_id(&self, di: usize) -> &str {
        let lo = self.doc_ids_off[di] as usize;
        let hi = self.doc_ids_off[di + 1] as usize;
        std::str::from_utf8(&self.doc_ids_text[lo..hi]).unwrap_or("")
    }

    pub fn doc_snip(&self, di: usize) -> &str {
        let lo = self.doc_snips_off[di] as usize;
        let hi = self.doc_snips_off[di + 1] as usize;
        std::str::from_utf8(&self.doc_snips_text[lo..hi]).unwrap_or("")
    }

    /// True when this artifact stores per-doc payloads (the source documents).
    pub fn has_payload(&self) -> bool {
        self.payload_text.is_some()
    }

    /// The stored raw JSON payload for doc `di`, if present and non-empty.
    pub fn payload(&self, di: usize) -> Option<&str> {
        let text = self.payload_text?;
        let off = self.payload_off?;
        if di + 1 >= off.len() {
            return None;
        }
        let lo = off[di] as usize;
        let hi = off[di + 1] as usize;
        if lo >= hi || hi > text.len() {
            return None;
        }
        std::str::from_utf8(&text[lo..hi]).ok()
    }

    /// Tokenize `text` to a deduplicated list of content-token ids. Applies
    /// the same normalization used at index time so query vocab matches.
    pub fn tokenize_query(&self, text: &str) -> Vec<u32> {
        let normalized = normalize_text(text);
        let enc = match self.tokenizer.encode(normalized.as_str(), false) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let mut out: Vec<u32> = enc
            .get_ids()
            .iter()
            .copied()
            .filter(|&id| (id as usize) < self.keep.len() && self.keep[id as usize] == 1)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Like `tokenize_query` but preserves order and duplicates (used by the
    /// boolean / phrase parser).
    pub fn tokenize_query_keep_order(&self, text: &str) -> Vec<u32> {
        let normalized = normalize_text(text);
        let enc = match self.tokenizer.encode(normalized.as_str(), false) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        enc.get_ids()
            .iter()
            .copied()
            .filter(|&id| (id as usize) < self.keep.len() && self.keep[id as usize] == 1)
            .collect()
    }

    /// Parse a query string into (included_token_ids, excluded_token_ids).
    /// Terms prefixed with `-` are excluded ("tax -retirement"). Hyphens
    /// inside words (e.g., "covid-19") are unaffected because we split on
    /// whitespace, not punctuation.
    pub fn parse_query(&self, text: &str) -> (Vec<u32>, Vec<u32>) {
        let mut inc_text = String::new();
        let mut exc_text = String::new();
        for word in text.split_whitespace() {
            if let Some(rest) = word.strip_prefix('-') {
                if !rest.is_empty() {
                    if !exc_text.is_empty() {
                        exc_text.push(' ');
                    }
                    exc_text.push_str(rest);
                    continue;
                }
            }
            if !inc_text.is_empty() {
                inc_text.push(' ');
            }
            inc_text.push_str(word);
        }
        let inc = self.tokenize_query(&inc_text);
        let exc = if exc_text.is_empty() {
            Vec::new()
        } else {
            self.tokenize_query(&exc_text)
        };
        (inc, exc)
    }

    /// True iff doc `di` has any posting under term `tid` in the EXACT
    /// (un-expanded) inverted index. Used by the boolean-NOT path so that
    /// "-retirement" only excludes docs that literally say retirement, not
    /// docs that semantic-matched it via a neighbor.
    fn doc_has_exact_term(&self, di: u32, tid: u32) -> bool {
        let (e_indptr, e_indices) = match (self.exact_indptr, self.exact_indices) {
            (Some(a), Some(b)) => (a, b),
            _ => return false, // no exact CSR → can't safely exclude
        };
        let t = tid as usize;
        if t + 1 >= e_indptr.len() {
            return false;
        }
        let lo = e_indptr[t] as usize;
        let hi = e_indptr[t + 1] as usize;
        e_indices[lo..hi].binary_search(&di).is_ok()
    }

    /// Look up a bigram by hash. Returns (idf, posting_doc_ids_slice) if present.
    fn lookup_bigram(&self, hash: u64) -> Option<(f32, &[u32])> {
        let keys = self.bigram_keys?;
        let indptr = self.bigram_indptr?;
        let indices = self.bigram_indices?;
        let idfs = self.bigram_idf?;
        let pos = keys.binary_search(&hash).ok()?;
        let lo = indptr[pos] as usize;
        let hi = indptr[pos + 1] as usize;
        Some((idfs[pos], &indices[lo..hi]))
    }

    /// Bigram bonus for a single doc against ordered query tokens, scaled by
    /// `weight`. Cheap: O(query_bigrams × log(docs_per_bigram)). Used
    /// post-scoring on top-K only.
    pub fn bigram_bonus_for_doc(&self, di: u32, q_in_order: &[u32], weight: f32) -> f32 {
        if self.bigram_keys.is_none() || weight == 0.0 {
            return 0.0;
        }
        let mut bonus = 0.0_f32;
        for w in q_in_order.windows(2) {
            let h = ((w[0] as u64) << 32) | (w[1] as u64);
            if let Some((idf_bg, docs)) = self.lookup_bigram(h) {
                if docs.binary_search(&di).is_ok() {
                    bonus += weight * idf_bg;
                }
            }
        }
        bonus
    }

    /// Proximity bonus for a single doc: `weight / (1 + min_gap)` where
    /// min_gap is the smallest token distance between occurrences of two
    /// *distinct* query terms in the doc. Requires the positions sidecar
    /// (returns 0.0 without it). Used post-scoring on top-K only; cost is
    /// one walk of the doc's term sequence.
    pub fn proximity_bonus_for_doc(&self, di: u32, q_tokens: &[u32], weight: f32) -> f32 {
        if weight == 0.0 || q_tokens.len() < 2 {
            return 0.0;
        }
        let seq = match self.doc_term_sequence(di) {
            Some(s) => s,
            None => return 0.0,
        };
        // Walk the sequence remembering the last position of each query term;
        // the candidate gap at each match is (here - last_other_term_pos).
        let mut last_pos: Vec<(u32, usize)> = Vec::with_capacity(q_tokens.len());
        let mut min_gap = usize::MAX;
        for (pos, &t) in seq.iter().enumerate() {
            if !q_tokens.contains(&t) {
                continue;
            }
            for &(other, opos) in &last_pos {
                if other != t {
                    let gap = pos - opos;
                    if gap < min_gap {
                        min_gap = gap;
                    }
                }
            }
            match last_pos.iter_mut().find(|(tt, _)| *tt == t) {
                Some(e) => e.1 = pos,
                None => last_pos.push((t, pos)),
            }
        }
        if min_gap == usize::MAX {
            return 0.0;
        }
        weight / (1.0 + min_gap as f32)
    }

    /// True when the artifact carries the query-expansion sidecar.
    pub fn has_qexp(&self) -> bool {
        self.qexp_indptr.is_some() && self.qexp_terms.is_some() && self.qexp_sims.is_some()
    }

    /// Top-K embedding neighbors of `tid` (term ids + cosine sims) from the
    /// query-expansion sidecar, or None if absent / out of range / empty.
    pub fn qexp_neighbors(&self, tid: u32) -> Option<(&[u32], &[f32])> {
        let indptr = self.qexp_indptr?;
        let terms = self.qexp_terms?;
        let sims = self.qexp_sims?;
        let t = tid as usize;
        if t + 1 >= indptr.len() {
            return None;
        }
        let lo = indptr[t] as usize;
        let hi = indptr[t + 1] as usize;
        if lo == hi {
            return None;
        }
        Some((&terms[lo..hi], &sims[lo..hi]))
    }

    /// WAND-pruned BM25 scorer. Same result as `score()` modulo float
    /// associativity; skips posting-list entries that can't possibly enter
    /// the top-k by using a per-term upper-bound on contribution. Best
    /// returns on long posting lists with selective query terms - i.e.
    /// large-corpus deployments where the dense Vec<f32> accumulator in
    /// `score()` dominates latency.
    pub fn has_block_max(&self) -> bool {
        self.block_max_indptr.is_some() && self.block_max.is_some()
    }

    /// Reverse-map a token id to its surface form via the tokenizer.
    pub fn token_string(&self, tid: u32) -> Option<String> {
        self.tokenizer.id_to_token(tid)
    }

    /// Names of all ranking-attribute fields the artifact was built with.
    pub fn rank_field_names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.rank_attrs.keys().map(|s| s.as_str()).collect();
        v.sort();
        v
    }

    /// Return the value of ranking attribute `field` for doc `di`, or None
    /// if the field wasn't materialized at build time.
    pub fn rank_value(&self, field: &str, di: u32) -> Option<f32> {
        let slice = self.rank_attrs.get(field)?;
        slice.get(di as usize).copied()
    }

    pub fn has_positions(&self) -> bool {
        self.pos_indptr.is_some() && self.pos_terms.is_some()
    }

    pub fn has_dedup(&self) -> bool {
        self.dedup_canonical.is_some()
    }

    /// True iff doc `di` is canonical (not a flagged near-duplicate). Always
    /// returns true if the artifact wasn't built with --dedup.
    pub fn is_canonical(&self, di: u32) -> bool {
        match self.dedup_canonical {
            Some(m) => m.get(di as usize).map(|&b| b != 0).unwrap_or(true),
            None => true,
        }
    }

    /// Doc-order term sequence for doc `di`. None if positions sidecar
    /// wasn't built or `di` is out of range.
    pub fn doc_term_sequence(&self, di: u32) -> Option<&[u32]> {
        let indptr = self.pos_indptr?;
        let terms = self.pos_terms?;
        let i = di as usize;
        if i + 1 >= indptr.len() {
            return None;
        }
        let lo = indptr[i] as usize;
        let hi = indptr[i + 1] as usize;
        Some(&terms[lo..hi])
    }

    /// True iff doc `di`'s term sequence contains `phrase` as a contiguous
    /// subsequence. Returns false (rather than None) when the positional
    /// index isn't loaded, so callers can use this as a hard filter only when
    /// `has_positions()` is also true.
    pub fn doc_contains_phrase(&self, di: u32, phrase: &[u32]) -> bool {
        if phrase.is_empty() {
            return true;
        }
        let seq = match self.doc_term_sequence(di) {
            Some(s) => s,
            None => return false,
        };
        if phrase.len() > seq.len() {
            return false;
        }
        let last = seq.len() - phrase.len();
        for i in 0..=last {
            if seq[i] == phrase[0] && seq[i..i + phrase.len()] == *phrase {
                return true;
            }
        }
        false
    }

    pub fn has_forward(&self) -> bool {
        self.fwd_indptr.is_some() && self.fwd_indices.is_some() && self.fwd_data.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub doc_idx: u32,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct SearchResults {
    pub hits: Vec<Hit>,
    pub matched_query_terms: u32,
    pub elapsed_us: u64,
}

#[derive(Debug, Clone)]
pub struct FeatureHit {
    pub doc_idx: u32,
    pub bm25_combined: f32,
    pub bm25_exact: f32,
    pub bm25_semantic: f32,
    pub coverage: f32,
    pub doc_len: f32,
}

#[derive(Debug, Clone)]
pub struct FeatureResults {
    pub hits: Vec<FeatureHit>,
    pub matched_query_terms: u32,
    pub elapsed_us: u64,
}

#[derive(Debug, Clone)]
pub struct TermContribution {
    pub tid: u32,
    pub token: String,
    pub idf: f32,
    pub tf_in_doc: f32,
    pub contribution: f32,
}

#[derive(Debug, Clone)]
pub struct ExplainHit {
    pub doc_idx: u32,
    pub score: f32,
    pub doc_len: f32,
    pub terms: Vec<TermContribution>,
}

#[derive(Debug, Clone)]
pub struct ExplainResults {
    pub hits: Vec<ExplainHit>,
    pub matched_query_terms: u32,
    pub elapsed_us: u64,
}

#[cfg(test)]
mod properties {
    use super::*;
    use hegel::generators as gs;
    use hegel::TestCase;

    // u24 pack→unpack must round-trip exactly for every doc-id in range.
    // A bug here silently corrupts every posting list of a u24 artifact.
    #[hegel::test]
    fn u24_roundtrip(tc: TestCase) {
        let vals: Vec<u32> = tc
            .draw(gs::vecs(gs::integers::<u32>()))
            .into_iter()
            .map(|v| v & U24_MAX) // restrict to the documented in-range domain
            .collect();
        let packed = pack_u24(&vals);
        assert_eq!(packed.len(), vals.len() * 3, "packed length wrong");
        let back = unpack_u24(&packed).expect("unpack valid 3-byte stream");
        assert_eq!(back, vals, "u24 round-trip mismatch");
    }

    // unpack_u24 rejects byte streams whose length isn't a multiple of 3
    // rather than silently truncating/over-reading.
    #[hegel::test]
    fn u24_unpack_rejects_misaligned(tc: TestCase) {
        let bytes: Vec<u8> = tc.draw(gs::vecs(gs::integers::<u8>()));
        let res = unpack_u24(&bytes);
        if bytes.len() % 3 == 0 {
            assert!(res.is_ok());
        } else {
            assert!(res.is_err(), "misaligned u24 stream must error");
        }
    }

    // Normalising twice gives the same result as normalising once. If this
    // failed, query-time normalization could disagree with index-time
    // normalization on cleaned text and silently miss matches.
    #[hegel::test]
    fn normalize_text_idempotent(tc: TestCase) {
        let s: String = tc.draw(gs::text());
        let once = normalize_text(&s);
        let twice = normalize_text(&once);
        assert_eq!(
            once, twice,
            "normalize_text is not idempotent on input: {s:?}"
        );
    }

    // Reflexivity: any string is within edit distance 1 of itself.
    #[hegel::test]
    fn edit_distance_reflexive(tc: TestCase) {
        let s: String = tc.draw(gs::text());
        assert!(
            Index::edit_distance_le1(&s, &s),
            "edit_distance_le1(s, s) returned false for {s:?}",
        );
    }

    // Symmetry: edit_distance_le1(a, b) == edit_distance_le1(b, a). A failure
    // here would mean spell correction would non-deterministically depend on
    // argument order.
    #[hegel::test]
    fn edit_distance_symmetric(tc: TestCase) {
        let a: String = tc.draw(gs::text());
        let b: String = tc.draw(gs::text());
        let ab = Index::edit_distance_le1(&a, &b);
        let ba = Index::edit_distance_le1(&b, &a);
        assert_eq!(ab, ba, "edit_distance_le1 not symmetric on ({a:?}, {b:?})");
    }

    // Sanity floor: strings whose char-length differs by more than 1 cannot
    // be at edit distance ≤ 1.
    #[hegel::test]
    fn edit_distance_length_floor(tc: TestCase) {
        let a: String = tc.draw(gs::text());
        let b: String = tc.draw(gs::text());
        let la = a.chars().count();
        let lb = b.chars().count();
        if la.abs_diff(lb) > 1 {
            assert!(
                !Index::edit_distance_le1(&a, &b),
                "edit_distance_le1 claimed dist≤1 across char-lens {la} vs {lb}: ({a:?}, {b:?})",
            );
        }
    }
}
