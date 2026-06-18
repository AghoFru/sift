//! Core sparse retrieval for [`Index`].
//!
//! This module owns BM25 accumulation, posting-list traversal, WAND and
//! Block-Max WAND execution, facets, grouped terms, and ranking features.
//!
//! The dense scorers share two building blocks:
//!
//!   - [`Index::term_postings`] resolves a query token to its posting slice
//!     (cols + weights + idf), dispatching on the on-disk weight dtype
//!     (f32 or f16) via [`DataView`].
//!   - A *touched list*: while accumulating, the first time a doc's score
//!     leaves zero its id is recorded. Top-k selection then runs over only
//!     the touched docs instead of all `n_docs`, which on large corpora is
//!     the difference between scanning millions of entries per query and
//!     scanning just the matched set.

use crate::{
    DataView, FeatureHit, FeatureResults, Hit, Index, SearchResults, BLOCK_MAX_BLOCK_SIZE,
};
use std::time::Instant;

/// Widening accessor over posting-weight storage so the accumulation loops
/// monomorphize per dtype (no per-element branch).
trait Weight: Copy {
    fn widen(self) -> f32;
}
impl Weight for f32 {
    #[inline(always)]
    fn widen(self) -> f32 {
        self
    }
}
impl Weight for half::f16 {
    #[inline(always)]
    fn widen(self) -> f32 {
        self.to_f32()
    }
}

/// BM25 saturation constants, hoisted out of the per-posting loop.
#[derive(Clone, Copy)]
pub(crate) struct Bm25 {
    k1: f32,
    b: f32,
    delta: f32,
    avgdl: f32,
}

impl Bm25 {
    #[inline(always)]
    pub(crate) fn of(idx: &Index) -> Self {
        Self::with_avgdl(idx, idx.meta.avgdl)
    }

    /// Like [`Bm25::of`] but with an externally supplied average document
    /// length. Used by the merged path so every segment normalizes against
    /// the collection-wide avgdl instead of its own.
    #[inline(always)]
    pub(crate) fn with_avgdl(idx: &Index, avgdl: f32) -> Self {
        Bm25 {
            k1: idx.meta.bm25_k1,
            b: idx.meta.bm25_b,
            delta: idx.meta.bm25_delta,
            avgdl,
        }
    }

    /// Saturated-TF contribution for a single posting.
    #[inline(always)]
    pub(crate) fn stf(&self, wtf: f32, dl: f32) -> f32 {
        let denom = wtf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl);
        (wtf * (self.k1 + 1.0)) / denom + self.delta
    }
}

/// Sum-accumulate one posting list into `scores`, recording each doc the
/// first time its score leaves zero. All contributions are non-negative, so
/// the zero-to-positive transition fires exactly once per doc.
#[inline]
fn accumulate_into<W: Weight>(
    cols: &[u32],
    vals: &[W],
    doc_lens: &[f32],
    p: Bm25,
    wi: f32, // weight * idf, folded
    scores: &mut [f32],
    touched: &mut Vec<u32>,
) {
    for (col, val) in cols.iter().zip(vals.iter()) {
        let di = *col as usize;
        let contribution = wi * p.stf(val.widen(), doc_lens[di]);
        let s = &mut scores[di];
        let old = *s;
        *s = old + contribution;
        if old == 0.0 && *s > 0.0 {
            touched.push(*col);
        }
    }
}

/// Max-accumulate (used by OR groups): keep the largest single contribution
/// per doc instead of the sum.
#[inline]
fn max_into<W: Weight>(
    cols: &[u32],
    vals: &[W],
    doc_lens: &[f32],
    p: Bm25,
    idf: f32,
    scores: &mut [f32],
    touched: &mut Vec<u32>,
) {
    for (col, val) in cols.iter().zip(vals.iter()) {
        let di = *col as usize;
        let contribution = idf * p.stf(val.widen(), doc_lens[di]);
        let cur = &mut scores[di];
        if contribution > *cur {
            if *cur == 0.0 {
                touched.push(*col);
            }
            *cur = contribution;
        }
    }
}

#[inline]
pub(crate) fn accumulate(
    cols: &[u32],
    vals: DataView,
    doc_lens: &[f32],
    p: Bm25,
    wi: f32,
    scores: &mut [f32],
    touched: &mut Vec<u32>,
) {
    match vals {
        DataView::F32(v) => accumulate_into(cols, v, doc_lens, p, wi, scores, touched),
        DataView::F16(v) => accumulate_into(cols, v, doc_lens, p, wi, scores, touched),
    }
}

/// Resolve the touched set into the top-k hits, sorted by descending score.
/// Consumes `touched` (reorders it in place).
pub(crate) fn top_hits(scores: &[f32], touched: &mut Vec<u32>, k: usize) -> Vec<Hit> {
    if k == 0 {
        return Vec::new();
    }
    let desc = |a: &u32, b: &u32| {
        scores[*b as usize]
            .partial_cmp(&scores[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    if touched.len() > k {
        touched.select_nth_unstable_by(k - 1, desc);
        touched.truncate(k);
    }
    touched.sort_unstable_by(desc);
    touched
        .iter()
        .map(|&di| Hit {
            doc_idx: di,
            score: scores[di as usize],
        })
        .collect()
}

/// Min-heap entry for the WAND scorers: orders by score, doc id as the
/// tie-break so ordering is total.
#[derive(PartialEq)]
struct HeapEntry(f32, u32);
impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Index {
    /// Posting slice for one query token against the expanded index, looking
    /// the term's idf up in `idf_tab` (the segment's own `idf` for normal
    /// queries, or a collection-wide idf for the merged path). None when the
    /// token can't contribute. Returns `(doc_ids, weights, idf)`.
    #[inline]
    pub(crate) fn term_postings_idf(
        &self,
        tid: u32,
        idf_tab: &[f32],
    ) -> Option<(&'static [u32], DataView, f32)> {
        let t = tid as usize;
        if t >= idf_tab.len() || t + 1 >= self.indptr.len() {
            return None;
        }
        let idf = idf_tab[t];
        if idf <= 0.0 {
            return None;
        }
        let lo = self.indptr[t] as usize;
        let hi = self.indptr[t + 1] as usize;
        if lo == hi {
            return None;
        }
        Some((&self.indices[lo..hi], self.data.slice(lo, hi), idf))
    }

    #[inline]
    pub(crate) fn term_postings(&self, tid: u32) -> Option<(&'static [u32], DataView, f32)> {
        self.term_postings_idf(tid, self.idf)
    }

    /// Same as [`Index::term_postings_idf`] but over the exact (un-expanded)
    /// CSR. None when the sidecar is absent or the row is empty.
    #[inline]
    pub(crate) fn term_postings_exact_idf(
        &self,
        tid: u32,
        idf_tab: &[f32],
    ) -> Option<(&'static [u32], DataView, f32)> {
        let (e_indptr, e_indices, e_data) =
            match (self.exact_indptr, self.exact_indices, self.exact_data) {
                (Some(a), Some(b), Some(c)) => (a, b, c),
                _ => return None,
            };
        let t = tid as usize;
        if t >= idf_tab.len() || t + 1 >= e_indptr.len() {
            return None;
        }
        let idf = idf_tab[t];
        if idf <= 0.0 {
            return None;
        }
        let lo = e_indptr[t] as usize;
        let hi = e_indptr[t + 1] as usize;
        if lo == hi {
            return None;
        }
        Some((&e_indices[lo..hi], e_data.slice(lo, hi), idf))
    }

    #[inline]
    fn term_postings_exact(&self, tid: u32) -> Option<(&'static [u32], DataView, f32)> {
        self.term_postings_exact_idf(tid, self.idf)
    }

    /// BM25 score with optional excluded terms. Documents containing any
    /// excluded term in the exact (un-expanded) index are filtered out.
    /// Oversamples 5*k internally to absorb the filter cost.
    pub fn score_excluding(&self, included: &[u32], excluded: &[u32], k: usize) -> SearchResults {
        if excluded.is_empty() {
            return self.score(included, k);
        }
        let oversample = (k.saturating_mul(5)).max(50);
        let mut raw = self.score(included, oversample);
        raw.hits.retain(|h| {
            for &x in excluded {
                if self.doc_has_exact_term(h.doc_idx, x) {
                    return false;
                }
            }
            true
        });
        raw.hits.truncate(k);
        raw
    }

    /// BM25 score `query_tokens` against the expanded index, returning top-K
    /// `(doc_idx, score)` pairs sorted descending.
    pub fn score(&self, query_tokens: &[u32], k: usize) -> SearchResults {
        self.score_with(query_tokens, k, self.idf, self.meta.avgdl)
    }

    /// [`Index::score`] using an externally supplied idf table and average
    /// document length instead of the segment's own. The merged path passes
    /// collection-wide stats so every segment scores on the same footing
    /// (fixes per-segment idf/avgdl drift across a multi-segment index).
    pub fn score_with(
        &self,
        query_tokens: &[u32],
        k: usize,
        idf: &[f32],
        avgdl: f32,
    ) -> SearchResults {
        let t0 = Instant::now();
        let n = self.n_docs();
        let p = Bm25::with_avgdl(self, avgdl);
        let mut scores = vec![0.0f32; n];
        let mut touched: Vec<u32> = Vec::new();
        let mut matched_terms = 0u32;

        for &tid in query_tokens {
            if let Some((cols, vals, term_idf)) = self.term_postings_idf(tid, idf) {
                matched_terms += 1;
                accumulate(
                    cols,
                    vals,
                    self.doc_lens,
                    p,
                    term_idf,
                    &mut scores,
                    &mut touched,
                );
            }
        }

        let hits = top_hits(&scores, &mut touched, k.min(n));
        SearchResults {
            hits,
            matched_query_terms: matched_terms,
            elapsed_us: t0.elapsed().as_micros() as u64,
        }
    }

    /// Compute facet bucket counts (value → number of docs) for each named
    /// rank-attribute field. Counts every doc whose final BM25 score (same
    /// formula as `score()`) is strictly positive - i.e. the full matched
    /// set, not just the displayed top-K. Returns `Err(field_name)` if any
    /// named field wasn't built into the artifact.
    pub fn facet_counts(
        &self,
        query_tokens: &[u32],
        fields: &[String],
    ) -> Result<Vec<Vec<(f32, u32)>>, String> {
        // Up-front validate the fields and pull their slices.
        let resolved: Vec<&'static [f32]> = fields
            .iter()
            .map(|name| {
                self.rank_attrs
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| name.clone())
            })
            .collect::<Result<_, _>>()?;
        if resolved.is_empty() {
            return Ok(Vec::new());
        }

        // Same scoring pass shape as score(), but instead of a top-k we
        // bucket-count every touched (matched) doc per field.
        let n = self.n_docs();
        let p = Bm25::of(self);
        let mut scores = vec![0.0f32; n];
        let mut touched: Vec<u32> = Vec::new();
        for &tid in query_tokens {
            if let Some((cols, vals, idf)) = self.term_postings(tid) {
                accumulate(cols, vals, self.doc_lens, p, idf, &mut scores, &mut touched);
            }
        }

        let mut out: Vec<Vec<(f32, u32)>> = Vec::with_capacity(fields.len());
        for slice in &resolved {
            let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for &di in &touched {
                // Hash f32 by its bit pattern so NaN/-0 don't merge
                // unexpectedly. Values exposed to the client are still
                // the f32.
                let v = slice[di as usize];
                *counts.entry(v.to_bits()).or_insert(0) += 1;
            }
            let mut buckets: Vec<(f32, u32)> = counts
                .into_iter()
                .map(|(b, c)| (f32::from_bits(b), c))
                .collect();
            buckets.sort_by(|a, b| {
                b.1.cmp(&a.1)
                    .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            });
            out.push(buckets);
        }
        Ok(out)
    }

    /// Block-Max WAND. Same algorithm as `score_wand` but the per-cursor
    /// upper bound is read from the precomputed `block_max` sidecar (one
    /// f32 per 128 postings) instead of the global `idf * (k1+1+delta)`
    /// ceiling. Tighter bounds → bigger gallop skips → fewer dot products.
    /// Falls back to `score_wand` if the sidecar isn't present.
    pub fn score_block_max_wand(&self, query_tokens: &[u32], k: usize) -> SearchResults {
        let (bm_indptr, bm_data) = match (self.block_max_indptr, self.block_max) {
            (Some(a), Some(b)) => (a, b),
            _ => return self.score_wand(query_tokens, k),
        };
        let t0 = Instant::now();

        struct Cursor {
            indices: &'static [u32],
            data: DataView,
            block_max: &'static [f32],
            idf: f32,
            pos: usize,
            current_upper: f32,
        }
        let mut cursors: Vec<Cursor> = Vec::with_capacity(query_tokens.len());
        let mut matched_terms = 0u32;
        for &tid in query_tokens {
            let (cols, vals, idf) = match self.term_postings(tid) {
                Some(t) => t,
                None => continue,
            };
            let tu = tid as usize;
            let bm_lo = bm_indptr[tu] as usize;
            let bm_hi = bm_indptr[tu + 1] as usize;
            if bm_lo == bm_hi {
                // Posting list exists but no block max - possibly mismatched
                // artifact; fall back rather than risk silent wrong results.
                return self.score_wand(query_tokens, k);
            }
            let block_max_slice: &'static [f32] = &bm_data[bm_lo..bm_hi];
            let current_upper = block_max_slice[0];
            matched_terms += 1;
            cursors.push(Cursor {
                indices: cols,
                data: vals,
                block_max: block_max_slice,
                idf,
                pos: 0,
                current_upper,
            });
        }
        if cursors.is_empty() {
            return SearchResults {
                hits: Vec::new(),
                matched_query_terms: 0,
                elapsed_us: t0.elapsed().as_micros() as u64,
            };
        }

        let p = Bm25::of(self);

        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(k + 1);
        let mut threshold: f32 = 0.0;

        let cur_doc = |c: &Cursor| -> u32 { c.indices[c.pos] };

        /// Refresh `current_upper` from the cursor's new position.
        fn refresh_upper(c: &mut Cursor) {
            if c.pos >= c.indices.len() {
                c.current_upper = 0.0;
                return;
            }
            let blk = c.pos / BLOCK_MAX_BLOCK_SIZE;
            c.current_upper = c.block_max[blk.min(c.block_max.len() - 1)];
        }

        fn advance_to(c: &mut Cursor, target: u32) {
            if c.pos >= c.indices.len() {
                return;
            }
            if c.indices[c.pos] >= target {
                return;
            }
            let mut step = 1usize;
            let mut p = c.pos;
            while p + step < c.indices.len() && c.indices[p + step] < target {
                p += step;
                step *= 2;
            }
            let hi = (p + step + 1).min(c.indices.len());
            let off = match c.indices[p..hi].binary_search(&target) {
                Ok(i) => i,
                Err(i) => i,
            };
            c.pos = p + off;
            refresh_upper(c);
        }

        loop {
            cursors.retain(|c| c.pos < c.indices.len());
            if cursors.is_empty() {
                break;
            }
            cursors.sort_unstable_by_key(|c| cur_doc(c));

            let mut acc = 0.0f32;
            let mut pivot_idx: Option<usize> = None;
            for (i, c) in cursors.iter().enumerate() {
                acc += c.current_upper;
                if acc > threshold {
                    pivot_idx = Some(i);
                    break;
                }
            }
            let pivot_idx = match pivot_idx {
                Some(p) => p,
                None => break,
            };
            let pivot_doc = cur_doc(&cursors[pivot_idx]);

            if cur_doc(&cursors[0]) == pivot_doc {
                let dl = self.doc_lens[pivot_doc as usize];
                let mut score = 0.0f32;
                for c in cursors.iter() {
                    if cur_doc(c) != pivot_doc {
                        break;
                    }
                    score += c.idf * p.stf(c.data.get(c.pos), dl);
                }
                if score > 0.0 {
                    if heap.len() < k {
                        heap.push(Reverse(HeapEntry(score, pivot_doc)));
                        if heap.len() == k {
                            threshold = heap.peek().unwrap().0 .0;
                        }
                    } else if score > threshold {
                        heap.pop();
                        heap.push(Reverse(HeapEntry(score, pivot_doc)));
                        threshold = heap.peek().unwrap().0 .0;
                    }
                }
                for c in cursors.iter_mut() {
                    if c.pos < c.indices.len() && c.indices[c.pos] == pivot_doc {
                        c.pos += 1;
                        refresh_upper(c);
                    }
                }
            } else {
                for c in cursors[..pivot_idx].iter_mut() {
                    advance_to(c, pivot_doc);
                }
            }
        }

        let mut hits: Vec<Hit> = heap
            .into_iter()
            .map(|Reverse(HeapEntry(s, d))| Hit {
                doc_idx: d,
                score: s,
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        SearchResults {
            hits,
            matched_query_terms: matched_terms,
            elapsed_us: t0.elapsed().as_micros() as u64,
        }
    }

    /// WAND-pruned BM25 scorer. Same result as `score()` modulo float
    /// associativity; skips posting-list entries that can't possibly enter
    /// the top-k by using a per-term upper-bound on contribution. Best
    /// returns on long posting lists with selective query terms - i.e.
    /// large-corpus deployments.
    pub fn score_wand(&self, query_tokens: &[u32], k: usize) -> SearchResults {
        let t0 = Instant::now();
        let p = Bm25::of(self);

        // Per-query-term cursor. `pos` is the index into the term's posting
        // slice; `upper` is an upper bound on the BM25 contribution any doc
        // can receive from this term (stf saturates at k1+1, so the bound is
        // idf * (k1+1 + delta)).
        struct Cursor {
            idf: f32,
            upper: f32,
            indices: &'static [u32],
            data: DataView,
            pos: usize,
        }
        let mut cursors: Vec<Cursor> = Vec::with_capacity(query_tokens.len());
        let mut matched_terms = 0u32;
        for &tid in query_tokens {
            let (cols, vals, idf) = match self.term_postings(tid) {
                Some(t) => t,
                None => continue,
            };
            matched_terms += 1;
            cursors.push(Cursor {
                idf,
                upper: idf * (p.k1 + 1.0 + p.delta),
                indices: cols,
                data: vals,
                pos: 0,
            });
        }
        if cursors.is_empty() {
            return SearchResults {
                hits: Vec::new(),
                matched_query_terms: 0,
                elapsed_us: t0.elapsed().as_micros() as u64,
            };
        }

        // Top-k as a min-heap keyed on score so the cheapest survivor is at
        // the root. When the heap is full and the cumulative upper bound for
        // a candidate doc is below the heap root, we can safely skip.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(k + 1);
        let mut threshold: f32 = 0.0;

        let cur_doc = |c: &Cursor| -> u32 { c.indices[c.pos] };

        // Advance `c` to the first posting whose doc_id ≥ `target`. Galloping
        // probe + binary refinement, which beats linear advance once the gap
        // exceeds ~6 entries.
        fn advance_to(c: &mut Cursor, target: u32) {
            if c.pos >= c.indices.len() {
                return;
            }
            if c.indices[c.pos] >= target {
                return;
            }
            let mut step = 1usize;
            let mut p = c.pos;
            while p + step < c.indices.len() && c.indices[p + step] < target {
                p += step;
                step *= 2;
            }
            // Binary search in [p, min(p+step, len)] for the first index whose
            // value is >= target.
            let hi = (p + step + 1).min(c.indices.len());
            let slice = &c.indices[p..hi];
            let off = match slice.binary_search(&target) {
                Ok(i) => i,
                Err(i) => i,
            };
            c.pos = p + off;
        }

        fn cur_doc_or_max(c: &Cursor) -> u32 {
            if c.pos < c.indices.len() {
                c.indices[c.pos]
            } else {
                u32::MAX
            }
        }

        loop {
            // Drop exhausted cursors; sort by current doc so cursors[0] is
            // the candidate (smallest doc) and we can scan a prefix.
            cursors.retain(|c| c.pos < c.indices.len());
            if cursors.is_empty() {
                break;
            }
            cursors.sort_unstable_by_key(|c| cur_doc(c));

            // Find the pivot: smallest prefix index whose cumulative upper
            // bound can match the threshold. No prefix → done.
            let mut acc = 0.0f32;
            let mut pivot_idx: Option<usize> = None;
            for (i, c) in cursors.iter().enumerate() {
                acc += c.upper;
                if acc > threshold {
                    pivot_idx = Some(i);
                    break;
                }
            }
            let pivot_idx = match pivot_idx {
                Some(p) => p,
                None => break,
            };
            let pivot_doc = cur_doc(&cursors[pivot_idx]);

            if cur_doc(&cursors[0]) == pivot_doc {
                // Score `pivot_doc`. Walk cursors in order; only the prefix
                // currently at `pivot_doc` contributes.
                let dl = self.doc_lens[pivot_doc as usize];
                let mut score = 0.0f32;
                for c in cursors.iter() {
                    if cur_doc(c) != pivot_doc {
                        break;
                    }
                    score += c.idf * p.stf(c.data.get(c.pos), dl);
                }
                if score > 0.0 {
                    if heap.len() < k {
                        heap.push(Reverse(HeapEntry(score, pivot_doc)));
                        if heap.len() == k {
                            threshold = heap.peek().unwrap().0 .0;
                        }
                    } else if score > threshold {
                        heap.pop();
                        heap.push(Reverse(HeapEntry(score, pivot_doc)));
                        threshold = heap.peek().unwrap().0 .0;
                    }
                }
                // Advance every cursor currently at pivot_doc.
                for c in cursors.iter_mut() {
                    if c.pos < c.indices.len() && c.indices[c.pos] == pivot_doc {
                        c.pos += 1;
                    } else if cur_doc_or_max(c) > pivot_doc {
                        break;
                    }
                }
            } else {
                // Advance prefix terms (before pivot) up to pivot_doc.
                for c in cursors[..pivot_idx].iter_mut() {
                    advance_to(c, pivot_doc);
                }
            }
        }

        // Drain heap into a descending-by-score Vec.
        let mut hits: Vec<Hit> = heap
            .into_iter()
            .map(|Reverse(HeapEntry(s, d))| Hit {
                doc_idx: d,
                score: s,
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        SearchResults {
            hits,
            matched_query_terms: matched_terms,
            elapsed_us: t0.elapsed().as_micros() as u64,
        }
    }

    /// Score with OR groups. Each group's contribution to a doc is
    /// `max(idf * stf)` over its tokens (instead of the regular sum), so
    /// `covid|covid19 vaccine` doesn't double-count docs that mention both
    /// covid spellings. Singleton groups (no `|`) behave identically to
    /// `score()`. Returns the same `SearchResults` shape.
    pub fn score_with_groups(&self, groups: &[Vec<u32>], k: usize) -> SearchResults {
        let t0 = Instant::now();
        let n = self.n_docs();
        let p = Bm25::of(self);
        let mut scores = vec![0.0f32; n];
        let mut touched: Vec<u32> = Vec::new();
        // Reused per-group max accumulator; only the entries this group
        // touched are reset between groups.
        let mut group_score = vec![0.0f32; n];
        let mut group_touched: Vec<u32> = Vec::new();
        let mut matched_terms = 0u32;

        for group in groups {
            let mut group_had_match = false;
            for &tid in group {
                if let Some((cols, vals, idf)) = self.term_postings(tid) {
                    group_had_match = true;
                    match vals {
                        DataView::F32(v) => max_into(
                            cols,
                            v,
                            self.doc_lens,
                            p,
                            idf,
                            &mut group_score,
                            &mut group_touched,
                        ),
                        DataView::F16(v) => max_into(
                            cols,
                            v,
                            self.doc_lens,
                            p,
                            idf,
                            &mut group_score,
                            &mut group_touched,
                        ),
                    }
                }
            }
            if group_had_match {
                matched_terms += 1;
            }
            // Fold this group's max contributions into the running sum and
            // zero the group buffer for the next group.
            for &di in &group_touched {
                let d = di as usize;
                let s = &mut scores[d];
                let old = *s;
                *s = old + group_score[d];
                if old == 0.0 && *s > 0.0 {
                    touched.push(di);
                }
                group_score[d] = 0.0;
            }
            group_touched.clear();
        }

        let hits = top_hits(&scores, &mut touched, k.min(n));
        SearchResults {
            hits,
            matched_query_terms: matched_terms,
            elapsed_us: t0.elapsed().as_micros() as u64,
        }
    }

    /// Like `score()`, but also computes per-doc feature breakdown for the
    /// learned reranker. Requires the optional exact CSR to have been
    /// written at build time. Returns `None` if the exact CSR isn't loaded.
    pub fn score_with_features(&self, query_tokens: &[u32], k: usize) -> Option<FeatureResults> {
        self.exact_indptr?;
        let t0 = Instant::now();
        let n = self.n_docs();
        let p = Bm25::of(self);

        let mut combined = vec![0.0f32; n];
        let mut exact = vec![0.0f32; n];
        // The exact postings of a term are a subset of its expanded row
        // (expansion only adds neighbour contributions), so the combined
        // pass's touched list covers the exact pass too.
        let mut touched: Vec<u32> = Vec::new();
        let mut exact_touched: Vec<u32> = Vec::new();
        let mut matched = 0u32;

        for &tid in query_tokens {
            if let Some((cols, vals, idf)) = self.term_postings(tid) {
                matched += 1;
                accumulate(
                    cols,
                    vals,
                    self.doc_lens,
                    p,
                    idf,
                    &mut combined,
                    &mut touched,
                );
            }
            if let Some((cols, vals, idf)) = self.term_postings_exact(tid) {
                accumulate(
                    cols,
                    vals,
                    self.doc_lens,
                    p,
                    idf,
                    &mut exact,
                    &mut exact_touched,
                );
            }
        }

        // top-K by combined score
        let hits_base = top_hits(&combined, &mut touched, k.min(n));

        let unique_q = query_tokens.len().max(1) as f32;
        let coverage = (matched as f32) / unique_q;

        let hits = hits_base
            .into_iter()
            .map(|h| {
                let c = h.score;
                let e = exact[h.doc_idx as usize].max(0.0);
                FeatureHit {
                    doc_idx: h.doc_idx,
                    bm25_combined: c,
                    bm25_exact: e,
                    bm25_semantic: (c - e).max(0.0),
                    coverage,
                    doc_len: self.doc_lens[h.doc_idx as usize],
                }
            })
            .collect();

        Some(FeatureResults {
            hits,
            matched_query_terms: matched,
            elapsed_us: t0.elapsed().as_micros() as u64,
        })
    }
}
