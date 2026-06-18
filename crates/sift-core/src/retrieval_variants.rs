//! Advanced ranking strategies layered on the core sparse scorer.

//! Optional retrieval and ranking modes built on the core sparse scorer.
//!
//! This module owns semantic score blending, custom ranking, MMR,
//! pseudo-relevance feedback, and score explanations.

use crate::retrieval::{accumulate, top_hits, Bm25};
use crate::{ExplainHit, ExplainResults, Hit, Index, SearchResults, TermContribution};
use std::time::Instant;

impl Index {
    /// Blended exact+expansion scorer. Ranks docs by
    /// `alpha * combined_bm25 + (1 - alpha) * exact_bm25` in one pass.
    /// `alpha = 1` is fully expanded and `alpha = 0` is exact BM25.
    pub fn score_blended(&self, query_tokens: &[u32], k: usize, alpha: f32) -> SearchResults {
        self.score_blended_qexp(query_tokens, k, alpha, &[])
    }

    /// [`Index::score_blended`] plus weighted query-side expansion terms.
    /// Each `(term, weight)` in `qexp` contributes `weight * idf * stf`
    /// against the EXACT (un-expanded) index, so expansion never compounds
    /// two hops (a query neighbor matching through its own doc-side
    /// expansion row). Falls back to `score()` (ignoring `qexp`) if the
    /// exact CSR isn't present.
    pub fn score_blended_qexp(
        &self,
        query_tokens: &[u32],
        k: usize,
        alpha: f32,
        qexp: &[(u32, f32)],
    ) -> SearchResults {
        self.score_blended_qexp_with(query_tokens, k, alpha, qexp, self.idf, self.meta.avgdl)
    }

    /// [`Index::score_blended_qexp`] with an externally supplied idf table and
    /// avgdl. The merged path passes collection-wide stats so segments score
    /// consistently.
    #[allow(clippy::too_many_arguments)]
    pub fn score_blended_qexp_with(
        &self,
        query_tokens: &[u32],
        k: usize,
        alpha: f32,
        qexp: &[(u32, f32)],
        idf: &[f32],
        avgdl: f32,
    ) -> SearchResults {
        if self.exact_indptr.is_none() || self.exact_data.is_none() {
            return self.score_with(query_tokens, k, idf, avgdl);
        }
        let t0 = Instant::now();
        let n = self.n_docs();
        let p = Bm25::with_avgdl(self, avgdl);
        let alpha = alpha.clamp(0.0, 1.0);

        let mut scores = vec![0.0f32; n];
        let mut touched: Vec<u32> = Vec::new();
        let mut matched = 0u32;
        for &tid in query_tokens {
            let expanded = self.term_postings_idf(tid, idf);
            if expanded.is_some() {
                matched += 1;
            }
            if alpha > 0.0 {
                if let Some((cols, vals, term_idf)) = expanded {
                    accumulate(
                        cols,
                        vals,
                        self.doc_lens,
                        p,
                        alpha * term_idf,
                        &mut scores,
                        &mut touched,
                    );
                }
            }
            if alpha < 1.0 {
                if let Some((cols, vals, term_idf)) = self.term_postings_exact_idf(tid, idf) {
                    accumulate(
                        cols,
                        vals,
                        self.doc_lens,
                        p,
                        (1.0 - alpha) * term_idf,
                        &mut scores,
                        &mut touched,
                    );
                }
            }
        }
        for &(tid, weight) in qexp {
            if weight <= 0.0 {
                continue;
            }
            if let Some((cols, vals, term_idf)) = self.term_postings_exact_idf(tid, idf) {
                accumulate(
                    cols,
                    vals,
                    self.doc_lens,
                    p,
                    weight * term_idf,
                    &mut scores,
                    &mut touched,
                );
            }
        }

        let hits = top_hits(&scores, &mut touched, k.min(n));
        SearchResults {
            hits,
            matched_query_terms: matched,
            elapsed_us: t0.elapsed().as_micros() as u64,
        }
    }

    /// Run plain BM25, then re-sort hits inside each BM25 band (width =
    /// `band_frac * top_bm25`) by the user-supplied ranking-attribute tuple.
    /// Each entry of `tiers` is `(field_name, ascending_flag)`:
    /// `ascending_flag = false` means higher value wins (e.g. popularity);
    /// `true` means lower wins (e.g. price). Returns an error string if any
    /// named field wasn't materialized.
    pub fn score_with_rank(
        &self,
        query_tokens: &[u32],
        k: usize,
        tiers: &[(String, bool)],
        band_frac: f32,
    ) -> Result<SearchResults, String> {
        // Validate tier fields up front.
        let resolved: Vec<&'static [f32]> = tiers
            .iter()
            .map(|(name, _)| {
                self.rank_attrs
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| format!("unknown rank field '{name}'"))
            })
            .collect::<Result<_, _>>()?;
        if tiers.is_empty() {
            return Ok(self.score(query_tokens, k));
        }

        let oversample = (k * 5).max(k + 50).min(self.n_docs());
        let mut base = self.score(query_tokens, oversample);
        if base.hits.is_empty() {
            return Ok(base);
        }

        let band_frac = band_frac.clamp(0.0, 1.0);
        let top = base
            .hits
            .iter()
            .map(|h| h.score)
            .fold(0.0f32, f32::max)
            .max(1e-9);
        let band = top * band_frac;

        // Bucket #0 holds the top score plus every doc within `band` of it;
        // bucket #1 holds docs in [top-2*band, top-band); and so on. Docs in
        // the same bucket are then ordered by the user's tier tuple, with
        // BM25 as the final tie-break.
        base.hits.sort_by(|a, b| {
            let bucket = |s: f32| {
                if band > 0.0 {
                    ((top - s) / band) as i32
                } else {
                    0
                }
            };
            let mut ord = bucket(a.score).cmp(&bucket(b.score));
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
            for (i, (_, ascending)) in tiers.iter().enumerate() {
                let va = resolved[i][a.doc_idx as usize];
                let vb = resolved[i][b.doc_idx as usize];
                ord = if *ascending {
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        base.hits.truncate(k);
        Ok(base)
    }

    /// MMR diversity rerank. Oversamples 3×k internally, then greedily
    /// rebuilds the top-k by lambda * rel - (1-lambda) * max-cosine to any
    /// previously selected doc, where the doc representation is the per-doc
    /// query-term contribution vector (i.e. how much each query term scored
    /// this doc). lambda=1.0 reduces to the standard top-k.
    pub fn score_with_mmr(&self, query_tokens: &[u32], k: usize, lambda: f32) -> SearchResults {
        let t0 = Instant::now();
        let lambda = lambda.clamp(0.0, 1.0);
        let oversample = (k * 3).max(k + 20).min(self.n_docs());
        let r = self.score(query_tokens, oversample);
        if r.hits.is_empty() || lambda >= 0.9999 {
            let mut out = r.clone();
            out.hits.truncate(k);
            return out;
        }

        let p = Bm25::of(self);

        let n_cand = r.hits.len();
        let n_terms = query_tokens.len();
        let mut vecs: Vec<Vec<f32>> = vec![vec![0.0; n_terms]; n_cand];
        let mut norms: Vec<f32> = vec![0.0; n_cand];
        let mut rel: Vec<f32> = Vec::with_capacity(n_cand);
        let mut doc_idxs: Vec<u32> = Vec::with_capacity(n_cand);
        for (i, h) in r.hits.iter().enumerate() {
            rel.push(h.score);
            doc_idxs.push(h.doc_idx);
            let di = h.doc_idx;
            let dl = self.doc_lens[di as usize];
            for (j, &tid) in query_tokens.iter().enumerate() {
                let (cols, vals, idf) = match self.term_postings(tid) {
                    Some(t) => t,
                    None => continue,
                };
                if let Ok(pos) = cols.binary_search(&di) {
                    let c = idf * p.stf(vals.get(pos), dl);
                    vecs[i][j] = c;
                    norms[i] += c * c;
                }
            }
            norms[i] = norms[i].sqrt().max(1e-9);
        }
        // Normalize relevance to [0,1] so it composes with cosine.
        let rel_max = rel.iter().copied().fold(0.0f32, f32::max).max(1e-9);
        let rel_norm: Vec<f32> = rel.iter().map(|&x| x / rel_max).collect();

        let mut selected: Vec<usize> = Vec::with_capacity(k);
        let mut max_sim: Vec<f32> = vec![0.0; n_cand];
        let mut alive: Vec<bool> = vec![true; n_cand];
        let k_target = k.min(n_cand);

        // First pick: highest relevance.
        if let Some((i0, _)) = rel_norm
            .iter()
            .enumerate()
            .max_by(|a, c| a.1.partial_cmp(c.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            selected.push(i0);
            alive[i0] = false;
            // update max_sim relative to i0
            for j in 0..n_cand {
                if !alive[j] {
                    continue;
                }
                let dot: f32 = vecs[i0].iter().zip(&vecs[j]).map(|(a, b)| a * b).sum();
                let s = dot / (norms[i0] * norms[j]);
                if s > max_sim[j] {
                    max_sim[j] = s;
                }
            }
        }

        while selected.len() < k_target {
            let mut best_i: Option<usize> = None;
            let mut best_score = f32::NEG_INFINITY;
            for j in 0..n_cand {
                if !alive[j] {
                    continue;
                }
                let mmr = lambda * rel_norm[j] - (1.0 - lambda) * max_sim[j];
                if mmr > best_score {
                    best_score = mmr;
                    best_i = Some(j);
                }
            }
            let pick = match best_i {
                Some(p) => p,
                None => break,
            };
            selected.push(pick);
            alive[pick] = false;
            for j in 0..n_cand {
                if !alive[j] {
                    continue;
                }
                let dot: f32 = vecs[pick].iter().zip(&vecs[j]).map(|(a, b)| a * b).sum();
                let s = dot / (norms[pick] * norms[j]);
                if s > max_sim[j] {
                    max_sim[j] = s;
                }
            }
        }

        let hits: Vec<Hit> = selected
            .into_iter()
            .map(|i| Hit {
                doc_idx: doc_idxs[i],
                score: rel[i],
            })
            .collect();

        SearchResults {
            hits,
            matched_query_terms: r.matched_query_terms,
            elapsed_us: t0.elapsed().as_micros() as u64,
        }
    }

    /// PRF / Rocchio-style query expansion. Requires the forward index
    /// (--forward at build). Procedure:
    ///   1. Initial retrieval: top `prf_k` docs by `score()`.
    ///   2. Collect candidate expansion terms by walking the forward index
    ///      of each feedback doc. Weight each candidate by
    ///      `sum_d idf[t] * tf_d(t) / dl_d`
    ///      (Rocchio with IDF reweighting and length normalization).
    ///   3. Drop terms already in the query; pick top `prf_t` by weight.
    ///   4. Re-retrieve combining original tokens (weight 1.0) with the
    ///      new tokens (weight `alpha`) using a single weighted-sum BM25
    ///      pass. Returns the final top-`k`.
    ///
    /// Falls back to plain `score()` if the forward index is missing.
    pub fn score_with_prf(
        &self,
        query_tokens: &[u32],
        k: usize,
        prf_k: usize,
        prf_t: usize,
        alpha: f32,
    ) -> SearchResults {
        let t0 = Instant::now();
        let (fwd_indptr, fwd_indices, fwd_data) =
            match (self.fwd_indptr, self.fwd_indices, self.fwd_data) {
                (Some(a), Some(b), Some(c)) => (a, b, c),
                _ => return self.score(query_tokens, k),
            };

        let initial = self.score(query_tokens, prf_k.max(k));
        let feedback: Vec<u32> = initial.hits.iter().take(prf_k).map(|h| h.doc_idx).collect();
        if feedback.is_empty() {
            return SearchResults {
                hits: initial.hits.into_iter().take(k).collect(),
                matched_query_terms: initial.matched_query_terms,
                elapsed_us: t0.elapsed().as_micros() as u64,
            };
        }

        let q_set: std::collections::HashSet<u32> = query_tokens.iter().copied().collect();
        let mut term_weights: std::collections::HashMap<u32, f32> =
            std::collections::HashMap::new();
        for &d in &feedback {
            let du = d as usize;
            if du + 1 >= fwd_indptr.len() {
                continue;
            }
            let lo = fwd_indptr[du] as usize;
            let hi = fwd_indptr[du + 1] as usize;
            let dl = self.doc_lens[du].max(1.0);
            for j in lo..hi {
                let tid = fwd_indices[j];
                if q_set.contains(&tid) {
                    continue;
                }
                let tf = fwd_data[j];
                let tu = tid as usize;
                if tu >= self.idf.len() {
                    continue;
                }
                let idf = self.idf[tu];
                if idf <= 0.0 {
                    continue;
                }
                let w = idf * tf / dl;
                *term_weights.entry(tid).or_insert(0.0) += w;
            }
        }
        let mut ranked: Vec<(u32, f32)> = term_weights.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(prf_t);

        if ranked.is_empty() || alpha <= 0.0 {
            return SearchResults {
                hits: initial.hits.into_iter().take(k).collect(),
                matched_query_terms: initial.matched_query_terms,
                elapsed_us: t0.elapsed().as_micros() as u64,
            };
        }

        // Build weighted (token, weight) list and re-score with one pass.
        let mut tw: Vec<(u32, f32)> = query_tokens.iter().map(|&t| (t, 1.0)).collect();
        // Normalize expansion weights so the largest equals alpha. Keeps
        // alpha interpretable as a hard "expansion contribution ceiling".
        let max_w = ranked
            .iter()
            .map(|(_, w)| *w)
            .fold(0.0f32, f32::max)
            .max(1e-9);
        for (t, w) in ranked {
            tw.push((t, alpha * w / max_w));
        }

        let n = self.n_docs();
        let p = Bm25::of(self);
        let mut scores = vec![0.0f32; n];
        let mut touched: Vec<u32> = Vec::new();
        let mut matched = 0u32;
        for (tid, weight) in &tw {
            if let Some((cols, vals, idf)) = self.term_postings(*tid) {
                matched += 1;
                accumulate(
                    cols,
                    vals,
                    self.doc_lens,
                    p,
                    *weight * idf,
                    &mut scores,
                    &mut touched,
                );
            }
        }

        let hits = top_hits(&scores, &mut touched, k.min(n));
        SearchResults {
            hits,
            matched_query_terms: matched,
            elapsed_us: t0.elapsed().as_micros() as u64,
        }
    }

    /// Score like `score()`, but record per-term contributions for each
    /// top-K hit. Walks postings twice (once for scoring, once for the
    /// focused per-hit pull) so this is ~2× slower than `score()`. Intended
    /// for the `/explain` debugging endpoint, not the hot search path.
    pub fn score_with_explain(&self, query_tokens: &[u32], k: usize) -> ExplainResults {
        let t0 = Instant::now();
        let r = self.score(query_tokens, k);
        let p = Bm25::of(self);

        let mut hits = Vec::with_capacity(r.hits.len());
        for h in &r.hits {
            let di = h.doc_idx;
            let dl = self.doc_lens[di as usize];
            let mut terms = Vec::with_capacity(query_tokens.len());
            for &tid in query_tokens {
                let (cols, vals, idf) = match self.term_postings(tid) {
                    Some(t) => t,
                    None => continue,
                };
                if let Ok(pos) = cols.binary_search(&di) {
                    let wtf = vals.get(pos);
                    let contribution = idf * p.stf(wtf, dl);
                    if contribution > 0.0 {
                        terms.push(TermContribution {
                            tid,
                            token: self.token_string(tid).unwrap_or_default(),
                            idf,
                            tf_in_doc: wtf,
                            contribution,
                        });
                    }
                }
            }
            terms.sort_by(|a, c| {
                c.contribution
                    .partial_cmp(&a.contribution)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            hits.push(ExplainHit {
                doc_idx: di,
                score: h.score,
                doc_len: dl,
                terms,
            });
        }

        ExplainResults {
            hits,
            matched_query_terms: r.matched_query_terms,
            elapsed_us: t0.elapsed().as_micros() as u64,
        }
    }
}
