//! Read-side handlers: /search, /explain, /suggest, /failures, /datasets,
//! plus the query-string helpers they share (OR groups, quoted phrases,
//! snippet highlighting).

use super::filters::{passes_filters, FilterClause};
use super::{resolve_alias, AppState, CachedSearch, IndexEntry, SlowEntry};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Json as RespJson, Response},
};
use serde::{Deserialize, Serialize};
use sift_core::{Index, ScoreMode};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;
use std::time::Instant;

include!("query/request.rs");

/// Escape a plain snippet before adding trusted `<mark>` tags.
fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Parse a query for `|`-OR groups. Each whitespace-separated word can be
/// split on `|` to form an OR group; words without `|` become singleton
/// groups. Group tokens are tokenized via `tokenize_query` (same path the
/// regular AND chain uses) so phrase content + casing + normalization
/// match the index. Returns `(groups, flat_tokens, has_or)` so the caller
/// can drop into the right scoring path.
fn parse_or_groups(idx: &Index, q: &str) -> (Vec<Vec<u32>>, Vec<u32>, bool) {
    let mut groups: Vec<Vec<u32>> = Vec::new();
    let mut flat: Vec<u32> = Vec::new();
    let mut has_or = false;
    for word in q.split_whitespace() {
        if word.contains('|') {
            let parts: Vec<&str> = word.split('|').filter(|s| !s.is_empty()).collect();
            if parts.len() < 2 {
                // `foo|` or `|foo` - degrade to a singleton.
                let toks = idx.tokenize_query(word);
                for t in &toks {
                    flat.push(*t);
                    groups.push(vec![*t]);
                }
                continue;
            }
            has_or = true;
            let mut group: Vec<u32> = Vec::new();
            for p in parts {
                let toks = idx.tokenize_query(p);
                group.extend_from_slice(&toks);
                flat.extend_from_slice(&toks);
            }
            group.sort_unstable();
            group.dedup();
            if !group.is_empty() {
                groups.push(group);
            }
        } else {
            let toks = idx.tokenize_query(word);
            for t in &toks {
                flat.push(*t);
                groups.push(vec![*t]);
            }
        }
    }
    flat.sort_unstable();
    flat.dedup();
    (groups, flat, has_or)
}

/// Extract `"…"` runs from a query. Each returned string is the literal
/// content between a pair of double quotes (in order); unclosed quotes are
/// ignored. The caller is responsible for tokenizing each phrase the same
/// way the indexed corpus was.
fn extract_phrases(q: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = q.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            let mut buf = String::new();
            let mut closed = false;
            for c2 in chars.by_ref() {
                if c2 == '"' {
                    closed = true;
                    break;
                }
                buf.push(c2);
            }
            if closed && !buf.trim().is_empty() {
                out.push(buf);
            }
        }
    }
    out
}

/// Split the query on whitespace and punctuation, lowercase each fragment,
/// drop empties and any leading `-` (treated as an excluded term elsewhere).
fn highlight_words(q: &str) -> Vec<String> {
    q.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .map(|w| w.trim_start_matches('-'))
        .filter(|w| !w.is_empty() && w.chars().any(|c| c.is_alphanumeric()))
        .map(|w| w.to_lowercase())
        .collect()
}

/// Wrap case-insensitive matches of `words` in the snippet with
/// `<mark>…</mark>` over an HTML-escaped base. Greedy longest-prefix match at
/// each position so overlapping query words don't produce nested marks.
fn highlight_snippet(snippet: &str, words: &[String]) -> String {
    if words.is_empty() {
        return html_escape(snippet);
    }
    // Pre-lowercase the snippet once for case-insensitive position matching.
    let lower: Vec<char> = snippet.to_lowercase().chars().collect();
    let original: Vec<char> = snippet.chars().collect();
    // Word-lengths in chars (snippet was lowercased the same way).
    let lc_words: Vec<Vec<char>> = words.iter().map(|w| w.chars().collect()).collect();

    let mut out = String::with_capacity(snippet.len() + 32);
    let mut i = 0;
    while i < original.len() {
        // Only consider a match starting at a non-alphanumeric boundary so
        // we don't highlight `cat` inside `category`.
        let on_boundary = i == 0 || !original[i - 1].is_alphanumeric();
        let mut best: Option<usize> = None;
        if on_boundary {
            for w in &lc_words {
                if w.len() > lower.len() - i {
                    continue;
                }
                if lower[i..i + w.len()] == w[..] {
                    let after = i + w.len();
                    let off_boundary =
                        after == original.len() || !original[after].is_alphanumeric();
                    if off_boundary && best.map_or(true, |b| w.len() > b) {
                        best = Some(w.len());
                    }
                }
            }
        }
        if let Some(len) = best {
            let slice: String = original[i..i + len].iter().collect();
            out.push_str("<mark>");
            out.push_str(&html_escape(&slice));
            out.push_str("</mark>");
            i += len;
        } else {
            out.push_str(&html_escape(&original[i].to_string()));
            i += 1;
        }
    }
    out
}

#[derive(Serialize, Clone)]
pub(crate) struct SearchHit {
    doc_id: String,
    score: f32,
    snippet: String,
    /// Internal doc index. Skipped from serialization (it's an opaque
    /// build-time id) but kept so the phrase filter can hit the positional
    /// index without re-resolving by string.
    #[serde(skip)]
    doc_idx: u32,
    /// The stored source document, when the artifact has payloads and the
    /// request didn't opt out via `"with_payload": false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<serde_json::Value>,
    /// HTML-escaped snippet with case-insensitive matches of each query word
    /// wrapped in `<mark>…</mark>`. Present only when the caller set
    /// `"highlight": true`. The plain `snippet` is always included alongside.
    #[serde(skip_serializing_if = "Option::is_none")]
    snippet_html: Option<String>,
    /// Per-doc feature breakdown for the optional reranker. Present only when
    /// the artifact has the exact-CSR sidecar written (rebuild required for
    /// older artifacts).
    #[serde(skip_serializing_if = "Option::is_none")]
    features: Option<Features>,
}

#[derive(Serialize, Clone)]
struct Features {
    bm25_combined: f32,
    bm25_exact: f32,
    bm25_semantic: f32,
    coverage: f32,
    doc_len: f32,
}

#[derive(Serialize, Clone)]
struct FacetBucket {
    value: f32,
    count: u32,
}

#[derive(Serialize)]
struct SearchResp {
    index: String,
    matched_terms: u32,
    /// Number of results that matched the query and passed all filters, before
    /// pagination. Exact for typical result sizes; for very large result sets
    /// it is capped at the internal candidate window.
    total: usize,
    latency_us: u64,
    hits: Vec<SearchHit>,
    /// Present only when the caller passed `"spell": true` and the artifact
    /// has a spell sidecar. Echoes the query as it was after correction so
    /// the client can render "showing results for …" UX.
    #[serde(skip_serializing_if = "Option::is_none")]
    spell_corrected: Option<String>,
    /// Per-field bucket counts over every doc that scored > 0, populated
    /// when the request set `facets`. Empty when no facets were requested.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    facets: HashMap<String, Vec<FacetBucket>>,
}

/// Public dataset listing. Intentionally minimal: anything beyond
/// `name` and `n_docs` leaks implementation parameters.
#[derive(Serialize)]
struct DatasetInfo {
    name: String,
    n_docs: u64,
    /// Names of per-doc ranking-attribute fields available for "rank" in
    /// /search. Empty if the artifact wasn't built with --rank-fields.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    rank_fields: Vec<String>,
}

pub(crate) async fn list_datasets(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let map = state.indices.read().unwrap();
    let mut out: Vec<DatasetInfo> = map
        .iter()
        .map(|(name, e)| DatasetInfo {
            name: name.clone(),
            n_docs: e.set.n_docs_total() as u64,
            rank_fields: e
                .idx()
                .rank_field_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    RespJson(out)
}

pub(crate) async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchReq>,
) -> Result<Response, (StatusCode, String)> {
    validate_search_req(&req).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let idx_name = req.index.unwrap_or_else(|| state.default.clone());
    let idx_name = resolve_alias(&state, &idx_name);
    let entry: Arc<IndexEntry> = {
        let map = state.indices.read().unwrap();
        map.get(&idx_name).cloned()
    }
    .ok_or((StatusCode::NOT_FOUND, format!("unknown index '{idx_name}'")))?;

    // If the caller asked for spell correction and the artifact supports it,
    // rewrite the query string before parsing.
    let req_start = Instant::now();
    entry.queries_total.fetch_add(1, AtomicOrdering::Relaxed);

    let (effective_q, spell_corrected): (String, Option<String>) =
        if req.spell && entry.idx().has_spell() {
            let c = entry.idx().spell_correct_query(&req.q);
            let echoed = if c != req.q { Some(c.clone()) } else { None };
            (c, echoed)
        } else {
            (req.q.clone(), None)
        };

    // Multi-segment indices (built up with `sift add` / `sift delete`) are
    // served through the cross-segment merge scorer. The rich single-segment
    // features need one compacted segment; reject them with a clear pointer.
    if !entry.set.is_single() {
        if req.features
            || req.mmr
            || req.prf
            || req.dedup
            || !req.rank.is_empty()
            || !req.facets.is_empty()
        {
            return Err((
                StatusCode::CONFLICT,
                "index has multiple segments; facets, rank tiers, mmr, prf and dedup \
                 require a single segment - run `sift compact` first"
                    .to_string(),
            ));
        }
        let want = req.k.clamp(1, 200);
        let offset = req.offset.min(10_000);
        let mode = if req.wand {
            if entry.set.segments.iter().all(|s| s.has_block_max()) {
                ScoreMode::BlockMaxWand
            } else {
                ScoreMode::Wand
            }
        } else {
            ScoreMode::Plain
        };
        // Oversample for pagination and to keep the post-filter top-k full.
        let base = want + offset;
        let fetch = if req.filter.is_empty() {
            base
        } else {
            (base * 5).clamp(base, 4000)
        };
        let merged = entry
            .set
            .search_merged(&effective_q, fetch, mode, req.blend_alpha);
        let mut mhits = merged.hits;
        if !req.filter.is_empty() {
            let has_payload = entry.set.segments.iter().any(|s| s.has_payload());
            for f in &req.filter {
                let known = has_payload || entry.set.primary().rank_value(&f.field, 0).is_some();
                if !known {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("unknown filter field '{}'", f.field),
                    ));
                }
            }
            mhits.retain(|h| {
                let seg = &entry.set.segments[h.segment];
                passes_filters(seg.payload(h.doc_idx as usize), &req.filter, |field| {
                    seg.rank_value(field, h.doc_idx)
                })
            });
        }
        // Without filters, the merge reports the distinct match count; with
        // filters, the surviving candidate count is the total.
        let total = if req.filter.is_empty() {
            merged.total
        } else {
            mhits.len()
        };
        // Pagination: drop the head, then keep one page.
        if offset >= mhits.len() {
            mhits.clear();
        } else {
            mhits.drain(..offset);
            mhits.truncate(want);
        }
        let hi_words: Vec<String> = if req.highlight {
            highlight_words(&effective_q)
        } else {
            Vec::new()
        };
        let hits: Vec<SearchHit> = mhits
            .iter()
            .map(|h| {
                let seg = &entry.set.segments[h.segment];
                SearchHit {
                    doc_id: h.doc_id.clone(),
                    doc_idx: h.doc_idx,
                    score: h.score,
                    snippet: h.snippet.clone(),
                    payload: if req.with_payload {
                        seg.payload(h.doc_idx as usize)
                            .and_then(|s| serde_json::from_str(s).ok())
                    } else {
                        None
                    },
                    snippet_html: if req.highlight {
                        Some(highlight_snippet(&h.snippet, &hi_words))
                    } else {
                        None
                    },
                    features: None,
                }
            })
            .collect();
        {
            let mut lat = entry.latencies_us.lock().unwrap();
            if lat.len() >= 1024 {
                lat.remove(0);
            }
            lat.push(merged.elapsed_us);
        }
        let elapsed = req_start.elapsed().as_micros() as u64;
        let body = RespJson(SearchResp {
            index: idx_name,
            matched_terms: merged.matched_query_terms,
            total,
            latency_us: elapsed,
            hits,
            spell_corrected,
            facets: HashMap::new(),
        });
        let st = format!(
            "score;dur={:.3},total;dur={:.3}",
            merged.elapsed_us as f64 / 1000.0,
            elapsed as f64 / 1000.0
        );
        return Ok(([("server-timing", st)], body).into_response());
    }

    // Cache key over everything that can change the result.
    // Facets are deliberately not cached: CachedSearch stores the ranked page,
    // while facets describe the complete matched set and have a different
    // invalidation/size profile.
    let cache_key = if req.cache && req.facets.is_empty() {
        let mut h = ahash::AHasher::default();
        idx_name.hash(&mut h);
        effective_q.hash(&mut h);
        req.k.hash(&mut h);
        req.features.hash(&mut h);
        req.spell.hash(&mut h);
        req.mmr.hash(&mut h);
        ((req.mmr_lambda * 1e6) as i32).hash(&mut h);
        req.prf.hash(&mut h);
        req.prf_k.hash(&mut h);
        req.prf_t.hash(&mut h);
        ((req.prf_alpha * 1e6) as i32).hash(&mut h);
        for t in &req.rank {
            t.field.hash(&mut h);
            t.order.hash(&mut h);
        }
        ((req.rank_band * 1e6) as i32).hash(&mut h);
        req.highlight.hash(&mut h);
        req.phrase.hash(&mut h);
        ((req.blend_alpha * 1e6) as i32).hash(&mut h);
        ((req.bigram_weight * 1e6) as i32).hash(&mut h);
        ((req.proximity_weight * 1e6) as i32).hash(&mut h);
        ((req.qexp_weight * 1e6) as i32).hash(&mut h);
        req.dedup.hash(&mut h);
        req.wand.hash(&mut h);
        req.offset.hash(&mut h);
        req.with_payload.hash(&mut h);
        req.rerank.unwrap_or(true).hash(&mut h);
        state.reranker.is_some().hash(&mut h);
        state.cross_encoder.is_some().hash(&mut h);
        // Filters can now carry JSON values (eq/in); hash their serialized form.
        serde_json::to_string(&req.filter)
            .unwrap_or_default()
            .hash(&mut h);
        Some(h.finish())
    } else {
        None
    };
    if let Some(key) = cache_key {
        let mut c = entry.cache.lock().unwrap();
        if let Some(cached) = c.get(&key) {
            entry.cache_hits.fetch_add(1, AtomicOrdering::Relaxed);
            let elapsed = req_start.elapsed().as_micros() as u64;
            let body = RespJson(SearchResp {
                index: idx_name,
                matched_terms: cached.matched_terms,
                total: cached.total,
                latency_us: elapsed,
                hits: cached.hits.clone(),
                spell_corrected: cached.spell_corrected.clone(),
                facets: HashMap::new(),
            });
            let st = format!(
                "cache;dur={:.3},total;dur={:.3}",
                0.0_f64,
                elapsed as f64 / 1000.0
            );
            return Ok(([("server-timing", st)], body).into_response());
        }
    }

    let (tokens, excluded) = entry.idx().parse_query(&effective_q);
    if tokens.is_empty() {
        return Ok(RespJson(SearchResp {
            index: idx_name,
            matched_terms: 0,
            total: 0,
            latency_us: 0,
            hits: vec![],
            spell_corrected,
            facets: HashMap::new(),
        })
        .into_response());
    }
    // For bigram bonus we need the query in original order. Filter out the
    // excluded terms so a "-foo" doesn't accidentally make "foo bar" a bigram.
    let order_all = entry.idx().tokenize_query_keep_order(&effective_q);
    let exc_set: std::collections::HashSet<u32> = excluded.iter().copied().collect();
    let q_ordered: Vec<u32> = order_all
        .into_iter()
        .filter(|t| !exc_set.contains(t))
        .collect();

    let k = req.k.clamp(1, 200);
    // Pagination: oversample by offset, then trim the head before returning.
    // The cap mirrors the BM25 result depth most clients ever consume.
    let offset = req.offset.min(10_000);
    // Oversample the score pass so the bigram-bonus rerank that runs after
    // score() has room to promote docs that won the bigram contribution but
    // didn't make the raw-BM25 top-K. Without this, page N+1 can repeat a
    // doc that appeared at the bottom of page N (because score(k_small) and
    // score(k_large) return different prefixes once bigram bonus reshuffles).
    let inner_k = (k + offset).clamp(100, 10_000);
    let hi_words: Vec<String> = if req.highlight {
        highlight_words(&effective_q)
    } else {
        Vec::new()
    };
    let mk_snip_html = |s: &str| -> Option<String> {
        if req.highlight {
            Some(highlight_snippet(s, &hi_words))
        } else {
            None
        }
    };
    let mk_hit = |doc_idx: u32, score: f32| -> SearchHit {
        let idx = entry.idx();
        let snippet = idx.doc_snip(doc_idx as usize).to_string();
        SearchHit {
            doc_id: idx.doc_id(doc_idx as usize).to_string(),
            doc_idx,
            score,
            snippet_html: mk_snip_html(&snippet),
            snippet,
            payload: None,
            features: None,
        }
    };

    // A request shape that plain ranked search (and therefore reranking)
    // can serve: no exotic scoring mode that owns its own ordering.
    let plain_shape = req.rerank.unwrap_or(true)
        && excluded.is_empty()
        && !req.features
        && req.rank.is_empty()
        && !req.mmr
        && !req.prf
        && !effective_q.contains('|');

    // Query-side expansion: pull each query term's embedding neighbors from
    // the sidecar as weighted extra terms (scored against the exact index
    // inside score_blended_qexp). Terms already in the query are skipped;
    // duplicates keep their max weight.
    let qexp_terms: Vec<(u32, f32)> =
        if req.qexp_weight > 0.0 && excluded.is_empty() && entry.idx().has_qexp() {
            let mut acc: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
            let tok_set: std::collections::HashSet<u32> = tokens.iter().copied().collect();
            for &t in &tokens {
                if let Some((nbrs, sims)) = entry.idx().qexp_neighbors(t) {
                    for (&j, &sim) in nbrs.iter().zip(sims.iter()) {
                        if tok_set.contains(&j) {
                            continue;
                        }
                        let w = req.qexp_weight * sim;
                        let e = acc.entry(j).or_insert(0.0);
                        if w > *e {
                            *e = w;
                        }
                    }
                }
            }
            acc.into_iter().collect()
        } else {
            Vec::new()
        };

    // Cross-encoder rerank: retrieve a candidate window with the normal
    // blended scorer, then let the ONNX model rescore (query, doc) pairs
    // jointly. The window keeps base order below ce_depth.
    let ce_hits: Option<sift_core::SearchResults> = match (plain_shape, &state.cross_encoder) {
        (true, Some(ce)) => {
            let depth = (k + offset).max(state.ce_depth).min(10_000);
            let mut r =
                entry
                    .idx()
                    .score_blended_qexp(&tokens, depth, req.blend_alpha, &qexp_terms);
            let window = r.hits.len().min(state.ce_depth);
            let texts: Vec<String> = r.hits[..window]
                .iter()
                .map(|h| {
                    let idx = entry.idx();
                    let di = h.doc_idx as usize;
                    let t = idx
                        .payload(di)
                        .and_then(|p| serde_json::from_str::<serde_json::Value>(p).ok())
                        .and_then(|v| {
                            v.get("text")
                                .or_else(|| v.get("body"))
                                .and_then(|t| t.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| idx.doc_snip(di).to_string());
                    t.chars().take(1200).collect()
                })
                .collect();
            match ce.score_pairs(&effective_q, &texts) {
                Ok(scores) if scores.len() == window => {
                    let mut order: Vec<usize> = (0..window).collect();
                    order.sort_by(|&a, &b| {
                        scores[b]
                            .partial_cmp(&scores[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let reranked: Vec<sift_core::Hit> = order
                        .iter()
                        .map(|&i| sift_core::Hit {
                            doc_idx: r.hits[i].doc_idx,
                            score: scores[i],
                        })
                        .chain(r.hits[window..].iter().cloned())
                        .collect();
                    r.hits = reranked;
                    Some(r)
                }
                Ok(_) => Some(r), // model returned nothing (stub build); keep base order
                Err(e) => {
                    tracing::warn!("cross-encoder rerank failed: {e}; serving base order");
                    Some(r)
                }
            }
        }
        _ => None,
    };

    // GBDT rerank: when the server carries a model (and no cross-encoder
    // claimed the request), score top candidates via the feature path and
    // reorder by the model. Opt out per request with `"rerank": false`.
    let do_rerank = ce_hits.is_none() && state.reranker.is_some() && plain_shape;

    // None when reranking is off OR the artifact lacks the exact CSR the
    // feature pass needs (falls through to the normal branches below).
    let rerank_results = if do_rerank {
        // Rerank depth: enough for the page requested, at least 100.
        let depth = (k + offset).clamp(100, 10_000);
        entry.idx().score_with_features(&tokens, depth)
    } else {
        None
    };

    // If caller asked for features, route through the feature-aware path.
    // Excludes are applied as a post-filter regardless of feature mode.
    let (matched_terms, elapsed_us, hits) = if let Some(r) = ce_hits {
        let h: Vec<SearchHit> = r.hits.iter().map(|h| mk_hit(h.doc_idx, h.score)).collect();
        (r.matched_query_terms, r.elapsed_us, h)
    } else if let Some(fr) = rerank_results {
        let model = state.reranker.as_ref().unwrap();
        let mut scored: Vec<(f32, u32)> = fr
            .hits
            .iter()
            .enumerate()
            .map(|(rank, h)| {
                let f = super::rerank::feature_vector(
                    h.bm25_combined,
                    h.bm25_exact,
                    h.bm25_semantic,
                    h.coverage,
                    h.doc_len,
                    rank,
                );
                (model.score(&f), h.doc_idx)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let h: Vec<SearchHit> = scored.iter().map(|&(s, di)| mk_hit(di, s)).collect();
        (fr.matched_query_terms, fr.elapsed_us, h)
    } else if req.features && excluded.is_empty() {
        if let Some(fr) = entry.idx().score_with_features(&tokens, inner_k) {
            let h: Vec<SearchHit> = fr
                .hits
                .iter()
                .map(|h| {
                    let mut hit = mk_hit(h.doc_idx, h.bm25_combined);
                    hit.features = Some(Features {
                        bm25_combined: h.bm25_combined,
                        bm25_exact: h.bm25_exact,
                        bm25_semantic: h.bm25_semantic,
                        coverage: h.coverage,
                        doc_len: h.doc_len,
                    });
                    hit
                })
                .collect();
            (fr.matched_query_terms, fr.elapsed_us, h)
        } else {
            // artifact built before exact-CSR was added; fall through
            let r = entry.idx().score(&tokens, inner_k);
            let h: Vec<SearchHit> = r.hits.iter().map(|h| mk_hit(h.doc_idx, h.score)).collect();
            (r.matched_query_terms, r.elapsed_us, h)
        }
    } else if !req.rank.is_empty() && excluded.is_empty() {
        let tiers: Vec<(String, bool)> = req
            .rank
            .iter()
            .map(|t| {
                let ascending = matches!(t.order.as_str(), "asc" | "ascending" | "ASC");
                (t.field.clone(), ascending)
            })
            .collect();
        match entry
            .idx()
            .score_with_rank(&tokens, inner_k, &tiers, req.rank_band)
        {
            Ok(mut r) => {
                if q_ordered.len() >= 2 && req.bigram_weight > 0.0 {
                    for h in &mut r.hits {
                        h.score += entry.idx().bigram_bonus_for_doc(
                            h.doc_idx,
                            &q_ordered,
                            req.bigram_weight,
                        );
                    }
                }
                let h: Vec<SearchHit> = r.hits.iter().map(|h| mk_hit(h.doc_idx, h.score)).collect();
                (r.matched_query_terms, r.elapsed_us, h)
            }
            Err(e) => {
                return Err((StatusCode::BAD_REQUEST, e));
            }
        }
    } else if effective_q.contains('|') && excluded.is_empty() {
        // OR-group path: any whitespace-separated word containing `|` becomes
        // a max-within group instead of summing each variant's contribution.
        let (groups, _flat, has_or) = parse_or_groups(entry.idx(), &effective_q);
        let r = if has_or {
            entry.idx().score_with_groups(&groups, inner_k)
        } else {
            entry.idx().score(&tokens, inner_k)
        };
        let h: Vec<SearchHit> = r.hits.iter().map(|h| mk_hit(h.doc_idx, h.score)).collect();
        (r.matched_query_terms, r.elapsed_us, h)
    } else {
        let mut r = if (req.blend_alpha < 1.0 || !qexp_terms.is_empty()) && excluded.is_empty() {
            entry
                .idx()
                .score_blended_qexp(&tokens, inner_k, req.blend_alpha, &qexp_terms)
        } else if req.wand && excluded.is_empty() {
            if entry.idx().has_block_max() {
                entry.idx().score_block_max_wand(&tokens, inner_k)
            } else {
                entry.idx().score_wand(&tokens, inner_k)
            }
        } else if req.prf && entry.idx().has_forward() && excluded.is_empty() {
            entry
                .idx()
                .score_with_prf(&tokens, inner_k, req.prf_k, req.prf_t, req.prf_alpha)
        } else if req.mmr && excluded.is_empty() {
            entry.idx().score_with_mmr(&tokens, inner_k, req.mmr_lambda)
        } else if excluded.is_empty() {
            entry.idx().score(&tokens, inner_k)
        } else {
            entry.idx().score_excluding(&tokens, &excluded, inner_k)
        };
        // Apply bigram + proximity bonuses to the top-K (cheap per hit), then
        // re-sort since bonuses can change the order of close-scored hits.
        if q_ordered.len() >= 2 && (req.bigram_weight > 0.0 || req.proximity_weight > 0.0) {
            for h in &mut r.hits {
                h.score +=
                    entry
                        .idx()
                        .bigram_bonus_for_doc(h.doc_idx, &q_ordered, req.bigram_weight);
                h.score +=
                    entry
                        .idx()
                        .proximity_bonus_for_doc(h.doc_idx, &tokens, req.proximity_weight);
            }
            r.hits.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let h: Vec<SearchHit> = r.hits.iter().map(|h| mk_hit(h.doc_idx, h.score)).collect();
        (r.matched_query_terms, r.elapsed_us, h)
    };

    // Filter: drop hits failing any predicate, evaluated against the stored
    // payload (arbitrary fields) and numeric rank attributes. A field is
    // "known" if the artifact has payloads (any field may exist) or it is a
    // declared rank attribute; otherwise a typo is a clean 400.
    let mut hits = hits;
    if !req.filter.is_empty() {
        let idx = entry.idx();
        for f in &req.filter {
            let known = idx.has_payload() || idx.rank_value(&f.field, 0).is_some();
            if !known {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("unknown filter field '{}'", f.field),
                ));
            }
        }
        hits.retain(|h| {
            passes_filters(idx.payload(h.doc_idx as usize), &req.filter, |field| {
                idx.rank_value(field, h.doc_idx)
            })
        });
    }

    // Dedup filter: drop near-duplicate hits when --dedup was set at build.
    if req.dedup && entry.idx().has_dedup() {
        hits.retain(|h| entry.idx().is_canonical(h.doc_idx));
    }
    // Phrase filter: drop any hit that doesn't contain every quoted phrase
    // as a contiguous run of tokens. Only fires when the artifact has the
    // positional index and the query actually contains a "…" run.
    if req.phrase && entry.idx().has_positions() {
        let phrases = extract_phrases(&effective_q);
        if !phrases.is_empty() {
            let phrase_tokens: Vec<Vec<u32>> = phrases
                .iter()
                .map(|p| entry.idx().tokenize_query_keep_order(p))
                .filter(|t| !t.is_empty())
                .collect();
            if !phrase_tokens.is_empty() {
                hits.retain(|h| {
                    phrase_tokens
                        .iter()
                        .all(|p| entry.idx().doc_contains_phrase(h.doc_idx, p))
                });
            }
        }
    }

    // Matched-result count, after every result filter and before pagination.
    // It remains bounded by the internal candidate window, as documented on
    // SearchResp::total.
    let total = hits.len();
    // Pagination trim. All filters run first so a page contains up to k
    // matching results rather than candidates that are subsequently dropped.
    if offset >= hits.len() {
        hits.clear();
    } else {
        hits.drain(..offset);
        hits.truncate(k);
    }

    // Attach stored payloads to the returned page (after pagination so we only
    // parse the hits we return).
    if req.with_payload && entry.idx().has_payload() {
        let idx = entry.idx();
        for h in &mut hits {
            h.payload = idx
                .payload(h.doc_idx as usize)
                .and_then(|s| serde_json::from_str(s).ok());
        }
    }

    {
        let mut lat = entry.latencies_us.lock().unwrap();
        if lat.len() >= 1024 {
            lat.remove(0);
        }
        lat.push(elapsed_us);
    }

    let total_us = req_start.elapsed().as_micros() as u64;
    if total_us >= state.slow_query_us {
        entry.slow_queries.fetch_add(1, AtomicOrdering::Relaxed);
        tracing::warn!(
            target: "sift::slow",
            index = %idx_name,
            latency_us = total_us,
            score_us = elapsed_us,
            q_len = effective_q.len(),
            k = req.k,
            "slow query",
        );
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let entry_row = SlowEntry {
            ts: now_secs,
            q: effective_q.clone(),
            latency_us: total_us,
            score_us: elapsed_us,
            n_hits: hits.len(),
            k: req.k,
        };
        let mut log = entry.slow_log.lock().unwrap();
        if log.len() >= 128 {
            log.pop_front();
        }
        log.push_back(entry_row);
    }

    if let Some(key) = cache_key {
        let cached = CachedSearch {
            matched_terms,
            total,
            hits: hits.clone(),
            spell_corrected: spell_corrected.clone(),
        };
        entry.cache.lock().unwrap().put(key, cached);
    }

    let facets = if req.facets.is_empty() {
        HashMap::new()
    } else {
        match entry.idx().facet_counts(&tokens, &req.facets) {
            Ok(per_field) => {
                let mut map: HashMap<String, Vec<FacetBucket>> = HashMap::new();
                for (name, buckets) in req.facets.iter().zip(per_field) {
                    let bucket_vec: Vec<FacetBucket> = buckets
                        .into_iter()
                        .map(|(value, count)| FacetBucket { value, count })
                        .collect();
                    map.insert(name.clone(), bucket_vec);
                }
                map
            }
            Err(name) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("unknown facet field '{name}'"),
                ));
            }
        }
    };

    let body = RespJson(SearchResp {
        index: idx_name,
        matched_terms,
        total,
        latency_us: elapsed_us,
        hits,
        spell_corrected,
        facets,
    });
    let total_ms = req_start.elapsed().as_secs_f64() * 1000.0;
    let score_ms = elapsed_us as f64 / 1000.0;
    let st = format!("score;dur={:.3},total;dur={:.3}", score_ms, total_ms);
    Ok(([("server-timing", st)], body).into_response())
}
include!("query/endpoints.rs");

#[cfg(test)]
mod tests {
    use super::*;

    fn request(extra: &str) -> SearchReq {
        let json = if extra.is_empty() {
            r#"{"q":"test"}"#.to_string()
        } else {
            format!(r#"{{"q":"test",{extra}}}"#)
        };
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn search_defaults_are_valid() {
        let req = request("");
        assert_eq!(req.blend_alpha, 0.5);
        assert!(validate_search_req(&req).is_ok());
    }

    #[test]
    fn rejects_weights_outside_documented_ranges() {
        let req = request(r#""blend_alpha":1.1"#);
        assert!(validate_search_req(&req)
            .unwrap_err()
            .contains("blend_alpha"));

        let req = request(r#""proximity_weight":-0.1"#);
        assert!(validate_search_req(&req)
            .unwrap_err()
            .contains("proximity_weight"));
    }

    #[test]
    fn rejects_unknown_rank_direction_and_empty_fields() {
        let req = request(r#""rank":[{"field":"price","order":"sideways"}]"#);
        assert!(validate_search_req(&req)
            .unwrap_err()
            .contains("rank order"));

        let req = request(r#""filter":[{"field":""}]"#);
        assert!(validate_search_req(&req)
            .unwrap_err()
            .contains("filter field"));
    }
}
