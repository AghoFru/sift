//! Plain term scoring preserves the CLI's established search contract.

use super::*;

pub(super) fn execute(
    entry: &IndexEntry,
    req: SearchOptions,
    name: String,
) -> Result<SearchResponse, (SearchErrorKind, String)> {
    let (hits, matched_terms, latency_us) = if entry.set.is_single() {
        let index = entry.idx();
        let tokens = index.tokenize_query(&req.q);
        let results = if req.composition_weight > 0.0 {
            index.score_blended_qexp_compositional(
                &tokens,
                &index.tokenize_query_keep_order(&req.q),
                req.k,
                req.blend_alpha,
                &[],
                req.composition_weight,
            )
        } else {
            index.score_blended(&tokens, req.k, req.blend_alpha)
        };
        let hits = results
            .hits
            .iter()
            .map(|hit| SearchHit {
                doc_id: index.doc_id(hit.doc_idx as usize).to_string(),
                doc_idx: hit.doc_idx,
                score: hit.score,
                snippet: index.doc_snip(hit.doc_idx as usize).to_string(),
                payload: None,
                snippet_html: None,
                features: None,
            })
            .collect::<Vec<_>>();
        (hits, results.matched_query_terms, results.elapsed_us)
    } else {
        if req.composition_weight > 0.0 {
            return Err((
                SearchErrorKind::Conflict,
                "Composition reranking requires a single segment. Run `sift compact` first.".into(),
            ));
        }
        let results = entry
            .set
            .search_merged(&req.q, req.k, ScoreMode::Plain, req.blend_alpha);
        let hits = results
            .hits
            .into_iter()
            .map(|hit| SearchHit {
                doc_id: hit.doc_id,
                doc_idx: hit.doc_idx,
                score: hit.score,
                snippet: hit.snippet,
                payload: None,
                snippet_html: None,
                features: None,
            })
            .collect::<Vec<_>>();
        (hits, results.matched_query_terms, results.elapsed_us)
    };
    Ok(SearchResponse {
        index: name,
        total: hits.len(),
        hits,
        matched_terms,
        latency_us,
        spell_corrected: None,
        facets: HashMap::new(),
        timing: None,
    })
}
