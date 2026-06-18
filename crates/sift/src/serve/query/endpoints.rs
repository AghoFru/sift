// Read-only diagnostic, suggestion, and explain endpoints.


#[derive(Deserialize)]
pub(crate) struct ExplainReq {
    index: Option<String>,
    q: String,
    #[serde(default = "default_explain_k")]
    k: usize,
    #[serde(default)]
    spell: bool,
}

fn default_explain_k() -> usize {
    5
}

#[derive(Serialize)]
struct ExplainTerm {
    token: String,
    idf: f32,
    tf_in_doc: f32,
    contribution: f32,
}

#[derive(Serialize)]
struct ExplainHitResp {
    doc_id: String,
    score: f32,
    snippet: String,
    doc_len: f32,
    terms: Vec<ExplainTerm>,
}

#[derive(Serialize)]
pub(crate) struct ExplainResp {
    index: String,
    query_tokens: Vec<String>,
    matched_terms: u32,
    latency_us: u64,
    hits: Vec<ExplainHitResp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    spell_corrected: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct FailuresReq {
    #[serde(default)]
    index: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct FailuresResp {
    index: String,
    entries: Vec<SlowEntry>,
}

pub(crate) async fn failures(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FailuresReq>,
) -> Result<RespJson<FailuresResp>, (StatusCode, String)> {
    let idx_name = req.index.unwrap_or_else(|| state.default.clone());
    let idx_name = resolve_alias(&state, &idx_name);
    let entry: Arc<IndexEntry> = {
        let map = state.indices.read().unwrap();
        map.get(&idx_name).cloned()
    }
    .ok_or((StatusCode::NOT_FOUND, format!("unknown index '{idx_name}'")))?;
    let log = entry.slow_log.lock().unwrap();
    let entries: Vec<SlowEntry> = log.iter().cloned().collect();
    Ok(RespJson(FailuresResp {
        index: idx_name,
        entries,
    }))
}

#[derive(Deserialize)]
pub(crate) struct SuggestReq {
    index: Option<String>,
    prefix: String,
    #[serde(default = "default_suggest_k")]
    k: usize,
}

fn default_suggest_k() -> usize {
    8
}

#[derive(Serialize)]
struct SuggestItem {
    word: String,
    df: u32,
}

#[derive(Serialize)]
pub(crate) struct SuggestResp {
    index: String,
    items: Vec<SuggestItem>,
}

pub(crate) async fn suggest(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SuggestReq>,
) -> Result<RespJson<SuggestResp>, (StatusCode, String)> {
    let idx_name = req.index.unwrap_or_else(|| state.default.clone());
    let idx_name = resolve_alias(&state, &idx_name);
    let entry: Arc<IndexEntry> = {
        let map = state.indices.read().unwrap();
        map.get(&idx_name).cloned()
    }
    .ok_or((StatusCode::NOT_FOUND, format!("unknown index '{idx_name}'")))?;
    if !entry.idx().has_spell() {
        return Err((
            StatusCode::BAD_REQUEST,
            "index has no spell vocab; rebuild with --spell to enable /suggest".into(),
        ));
    }
    let k = req.k.clamp(1, 50);
    let items: Vec<SuggestItem> = entry
        .idx()
        .suggest(&req.prefix, k)
        .unwrap_or_default()
        .into_iter()
        .map(|(word, df)| SuggestItem { word, df })
        .collect();
    Ok(RespJson(SuggestResp {
        index: idx_name,
        items,
    }))
}

pub(crate) async fn explain(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExplainReq>,
) -> Result<RespJson<ExplainResp>, (StatusCode, String)> {
    let idx_name = req.index.unwrap_or_else(|| state.default.clone());
    let idx_name = resolve_alias(&state, &idx_name);
    let entry: Arc<IndexEntry> = {
        let map = state.indices.read().unwrap();
        map.get(&idx_name).cloned()
    }
    .ok_or((StatusCode::NOT_FOUND, format!("unknown index '{idx_name}'")))?;

    let (effective_q, spell_corrected): (String, Option<String>) =
        if req.spell && entry.idx().has_spell() {
            let c = entry.idx().spell_correct_query(&req.q);
            let echoed = if c != req.q { Some(c.clone()) } else { None };
            (c, echoed)
        } else {
            (req.q.clone(), None)
        };

    let (tokens, _excluded) = entry.idx().parse_query(&effective_q);
    let query_tokens: Vec<String> = tokens
        .iter()
        .filter_map(|&t| entry.idx().token_string(t))
        .collect();

    if tokens.is_empty() {
        return Ok(RespJson(ExplainResp {
            index: idx_name,
            query_tokens,
            matched_terms: 0,
            latency_us: 0,
            hits: vec![],
            spell_corrected,
        }));
    }

    let k = req.k.clamp(1, 50);
    let r = entry.idx().score_with_explain(&tokens, k);
    let hits: Vec<ExplainHitResp> = r
        .hits
        .iter()
        .map(|h| ExplainHitResp {
            doc_id: entry.idx().doc_id(h.doc_idx as usize).to_string(),
            score: h.score,
            snippet: entry.idx().doc_snip(h.doc_idx as usize).to_string(),
            doc_len: h.doc_len,
            terms: h
                .terms
                .iter()
                .map(|t| ExplainTerm {
                    token: t.token.clone(),
                    idf: t.idf,
                    tf_in_doc: t.tf_in_doc,
                    contribution: t.contribution,
                })
                .collect(),
        })
        .collect();

    Ok(RespJson(ExplainResp {
        index: idx_name,
        query_tokens,
        matched_terms: r.matched_query_terms,
        latency_us: r.elapsed_us,
        hits,
        spell_corrected,
    }))
}
