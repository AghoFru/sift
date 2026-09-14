//! HTTP adapters for search and diagnostic operations.

use super::{resolve_alias, AppState, IndexEntry, SlowEntry};
use crate::query::{SearchErrorKind, SearchOptions};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Json as RespJson, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    Json(req): Json<SearchOptions>,
) -> Result<Response, (StatusCode, String)> {
    crate::query::validate_search_req(&req).map_err(|error| (StatusCode::BAD_REQUEST, error))?;
    let name = req.index.clone().unwrap_or_else(|| state.default.clone());
    let name = resolve_alias(&state, &name);
    let entry = state
        .indices
        .read()
        .unwrap()
        .get(&name)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, format!("unknown index '{name}'")))?;
    let result = crate::query::execute(&state.search_context, &entry, req, name).map_err(
        |(kind, message)| {
            let code = match kind {
                SearchErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
                SearchErrorKind::Conflict => StatusCode::CONFLICT,
                SearchErrorKind::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
                SearchErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (code, message)
        },
    )?;
    let timing = result.timing.as_ref().map(|timing| {
        let stage = if timing.cache_hit { "cache" } else { "score" };
        format!(
            "{stage};dur={:.3},total;dur={:.3}",
            timing.score_us as f64 / 1000.0,
            timing.total_us as f64 / 1000.0
        )
    });
    let body = RespJson(result);
    Ok(match timing {
        Some(value) => ([("server-timing", value)], body).into_response(),
        None => body.into_response(),
    })
}

include!("query/endpoints.rs");
