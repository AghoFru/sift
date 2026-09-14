//! Live write path: /add, /delete, /compact. All writes serialize on the
//! global write lock, commit durably on disk (segment fsync before the
//! manifest), then atomically swap the in-memory entry so the change is
//! immediately queryable.

use super::{make_entry, AppState, IndexEntry};
use crate::build::Model;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Json as RespJson, Response},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Deserialize)]
pub(crate) struct AddReq {
    index: String,
    /// Documents to add. Each must have an `id` (string or number) and a string
    /// `text`. Extra fields are ignored for now.
    docs: Vec<serde_json::Value>,
    /// Embedding model for a brand-new index. Ignored when the index exists
    /// (its existing model is used). Defaults to the English potion model.
    #[serde(default)]
    model: Option<String>,
    /// Treat these as upserts: any older copy of a posted id (in an earlier
    /// segment) is superseded so it no longer matches any query, even by terms
    /// that only appear in the old version. Default false (pure append/insert).
    #[serde(default)]
    upsert: bool,
}

#[derive(Deserialize)]
pub(crate) struct DeleteReq {
    index: String,
    ids: Vec<String>,
}

#[derive(Deserialize)]
pub(crate) struct CompactReq {
    index: String,
}

#[derive(Serialize)]
struct WriteResp {
    index: String,
    segments: usize,
    generation: u64,
    /// Docs added (for /add) or ids newly tombstoned (for /delete).
    affected: usize,
}

/// Re-open `path` and atomically replace the in-memory entry for `name`.
pub(crate) fn swap_entry(
    state: &AppState,
    name: &str,
    path: &Path,
) -> Result<Arc<IndexEntry>, String> {
    let entry = Arc::new(make_entry(path)?);
    state
        .indices
        .write()
        .unwrap()
        .insert(name.to_string(), entry.clone());
    Ok(entry)
}

/// Get a resident model by name, loading and caching it on first use.
async fn get_model(
    state: &Arc<AppState>,
    model_name: &str,
) -> Result<Arc<Model>, (StatusCode, String)> {
    if let Some(m) = state.models.read().unwrap().get(model_name).cloned() {
        return Ok(m);
    }
    let mn = model_name.to_string();
    let loaded = tokio::task::spawn_blocking(move || crate::build::load_model(&mn))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("model load task: {e}"),
            )
        })?
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("loading model '{model_name}': {e}"),
            )
        })?;
    let arc = Arc::new(loaded);
    state
        .models
        .write()
        .unwrap()
        .insert(model_name.to_string(), arc.clone());
    Ok(arc)
}

/// Schedule a background compaction if the index now has too many segments.
fn spawn_auto_compact(state: Arc<AppState>, name: String, path: PathBuf) {
    if state.compact_threshold == 0 {
        return;
    }
    tokio::spawn(async move {
        let _guard = state.write_lock.lock().await;
        let segs = {
            state
                .indices
                .read()
                .unwrap()
                .get(&name)
                .map(|e| e.set.n_segments())
                .unwrap_or(0)
        };
        if segs <= state.compact_threshold {
            return;
        }
        let p = path.clone();
        let res = tokio::task::spawn_blocking(move || {
            crate::index_cmd::run_compact(crate::index_cmd::CompactArgs { index: p })
        })
        .await;
        match res {
            Ok(Ok(())) => match swap_entry(&state, &name, &path) {
                Ok(_) => tracing::info!("auto-compacted '{name}' ({segs} segments)"),
                Err(e) => tracing::warn!("auto-compact '{name}': reload failed: {e}"),
            },
            Ok(Err(e)) => tracing::warn!("auto-compact '{name}' skipped: {e}"),
            Err(e) => tracing::warn!("auto-compact '{name}' task error: {e}"),
        }
    });
}

pub(crate) async fn add_docs(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddReq>,
) -> Result<Response, (StatusCode, String)> {
    if req.docs.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "no docs provided".into()));
    }
    let name = req.index.clone();
    if name.is_empty()
        || name.len() > 128
        || name == "."
        || name == ".."
        || name.contains(['/', '\\'])
        || name.chars().any(char::is_control)
    {
        return Err((StatusCode::BAD_REQUEST, "Invalid index name.".into()));
    }
    let jsonl = crate::write::encode_documents(&req.docs)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    let existing = { state.indices.read().unwrap().get(&name).cloned() };
    let index_dir = existing
        .as_ref()
        .map(|e| e.path.clone())
        .unwrap_or_else(|| state.artifacts_dir.join(format!("{name}.sift")));
    let model_name = if let Some(e) = &existing {
        e.idx().meta.model_name.clone()
    } else {
        req.model
            .clone()
            .unwrap_or_else(|| "minishlab/potion-base-8M".to_string())
    };
    let model = get_model(&state, &model_name).await?;

    let added = req.docs.len();
    let mode = if req.upsert {
        crate::WriteMode::Upsert
    } else {
        crate::WriteMode::Insert
    };

    let _guard = state.write_lock.lock().await;
    let path = index_dir.clone();
    let mname = model_name.clone();
    let model2 = model.clone();
    let built = tokio::task::spawn_blocking(move || -> Result<(u64, usize), String> {
        let outcome = crate::write::append_documents(&path, &jsonl, &mname, &model2, mode)
            .map_err(|error| error.to_string())?;
        Ok((outcome.generation, outcome.segments))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("build task: {e}"),
        )
    })?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let (generation, segments) = built;

    swap_entry(&state, &name, &index_dir).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    drop(_guard);
    spawn_auto_compact(state.clone(), name.clone(), index_dir);

    Ok(RespJson(WriteResp {
        index: name,
        segments,
        generation,
        affected: added,
    })
    .into_response())
}

pub(crate) async fn delete_docs(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeleteReq>,
) -> Result<Response, (StatusCode, String)> {
    if req.ids.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "no ids provided".into()));
    }
    let name = req.index.clone();
    let existing = { state.indices.read().unwrap().get(&name).cloned() };
    let index_dir = existing
        .as_ref()
        .map(|e| e.path.clone())
        .ok_or((StatusCode::NOT_FOUND, format!("unknown index '{name}'")))?;

    let _guard = state.write_lock.lock().await;
    let path = index_dir.clone();
    let ids = req.ids.clone();
    let done = tokio::task::spawn_blocking(move || -> Result<(u64, usize, usize), String> {
        let outcome = crate::Engine::delete_from(&path, &ids).map_err(|error| error.to_string())?;
        Ok((outcome.generation, outcome.segments, outcome.affected))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("delete task: {e}"),
        )
    })?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let (generation, segments, added) = done;

    swap_entry(&state, &name, &index_dir).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(RespJson(WriteResp {
        index: name,
        segments,
        generation,
        affected: added,
    })
    .into_response())
}

pub(crate) async fn compact_index(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompactReq>,
) -> Result<Response, (StatusCode, String)> {
    let name = req.index.clone();
    let existing = { state.indices.read().unwrap().get(&name).cloned() };
    let index_dir = existing
        .as_ref()
        .map(|e| e.path.clone())
        .ok_or((StatusCode::NOT_FOUND, format!("unknown index '{name}'")))?;

    let _guard = state.write_lock.lock().await;
    let p = index_dir.clone();
    tokio::task::spawn_blocking(move || {
        crate::index_cmd::run_compact(crate::index_cmd::CompactArgs { index: p })
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("compact task: {e}"),
        )
    })?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("compact: {e}")))?;

    let entry = swap_entry(&state, &name, &index_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(RespJson(WriteResp {
        index: name,
        segments: entry.set.n_segments(),
        generation: 0,
        affected: 0,
    })
    .into_response())
}
