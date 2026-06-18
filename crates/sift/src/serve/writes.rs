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
use sift_core::{add_tombstones, next_segment_name, Manifest, SegmentRef};
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

/// Render a JSON id value (string or number) to the canonical string id.
fn json_to_id(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
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

    // Validate + serialize the posted docs to JSONL now so a bad payload fails
    // fast, before we touch the index.
    let mut jsonl = String::new();
    let mut ids: Vec<String> = Vec::with_capacity(req.docs.len());
    for (i, d) in req.docs.iter().enumerate() {
        let obj = d.as_object().ok_or((
            StatusCode::BAD_REQUEST,
            format!("doc {i} must be a JSON object"),
        ))?;
        let id = obj
            .get("id")
            .map(json_to_id)
            .ok_or((StatusCode::BAD_REQUEST, format!("doc {i} missing 'id'")))?;
        if obj.get("text").and_then(|v| v.as_str()).is_none() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("doc {i} missing string 'text'"),
            ));
        }
        // Store the whole document as the payload (all fields preserved); coerce
        // `id` to a string so the builder parses it.
        let mut row = obj.clone();
        row.insert("id".to_string(), serde_json::Value::String(id.clone()));
        jsonl.push_str(&serde_json::Value::Object(row).to_string());
        jsonl.push('\n');
        ids.push(id);
    }
    let added = req.docs.len();
    let upsert = req.upsert;

    let _guard = state.write_lock.lock().await;
    let path = index_dir.clone();
    let mname = model_name.clone();
    let model2 = model.clone();
    let built = tokio::task::spawn_blocking(move || -> Result<(u64, usize), String> {
        let mut manifest = Manifest::ensure(&path).map_err(|e| e.to_string())?;
        let seg_name = next_segment_name(&path).map_err(|e| e.to_string())?;
        let ordinal = sift_core::parse_seg_ordinal(&seg_name);
        let seg_dir = path.join(&seg_name);
        std::fs::create_dir_all(&seg_dir).map_err(|e| e.to_string())?;
        let src = seg_dir.join("source.jsonl");
        std::fs::write(&src, jsonl.as_bytes()).map_err(|e| e.to_string())?;
        let recipe = vec![
            "--model".to_string(),
            mname.clone(),
            "--block-max".to_string(),
        ];
        let args = crate::index_cmd::build_args_from(&src, &seg_dir, &recipe)
            .map_err(|e| e.to_string())?;
        crate::build::run_with_model(args, &model2).map_err(|e| e.to_string())?;
        manifest.segments.push(SegmentRef {
            dir: seg_name,
            source: Some(src.display().to_string()),
            build_args: recipe,
        });
        manifest.generation += 1;
        manifest.write(&path).map_err(|e| e.to_string())?;
        // Upsert: supersede older copies of these ids (dead below this segment).
        if upsert {
            add_tombstones(&path, &ids, ordinal).map_err(|e| e.to_string())?;
        }
        Ok((manifest.generation, manifest.segments.len()))
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
        let mut manifest = Manifest::ensure(&path).map_err(|e| e.to_string())?;
        let added =
            add_tombstones(&path, &ids, sift_core::TOMBSTONE_ALL).map_err(|e| e.to_string())?;
        manifest.generation += 1;
        manifest.write(&path).map_err(|e| e.to_string())?;
        Ok((manifest.generation, manifest.segments.len(), added))
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
