//! Server side of HTTP pull replication. Three read-only endpoints let a
//! `sift replicate --from <url>` client mirror an index over the network with
//! the same commit-point ordering the filesystem path uses:
//!
//!   GET /replicate/{index}/listing            generation + file inventory
//!   GET /replicate/{index}/file?path=<rel>     stream one file's bytes
//!
//! These expose the full on-disk index (segment files carry document text and
//! payloads), so they are mounted only when the operator passes
//! `--enable-replication`, and they sit behind the same auth layer as the
//! other protected routes. Path access is confined to the index directory:
//! the requested relative path is rejected if it escapes via `..`, is
//! absolute, or resolves (canonicalized) outside the index root.

use super::{resolve_alias, AppState};
use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Json as RespJson, Response},
};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

#[derive(Serialize)]
pub(crate) struct FileEntry {
    /// Path relative to the index directory, forward-slash separated.
    pub path: String,
    pub len: u64,
}

#[derive(Serialize)]
pub(crate) struct ListingResp {
    /// Manifest generation, or 0 for a legacy single-artifact index.
    pub generation: u64,
    /// "manifest" (multi-segment) or "legacy" (single artifact).
    pub layout: String,
    /// Every replicable file under the index, relative paths.
    pub files: Vec<FileEntry>,
}

/// Resolve the index name (after alias) to its on-disk directory.
fn index_dir(state: &AppState, name: &str) -> Option<PathBuf> {
    let real = resolve_alias(state, name);
    let map = state.indices.read().unwrap();
    map.get(&real).map(|e| e.path.clone())
}

/// Recursively collect regular files under `root`, returning paths relative to
/// `root` with `/` separators. Skips temp/hidden churn (`.tmp`, dotfiles).
fn collect_files(root: &Path, dir: &Path, out: &mut Vec<FileEntry>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name.ends_with(".tmp") {
            continue;
        }
        let p = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect_files(root, &p, out)?;
        } else if meta.is_file() {
            if let Ok(rel) = p.strip_prefix(root) {
                let rel = rel.to_string_lossy().replace('\\', "/");
                out.push(FileEntry {
                    path: rel,
                    len: meta.len(),
                });
            }
        }
    }
    Ok(())
}

pub(crate) async fn listing(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(index): axum::extract::Path<String>,
) -> Result<RespJson<ListingResp>, (StatusCode, String)> {
    let dir = index_dir(&state, &index)
        .ok_or((StatusCode::NOT_FOUND, format!("unknown index '{index}'")))?;
    let manifested = dir.join("manifest.json").exists();
    let generation = if manifested {
        sift_core::Manifest::read(&dir)
            .map(|m| m.generation)
            .unwrap_or(0)
    } else {
        0
    };
    let mut files = Vec::new();
    collect_files(&dir, &dir, &mut files)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("listing: {e}")))?;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(RespJson(ListingResp {
        generation,
        layout: if manifested { "manifest" } else { "legacy" }.to_string(),
        files,
    }))
}

#[derive(Deserialize)]
pub(crate) struct FileQuery {
    path: String,
}

/// True if `rel` is a safe in-tree relative path: not absolute, no `..` or
/// other non-normal components, no Windows prefix/root.
fn safe_relative(rel: &str) -> bool {
    let p = Path::new(rel);
    p.components().all(|c| matches!(c, Component::Normal(_)))
}

pub(crate) async fn file(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(index): axum::extract::Path<String>,
    Query(q): Query<FileQuery>,
) -> Result<Response, (StatusCode, String)> {
    let dir = index_dir(&state, &index)
        .ok_or((StatusCode::NOT_FOUND, format!("unknown index '{index}'")))?;
    if !safe_relative(&q.path) {
        return Err((StatusCode::BAD_REQUEST, "illegal path".into()));
    }
    let target = dir.join(&q.path);
    // Defense in depth: canonicalize and confirm the resolved file is inside
    // the (canonical) index directory, so symlinks can't escape it either.
    let root = std::fs::canonicalize(&dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("canonicalize: {e}"),
        )
    })?;
    let real = std::fs::canonicalize(&target)
        .map_err(|_| (StatusCode::NOT_FOUND, "no such file".into()))?;
    if !real.starts_with(&root) || !real.is_file() {
        return Err((StatusCode::NOT_FOUND, "no such file".into()));
    }
    let bytes = tokio::fs::read(&real)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("read: {e}")))?;
    Ok((
        [(header::CONTENT_TYPE, "application/octet-stream")],
        Body::from(bytes),
    )
        .into_response())
}
