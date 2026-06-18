//! Operational handlers: /stats, /metrics, /reload, /snapshot, /alias.

use super::{discover_artifacts, make_entry, AppState};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Json as RespJson},
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;
use std::time::Instant;

#[derive(Serialize)]
pub(crate) struct StatsResp {
    indices: Vec<IndexStats>,
}

#[derive(Serialize)]
struct IndexStats {
    name: String,
    n_docs: u64,
    queries_seen: u64,
    last_window_mean_us: f64,
    last_window_p95_us: u64,
    last_window_size: usize,
}

pub(crate) async fn stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut out = Vec::new();
    for (name, entry) in state.indices.read().unwrap().iter() {
        let lat = entry.latencies_us.lock().unwrap();
        let n = lat.len();
        let mean = if n > 0 {
            lat.iter().copied().map(|x| x as f64).sum::<f64>() / n as f64
        } else {
            0.0
        };
        let p95 = if n > 0 {
            let mut sorted = lat.clone();
            sorted.sort_unstable();
            let i = ((n as f64) * 0.95).floor() as usize;
            sorted[i.min(n - 1)]
        } else {
            0
        };
        out.push(IndexStats {
            name: name.clone(),
            n_docs: entry.set.n_docs_total() as u64,
            queries_seen: n as u64,
            last_window_mean_us: mean,
            last_window_p95_us: p95,
            last_window_size: n,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    RespJson(StatsResp { indices: out })
}

/// Prometheus-style text metrics. One line per (index, metric) pair.
pub(crate) async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut out = String::new();
    out.push_str("# HELP sift_queries_total Total search requests received.\n");
    out.push_str("# TYPE sift_queries_total counter\n");
    for (name, e) in state.indices.read().unwrap().iter() {
        let v = e.queries_total.load(AtomicOrdering::Relaxed);
        out.push_str(&format!("sift_queries_total{{index=\"{name}\"}} {v}\n"));
    }
    out.push_str("# HELP sift_cache_hits_total Result cache hits.\n");
    out.push_str("# TYPE sift_cache_hits_total counter\n");
    for (name, e) in state.indices.read().unwrap().iter() {
        let v = e.cache_hits.load(AtomicOrdering::Relaxed);
        out.push_str(&format!("sift_cache_hits_total{{index=\"{name}\"}} {v}\n"));
    }
    out.push_str("# HELP sift_slow_queries_total Queries exceeding the slow threshold.\n");
    out.push_str("# TYPE sift_slow_queries_total counter\n");
    for (name, e) in state.indices.read().unwrap().iter() {
        let v = e.slow_queries.load(AtomicOrdering::Relaxed);
        out.push_str(&format!(
            "sift_slow_queries_total{{index=\"{name}\"}} {v}\n"
        ));
    }
    out.push_str("# HELP sift_latency_us Recent-window latency in microseconds.\n");
    out.push_str("# TYPE sift_latency_us gauge\n");
    for (name, e) in state.indices.read().unwrap().iter() {
        let lat = e.latencies_us.lock().unwrap();
        let n = lat.len();
        if n == 0 {
            continue;
        }
        let mut sorted = lat.clone();
        sorted.sort_unstable();
        let p50 = sorted[n / 2];
        let p95 = sorted[((n as f64) * 0.95) as usize].min(*sorted.last().unwrap());
        let p99 = sorted[((n as f64) * 0.99) as usize].min(*sorted.last().unwrap());
        out.push_str(&format!(
            "sift_latency_us{{index=\"{name}\",q=\"p50\"}} {p50}\nsift_latency_us{{index=\"{name}\",q=\"p95\"}} {p95}\nsift_latency_us{{index=\"{name}\",q=\"p99\"}} {p99}\n"
        ));
    }
    ([("content-type", "text/plain; version=0.0.4")], out)
}

#[derive(Deserialize)]
pub(crate) struct ReloadReq {
    /// Either the name of a single loaded index to refresh, or omit to
    /// rediscover the artifacts directory and reload every index.
    #[serde(default)]
    name: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct ReloadResp {
    reloaded: Vec<String>,
    added: Vec<String>,
    removed: Vec<String>,
    elapsed_ms: u64,
}

pub(crate) async fn reload(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ReloadReq>,
) -> Result<RespJson<ReloadResp>, (StatusCode, String)> {
    let t0 = Instant::now();
    let mut reloaded: Vec<String> = Vec::new();
    let mut added: Vec<String> = Vec::new();
    let mut removed: Vec<String> = Vec::new();

    match req.name {
        Some(name) => {
            // Single-index path: re-open the same on-disk directory the
            // entry was originally loaded from. Bails out cleanly if the
            // name isn't currently loaded (caller should use the no-name
            // form to pick up brand-new artifacts).
            let path = {
                let map = state.indices.read().unwrap();
                map.get(&name).map(|e| e.path.clone())
            }
            .ok_or((
                StatusCode::NOT_FOUND,
                format!("index '{name}' is not currently loaded"),
            ))?;
            let new_entry =
                Arc::new(make_entry(&path).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?);
            {
                let mut map = state.indices.write().unwrap();
                map.insert(name.clone(), new_entry);
            }
            reloaded.push(name);
        }
        None => {
            // Rediscover everything. New names are added; existing names
            // get fresh entries; names that vanished from disk are removed.
            let discovered = discover_artifacts(&state.artifacts_dir).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("rediscover: {e}"),
                )
            })?;
            let new_names: HashSet<String> = discovered.keys().cloned().collect();
            let mut map = state.indices.write().unwrap();
            let old_names: HashSet<String> = map.keys().cloned().collect();
            for (k, v) in discovered {
                if old_names.contains(&k) {
                    reloaded.push(k.clone());
                } else {
                    added.push(k.clone());
                }
                map.insert(k, Arc::new(v));
            }
            for k in old_names.difference(&new_names) {
                map.remove(k);
                removed.push(k.clone());
            }
            reloaded.sort();
            added.sort();
            removed.sort();
        }
    }

    Ok(RespJson(ReloadResp {
        reloaded,
        added,
        removed,
        elapsed_ms: t0.elapsed().as_millis() as u64,
    }))
}

#[derive(Deserialize)]
pub(crate) struct SnapshotReq {
    /// Index name to snapshot; required.
    name: String,
    /// Output tarball path. Defaults to `<artifacts_dir>/<name>.sift.tar`.
    #[serde(default)]
    out: Option<PathBuf>,
}

#[derive(Serialize)]
pub(crate) struct SnapshotResp {
    name: String,
    out: String,
    bytes: u64,
    elapsed_ms: u64,
}

pub(crate) async fn snapshot(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SnapshotReq>,
) -> Result<RespJson<SnapshotResp>, (StatusCode, String)> {
    let t0 = Instant::now();
    let src = {
        let map = state.indices.read().unwrap();
        map.get(&req.name).map(|e| e.path.clone())
    }
    .ok_or((
        StatusCode::NOT_FOUND,
        format!("index '{}' is not currently loaded", req.name),
    ))?;
    let out = req
        .out
        .unwrap_or_else(|| state.artifacts_dir.join(format!("{}.sift.tar", req.name)));
    if !src.is_dir() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("source path {} is not a directory", src.display()),
        ));
    }
    // Run `tar -cf <out> -C <parent> <basename>` so the archive layout is
    // a single .sift directory ready to be untarred next to other indices.
    let parent = src.parent().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("source path {} has no parent", src.display()),
    ))?;
    let base = src.file_name().ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("source path {} has no basename", src.display()),
    ))?;
    let status = std::process::Command::new("tar")
        .arg("-cf")
        .arg(&out)
        .arg("-C")
        .arg(parent)
        .arg(base)
        .status()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("tar spawn: {e}")))?;
    if !status.success() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("tar exited with {status}"),
        ));
    }
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    Ok(RespJson(SnapshotResp {
        name: req.name,
        out: out.display().to_string(),
        bytes,
        elapsed_ms: t0.elapsed().as_millis() as u64,
    }))
}

#[derive(Deserialize)]
pub(crate) struct AliasReq {
    /// "set", "delete", or "list".
    action: String,
    /// Alias name (the public-facing handle clients hit).
    #[serde(default)]
    name: Option<String>,
    /// Real index name the alias should resolve to. Required for "set".
    #[serde(default)]
    target: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct AliasResp {
    aliases: HashMap<String, String>,
}

pub(crate) async fn alias(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AliasReq>,
) -> Result<RespJson<AliasResp>, (StatusCode, String)> {
    match req.action.as_str() {
        "list" => {
            let map = state.aliases.read().unwrap();
            Ok(RespJson(AliasResp {
                aliases: map.clone(),
            }))
        }
        "set" => {
            let name = req
                .name
                .ok_or((StatusCode::BAD_REQUEST, "missing 'name'".into()))?;
            let target = req
                .target
                .ok_or((StatusCode::BAD_REQUEST, "missing 'target'".into()))?;
            {
                let indices = state.indices.read().unwrap();
                if !indices.contains_key(&target) {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("target index '{target}' is not currently loaded"),
                    ));
                }
            }
            let mut map = state.aliases.write().unwrap();
            map.insert(name, target);
            Ok(RespJson(AliasResp {
                aliases: map.clone(),
            }))
        }
        "delete" => {
            let name = req
                .name
                .ok_or((StatusCode::BAD_REQUEST, "missing 'name'".into()))?;
            let mut map = state.aliases.write().unwrap();
            map.remove(&name);
            Ok(RespJson(AliasResp {
                aliases: map.clone(),
            }))
        }
        other => Err((
            StatusCode::BAD_REQUEST,
            format!("unknown action '{other}'; want 'set'|'delete'|'list'"),
        )),
    }
}
