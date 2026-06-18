//! sift - Axum HTTP server fronting one or more `.sift` artifacts.
//!
//! Discovers `*.sift` directories under `--artifacts-dir` (default `./artifacts`),
//! loads them all into memory (via mmap), and exposes:
//!   GET  /                version and loaded-index metadata
//!   GET  /healthz         liveness
//!   GET  /datasets        list of loaded indices
//!   POST /search          {"index": "scifact", "q": "...", "k": 10}
//!   GET  /stats           per-index size + last-N-query mean latency
//!
//! Module layout:
//!   - `query`   read-side handlers (/search, /explain, /suggest, /failures)
//!   - `filters` payload filter clauses + evaluation
//!   - `admin`   operational handlers (/stats, /metrics, /reload, /snapshot, /alias)
//!   - `writes`  live write path (/add, /delete, /compact)

mod admin;
#[cfg(feature = "cross-encoder")]
mod ce;
mod filters;
mod query;
mod replicate_srv;
mod rerank;
mod writes;

/// Stub so the rest of the server compiles identically without the
/// cross-encoder feature; loading reports the missing feature.
#[cfg(not(feature = "cross-encoder"))]
mod ce {
    use anyhow::{anyhow, Result};
    use std::path::Path;

    pub(crate) struct CrossEncoder;

    impl CrossEncoder {
        pub(crate) fn load(_dir: &Path, _threads: usize, _max_len: usize) -> Result<Self> {
            Err(anyhow!(
                "this binary was built without the cross-encoder feature \
                 (rebuild with --features cross-encoder)"
            ))
        }

        pub(crate) fn score_pairs(&self, _q: &str, _docs: &[String]) -> Result<Vec<f32>> {
            Ok(Vec::new())
        }
    }
}

use crate::build::Model;
use anyhow::{Context, Result};
use axum::body::Body;
use axum::http::Request;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Json as RespJson, Response},
    routing::{get, post},
    Router,
};
use clap::Args;
use serde::Serialize;
use sift_core::{Index, IndexSet, Manifest};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{
    atomic::{AtomicU64, Ordering as AtomicOrdering},
    Arc, Mutex, RwLock,
};
use std::time::Instant;
use tokio::sync::Mutex as AsyncMutex;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

#[derive(Clone)]
pub(crate) struct CachedSearch {
    pub(crate) matched_terms: u32,
    pub(crate) total: usize,
    pub(crate) hits: Vec<query::SearchHit>,
    pub(crate) spell_corrected: Option<String>,
}

/// A single entry in the rolling slow-query buffer.
#[derive(Clone, Serialize)]
pub(crate) struct SlowEntry {
    /// Seconds since UNIX epoch when the query was served.
    pub(crate) ts: u64,
    /// The effective query string (after spell correction, if any).
    pub(crate) q: String,
    /// Total request latency (server-side) in microseconds.
    pub(crate) latency_us: u64,
    /// Score-pass latency reported by the scoring function in microseconds.
    pub(crate) score_us: u64,
    /// Number of hits returned.
    pub(crate) n_hits: usize,
    /// Top-K bound the request asked for.
    pub(crate) k: usize,
}

pub(crate) struct IndexEntry {
    /// One or more immutable segments plus a tombstone set. Single-segment
    /// indices (the common case) expose the full single-`Index` feature set via
    /// [`IndexEntry::idx`]; multi-segment indices are served through the merged
    /// path in `query::search`.
    pub(crate) set: IndexSet,
    /// On-disk path of the .sift directory, kept so /reload can re-open it.
    pub(crate) path: PathBuf,
    /// Rolling last-K query latencies in microseconds (for /stats).
    pub(crate) latencies_us: Mutex<Vec<u64>>,
    /// LRU result cache. Key is the hash of the request shape. Bypassed
    /// when the request opts out via `"cache": false`.
    pub(crate) cache: Mutex<lru::LruCache<u64, CachedSearch>>,
    /// Rolling buffer of slow-query metadata for /failures. Capped to keep
    /// memory bounded under load; oldest entries get evicted.
    pub(crate) slow_log: Mutex<std::collections::VecDeque<SlowEntry>>,
    /// Counters for /metrics. Atomic to avoid contention on the hot path.
    pub(crate) queries_total: AtomicU64,
    pub(crate) cache_hits: AtomicU64,
    pub(crate) slow_queries: AtomicU64,
}

impl IndexEntry {
    /// The primary (and, for single-segment indices, only) segment. Callers in
    /// the rich single-segment query path use this after confirming
    /// `self.set.is_single()`.
    pub(crate) fn idx(&self) -> &Index {
        self.set.primary()
    }
}

struct RateBucket {
    tokens: f64,
    last: Instant,
}

pub(crate) struct AppState {
    /// Held behind an RwLock so /reload can atomically swap a single entry
    /// while live queries clone an Arc out of the map under a brief read lock.
    pub(crate) indices: RwLock<HashMap<String, Arc<IndexEntry>>>,
    /// Root directory holding the .sift artifact subdirectories. Used by
    /// /reload to locate an index by name on disk.
    pub(crate) artifacts_dir: PathBuf,
    pub(crate) default: String,
    pub(crate) slow_query_us: u64,
    /// When `Some`, every protected request must present
    /// `Authorization: Bearer <one-of-these>`. When `None`, auth is disabled.
    api_keys: Option<HashSet<String>>,
    /// Per-key token bucket. `qps == 0` disables rate limiting.
    rate_qps: f64,
    rate_burst: f64,
    rate_buckets: Mutex<HashMap<String, RateBucket>>,
    /// Flips to true once background prewarm completes. `/readyz` returns
    /// 503 until then.
    ready: AtomicBool,
    /// alias name → real index name. Resolved before looking up the
    /// indices map so clients can hit "production" while operators
    /// rotate the underlying artifact.
    pub(crate) aliases: RwLock<HashMap<String, String>>,
    /// When true, no mutating routes are mounted: the server is search-only.
    read_only: bool,
    /// Resident embedding models, keyed by model name, so live writes don't
    /// reload the model per request.
    pub(crate) models: RwLock<HashMap<String, Arc<Model>>>,
    /// Serializes all write operations (add / delete / compact) across every
    /// index. Writes are infrequent and heavy; one global lock keeps manifest
    /// read-modify-write and the in-memory swap race-free.
    pub(crate) write_lock: AsyncMutex<()>,
    /// Auto-compact an index once it exceeds this many segments (0 disables).
    pub(crate) compact_threshold: usize,
    /// Optional GBDT reranker (LightGBM dump_model JSON), applied to the
    /// top-K of /search by default when loaded (`"rerank": false` opts out).
    pub(crate) reranker: Option<Arc<rerank::GbdtModel>>,
    /// Optional ONNX cross-encoder. Takes precedence over the GBDT model
    /// when both are loaded; same `"rerank": false` opt-out.
    pub(crate) cross_encoder: Option<Arc<ce::CrossEncoder>>,
    /// How many top candidates the cross-encoder rescores per query.
    pub(crate) ce_depth: usize,
}

/// Walk one hop through the alias map: "production" → "scifact-v3".
/// Returns the raw input unchanged if not aliased. One hop is enough
/// for the operational pattern we care about (rotate target on deploy);
/// multi-hop aliasing is intentionally unsupported.
pub(crate) fn resolve_alias(state: &AppState, name: &str) -> String {
    let aliases = state.aliases.read().unwrap();
    aliases
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.to_string())
}

async fn healthz() -> &'static str {
    "ok"
}

#[derive(Serialize)]
struct VersionResp {
    sift_version: &'static str,
    artifact_schema_version: u32,
    n_indices: usize,
    indices: Vec<String>,
}

async fn version(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let map = state.indices.read().unwrap();
    let mut names: Vec<String> = map.keys().cloned().collect();
    names.sort();
    let n = names.len();
    RespJson(VersionResp {
        sift_version: env!("CARGO_PKG_VERSION"),
        artifact_schema_version: sift_core::SCHEMA_VERSION,
        n_indices: n,
        indices: names,
    })
}

async fn readyz(State(state): State<Arc<AppState>>) -> Response {
    if state.ready.load(AtomicOrdering::Relaxed) {
        (StatusCode::OK, "ready").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "warming").into_response()
    }
}

/// Request-ID middleware. Reads `x-request-id` from the incoming request;
/// if absent, generates one (compact base32 of nanos + a small counter).
/// Echoes the chosen id back in the response and stashes it in the
/// `tracing` span so slow-query / error logs carry the same correlation id
/// downstream observability already saw.
async fn request_id_layer(headers: HeaderMap, req: Request<Body>, next: Next) -> Response {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let id = headers
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            let seq = CTR.fetch_add(1, Ordering::Relaxed);
            format!("sift-{:016x}-{:04x}", now, seq & 0xffff)
        });
    let id_for_resp = id.clone();
    tracing::Span::current().record("request_id", tracing::field::display(&id));
    let mut resp = next.run(req).await;
    if let Ok(v) = id_for_resp.parse() {
        resp.headers_mut().insert("x-request-id", v);
    }
    resp
}

/// Bearer-token gate + per-key token-bucket rate limit. When `api_keys` is
/// None auth is skipped; when `rate_qps == 0` rate limiting is skipped.
async fn auth_layer(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request<Body>,
    next: Next,
) -> Response {
    // 1. Auth.
    let validated_key: Option<String> = if let Some(allow) = state.api_keys.as_ref() {
        let presented = headers
            .get("authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .map(|s| s.trim().to_string());
        match presented {
            Some(k) if allow.contains(&k) => Some(k),
            _ => {
                return (
                    StatusCode::UNAUTHORIZED,
                    [("www-authenticate", "Bearer realm=\"sift\"")],
                    "missing or invalid API key",
                )
                    .into_response();
            }
        }
    } else {
        None
    };

    // 2. Rate limit (only meaningful when keys are configured).
    if state.rate_qps > 0.0 {
        if let Some(key) = validated_key {
            let now = Instant::now();
            let mut buckets = state.rate_buckets.lock().unwrap();
            let b = buckets.entry(key).or_insert_with(|| RateBucket {
                tokens: state.rate_burst,
                last: now,
            });
            let elapsed = now.duration_since(b.last).as_secs_f64();
            b.tokens = (b.tokens + elapsed * state.rate_qps).min(state.rate_burst);
            b.last = now;
            if b.tokens < 1.0 {
                let retry_after = ((1.0 - b.tokens) / state.rate_qps).ceil() as u64;
                let retry_after = retry_after.max(1);
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [("retry-after", retry_after.to_string())],
                    "rate limit exceeded",
                )
                    .into_response();
            }
            b.tokens -= 1.0;
        }
    }

    next.run(req).await
}

/// Build a fresh `IndexEntry` from an index directory (fresh stats + cache; the
/// cache must reset on every write so stale results can't survive a mutation).
pub(crate) fn make_entry(path: &Path) -> Result<IndexEntry, String> {
    let set = IndexSet::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
    Ok(IndexEntry {
        set,
        path: path.to_path_buf(),
        latencies_us: Mutex::new(Vec::with_capacity(1024)),
        cache: Mutex::new(lru::LruCache::new(NonZeroUsize::new(1024).unwrap())),
        slow_log: Mutex::new(std::collections::VecDeque::with_capacity(128)),
        queries_total: AtomicU64::new(0),
        cache_hits: AtomicU64::new(0),
        slow_queries: AtomicU64::new(0),
    })
}

pub(crate) fn discover_artifacts(root: &PathBuf) -> Result<HashMap<String, IndexEntry>> {
    let mut out = HashMap::new();
    for entry in std::fs::read_dir(root).with_context(|| format!("reading {}", root.display()))? {
        let entry = entry?;
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        let name = match p.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        let stripped = name.strip_suffix(".sift").unwrap_or(name);
        let e = make_entry(&p).map_err(|e| anyhow::anyhow!(e))?;
        tracing::info!(
            "loaded index '{}' from {} (segments={}, n_docs={})",
            stripped,
            p.display(),
            e.set.n_segments(),
            e.set.primary().meta.n_docs
        );
        out.insert(stripped.to_string(), e);
    }
    Ok(out)
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Directory containing `.sift` artifact subdirectories.
    #[arg(long, default_value = "./artifacts", env = "SIFT_ARTIFACTS")]
    pub artifacts: PathBuf,
    /// Address to bind.
    #[arg(long, default_value = "0.0.0.0:8080", env = "SIFT_BIND")]
    pub bind: String,
    /// Log a slow-query warning when total request latency exceeds this
    /// many microseconds. Default 5000us (5ms).
    #[arg(long, default_value_t = 5000, env = "SIFT_SLOW_QUERY_US")]
    pub slow_query_us: u64,
    /// Optional path to a text file with one API key per line. When set,
    /// /search /explain /stats /metrics require `Authorization: Bearer <key>`.
    /// Blank lines and lines starting with `#` are ignored.
    #[arg(long, env = "SIFT_API_KEYS")]
    pub api_keys: Option<PathBuf>,
    /// Emit logs as JSON (one event per line) for structured log shipping.
    #[arg(long, default_value_t = false, env = "SIFT_LOG_JSON")]
    pub log_json: bool,
    /// Per-API-key token-bucket refill rate in QPS. 0 = unlimited. Only
    /// takes effect when --api-keys is also set.
    #[arg(long, default_value_t = 0.0, env = "SIFT_RATE_QPS")]
    pub rate_qps: f64,
    /// Per-API-key bucket capacity (max burst). Defaults to 5 * rate_qps
    /// when 0 and rate_qps > 0.
    #[arg(long, default_value_t = 0.0, env = "SIFT_RATE_BURST")]
    pub rate_burst: f64,
    /// mlock the indices.bin and data*.bin mappings so they can't be paged
    /// out under memory pressure. Requires `ulimit -l` headroom for the
    /// full size of those files. Best-effort: failures are logged and
    /// search continues to work, just with normal page-cache eviction.
    #[arg(long, default_value_t = false, env = "SIFT_MLOCK_POSTINGS")]
    pub mlock_postings: bool,
    /// Allowed CORS origins (comma-separated). Empty = allow Any.
    /// Specify one or more origins to restrict browser-based clients to
    /// known domains.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "",
        env = "SIFT_CORS_ORIGINS"
    )]
    pub cors_origin: Vec<String>,
    /// Reject incoming request bodies larger than this many bytes (default
    /// 1 MB). Memory-DoS guard. Most /search bodies are < 4 KB.
    #[arg(long, default_value_t = 1_048_576, env = "SIFT_MAX_BODY_BYTES")]
    pub max_body_bytes: usize,
    /// Search-only mode: do not mount any write or admin route (/add, /delete,
    /// /compact, /reload, /snapshot, /alias). Accepts 1/0, true/false, yes/no
    /// via SIFT_READONLY.
    #[arg(
        long,
        default_value_t = false,
        env = "SIFT_READONLY",
        value_parser = clap::builder::BoolishValueParser::new()
    )]
    pub read_only: bool,
    /// Auto-compact an index once it grows past this many segments (0 disables).
    /// Live writes append segments; this bounds query fan-out by merging in the
    /// background.
    #[arg(long, default_value_t = 8, env = "SIFT_COMPACT_THRESHOLD")]
    pub compact_threshold: usize,
    /// Optional path to a GBDT reranker model (LightGBM dump_model JSON).
    /// When set, /search reranks the top candidates with it by default;
    /// requests can opt out with `"rerank": false`. Requires artifacts with
    /// the exact-CSR sidecar (default builds have it).
    #[arg(long, env = "SIFT_RERANKER")]
    pub reranker: Option<PathBuf>,
    /// Optional directory holding an ONNX cross-encoder (`model-int8.onnx`
    /// or `model.onnx` + `tokenizer.json`). When set, /search reranks the
    /// top `--ce-depth` candidates with it by default (~60ms for depth 50
    /// with the int8 MiniLM-L6 model); requests opt out with
    /// `"rerank": false`. Takes precedence over --reranker.
    #[arg(long, env = "SIFT_CROSS_ENCODER")]
    pub cross_encoder: Option<PathBuf>,
    /// Cross-encoder rerank depth (top-N candidates rescored).
    #[arg(long, default_value_t = 50, env = "SIFT_CE_DEPTH")]
    pub ce_depth: usize,
    /// Threads for cross-encoder inference. More is not always faster;
    /// ~half the cores is a good default on big machines.
    #[arg(long, default_value_t = 10, env = "SIFT_CE_THREADS")]
    pub ce_threads: usize,
    /// Max (query + doc) tokens per cross-encoder pair; longer docs truncate.
    #[arg(long, default_value_t = 192, env = "SIFT_CE_MAX_LEN")]
    pub ce_max_len: usize,
    /// Mount the read-only HTTP replication endpoints
    /// (`/replicate/{index}/listing`, `/replicate/{index}/file`) so a
    /// `sift replicate --from <url>` client can pull this server's indices.
    /// Off by default: these serve the full on-disk index (document text and
    /// payloads), so enable only where that exposure is acceptable. Subject to
    /// the same auth as other protected routes; works in read-only mode.
    #[arg(long, default_value_t = false, env = "SIFT_ENABLE_REPLICATION")]
    pub enable_replication: bool,
}

pub fn run(args: ServeArgs) -> Result<()> {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "sift=info,tower_http=info".into());
    if args.log_json {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .json()
            .flatten_event(true)
            .try_init()
            .ok();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .try_init()
            .ok();
    }

    let rt = tokio::runtime::Runtime::new().context("creating tokio runtime")?;
    rt.block_on(run_async(args))
}

async fn run_async(args: ServeArgs) -> Result<()> {
    let bind: SocketAddr = args.bind.parse().context("parsing bind address")?;
    let discovered = discover_artifacts(&args.artifacts)?;
    if discovered.is_empty() {
        anyhow::bail!(
            "no .sift artifacts found under {}",
            args.artifacts.display()
        );
    }
    // pick a deterministic default (prefer "scifact" if present, else min by name)
    let default = if discovered.contains_key("scifact") {
        "scifact".to_string()
    } else {
        discovered.keys().min().unwrap().clone()
    };
    let indices: HashMap<String, Arc<IndexEntry>> = discovered
        .into_iter()
        .map(|(k, v)| (k, Arc::new(v)))
        .collect();

    // Crash recovery: drop orphan segment directories left by a crash mid-build
    // (a segment that was being written when the process died, before its
    // manifest commit; the write was never acked). Write mode only.
    if !args.read_only {
        for (name, entry) in indices.iter() {
            if entry.path.join("manifest.json").exists() {
                if let Ok(m) = Manifest::read(&entry.path) {
                    match sift_core::recover_orphans(&entry.path, &m) {
                        Ok(n) if n > 0 => {
                            tracing::info!(
                                "recovered index '{name}': removed {n} orphan segment(s)"
                            )
                        }
                        Ok(_) => {}
                        Err(e) => tracing::warn!("orphan recovery for '{name}' failed: {e}"),
                    }
                }
            }
        }
    }

    let api_keys = if let Some(p) = &args.api_keys {
        let txt = std::fs::read_to_string(p)
            .with_context(|| format!("reading api-keys file {}", p.display()))?;
        let set: HashSet<String> = txt
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.to_string())
            .collect();
        if set.is_empty() {
            anyhow::bail!("api-keys file {} contained no usable keys", p.display());
        }
        tracing::info!("API key auth enabled ({} keys loaded)", set.len());
        Some(set)
    } else {
        None
    };

    let rate_qps = args.rate_qps.max(0.0);
    let rate_burst = if args.rate_burst > 0.0 {
        args.rate_burst
    } else {
        (rate_qps * 5.0).max(1.0)
    };
    if rate_qps > 0.0 {
        tracing::info!(
            "rate limit enabled: {:.1} qps, burst {:.1} per key",
            rate_qps,
            rate_burst,
        );
    }

    // Optional alias map: <artifacts_dir>/aliases.json mapping
    // {"alias": "real_index_name", ...}.
    let aliases: HashMap<String, String> = {
        let p = args.artifacts.join("aliases.json");
        match std::fs::read_to_string(&p) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
                tracing::warn!("aliases.json parse failed ({e}); ignoring");
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        }
    };
    if !aliases.is_empty() {
        tracing::info!("loaded {} alias(es)", aliases.len());
    }

    let reranker = match &args.reranker {
        Some(p) => {
            let m = rerank::GbdtModel::load(p)
                .with_context(|| format!("loading reranker {}", p.display()))?;
            tracing::info!(
                "reranker loaded from {} ({} features)",
                p.display(),
                m.n_features
            );
            Some(Arc::new(m))
        }
        None => None,
    };
    let cross_encoder = match &args.cross_encoder {
        Some(dir) => Some(Arc::new(ce::CrossEncoder::load(
            dir,
            args.ce_threads.max(1),
            args.ce_max_len.max(16),
        )?)),
        None => None,
    };

    let state = Arc::new(AppState {
        indices: RwLock::new(indices),
        artifacts_dir: args.artifacts.clone(),
        default,
        slow_query_us: args.slow_query_us,
        api_keys,
        rate_qps,
        rate_burst,
        rate_buckets: Mutex::new(HashMap::new()),
        ready: AtomicBool::new(false),
        aliases: RwLock::new(aliases),
        read_only: args.read_only,
        models: RwLock::new(HashMap::new()),
        write_lock: AsyncMutex::new(()),
        compact_threshold: args.compact_threshold,
        reranker,
        cross_encoder,
        ce_depth: args.ce_depth.clamp(10, 500),
    });
    if state.read_only {
        tracing::info!("read-only mode: write/admin routes are not mounted (search-only)");
    }
    let warm_state = state.clone();

    let protected = Router::new()
        .route("/search", post(query::search))
        .route("/explain", post(query::explain))
        .route("/suggest", post(query::suggest))
        .route("/failures", post(query::failures))
        .route("/stats", get(admin::stats))
        .route("/metrics", get(admin::metrics))
        .route_layer(from_fn_with_state(state.clone(), auth_layer));

    let mut app = Router::new()
        .route("/", get(version))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/version", get(version))
        .route("/datasets", get(query::list_datasets))
        .merge(protected);

    // Mutating and admin routes are omitted entirely in read-only mode.
    if !state.read_only {
        let mutating = Router::new()
            .route("/add", post(writes::add_docs))
            .route("/delete", post(writes::delete_docs))
            .route("/compact", post(writes::compact_index))
            .route("/reload", post(admin::reload))
            .route("/snapshot", post(admin::snapshot))
            .route("/alias", post(admin::alias))
            .route_layer(from_fn_with_state(state.clone(), auth_layer));
        app = app.merge(mutating);
    }

    // Read-only HTTP replication endpoints (opt-in). Available even in
    // read-only mode since they only read; behind the same auth layer.
    if args.enable_replication {
        let repl = Router::new()
            .route("/replicate/:index/listing", get(replicate_srv::listing))
            .route("/replicate/:index/file", get(replicate_srv::file))
            .route_layer(from_fn_with_state(state.clone(), auth_layer));
        app = app.merge(repl);
        tracing::info!("HTTP replication endpoints enabled (/replicate/{{index}}/...)");
    }

    let app = app
        .layer(axum::middleware::from_fn(request_id_layer))
        .layer(CompressionLayer::new())
        .layer(axum::extract::DefaultBodyLimit::max(args.max_body_bytes))
        .layer({
            let allowed: Vec<String> = args
                .cors_origin
                .iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if allowed.is_empty() {
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any)
            } else {
                let origins: Vec<axum::http::HeaderValue> =
                    allowed.into_iter().filter_map(|o| o.parse().ok()).collect();
                CorsLayer::new()
                    .allow_origin(origins)
                    .allow_methods(Any)
                    .allow_headers(Any)
            }
        })
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Background warmup: fault the big posting files into the page cache so
    // queries don't pay the cold-start tax. Runs after we start listening, so
    // the server is reachable immediately.
    let mlock = args.mlock_postings;
    tokio::task::spawn_blocking(move || {
        for (name, entry) in warm_state.indices.read().unwrap().iter() {
            let t0 = std::time::Instant::now();
            for seg in &entry.set.segments {
                seg.prewarm_postings();
                if mlock {
                    seg.mlock_postings();
                }
            }
            tracing::info!(
                "warmed{} index '{}' in {:.2}s",
                if mlock { " + mlocked" } else { "" },
                name,
                t0.elapsed().as_secs_f64()
            );
        }
        warm_state.ready.store(true, AtomicOrdering::Relaxed);
        tracing::info!("all indices warmed; /readyz now returns 200");
    });

    tracing::info!("sift listening on http://{}", bind);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    // Graceful shutdown: wait for SIGTERM (or Ctrl-C in foreground); when
    // received, stop accepting new connections and let in-flight requests
    // finish naturally up to axum's default timeout.
    let shutdown = async {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        let mut int = signal(SignalKind::interrupt()).expect("install SIGINT handler");
        tokio::select! {
            _ = term.recv() => tracing::info!("SIGTERM received; draining"),
            _ = int.recv() => tracing::info!("SIGINT received; draining"),
        }
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    tracing::info!("sift shut down cleanly");
    Ok(())
}
