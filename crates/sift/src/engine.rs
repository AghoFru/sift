//! Embedded index ownership and reusable search configuration.

use crate::query::{SearchError, SearchOptions, SearchResponse};
use crate::{ce, rerank};
use anyhow::{Context, Result};
use serde::Serialize;
use sift_core::{Index, IndexSet};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

pub(crate) struct SearchContext {
    pub(crate) slow_query_us: u64,
    pub(crate) reranker: Option<Arc<rerank::GbdtModel>>,
    pub(crate) cross_encoder: Option<Arc<ce::CrossEncoder>>,
    pub(crate) ce_depth: usize,
}

impl Default for SearchContext {
    fn default() -> Self {
        Self {
            slow_query_us: u64::MAX,
            reranker: None,
            cross_encoder: None,
            ce_depth: 50,
        }
    }
}

/// An open index. Searches use a consistent snapshot until `reload` succeeds.
pub struct Engine {
    pub(crate) entry: IndexEntry,
    context: SearchContext,
}

impl Engine {
    /// Create an index from JSON documents and a local model or enabled model download.
    pub fn create(
        path: impl AsRef<Path>,
        model_name: &str,
        documents: &[serde_json::Value],
    ) -> Result<Self> {
        let path = path.as_ref();
        if path.join("meta.json").exists() || path.join("manifest.json").exists() {
            anyhow::bail!("An index already exists at {}", path.display());
        }
        let jsonl = crate::write::encode_documents(documents)?;
        let model = crate::build::load_model(model_name)?;
        crate::write::append_documents(path, &jsonl, model_name, &model, crate::WriteMode::Create)?;
        Self::open(path)
    }

    pub fn write_documents(
        &mut self,
        documents: &[serde_json::Value],
        mode: crate::WriteMode,
    ) -> Result<crate::WriteOutcome> {
        let jsonl = crate::write::encode_documents(documents)?;
        let name = &self.entry.idx().meta.model_name;
        let model = crate::build::load_model(name)?;
        let result = crate::write::append_documents(&self.entry.path, &jsonl, name, &model, mode)?;
        self.reload()?;
        Ok(result)
    }

    /// Build a new index and open it for searches.
    pub fn build(options: crate::BuildOptions) -> Result<Self> {
        let path = options
            .output
            .clone()
            .context("An output directory is required.")?;
        let guard = crate::write::WriteGuard::acquire(&path)?;
        crate::build::run(options)?;
        drop(guard);
        Self::open(path)
    }

    /// Append a source file to an index, creating the index if necessary.
    pub fn append_to(
        path: impl AsRef<Path>,
        options: crate::BuildOptions,
        mode: crate::WriteMode,
    ) -> Result<crate::WriteOutcome> {
        let model = crate::build::load_model(&options.model)?;
        crate::write::append(path.as_ref(), options, &model, mode)
    }

    pub fn append(
        &mut self,
        options: crate::BuildOptions,
        mode: crate::WriteMode,
    ) -> Result<crate::WriteOutcome> {
        let result = Self::append_to(&self.entry.path, options, mode)?;
        self.reload()?;
        Ok(result)
    }

    pub fn delete_from(path: impl AsRef<Path>, ids: &[String]) -> Result<crate::WriteOutcome> {
        crate::write::delete(path.as_ref(), ids)
    }

    pub fn delete(&mut self, ids: &[String]) -> Result<crate::WriteOutcome> {
        let result = Self::delete_from(&self.entry.path, ids)?;
        self.reload()?;
        Ok(result)
    }

    pub fn compact_at(path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let _guard = crate::write::WriteGuard::acquire(path)?;
        crate::write::compact(path)
    }

    pub fn compact(&mut self) -> Result<()> {
        Self::compact_at(&self.entry.path)?;
        self.reload()
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let entry = make_entry(path.as_ref()).map_err(anyhow::Error::msg)?;
        Ok(Self {
            entry,
            context: SearchContext::default(),
        })
    }

    pub fn reload(&mut self) -> Result<()> {
        self.entry = make_entry(&self.entry.path).map_err(anyhow::Error::msg)?;
        Ok(())
    }

    pub fn search(&self, options: SearchOptions) -> Result<SearchResponse, SearchError> {
        let name = options.index.clone().unwrap_or_else(|| {
            self.entry
                .path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        });
        crate::query::execute(&self.context, &self.entry, options, name).map_err(SearchError::from)
    }

    pub fn load_reranker(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let model = rerank::GbdtModel::load(path.as_ref()).context("Loading the reranker.")?;
        tracing::info!(features = model.n_features, "Loaded reranker.");
        self.context.reranker = Some(Arc::new(model));
        self.entry.cache.lock().unwrap().clear();
        Ok(())
    }

    pub fn load_cross_encoder(&mut self, path: impl AsRef<Path>, threads: usize) -> Result<()> {
        let model = ce::CrossEncoder::load(path.as_ref(), threads.max(1), 512)?;
        self.context.cross_encoder = Some(Arc::new(model));
        self.entry.cache.lock().unwrap().clear();
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct CachedSearch {
    pub(crate) matched_terms: u32,
    pub(crate) total: usize,
    pub(crate) hits: Vec<crate::query::SearchHit>,
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
