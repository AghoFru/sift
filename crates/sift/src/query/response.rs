#[derive(Serialize, Clone)]
pub struct SearchHit {
    pub doc_id: String,
    pub score: f32,
    pub snippet: String,
    /// Internal doc index. Skipped from serialization (it's an opaque
    /// build-time id) but kept so the phrase filter can hit the positional
    /// index without re-resolving by string.
    #[serde(skip)]
    pub(crate) doc_idx: u32,
    /// The stored source document, when the artifact has payloads and the
    /// request didn't opt out via `"with_payload": false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
    /// HTML-escaped snippet with case-insensitive matches of each query word
    /// wrapped in `<mark>…</mark>`. Present only when the caller set
    /// `"highlight": true`. The plain `snippet` is always included alongside.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet_html: Option<String>,
    /// Per-doc feature breakdown for the optional reranker. Present only when
    /// the artifact has the exact-CSR sidecar written (rebuild required for
    /// older artifacts).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Features>,
}

#[derive(Serialize, Clone)]
pub struct Features {
    pub bm25_combined: f32,
    pub bm25_exact: f32,
    pub bm25_semantic: f32,
    pub bm25_blended: f32,
    pub bigram_bonus: f32,
    pub qexp_score: f32,
    pub composition_similarity: f32,
    pub retrieval_score: f32,
    pub coverage: f32,
    pub doc_len: f32,
}

#[derive(Serialize, Clone)]
pub struct FacetBucket {
    pub value: f32,
    pub count: u32,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub index: String,
    pub matched_terms: u32,
    /// Number of results that matched the query and passed all filters, before
    /// pagination. Exact for typical result sizes; for very large result sets
    /// it is capped at the internal candidate window.
    pub total: usize,
    pub latency_us: u64,
    pub hits: Vec<SearchHit>,
    /// Present only when the caller passed `"spell": true` and the artifact
    /// has a spell sidecar. Echoes the query as it was after correction so
    /// the client can render "showing results for …" UX.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spell_corrected: Option<String>,
    /// Per-field bucket counts over every doc that scored > 0, populated
    /// when the request set `facets`. Empty when no facets were requested.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub facets: HashMap<String, Vec<FacetBucket>>,
    #[serde(skip)]
    pub timing: Option<SearchTiming>,
}
