// Search request schema, defaults, and boundary validation.

#[derive(Deserialize, Serialize)]
pub struct SearchOptions {
    /// Plain terms preserve CLI behavior. Query syntax enables the HTTP query language.
    #[serde(skip)]
    pub syntax: SearchSyntax,
    pub index: Option<String>,
    pub q: String,
    #[serde(default = "default_k")]
    pub k: usize,
    /// If true and the artifact has the feature sidecar, include per-doc feature
    /// breakdown in each hit (used by reranker training and downstream models).
    #[serde(default)]
    pub features: bool,
    /// If true and the artifact has a spell-correction sidecar, the query is
    /// corrected against the corpus vocabulary before tokenization. The
    /// corrected form is echoed back in the response.
    #[serde(default)]
    pub spell: bool,
    /// If true, apply MMR diversity rerank on an oversampled top-K before
    /// truncating. Trades some BM25-rank for less near-duplicate output.
    #[serde(default)]
    pub mmr: bool,
    /// MMR relevance weight in [0,1]. 1.0 = pure relevance; 0.0 = pure
    /// diversity. Only used when `mmr: true`.
    #[serde(default = "default_mmr_lambda")]
    pub mmr_lambda: f32,
    /// If false, skip the result cache (both read and write). Useful for
    /// A/B testing and benchmarks. Default true.
    #[serde(default = "default_true")]
    pub cache: bool,
    /// Pseudo-relevance feedback (Rocchio-style query expansion). Requires
    /// the forward index sidecar (built with --forward).
    #[serde(default)]
    pub prf: bool,
    #[serde(default = "default_prf_k")]
    pub prf_k: usize,
    #[serde(default = "default_prf_t")]
    pub prf_t: usize,
    #[serde(default = "default_prf_alpha")]
    pub prf_alpha: f32,
    /// Custom-ranking tiers. Each entry: {"field":"popularity","order":"desc"}.
    /// Higher value wins for "desc"; lower wins for "asc". Tie-breaks BM25
    /// inside a band of `rank_band` × top-BM25 wide.
    #[serde(default)]
    pub rank: Vec<RankTier>,
    #[serde(default = "default_rank_band")]
    pub rank_band: f32,
    /// When true, every hit gets a `snippet_html` field with the matched
    /// query words wrapped in <mark>…</mark>. Off by default; clients that
    /// render plain text shouldn't pay the per-hit escape cost.
    #[serde(default)]
    pub highlight: bool,
    /// When true and the artifact was built with --positions, any
    /// "double-quoted" sub-phrases in `q` become exact-sequence filters:
    /// a hit must literally contain every phrase as a contiguous run of
    /// tokens (after the same tokenization and normalization the index used).
    /// When `q` has no quoted phrases this is a no-op. Default true so the
    /// quoted-phrase syntax "just works".
    #[serde(default = "default_true")]
    pub phrase: bool,
    /// Blended exact+expansion weight in [0,1]. <1 down-weights the
    /// expansion (synonym/neighbor) signal so literal matches keep the top
    /// ranks while expansion only surfaces when no exact match exists.
    /// Default 0.5 balances literal and expanded scoring. Requires the
    /// exact-CSR sidecar.
    #[serde(default = "default_blend_alpha")]
    pub blend_alpha: f32,
    /// Weight of the adjacent-pair (bigram) bonus applied to the top-K.
    /// 0 disables. Requires the bigram sidecar (built by default).
    #[serde(default = "default_bigram_weight")]
    pub bigram_weight: f32,
    /// Weight of the term-proximity bonus applied to the top-K:
    /// `weight / (1 + min_gap)` between two distinct query terms. Requires
    /// the positions sidecar (--positions at build); 0 disables.
    #[serde(default)]
    pub proximity_weight: f32,
    /// Query-side expansion weight. When > 0 and the artifact has the
    /// query-expansion sidecar, each query term's embedding neighbors are
    /// added as weighted query terms (weight = qexp_weight * sim) scored
    /// against the exact index. 0 disables.
    #[serde(default)]
    pub qexp_weight: f32,
    /// Weight of the optional order-aware composition rerank. Requires an
    /// artifact built with `--compositional`. 0 disables.
    #[serde(default)]
    pub composition_weight: f32,
    /// Weight of the contextual query-document rerank. Requires a server
    /// started with `--cross-encoder`. 0 disables.
    #[serde(default)]
    pub contextual_weight: f32,
    /// When true and the artifact was built with --dedup, drop hits whose
    /// content hash matches an earlier doc. Default off so the response
    /// includes every match by default; clients that want unique results
    /// opt in.
    #[serde(default)]
    pub dedup: bool,
    /// Use the WAND-pruned BM25 scorer (top-k via posting-list skipping
    /// over per-term upper bounds). Same result as the default modulo
    /// float associativity; only meaningfully faster on large posting
    /// lists where `score()`'s dense accumulator dominates.
    #[serde(default)]
    pub wand: bool,
    /// Per-rank-attribute predicates that hits must satisfy. Each entry
    /// names a field declared at build time and specifies a single
    /// comparator. All predicates must pass for a hit to survive
    /// (logical AND). Hits referencing an unknown field 400.
    /// Example: [{"field":"price","lt":100},
    ///           {"field":"rating","gte":4.0},
    ///           {"field":"category","eq":3}]
    #[serde(default)]
    pub filter: Vec<FilterClause>,
    /// Names of rank-attribute fields to faceted-count over the full
    /// matched set. The response gets a `facets` field with
    /// `{field: [{value, count}, ...]}` sorted by descending count.
    /// Unknown field names 400 the request.
    #[serde(default)]
    pub facets: Vec<String>,
    /// Skip this many hits from the top before returning `k`. Implemented
    /// by retrieving `offset + k` internally and dropping the head; safe
    /// for hosted pagination as long as the index doesn't change between
    /// page requests.
    #[serde(default)]
    pub offset: usize,
    /// Include each hit's stored JSON payload (the source document) in the
    /// response when the artifact has payloads. Default true.
    #[serde(default = "default_true")]
    pub with_payload: bool,
    /// Rerank the top candidates with the server-loaded GBDT model
    /// (`--reranker`). Defaults to on when a model is loaded; pass false to
    /// get the raw BM25 ordering. Ignored when no model is loaded.
    #[serde(default)]
    pub rerank: Option<bool>,
}

#[derive(Deserialize, Serialize)]
pub struct RankTier {
    pub field: String,
    #[serde(default = "default_rank_order")]
    pub order: String,
}

fn default_rank_order() -> String {
    "desc".into()
}
fn default_rank_band() -> f32 {
    0.10
}
fn default_blend_alpha() -> f32 {
    0.5
}
fn default_bigram_weight() -> f32 {
    0.4
}

fn default_prf_k() -> usize {
    10
}
fn default_prf_t() -> usize {
    10
}
fn default_prf_alpha() -> f32 {
    0.3
}

fn default_true() -> bool {
    true
}

fn default_k() -> usize {
    10
}

fn default_mmr_lambda() -> f32 {
    0.7
}

pub(crate) fn validate_search_req(req: &SearchOptions) -> Result<(), String> {
    fn in_unit_interval(name: &str, value: f32) -> Result<(), String> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(())
        } else {
            Err(format!("{name} must be a finite number in [0, 1]"))
        }
    }

    fn non_negative(name: &str, value: f32) -> Result<(), String> {
        if value.is_finite() && value >= 0.0 {
            Ok(())
        } else {
            Err(format!("{name} must be a finite non-negative number"))
        }
    }

    in_unit_interval("blend_alpha", req.blend_alpha)?;
    in_unit_interval("mmr_lambda", req.mmr_lambda)?;
    non_negative("prf_alpha", req.prf_alpha)?;
    non_negative("rank_band", req.rank_band)?;
    non_negative("bigram_weight", req.bigram_weight)?;
    non_negative("proximity_weight", req.proximity_weight)?;
    non_negative("qexp_weight", req.qexp_weight)?;
    in_unit_interval("composition_weight", req.composition_weight)?;
    in_unit_interval("contextual_weight", req.contextual_weight)?;

    for tier in &req.rank {
        if tier.field.trim().is_empty() {
            return Err("rank field must not be empty".to_string());
        }
        if !matches!(
            tier.order.as_str(),
            "asc" | "ascending" | "ASC" | "desc" | "descending" | "DESC"
        ) {
            return Err(format!(
                "rank order for '{}' must be 'asc' or 'desc'",
                tier.field
            ));
        }
    }
    for clause in &req.filter {
        if clause.field.trim().is_empty() {
            return Err("filter field must not be empty".to_string());
        }
    }
    if req.facets.iter().any(|field| field.trim().is_empty()) {
        return Err("facet field must not be empty".to_string());
    }
    Ok(())
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            syntax: SearchSyntax::Query,
            index: Default::default(),
            q: Default::default(),
            k: default_k(),
            features: Default::default(),
            spell: Default::default(),
            mmr: Default::default(),
            mmr_lambda: default_mmr_lambda(),
            cache: default_true(),
            prf: Default::default(),
            prf_k: default_prf_k(),
            prf_t: default_prf_t(),
            prf_alpha: default_prf_alpha(),
            rank: Default::default(),
            rank_band: default_rank_band(),
            highlight: Default::default(),
            phrase: default_true(),
            blend_alpha: default_blend_alpha(),
            bigram_weight: default_bigram_weight(),
            proximity_weight: Default::default(),
            qexp_weight: Default::default(),
            composition_weight: Default::default(),
            contextual_weight: Default::default(),
            dedup: Default::default(),
            wand: Default::default(),
            filter: Default::default(),
            facets: Default::default(),
            offset: Default::default(),
            with_payload: default_true(),
            rerank: Default::default(),
        }
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum SearchSyntax {
    #[default]
    Query,
    Terms,
}

impl SearchOptions {
    pub fn new(query: impl Into<String>) -> Self {
        Self { q: query.into(), ..Self::default() }
    }
}
