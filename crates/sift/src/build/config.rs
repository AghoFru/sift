// CLI configuration, artifact metadata, and reusable model loading.

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// Input corpus (JSONL).
    #[arg(long)]
    pub input: PathBuf,
    /// Output artifact directory. Required for `sift build`; `sift add`
    /// chooses the segment directory itself and overrides this.
    #[arg(long = "out")]
    pub output: Option<PathBuf>,
    /// Input format: `jsonl` (`{id, text}`) or `beir` (`{_id, title, text}`).
    #[arg(long, default_value = "jsonl")]
    pub format: String,
    /// m2v model (Hugging Face repo or local path).
    #[arg(long, default_value = "minishlab/potion-base-8M")]
    pub model: String,
    /// Semantic neighbors per term during expansion.
    #[arg(short = 'k', long = "k-expand", default_value_t = 10)]
    pub k_expand: usize,
    /// Min cosine similarity for an expansion edge.
    #[arg(long, default_value_t = 0.65)]
    pub threshold: f32,
    /// Stopword cutoff (drop tokens with df/N greater than this).
    #[arg(long, default_value_t = 0.4)]
    pub stop_df: f32,
    /// Threadpool cap. 0 = use all cores. Set to 1 for edge / battery-bound.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
    /// Chars of source text to keep as a snippet per doc.
    #[arg(long, default_value_t = 320)]
    pub snippet_chars: usize,
    /// Skip writing the snippets table to reduce artifact size.
    #[arg(long, default_value_t = false)]
    pub no_snippets: bool,
    /// Weight of title-field term occurrences relative to body occurrences
    /// (BM25F-style). 2.0 counts each title token twice in tf and doc length.
    /// Only meaningful for formats that carry a title (`--format beir`).
    #[arg(long, default_value_t = 2.0)]
    pub title_weight: f32,
    /// Multiplier applied to the IDF of WordPiece continuation tokens
    /// (`##`-prefixed subwords). Continuation fragments carry far less signal
    /// than whole words and can surface noise matches; values < 1 damp them.
    /// Applies to both exact and expanded scoring (baked into idf.bin).
    /// Default measured on English BEIR; raise toward 1.0 for heavily
    /// agglutinative or multilingual corpora where subwords carry meaning.
    #[arg(long, default_value_t = 0.4)]
    pub subword_weight: f32,
    /// BM25 k1.
    #[arg(long, default_value_t = 1.5)]
    pub bm25_k1: f32,
    /// BM25 b.
    #[arg(long, default_value_t = 0.75)]
    pub bm25_b: f32,
    /// BM25 delta. 0 = classic BM25. >0 = BM25+ (helps low-freq terms in long docs;
    /// typical value 1.0).
    #[arg(long, default_value_t = 0.0)]
    pub bm25_delta: f32,
    /// Skip the identifier/number/dot normalization pass. Used for isolating
    /// the normalization's contribution to retrieval quality.
    #[arg(long, default_value_t = false)]
    pub no_normalize: bool,
    /// Skip building the bigram inverted index. Used for ablation.
    #[arg(long, default_value_t = false)]
    pub no_bigrams: bool,
    /// Build the SymSpell word-level vocab + deletion table. When present at
    /// serve time, queries are spell-corrected against the corpus's own vocabulary
    /// before WordPiece tokenization. Off by default until measured per corpus.
    #[arg(long, default_value_t = false)]
    pub spell: bool,
    /// Minimum doc-frequency for a word to enter the spell vocabulary. Filters
    /// out singletons (often typos themselves) while keeping legitimate rare
    /// terms like proper nouns. Only used when --spell is set.
    #[arg(long, default_value_t = 2)]
    pub spell_min_df: u32,
    /// Optional wordlist file (one word per line) merged into the spell
    /// vocab on top of the corpus words. Lets a typo whose intended form
    /// only appears in the dictionary (and not the corpus) still get
    /// corrected. Trade-off: dictionary-driven corrections can "fix" valid
    /// dialect or domain-specific spellings (e.g. `colour → color` if the
    /// dictionary is US-English). Off by default; user picks the wordlist.
    #[arg(long)]
    pub spell_dictionary: Option<PathBuf>,
    /// Build the doc-major forward index (transpose of the exact term-major
    /// CSR). Required for pseudo-relevance feedback (PRF) at query time.
    /// Costs ~same storage as the exact CSR. Off by default.
    #[arg(long, default_value_t = false)]
    pub forward: bool,
    /// Pre-tokenize content cleaning. `off` (default) feeds raw text into the
    /// tokenizer. `html` strips tags and decodes entities. `md` strips
    /// CommonMark markup. `auto` sniffs each doc and picks one.
    #[arg(long, default_value = "off", value_parser = ["off", "html", "md", "auto"])]
    pub clean: String,
    /// Comma-separated list of per-doc numeric fields to extract as ranking
    /// attributes. Each named field is read as f32 from the JSONL row;
    /// missing values become 0.0. At query time pass a "rank":
    /// [{"field":..,"order":"desc"|"asc"}, ...] tuple to tie-break BM25.
    /// Example: --rank-fields popularity,rating,price
    #[arg(long, value_delimiter = ',', default_value = "")]
    pub rank_fields: Vec<String>,
    /// Materialize a per-doc positional sequence (doc -> [term_id] in document
    /// order). Required for "double-quoted" phrase queries at serve time.
    /// Adds ~4 bytes per indexed token to the artifact.
    #[arg(long, default_value_t = false)]
    pub positions: bool,
    /// Pack doc-id indices as 24-bit little-endian integers instead of u32
    /// when n_docs fits in 24 bits (≤ 16,777,215). Saves ~25% on the
    /// biggest file in the artifact (indices.bin, exact_indices.bin) and
    /// the same ~25% on resident memory for those mappings. Decode is one
    /// 3-byte load + bit pad per posting; the score inner loop is
    /// otherwise unchanged.
    #[arg(long, default_value_t = false)]
    pub u24_indices: bool,
    /// Precompute per-block max BM25 contribution and write `block_max.bin` +
    /// `block_max_indptr.bin`. Enables the Block-Max WAND scorer at query
    /// time which uses per-block upper bounds instead of the global per-term
    /// bound, allowing much tighter pivot skipping on long posting lists.
    /// Block size is hard-coded to 128 (industry standard).
    #[arg(long, default_value_t = false)]
    pub block_max: bool,
    /// Quantize embeddings to i8 + per-vector scale before the brute top-K
    /// pass. Reduces memory bandwidth ~4× (256 dims × 1 byte vs 256 × 4 bytes)
    /// so the inner dot loop on long vocabularies finishes faster. Off by
    /// default; evaluate per-corpus before enabling.
    #[arg(long, default_value_t = false)]
    pub quantize_embeddings: bool,
    /// Detect exact-content duplicates at build time. Computes an ahash
    /// over the normalized doc text per row; for each cluster of docs that
    /// hash identical, only the lowest-id member is marked canonical.
    /// At serve time, `"dedup": true` on /search drops non-canonical hits.
    #[arg(long, default_value_t = false)]
    pub dedup: bool,
    /// Store posting weights as f16 (data_f16.bin / exact_data_f16.bin)
    /// instead of f32. Halves the two largest files of the artifact and the
    /// memory bandwidth of the score inner loop, which reads the weights
    /// directly off the f16 mmap. BM25 weights fit comfortably in f16's
    /// dynamic range; nDCG is unchanged within tolerance. On by default;
    /// pass `--f16-postings=false` for exact f32 weights.
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub f16_postings: bool,
    /// Skip storing per-doc JSON payloads. By default each source document is
    /// stored in the artifact (payload_text.bin) so it can be returned in
    /// results and filtered on arbitrary fields. Pass this for a smaller
    /// artifact when you don't need payload return or rich filtering.
    #[arg(long, default_value_t = false)]
    pub no_payload: bool,
    /// Add corpus-fitted expansion edges from term co-occurrence (PPMI) on
    /// top of the static-embedding neighbours, scaled by this weight (0 =
    /// off). Where the embedding table gives generic substitutability
    /// (run->running), PPMI gives domain association (insulin<->diabetes)
    /// learned from this corpus. It expands by topical co-occurrence, which
    /// raises recall but can dilute precision, so it is off by default;
    /// measure per corpus. The top neighbour's edge weight equals this value.
    #[arg(long, default_value_t = 0.0)]
    pub corpus_expand_weight: f32,
    /// Co-occurrence window (tokens to each side) for PPMI corpus expansion.
    #[arg(long, default_value_t = 5)]
    pub corpus_window: usize,
    /// Top-K corpus associates kept per term for PPMI expansion.
    #[arg(long, default_value_t = 5)]
    pub corpus_expand_k: usize,
    /// Minimum co-occurrence count for a PPMI edge (drops noise pairs).
    #[arg(long, default_value_t = 3)]
    pub corpus_min_cooc: u32,
}

#[derive(Serialize)]
struct Meta {
    schema_version: u32,
    model_name: String,
    vocab_size: u64,
    n_docs: u64,
    n_nonzero: u64,
    n_nonzero_exact: u64,
    n_active_terms: u64,
    n_stopwords: u64,
    avgdl: f32,
    bm25_k1: f32,
    bm25_b: f32,
    bm25_delta: f32,
    k_expand: u32,
    sim_threshold: f32,
    build_seconds: f64,
    indices_packing: String,
    subword_weight: f32,
}

/// A loaded embedding model: tokenizer, keep-mask, and the m2v embedding table.
/// Loaded once and reused across many segment builds. The serve write path keeps
/// one resident per model name so live `add`s don't reload it.
pub struct Model {
    pub tokenizer: Tokenizer,
    pub tokenizer_path: PathBuf,
    pub embeddings: Vec<Vec<f32>>,
    pub keep_mask: Vec<u8>,
    pub vocab_size: usize,
}

/// Load an m2v model (HF repo or local path) into a reusable [`Model`].
pub fn load_model(name: &str) -> Result<Model> {
    let (tokenizer_path, model_bytes) = load_model_files(name)?;
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("loading tokenizer: {e}"))?;
    let vocab_size = tokenizer.get_vocab_size(true);
    let keep_mask = build_keep_mask(&tokenizer);
    let embeddings = load_embeddings(&model_bytes, vocab_size)?;
    Ok(Model {
        tokenizer,
        tokenizer_path,
        embeddings,
        keep_mask,
        vocab_size,
    })
}
