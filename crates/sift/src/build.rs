//! `sift build` - produce a `.sift` artifact from a JSONL corpus.
//!
//! Same recipe as the Python reference (`sift_build/build.py`):
//!   1. Tokenize each doc with the model's WordPiece tokenizer.
//!   2. Compute doc-frequency, cut stopwords (df/N > stop_df).
//!   3. Load and normalize m2v embeddings for the active subwords.
//!   4. For each active subword, find its exact top-K neighbors by cosine
//!      similarity and retain edges with sim ≥ threshold;
//!      add weighted contributions to those neighbors' posting lists.
//!   5. Sort + dedup triplets into a CSR sparse `[V × N]` matrix.
//!   6. Write artifact files matching the schema-v1 layout that `sift-core`
//!      mmaps at serve time.

use anyhow::{anyhow, Context, Result};
use clap::Args;
use half::f16;
use hf_hub::api::sync::Api;
use safetensors::{tensor::Dtype, SafeTensors};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

#[cfg(feature = "multithread")]
use rayon::prelude::*;

use sift_core::SCHEMA_VERSION;
const SPECIAL: &[&str] = &["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"];

include!("build/config.rs");

pub fn run(args: BuildArgs) -> Result<()> {
    println!("[1/6] loading model {}", args.model);
    let model = load_model(&args.model)?;
    run_with_model(args, &model)
}

/// Build one segment artifact using an already-loaded [`Model`], reading
/// documents from `args.input` and writing to `args.output`. This is the shared
/// core of `sift build`, `sift add`, and the server's live write path.
pub fn run_with_model(args: BuildArgs, model: &Model) -> Result<()> {
    let t_total = Instant::now();

    let out: PathBuf = args
        .output
        .clone()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| anyhow!("--out is required"))?;

    #[cfg(feature = "multithread")]
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;

    let tokenizer = &model.tokenizer;
    let keep_mask = &model.keep_mask;
    let embeddings = &model.embeddings;
    let vocab_size = model.vocab_size;
    let tokenizer_path = &model.tokenizer_path;
    let n_keep: usize = keep_mask.iter().map(|&x| x as usize).sum();
    let emb_dim = embeddings[0].len();
    println!("      vocab={vocab_size}  content-tokens={n_keep}  emb_dim={emb_dim}");

    println!("[2/6] tokenizing corpus from {}", args.input.display());
    let t0 = Instant::now();
    // Title token streams are only materialized when title weighting is on.
    let want_titles = (args.title_weight - 1.0).abs() > f32::EPSILON;
    let (doc_ids, doc_snips, tokenized, tokenized_titles) = tokenize_corpus(
        &args.input,
        &args.format,
        tokenizer,
        keep_mask,
        args.snippet_chars,
        args.no_normalize,
        &args.clean,
        want_titles,
    )?;
    let n = doc_ids.len();
    if n == 0 {
        return Err(anyhow!("no documents read from {}", args.input.display()));
    }
    println!("      {n} docs in {:.1}s", t0.elapsed().as_secs_f64());

    // df pass - parallel reduce over docs
    let t0 = Instant::now();
    #[cfg(feature = "multithread")]
    let df: Vec<u64> = tokenized
        .par_iter()
        .fold(
            || vec![0u64; vocab_size],
            |mut local, toks| {
                let mut seen = ahash::AHashSet::default();
                for &t in toks {
                    if seen.insert(t) {
                        local[t as usize] += 1;
                    }
                }
                local
            },
        )
        .reduce(
            || vec![0u64; vocab_size],
            |mut a, b| {
                for t in 0..vocab_size {
                    a[t] += b[t];
                }
                a
            },
        );
    #[cfg(not(feature = "multithread"))]
    let df: Vec<u64> = {
        let mut df = vec![0u64; vocab_size];
        for toks in &tokenized {
            let mut seen = ahash::AHashSet::default();
            for &t in toks {
                if seen.insert(t) {
                    df[t as usize] += 1;
                }
            }
        }
        df
    };
    // Relative stopword cutoff: drop terms appearing in more than stop_df of the
    // docs. This is only meaningful with enough docs; on a tiny segment (e.g. a
    // single-doc live add) `floor(stop_df * n)` rounds to 0 and every term would
    // be dropped, leaving an empty, unsearchable index. Below a small floor we
    // keep all terms; segments this size get merged by compaction anyway, and
    // real corpora (n >= STOP_MIN_DOCS) are unaffected.
    const STOP_MIN_DOCS: usize = 10;
    let stop_threshold = if n >= STOP_MIN_DOCS {
        (args.stop_df * n as f32) as u64
    } else {
        u64::MAX
    };
    let stop_mask: Vec<u8> = df
        .iter()
        .map(|&c| if c > stop_threshold { 1 } else { 0 })
        .collect();
    let n_stops: u64 = stop_mask.iter().map(|&x| x as u64).sum();
    println!(
        "      df pass {:.2}s  stopwords (df/N > {}): {n_stops}",
        t0.elapsed().as_secs_f64(),
        args.stop_df
    );

    // Parallel single pass: apply stop_mask, build per-thread (df_kept, inverted, doc_lens).
    // Merge in chunk order so doc_lens stays in original doc order.
    let t0 = Instant::now();
    #[cfg(feature = "multithread")]
    let n_threads = rayon::current_num_threads().max(1);
    #[cfg(not(feature = "multithread"))]
    let n_threads = 1usize;
    let chunk_size = (n + n_threads - 1) / n_threads.max(1);

    type ChunkOut = (
        Vec<u64>,
        ahash::AHashMap<u32, Vec<(u32, u32)>>,
        Vec<f32>,
        // bigram hash → list of doc ids that contain it (deduped within doc).
        ahash::AHashMap<u64, Vec<u32>>,
    );

    let no_bigrams = args.no_bigrams;
    // Extra weight per title-token occurrence (BM25F-lite). 0.0 = off.
    let title_extra = if want_titles {
        args.title_weight - 1.0
    } else {
        0.0
    };
    let titles_ref = &tokenized_titles;
    let process_chunk = |chunk_i: usize, docs: &[Vec<u32>]| -> ChunkOut {
        let mut df_local = vec![0u64; vocab_size];
        let mut inv_local: ahash::AHashMap<u32, Vec<(u32, u32)>> = ahash::AHashMap::default();
        let mut dl_local: Vec<f32> = Vec::with_capacity(docs.len());
        let mut bg_local: ahash::AHashMap<u64, Vec<u32>> = ahash::AHashMap::default();
        let chunk_start = chunk_i * chunk_size;
        for (i, toks) in docs.iter().enumerate() {
            let di = (chunk_start + i) as u32;
            let mut tf_map: ahash::AHashMap<u32, u32> = ahash::AHashMap::default();
            let mut kept: Vec<u32> = Vec::with_capacity(toks.len());
            for &t in toks {
                if stop_mask[t as usize] == 1 {
                    continue;
                }
                *tf_map.entry(t).or_insert(0) += 1;
                kept.push(t);
            }
            // Title field weighting: each title occurrence counts an extra
            // (title_weight - 1) in tf and in the doc length, mirroring
            // BM25F's weighted field frequencies.
            let mut extra_len = 0f32;
            if title_extra != 0.0 {
                let mut t_counts: ahash::AHashMap<u32, u32> = ahash::AHashMap::default();
                for &t in &titles_ref[di as usize] {
                    if stop_mask[t as usize] == 1 {
                        continue;
                    }
                    *t_counts.entry(t).or_insert(0) += 1;
                }
                for (t, c) in t_counts {
                    let extra = (title_extra * c as f32).round() as i64;
                    if extra == 0 {
                        continue;
                    }
                    extra_len += extra as f32;
                    let cur = tf_map.get(&t).copied().unwrap_or(0) as i64;
                    let nv = (cur + extra).max(0) as u32;
                    if nv == 0 {
                        tf_map.remove(&t);
                    } else {
                        tf_map.insert(t, nv);
                    }
                }
            }
            dl_local.push(kept.len() as f32 + extra_len);
            for (t, tf) in tf_map {
                inv_local.entry(t).or_default().push((di, tf));
                df_local[t as usize] += 1;
            }
            // bigrams: emit each unique (t_i, t_{i+1}) once per doc.
            // Skipped entirely when --no-bigrams is passed (saves work + storage).
            if !no_bigrams {
                let mut seen_bg: ahash::AHashSet<u64> = ahash::AHashSet::default();
                for w in kept.windows(2) {
                    let h = ((w[0] as u64) << 32) | (w[1] as u64);
                    if seen_bg.insert(h) {
                        bg_local.entry(h).or_default().push(di);
                    }
                }
            }
        }
        (df_local, inv_local, dl_local, bg_local)
    };

    #[cfg(feature = "multithread")]
    let chunks: Vec<ChunkOut> = tokenized
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(i, docs)| process_chunk(i, docs))
        .collect();
    #[cfg(not(feature = "multithread"))]
    let chunks: Vec<ChunkOut> = tokenized
        .chunks(chunk_size)
        .enumerate()
        .map(|(i, docs)| process_chunk(i, docs))
        .collect();

    // Optional positional forward index: doc -> [term_id] in document order.
    // Required for phrase-query verification. Built directly from the
    // tokenized buffer before we drop it.
    let (pos_indptr, pos_terms): (Vec<u64>, Vec<u32>) = if args.positions {
        let t_pos = Instant::now();
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0u64);
        let mut total = 0u64;
        for toks in &tokenized {
            total += toks.len() as u64;
            indptr.push(total);
        }
        let mut terms = Vec::with_capacity(total as usize);
        for toks in &tokenized {
            terms.extend_from_slice(toks);
        }
        println!(
            "      positional index {:.2}s  tokens={}",
            t_pos.elapsed().as_secs_f64(),
            total
        );
        (indptr, terms)
    } else {
        (Vec::new(), Vec::new())
    };

    // Corpus-fitted expansion: accumulate PPMI co-occurrence edges from the
    // tokenized stream before it is freed. Stashed per source term as
    // (associate, weight) and merged into the neighbour graph after the
    // embedding edges. Off (empty) unless --corpus-expand-weight > 0.
    let corpus_edges: Vec<Vec<(u32, f32)>> = if args.corpus_expand_weight > 0.0 {
        let t_ppmi = Instant::now();
        let e = build_ppmi_edges(
            &tokenized,
            &stop_mask,
            vocab_size,
            args.corpus_window.max(1),
            args.corpus_expand_k.max(1),
            args.corpus_min_cooc.max(1),
            args.corpus_expand_weight,
        );
        let n_edges: usize = e.iter().map(|v| v.len()).sum();
        println!(
            "      PPMI corpus edges: {n_edges} ({:.2}s)",
            t_ppmi.elapsed().as_secs_f64()
        );
        e
    } else {
        Vec::new()
    };

    // free the now-unused tokenized buffer (large for msmarco)
    drop(tokenized);

    // merge
    let mut df: Vec<u64> = vec![0u64; vocab_size];
    let mut inverted: Vec<Vec<(u32, u32)>> = (0..vocab_size).map(|_| Vec::new()).collect();
    let mut doc_lens: Vec<f32> = Vec::with_capacity(n);
    let mut bigram_index: ahash::AHashMap<u64, Vec<u32>> = ahash::AHashMap::default();
    for (df_p, inv_p, dl_p, bg_p) in chunks {
        for t in 0..vocab_size {
            df[t] += df_p[t];
        }
        for (t, postings) in inv_p {
            inverted[t as usize].extend(postings);
        }
        doc_lens.extend(dl_p);
        for (h, docs) in bg_p {
            bigram_index.entry(h).or_default().extend(docs);
        }
    }
    // Filter bigrams: drop too-rare (df < 2) and too-common (df > 0.1 * N).
    // Keeps the bigram index small while still catching meaningful compounds.
    let bg_min_df: usize = 2;
    let bg_max_df: usize = (0.1 * n as f64).max(2.0) as usize;
    bigram_index.retain(|_, v| {
        let len = v.len();
        len >= bg_min_df && len <= bg_max_df
    });
    // Sort kept bigram-postings by doc id so binary search works at query time.
    for v in bigram_index.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    let n_bigrams = bigram_index.len();
    let bg_nnz: usize = bigram_index.values().map(|v| v.len()).sum();
    println!("      bigrams: {n_bigrams} kept (df∈[{bg_min_df}, {bg_max_df}]), {bg_nnz} postings");
    let total_tokens: f32 = doc_lens.iter().sum();
    let avgdl = if n > 0 { total_tokens / n as f32 } else { 1.0 };
    println!(
        "      single-pass df+inverted+doc_lens {:.2}s",
        t0.elapsed().as_secs_f64()
    );
    println!("[3/6] stopwords applied");

    let mut idf = vec![0f32; vocab_size];
    let n_f = n as f32;
    for t in 0..vocab_size {
        if df[t] > 0 {
            let c = df[t] as f32;
            idf[t] = ((n_f - c + 0.5) / (c + 0.5) + 1.0).ln();
        }
    }
    // Damp WordPiece continuation fragments: `##bits` matching across docs is
    // mostly noise relative to whole-word matches. Scaling idf here bakes the
    // damping into everything downstream (scoring, WAND bounds, expansion).
    if (args.subword_weight - 1.0).abs() > f32::EPSILON {
        let vocab = tokenizer.get_vocab(true);
        let mut n_scaled = 0u64;
        for (tok, id) in vocab.iter() {
            if tok.starts_with("##") {
                let i = *id as usize;
                if i < idf.len() && idf[i] > 0.0 {
                    idf[i] *= args.subword_weight;
                    n_scaled += 1;
                }
            }
        }
        println!(
            "      subword idf x{}: {n_scaled} continuation tokens damped",
            args.subword_weight
        );
    }
    let active_ids: Vec<u32> = (0..vocab_size as u32)
        .filter(|&t| df[t as usize] > 0)
        .collect();
    let n_active = active_ids.len();
    println!("      active vocab: {n_active}  avgdl={avgdl:.1}");

    // Build the EXACT CSR matrix (pre-expansion). Each row is the original
    // inverted-index postings for that term, sorted by doc id. Used at query
    // time to split bm25_combined into bm25_exact + bm25_semantic, which are
    // input features for the learned reranker.
    let t_exact = Instant::now();
    let build_exact_row = |&term: &u32| {
        let mut postings: Vec<(u32, u32)> = inverted[term as usize].clone();
        postings.sort_unstable_by_key(|&(d, _)| d);
        let indices: Vec<u32> = postings.iter().map(|&(d, _)| d).collect();
        let data: Vec<f32> = postings.iter().map(|&(_, tf)| tf as f32).collect();
        (term, indices, data)
    };
    #[cfg(feature = "multithread")]
    let exact_rows: Vec<(u32, Vec<u32>, Vec<f32>)> =
        active_ids.par_iter().map(build_exact_row).collect();
    #[cfg(not(feature = "multithread"))]
    let exact_rows: Vec<(u32, Vec<u32>, Vec<f32>)> =
        active_ids.iter().map(build_exact_row).collect();
    let mut exact_row_sizes = vec![0u64; vocab_size];
    for (term, idx, _) in &exact_rows {
        exact_row_sizes[*term as usize] = idx.len() as u64;
    }
    let mut exact_indptr = vec![0u64; vocab_size + 1];
    for r in 0..vocab_size {
        exact_indptr[r + 1] = exact_indptr[r] + exact_row_sizes[r];
    }
    let exact_nnz = exact_indptr[vocab_size] as usize;
    let mut exact_indices = vec![0u32; exact_nnz];
    let mut exact_data = vec![0f32; exact_nnz];
    for (term, idx, dat) in exact_rows {
        let off = exact_indptr[term as usize] as usize;
        let len = idx.len();
        exact_indices[off..off + len].copy_from_slice(&idx);
        exact_data[off..off + len].copy_from_slice(&dat);
    }
    // When shipping f16 weights, quantize in memory now so everything
    // derived from these values downstream (forward index, block-max
    // bounds) is computed over exactly what queries will read.
    if args.f16_postings {
        for x in exact_data.iter_mut() {
            *x = f16::from_f32(*x).to_f32();
        }
    }
    println!(
        "      exact-CSR (pre-expansion) {:.2}s  nnz={}",
        t_exact.elapsed().as_secs_f64(),
        exact_nnz
    );

    // Optional forward index: transpose of the exact CSR. Doc-major
    // (doc_id -> [(term_id, tf)]). Required for PRF / Rocchio expansion.
    let (fwd_indptr, fwd_indices, fwd_data): (Vec<u64>, Vec<u32>, Vec<f32>) = if args.forward {
        let t_fwd = Instant::now();
        let mut counts = vec![0u64; n + 1];
        for &d in &exact_indices {
            counts[d as usize + 1] += 1;
        }
        for i in 1..=n {
            counts[i] += counts[i - 1];
        }
        let total = counts[n] as usize;
        let mut fwd_indices = vec![0u32; total];
        let mut fwd_data = vec![0f32; total];
        let mut cursor = counts[..n].to_vec();
        for term in 0..vocab_size {
            let lo = exact_indptr[term] as usize;
            let hi = exact_indptr[term + 1] as usize;
            for j in lo..hi {
                let d = exact_indices[j] as usize;
                let off = cursor[d] as usize;
                fwd_indices[off] = term as u32;
                fwd_data[off] = exact_data[j];
                cursor[d] += 1;
            }
        }
        println!(
            "      forward index {:.2}s  nnz={}",
            t_fwd.elapsed().as_secs_f64(),
            total
        );
        (counts, fwd_indices, fwd_data)
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };

    println!("[4/6] computing neighbour graph over {n_active} active subwords");
    let t0 = Instant::now();
    // Flat embedding storage: emb_dim rows packed contiguously.
    // active_emb_flat[i * emb_dim .. (i+1) * emb_dim] = embedding for active_ids[i].
    let active_emb_flat: Vec<f32> = active_ids
        .iter()
        .flat_map(|&tid| embeddings[tid as usize].iter().copied())
        .collect();

    // Brute-force exact top-K by cosine similarity. Embeddings are L2-normalized
    // upstream so cosine = dot product. n_active rarely exceeds 50k for BEIR-size
    // corpora; brute-force at that size is dominated by autovectorized dot
    // products and out-runs HNSW build+search wall-clock by 5-10× on
    // multi-socket machines. For very large vocabularies the algorithm is still
    // O(n²) so future work can re-introduce HNSW above some threshold.
    let k_query = (args.k_expand + 1).min(n_active);
    let neighbour_lists: Vec<Vec<(usize, f32)>> = if args.quantize_embeddings {
        // Per-vector i8 quantization: encode v[k] as round(v[k] * 127 / max_abs)
        // with the per-vector scale = max_abs / 127. Dot(a, b) is then
        //   sum(i8_a[k] * i8_b[k]) * (scale_a * scale_b).
        // 4× smaller working set → far less DRAM bandwidth.
        let mut q: Vec<i8> = vec![0i8; n_active * emb_dim];
        let mut scales: Vec<f32> = Vec::with_capacity(n_active);
        for i in 0..n_active {
            let row = &active_emb_flat[i * emb_dim..(i + 1) * emb_dim];
            let max_abs = row.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-9);
            let s = max_abs / 127.0;
            scales.push(s);
            for k in 0..emb_dim {
                let v = (row[k] / s).round();
                q[i * emb_dim + k] = v.clamp(-127.0, 127.0) as i8;
            }
        }
        let q = q;
        let scales = scales;
        let topk = |i: usize| -> Vec<(usize, f32)> {
            let qv = &q[i * emb_dim..(i + 1) * emb_dim];
            let qs = scales[i];
            let mut top: Vec<(usize, f32)> = Vec::with_capacity(k_query);
            for j in 0..n_active {
                let dv = &q[j * emb_dim..(j + 1) * emb_dim];
                let mut acc: i32 = 0;
                for k in 0..emb_dim {
                    acc += (qv[k] as i32) * (dv[k] as i32);
                }
                let s = acc as f32 * qs * scales[j];
                if top.len() < k_query {
                    top.push((j, s));
                    if top.len() == k_query {
                        top.sort_unstable_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                } else if s > top[k_query - 1].1 {
                    top[k_query - 1] = (j, s);
                    let mut k = k_query - 1;
                    while k > 0 && top[k].1 > top[k - 1].1 {
                        top.swap(k, k - 1);
                        k -= 1;
                    }
                }
            }
            top
        };
        #[cfg(feature = "multithread")]
        {
            (0..n_active).into_par_iter().map(topk).collect()
        }
        #[cfg(not(feature = "multithread"))]
        {
            (0..n_active).map(topk).collect()
        }
    } else {
        let topk = |i: usize| -> Vec<(usize, f32)> {
            let qv = &active_emb_flat[i * emb_dim..(i + 1) * emb_dim];
            let mut top: Vec<(usize, f32)> = Vec::with_capacity(k_query);
            for j in 0..n_active {
                let dv = &active_emb_flat[j * emb_dim..(j + 1) * emb_dim];
                let mut s = 0.0f32;
                for k in 0..emb_dim {
                    s += qv[k] * dv[k];
                }
                if top.len() < k_query {
                    top.push((j, s));
                    if top.len() == k_query {
                        top.sort_unstable_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                } else if s > top[k_query - 1].1 {
                    top[k_query - 1] = (j, s);
                    let mut k = k_query - 1;
                    while k > 0 && top[k].1 > top[k - 1].1 {
                        top.swap(k, k - 1);
                        k -= 1;
                    }
                }
            }
            top
        };
        #[cfg(feature = "multithread")]
        {
            (0..n_active).into_par_iter().map(topk).collect()
        }
        #[cfg(not(feature = "multithread"))]
        {
            (0..n_active).map(topk).collect()
        }
    };
    println!("      brute top-k {:.2}s", t0.elapsed().as_secs_f64());

    println!(
        "[5/6] index-time expansion (k={}, sim>={})  [per-row]",
        args.k_expand, args.threshold
    );
    let t0 = Instant::now();

    // Serialize the forward neighbor graph as the query-expansion sidecar:
    // term -> its top-K neighbors with sims. At query time this lets sift
    // expand the query by table lookup (no model), covering the asymmetric
    // half of the top-K graph that doc-side expansion structurally misses
    // (q's neighbor j where q is not among j's top-K).
    {
        let mut qexp_indptr = vec![0u64; vocab_size + 1];
        let mut qexp_terms: Vec<u32> = Vec::new();
        let mut qexp_sims: Vec<f32> = Vec::new();
        let mut per_term: Vec<Vec<(u32, f32)>> = vec![Vec::new(); vocab_size];
        for (i, neighbours) in neighbour_lists.iter().enumerate() {
            let source_vocab = active_ids[i] as usize;
            for &(j, sim) in neighbours {
                if sim < args.threshold {
                    continue;
                }
                let target_vocab = active_ids[j];
                if target_vocab as usize == source_vocab {
                    continue;
                }
                per_term[source_vocab].push((target_vocab, sim));
            }
        }
        for (t, row) in per_term.into_iter().enumerate() {
            for (j, sim) in row {
                qexp_terms.push(j);
                qexp_sims.push(sim);
            }
            qexp_indptr[t + 1] = qexp_terms.len() as u64;
        }
        write_bin(out.join("qexp_indptr.bin"), &qexp_indptr)?;
        write_bin(out.join("qexp_terms.bin"), &qexp_terms)?;
        write_bin(out.join("qexp_sims.bin"), &qexp_sims)?;
        println!("      query-expansion sidecar: {} edges", qexp_terms.len());
    }

    // Invert the neighbor graph: for each target vocab id n, list of (source_vocab, sim)
    // such that source has n as one of its top-K neighbors with sim >= threshold.
    // We store this sparse: only fill rows for active vocab ids.
    let mut neighbor_inverse: Vec<Vec<(u32, f32)>> = vec![Vec::new(); vocab_size];
    for (i, neighbours) in neighbour_lists.iter().enumerate() {
        let source_vocab = active_ids[i];
        for &(j, sim) in neighbours {
            if sim < args.threshold {
                continue;
            }
            let target_vocab = active_ids[j];
            if target_vocab == source_vocab {
                continue;
            }
            neighbor_inverse[target_vocab as usize].push((source_vocab, sim));
        }
    }
    drop(neighbour_lists);

    // Merge corpus-fitted PPMI edges into the neighbour graph. corpus_edges[t]
    // holds t's associates; an edge (t, assoc, w) means a query for `t` should
    // also retrieve docs containing `assoc`, i.e. neighbor_inverse[t] gains
    // (assoc, w). The PPMI pass already emitted both directions, so symmetry
    // is preserved. These sum with the embedding edges in build_row.
    if !corpus_edges.is_empty() {
        for (t, assocs) in corpus_edges.iter().enumerate() {
            for &(assoc, w) in assocs {
                if assoc as usize != t {
                    neighbor_inverse[t].push((assoc, w));
                }
            }
        }
    }
    println!("      inverse graph {:.2}s", t0.elapsed().as_secs_f64());

    // Per-row construction. For each active target term, accumulate exact postings +
    // semantic contributions into an AHashMap<doc_id, weight>, then sort by doc id and
    // emit (indices, data). All target rows are built independently → trivial rayon.
    let t_rows = Instant::now();
    let build_row = |target_vocab: u32| -> (u32, Vec<u32>, Vec<f32>) {
        let mut accum: ahash::AHashMap<u32, f32> = ahash::AHashMap::default();
        // exact postings
        for &(doc, tf) in &inverted[target_vocab as usize] {
            accum.insert(doc, tf as f32);
        }
        // semantic contributions
        for &(source_vocab, sim) in &neighbor_inverse[target_vocab as usize] {
            for &(doc, tf) in &inverted[source_vocab as usize] {
                *accum.entry(doc).or_insert(0.0) += tf as f32 * sim;
            }
        }
        let mut entries: Vec<(u32, f32)> = accum.into_iter().collect();
        entries.sort_unstable_by_key(|&(d, _)| d);
        let mut indices = Vec::with_capacity(entries.len());
        let mut data = Vec::with_capacity(entries.len());
        for (d, w) in entries {
            indices.push(d);
            data.push(w);
        }
        (target_vocab, indices, data)
    };
    #[cfg(feature = "multithread")]
    let row_results: Vec<(u32, Vec<u32>, Vec<f32>)> =
        active_ids.par_iter().map(|&t| build_row(t)).collect();
    #[cfg(not(feature = "multithread"))]
    let row_results: Vec<(u32, Vec<u32>, Vec<f32>)> =
        active_ids.iter().map(|&t| build_row(t)).collect();
    println!("      per-row build {:.2}s", t_rows.elapsed().as_secs_f64());

    // Assemble CSR. indptr[i+1] - indptr[i] = row i size; row data lives at indptr[i]..
    let mut row_sizes = vec![0u64; vocab_size];
    for (target, idx, _) in &row_results {
        row_sizes[*target as usize] = idx.len() as u64;
    }
    let mut indptr = vec![0u64; vocab_size + 1];
    for r in 0..vocab_size {
        indptr[r + 1] = indptr[r] + row_sizes[r];
    }
    let nnz = indptr[vocab_size] as usize;
    let mut indices = vec![0u32; nnz];
    let mut data = vec![0f32; nnz];
    for (target, idx, dat) in row_results {
        let off = indptr[target as usize] as usize;
        let len = idx.len();
        indices[off..off + len].copy_from_slice(&idx);
        data[off..off + len].copy_from_slice(&dat);
    }
    // Quantize before block-max so the precomputed upper bounds hold
    // exactly for the f16 weights queries will score against.
    if args.f16_postings {
        for x in data.iter_mut() {
            *x = f16::from_f32(*x).to_f32();
        }
    }
    println!(
        "      expansion+CSR {:.2}s  nnz={nnz}",
        t0.elapsed().as_secs_f64()
    );

    // Optional: precompute per-block max BM25 contribution. For each term's
    // posting list, group every 128 consecutive postings into a block and
    // store the maximum idf * stf any doc in the block can produce.
    // Block-Max WAND uses these as tighter per-cursor upper bounds.
    const BLOCK_SIZE: usize = 128;
    let (block_max_indptr, block_max_data): (Vec<u64>, Vec<f32>) = if args.block_max {
        let t_bm = Instant::now();
        // Per-term IDF for the final (expanded) postings.
        let n_f = n as f32;
        let term_idf = |t: usize| -> f32 {
            // Reuse the idf vector computed earlier; falls back to 0 for stopwords.
            idf[t]
        };
        // BM25 saturation constants from the build args.
        let k1 = args.bm25_k1;
        let b_param = args.bm25_b;
        let delta = args.bm25_delta;
        let avgdl_v = if avgdl > 0.0 { avgdl } else { 1.0 };

        let mut bm_indptr = vec![0u64; vocab_size + 1];
        let mut bm_data: Vec<f32> = Vec::new();
        let mut total_blocks = 0u64;
        for t in 0..vocab_size {
            let lo = indptr[t] as usize;
            let hi = indptr[t + 1] as usize;
            let term_len = hi - lo;
            if term_len == 0 || term_idf(t) <= 0.0 {
                bm_indptr[t + 1] = total_blocks;
                continue;
            }
            let n_blocks = term_len.div_ceil(BLOCK_SIZE);
            let idf_t = term_idf(t);
            for blk in 0..n_blocks {
                let b_lo = lo + blk * BLOCK_SIZE;
                let b_hi = (b_lo + BLOCK_SIZE).min(hi);
                let mut bmax = 0.0f32;
                for k in b_lo..b_hi {
                    let wtf = data[k];
                    let doc = indices[k] as usize;
                    let dl = doc_lens[doc];
                    let denom = wtf + k1 * (1.0 - b_param + b_param * dl / avgdl_v);
                    let stf = (wtf * (k1 + 1.0)) / denom + delta;
                    let c = idf_t * stf;
                    if c > bmax {
                        bmax = c;
                    }
                }
                bm_data.push(bmax);
            }
            total_blocks += n_blocks as u64;
            bm_indptr[t + 1] = total_blocks;
        }
        let _ = n_f;
        println!(
            "      block-max {:.2}s  blocks={total_blocks}",
            t_bm.elapsed().as_secs_f64()
        );
        (bm_indptr, bm_data)
    } else {
        (Vec::new(), Vec::new())
    };

    // Serialize the bigram inverted index. Sort by hash for binary-search lookup.
    let mut sorted_keys: Vec<u64> = bigram_index.keys().copied().collect();
    sorted_keys.sort_unstable();
    let mut bg_indptr = vec![0u64; sorted_keys.len() + 1];
    let mut bg_indices: Vec<u32> = Vec::with_capacity(bg_nnz);
    let mut bg_idf: Vec<f32> = Vec::with_capacity(sorted_keys.len());
    let n_f = n as f32;
    for (i, h) in sorted_keys.iter().enumerate() {
        let docs = &bigram_index[h];
        bg_indices.extend_from_slice(docs);
        bg_indptr[i + 1] = bg_indices.len() as u64;
        let c = docs.len() as f32;
        bg_idf.push(((n_f - c + 0.5) / (c + 0.5) + 1.0).ln());
    }

    // Optional: extract per-doc custom-rank attributes from the JSONL.
    // Each named field becomes an f32 array of length n_docs written as
    // rank_<name>.bin. The set of available fields is recorded in
    // rank_fields.json so the server knows what's loadable.
    let rank_fields: Vec<String> = args
        .rank_fields
        .iter()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    let rank_values: Vec<Vec<f32>> = if !rank_fields.is_empty() {
        let t_rank = Instant::now();
        let values = extract_rank_fields(&args.input, &args.format, &rank_fields, n)?;
        println!(
            "      rank attributes: {} field(s) over {} docs ({:.2}s)",
            rank_fields.len(),
            n,
            t_rank.elapsed().as_secs_f64()
        );
        values
    } else {
        Vec::new()
    };

    // Optional: exact-content dedup. ahash the normalized doc text; in each
    // cluster of identical hashes, the lowest-index doc is canonical (1) and
    // the rest are non-canonical (0). The server can drop non-canonical docs
    // from results on request.
    let dedup_mask: Vec<u8> = if args.dedup {
        let t_dedup = Instant::now();
        let texts = reread_doc_texts(&args.input, &args.format, args.no_normalize, &args.clean)?;
        if texts.len() != n {
            anyhow::bail!(
                "dedup: re-read produced {} texts but corpus has {} docs",
                texts.len(),
                n
            );
        }
        let mut hashes: Vec<u64> = Vec::with_capacity(n);
        for t in &texts {
            use std::hash::Hasher;
            let mut h = ahash::AHasher::default();
            h.write(t.as_bytes());
            hashes.push(h.finish());
        }
        let mut seen: ahash::AHashMap<u64, usize> = ahash::AHashMap::default();
        let mut mask = vec![1u8; n];
        let mut dup_count = 0u64;
        for (i, &h) in hashes.iter().enumerate() {
            match seen.get(&h) {
                Some(&_canon) => {
                    mask[i] = 0;
                    dup_count += 1;
                }
                None => {
                    seen.insert(h, i);
                }
            }
        }
        println!(
            "      dedup mask: {dup_count} non-canonical docs flagged ({:.2}s)",
            t_dedup.elapsed().as_secs_f64()
        );
        mask
    } else {
        Vec::new()
    };

    // Optional: build the spell-correction word vocab + deletion table.
    // Re-reads the input file in a stripped-down pass so the chunk loop stays clean.
    if args.spell {
        let t_spell = Instant::now();
        let texts = reread_doc_texts(&args.input, &args.format, args.no_normalize, &args.clean)?;
        let extra: Vec<String> = if let Some(p) = &args.spell_dictionary {
            let raw = std::fs::read_to_string(p)
                .with_context(|| format!("reading spell dictionary {}", p.display()))?;
            raw.lines()
                .map(|l| l.trim().to_lowercase())
                .filter(|l| {
                    !l.is_empty()
                        && l.chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '\'')
                })
                .collect()
        } else {
            Vec::new()
        };
        if !extra.is_empty() {
            println!(
                "      dictionary augmentation: {} words from {}",
                extra.len(),
                args.spell_dictionary.as_ref().unwrap().display()
            );
        }
        let table = crate::spell::build_spell_table(&texts, args.spell_min_df, 40, &extra);
        let (nw, nd, np) = crate::spell::write_spell_table(&out, &table)?;
        println!(
            "      spell table: {nw} words / {nd} deletions / {np} postings ({:.2}s)",
            t_spell.elapsed().as_secs_f64()
        );
    }

    println!("[6/6] writing artifact to {}", out.display());
    let t0 = Instant::now();
    write_bin(out.join("indptr.bin"), &indptr)?;
    if args.u24_indices && (n as u64) <= sift_core::U24_MAX as u64 + 1 {
        write_bin(out.join("indices.bin"), &sift_core::pack_u24(&indices))?;
    } else {
        write_bin(out.join("indices.bin"), &indices)?;
    }
    if args.f16_postings {
        let half: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
        write_bin(out.join("data_f16.bin"), &half)?;
    } else {
        write_bin(out.join("data.bin"), &data)?;
    }
    write_bin(out.join("exact_indptr.bin"), &exact_indptr)?;
    write_bin(out.join("exact_indices.bin"), &exact_indices)?;
    if args.f16_postings {
        let half: Vec<f16> = exact_data.iter().map(|&x| f16::from_f32(x)).collect();
        write_bin(out.join("exact_data_f16.bin"), &half)?;
    } else {
        write_bin(out.join("exact_data.bin"), &exact_data)?;
    }
    if args.forward {
        write_bin(out.join("fwd_indptr.bin"), &fwd_indptr)?;
        write_bin(out.join("fwd_indices.bin"), &fwd_indices)?;
        write_bin(out.join("fwd_data.bin"), &fwd_data)?;
    }
    if args.positions {
        write_bin(out.join("pos_indptr.bin"), &pos_indptr)?;
        write_bin(out.join("pos_terms.bin"), &pos_terms)?;
    }
    if args.block_max {
        write_bin(out.join("block_max_indptr.bin"), &block_max_indptr)?;
        write_bin(out.join("block_max.bin"), &block_max_data)?;
    }
    if args.dedup {
        write_bin(out.join("dedup_canonical.bin"), &dedup_mask)?;
    }
    if !rank_fields.is_empty() {
        for (field, values) in rank_fields.iter().zip(rank_values.iter()) {
            let safe = sanitize_field_name(field);
            write_bin(out.join(format!("rank_{safe}.bin")), values)?;
        }
        let manifest = serde_json::json!({ "fields": rank_fields });
        fs::write(
            out.join("rank_fields.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;
    }
    write_bin(out.join("bigram_keys.bin"), &sorted_keys)?;
    write_bin(out.join("bigram_indptr.bin"), &bg_indptr)?;
    write_bin(out.join("bigram_indices.bin"), &bg_indices)?;
    write_bin(out.join("bigram_idf.bin"), &bg_idf)?;
    write_bin(out.join("idf.bin"), &idf)?;
    write_bin(out.join("vocab_keep.bin"), keep_mask)?;
    write_bin(out.join("doc_lens.bin"), &doc_lens)?;

    // packed doc ids
    let (text_buf, off_buf) = pack_strings(&doc_ids);
    fs::write(out.join("doc_ids_text.bin"), &text_buf)?;
    write_bin(out.join("doc_ids_off.bin"), &off_buf)?;

    // packed doc snippets (optional)
    if args.no_snippets {
        // still need files for serve to load
        fs::write(out.join("doc_snips_text.bin"), [] as [u8; 0])?;
        let zeros = vec![0u64; n + 1];
        write_bin(out.join("doc_snips_off.bin"), &zeros)?;
    } else {
        let (sbuf, soff) = pack_strings(&doc_snips);
        fs::write(out.join("doc_snips_text.bin"), &sbuf)?;
        write_bin(out.join("doc_snips_off.bin"), &soff)?;
    }

    // per-doc JSON payload (stored source), unless disabled
    if !args.no_payload {
        let payloads = reread_payloads(&args.input)?;
        if payloads.len() != n {
            return Err(anyhow!(
                "payload count {} != doc count {n} (source/doc order mismatch)",
                payloads.len()
            ));
        }
        let (pbuf, poff) = pack_strings(&payloads);
        fs::write(out.join("payload_text.bin"), &pbuf)?;
        write_bin(out.join("payload_off.bin"), &poff)?;
    }

    // tokenizer.json (copy file we already have on disk)
    fs::copy(tokenizer_path, out.join("tokenizer.json")).context("copying tokenizer.json")?;
    // Flush the whole segment to stable storage so a write that gets ack'd
    // (after the manifest commit that references this segment) is durable.
    sift_core::fsync_dir_contents(&out).context("fsync segment")?;
    println!("      write {:.2}s", t0.elapsed().as_secs_f64());

    let meta = Meta {
        schema_version: SCHEMA_VERSION,
        model_name: args.model.clone(),
        vocab_size: vocab_size as u64,
        n_docs: n as u64,
        n_nonzero: nnz as u64,
        n_nonzero_exact: exact_nnz as u64,
        n_active_terms: n_active as u64,
        n_stopwords: n_stops,
        avgdl,
        bm25_k1: args.bm25_k1,
        bm25_b: args.bm25_b,
        bm25_delta: args.bm25_delta,
        k_expand: args.k_expand as u32,
        sim_threshold: args.threshold,
        build_seconds: t_total.elapsed().as_secs_f64(),
        indices_packing: if args.u24_indices && (n as u64) <= sift_core::U24_MAX as u64 + 1 {
            "u24".to_string()
        } else {
            "u32".to_string()
        },
        subword_weight: args.subword_weight,
    };
    fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;

    println!(
        "      done: {n} docs, {nnz} postings, total {:.1}s",
        t_total.elapsed().as_secs_f64()
    );
    Ok(())
}

// Helpers are split out to keep the build pipeline readable as six phases.
include!("build/support.rs");
