//! Read and write the existing manifest build recipe.

use super::BuildArgs;
use anyhow::{anyhow, Result};
use std::path::Path;

/// Reconstruct the `sift build` flags (minus `--input`/`--out`) for a
/// BuildArgs, emitting only non-default values. Used to record how a segment
/// was built so `compact` can reproduce it.
pub(crate) fn build_args_to_argv(b: &BuildArgs) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    macro_rules! push {
        ($k:expr, $v:expr) => {{
            a.push($k.to_string());
            a.push($v);
        }};
    }
    if b.format != "jsonl" {
        push!("--format", b.format.clone());
    }
    if b.model != "minishlab/potion-base-8M" {
        push!("--model", b.model.clone());
    }
    if b.k_expand != 10 {
        push!("--k-expand", b.k_expand.to_string());
    }
    if (b.threshold - 0.65).abs() > f32::EPSILON {
        push!("--threshold", b.threshold.to_string());
    }
    if (b.stop_df - 0.4).abs() > f32::EPSILON {
        push!("--stop-df", b.stop_df.to_string());
    }
    if b.snippet_chars != 320 {
        push!("--snippet-chars", b.snippet_chars.to_string());
    }
    if b.no_snippets {
        a.push("--no-snippets".into());
    }
    if (b.title_weight - 2.0).abs() > f32::EPSILON {
        push!("--title-weight", b.title_weight.to_string());
    }
    if (b.subword_weight - 0.4).abs() > f32::EPSILON {
        push!("--subword-weight", b.subword_weight.to_string());
    }
    if b.corpus_expand_weight != 0.0 {
        push!("--corpus-expand-weight", b.corpus_expand_weight.to_string());
        if b.corpus_window != 5 {
            push!("--corpus-window", b.corpus_window.to_string());
        }
        if b.corpus_expand_k != 5 {
            push!("--corpus-expand-k", b.corpus_expand_k.to_string());
        }
        if b.corpus_min_cooc != 3 {
            push!("--corpus-min-cooc", b.corpus_min_cooc.to_string());
        }
    }
    if (b.bm25_k1 - 1.5).abs() > f32::EPSILON {
        push!("--bm25-k1", b.bm25_k1.to_string());
    }
    if (b.bm25_b - 0.75).abs() > f32::EPSILON {
        push!("--bm25-b", b.bm25_b.to_string());
    }
    if b.bm25_delta.abs() > f32::EPSILON {
        push!("--bm25-delta", b.bm25_delta.to_string());
    }
    if b.no_normalize {
        a.push("--no-normalize".into());
    }
    if b.no_bigrams {
        a.push("--no-bigrams".into());
    }
    if b.spell {
        a.push("--spell".into());
    }
    if b.spell_min_df != 2 {
        push!("--spell-min-df", b.spell_min_df.to_string());
    }
    if let Some(d) = &b.spell_dictionary {
        push!("--spell-dictionary", d.display().to_string());
    }
    if b.forward {
        a.push("--forward".into());
    }
    if b.clean != "off" {
        push!("--clean", b.clean.clone());
    }
    let rank_fields: Vec<&String> = b.rank_fields.iter().filter(|s| !s.is_empty()).collect();
    if !rank_fields.is_empty() {
        let joined = rank_fields
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(",");
        push!("--rank-fields", joined);
    }
    if b.positions {
        a.push("--positions".into());
    }
    if b.compositional {
        a.push("--compositional".into());
    }
    if b.u24_indices {
        a.push("--u24-indices".into());
    }
    if b.block_max {
        a.push("--block-max".into());
    }
    if b.quantize_embeddings {
        a.push("--quantize-embeddings".into());
    }
    if b.dedup {
        a.push("--dedup".into());
    }
    if !b.f16_postings {
        // f16 is the default; only the opt-out needs recording. Older
        // manifests may carry a bare `--f16-postings`, which still parses.
        push!("--f16-postings", "false".to_string());
    }
    if b.no_payload {
        a.push("--no-payload".into());
    }
    a
}

/// Re-parse a recorded build argv into a `BuildArgs` for a given input/out.
pub(crate) fn build_args_from(input: &Path, out: &Path, argv: &[String]) -> Result<BuildArgs> {
    use clap::Parser;
    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        b: BuildArgs,
    }
    let mut full: Vec<String> = vec![
        "sift-build".to_string(),
        "--input".to_string(),
        input.display().to_string(),
        "--out".to_string(),
        out.display().to_string(),
    ];
    full.extend(argv.iter().cloned());
    let w = Wrap::try_parse_from(&full)
        .map_err(|e| anyhow!("reconstructing build args from manifest: {e}"))?;
    Ok(w.b)
}
