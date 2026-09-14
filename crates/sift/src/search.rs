//! `sift search` - one-shot CLI search against a single `.sift` index.

use crate::query::SearchSyntax;
use crate::{Engine, SearchOptions};
use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Path to a `.sift` artifact directory.
    #[arg(value_name = "INDEX_DIR")]
    pub index: PathBuf,
    /// Query string.
    pub query: String,
    /// Top-K to return.
    #[arg(short = 'k', long, default_value_t = 10)]
    pub k: usize,
    /// Semantic expansion weight. 0 is exact BM25, 1 is the fully expanded
    /// index, and the default balances exact matches with semantic recall.
    #[arg(long, default_value_t = 0.5, value_parser = parse_semantic_weight)]
    pub semantic_weight: f32,
    /// Weight of the optional order-aware composition rerank. Requires an
    /// artifact built with `--compositional`. 0 disables.
    #[arg(long, default_value_t = 0.0, value_parser = parse_composition_weight)]
    pub composition_weight: f32,
    /// Show snippets in output.
    #[arg(long, default_value_t = true)]
    pub snippets: bool,
}

fn parse_semantic_weight(raw: &str) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|_| "semantic weight must be a number in [0, 1]".to_string())?;
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("semantic weight must be a number in [0, 1]".to_string())
    }
}

fn parse_composition_weight(raw: &str) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|_| "composition weight must be a number in [0, 1]".to_string())?;
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("composition weight must be a number in [0, 1]".to_string())
    }
}

pub fn run(args: SearchArgs) -> Result<()> {
    let engine =
        Engine::open(&args.index).with_context(|| format!("opening {}", args.index.display()))?;
    let single = engine.entry.set.is_single();
    if single && engine.entry.idx().tokenize_query(&args.query).is_empty() {
        eprintln!("(no content tokens in query)");
        return Ok(());
    }
    let result = engine.search(SearchOptions {
        q: args.query,
        k: args.k,
        syntax: SearchSyntax::Terms,
        blend_alpha: args.semantic_weight,
        composition_weight: args.composition_weight,
        ..SearchOptions::default()
    })?;
    if single {
        println!(
            "# {} hits  matched_terms={}  latency={}µs",
            result.hits.len(),
            result.matched_terms,
            result.latency_us
        );
    } else {
        println!(
            "# {} hits  matched_terms={}  latency={}µs  ({} segments)",
            result.hits.len(),
            result.matched_terms,
            result.latency_us,
            engine.entry.set.n_segments()
        );
    }
    for hit in result.hits {
        if args.snippets {
            println!(
                "{:>8.3}  {}  {}",
                hit.score,
                hit.doc_id,
                truncate(&hit.snippet, 200)
            );
        } else {
            println!("{:>8.3}  {}", hit.score, hit.doc_id);
        }
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_composition_weight, parse_semantic_weight};

    #[test]
    fn semantic_weight_stays_in_unit_interval() {
        assert_eq!(parse_semantic_weight("0").unwrap(), 0.0);
        assert_eq!(parse_semantic_weight("0.5").unwrap(), 0.5);
        assert_eq!(parse_semantic_weight("1").unwrap(), 1.0);
        assert!(parse_semantic_weight("-0.1").is_err());
        assert!(parse_semantic_weight("1.1").is_err());
        assert!(parse_semantic_weight("NaN").is_err());
    }

    #[test]
    fn composition_weight_stays_in_unit_interval() {
        assert_eq!(parse_composition_weight("0").unwrap(), 0.0);
        assert_eq!(parse_composition_weight("0.7").unwrap(), 0.7);
        assert_eq!(parse_composition_weight("1").unwrap(), 1.0);
        assert!(parse_composition_weight("-0.1").is_err());
        assert!(parse_composition_weight("1.1").is_err());
        assert!(parse_composition_weight("NaN").is_err());
    }
}
