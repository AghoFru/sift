//! `sift search` - one-shot CLI search against a single `.sift` index.

use anyhow::{Context, Result};
use clap::Args;
use sift_core::{IndexSet, ScoreMode};
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

pub fn run(args: SearchArgs) -> Result<()> {
    let set =
        IndexSet::open(&args.index).with_context(|| format!("opening {}", args.index.display()))?;

    // Single-segment: score the one artifact directly. Multi-segment: merge.
    if set.is_single() {
        let idx = set.primary();
        let tokens = idx.tokenize_query(&args.query);
        if tokens.is_empty() {
            eprintln!("(no content tokens in query)");
            return Ok(());
        }
        let r = idx.score_blended(&tokens, args.k, args.semantic_weight);
        println!(
            "# {} hits  matched_terms={}  latency={}µs",
            r.hits.len(),
            r.matched_query_terms,
            r.elapsed_us
        );
        for h in &r.hits {
            let id = idx.doc_id(h.doc_idx as usize);
            if args.snippets {
                let snip = idx.doc_snip(h.doc_idx as usize);
                println!("{:>8.3}  {}  {}", h.score, id, truncate(snip, 200));
            } else {
                println!("{:>8.3}  {}", h.score, id);
            }
        }
    } else {
        let r = set.search_merged(&args.query, args.k, ScoreMode::Plain, args.semantic_weight);
        println!(
            "# {} hits  matched_terms={}  latency={}µs  ({} segments)",
            r.hits.len(),
            r.matched_query_terms,
            r.elapsed_us,
            set.n_segments()
        );
        for h in &r.hits {
            if args.snippets {
                println!(
                    "{:>8.3}  {}  {}",
                    h.score,
                    h.doc_id,
                    truncate(&h.snippet, 200)
                );
            } else {
                println!("{:>8.3}  {}", h.score, h.doc_id);
            }
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
    use super::parse_semantic_weight;

    #[test]
    fn semantic_weight_stays_in_unit_interval() {
        assert_eq!(parse_semantic_weight("0").unwrap(), 0.0);
        assert_eq!(parse_semantic_weight("0.5").unwrap(), 0.5);
        assert_eq!(parse_semantic_weight("1").unwrap(), 1.0);
        assert!(parse_semantic_weight("-0.1").is_err());
        assert!(parse_semantic_weight("1.1").is_err());
        assert!(parse_semantic_weight("NaN").is_err());
    }
}
