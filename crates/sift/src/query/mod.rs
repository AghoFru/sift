//! Search request validation and ranking, independent of HTTP transport.

use crate::engine::{CachedSearch, IndexEntry, SearchContext, SlowEntry};
use crate::filters::{passes_filters, FilterClause};
use crate::query_negation::NegationParts;
use serde::{Deserialize, Serialize};
use sift_core::{Index, ScoreMode};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::Instant;
mod terms;

include!("request.rs");
include!("response.rs");
include!("error.rs");

/// Add high-confidence static neighbors of natural-language negative terms.
/// The exact seed is always kept. Only whole-word neighbors above this
/// threshold are added, because a broad semantic exclusion can hide valid
/// documents.
fn add_natural_exclusion_neighbors(index: &Index, seeds: &[u32], excluded: &mut Vec<u32>) {
    const MIN_SIMILARITY: f32 = 0.88;
    const MAX_NEIGHBORS_PER_TERM: usize = 4;
    let mut seen: std::collections::HashSet<u32> = excluded.iter().copied().collect();
    for &seed in seeds {
        if let Some((neighbors, similarities)) = index.qexp_neighbors(seed) {
            let mut added = 0usize;
            for (&neighbor, &similarity) in neighbors.iter().zip(similarities) {
                if added == MAX_NEIGHBORS_PER_TERM {
                    break;
                }
                if similarity < MIN_SIMILARITY {
                    continue;
                }
                let Some(token) = index.token_string(neighbor) else {
                    continue;
                };
                if token.starts_with("##") || !token.chars().any(|c| c.is_alphanumeric()) {
                    continue;
                }
                if seen.insert(neighbor) {
                    excluded.push(neighbor);
                    added += 1;
                }
            }
        }
    }
}

/// Add simple inflection variants for a negative whole-word token. Static
/// vocabularies often keep cat and cats as separate entries, so exact
/// negative matching must account for common plural forms.
fn add_natural_exclusion_inflections(index: &Index, seeds: &[u32], excluded: &mut Vec<u32>) {
    let mut seen: std::collections::HashSet<u32> = excluded.iter().copied().collect();
    for &seed in seeds {
        let Some(token) = index.token_string(seed) else {
            continue;
        };
        let surface = token.trim_start_matches('▁');
        if surface.len() < 4 || !surface.chars().all(|c| c.is_alphabetic()) {
            continue;
        }
        let singular = if let Some(stem) = surface.strip_suffix("ies") {
            format!("{stem}y")
        } else if surface.ends_with("ses")
            || surface.ends_with("xes")
            || surface.ends_with("zes")
            || surface.ends_with("ches")
            || surface.ends_with("shes")
        {
            surface[..surface.len() - 2].to_string()
        } else if let Some(stem) = surface.strip_suffix('s') {
            stem.to_string()
        } else {
            continue;
        };
        for variant in index.tokenize_query(&singular) {
            if seen.insert(variant) {
                excluded.push(variant);
            }
        }
    }
}

fn parse_query_with_natural_negation(index: &Index, query: &str) -> (Vec<u32>, Vec<u32>) {
    let NegationParts { positive, negative } = crate::query_negation::extract(query);
    let (included, explicit_excluded) = index.parse_query(&positive);
    let natural_seeds: Vec<u32> = negative
        .iter()
        .flat_map(|text| index.tokenize_query(text))
        .collect();
    let mut excluded = explicit_excluded;
    excluded.extend(natural_seeds.iter().copied());
    add_natural_exclusion_inflections(index, &natural_seeds, &mut excluded);
    add_natural_exclusion_neighbors(index, &natural_seeds, &mut excluded);
    (included, excluded)
}

/// Escape a plain snippet before adding trusted `<mark>` tags.
fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Parse a query for `|`-OR groups. Each whitespace-separated word can be
/// split on `|` to form an OR group; words without `|` become singleton
/// groups. Group tokens are tokenized via `tokenize_query` (same path the
/// regular AND chain uses) so phrase content + casing + normalization
/// match the index. Returns `(groups, flat_tokens, has_or)` so the caller
/// can drop into the right scoring path.
fn parse_or_groups(idx: &Index, q: &str) -> (Vec<Vec<u32>>, Vec<u32>, bool) {
    let mut groups: Vec<Vec<u32>> = Vec::new();
    let mut flat: Vec<u32> = Vec::new();
    let mut has_or = false;
    for word in q.split_whitespace() {
        if word.contains('|') {
            let parts: Vec<&str> = word.split('|').filter(|s| !s.is_empty()).collect();
            if parts.len() < 2 {
                // `foo|` or `|foo` - degrade to a singleton.
                let toks = idx.tokenize_query(word);
                for t in &toks {
                    flat.push(*t);
                    groups.push(vec![*t]);
                }
                continue;
            }
            has_or = true;
            let mut group: Vec<u32> = Vec::new();
            for p in parts {
                let toks = idx.tokenize_query(p);
                group.extend_from_slice(&toks);
                flat.extend_from_slice(&toks);
            }
            group.sort_unstable();
            group.dedup();
            if !group.is_empty() {
                groups.push(group);
            }
        } else {
            let toks = idx.tokenize_query(word);
            for t in &toks {
                flat.push(*t);
                groups.push(vec![*t]);
            }
        }
    }
    flat.sort_unstable();
    flat.dedup();
    (groups, flat, has_or)
}

/// Extract `"…"` runs from a query. Each returned string is the literal
/// content between a pair of double quotes (in order); unclosed quotes are
/// ignored. The caller is responsible for tokenizing each phrase the same
/// way the indexed corpus was.
fn extract_phrases(q: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = q.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            let mut buf = String::new();
            let mut closed = false;
            for c2 in chars.by_ref() {
                if c2 == '"' {
                    closed = true;
                    break;
                }
                buf.push(c2);
            }
            if closed && !buf.trim().is_empty() {
                out.push(buf);
            }
        }
    }
    out
}

/// Split the query on whitespace and punctuation, lowercase each fragment,
/// drop empties and any leading `-` (treated as an excluded term elsewhere).
fn highlight_words(q: &str) -> Vec<String> {
    q.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .map(|w| w.trim_start_matches('-'))
        .filter(|w| !w.is_empty() && w.chars().any(|c| c.is_alphanumeric()))
        .map(|w| w.to_lowercase())
        .collect()
}

/// Wrap case-insensitive matches of `words` in the snippet with
/// `<mark>…</mark>` over an HTML-escaped base. Greedy longest-prefix match at
/// each position so overlapping query words don't produce nested marks.
fn highlight_snippet(snippet: &str, words: &[String]) -> String {
    if words.is_empty() {
        return html_escape(snippet);
    }
    // Pre-lowercase the snippet once for case-insensitive position matching.
    let lower: Vec<char> = snippet.to_lowercase().chars().collect();
    let original: Vec<char> = snippet.chars().collect();
    // Word-lengths in chars (snippet was lowercased the same way).
    let lc_words: Vec<Vec<char>> = words.iter().map(|w| w.chars().collect()).collect();

    let mut out = String::with_capacity(snippet.len() + 32);
    let mut i = 0;
    while i < original.len() {
        // Only consider a match starting at a non-alphanumeric boundary so
        // we don't highlight `cat` inside `category`.
        let on_boundary = i == 0 || !original[i - 1].is_alphanumeric();
        let mut best: Option<usize> = None;
        if on_boundary {
            for w in &lc_words {
                if w.len() > lower.len() - i {
                    continue;
                }
                if lower[i..i + w.len()] == w[..] {
                    let after = i + w.len();
                    let off_boundary =
                        after == original.len() || !original[after].is_alphanumeric();
                    if off_boundary && best.map_or(true, |b| w.len() > b) {
                        best = Some(w.len());
                    }
                }
            }
        }
        if let Some(len) = best {
            let slice: String = original[i..i + len].iter().collect();
            out.push_str("<mark>");
            out.push_str(&html_escape(&slice));
            out.push_str("</mark>");
            i += len;
        } else {
            out.push_str(&html_escape(&original[i].to_string()));
            i += 1;
        }
    }
    out
}

include!("execute.rs");
#[cfg(test)]
mod tests {
    use super::*;

    fn request(extra: &str) -> SearchOptions {
        let json = if extra.is_empty() {
            r#"{"q":"test"}"#.to_string()
        } else {
            format!(r#"{{"q":"test",{extra}}}"#)
        };
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn search_defaults_are_valid() {
        let req = request("");
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            serde_json::to_value(SearchOptions::new("test")).unwrap()
        );
        assert_eq!(req.blend_alpha, 0.5);
        assert_eq!(req.bigram_weight, 0.4);
        assert_eq!(req.composition_weight, 0.0);
        assert_eq!(req.contextual_weight, 0.0);
        assert!(validate_search_req(&req).is_ok());
    }

    #[test]
    fn rejects_weights_outside_documented_ranges() {
        let req = request(r#""blend_alpha":1.1"#);
        assert!(validate_search_req(&req)
            .unwrap_err()
            .contains("blend_alpha"));

        let req = request(r#""proximity_weight":-0.1"#);
        assert!(validate_search_req(&req)
            .unwrap_err()
            .contains("proximity_weight"));

        let req = request(r#""composition_weight":1.1"#);
        assert!(validate_search_req(&req)
            .unwrap_err()
            .contains("composition_weight"));

        let req = request(r#""contextual_weight":1.1"#);
        assert!(validate_search_req(&req)
            .unwrap_err()
            .contains("contextual_weight"));
    }

    #[test]
    fn rejects_unknown_rank_direction_and_empty_fields() {
        let req = request(r#""rank":[{"field":"price","order":"sideways"}]"#);
        assert!(validate_search_req(&req)
            .unwrap_err()
            .contains("rank order"));

        let req = request(r#""filter":[{"field":""}]"#);
        assert!(validate_search_req(&req)
            .unwrap_err()
            .contains("filter field"));
    }
}
