//! Corpus-only SymSpell spell-correction support for sift.
//!
//! At build time we walk the (already normalized) document text, extract a
//! word-level vocabulary with frequencies, and emit a single-deletion lookup
//! table. At serve time the table is mmapped (in `sift-core`) and queried
//! per word that doesn't match the vocab.
//!
//! The artifact format here is a tight set of 5 binary files:
//!   spell_words.bin       concatenated UTF-8 bytes of the sorted word list
//!   spell_word_offs.bin   u64[V+1] byte offsets per word
//!   spell_word_df.bin     u32[V]   doc-frequency per word
//!   spell_del_keys.bin    concatenated UTF-8 bytes of sorted deletion strings
//!   spell_del_offs.bin    u64[D+1] byte offsets per deletion
//!   spell_del_ptr.bin     u64[D+1] offsets into the postings array
//!   spell_del_posts.bin   u32[N]   word-vocab indices that map back from each
//!                                  deletion (one posting per (deletion, word) pair)

use ahash::{AHashMap, AHashSet};
use anyhow::Result;
use std::path::Path;

#[cfg(feature = "multithread")]
use rayon::prelude::*;

/// Split `text` into "words" using whitespace and punctuation as separators.
/// Same boundaries the WordPiece tokenizer effectively uses, applied after the
/// existing normalize_text() pass.
pub fn split_into_words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
}

/// Walk a single doc, returning the unique set of word strings in it.
fn doc_unique_words(text: &str, max_word_len: usize) -> AHashSet<String> {
    let mut out = AHashSet::default();
    for w in split_into_words(text) {
        if w.len() > max_word_len {
            continue;
        }
        // lowercase normalizes spelling-correction lookups
        out.insert(w.to_lowercase());
    }
    out
}

/// Generate all single-character deletions of `word` (Unicode-safe).
fn one_deletions(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut out = Vec::with_capacity(chars.len());
    for i in 0..chars.len() {
        let mut s = String::with_capacity(word.len());
        for (j, c) in chars.iter().enumerate() {
            if i != j {
                s.push(*c);
            }
        }
        out.push(s);
    }
    out
}

/// Result of the build-time spell vocab + deletion table construction.
pub struct SpellTable {
    /// Sorted word strings.
    pub words: Vec<String>,
    /// df per word (parallel to `words`).
    pub df: Vec<u32>,
    /// Sorted deletion strings.
    pub deletions: Vec<String>,
    /// Per-deletion list of word-vocab indices (sorted within each deletion).
    pub postings_per_deletion: Vec<Vec<u32>>,
}

/// Build the spell table by walking already-tokenized (and normalized) doc
/// texts. We rebuild the word list from the raw text instead of relying on
/// the WordPiece subword stream - WordPiece atomization defeats the spell
/// check (`cancerr` → `cancer ##r`, both "valid").
///
/// `doc_texts` is the *normalized* doc text. The build pipeline already runs
/// `sift_core::normalize_text` so we don't repeat it here.
pub fn build_spell_table(
    doc_texts: &[String],
    min_df: u32,
    max_word_len: usize,
    extra_words: &[String],
) -> SpellTable {
    // Pass 1: word frequencies (df, not total tf).
    let mut df_map: AHashMap<String, u32> = AHashMap::default();
    #[cfg(feature = "multithread")]
    let partials: Vec<AHashMap<String, u32>> = doc_texts
        .par_iter()
        .map(|t| {
            let mut local: AHashMap<String, u32> = AHashMap::default();
            for w in doc_unique_words(t, max_word_len) {
                *local.entry(w).or_insert(0) += 1;
            }
            local
        })
        .collect();
    #[cfg(not(feature = "multithread"))]
    let partials: Vec<AHashMap<String, u32>> = doc_texts
        .iter()
        .map(|t| {
            let mut local: AHashMap<String, u32> = AHashMap::default();
            for w in doc_unique_words(t, max_word_len) {
                *local.entry(w).or_insert(0) += 1;
            }
            local
        })
        .collect();
    for p in partials {
        for (w, c) in p {
            *df_map.entry(w).or_insert(0) += c;
        }
    }

    // Filter by min_df and sort lexically so we can binary-search on the
    // serve side.
    let mut words: Vec<(String, u32)> = df_map
        .into_iter()
        .filter(|(w, c)| *c >= min_df && !w.is_empty() && w.chars().count() <= max_word_len)
        .collect();
    // Dictionary augmentation: any word in `extra_words` not already in the
    // corpus vocab is added with df=min_df so it survives the filter. Words
    // that ARE in the corpus keep their corpus df (no double-counting).
    if !extra_words.is_empty() {
        let in_corpus: ahash::AHashSet<String> = words.iter().map(|(w, _)| w.clone()).collect();
        for w in extra_words {
            if w.is_empty() || w.chars().count() > max_word_len {
                continue;
            }
            if !in_corpus.contains(w) {
                words.push((w.clone(), min_df));
            }
        }
    }
    words.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let word_strs: Vec<String> = words.iter().map(|(w, _)| w.clone()).collect();
    let word_df: Vec<u32> = words.iter().map(|(_, c)| *c).collect();

    // Deletion table: for each word, generate single-character deletions; map
    // each deletion → list of word-vocab indices.
    let mut del_map: AHashMap<String, Vec<u32>> = AHashMap::default();
    for (i, w) in word_strs.iter().enumerate() {
        // also include the word itself as a "0-edit" candidate so the lookup
        // round-trips correctly for an exact-match query word.
        del_map.entry(w.clone()).or_default().push(i as u32);
        if w.chars().count() < 2 {
            continue;
        }
        for d in one_deletions(w) {
            del_map.entry(d).or_default().push(i as u32);
        }
    }

    // Sort deletion keys + dedup posting lists.
    let mut sorted_keys: Vec<String> = del_map.keys().cloned().collect();
    sorted_keys.sort_unstable();
    let postings: Vec<Vec<u32>> = sorted_keys
        .iter()
        .map(|k| {
            let mut v = del_map.remove(k).unwrap();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();

    SpellTable {
        words: word_strs,
        df: word_df,
        deletions: sorted_keys,
        postings_per_deletion: postings,
    }
}

/// Serialize a SpellTable to the on-disk binary format.
pub fn write_spell_table(out_dir: &Path, t: &SpellTable) -> Result<(usize, usize, usize)> {
    use std::fs::File;
    use std::io::Write;

    // words + offsets + df
    let mut words_buf: Vec<u8> = Vec::new();
    let mut word_offs: Vec<u64> = Vec::with_capacity(t.words.len() + 1);
    word_offs.push(0);
    for w in &t.words {
        words_buf.extend_from_slice(w.as_bytes());
        word_offs.push(words_buf.len() as u64);
    }
    File::create(out_dir.join("spell_words.bin"))?.write_all(&words_buf)?;
    write_u64(out_dir, "spell_word_offs.bin", &word_offs)?;
    write_u32(out_dir, "spell_word_df.bin", &t.df)?;

    // deletions + offsets
    let mut del_buf: Vec<u8> = Vec::new();
    let mut del_offs: Vec<u64> = Vec::with_capacity(t.deletions.len() + 1);
    del_offs.push(0);
    for d in &t.deletions {
        del_buf.extend_from_slice(d.as_bytes());
        del_offs.push(del_buf.len() as u64);
    }
    File::create(out_dir.join("spell_del_keys.bin"))?.write_all(&del_buf)?;
    write_u64(out_dir, "spell_del_offs.bin", &del_offs)?;

    // postings + their indptr
    let mut postings_flat: Vec<u32> = Vec::new();
    let mut posts_ptr: Vec<u64> = Vec::with_capacity(t.postings_per_deletion.len() + 1);
    posts_ptr.push(0);
    for p in &t.postings_per_deletion {
        postings_flat.extend_from_slice(p);
        posts_ptr.push(postings_flat.len() as u64);
    }
    write_u32(out_dir, "spell_del_posts.bin", &postings_flat)?;
    write_u64(out_dir, "spell_del_ptr.bin", &posts_ptr)?;

    Ok((t.words.len(), t.deletions.len(), postings_flat.len()))
}

fn write_u64(dir: &Path, name: &str, data: &[u64]) -> Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut f = File::create(dir.join(name))?;
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    f.write_all(bytes)?;
    Ok(())
}

fn write_u32(dir: &Path, name: &str, data: &[u32]) -> Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut f = File::create(dir.join(name))?;
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    f.write_all(bytes)?;
    Ok(())
}
