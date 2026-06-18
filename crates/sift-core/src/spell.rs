//! Corpus-backed spelling correction and prefix suggestions.

use crate::Index;

impl Index {
    /// True if this artifact carries a spell-correction sidecar.
    pub fn has_spell(&self) -> bool {
        self.spell_words.is_some() && self.spell_del_keys.is_some()
    }

    /// Prefix-completion suggestions from the corpus spell vocab. Returns
    /// up to `k` words that start with `prefix` (case-insensitive on ASCII),
    /// ranked by descending document frequency. None if the artifact wasn't
    /// built with --spell.
    pub fn suggest(&self, prefix: &str, k: usize) -> Option<Vec<(String, u32)>> {
        let strs = self.spell_words?;
        let offs = self.spell_word_offs?;
        let df = self.spell_word_df?;
        let n = offs.len().saturating_sub(1);
        if n == 0 || prefix.is_empty() {
            return Some(Vec::new());
        }
        let prefix_lc = prefix.to_lowercase();
        let target = prefix_lc.as_bytes();
        // lower_bound: smallest index whose word ≥ prefix.
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let s = offs[mid] as usize;
            let e = offs[mid + 1] as usize;
            if strs[s..e] < *target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // Scan forward, accumulating prefix matches.
        let mut cands: Vec<(String, u32)> = Vec::new();
        let mut i = lo;
        while i < n {
            let s = offs[i] as usize;
            let e = offs[i + 1] as usize;
            let w = &strs[s..e];
            if !w.starts_with(target) {
                break;
            }
            if let Ok(s) = std::str::from_utf8(w) {
                cands.push((s.to_string(), df[i]));
            }
            i += 1;
        }
        cands.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        cands.truncate(k);
        Some(cands)
    }

    /// Look up a word in the spell vocab by binary searching the sorted
    /// strings table. Returns `Some(index)` if found.
    fn spell_lookup_word(&self, w: &str) -> Option<usize> {
        let strs = self.spell_words?;
        let offs = self.spell_word_offs?;
        let n = offs.len().saturating_sub(1);
        if n == 0 {
            return None;
        }
        let target = w.as_bytes();
        // Binary search on offsets, comparing slices in `strs`.
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let s = offs[mid] as usize;
            let e = offs[mid + 1] as usize;
            let cur = &strs[s..e];
            match cur.cmp(target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// Look up a deletion string in the sorted deletion table; if found, return
    /// the slice of word-vocab indices that map back.
    fn spell_lookup_deletion(&self, d: &[u8]) -> Option<&'static [u32]> {
        let strs = self.spell_del_keys?;
        let offs = self.spell_del_offs?;
        let ptr = self.spell_del_ptr?;
        let posts = self.spell_del_posts?;
        let n = offs.len().saturating_sub(1);
        if n == 0 {
            return None;
        }
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let s = offs[mid] as usize;
            let e = offs[mid + 1] as usize;
            let cur = &strs[s..e];
            match cur.cmp(d) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let ps = ptr[mid] as usize;
                    let pe = ptr[mid + 1] as usize;
                    return Some(&posts[ps..pe]);
                }
            }
        }
        None
    }

    /// Return the word string for vocab index `i`.
    fn spell_word_at(&self, i: usize) -> Option<&str> {
        let strs = self.spell_words?;
        let offs = self.spell_word_offs?;
        if i + 1 >= offs.len() {
            return None;
        }
        let s = offs[i] as usize;
        let e = offs[i + 1] as usize;
        std::str::from_utf8(&strs[s..e]).ok()
    }

    /// Levenshtein distance between two ASCII byte strings, bounded to 2 for
    /// early-exit. We only care about edit distance ≤ 1 for spell correction.
    pub(crate) fn edit_distance_le1(a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        // All length logic is on CHAR counts, not byte lengths - mixing the
        // two breaks on multi-byte UTF-8 (e.g. a 1-char 3-byte glyph vs a
        // 3-char ASCII string have equal byte length but differ by 2 chars).
        let mut ac: Vec<char> = a.chars().collect();
        let mut bc: Vec<char> = b.chars().collect();
        // Order so `ac` is the shorter (by char count); keeps the
        // insertion-walk and the asymmetry-free.
        if ac.len() > bc.len() {
            std::mem::swap(&mut ac, &mut bc);
        }
        if bc.len() - ac.len() > 1 {
            return false;
        }
        if ac.len() == bc.len() {
            let mut diffs = 0;
            for (x, y) in ac.iter().zip(bc.iter()) {
                if x != y {
                    diffs += 1;
                    if diffs > 1 {
                        return false;
                    }
                }
            }
            return diffs == 1;
        }
        // Length differs by 1 → exactly one insertion in b vs a.
        let mut i = 0;
        let mut j = 0;
        let mut skipped = false;
        while i < ac.len() && j < bc.len() {
            if ac[i] == bc[j] {
                i += 1;
                j += 1;
            } else if !skipped {
                skipped = true;
                j += 1;
            } else {
                return false;
            }
        }
        true
    }

    /// Attempt to correct a single word against the spell vocab. Returns the
    /// corrected word, or the original if no correction is found within edit
    /// distance 1, or if the word is already an exact match.
    pub fn spell_correct_word(&self, word: &str) -> String {
        if !self.has_spell() {
            return word.to_string();
        }
        let lower = word.to_lowercase();
        // Exact match: keep.
        if self.spell_lookup_word(&lower).is_some() {
            return lower;
        }
        // Candidate pool: union of postings under (lower itself) + its single deletions.
        let mut candidates: std::collections::HashSet<u32> = std::collections::HashSet::new();
        if let Some(p) = self.spell_lookup_deletion(lower.as_bytes()) {
            for &i in p {
                candidates.insert(i);
            }
        }
        let chars: Vec<char> = lower.chars().collect();
        if chars.len() > 1 {
            for i in 0..chars.len() {
                let mut s = String::with_capacity(lower.len());
                for (j, c) in chars.iter().enumerate() {
                    if j != i {
                        s.push(*c);
                    }
                }
                if let Some(p) = self.spell_lookup_deletion(s.as_bytes()) {
                    for &k in p {
                        candidates.insert(k);
                    }
                }
            }
        }
        // Pick best candidate by (edit distance ≤ 1, then highest df).
        let df = match self.spell_word_df {
            Some(d) => d,
            None => return word.to_string(),
        };
        let mut best: Option<(u32, usize)> = None;
        for &idx in &candidates {
            let i = idx as usize;
            let cand = match self.spell_word_at(i) {
                Some(s) => s,
                None => continue,
            };
            if !Self::edit_distance_le1(&lower, cand) {
                continue;
            }
            let c_df = df[i];
            if best.map_or(true, |(b_df, _)| c_df > b_df) {
                best = Some((c_df, i));
            }
        }
        match best.and_then(|(_, i)| self.spell_word_at(i)) {
            Some(s) => s.to_string(),
            None => word.to_string(),
        }
    }

    /// Spell-correct an entire query string. Splits on whitespace and
    /// punctuation, corrects each word independently, joins with spaces.
    /// No-op when no spell sidecar is loaded.
    pub fn spell_correct_query(&self, q: &str) -> String {
        if !self.has_spell() {
            return q.to_string();
        }
        let mut out = String::with_capacity(q.len());
        let mut buf = String::new();
        let flush = |buf: &mut String, out: &mut String, idx: &Self| {
            if buf.is_empty() {
                return;
            }
            let corrected = idx.spell_correct_word(buf);
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(&corrected);
            buf.clear();
        };
        for c in q.chars() {
            if c.is_alphanumeric() || c == '\'' {
                buf.push(c);
            } else {
                flush(&mut buf, &mut out, self);
                if !out.is_empty() && !out.ends_with(' ') {
                    out.push(' ');
                }
            }
        }
        flush(&mut buf, &mut out, self);
        out.trim().to_string()
    }
}
