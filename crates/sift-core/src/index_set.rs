//! Multi-segment indices: incremental add / delete on top of immutable segments.
//!
//! An index on disk is one of two shapes:
//!
//!   * **Legacy single-segment** - the directory *is* a `.sift` artifact
//!     (it has `meta.json` at its root). This is what `sift build` produces
//!     and what every existing artifact looks like. It loads as one segment.
//!
//!   * **Multi-segment** - the directory has a `manifest.json` listing one or
//!     more segment subdirectories, each itself a `.sift` artifact, plus a
//!     `tombstones` file of deleted external doc-ids. `sift add` appends a new
//!     segment; `sift delete` appends tombstones; `sift compact` rebuilds the
//!     whole thing back down to a single segment.
//!
//! Segments are never mutated in place, so an index directory is always safe to
//! copy (`cp`/`rsync`) for backup or replication: the manifest is written
//! atomically (temp file + rename) and points only at already-complete
//! segment directories.
//!
//! Query merge: each segment is scored independently (token ids are per-segment
//! vocabulary, so the query is tokenized per segment), results are merged by
//! score, tombstoned doc-ids are dropped, and when the same external doc-id
//! appears in more than one segment the newest segment wins (an "update" is a
//! delete of the old id plus an add of the new doc).

use crate::Index;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Current manifest schema version.
pub const MANIFEST_FORMAT: u32 = 1;

/// One segment in a multi-segment index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentRef {
    /// Subdirectory name (relative to the index directory) holding the artifact.
    pub dir: String,
    /// Source file this segment was built from, if recorded. Used by `compact`
    /// to rebuild. Absolute or relative to the working directory at add time.
    #[serde(default)]
    pub source: Option<String>,
    /// Full `sift build` argument vector used to produce this segment, minus
    /// `--input`/`--out`. Lets `compact` reproduce the build recipe.
    #[serde(default)]
    pub build_args: Vec<String>,
}

/// On-disk manifest for a multi-segment index directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest schema version.
    pub format: u32,
    /// Bumped on every mutation (add / delete / compact). Lets a replica tell
    /// whether its copy is current.
    pub generation: u64,
    /// Segments in insertion order; later segments shadow earlier ones on a
    /// duplicate external doc-id.
    pub segments: Vec<SegmentRef>,
}

impl Default for Manifest {
    fn default() -> Self {
        Manifest {
            format: MANIFEST_FORMAT,
            generation: 0,
            segments: Vec::new(),
        }
    }
}

impl Manifest {
    /// Read a manifest from `dir/manifest.json`.
    pub fn read(dir: &Path) -> Result<Manifest> {
        let p = dir.join("manifest.json");
        let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        let m: Manifest =
            serde_json::from_str(&s).with_context(|| format!("parsing {}", p.display()))?;
        Ok(m)
    }

    /// Load the manifest for `dir`, migrating a legacy single-artifact directory
    /// (one with `meta.json` at its root) into the segment layout first: its
    /// files move into `seg-00000/` and a manifest is written. Returns a fresh
    /// empty manifest if `dir` does not exist yet (it is created).
    pub fn ensure(dir: &Path) -> Result<Manifest> {
        if dir.join("manifest.json").exists() {
            return Manifest::read(dir);
        }
        if dir.join("meta.json").exists() {
            let seg = dir.join("seg-00000");
            std::fs::create_dir_all(&seg).with_context(|| format!("creating {}", seg.display()))?;
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let name = entry.file_name();
                if name == "seg-00000" || name == "manifest.json" || name == "tombstones" {
                    continue;
                }
                let from = entry.path();
                let to = seg.join(&name);
                std::fs::rename(&from, &to)
                    .with_context(|| format!("moving {} into segment", from.display()))?;
            }
            let manifest = Manifest {
                format: MANIFEST_FORMAT,
                generation: 1,
                segments: vec![SegmentRef {
                    dir: "seg-00000".to_string(),
                    source: None,
                    build_args: Vec::new(),
                }],
            };
            manifest.write(dir)?;
            return Ok(manifest);
        }
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(Manifest::default())
    }

    /// Write the manifest to `dir/manifest.json` durably: write a temp file,
    /// fsync it, atomically rename it into place, then fsync the directory so
    /// the rename itself survives a power loss. After this returns, the manifest
    /// is on stable storage, which is what makes an ack'd write durable.
    pub fn write(&self, dir: &Path) -> Result<()> {
        let tmp = dir.join("manifest.json.tmp");
        let final_path = dir.join("manifest.json");
        let body = serde_json::to_string_pretty(self)?;
        {
            let mut f = std::fs::File::create(&tmp)
                .with_context(|| format!("creating {}", tmp.display()))?;
            std::io::Write::write_all(&mut f, body.as_bytes())?;
            f.sync_all()
                .with_context(|| format!("fsync {}", tmp.display()))?;
        }
        std::fs::rename(&tmp, &final_path)
            .with_context(|| format!("renaming into {}", final_path.display()))?;
        fsync_dir(dir)?;
        Ok(())
    }
}

/// fsync a directory so a rename/creation inside it is durable. Best-effort on
/// platforms where opening a directory for fsync isn't supported.
pub fn fsync_dir(dir: &Path) -> Result<()> {
    if let Ok(f) = std::fs::File::open(dir) {
        let _ = f.sync_all();
    }
    Ok(())
}

/// fsync every regular file directly in `dir`, then the directory itself. Used
/// after building a segment so the segment is on stable storage before the
/// manifest that references it is committed.
pub fn fsync_dir_contents(dir: &Path) -> Result<()> {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Ok(f) = std::fs::File::open(&p) {
                    let _ = f.sync_all();
                }
            }
        }
    }
    fsync_dir(dir)
}

/// Remove `seg-*` directories under `dir` that the manifest doesn't reference.
/// These are orphans left by a crash mid-build (the segment was being written
/// when the process died, before its manifest commit). The write was never
/// acked, so dropping the orphan is safe. Returns the number removed.
pub fn recover_orphans(dir: &Path, manifest: &Manifest) -> Result<usize> {
    use std::collections::HashSet;
    let referenced: HashSet<&str> = manifest.segments.iter().map(|s| s.dir.as_str()).collect();
    let mut removed = 0usize;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy().to_string();
            if name.starts_with("seg-")
                && !referenced.contains(name.as_str())
                && entry.path().is_dir()
            {
                std::fs::remove_dir_all(entry.path())
                    .with_context(|| format!("removing orphan segment {name}"))?;
                removed += 1;
            }
        }
    }
    Ok(removed)
}

/// Next free `seg-NNNNN` directory name in `dir` (max existing index + 1).
pub fn next_segment_name(dir: &Path) -> Result<String> {
    let mut max_id: i64 = -1;
    if dir.exists() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(num) = name.strip_prefix("seg-") {
                if let Ok(n) = num.parse::<i64>() {
                    max_id = max_id.max(n);
                }
            }
        }
    }
    Ok(format!("seg-{:05}", max_id + 1))
}

/// Sentinel tombstone threshold meaning "dead in every segment" (a hard delete).
pub const TOMBSTONE_ALL: u32 = u32::MAX;

/// Numeric ordinal of a `seg-NNNNN` directory name. Ordinals are monotonic
/// (each new segment takes max+1), so a larger ordinal is a newer segment.
pub fn parse_seg_ordinal(dir_name: &str) -> u32 {
    dir_name
        .strip_prefix("seg-")
        .and_then(|n| n.parse::<u32>().ok())
        .unwrap_or(0)
}

/// Read the tombstones from `dir/tombstones`. Each entry maps a doc-id to a
/// threshold ordinal: the id is dead in every segment whose ordinal is strictly
/// less than the threshold. A hard delete uses [`TOMBSTONE_ALL`]. On-disk format
/// is one `id` (= dead everywhere) or `id<TAB>ordinal` per line. Missing file =
/// empty.
pub fn read_tombstones(dir: &Path) -> Result<std::collections::HashMap<String, u32>> {
    let p = dir.join("tombstones");
    let s = match std::fs::read_to_string(&p) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Default::default()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", p.display())),
    };
    let mut map: std::collections::HashMap<String, u32> = Default::default();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (id, ord) = match line.split_once('\t') {
            Some((id, o)) => (id.to_string(), o.parse::<u32>().unwrap_or(TOMBSTONE_ALL)),
            None => (line.to_string(), TOMBSTONE_ALL),
        };
        let e = map.entry(id).or_insert(0);
        if ord > *e {
            *e = ord;
        }
    }
    Ok(map)
}

/// Tombstone `ids` with threshold `ordinal`: each id becomes dead in every
/// segment older than `ordinal`. Use [`TOMBSTONE_ALL`] for a hard delete (dead
/// everywhere) or the new segment's ordinal for an upsert (older copies die,
/// the new copy survives). Returns how many entries were raised. Durable.
pub fn add_tombstones(dir: &Path, ids: &[String], ordinal: u32) -> Result<usize> {
    let mut existing = read_tombstones(dir)?;
    let mut changed = 0usize;
    for id in ids {
        let e = existing.entry(id.clone()).or_insert(0);
        if ordinal > *e {
            *e = ordinal;
            changed += 1;
        }
    }
    if changed > 0 {
        let mut all: Vec<(&String, &u32)> = existing.iter().collect();
        all.sort();
        let mut body = String::new();
        for (id, ord) in all {
            body.push_str(id);
            if *ord != TOMBSTONE_ALL {
                body.push('\t');
                body.push_str(&ord.to_string());
            }
            body.push('\n');
        }
        let tmp = dir.join("tombstones.tmp");
        {
            let mut f = std::fs::File::create(&tmp)
                .with_context(|| format!("creating {}", tmp.display()))?;
            std::io::Write::write_all(&mut f, body.as_bytes())?;
            f.sync_all()
                .with_context(|| format!("fsync {}", tmp.display()))?;
        }
        std::fs::rename(&tmp, dir.join("tombstones"))?;
        fsync_dir(dir)?;
    }
    Ok(changed)
}

/// True if doc-id is dead in a segment with the given ordinal, per the map.
pub fn is_tombstoned(
    tombstones: &std::collections::HashMap<String, u32>,
    id: &str,
    ordinal: u32,
) -> bool {
    tombstones.get(id).is_some_and(|&t| ordinal < t)
}

/// Recompute collection-wide idf and avgdl across `segments`, returning
/// `(Some(idf), avgdl)` only when every segment shares the same vocab and
/// carries the exact (un-expanded) CSR. The exact CSR's per-term row length
/// is that term's document frequency, so summing row lengths across segments
/// gives the global df; idf follows the BM25 formula and is then re-damped
/// for `##` continuation tokens by the recorded `subword_weight`, matching
/// what each segment baked into its own idf.bin. Returns `(None, _)` when
/// global stats can't be formed (one segment, mixed vocab, or missing exact
/// CSR), so the merged path falls back to per-segment idf.
fn compute_global_stats(segments: &[Index]) -> (Option<Vec<f32>>, f32) {
    if segments.len() < 2 {
        let avgdl = segments.first().map(|s| s.meta.avgdl).unwrap_or(1.0);
        return (None, avgdl);
    }
    let vocab = segments[0].meta.vocab_size as usize;
    for s in segments {
        if s.meta.vocab_size as usize != vocab || s.exact_indptr.is_none() {
            let avgdl = segments[0].meta.avgdl;
            return (None, avgdl);
        }
    }
    let mut df = vec![0u64; vocab];
    let mut n_docs: u64 = 0;
    let mut total_len: f64 = 0.0;
    for s in segments {
        let eip = s.exact_indptr.expect("checked above");
        if eip.len() < vocab + 1 {
            return (None, segments[0].meta.avgdl);
        }
        for (t, d) in df.iter_mut().enumerate() {
            *d += eip[t + 1] - eip[t];
        }
        n_docs += s.meta.n_docs;
        total_len += s.meta.avgdl as f64 * s.meta.n_docs as f64;
    }
    if n_docs == 0 {
        return (None, segments[0].meta.avgdl);
    }
    let n_f = n_docs as f32;
    let mut idf = vec![0f32; vocab];
    for (t, &c_u) in df.iter().enumerate() {
        if c_u > 0 {
            let c = c_u as f32;
            idf[t] = ((n_f - c + 0.5) / (c + 0.5) + 1.0).ln();
        }
    }
    // Re-apply the subword damping each segment baked into its idf.bin, so the
    // global idf has the same shape (just with collection-wide df).
    let sw = segments[0].meta.subword_weight;
    if (sw - 1.0).abs() > f32::EPSILON {
        let tok = &segments[0].tokenizer;
        for (t, v) in idf.iter_mut().enumerate() {
            if *v > 0.0 {
                if let Some(s) = tok.id_to_token(t as u32) {
                    if s.starts_with("##") {
                        *v *= sw;
                    }
                }
            }
        }
    }
    let avgdl = (total_len / n_docs as f64) as f32;
    (Some(idf), avgdl.max(1.0))
}

/// A hit merged across segments, with external identifiers resolved to owned
/// strings (segment-local `doc_idx` can't survive a cross-segment merge).
#[derive(Debug, Clone)]
pub struct MergedHit {
    pub doc_id: String,
    pub score: f32,
    pub snippet: String,
    /// Which segment (index into `IndexSet::segments`) produced this hit.
    pub segment: usize,
    /// Segment-local document index, for callers that need to reach back into
    /// the originating segment (e.g. feature extraction).
    pub doc_idx: u32,
}

#[derive(Debug, Clone)]
pub struct MergedResults {
    pub hits: Vec<MergedHit>,
    pub matched_query_terms: u32,
    /// Distinct matched docs across all segments before truncation to `k`
    /// (bounded by the per-segment oversample window for very large result sets).
    pub total: usize,
    pub elapsed_us: u64,
}

/// How to score each segment in a merged query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreMode {
    /// Plain BM25 over the expanded index.
    Plain,
    /// WAND.
    Wand,
    /// Block-Max WAND (falls back to plain if a segment lacks the sidecar).
    BlockMaxWand,
}

/// A loaded index: one or more immutable segments plus a tombstone set.
pub struct IndexSet {
    /// Insertion order; later segments shadow earlier ones on duplicate doc-id.
    pub segments: Vec<Index>,
    /// Numeric ordinal of each segment (parallel to `segments`), parsed from the
    /// `seg-NNNNN` directory name. A hit is dead if its segment ordinal is below
    /// the doc-id's tombstone threshold.
    pub seg_ordinals: Vec<u32>,
    /// doc-id -> tombstone threshold ordinal (dead in segments below it).
    pub tombstones: std::collections::HashMap<String, u32>,
    /// The index directory (the parent of the segment dirs, or the artifact
    /// dir itself in the legacy single-segment case).
    pub dir: PathBuf,
    /// True when this directory uses the manifest layout (vs. a legacy artifact).
    pub manifested: bool,
    /// Collection-wide idf over the shared vocab, recomputed across all
    /// segments at open. `None` when there's only one segment, or segments
    /// don't share a vocab, or any segment lacks the exact CSR; the merged
    /// path then falls back to each segment's own idf. Slightly approximate:
    /// tombstoned / upsert-superseded docs still count toward df until a
    /// `compact` rebuilds the collection.
    pub global_idf: Option<Vec<f32>>,
    /// Collection-wide average document length, paired with `global_idf`.
    pub global_avgdl: f32,
}

impl IndexSet {
    /// Open an index directory in either layout.
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<IndexSet> {
        let dir = dir.as_ref().to_path_buf();
        if dir.join("manifest.json").exists() {
            let manifest = Manifest::read(&dir)?;
            if manifest.segments.is_empty() {
                return Err(anyhow!("manifest at {} lists no segments", dir.display()));
            }
            let mut segments = Vec::with_capacity(manifest.segments.len());
            let mut seg_ordinals = Vec::with_capacity(manifest.segments.len());
            for seg in &manifest.segments {
                let seg_dir = dir.join(&seg.dir);
                let idx = Index::open(&seg_dir)
                    .with_context(|| format!("opening segment {}", seg_dir.display()))?;
                segments.push(idx);
                seg_ordinals.push(parse_seg_ordinal(&seg.dir));
            }
            let tombstones = read_tombstones(&dir)?;
            let (global_idf, global_avgdl) = compute_global_stats(&segments);
            Ok(IndexSet {
                segments,
                seg_ordinals,
                tombstones,
                dir,
                manifested: true,
                global_idf,
                global_avgdl,
            })
        } else if dir.join("meta.json").exists() {
            // Legacy single-segment artifact: the directory is the segment.
            let idx = Index::open(&dir)?;
            let avgdl = idx.meta.avgdl;
            Ok(IndexSet {
                segments: vec![idx],
                seg_ordinals: vec![0],
                tombstones: std::collections::HashMap::new(),
                dir,
                manifested: false,
                global_idf: None,
                global_avgdl: avgdl,
            })
        } else {
            Err(anyhow!(
                "{} is neither a .sift artifact (no meta.json) nor a multi-segment index (no manifest.json)",
                dir.display()
            ))
        }
    }

    /// True when this index is a single segment with no tombstones: the common
    /// case, where every advanced query feature can run directly on the one
    /// underlying [`Index`] with no merge.
    pub fn is_single(&self) -> bool {
        self.segments.len() == 1 && self.tombstones.is_empty()
    }

    /// The sole segment. Callers that need the rich single-segment API should
    /// guard with [`IndexSet::is_single`] first; this returns the first segment
    /// regardless.
    pub fn primary(&self) -> &Index {
        &self.segments[0]
    }

    /// Live document count: the sum across segments minus tombstoned ids that
    /// are actually present. Cheap upper bound (ignores cross-segment id
    /// shadowing); exact counts require a full scan, which callers rarely need.
    pub fn n_docs_total(&self) -> usize {
        self.segments.iter().map(|s| s.n_docs()).sum()
    }

    pub fn n_segments(&self) -> usize {
        self.segments.len()
    }

    /// Score `query` across every segment and merge into a single top-`k`,
    /// dropping tombstoned doc-ids and keeping the newest segment's copy of any
    /// duplicated external doc-id.
    pub fn search_merged(
        &self,
        query: &str,
        k: usize,
        mode: ScoreMode,
        blend_alpha: f32,
    ) -> MergedResults {
        let t0 = std::time::Instant::now();
        // Oversample per segment so cross-segment merge + dedup + tombstone
        // filtering still yields a full k.
        let inner_k = k.max(1).saturating_mul(3).max(k + 16);
        let mut matched = 0u32;
        // Keep, per external doc-id, the best (score, segment, doc_idx). Later
        // segments win ties because we iterate segments in order and only
        // replace when strictly newer OR higher-scoring within the same id.
        let mut merged: Vec<MergedHit> = Vec::new();
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for (si, seg) in self.segments.iter().enumerate() {
            let tokens = seg.tokenize_query(query);
            if tokens.is_empty() {
                continue;
            }
            // Score each segment on collection-wide stats when available, so a
            // term that is rare globally but common in one small segment (or
            // vice versa) ranks consistently across segments. The dense
            // blended scorer recomputes everything from the supplied idf/avgdl;
            // the WAND modes are skipped under global stats because their
            // pruning bounds were baked with each segment's local idf. When no
            // global stats exist (mixed vocab / missing exact CSR) we fall back
            // to the segment's own stats and honor the requested mode.
            let res = match &self.global_idf {
                Some(gidf) => seg.score_blended_qexp_with(
                    &tokens,
                    inner_k,
                    blend_alpha,
                    &[],
                    gidf,
                    self.global_avgdl,
                ),
                None => {
                    let local = match mode {
                        ScoreMode::Plain => seg.score(&tokens, inner_k),
                        ScoreMode::Wand => seg.score_wand(&tokens, inner_k),
                        ScoreMode::BlockMaxWand => seg.score_block_max_wand(&tokens, inner_k),
                    };
                    if blend_alpha < 1.0 {
                        seg.score_blended(&tokens, inner_k, blend_alpha)
                    } else {
                        local
                    }
                }
            };
            let ord = self.seg_ordinals[si];
            matched = matched.max(res.matched_query_terms);
            for h in res.hits {
                let doc_id = seg.doc_id(h.doc_idx as usize).to_string();
                // Dead if this segment is older than the id's tombstone threshold
                // (covers both hard deletes and upsert-superseded older copies).
                if is_tombstoned(&self.tombstones, &doc_id, ord) {
                    continue;
                }
                match seen.get(&doc_id).copied() {
                    Some(pos) => {
                        // Same id seen in an earlier (older) segment. Newest
                        // segment wins regardless of score (it's the update).
                        merged[pos] = MergedHit {
                            doc_id,
                            score: h.score,
                            snippet: seg.doc_snip(h.doc_idx as usize).to_string(),
                            segment: si,
                            doc_idx: h.doc_idx,
                        };
                    }
                    None => {
                        seen.insert(doc_id.clone(), merged.len());
                        merged.push(MergedHit {
                            doc_id,
                            score: h.score,
                            snippet: seg.doc_snip(h.doc_idx as usize).to_string(),
                            segment: si,
                            doc_idx: h.doc_idx,
                        });
                    }
                }
            }
        }

        merged.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total = merged.len();
        merged.truncate(k);

        MergedResults {
            hits: merged,
            matched_query_terms: matched,
            total,
            elapsed_us: t0.elapsed().as_micros() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sift_idxset_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn manifest_roundtrip() {
        let d = scratch("manifest");
        let m = Manifest {
            format: MANIFEST_FORMAT,
            generation: 7,
            segments: vec![
                SegmentRef {
                    dir: "seg-00000".into(),
                    source: Some("a.jsonl".into()),
                    build_args: vec!["--threshold".into(), "0.5".into()],
                },
                SegmentRef {
                    dir: "seg-00001".into(),
                    source: None,
                    build_args: vec![],
                },
            ],
        };
        m.write(&d).unwrap();
        let back = Manifest::read(&d).unwrap();
        assert_eq!(back.generation, 7);
        assert_eq!(back.segments.len(), 2);
        assert_eq!(back.segments[0].dir, "seg-00000");
        assert_eq!(back.segments[0].build_args, vec!["--threshold", "0.5"]);
        assert_eq!(back.segments[1].source, None);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn orphan_recovery_removes_unreferenced_segments() {
        let d = scratch("orphans");
        // manifest references only seg-00000
        let m = Manifest {
            format: MANIFEST_FORMAT,
            generation: 1,
            segments: vec![SegmentRef {
                dir: "seg-00000".into(),
                source: None,
                build_args: vec![],
            }],
        };
        std::fs::create_dir_all(d.join("seg-00000")).unwrap();
        std::fs::create_dir_all(d.join("seg-00001")).unwrap(); // orphan
        std::fs::create_dir_all(d.join("seg-00002")).unwrap(); // orphan
        m.write(&d).unwrap();
        let removed = recover_orphans(&d, &m).unwrap();
        assert_eq!(removed, 2);
        assert!(d.join("seg-00000").exists());
        assert!(!d.join("seg-00001").exists());
        assert!(!d.join("seg-00002").exists());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn tombstones_dedup_and_persist() {
        let d = scratch("tombstones");
        assert!(read_tombstones(&d).unwrap().is_empty());
        // Hard-delete x and y.
        let added =
            add_tombstones(&d, &["x".into(), "y".into(), "x".into()], TOMBSTONE_ALL).unwrap();
        assert_eq!(added, 2, "duplicate within the same call counts once");
        let map = read_tombstones(&d).unwrap();
        assert_eq!(map.len(), 2);
        assert!(is_tombstoned(&map, "x", 0) && is_tombstoned(&map, "x", 100));
        // Upsert-style scoped tombstone: z dead only below ordinal 5.
        add_tombstones(&d, &["z".into()], 5).unwrap();
        let map = read_tombstones(&d).unwrap();
        assert!(is_tombstoned(&map, "z", 4), "z dead in older segment 4");
        assert!(!is_tombstoned(&map, "z", 5), "z live in its own segment 5");
        assert!(!is_tombstoned(&map, "z", 9), "z live in newer segment 9");
        std::fs::remove_dir_all(&d).ok();
    }
}
