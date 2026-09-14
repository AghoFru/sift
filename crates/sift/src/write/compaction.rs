//! Rebuild immutable segments through the shared index writer.

use crate::build::{self, build_args_from};
use anyhow::{anyhow, Context, Result};
use sift_core::{next_segment_name, Index, Manifest, SegmentRef};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

pub(crate) fn compact(index_dir: &Path) -> Result<()> {
    if !index_dir.join("manifest.json").exists() {
        if index_dir.join("meta.json").exists() {
            println!(
                "{} is a single artifact already; nothing to compact",
                index_dir.display()
            );
            return Ok(());
        }
        return Err(anyhow!("{} is not an index directory", index_dir.display()));
    }
    let manifest = Manifest::read(index_dir)?;
    if manifest.segments.is_empty() {
        return Err(anyhow!("manifest lists no segments"));
    }

    // Every segment must have a recorded source to rebuild from.
    for seg in &manifest.segments {
        if seg.source.is_none() {
            return Err(anyhow!(
                "segment {} has no recorded source (likely a migrated legacy base); \
                 cannot compact. Rebuild the index with `sift build` then `sift add`.",
                seg.dir
            ));
        }
    }
    // All segments must share a model (single CSR vocab in the output).
    let base_model = read_model_name(&index_dir.join(&manifest.segments[0].dir))?;
    for seg in &manifest.segments[1..] {
        let m = read_model_name(&index_dir.join(&seg.dir))?;
        if m != base_model {
            return Err(anyhow!(
                "segments use different models ({base_model} vs {m}); compaction needs one model"
            ));
        }
    }

    let tombstones = sift_core::read_tombstones(index_dir)?;
    let format = detect_format(&manifest);

    // Concatenate every segment's live source rows into one temp JSONL,
    // dropping tombstoned ids and shadowed (re-added) ids. Later segments win,
    // so process newest-first and skip ids already emitted.
    let tmp_jsonl = index_dir.join(".compact-source.jsonl");
    let mut out = std::io::BufWriter::new(
        fs::File::create(&tmp_jsonl)
            .with_context(|| format!("creating {}", tmp_jsonl.display()))?,
    );
    let mut emitted: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut kept = 0usize;
    for seg in manifest.segments.iter().rev() {
        let ord = sift_core::parse_seg_ordinal(&seg.dir);
        // New segments retain their source locally so copied indexes can compact.
        let local_source = index_dir.join(&seg.dir).join("source.jsonl");
        let src = if local_source.is_file() {
            local_source
        } else {
            PathBuf::from(seg.source.as_ref().unwrap())
        };
        let f = fs::File::open(&src)
            .with_context(|| format!("reading source {} for compaction", src.display()))?;
        for line in BufReader::new(f).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let id = extract_id(&line, &format);
            if let Some(id) = id {
                // Drop if dead in this segment (hard delete or superseded by a
                // newer copy via upsert) or already emitted from a newer segment.
                if sift_core::is_tombstoned(&tombstones, &id, ord) || !emitted.insert(id) {
                    continue;
                }
            }
            out.write_all(line.as_bytes())?;
            out.write_all(b"\n")?;
            kept += 1;
        }
    }
    out.flush()?;
    drop(out);

    // Build a fresh single segment from the base segment's recipe.
    let new_seg_name = next_segment_name(index_dir)?;
    let new_seg_dir = index_dir.join(&new_seg_name);
    let base_argv = &manifest.segments[0].build_args;
    let build_args = build_args_from(&tmp_jsonl, &new_seg_dir, base_argv)?;
    build::run(build_args).context("building compacted segment")?;

    // Swap the manifest to point only at the new segment, clear tombstones,
    // then remove the old segment dirs and temp source.
    let new_manifest = Manifest {
        format: sift_core::MANIFEST_FORMAT,
        generation: manifest
            .generation
            .checked_add(1)
            .ok_or_else(|| anyhow!("Index generation limit reached."))?,
        segments: vec![SegmentRef {
            dir: new_seg_name.clone(),
            source: Some(new_seg_dir.join("source.jsonl").display().to_string()),
            build_args: base_argv.clone(),
        }],
    };
    // Persist the compacted source so a future compaction still has a source.
    let final_src = new_seg_dir.join("source.jsonl");
    fs::rename(&tmp_jsonl, &final_src)
        .with_context(|| format!("renaming compacted source to {}", final_src.display()))?;
    sift_core::fsync_dir_contents(&new_seg_dir)?;
    new_manifest.write(index_dir)?;
    let _ = fs::remove_file(index_dir.join("tombstones"));
    for seg in &manifest.segments {
        let _ = fs::remove_dir_all(index_dir.join(&seg.dir));
    }
    println!(
        "compacted {} segment(s) into {} ({kept} live docs), generation {}",
        manifest.segments.len(),
        new_seg_name,
        new_manifest.generation
    );
    Ok(())
}

fn read_model_name(seg_dir: &Path) -> Result<String> {
    let idx = Index::open(seg_dir)
        .with_context(|| format!("opening {} to read model name", seg_dir.display()))?;
    Ok(idx.meta.model_name.clone())
}

/// Guess the input format from the base segment's recorded build args.
fn detect_format(manifest: &Manifest) -> String {
    let argv = &manifest.segments[0].build_args;
    for w in argv.windows(2) {
        if w[0] == "--format" {
            return w[1].clone();
        }
    }
    "jsonl".to_string()
}

/// Pull the external doc-id out of one JSONL row for the given format.
fn extract_id(line: &str, format: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let key = if format == "beir" { "_id" } else { "id" };
    v.get(key).map(|x| match x {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}
