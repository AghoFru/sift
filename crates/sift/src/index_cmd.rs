//! `sift add` / `sift delete` / `sift compact` - incremental updates.
//!
//! An index directory holds immutable segment subdirectories plus a
//! `manifest.json` and a `tombstones` file (see `sift_core::index_set`).
//!
//!   * `add` builds a new segment from a JSONL file and appends it.
//!   * `delete` appends external doc-ids to the tombstone set.
//!   * `compact` rebuilds the whole index back into a single segment, dropping
//!     tombstoned docs and reclaiming their space.
//!
//! Adding to or deleting from a *legacy* single-segment artifact (a directory
//! that is itself a `.sift` build output) transparently migrates it into the
//! manifest layout first: the existing files move into `seg-00000/` and a
//! manifest is written. The migration is atomic enough for local use (a crash
//! mid-move leaves the original files in place or fully moved; the manifest is
//! written last).

use crate::build::{self, BuildArgs};
use anyhow::{anyhow, Context, Result};
use clap::Args;
use sift_core::{add_tombstones, next_segment_name, Index, Manifest, SegmentRef};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct AddArgs {
    /// Index directory to append to (created if it does not exist).
    #[arg(long)]
    pub index: PathBuf,
    /// Build options for the new segment. `--input` is the JSONL to add;
    /// `--out` is ignored (the segment directory is chosen automatically).
    #[command(flatten)]
    pub build: BuildArgs,
}

#[derive(Args, Debug)]
pub struct DeleteArgs {
    /// Index directory to delete from.
    #[arg(long)]
    pub index: PathBuf,
    /// External doc-id to tombstone (repeatable).
    #[arg(long = "id")]
    pub ids: Vec<String>,
    /// File of external doc-ids to tombstone, one per line.
    #[arg(long)]
    pub ids_file: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct CompactArgs {
    /// Index directory to compact.
    #[arg(long)]
    pub index: PathBuf,
}

/// Reconstruct the `sift build` flags (minus `--input`/`--out`) for a
/// BuildArgs, emitting only non-default values. Used to record how a segment
/// was built so `compact` can reproduce it.
fn build_args_to_argv(b: &BuildArgs) -> Vec<String> {
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

pub fn run_add(args: AddArgs) -> Result<()> {
    let index_dir = args.index.clone();
    let mut manifest = Manifest::ensure(&index_dir)?;

    let source = args.build.input.clone();
    let argv = build_args_to_argv(&args.build);
    let seg_name = next_segment_name(&index_dir)?;
    let seg_dir = index_dir.join(&seg_name);

    // Build the new segment into its subdirectory.
    let mut build = args.build;
    build.output = Some(seg_dir.clone());
    build::run(build).with_context(|| format!("building segment {}", seg_dir.display()))?;

    manifest.segments.push(SegmentRef {
        dir: seg_name.clone(),
        source: Some(source.display().to_string()),
        build_args: argv,
    });
    manifest.generation += 1;
    manifest.write(&index_dir)?;
    println!(
        "added {} ({} segment(s), generation {})",
        seg_name,
        manifest.segments.len(),
        manifest.generation
    );
    Ok(())
}

pub fn run_delete(args: DeleteArgs) -> Result<()> {
    let index_dir = args.index.clone();
    let mut manifest = Manifest::ensure(&index_dir)?;

    let mut ids = args.ids.clone();
    if let Some(f) = &args.ids_file {
        let file = fs::File::open(f).with_context(|| format!("opening {}", f.display()))?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            let t = line.trim();
            if !t.is_empty() {
                ids.push(t.to_string());
            }
        }
    }
    if ids.is_empty() {
        return Err(anyhow!("no ids given (use --id or --ids-file)"));
    }
    let added = add_tombstones(&index_dir, &ids, sift_core::TOMBSTONE_ALL)?;
    manifest.generation += 1;
    manifest.write(&index_dir)?;
    println!(
        "tombstoned {} new id(s) ({} requested), generation {}",
        added,
        ids.len(),
        manifest.generation
    );
    Ok(())
}

pub fn run_compact(args: CompactArgs) -> Result<()> {
    let index_dir = args.index.clone();
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
    let manifest = Manifest::read(&index_dir)?;
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

    let tombstones = sift_core::read_tombstones(&index_dir)?;
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
        let src = PathBuf::from(seg.source.as_ref().unwrap());
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
    let new_seg_name = next_segment_name(&index_dir)?;
    let new_seg_dir = index_dir.join(&new_seg_name);
    let base_argv = &manifest.segments[0].build_args;
    let build_args = build_args_from(&tmp_jsonl, &new_seg_dir, base_argv)?;
    build::run(build_args).context("building compacted segment")?;

    // Swap the manifest to point only at the new segment, clear tombstones,
    // then remove the old segment dirs and temp source.
    let new_manifest = Manifest {
        format: sift_core::MANIFEST_FORMAT,
        generation: manifest.generation + 1,
        segments: vec![SegmentRef {
            dir: new_seg_name.clone(),
            source: Some(tmp_jsonl_final(&index_dir)),
            build_args: base_argv.clone(),
        }],
    };
    // Persist the compacted source so a future compaction still has a source.
    let final_src = PathBuf::from(tmp_jsonl_final(&index_dir));
    fs::rename(&tmp_jsonl, &final_src)
        .with_context(|| format!("renaming compacted source to {}", final_src.display()))?;
    new_manifest.write(&index_dir)?;
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

fn tmp_jsonl_final(index_dir: &Path) -> String {
    index_dir
        .join("compacted-source.jsonl")
        .display()
        .to_string()
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
