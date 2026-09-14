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

use crate::build::BuildArgs;
use anyhow::{anyhow, Context, Result};
use clap::Args;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

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

pub fn run_add(args: AddArgs) -> Result<()> {
    let outcome = crate::Engine::append_to(&args.index, args.build, crate::WriteMode::Insert)?;
    println!(
        "added {} ({} segment(s), generation {})",
        outcome
            .segment
            .as_deref()
            .expect("Append returns its segment name."),
        outcome.segments,
        outcome.generation
    );
    Ok(())
}

pub fn run_delete(args: DeleteArgs) -> Result<()> {
    let index_dir = args.index.clone();
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
    let outcome = crate::Engine::delete_from(&index_dir, &ids)?;
    println!(
        "tombstoned {} new id(s) ({} requested), generation {}",
        outcome.affected,
        ids.len(),
        outcome.generation
    );
    Ok(())
}

pub fn run_compact(args: CompactArgs) -> Result<()> {
    crate::Engine::compact_at(&args.index)
}
