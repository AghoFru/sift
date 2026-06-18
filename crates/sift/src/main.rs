//! sift - single-binary BM25 with semantic recall, no model, one store.
//!
//! Subcommands:
//!   sift build  --input corpus.jsonl --out artifacts/my.sift
//!   sift serve  --artifacts ./artifacts --bind 0.0.0.0:8080
//!   sift search artifacts/my.sift "your query"

use anyhow::Result;
use clap::{Parser, Subcommand};

mod build;
mod clean;
mod index_cmd;
mod replicate;
mod search;
mod serve;
mod spell;

#[derive(Parser, Debug)]
#[command(
    name = "sift",
    version,
    about = "BM25 with semantic recall, no model, one store.",
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Build a `.sift` index from a JSONL corpus.
    Build(build::BuildArgs),
    /// Serve one or more `.sift` indices over HTTP.
    Serve(serve::ServeArgs),
    /// One-shot CLI search against a single `.sift` index.
    Search(search::SearchArgs),
    /// Append a new segment of documents to an index (incremental add).
    Add(index_cmd::AddArgs),
    /// Tombstone documents by external id (incremental delete).
    Delete(index_cmd::DeleteArgs),
    /// Rebuild a multi-segment index back into a single segment.
    Compact(index_cmd::CompactArgs),
    /// Replicate an index to another directory (consistent, incremental).
    Replicate(replicate::ReplicateArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Build(a) => build::run(a),
        Cmd::Serve(a) => serve::run(a),
        Cmd::Search(a) => search::run(a),
        Cmd::Add(a) => index_cmd::run_add(a),
        Cmd::Delete(a) => index_cmd::run_delete(a),
        Cmd::Compact(a) => index_cmd::run_compact(a),
        Cmd::Replicate(a) => replicate::run(a),
    }
}
