//! Shared index mutations for embedded callers, the CLI, and HTTP.

use crate::build::build_args_to_argv;
use crate::build::{self, BuildArgs, Model};
mod compaction;
use anyhow::{anyhow, Context, Result};
pub(crate) use compaction::compact;
use fs2::FileExt;
use serde::Serialize;
use sift_core::{add_tombstones, next_segment_name, Manifest, SegmentRef};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// An index write lock released by the operating system when its file closes.
pub(crate) struct WriteGuard {
    _file: File,
}

impl WriteGuard {
    pub(crate) fn acquire(path: &Path) -> Result<Self> {
        fs::create_dir_all(path).with_context(|| format!("Creating {}", path.display()))?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path.join(".sift-write.lock"))?;
        file.try_lock_exclusive()
            .with_context(|| format!("Cannot acquire the write lock for {}", path.display()))?;
        Ok(Self { _file: file })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    Create,
    Insert,
    Upsert,
}

#[derive(Serialize)]
pub struct WriteOutcome {
    pub generation: u64,
    pub segments: usize,
    pub affected: usize,
    pub segment: Option<String>,
}

pub(crate) fn validate_id(id: &str) -> Result<()> {
    if id.trim() != id || id.is_empty() || id.contains(['\n', '\r', '\t']) {
        return Err(anyhow!("Document identifiers cannot be empty. Remove surrounding whitespace, tabs, and newlines."));
    }
    Ok(())
}

pub(crate) fn encode_documents(documents: &[serde_json::Value]) -> Result<String> {
    if documents.is_empty() || documents.len() > 100_000 {
        return Err(anyhow!(
            "Provide from 1 through 100000 documents per write."
        ));
    }
    let mut jsonl = String::new();
    for (number, document) in documents.iter().enumerate() {
        let mut row = document
            .as_object()
            .cloned()
            .ok_or_else(|| anyhow!("Document {number} must be a JSON object."))?;
        let id = match row.get("id") {
            Some(serde_json::Value::String(value)) => value.clone(),
            Some(serde_json::Value::Number(value)) => value.to_string(),
            _ => return Err(anyhow!("Document {number} requires a string or number id.")),
        };
        validate_id(&id)?;
        if !row.get("text").is_some_and(|value| value.is_string()) {
            return Err(anyhow!("Document {number} requires a text string."));
        }
        row.insert("id".into(), id.into());
        jsonl.push_str(&serde_json::to_string(&row)?);
        jsonl.push('\n');
        if jsonl.len() > 64 * 1024 * 1024 {
            return Err(anyhow!("Serialized documents exceed 64 MiB."));
        }
    }
    Ok(jsonl)
}

struct InputFile(PathBuf);

impl Drop for InputFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub(crate) fn append_documents(
    path: &Path,
    jsonl: &str,
    model_name: &str,
    model: &Model,
    mode: WriteMode,
) -> Result<WriteOutcome> {
    static NEXT_INPUT: AtomicU64 = AtomicU64::new(0);
    fs::create_dir_all(path)?;
    let ordinal = NEXT_INPUT.fetch_add(1, Ordering::Relaxed);
    let source = InputFile(path.join(format!(
        ".sift-input-{}-{ordinal}.jsonl",
        std::process::id()
    )));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&source.0)?;
    file.write_all(jsonl.as_bytes())?;
    drop(file);
    let mut options = BuildArgs::new(&source.0, path)?;
    options.model = model_name.to_string();
    options.block_max = true;
    append(path, options, model, mode)
}

fn source_ids(input: &Path) -> Result<Vec<String>> {
    let mut ids = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let file = File::open(input).with_context(|| format!("Opening {}", input.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(&line)?;
        let id = row
            .get("id")
            .or_else(|| row.get("_id"))
            .and_then(|id| id.as_str())
            .ok_or_else(|| anyhow!("Each document requires a string id or _id."))?;
        validate_id(id)?;
        if !seen.insert(id.to_string()) {
            return Err(anyhow!("Duplicate document identifier: {id}"));
        }
        ids.push(id.to_string());
    }
    if ids.is_empty() {
        return Err(anyhow!("No documents were provided."));
    }
    Ok(ids)
}

pub(crate) fn append(
    path: &Path,
    mut options: BuildArgs,
    model: &Model,
    mode: WriteMode,
) -> Result<WriteOutcome> {
    let ids = source_ids(&options.input)?;
    let _guard = WriteGuard::acquire(path)?;
    if mode == WriteMode::Create
        && (path.join("meta.json").exists() || path.join("manifest.json").exists())
    {
        return Err(anyhow!("An index already exists at {}", path.display()));
    }
    let mut manifest = Manifest::ensure(path)?;
    let recipe = build_args_to_argv(&options);
    let name = next_segment_name(path)?;
    let segment = path.join(&name);
    fs::create_dir(&segment)?;
    let source = segment.join("source.jsonl");
    let built = (|| -> Result<()> {
        fs::copy(&options.input, &source).context("Copying the segment source.")?;
        options.input = source.clone();
        options.output = Some(segment.clone());
        build::run_with_model(options, model)?;
        sift_core::fsync_dir_contents(&segment)?;
        Ok(())
    })();
    if let Err(error) = built {
        fs::remove_dir_all(&segment).context("Removing the failed segment.")?;
        return Err(error);
    }
    manifest.segments.push(SegmentRef {
        dir: name.clone(),
        source: Some(source.canonicalize()?.display().to_string()),
        build_args: recipe,
    });
    manifest.generation = manifest
        .generation
        .checked_add(1)
        .ok_or_else(|| anyhow!("Index generation limit reached."))?;
    manifest.write(path)?;
    if mode == WriteMode::Upsert {
        add_tombstones(path, &ids, sift_core::parse_seg_ordinal(&name))?;
    }
    Ok(WriteOutcome {
        generation: manifest.generation,
        segments: manifest.segments.len(),
        affected: ids.len(),
        segment: Some(name),
    })
}

pub(crate) fn delete(path: &Path, ids: &[String]) -> Result<WriteOutcome> {
    if ids.is_empty() {
        return Err(anyhow!("No document identifiers were provided."));
    }
    for id in ids {
        validate_id(id)?;
    }
    if !path.join("meta.json").exists() && !path.join("manifest.json").exists() {
        return Err(anyhow!("No index exists at {}", path.display()));
    }
    let _guard = WriteGuard::acquire(path)?;
    let mut manifest = Manifest::ensure(path)?;
    let affected = add_tombstones(path, ids, sift_core::TOMBSTONE_ALL)?;
    manifest.generation = manifest
        .generation
        .checked_add(1)
        .ok_or_else(|| anyhow!("Index generation limit reached."))?;
    manifest.write(path)?;
    Ok(WriteOutcome {
        generation: manifest.generation,
        segments: manifest.segments.len(),
        affected,
        segment: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn document_validation_preserves_fields_and_rejects_invalid_identifiers() {
        let encoded = encode_documents(&[json!({"id":42,"text":"cat","tag":"pet"})]).unwrap();
        let decoded: serde_json::Value = serde_json::from_str(encoded.trim()).unwrap();
        assert_eq!(decoded, json!({"id":"42","text":"cat","tag":"pet"}));
        for id in [
            json!(true),
            json!([]),
            json!(""),
            json!("a\nb"),
            json!(" a"),
        ] {
            assert!(encode_documents(&[json!({"id":id,"text":"cat"})]).is_err());
        }
        assert!(encode_documents(&[]).is_err());
        assert!(encode_documents(&[json!({"id":"cat","text":null})]).is_err());
    }

    #[test]
    fn kernel_lock_rejects_another_writer_and_releases_on_close() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target")
            .join(format!("write-lock-test-{}", std::process::id()));
        let guard = WriteGuard::acquire(&path)?;
        assert!(WriteGuard::acquire(&path).is_err());
        drop(guard);
        drop(WriteGuard::acquire(&path)?);
        fs::remove_dir_all(path)?;
        Ok(())
    }
}
