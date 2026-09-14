//! Runnable check of the embedded API with a local model and an empty output directory.

use anyhow::{Context, Result};
use serde_json::json;
use sift::{Engine, SearchOptions, WriteMode};

fn exact(index: &Engine, query: &str) -> Result<Vec<String>> {
    let options = SearchOptions {
        blend_alpha: 0.0,
        cache: false,
        ..SearchOptions::new(query)
    };
    Ok(index
        .search(options)?
        .hits
        .into_iter()
        .map(|hit| hit.doc_id)
        .collect())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context("Provide a local model directory.")?;
    let output = args.next().context("Provide an empty output directory.")?;
    anyhow::ensure!(args.next().is_none(), "Unexpected argument.");
    let documents = [
        json!({"id":"cat", "text":"A domestic cat sleeps on the windowsill."}),
        json!({"id":"dog", "text":"A dog plays in the park."}),
        json!({"id":"bird", "text":"A sparrow flies over a tree."}),
    ];
    let mut index = Engine::create(&output, &model, &documents)?;
    assert_eq!(exact(&index, "cat")?, ["cat"]);
    assert!(Engine::create(&output, &model, &documents).is_err());
    let mut snapshot = Engine::open(&output)?;
    index.write_documents(
        &[json!({"id":"cat", "text":"A horse rests near a stable."})],
        WriteMode::Upsert,
    )?;
    assert!(exact(&index, "cat")?.is_empty());
    assert_eq!(exact(&index, "horse")?, ["cat"]);
    index.delete(&["dog".into()])?;
    assert!(exact(&index, "dog")?.is_empty());
    // Compaction must use local sources after the artifact changes location.
    drop(index);
    let moved = std::path::PathBuf::from(&output).with_extension("moved");
    anyhow::ensure!(
        !moved.exists(),
        "The moved output directory already exists."
    );
    std::fs::rename(&output, &moved)?;
    let mut index = Engine::open(&moved)?;
    index.compact()?;
    assert_eq!(exact(&index, "horse")?, ["cat"]);
    index.compact()?;
    assert_eq!(exact(&index, "sparrow")?, ["bird"]);
    drop(index);
    std::fs::rename(&moved, &output)?;
    let mut index = Engine::open(&output)?;
    index.compact()?;
    drop(index);
    let reopened = Engine::open(&output)?;
    assert_eq!(exact(&reopened, "horse")?, ["cat"]);
    assert_eq!(exact(&reopened, "sparrow")?, ["bird"]);
    assert!(exact(&reopened, "dog")?.is_empty());
    assert_eq!(exact(&snapshot, "cat")?, ["cat"]);
    snapshot.reload()?;
    assert!(exact(&snapshot, "cat")?.is_empty());
    assert_eq!(exact(&snapshot, "horse")?, ["cat"]);
    println!("Embedded create, search, upsert, delete, compact, and reopen checks passed.");
    Ok(())
}
