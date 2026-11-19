//! In-process Sift latency: times `Engine::search` over every judged query.
use anyhow::{bail, ensure, Context, Result};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::{json, Value};
use sift::{Engine, SearchOptions};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{Duration, Instant};

const DEPTH: usize = 100;
const MAX_PASSES: usize = 100;
const MAX_ITERATIONS: usize = 100_000;
const WARMUP_MIN: Duration = Duration::from_secs(2);
const PASS_MIN: Duration = Duration::from_secs(1);

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        bail!("usage: siftbench INDEX_DIR DATASET_DIR OUTPUT_JSON SEED PASSES");
    }
    let seed: u64 = args[3].parse().context("seed")?;
    let passes: usize = args[4].parse().context("passes")?;
    ensure!(
        (1..=MAX_PASSES).contains(&passes),
        "passes must be 1..={MAX_PASSES}"
    );
    let engine = Engine::open(&args[0])?;
    let (queries, mut order) = judged_queries(Path::new(&args[1]))?;
    let mut search = |qid: &str| -> Result<Vec<String>> {
        let options = SearchOptions {
            k: DEPTH,
            cache: false,
            with_payload: false,
            ..SearchOptions::new(queries[qid].clone())
        };
        match engine.search(options) {
            Ok(r) => Ok(r.hits.into_iter().map(|h| h.doc_id).collect()),
            Err(e) => bail!("query {qid} failed: {e:?}"),
        }
    };
    let mut rng = StdRng::seed_from_u64(seed);
    // One iteration over a small dataset lasts a few milliseconds. Warmup and each measured pass
    // therefore repeat the query set for a minimum time, so a pass median is not sampling noise.
    let (_, warmup_iterations) = timed_iterations(&mut order, &mut rng, WARMUP_MIN, &mut search)?;
    let mut measured = Vec::new();
    for _ in 0..passes {
        let (samples_us, iterations) =
            timed_iterations(&mut order, &mut rng, PASS_MIN, &mut search)?;
        measured.push(json!({ "samples_us": samples_us, "iterations": iterations }));
    }
    // Rankings for the identity check come from one more iteration, after all timing.
    let mut rankings = BTreeMap::new();
    for qid in &order {
        rankings.insert(qid.clone(), search(qid)?);
    }
    let report = json!({
        "system": "sift-rust",
        "seed": seed,
        "warmup_iterations": warmup_iterations,
        "passes": measured,
        "rankings": rankings,
    });
    std::fs::write(&args[2], serde_json::to_string(&report)?)?;
    Ok(())
}

fn judged_queries(dataset: &Path) -> Result<(BTreeMap<String, String>, Vec<String>)> {
    let mut queries = BTreeMap::new();
    for line in std::fs::read_to_string(dataset.join("queries.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let id = row["_id"].as_str().context("query id")?.to_owned();
        queries.insert(id, row["text"].as_str().context("query text")?.to_owned());
    }
    let mut judged = BTreeSet::new();
    for line in std::fs::read_to_string(dataset.join("qrels.txt"))?.lines() {
        if let Some(id) = line.split_whitespace().next() {
            ensure!(queries.contains_key(id), "judged query {id} has no text");
            judged.insert(id.to_owned());
        }
    }
    Ok((queries, judged.into_iter().collect()))
}

/// Runs the judged queries in a new random order, repeatedly, until at least `min` has passed.
/// Hits are dropped after each query, so no pass keeps memory that a later pass must allocate.
fn timed_iterations(
    order: &mut [String],
    rng: &mut StdRng,
    min: Duration,
    search: &mut impl FnMut(&str) -> Result<Vec<String>>,
) -> Result<(Vec<f64>, usize)> {
    let started = Instant::now();
    let mut samples_us = Vec::new();
    let mut iterations = 0;
    while iterations < MAX_ITERATIONS && (iterations == 0 || started.elapsed() < min) {
        order.shuffle(rng);
        for qid in order.iter() {
            let begun = Instant::now();
            let hits = search(qid)?;
            samples_us.push(begun.elapsed().as_nanos() as f64 / 1000.0);
            drop(hits);
        }
        iterations += 1;
    }
    Ok((samples_us, iterations))
}
