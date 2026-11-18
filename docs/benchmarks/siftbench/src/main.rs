//! In-process Sift latency: times Engine::search over every judged query.
use anyhow::{bail, Context, Result};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::{json, Value};
use sift::{Engine, SearchOptions};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        bail!("usage: siftbench INDEX_DIR DATASET_DIR OUTPUT_JSON");
    }
    let engine = Engine::open(&args[0])?;
    let dataset = std::path::Path::new(&args[1]);
    let mut queries = BTreeMap::new();
    for line in std::fs::read_to_string(dataset.join("queries.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let id = row["_id"].as_str().context("query id")?.to_owned();
        queries.insert(id, row["text"].as_str().context("query text")?.to_owned());
    }
    let mut judged = BTreeSet::new();
    for line in std::fs::read_to_string(dataset.join("qrels.txt"))?.lines() {
        if let Some(id) = line.split_whitespace().next() {
            judged.insert(id.to_owned());
        }
    }
    let mut order: Vec<String> = judged.into_iter().collect();
    let mut rng = StdRng::seed_from_u64(0);
    let (warmup, repeats) = (1, 2);
    let mut samples_us = Vec::new();
    let mut rankings = serde_json::Map::new();
    for pass in 0..(warmup + repeats) {
        order.shuffle(&mut rng);
        for id in &order {
            let options = SearchOptions {
                k: 100,
                cache: false,
                with_payload: false,
                ..SearchOptions::new(queries[id].clone())
            };
            let started = Instant::now();
            let response = engine.search(options);
            let elapsed = started.elapsed().as_nanos() as f64 / 1000.0;
            let hits: Vec<String> = match response {
                Ok(r) => r.hits.into_iter().map(|h| h.doc_id).collect(),
                Err(e) => bail!("query {id} failed: {e:?}"),
            };
            if pass >= warmup {
                samples_us.push(elapsed);
            }
            if pass == warmup {
                rankings.insert(id.clone(), json!(hits));
            }
        }
    }
    let mut sorted = samples_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median = if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 };
    let p95 = sorted[((n as f64 * 0.95).ceil() as usize) - 1];
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let out = json!({
        "boundary": "Rust Engine::search wall time in process, k=100, cache off, no payload, one thread",
        "warmup_passes": warmup, "measured_passes": repeats, "query_order_seed": 0,
        "samples": n, "p50_us": median, "p95_us": p95, "mean_us": mean,
        "rankings": rankings,
    });
    std::fs::write(&args[2], serde_json::to_string(&out)?)?;
    println!("{} samples p50 {:.1} us p95 {:.1} us mean {:.1} us", n, median, p95, mean);
    Ok(())
}
