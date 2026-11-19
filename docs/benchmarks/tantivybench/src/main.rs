//! Tantivy BM25 baseline: builds an index from a BEIR-style corpus, then times in-process search.
//!
//! The query is the OR of its terms after the `en_stem` analyzer, scored with Tantivy's BM25.
//! Timing covers query analysis, search, and reading the 100 document IDs from a fast field.
use anyhow::{bail, ensure, Context, Result};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{Duration, Instant};
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, FAST};
use tantivy::tokenizer::TextAnalyzer;
use tantivy::{Index, IndexReader, ReloadPolicy, Searcher, Term};

const DEPTH: usize = 100;
const MAX_PASSES: usize = 100;
const MAX_ITERATIONS: usize = 100_000;
const WARMUP_MIN: Duration = Duration::from_secs(2);
const PASS_MIN: Duration = Duration::from_secs(1);
const WRITER_BUDGET_BYTES: usize = 512 * 1024 * 1024;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build") if args.len() == 3 => build(Path::new(&args[1]), Path::new(&args[2])),
        Some("search") if args.len() == 7 => {
            let seed: u64 = args[4].parse().context("seed")?;
            let passes: usize = args[5].parse().context("passes")?;
            ensure!(
                (1..=MAX_PASSES).contains(&passes),
                "passes must be 1..={MAX_PASSES}"
            );
            let paths = [&args[1], &args[2], &args[3], &args[6]].map(Path::new);
            search(paths[0], paths[1], paths[2], seed, passes, paths[3])
        }
        _ => bail!(
            "usage: tantivybench build CORPUS_JSONL INDEX_DIR\n       \
             tantivybench search INDEX_DIR DATASET_DIR OUTPUT_JSON SEED PASSES TREC_RUN"
        ),
    }
}

fn schema() -> (Schema, Field, Field) {
    let mut builder = Schema::builder();
    let id = builder.add_text_field("id", tantivy::schema::STRING | FAST);
    let indexing = TextFieldIndexing::default()
        .set_tokenizer("en_stem")
        .set_index_option(IndexRecordOption::WithFreqs);
    let body = builder.add_text_field(
        "body",
        TextOptions::default().set_indexing_options(indexing),
    );
    (builder.build(), id, body)
}

/// Indexes title and text joined by a newline, as the IR Bench adapters do, then merges to one
/// segment so search does not depend on how the writer split the corpus.
fn build(corpus: &Path, index_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(index_dir)?;
    let (schema, id, body) = schema();
    let index = Index::create_in_dir(index_dir, schema)?;
    let mut writer =
        index.writer_with_num_threads::<tantivy::TantivyDocument>(1, WRITER_BUDGET_BYTES)?;
    let started = Instant::now();
    let mut count = 0usize;
    for line in std::fs::read_to_string(corpus)?
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let row: Value = serde_json::from_str(line)?;
        let doc_id = row["_id"].as_str().context("document id")?;
        let title = row.get("title").and_then(Value::as_str).unwrap_or("");
        let text = row["text"].as_str().context("document text")?;
        writer.add_document(tantivy::doc!(id => doc_id, body => format!("{title}\n{text}")))?;
        count += 1;
    }
    writer.commit()?;
    let segments = index.searchable_segment_ids()?;
    if segments.len() > 1 {
        writer.merge(&segments).wait()?;
    }
    writer.wait_merging_threads()?;
    let segments = Index::open_in_dir(index_dir)?
        .searchable_segment_ids()?
        .len();
    ensure!(
        segments == 1,
        "expected one segment after merge, found {segments}"
    );
    let seconds = started.elapsed().as_secs_f64();
    println!("{count} documents, one segment, {seconds:.2} s");
    Ok(())
}

/// The opened index. The analyzer and a table of document IDs are loaded once, outside the timed
/// region, as Sift loads its document ID table when it opens an index. Reading each ID from the
/// fast field dictionary at query time took about 1 ms for 100 hits, which would not measure
/// the search itself.
struct Searchable {
    searcher: Searcher,
    body: Field,
    analyzer: TextAnalyzer,
    ids: Vec<String>,
}

impl Searchable {
    fn open(index_dir: &Path) -> Result<(IndexReader, Self)> {
        let index = Index::open_in_dir(index_dir)?;
        let body = index.schema().get_field("body")?;
        let reader: IndexReader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let searcher = reader.searcher();
        ensure!(
            searcher.segment_readers().len() == 1,
            "index must have one segment"
        );
        let column = searcher
            .segment_reader(0)
            .fast_fields()
            .str("id")?
            .context("id fast field")?;
        let mut ids = Vec::with_capacity(searcher.segment_reader(0).max_doc() as usize);
        for doc in 0..searcher.segment_reader(0).max_doc() {
            let ord = column
                .term_ords(doc)
                .next()
                .context("document without id")?;
            let mut value = String::new();
            column.ord_to_str(ord, &mut value)?;
            ids.push(value);
        }
        let analyzer = index.tokenizer_for_field(body)?;
        Ok((
            reader,
            Self {
                searcher,
                body,
                analyzer,
                ids,
            },
        ))
    }

    fn query(&mut self, text: &str) -> Result<Vec<String>> {
        let mut stream = self.analyzer.token_stream(text);
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        while stream.advance() {
            let term = Term::from_field_text(self.body, &stream.token().text);
            let leaf = TermQuery::new(term, IndexRecordOption::WithFreqs);
            clauses.push((Occur::Should, Box::new(leaf)));
        }
        drop(stream);
        let query = BooleanQuery::new(clauses);
        let top = self
            .searcher
            .search(&query, &TopDocs::with_limit(DEPTH).order_by_score())?;
        let mut hits = Vec::with_capacity(top.len());
        for (_, address) in top {
            hits.push(self.ids[address.doc_id as usize].clone());
        }
        Ok(hits)
    }
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

/// Warmup for at least `WARMUP_MIN`, then `passes` measured passes of at least `PASS_MIN` each.
fn search(
    index_dir: &Path,
    dataset: &Path,
    out: &Path,
    seed: u64,
    passes: usize,
    run: &Path,
) -> Result<()> {
    let (_reader, mut engine) = Searchable::open(index_dir)?;
    let (queries, mut order) = judged_queries(dataset)?;
    let mut search = |qid: &str| engine.query(&queries[qid]);
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
    write_trec(run, &rankings)?;
    let report = json!({
        "system": "tantivy",
        "tantivy": "0.27",
        "seed": seed,
        "warmup_iterations": warmup_iterations,
        "passes": measured,
        "rankings": rankings,
    });
    std::fs::write(out, serde_json::to_string(&report)?)?;
    Ok(())
}

/// Scores are written as descending rank values so the evaluator keeps Tantivy's order exactly.
fn write_trec(path: &Path, rankings: &BTreeMap<String, Vec<String>>) -> Result<()> {
    let mut text = String::new();
    for (qid, hits) in rankings {
        for (rank, doc) in hits.iter().enumerate() {
            let score = hits.len() - rank;
            text.push_str(&format!("{qid} Q0 {doc} {} {score} tantivy\n", rank + 1));
        }
    }
    std::fs::write(path, text)?;
    Ok(())
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
