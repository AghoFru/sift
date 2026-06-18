// Corpus readers, model decoding, PPMI expansion, and binary writers.

/// Build corpus-fitted expansion edges from term co-occurrence PPMI.
///
/// Counts nearby terms, scores pairs with positive pointwise mutual
/// information, retains each term's strongest associates, and normalizes
/// their weights onto the same scale as embedding similarities.
fn build_ppmi_edges(
    tokenized: &[Vec<u32>],
    stop_mask: &[u8],
    vocab_size: usize,
    window: usize,
    k: usize,
    min_cooc: u32,
    weight: f32,
) -> Vec<Vec<(u32, f32)>> {
    let mut unigram = vec![0u64; vocab_size];
    let mut total_tokens: u64 = 0;
    let mut cooc: ahash::AHashMap<u64, u32> = ahash::AHashMap::default();
    let mut kept: Vec<u32> = Vec::new();
    for toks in tokenized {
        kept.clear();
        kept.extend(toks.iter().copied().filter(|&t| stop_mask[t as usize] == 0));
        for (idx, &a) in kept.iter().enumerate() {
            unigram[a as usize] += 1;
            total_tokens += 1;
            let hi = (idx + 1 + window).min(kept.len());
            for &b in &kept[idx + 1..hi] {
                if a == b {
                    continue;
                }
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let key = ((lo as u64) << 32) | hi as u64;
                *cooc.entry(key).or_insert(0) += 1;
            }
        }
    }
    if total_tokens == 0 || cooc.is_empty() {
        return Vec::new();
    }
    let total_pairs: u64 = cooc.values().map(|&c| c as u64).sum();
    let tt = total_tokens as f64;
    let tp = total_pairs as f64;
    // Per-term associate lists with raw PPMI; track the global max for scaling.
    let mut per_term: Vec<Vec<(u32, f32)>> = vec![Vec::new(); vocab_size];
    let mut max_ppmi = 1e-9f64;
    for (&key, &c) in &cooc {
        if c < min_cooc {
            continue;
        }
        let a = (key >> 32) as u32;
        let b = (key & 0xffff_ffff) as u32;
        let pa = unigram[a as usize] as f64 / tt;
        let pb = unigram[b as usize] as f64 / tt;
        if pa <= 0.0 || pb <= 0.0 {
            continue;
        }
        let pab = c as f64 / tp;
        let ppmi = (pab / (pa * pb)).ln();
        if ppmi <= 0.0 {
            continue;
        }
        if ppmi > max_ppmi {
            max_ppmi = ppmi;
        }
        per_term[a as usize].push((b, ppmi as f32));
        per_term[b as usize].push((a, ppmi as f32));
    }
    // Keep top-k per term and rescale weights to [0, weight] by the global max.
    let scale = weight / max_ppmi as f32;
    for assocs in per_term.iter_mut() {
        if assocs.len() > k {
            assocs.sort_unstable_by(|x, y| {
                y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            assocs.truncate(k);
        }
        for e in assocs.iter_mut() {
            e.1 *= scale;
        }
    }
    per_term
}

fn load_model_files(repo_or_path: &str) -> Result<(PathBuf, Vec<u8>)> {
    let p = Path::new(repo_or_path);
    let (tok_path, mdl_path) = if p.exists() {
        (p.join("tokenizer.json"), p.join("model.safetensors"))
    } else {
        let api = Api::new().context("hf-hub init")?;
        let repo = api.model(repo_or_path.to_string());
        let t = repo
            .get("tokenizer.json")
            .context("downloading tokenizer.json")?;
        let m = repo
            .get("model.safetensors")
            .context("downloading model.safetensors")?;
        (t, m)
    };
    let bytes = fs::read(&mdl_path).with_context(|| format!("reading {}", mdl_path.display()))?;
    Ok((tok_path, bytes))
}

fn load_embeddings(model_bytes: &[u8], vocab_size: usize) -> Result<Vec<Vec<f32>>> {
    let st = SafeTensors::deserialize(model_bytes).context("parsing safetensors")?;
    let tensor = st
        .tensor("embeddings")
        .or_else(|_| st.tensor("0"))
        .context("'embeddings' tensor not found")?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(anyhow!("embedding tensor is not 2-D, got shape {shape:?}"));
    }
    let rows = shape[0];
    let cols = shape[1];
    if rows != vocab_size {
        return Err(anyhow!(
            "embedding rows {rows} != tokenizer vocab_size {vocab_size}"
        ));
    }
    let raw = tensor.data();
    let floats: Vec<f32> = match tensor.dtype() {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
            .collect(),
        Dtype::I8 => raw.iter().map(|&b| (b as i8) as f32).collect(),
        other => return Err(anyhow!("unsupported tensor dtype {other:?}")),
    };
    if floats.len() != rows * cols {
        return Err(anyhow!(
            "embedding tensor length {} != {rows}*{cols}",
            floats.len()
        ));
    }
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut v: Vec<f32> = floats[r * cols..(r + 1) * cols].to_vec();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in &mut v {
                *x /= n;
            }
        }
        out.push(v);
    }
    Ok(out)
}

fn build_keep_mask(tokenizer: &Tokenizer) -> Vec<u8> {
    let vocab = tokenizer.get_vocab(true);
    let mut keep = vec![0u8; tokenizer.get_vocab_size(true)];
    for (tok_str, id) in vocab.iter() {
        if SPECIAL.iter().any(|s| s == tok_str) {
            continue;
        }
        let clean = tok_str.strip_prefix("##").unwrap_or(tok_str.as_str());
        if clean.is_empty() {
            continue;
        }
        if !clean.chars().any(|c| c.is_alphanumeric()) {
            continue;
        }
        let chars: Vec<char> = clean.chars().collect();
        if chars.len() == 1 && !chars[0].is_alphanumeric() {
            continue;
        }
        keep[*id as usize] = 1;
    }
    keep
}

#[derive(serde::Deserialize)]
struct JsonlDoc {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "_id")]
    underscore_id: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    title: Option<String>,
    /// Catches all remaining fields so build can pull custom-rank attribute
    /// values out of the same row without per-field structs.
    #[serde(flatten, default)]
    extras: std::collections::HashMap<String, serde_json::Value>,
}

/// Returns (doc_ids, snippets, tokenized_full_text).
/// Map any non-alphanumeric character in a field name to '_' so the filename
/// is portable. Caller is responsible for declaring the original name in the
/// rank_fields.json manifest so the server can map back.
fn sanitize_field_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Walk the JSONL once and pull out an f32 per (field, doc). Missing /
/// non-numeric values become 0.0. Errors out if the row count doesn't match
/// the corpus tokenization pass.
fn extract_rank_fields(
    path: &Path,
    format: &str,
    fields: &[String],
    expected_n: usize,
) -> Result<Vec<Vec<f32>>> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let rdr = BufReader::new(f);
    let _ = format; // currently no per-format dispatch; both jsonl and beir share the row shape
    let mut out: Vec<Vec<f32>> = vec![Vec::with_capacity(expected_n); fields.len()];
    for (i, line) in rdr.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.trim().is_empty() {
            continue;
        }
        let d: JsonlDoc = serde_json::from_str(&line)
            .with_context(|| format!("invalid JSONL line at row {i}: {line}"))?;
        for (idx, field) in fields.iter().enumerate() {
            let v = d
                .extras
                .get(field.as_str())
                .and_then(|v| match v {
                    serde_json::Value::Number(n) => n.as_f64().map(|x| x as f32),
                    serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                    serde_json::Value::String(s) => s.parse::<f32>().ok(),
                    _ => None,
                })
                .unwrap_or(0.0);
            out[idx].push(v);
        }
    }
    for (idx, field) in fields.iter().enumerate() {
        if out[idx].len() != expected_n {
            anyhow::bail!(
                "rank field '{field}' produced {} values; expected {expected_n} (corpus row count)",
                out[idx].len()
            );
        }
    }
    Ok(out)
}

fn apply_clean(text: &str, mode: &str) -> String {
    match mode {
        "html" => crate::clean::strip_html(text),
        "md" => crate::clean::strip_markdown(text),
        "auto" => crate::clean::detect_and_clean(text),
        _ => text.to_string(),
    }
}

/// `(doc_ids, snippets, per-doc token-id sequences, per-doc title token-id
/// sequences)` produced by the corpus tokenization pass. The title streams
/// are empty unless `want_titles` is set (used for BM25F title weighting).
type TokenizedCorpus = (Vec<String>, Vec<String>, Vec<Vec<u32>>, Vec<Vec<u32>>);

#[allow(clippy::too_many_arguments)]
fn tokenize_corpus(
    path: &Path,
    format: &str,
    tokenizer: &Tokenizer,
    keep_mask: &[u8],
    snippet_chars: usize,
    no_normalize: bool,
    clean_mode: &str,
    want_titles: bool,
) -> Result<TokenizedCorpus> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let rdr = BufReader::new(f);
    let lines: Vec<String> = rdr
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .collect();

    // first pass: parse → (id, title, body) - title kept separate so we can
    // build the per-doc title-term set for BM25F.
    let parsed: Vec<(String, String, String)> = lines
        .into_iter()
        .map(|line| -> Result<(String, String, String)> {
            let d: JsonlDoc = serde_json::from_str(&line)
                .with_context(|| format!("invalid JSONL line: {line}"))?;
            let id =
                d.id.or(d.underscore_id)
                    .ok_or_else(|| anyhow!("doc missing id/_id"))?;
            let title = if format == "beir" {
                d.title.unwrap_or_default()
            } else {
                String::new()
            };
            let body = d.text.or(d.body).unwrap_or_default();
            let (title, body) = if clean_mode == "off" {
                (title, body)
            } else {
                (
                    apply_clean(&title, clean_mode),
                    apply_clean(&body, clean_mode),
                )
            };
            Ok((id, title, body))
        })
        .collect::<Result<_>>()?;

    // snippets
    let doc_ids: Vec<String> = parsed.iter().map(|(id, _, _)| id.clone()).collect();
    // Snippets include both title and body when a title is present.
    let snips: Vec<String> = parsed
        .iter()
        .map(|(_, title, body)| {
            let combined = if title.is_empty() {
                body.clone()
            } else {
                format!("{title} {body}")
            };
            combined
                .chars()
                .take(snippet_chars)
                .collect::<String>()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();

    // tokenize: outer rayon over docs, single-doc encode_fast inside (no offsets).
    // Apply the same normalization the query path uses so vocab matches.
    let encode_one = |text: &str| -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let normalized_storage: String;
        let to_encode: &str = if no_normalize {
            text
        } else {
            normalized_storage = sift_core::normalize_text(text);
            &normalized_storage[..]
        };
        // SAFETY: normalized_storage outlives this call's use of to_encode
        // because the match runs immediately and produces an owned Vec<u32>.
        match tokenizer.encode_fast(to_encode, false) {
            Ok(enc) => enc
                .get_ids()
                .iter()
                .copied()
                .filter(|&id| (id as usize) < keep_mask.len() && keep_mask[id as usize] == 1)
                .collect(),
            Err(_) => Vec::new(),
        }
    };
    let encode_doc = |title: &str, body: &str| -> Vec<u32> {
        let combined_text: String = if title.is_empty() {
            body.to_string()
        } else {
            format!("{title} {body}")
        };
        encode_one(&combined_text)
    };
    #[cfg(feature = "multithread")]
    let tokenized: Vec<Vec<u32>> = parsed
        .par_iter()
        .map(|(_, ti, bo)| encode_doc(ti.as_str(), bo.as_str()))
        .collect();
    #[cfg(not(feature = "multithread"))]
    let tokenized: Vec<Vec<u32>> = parsed
        .iter()
        .map(|(_, ti, bo)| encode_doc(ti.as_str(), bo.as_str()))
        .collect();
    // Title-only streams for field weighting. Tokenized standalone, which can
    // differ from the combined stream at the title/body boundary by at most
    // one subword; the weighting is robust to that.
    let tokenized_titles: Vec<Vec<u32>> = if want_titles {
        #[cfg(feature = "multithread")]
        {
            parsed.par_iter().map(|(_, ti, _)| encode_one(ti)).collect()
        }
        #[cfg(not(feature = "multithread"))]
        {
            parsed.iter().map(|(_, ti, _)| encode_one(ti)).collect()
        }
    } else {
        Vec::new()
    };
    Ok((doc_ids, snips, tokenized, tokenized_titles))
}

/// Re-read the corpus JSONL and produce the per-doc normalized text strings
/// that the spell-vocab pass needs. Mirrors the parsing in `tokenize_corpus`.
fn reread_doc_texts(
    path: &Path,
    format: &str,
    no_normalize: bool,
    clean_mode: &str,
) -> Result<Vec<String>> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let rdr = BufReader::new(f);
    let mut out: Vec<String> = Vec::new();
    for line in rdr.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.trim().is_empty() {
            continue;
        }
        let d: JsonlDoc =
            serde_json::from_str(&line).with_context(|| format!("invalid JSONL line: {line}"))?;
        let title = if format == "beir" {
            d.title.unwrap_or_default()
        } else {
            String::new()
        };
        let body = d.text.or(d.body).unwrap_or_default();
        let (title, body) = if clean_mode == "off" {
            (title, body)
        } else {
            (
                apply_clean(&title, clean_mode),
                apply_clean(&body, clean_mode),
            )
        };
        let raw = if title.is_empty() {
            body
        } else {
            format!("{title} {body}")
        };
        let normalized = if no_normalize {
            raw
        } else {
            sift_core::normalize_text(&raw)
        };
        out.push(normalized);
    }
    Ok(out)
}

/// Re-read the source rows and return the canonical (compact) JSON for each, in
/// the same doc order as `tokenize_corpus` (same empty-line skipping). Stored as
/// the per-doc payload so results can return the source and filters can match on
/// arbitrary fields.
fn reread_payloads(path: &Path) -> Result<Vec<String>> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let rdr = BufReader::new(f);
    let mut out: Vec<String> = Vec::new();
    for line in rdr.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("invalid JSONL line for payload: {line}"))?;
        out.push(v.to_string());
    }
    Ok(out)
}

fn pack_strings(strings: &[String]) -> (Vec<u8>, Vec<u64>) {
    let mut buf = Vec::new();
    let mut off = Vec::with_capacity(strings.len() + 1);
    off.push(0u64);
    for s in strings {
        buf.extend_from_slice(s.as_bytes());
        off.push(buf.len() as u64);
    }
    (buf, off)
}

trait Pod {
    fn as_bytes_view(&self) -> &[u8];
}

impl Pod for Vec<f32> {
    fn as_bytes_view(&self) -> &[u8] {
        let ptr = self.as_ptr() as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, self.len() * 4) }
    }
}
impl Pod for Vec<u32> {
    fn as_bytes_view(&self) -> &[u8] {
        let ptr = self.as_ptr() as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, self.len() * 4) }
    }
}
impl Pod for Vec<u64> {
    fn as_bytes_view(&self) -> &[u8] {
        let ptr = self.as_ptr() as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, self.len() * 8) }
    }
}
impl Pod for Vec<u8> {
    fn as_bytes_view(&self) -> &[u8] {
        self.as_slice()
    }
}
impl Pod for Vec<f16> {
    fn as_bytes_view(&self) -> &[u8] {
        let ptr = self.as_ptr() as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, self.len() * 2) }
    }
}

fn write_bin<P: AsRef<Path>, T: Pod>(path: P, data: &T) -> Result<()> {
    let mut f =
        File::create(&path).with_context(|| format!("creating {}", path.as_ref().display()))?;
    f.write_all(data.as_bytes_view())?;
    Ok(())
}
