//! Optional ONNX cross-encoder reranker. Loads `<dir>/model-int8.onnx` (or
//! `model.onnx`) plus `<dir>/tokenizer.json`, and scores (query, doc) pairs
//! jointly. Reranking the top-50 costs ~60ms on CPU with the int8
//! MiniLM-L6 model, which is what closes the quality gap to dense
//! retrievers while keeping a single store: the model reorders, it never
//! retrieves.

use anyhow::{anyhow, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

pub(crate) struct CrossEncoder {
    /// ort sessions need &mut for run(); a Mutex serializes inference,
    /// which also stops concurrent reranks from oversubscribing the CPU.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_len: usize,
}

impl CrossEncoder {
    pub(crate) fn load(dir: &Path, threads: usize, max_len: usize) -> Result<Self> {
        let model_path = ["model-int8.onnx", "model.onnx"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| {
                anyhow!(
                    "no model-int8.onnx or model.onnx under {} (export with reranker/teacher tooling)",
                    dir.display()
                )
            })?;
        let session = Session::builder()
            .map_err(|e| anyhow!("ort session builder: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("setting intra threads: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow!("loading {}: {e}", model_path.display()))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading cross-encoder tokenizer: {e}"))?;
        tracing::info!(
            "cross-encoder loaded from {} ({} intra threads, max_len {max_len})",
            model_path.display(),
            threads
        );
        Ok(CrossEncoder {
            session: Mutex::new(session),
            tokenizer,
            max_len,
        })
    }

    /// Relevance logits for `(query, doc)` pairs, one per doc. Higher =
    /// more relevant. Order matches `docs`.
    pub(crate) fn score_pairs(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        // Tokenize each pair, truncating to max_len, then pad to the batch max.
        let mut encs = Vec::with_capacity(docs.len());
        let mut batch_max = 0usize;
        for d in docs {
            let enc = self
                .tokenizer
                .encode((query, d.as_str()), true)
                .map_err(|e| anyhow!("cross-encoder tokenize: {e}"))?;
            let len = enc.get_ids().len().min(self.max_len);
            batch_max = batch_max.max(len);
            encs.push(enc);
        }
        let b = encs.len();
        let s = batch_max.max(1);
        let mut ids = vec![0i64; b * s];
        let mut mask = vec![0i64; b * s];
        let mut ttype = vec![0i64; b * s];
        for (i, enc) in encs.iter().enumerate() {
            let toks = enc.get_ids();
            let types = enc.get_type_ids();
            let n = toks.len().min(s);
            for j in 0..n {
                ids[i * s + j] = toks[j] as i64;
                mask[i * s + j] = 1;
                ttype[i * s + j] = types[j] as i64;
            }
        }
        let shape = [b as i64, s as i64];
        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => Tensor::from_array((shape, ids))?,
                "attention_mask" => Tensor::from_array((shape, mask))?,
                "token_type_ids" => Tensor::from_array((shape, ttype))?,
            ])
            .context("cross-encoder inference")?;
        let (_, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .context("extracting logits")?;
        Ok(logits.to_vec())
    }
}
