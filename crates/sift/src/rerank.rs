//! Native GBDT reranker: evaluates a LightGBM `dump_model()` JSON over the
//! per-hit feature vector from `score_with_features`. A few hundred shallow
//! trees over 14 features cost ~1µs per doc, so reranking the top-100 adds
//! ~100µs to a request: no model server, no ONNX runtime, one store.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use sift_core::FeatureHit;
use std::path::Path;

/// Feature count for the distribution-safe model trained by
/// `reranker/train_lgbm.py`.
pub(crate) const N_FEATURES: usize = 14;
const LEGACY_FEATURES: usize = 8;

#[derive(Clone, Copy)]
pub(crate) enum FeatureSchema {
    LegacyRaw,
    RelativeV1,
}

/// One flattened decision tree. Internal nodes are stored in arrays; a
/// negative child index `-i` means "leaf i".
struct Tree {
    split_feature: Vec<u16>,
    threshold: Vec<f32>,
    left: Vec<i32>,
    right: Vec<i32>,
    default_left: Vec<bool>,
    leaf_value: Vec<f32>,
}

impl Tree {
    fn eval(&self, feats: &[f32]) -> f32 {
        if self.split_feature.is_empty() {
            // single-leaf tree
            return self.leaf_value.first().copied().unwrap_or(0.0);
        }
        let mut node: i32 = 0;
        loop {
            let i = node as usize;
            let f = feats[self.split_feature[i] as usize];
            let go_left = if f.is_nan() {
                self.default_left[i]
            } else {
                f <= self.threshold[i]
            };
            node = if go_left { self.left[i] } else { self.right[i] };
            if node < 0 {
                return self.leaf_value[(-node - 1) as usize];
            }
        }
    }
}

pub(crate) struct GbdtModel {
    trees: Vec<Tree>,
    pub(crate) n_features: usize,
    pub(crate) schema: FeatureSchema,
    pub(crate) tree_weight: f32,
}

impl GbdtModel {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let txt = std::fs::read_to_string(path)
            .with_context(|| format!("reading reranker model {}", path.display()))?;
        let v: Value = serde_json::from_str(&txt).context("parsing reranker JSON")?;
        let schema = match v
            .get("sift_feature_schema")
            .and_then(|value| value.as_str())
        {
            None => FeatureSchema::LegacyRaw,
            Some("relative-v1") => FeatureSchema::RelativeV1,
            Some(name) => return Err(anyhow!("unsupported Sift reranker schema '{name}'")),
        };
        let tree_weight = v
            .get("sift_tree_weight")
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let max_features = match schema {
            FeatureSchema::LegacyRaw => LEGACY_FEATURES,
            FeatureSchema::RelativeV1 => N_FEATURES,
        };
        let n_features = v
            .get("max_feature_idx")
            .and_then(|x| x.as_i64())
            .map(|x| (x + 1) as usize)
            .unwrap_or(N_FEATURES);
        if n_features > max_features {
            return Err(anyhow!(
                "reranker schema needs {n_features} features, but this Sift build supports {max_features}"
            ));
        }
        let tree_info = v
            .get("tree_info")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow!("model JSON has no tree_info (use booster.dump_model())"))?;
        let mut trees = Vec::with_capacity(tree_info.len());
        for t in tree_info {
            let root = t
                .get("tree_structure")
                .ok_or_else(|| anyhow!("tree missing tree_structure"))?;
            let mut tree = Tree {
                split_feature: Vec::new(),
                threshold: Vec::new(),
                left: Vec::new(),
                right: Vec::new(),
                default_left: Vec::new(),
                leaf_value: Vec::new(),
            };
            flatten(root, &mut tree)?;
            trees.push(tree);
        }
        Ok(GbdtModel {
            trees,
            n_features,
            schema,
            tree_weight,
        })
    }

    /// Sum of leaf values across trees, raw LambdaMART score. Only the
    /// ordering matters).
    pub(crate) fn score(&self, feats: &[f32]) -> f32 {
        self.trees.iter().map(|t| t.eval(feats)).sum()
    }
}

/// Recursively flatten a LightGBM node into the tree's arrays. Returns the
/// node's encoded index: >= 0 for internal nodes, -(leaf_idx + 1) for leaves.
fn flatten(node: &Value, tree: &mut Tree) -> Result<i32> {
    if let Some(lv) = node.get("leaf_value") {
        let val = lv.as_f64().ok_or_else(|| anyhow!("bad leaf_value"))? as f32;
        tree.leaf_value.push(val);
        return Ok(-(tree.leaf_value.len() as i32));
    }
    let sf = node
        .get("split_feature")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| anyhow!("node missing split_feature"))? as u16;
    let th = node
        .get("threshold")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| anyhow!("node missing threshold"))? as f32;
    let dt = node
        .get("decision_type")
        .and_then(|x| x.as_str())
        .unwrap_or("<=");
    if dt != "<=" {
        return Err(anyhow!(
            "unsupported decision_type '{dt}' (numeric splits only)"
        ));
    }
    let dl = node
        .get("default_left")
        .and_then(|x| x.as_bool())
        .unwrap_or(true);

    let idx = tree.split_feature.len();
    tree.split_feature.push(sf);
    tree.threshold.push(th);
    tree.default_left.push(dl);
    // placeholders; children may push more internal nodes before we know ours
    tree.left.push(0);
    tree.right.push(0);

    let lc = node
        .get("left_child")
        .ok_or_else(|| anyhow!("node missing left_child"))?;
    let rc = node
        .get("right_child")
        .ok_or_else(|| anyhow!("node missing right_child"))?;
    let l = flatten(lc, tree)?;
    let r = flatten(rc, tree)?;
    tree.left[idx] = l;
    tree.right[idx] = r;
    Ok(idx as i32)
}

/// Query-local statistics used to remove corpus-dependent score thresholds.
pub(crate) struct FeatureStats {
    max_combined: f32,
    max_exact: f32,
    max_semantic: f32,
    max_blended: f32,
    max_bigram: f32,
    max_qexp: f32,
    max_composition: f32,
    max_retrieval: f32,
    max_log_doc_len: f32,
    candidate_count: usize,
}

impl FeatureStats {
    pub(crate) fn from_hits(hits: &[FeatureHit]) -> Self {
        let mut stats = FeatureStats {
            max_combined: 0.0,
            max_exact: 0.0,
            max_semantic: 0.0,
            max_blended: 0.0,
            max_bigram: 0.0,
            max_qexp: 0.0,
            max_composition: 0.0,
            max_retrieval: 0.0,
            max_log_doc_len: 0.0,
            candidate_count: hits.len(),
        };
        for hit in hits {
            stats.max_combined = stats.max_combined.max(hit.bm25_combined);
            stats.max_exact = stats.max_exact.max(hit.bm25_exact);
            stats.max_semantic = stats.max_semantic.max(hit.bm25_semantic);
            stats.max_blended = stats.max_blended.max(hit.bm25_blended);
            stats.max_bigram = stats.max_bigram.max(hit.bigram_bonus);
            stats.max_qexp = stats.max_qexp.max(hit.qexp_score);
            stats.max_composition = stats.max_composition.max(hit.composition_similarity);
            stats.max_retrieval = stats.max_retrieval.max(hit.retrieval_score);
            stats.max_log_doc_len = stats.max_log_doc_len.max((1.0 + hit.doc_len).ln());
        }
        stats.max_combined = stats.max_combined.max(1e-6);
        stats.max_exact = stats.max_exact.max(1e-6);
        stats.max_semantic = stats.max_semantic.max(1e-6);
        stats.max_blended = stats.max_blended.max(1e-6);
        stats.max_bigram = stats.max_bigram.max(1e-6);
        stats.max_qexp = stats.max_qexp.max(1e-6);
        stats.max_composition = stats.max_composition.max(1e-6);
        stats.max_retrieval = stats.max_retrieval.max(1e-6);
        stats.max_log_doc_len = stats.max_log_doc_len.max(1e-6);
        stats
    }
}

/// Build the legacy 8-feature vector. It remains available for existing
/// models that predate the relative feature schema.
#[inline]
pub(crate) fn legacy_feature_vector(
    bm25_combined: f32,
    bm25_exact: f32,
    bm25_semantic: f32,
    coverage: f32,
    doc_len: f32,
    rank: usize,
) -> [f32; LEGACY_FEATURES] {
    [
        bm25_combined,
        bm25_exact,
        bm25_semantic,
        coverage,
        doc_len,
        (1.0 + doc_len).ln(),
        bm25_semantic / bm25_combined.max(1e-6),
        1.0 / (rank as f32 + 1.0),
    ]
}

/// Build the distribution-safe feature vector. Every score and document
/// length is relative to the current candidate window, while ratios preserve
/// the exact versus semantic evidence for the document.
#[inline]
pub(crate) fn feature_vector(
    hit: &FeatureHit,
    rank: usize,
    stats: &FeatureStats,
) -> [f32; N_FEATURES] {
    let rank_fraction = if stats.candidate_count <= 1 {
        1.0
    } else {
        1.0 - rank as f32 / (stats.candidate_count - 1) as f32
    };
    [
        hit.retrieval_score / stats.max_retrieval,
        hit.bm25_blended / stats.max_blended,
        hit.bm25_combined / stats.max_combined,
        hit.bm25_exact / stats.max_exact,
        hit.bm25_semantic / stats.max_semantic,
        hit.bigram_bonus / stats.max_bigram,
        hit.qexp_score / stats.max_qexp,
        hit.composition_similarity / stats.max_composition,
        hit.bm25_semantic / hit.bm25_combined.max(1e-6),
        hit.bm25_exact / hit.bm25_combined.max(1e-6),
        hit.coverage,
        (1.0 + hit.doc_len).ln() / stats.max_log_doc_len,
        1.0 / (rank as f32 + 1.0),
        rank_fraction,
    ]
}

/// Keep the sparse ordering as a corpus-independent prior. Tree outputs are
/// normalized per query because their raw scale depends on the training set.
pub(crate) fn blend_with_sparse_prior(
    scores: &[(f32, f32, u32)],
    tree_weight: f32,
) -> Vec<(f32, u32)> {
    let tree_min = scores
        .iter()
        .map(|(tree, _, _)| *tree)
        .fold(f32::INFINITY, f32::min);
    let tree_max = scores
        .iter()
        .map(|(tree, _, _)| *tree)
        .fold(f32::NEG_INFINITY, f32::max);
    let sparse_min = scores
        .iter()
        .map(|(_, sparse, _)| *sparse)
        .fold(f32::INFINITY, f32::min);
    let sparse_max = scores
        .iter()
        .map(|(_, sparse, _)| *sparse)
        .fold(f32::NEG_INFINITY, f32::max);
    let tree_range = (tree_max - tree_min).max(1e-6);
    let sparse_range = (sparse_max - sparse_min).max(1e-6);
    scores
        .iter()
        .map(|(tree, sparse, doc_idx)| {
            let tree_score = (*tree - tree_min) / tree_range;
            let sparse_score = (*sparse - sparse_min) / sparse_range;
            let score = tree_weight * tree_score + (1.0 - tree_weight) * sparse_score;
            (score, *doc_idx)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_matches_hand_tree() {
        // tree: f0 <= 1.0 ? 0.5 : (f1 <= 2.0 ? -0.25 : 0.75)
        let json = serde_json::json!({
            "max_feature_idx": 1,
            "tree_info": [{ "tree_structure": {
                "split_feature": 0, "threshold": 1.0, "decision_type": "<=",
                "default_left": true,
                "left_child": {"leaf_value": 0.5},
                "right_child": {
                    "split_feature": 1, "threshold": 2.0, "decision_type": "<=",
                    "default_left": true,
                    "left_child": {"leaf_value": -0.25},
                    "right_child": {"leaf_value": 0.75}
                }
            }}]
        });
        let tmp = std::env::temp_dir().join("sift-gbdt-test.json");
        std::fs::write(&tmp, json.to_string()).unwrap();
        let m = GbdtModel::load(&tmp).unwrap();
        assert_eq!(m.score(&[0.0, 0.0]), 0.5);
        assert_eq!(m.score(&[2.0, 1.0]), -0.25);
        assert_eq!(m.score(&[2.0, 3.0]), 0.75);
    }

    #[test]
    fn feature_vector_keeps_blended_score_explicit() {
        let hit = FeatureHit {
            doc_idx: 0,
            bm25_combined: 10.0,
            bm25_exact: 4.0,
            bm25_semantic: 6.0,
            bm25_blended: 7.0,
            bigram_bonus: 1.0,
            qexp_score: 0.5,
            composition_similarity: 0.75,
            retrieval_score: 8.0,
            coverage: 0.5,
            doc_len: 20.0,
        };
        let stats = FeatureStats::from_hits(std::slice::from_ref(&hit));
        let features = feature_vector(&hit, 0, &stats);
        assert_eq!(features.len(), N_FEATURES);
        assert_eq!(features[0], 1.0);
        assert_eq!(features[4], 1.0);
        assert_eq!(features[5], 1.0);
        assert_eq!(features[6], 1.0);
        assert_eq!(features[7], 1.0);
        assert_eq!(features[8], 0.6);
        assert_eq!(features[9], 0.4);
        assert_eq!(features[13], 1.0);
    }

    #[test]
    fn sparse_prior_limits_tree_score_shift() {
        let scores = blend_with_sparse_prior(&[(1.0, 0.0, 1), (0.0, 1.0, 2)], 0.5);
        assert_eq!(scores[0].0, 0.5);
        assert_eq!(scores[1].0, 0.5);
    }
}
