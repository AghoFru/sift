//! Native GBDT reranker: evaluates a LightGBM `dump_model()` JSON over the
//! per-hit feature vector from `score_with_features`. A few hundred shallow
//! trees over 8 features cost ~1µs per doc, so reranking the top-100 adds
//! ~100µs to a request: no model server, no ONNX runtime, one store.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::path::Path;

/// Feature order the model was trained with (reranker/train_lgbm.py).
pub(crate) const N_FEATURES: usize = 8;

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
}

impl GbdtModel {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let txt = std::fs::read_to_string(path)
            .with_context(|| format!("reading reranker model {}", path.display()))?;
        let v: Value = serde_json::from_str(&txt).context("parsing reranker JSON")?;
        let n_features = v
            .get("max_feature_idx")
            .and_then(|x| x.as_i64())
            .map(|x| (x + 1) as usize)
            .unwrap_or(N_FEATURES);
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
        Ok(GbdtModel { trees, n_features })
    }

    /// Sum of leaf values across trees (raw lambdarank score; only the
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

/// Build the 8-feature vector for one hit. Must match FEATURE_NAMES in
/// reranker/train_lgbm.py.
#[inline]
pub(crate) fn feature_vector(
    bm25_combined: f32,
    bm25_exact: f32,
    bm25_semantic: f32,
    coverage: f32,
    doc_len: f32,
    rank: usize,
) -> [f32; N_FEATURES] {
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
}
