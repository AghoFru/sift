#[cfg(feature = "cross-encoder")]
mod onnx;
#[cfg(not(feature = "cross-encoder"))]
pub(crate) use disabled::CrossEncoder;
#[cfg(feature = "cross-encoder")]
pub(crate) use onnx::CrossEncoder;

/// Stub so the rest of the server compiles identically without the
/// cross-encoder feature; loading reports the missing feature.
#[cfg(not(feature = "cross-encoder"))]
mod disabled {
    use anyhow::{anyhow, Result};
    use std::path::Path;

    pub(crate) struct CrossEncoder;

    impl CrossEncoder {
        pub(crate) fn load(_dir: &Path, _threads: usize, _max_len: usize) -> Result<Self> {
            Err(anyhow!(
                "this binary was built without the cross-encoder feature \
                 (rebuild with --features cross-encoder)"
            ))
        }

        pub(crate) fn score_pairs(&self, _q: &str, _docs: &[String]) -> Result<Vec<f32>> {
            Ok(Vec::new())
        }
    }
}
