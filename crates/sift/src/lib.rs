//! Embedded sparse search, indexing, and updates.
//!
//! Disable default features to use local models without the CLI, server, or model downloads.

pub mod build;
mod ce;
mod clean;
mod engine;
mod filters;
#[cfg(feature = "cli")]
pub mod index_cmd;
pub mod query;
mod query_negation;
mod rerank;
#[cfg(feature = "cli")]
pub mod search;
mod spell;
mod write;
pub use build::BuildArgs as BuildOptions;
pub use write::{WriteMode, WriteOutcome};

pub use engine::Engine;
pub use filters::FilterClause;
pub use query::{SearchError, SearchErrorKind, SearchHit, SearchOptions, SearchResponse};

#[cfg(feature = "cli")]
pub mod replicate;
#[cfg(feature = "server")]
pub mod serve;
