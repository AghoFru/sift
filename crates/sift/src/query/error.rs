/// Search failures are independent of the transport used by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchErrorKind {
    InvalidRequest,
    Conflict,
    Unavailable,
    Internal,
}

#[derive(Debug)]
pub struct SearchError {
    pub kind: SearchErrorKind,
    pub message: String,
}

impl From<(SearchErrorKind, String)> for SearchError {
    fn from((kind, message): (SearchErrorKind, String)) -> Self {
        Self { kind, message }
    }
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for SearchError {}

/// Timing metadata is separate from the serialized search result.
pub struct SearchTiming {
    pub score_us: u64,
    pub total_us: u64,
    pub cache_hit: bool,
}
