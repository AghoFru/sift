# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed

- Licensed the project under Apache-2.0.
- Made the Rust builder the documented source of truth for production defaults.
- Clarified that an index is one logical store packaged as an artifact
  directory, rather than one physical file.
- Added `--semantic-weight` to one-shot CLI search so exact BM25 and blended
  semantic retrieval can be compared from the same artifact.
- Added a vocabulary-mismatch example and same-machine BM25 comparison.
- Split artifact loading, scoring strategies, spelling, build support, request
  validation, and auxiliary handlers into focused source files.
- Added a reproducible same-machine BM25-versus-sift benchmark harness.

### Fixed

- Prevented incompatible search response shapes from sharing cache entries.
- Applied phrase filtering before pagination and result counting.
- Rejected invalid numeric search parameters and rank/filter field names.
- Removed developer-machine paths from the regression and parity harnesses.
- Restored fresh Hugging Face model downloads by updating the Hub client.
- Preserved builder diagnostics when a regression artifact fails to build.

## 0.1.0 - Unreleased

Initial public release.
