# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed

- Removed the legacy Python builder and its package. Rust is the only product implementation.
- Moved retrieval experiments to IR Bench and replaced Python test and packaging tools.
- Added the embedded `Engine` API and routed CLI and HTTP operations through shared code.
- Added a C interface and an Android offline search example.
- Moved reusable retrieval evaluation and training tools to the independent IR Bench repository.
- Removed historical nDCG claims that need measurement with the corrected evaluator.

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

- Reported filesystem synchronization failures instead of discarding them.
- Preserved original files until legacy index migration commits its manifest.
- Added an operating-system write lock and protected server orphan recovery with it.
- Kept new segment sources with their indexes so relocated indexes can compact.
- Used a local build thread pool instead of changing the host process pool.

- Prevented incompatible search response shapes from sharing cache entries.
- Applied phrase filtering before pagination and result counting.
- Rejected invalid numeric search parameters and rank/filter field names.
- Removed developer-machine paths from the regression and parity harnesses.
- Restored fresh Hugging Face model downloads by updating the Hub client.
- Preserved builder diagnostics when a regression artifact fails to build.

## 0.1.0 - Unreleased

Initial public release.
