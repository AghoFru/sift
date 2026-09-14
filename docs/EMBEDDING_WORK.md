# Embedded product work

This work succeeds when developers can embed local search and verify its quality with an
independent benchmark.

## Scope

1. Extract evaluation into the local `/Users/obsidian/code/ir-bench` repository. Verify the
   metrics, cache identity, and Sift and SQLite FTS5 adapters with real searches.
2. Keep product correctness tests. Remove obsolete duplication and update product documentation.
3. Make one library API own indexing, updates, deletes, and search. Route the CLI and server
   through that API without changing supported behavior.
4. Add a small C ABI. Verify an offline mobile integration and relevant native platform checks.

Keep both repositories local. Do not publish or create pull requests. Distributed infrastructure
and an SQLite extension are outside this work.

## Evidence required

- Correct metrics use all relevance judgments. Failed queries cannot improve the reported score.
- Cache entries change when the corpus, engine, model, or build settings change. Incomplete or
  damaged artifacts cannot count as successful cached builds.
- Independent engines use the same dataset and evaluation code.
- Existing API and artifact contracts remain covered by correctness tests.
- CLI and HTTP operations use the same embedded application operations.
- A native consumer exercises the C API, including invalid input and resource cleanup.
- A mobile target opens a local index and searches with no network dependency.
- Formatting, strict linting, relevant tests, and integration checks pass.

## Current findings

The original quality scripts compute ideal nDCG from retrieved documents. A query with two
relevant documents and one retrieved relevant document reports 1.0 instead of about 0.6131.
The old cache identifies builds by dataset name and joined build arguments. It omits corpus
content, engine identity, and model content. Historical quality reports require remeasurement.

## Verified progress

- Created `/Users/obsidian/code/ir-bench` as a local Git repository with no remote.
- Added one evaluator, content-verified artifact caching, and Sift and SQLite FTS5 adapters.
- All 13 benchmark tests pass against the final release binary and the local static model.
- The benchmark package builds as a wheel and source distribution. Its installed CLI works.
- Moved training tools and historical reports out of Sift. Removed duplicate evaluators.
- Retained the supported Python reference builder and optional product feature checks.
- Added `sift::Engine` for creation, JSONL builds, searches, writes, deletes, compaction, and reload.
- Routed CLI and HTTP search and writes through the shared implementation.
- Added optional CLI, server, and model-download features. The C build disables these features.
- Added an operating-system writer lock and protected server orphan recovery with it.
- Preserved original files until legacy migration validates and commits its manifest.
- Propagated filesystem synchronization failures and flushed metadata before segment publication.
- Kept source documents inside new segments so relocated indexes can compact.
- Replaced the process-wide build thread setting with a local thread pool.
- Added a C interface with explicit byte lengths, ownership, thread-local errors, and panic handling.
- All 49 workspace tests pass. All 32 library tests pass without default features.
- All 36 Sift tests pass with the edge feature set. The all-features workspace check passes.
- Rust formatting, strict Clippy, and the changed Python files' Ruff checks pass.
- The CLI and C release builds pass. All three Rust crates package successfully.
- The core package builds from its package contents. Sift and Sift FFI package without registry
  verification because publication is outside this work.
- Three native product contract tests pass with Python warnings treated as errors. They cover CLI
  and HTTP search, payloads, highlighting, phrases, filters, pagination, facets, validation,
  incremental writes, upserts, deletes, compaction, and reopen behavior.
- The Rust example passes creation, search, updates, deletes, relocation, repeated compaction,
  reopen, and snapshot reload checks without default features.
- The native C example passes creation, search, updates, deletes, compaction, reload, reopen,
  invalid JSON, invalid modes, null inputs, and result cleanup on macOS ARM64.
- The Android ARM64 instrumentation APK passes offline search, reopen, and error checks on API 36.
  Its manifest has no internet permission. The test checks that the permission is denied.
- Added the embedding guide, updated the product README and reference, and extended release checks.

The final Android rebuild and rerun passed. The test app was removed and the task-owned
emulator was stopped. Its writable storage and all APK signing stages were removed.
The scoped goal is complete. Both repositories remain local and no pull request was created.

## Evidence

Local logs are in `target/verification/` in Sift and `work/native-final.log` in IR Bench.
The local model is `/Users/obsidian/code/ir-bench/work/model`. No model files were added to Git.
See [EMBEDDING.md](EMBEDDING.md) for reproducible Rust, C, and Android commands.
The CLI and HTTP checks use:

```bash
SIFT_MODEL=/path/to/model python3 -W error tests/product_contract.py -v
```

From IR Bench, verify both adapters with:

```bash
SIFT_BIN=/path/to/sift/target/release/sift SIFT_MODEL=/path/to/model \
  python3 -W error -m unittest discover -s tests -v
```

## Limits

These checks establish local integration and correctness on small corpora. They do not establish
production retrieval quality, mobile performance, or support for every platform. Android coverage
is offline search in an application process. Mobile indexing and iOS remain unverified.

Sift still uses artifact directories and separate manifest and tombstone writes. It does not
provide database transactions. Full builds must not overwrite files used by open engines.
Compaction needs source documents and the build model. Migrated legacy segments without recorded
sources cannot compact. These storage restrictions are documented in the embedding guide.

The package command reported an existing yanked `der` version in the optional dependency lock.
The package checks passed. Dependency updates and registry publication were not performed.
