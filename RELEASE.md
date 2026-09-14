# Release checklist

Run these checks from a clean checkout before tagging a release:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo test -p sift --no-default-features --lib --locked
cargo test -p sift --no-default-features --features edge --locked
cargo check --workspace --all-features --locked
cargo build --workspace --release --locked
cargo package -p sift-core --locked
cargo package -p sift --locked --no-verify
cargo package -p sift-ffi --locked --no-verify
bash -n examples/android/build.sh
```

Run the CLI and HTTP contract checks with a local static model:

```bash
SIFT_BIN="$PWD/target/release/sift" cargo run -p sift --example product_contract --locked -- \
  /path/to/model
```

Run the Rust, C, and Android checks in the [embedding guide](docs/EMBEDDING.md).
Build `sift-ffi` separately for the Android check to exclude server and download features.

For retrieval-quality releases, run the independent IR Bench repository against
the release binary and a local model. Supply a configuration file with the
binary path, model path, build arguments, and query parameters:

```bash
cd /path/to/ir-bench
ir-bench run --adapter sift \
  --config work/sift.json --dataset work/data/scifact --output work/scifact-result.json
```

Before publishing:

- Confirm `README.md` examples and defaults match `sift --help`.
- Review benchmark deltas against a baseline from the corrected evaluator.
- Add user-visible changes to `CHANGELOG.md`.
- Confirm the worktree is clean, create an annotated `vX.Y.Z` tag, and build
  release artifacts from that exact tag.
- If publishing to crates.io, publish `sift-core` first. After it is available
  in the registry, package and publish `sift` with normal verification enabled.
  Package and publish `sift-ffi` after `sift` is available.
