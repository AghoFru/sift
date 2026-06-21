# Release checklist

Run these checks from a clean checkout before tagging a release:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo test -p sift --no-default-features --features edge --locked
cargo check --workspace --all-features --locked
cargo build --workspace --release --locked
cargo package -p sift-core --locked
cargo package -p sift --locked --no-verify
python3 -m compileall -q sift_build reranker tests
bash -n tests/*.sh
```

For retrieval-quality releases, also run the pinned BEIR regression suite with
`SIFT_REGRESSION_DATA` pointing at the local dataset root:

```bash
SIFT_REGRESSION_DATA=/path/to/beir python3 tests/regression.py
```

Before publishing:

- Confirm `README.md` examples and defaults match `sift --help`.
- Review benchmark deltas; update the pinned baseline only deliberately.
- Add user-visible changes to `CHANGELOG.md`.
- Confirm the worktree is clean, create an annotated `vX.Y.Z` tag, and build
  release artifacts from that exact tag.
- If publishing to crates.io, publish `sift-core` first. After it is available
  in the registry, package and publish `sift` with normal verification enabled.
