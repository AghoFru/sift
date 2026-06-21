# Contributing

Bug reports and focused pull requests are welcome. For ranking changes, include
the retrieval-quality impact as well as ordinary correctness tests.

Before opening a pull request, run the commands in `RELEASE.md`. Keep generated
artifacts and model files out of Git. Changes to the artifact layout must either
remain backward-compatible with schema version 1 or introduce a new schema
version with an explicit loader error for older binaries.

By submitting a contribution, you agree that it is licensed under Apache-2.0,
the license covering this repository.
