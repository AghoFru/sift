# Embed Sift

Search needs only an index directory. Creation, document writes, and compaction
also need the build model (`tokenizer.json` and `model.safetensors`).

## Rust

```toml
[dependencies]
sift = { path = "/path/to/sift/crates/sift", default-features = false }
serde_json = "1"
```

```rust
use serde_json::json;
use sift::{Engine, SearchOptions, WriteMode};

let mut index = Engine::create("notes.sift", "/path/to/model", &[
    json!({"id": "1", "text": "A cat sleeps."}),
    json!({"id": "2", "text": "A dog plays."}),
    json!({"id": "3", "text": "A bird flies."}),
])?;
let results = index.search(SearchOptions::new("cat"))?;
index.write_documents(&[
    json!({"id": "1", "text": "A dog plays."}),
], WriteMode::Upsert)?;
index.delete(&["1".to_owned()])?;
```

Use `Engine::open(path)` to reopen an index. For JSONL input, use
`Engine::build(BuildOptions::new(input, output))`.
Run the [complete Rust example](../crates/sift/examples/embedded.rs):

```sh
cargo run -p sift --example embedded --no-default-features --locked -- \
  /path/to/model target/embedded-check
```

Use a new output directory for each run.

## C

```sh
cargo build -p sift-ffi --release --locked
```

Include [sift.h](../crates/sift-ffi/include/sift.h) and link `libsift_ffi.a` or
the shared library. Search uses the [HTTP JSON format](REFERENCE.md#http-search).
[tests/c_api.c](../tests/c_api.c) is a complete example. On macOS:

```sh
cc -std=c11 -Wall -Wextra -Werror -I crates/sift-ffi/include tests/c_api.c \
  -L target/release -lsift_ffi -Wl,-rpath,"$PWD/target/release" -o target/c-api-check
target/c-api-check /path/to/model target/c-api-index
```

- Keep input memory valid and unchanged throughout each call.
- Free each returned result once with `sift_string_free`.
- Close each handle once with `sift_close`, after its active calls finish.
- Read `sift_last_error` before another fallible call on the same thread. Do not free this borrowed string.
- After `SIFT_PANIC`, close and reopen the handle.

Calls on one handle serialize.

## Android

The [JNI example](../examples/android/OfflineCheck.java) searches a packaged
index without internet permission. It requires JDK 17, SDK platform 36,
build tools 36.0.0, NDK r28, and the Rust `aarch64-linux-android` target.

Use the index produced by the C example. Build on macOS or Linux:

```sh
bash examples/android/build.sh target/c-api-index \
  /path/to/android-sdk /path/to/android-sdk/ndk/28.2.13676358
adb -s DEVICE_SERIAL install target/android/sift-offline.apk
adb -s DEVICE_SERIAL shell am instrument -w org.sift.example/.OfflineCheck
adb -s DEVICE_SERIAL uninstall org.sift.example
```

Success prints `SIFT_OFFLINE_OK`. Remove an old installation before installing
a new build, because signing keys are disposable.

## Storage rules

- An index is a directory. Do not modify or rebuild files while an engine maps them.
- Stop writes before copying an index. Keep the authoritative documents available.
- Open handles retain snapshots. Call `reload()` to see another handle's writes.
- Competing writers receive a lock error. Writes through an engine reload that engine.
- Writes are not transactions across files or with your database. After failure, inspect and reload before retrying.
- Compaction requires stored source documents.

See [query restrictions](REFERENCE.md#http-search) before using optional features.
