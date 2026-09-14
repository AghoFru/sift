# Embed Sift

Use `sift::Engine` to build, open, search, and change a local index. The CLI and
HTTP server use the same search and write implementation. Search needs the index
directory and its tokenizer. It does not need the build model or a server.

## Rust

For a local checkout, add these dependencies to your application:

```toml
[dependencies]
sift = { path = "/path/to/sift/crates/sift", default-features = false }
serde_json = "1"
```

Disable default features to exclude the CLI, HTTP server, and model downloads.
Supply a local model directory with `tokenizer.json` and `model.safetensors` for
index creation and document writes.

```rust
use sift::{Engine, SearchOptions, WriteMode};
use serde_json::json;

let mut index = Engine::create("artifacts/notes.sift", "/path/to/model", &[
    json!({"id": "1", "text": "A cat sleeps on the windowsill."}),
])?;
let results = index.search(SearchOptions::new("cat"))?;
index.write_documents(&[
    json!({"id": "1", "text": "A horse rests near a stable."}),
], WriteMode::Upsert)?;
index.delete(&["1".to_owned()])?;
```

Use `Engine::open(path)` for an existing index. Use `BuildOptions::new(input,
output)` and `Engine::build(options)` for JSONL input and build settings.
`Engine::append_to` accepts JSONL files for incremental writes. Use `Upsert`
when an identifier can already exist. `Insert` preserves the CLI append behavior.

The executable [Rust example](../crates/sift/examples/embedded.rs) checks creation,
search, upserts, deletes, compaction, relocation, and reopen behavior:

```bash
cargo run -p sift --example embedded --no-default-features --locked -- \
  /path/to/model target/embedded-check
```

Use a new output directory for each run.

## C and native bindings

Build the C library separately to keep server and download features disabled:

```bash
cargo build -p sift-ffi --release --locked
```

Include [sift.h](../crates/sift-ffi/include/sift.h). Link `libsift_ffi.a` or the
platform shared library. The interface provides opaque handles and UTF-8 inputs
with explicit byte lengths.

```c
Sift *index = NULL;
char *result = NULL;
if (sift_open(sift_text("artifacts/notes.sift"), &index) != SIFT_OK) {
    fprintf(stderr, "%s\n", sift_last_error());
    return 1;
}
int status = sift_search(index, sift_text("{\"q\":\"cat\",\"k\":10}"), &result);
if (status == SIFT_OK) puts(result);
else fprintf(stderr, "%s\n", sift_last_error());
sift_string_free(result);
sift_close(index);
```

Search requests and responses use the HTTP JSON schema in the
[reference guide](REFERENCE.md). `sift_create` and `sift_write` accept document
arrays. `sift_delete` accepts an array of string identifiers.

- Keep input memory valid and unchanged for the complete call.
- Free each successful search result once with `sift_string_free`.
- Close each handle once with `sift_close`.
- Do not close a handle while another call uses it.
- Read `sift_last_error` before the next fallible call on the same thread.
- Do not free the borrowed error string.

Calls on one C handle serialize. Separate handles can search independent
snapshots. Error status is `SIFT_ERROR`. A caught Rust panic returns `SIFT_PANIC`.
After a panic, close and reopen the handle. Invalid pointers remain a caller error.
Inputs accept at most 4096 path bytes or 64 MiB of JSON. Document writes accept
from 1 through 100000 documents per call.

The native [C check](../tests/c_api.c) exercises writes, searches, and errors.
For macOS, run these commands with a new output directory:

```bash
cc -std=c11 -Wall -Wextra -Werror -I crates/sift-ffi/include tests/c_api.c \
  -L target/release -lsift_ffi -Wl,-rpath,"$PWD/target/release" -o target/c-api-check
target/c-api-check /path/to/model target/c-api-index
```

## Android offline check

The [Android example](../examples/android/OfflineCheck.java) calls the C interface
through a small JNI function. It copies a packaged index into application storage
and searches for `sparrow`. It checks that the result is document `bird`.
The manifest has no internet permission. The test checks that permission is denied.
It also checks repeated reopen and native error propagation.

Use the index from the C check above. Install the Rust `aarch64-linux-android`
target, JDK 17, Android SDK platform 36, build tools 36.0.0, and NDK r28.
The build script supports macOS and Linux hosts. It uses the installed SDK tools
directly and removes its disposable signing key and staging directory.

```bash
python3 examples/android/build.py target/c-api-index \
  --sdk /path/to/android-sdk --ndk /path/to/android-sdk/ndk/28.2.13676358
adb -s DEVICE_SERIAL install target/android/sift-offline.apk
adb -s DEVICE_SERIAL shell am instrument -w org.sift.example/.OfflineCheck
adb -s DEVICE_SERIAL uninstall org.sift.example
```

The expected result contains `SIFT_OFFLINE_OK`. Uninstall a previous copy before
installing a newly built APK because each build uses a new signing key.
The test covers Android ARM64 offline search. It does not measure device performance
or test mobile indexing. The cross-build follows the
[Android NDK compiler configuration](https://developer.android.com/ndk/guides/other_build_systems).

## Storage and compatibility

An index is a directory of immutable segments. It is not a single database file.
Open engines retain their snapshot until `reload` succeeds. Writes through an
engine reload that engine. Other open engines must reload explicitly.
Writers take a nonblocking operating-system file lock. A competing writer gets an error.

Use a new directory for a full build. Do not rebuild files that an open engine
has memory-mapped. Copy indexes while writes are stopped. New segments retain
their source documents locally for compaction. Existing legacy segments without
recorded sources cannot compact. Compaction and document writes also need the
recorded build model to remain available.

Manifest and tombstone writes are separate. Sift does not provide transactions
across these files or with an application's database. A failed write can require
a reload and inspection before retry. Keep the authoritative documents available.
Artifact files must remain unchanged while mapped.

Some query features require one segment. The existing HTTP validation applies
to embedded searches too. See the reference guide for feature restrictions.
This integration has native checks on macOS ARM64 and an Android ARM64 emulator.
It does not establish Windows, iOS, or distributed operation support.
