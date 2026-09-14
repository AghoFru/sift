#!/usr/bin/env bash
# Build the offline Android check with installed SDK and NDK tools.
set -euo pipefail
if [ "$#" -ne 3 ]; then
    echo 'Usage: build.sh INDEX ANDROID_SDK ANDROID_NDK' >&2
    exit 2
fi
index=$(cd "$1" && pwd)
sdk=$(cd "$2" && pwd)
ndk=$(cd "$3" && pwd)
root=$(cd "$(dirname "$0")/../.." && pwd)
source_dir="$root/examples/android"
case "$(uname -s)" in
    Darwin) host=darwin-x86_64 ;;
    Linux) host=linux-x86_64 ;;
    *) echo 'Use a macOS or Linux build host.' >&2; exit 2 ;;
esac
compiler="$ndk/toolchains/llvm/prebuilt/$host/bin/aarch64-linux-android26-clang"
build_tools="$sdk/build-tools/36.0.0"
android_jar="$sdk/platforms/android-36/android.jar"
cd "$root"
env CC_aarch64_linux_android="$compiler" \
    AR_aarch64_linux_android="$ndk/toolchains/llvm/prebuilt/$host/bin/llvm-ar" \
    CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$compiler" \
    cargo build -p sift-ffi --target aarch64-linux-android --release --locked
mkdir -p "$root/target/android"
stage=$(mktemp -d "$root/target/android/package-XXXXXX")
trap 'rm -rf "$stage"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$stage/assets" "$stage/classes" "$stage/lib/arm64-v8a"
cp -R "$index" "$stage/assets/index"
javac --release 8 -Xlint:all -Werror -cp "$android_jar" \
    -d "$stage/classes" "$source_dir/OfflineCheck.java"
"$build_tools/d8" --min-api 26 --lib "$android_jar" --output "$stage" \
    "$stage/classes/org/sift/example/OfflineCheck.class"
cp target/aarch64-linux-android/release/libsift_ffi.so "$stage/lib/arm64-v8a/"
"$compiler" -std=c11 -Wall -Wextra -Werror -shared -fPIC \
    -I crates/sift-ffi/include "$source_dir/sift_example.c" \
    -L "$stage/lib/arm64-v8a" -lsift_ffi -o "$stage/lib/arm64-v8a/libsift_example.so"
"$build_tools/aapt2" link -I "$android_jar" --manifest "$source_dir/AndroidManifest.xml" \
    -A "$stage/assets" -o "$stage/unsigned.apk"
(cd "$stage" && zip -q unsigned.apk classes.dex lib/arm64-v8a/*.so)
"$build_tools/zipalign" -f 4 "$stage/unsigned.apk" "$stage/aligned.apk"
# The signing key is disposable and stays in the staging directory.
keytool -genkeypair -keystore "$stage/check.p12" -storepass android -keypass android \
    -alias check -dname 'CN=Sift local check' -keyalg RSA -validity 1 -noprompt
"$build_tools/apksigner" sign --ks "$stage/check.p12" --ks-key-alias check \
    --ks-pass pass:android --out "$root/target/android/sift-offline.apk" "$stage/aligned.apk"
"$build_tools/apksigner" verify "$root/target/android/sift-offline.apk"
echo "$root/target/android/sift-offline.apk"
