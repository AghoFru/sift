"""Build the offline Android check with installed SDK and NDK tools."""

import argparse
import os
import platform
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
TARGET = ROOT / "target" / "android"


def run(*command, env=None):
    subprocess.run(
        [str(part) for part in command], cwd=ROOT, env=env, check=True, timeout=600
    )


def build(index, sdk, ndk):
    host = "darwin-x86_64" if platform.system() == "Darwin" else "linux-x86_64"
    compiler_bin = ndk / "toolchains" / "llvm" / "prebuilt" / host / "bin"
    compiler = compiler_bin / "aarch64-linux-android26-clang"
    build_tools = sdk / "build-tools" / "36.0.0"
    android_jar = sdk / "platforms" / "android-36" / "android.jar"
    env = dict(os.environ)
    env.update(
        CC_aarch64_linux_android=str(compiler),
        AR_aarch64_linux_android=str(compiler_bin / "llvm-ar"),
        CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=str(compiler),
    )
    run(
        "cargo",
        "build",
        "-p",
        "sift-ffi",
        "--target",
        "aarch64-linux-android",
        "--release",
        "--locked",
        env=env,
    )
    TARGET.mkdir(parents=True, exist_ok=True)
    # This directory owns staging and the disposable APK signing key.
    with tempfile.TemporaryDirectory(prefix="package-", dir=TARGET) as scratch:
        stage = Path(scratch)
        assets = stage / "assets"
        shutil.copytree(
            index, assets / "index", ignore=shutil.ignore_patterns(".sift-*")
        )
        classes = stage / "classes"
        classes.mkdir()
        run(
            "javac",
            "--release",
            "8",
            "-Xlint:all",
            "-Werror",
            "-cp",
            android_jar,
            "-d",
            classes,
            SOURCE / "OfflineCheck.java",
        )
        run(
            build_tools / "d8",
            "--min-api",
            "26",
            "--lib",
            android_jar,
            "--output",
            stage,
            *sorted(classes.rglob("*.class")),
        )
        native = stage / "lib" / "arm64-v8a"
        native.mkdir(parents=True)
        library = (
            ROOT / "target" / "aarch64-linux-android" / "release" / "libsift_ffi.so"
        )
        shutil.copy2(library, native)
        run(
            compiler,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "-I",
            ROOT / "crates" / "sift-ffi" / "include",
            SOURCE / "sift_example.c",
            "-L",
            native,
            "-lsift_ffi",
            "-o",
            native / "libsift_example.so",
        )
        unsigned = stage / "unsigned.apk"
        run(
            build_tools / "aapt2",
            "link",
            "-I",
            android_jar,
            "--manifest",
            SOURCE / "AndroidManifest.xml",
            "-A",
            assets,
            "-o",
            unsigned,
        )
        with zipfile.ZipFile(unsigned, "a", compression=zipfile.ZIP_DEFLATED) as apk:
            apk.write(stage / "classes.dex", "classes.dex")
            for library in sorted(native.glob("*.so")):
                apk.write(library, library.relative_to(stage).as_posix())
        aligned = stage / "aligned.apk"
        run(build_tools / "zipalign", "-f", "4", unsigned, aligned)
        key = stage / "check.p12"
        run(
            "keytool",
            "-genkeypair",
            "-keystore",
            key,
            "-storepass",
            "android",
            "-keypass",
            "android",
            "-alias",
            "check",
            "-dname",
            "CN=Sift local check",
            "-keyalg",
            "RSA",
            "-validity",
            "1",
            "-noprompt",
        )
        output = TARGET / "sift-offline.apk"
        run(
            build_tools / "apksigner",
            "sign",
            "--ks",
            key,
            "--ks-key-alias",
            "check",
            "--ks-pass",
            "pass:android",
            "--out",
            output,
            aligned,
        )
        run(build_tools / "apksigner", "verify", output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "index", type=Path, help="Index with a bird document that matches sparrow."
    )
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--ndk", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.index.resolve(), arguments.sdk.resolve(), arguments.ndk.resolve())
