#!/bin/bash
# Build libptts_ffi.so, the C interface in ptts-ffi (include/ptts.h), for Android apps, laid out
# as a jniLibs folder: arm64-v8a for phones, x86_64 for the emulator on an x86 machine.
# Needs cargo-ndk (`cargo install cargo-ndk`) and an NDK it can find (ANDROID_NDK_HOME).
#
# Output: android/jniLibs/<abi>/libptts_ffi.so, to copy to an app's src/main/jniLibs.
#
# FEATURES=litert adds the LiteRT engine, for bundles ptts-litert writes (README.md).
set -euo pipefail
cd "$(dirname "$0")/.."

# xn picks its kernels at compile time, so the CPU features are fixed here rather than detected
# on the phone. dotprod and fp16 (ARMv8.2) are on about every phone since 2019; set
# ARM64_FEATURES= to run on older ones, at the cost of xn's fast q8_0 kernels.
ARM64_FEATURES="${ARM64_FEATURES-+dotprod,+fp16}"
# These replace .cargo/config.toml's target-cpu=native, which is the build machine's CPU.
# Stripping keeps the exported ptts_* functions. Google Play requires 16 KB pages from Android 15;
# NDKs before r28 do not align to them by default.
COMMON="-C strip=symbols -C link-arg=-Wl,-z,max-page-size=16384"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS="$COMMON -C target-cpu=generic${ARM64_FEATURES:+ -C target-feature=$ARM64_FEATURES}"
export CARGO_TARGET_X86_64_LINUX_ANDROID_RUSTFLAGS="$COMMON -C target-cpu=x86-64"
# Either of these would take precedence over the flags above and drop them.
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS

rustup target add aarch64-linux-android x86_64-linux-android >/dev/null
# cargo-ndk skips the copy when the file already there is newer, so a cached build would leave a
# library of other flags in place.
rm -rf android/jniLibs
cargo ndk -t arm64-v8a -t x86_64 -o android/jniLibs build --release -p ptts-ffi ${FEATURES:+--features "$FEATURES"}
du -sh android/jniLibs/*/libptts_ffi.so
