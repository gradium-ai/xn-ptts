#!/usr/bin/env bash
# Build libptts_text for macOS (host), Android arm64 (API 31) and Linux arm64 (glibc 2.31).
#
#   ./build.sh [host|android|linux]...   (default: all three)
#
# Outputs land in dist/<target>/, next to a copy of include/ptts_text.h in dist/include/.
# Needs: rustup with aarch64-linux-android, cargo-ndk, the NDK (ANDROID_NDK_HOME, defaults to
# Homebrew's), and Docker for Linux, which builds inside ubuntu:20.04 arm64 so the library
# only asks for the glibc that image has.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# ptts is a path dependency in this repository; mount it at the same path in Docker.
PTTS_ROOT="$(cd "$HERE/../.." && pwd)"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-/opt/homebrew/share/android-ndk}"
ANDROID_API="${ANDROID_API:-31}"
# Never inherit host CPU flags into the cross builds.
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS

cd "$HERE"
mkdir -p dist/include
cp include/ptts_text.h dist/include/

build_host() {
    cargo build --release --locked
    mkdir -p dist/macos-arm64
    cp target/release/libptts_text.dylib dist/macos-arm64/
    install_name_tool -id @rpath/libptts_text.dylib dist/macos-arm64/libptts_text.dylib
    echo "host: dist/macos-arm64/libptts_text.dylib"
}

build_android() {
    rustup target add aarch64-linux-android >/dev/null
    cargo ndk -t arm64-v8a -P "$ANDROID_API" build --release --locked
    mkdir -p dist/android-arm64-v8a
    cp target/aarch64-linux-android/release/libptts_text.so dist/android-arm64-v8a/
    echo "android: dist/android-arm64-v8a/libptts_text.so"
}

build_linux() {
    docker build --platform linux/arm64 -t ptts-text-build:20.04 -f docker/Dockerfile.linux-arm64 docker
    docker volume create ptts-text-cargo-registry >/dev/null
    docker run --rm --platform linux/arm64 \
        -v "$HERE:$HERE" -v "$PTTS_ROOT:$PTTS_ROOT:ro" \
        -v ptts-text-cargo-registry:/opt/cargo/registry \
        -e CARGO_TARGET_DIR="$HERE/target/docker" -w "$HERE" \
        ptts-text-build:20.04 \
        cargo build --release --locked --target aarch64-unknown-linux-gnu
    mkdir -p dist/linux-arm64
    cp target/docker/aarch64-unknown-linux-gnu/release/libptts_text.so dist/linux-arm64/
    echo "linux: dist/linux-arm64/libptts_text.so"
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(host android linux)
for t in "${targets[@]}"; do
    case "$t" in
        host) build_host ;;
        android) build_android ;;
        linux) build_linux ;;
        *) echo "unknown target $t (host, android, linux)" >&2; exit 2 ;;
    esac
done
