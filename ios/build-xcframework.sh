#!/bin/bash
# Build PhononCore.xcframework, the compiled Rust core the ptts Swift package wraps:
# iPhone, the iOS simulator, Apple-silicon Macs and Mac Catalyst on Apple silicon, each a static
# library with the C header.
#
# Output: ios/PhononTTS/PhononCore.xcframework
set -euo pipefail
cd "$(dirname "$0")/.."

TARGETS=(aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin aarch64-apple-ios-macabi)
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
PROFILE=release

# Each target's default CPU, as in CI. .cargo/config.toml asks for target-cpu=native, the build
# machine's CPU, and xn picks its kernels at compile time, so a framework built on a recent Mac
# could use instructions older iPhones lack. An empty RUSTFLAGS takes precedence over that file
# (an empty CARGO_TARGET_<triple>_RUSTFLAGS does not); CARGO_ENCODED_RUSTFLAGS would win over both.
export RUSTFLAGS=""
unset CARGO_ENCODED_RUSTFLAGS

rustup target add "${TARGETS[@]}" >/dev/null
for t in "${TARGETS[@]}"; do
  echo "building $t"
  cargo build --profile "$PROFILE" -p ptts-ffi --target "$t"
done

HEADERS="$(mktemp -d)"
cp ptts-ffi/include/ptts.h "$HEADERS/"
cat > "$HEADERS/module.modulemap" <<'MAP'
module PhononCore {
    header "ptts.h"
    export *
}
MAP

OUT=ios/PhononTTS/PhononCore.xcframework
rm -rf "$OUT"
args=()
for t in "${TARGETS[@]}"; do
  args+=(-library "$TARGET_DIR/$t/$PROFILE/libptts_ffi.a" -headers "$HEADERS")
done
xcodebuild -create-xcframework "${args[@]}" -output "$OUT" >/dev/null
rm -rf "$HEADERS"
du -sh "$OUT"/*/libptts_ffi.a

# The same framework zipped, with the checksum a client's Package.swift needs to fetch it by
# URL instead of by path: `.binaryTarget(name: "PhononCore", url: ..., checksum: ...)`.
ZIP=ios/PhononTTS/PhononCore.xcframework.zip
rm -f "$ZIP"
(cd ios/PhononTTS && zip -qry PhononCore.xcframework.zip PhononCore.xcframework)
echo "zip: $ZIP ($(du -h "$ZIP" | cut -f1)), checksum $(swift package compute-checksum "$ZIP")"
