#!/usr/bin/env bash
# Check the libraries build.sh produced against ptts itself.
#
# 1. cargo test: the C interface, ptts::plan::chunks and the steps spelled out agree on every
#    line of tests/inputs.txt, and the ptts reference is written to target/expected.txt.
# 2. examples/dump.c, linked against each library, must print exactly that reference: on macOS
#    natively, and on Linux arm64 inside the ubuntu:20.04 runner image.
# 3. The Android library must need only system libraries.
#
# PTTS_TOKENIZER, required, is a checkpoint's tokenizer.json. RUNNER_IMAGE is the Linux image
# (default: phonon-runner, built with
# `docker build -f ../runner/Dockerfile.linux -t phonon-runner ../runner`).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TOK="${PTTS_TOKENIZER:?set PTTS_TOKENIZER to the tokenizer.json of a checkpoint}"
RUNNER_IMAGE="${RUNNER_IMAGE:-phonon-runner}"
NDK="${ANDROID_NDK_HOME:-/opt/homebrew/share/android-ndk}"
export PTTS_TOKENIZER="$TOK"
cd "$HERE"
mkdir -p target/c

cargo test --release --locked
echo "== ptts reference: $(grep -c '^#' target/expected.txt) inputs"

cc -O2 -Wall -Wextra -Werror -Idist/include examples/dump.c -Ldist/macos-arm64 -lptts_text \
    -Wl,-rpath,"$HERE/dist/macos-arm64" -o target/c/dump-macos
target/c/dump-macos "$TOK" en tests/inputs.txt > target/c/macos.txt
diff target/expected.txt target/c/macos.txt
echo "== macOS: dump.c output equals ptts"

docker run --rm --platform linux/arm64 \
    -v "$HERE:/work" -v "$TOK:/model/tokenizer.json:ro" -w /work "$RUNNER_IMAGE" sh -c '
        set -e
        gcc -O2 -Wall -Wextra -Werror -Idist/include examples/dump.c -Ldist/linux-arm64 \
            -lptts_text -Wl,-rpath,/work/dist/linux-arm64 -o /tmp/dump
        ldd dist/linux-arm64/libptts_text.so
        /tmp/dump /model/tokenizer.json en tests/inputs.txt > target/c/linux.txt'
diff target/expected.txt target/c/linux.txt
echo "== Linux arm64 (ubuntu:20.04): dump.c output equals ptts"

READELF="$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-readelf | head -1)"
needed="$("$READELF" -d dist/android-arm64-v8a/libptts_text.so | grep NEEDED | sed 's/.*\[\(.*\)\]/\1/' | sort | tr '\n' ' ')"
echo "== Android NEEDED: $needed"
for lib in $needed; do
    case "$lib" in
        libc.so|libm.so|libdl.so|liblog.so) ;;
        *) echo "unexpected Android dependency: $lib" >&2; exit 1 ;;
    esac
done
echo "== all checks passed"
