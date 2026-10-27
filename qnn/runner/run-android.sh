#!/bin/sh
# Push the runner, a Phonon bundle and the QNN libraries to an Android phone over adb, and speak.
#
#   QAIRT_ROOT=.../qairt/2.50.0.260828 ./run-android.sh BUNDLE_DIR "Hello world." LANG [htp|cpu|gpu] [SOC_MODEL] [hexagon-v79]
#
# htp runs the context binary on the NPU (Snapdragon only; the Hexagon version must
# match the SoC the binary was compiled for: v79 for 8 Elite, v75 for 8 Gen 3,
# v73 for 8 Gen 2, v69 for 8 Gen 1). cpu runs the fp32 DLCs on QNN's CPU backend.
# File names come from the bundle's metadata.json. The WAV comes back as ./out.wav.
set -eu

BUNDLE=${1:?bundle dir}
TEXT=${2:?text}
LANGUAGE=${3:?normalization language or none}
BACKEND=${4:-htp}
SOC_MODEL=${5:-}
HEXAGON=${6:-hexagon-v79}
case "$BACKEND" in
  htp) : "${SOC_MODEL:?compiled target SoC, e.g. SM8750}" ;;
  cpu|gpu) ;;
  *) echo "backend must be htp, cpu or gpu" >&2; exit 2 ;;
esac
: "${QAIRT_ROOT:?set QAIRT_ROOT to the QAIRT SDK root}"
HERE=$(cd "$(dirname "$0")" && pwd)
BIN="$HERE/build-android/phonon"
LIB="$QAIRT_ROOT/lib/aarch64-android"
DEV=/data/local/tmp/phonon

# meta KEY...: a value from metadata.json, following the keys (an object's first value for "*").
meta() {
  python3 -c '
import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    v = next(iter(v.values())) if k == "*" else v[k]
print(v)' "$BUNDLE/metadata.json" "$@"
}

push() {  # push FILE (a bundle-relative path) to the same place under $DEV/bundle
  # adb reads stdin; </dev/null keeps it from eating the voice list piped to the loop below.
  adb shell mkdir -p "$DEV/bundle/$(dirname "$1")" </dev/null
  adb push "$BUNDLE/$1" "$DEV/bundle/$1" </dev/null >/dev/null
}

adb shell mkdir -p $DEV/lib $DEV/dsp
adb push "$BIN" $DEV/ >/dev/null
push metadata.json
push "$(meta text tokenizer)"
push "$(meta text library android-arm64)"
push "$(meta generation bos file)"
python3 -c 'import json, sys; [print(v["file"]) for v in json.load(open(sys.argv[1]))["voices"]]' \
  "$BUNDLE/metadata.json" | while read -r f; do push "$f"; done

# The QNN libraries load each other by name at run time (QAIRT 2.50 forwards the
# libQnn* entry points to libQairt*), so both sets go along.
adb push "$LIB/libQnnSystem.so" "$LIB/libQairtSystem.so" $DEV/lib/ >/dev/null
if [ "$BACKEND" = htp ]; then
  V=$(echo "$HEXAGON" | sed 's/hexagon-v//')
  CONTEXT=$(python3 -c '
import json, sys
entries = json.load(open(sys.argv[1]))["runtime"]["context_binaries"].values()
files = [v["file"] for v in entries if sys.argv[2].upper() in v.get("soc_models", [])]
if len(files) != 1:
    raise SystemExit("bundle must have exactly one context binary for " + sys.argv[2])
print(files[0])' "$BUNDLE/metadata.json" "$SOC_MODEL")
  push "$CONTEXT"
  adb push "$LIB/libQnnHtp.so" "$LIB/libQairtHtp.so" "$LIB/libQnnHtpV${V}Stub.so" "$LIB/libQairtHtpV${V}Stub.so" \
    $DEV/lib/ >/dev/null
  adb push "$QAIRT_ROOT/lib/$HEXAGON/unsigned/." $DEV/dsp/ >/dev/null
else
  push "$(meta runtime dlcs prefill)"
  push "$(meta runtime dlcs step)"
  case "$BACKEND" in
    cpu) adb push "$LIB/libQnnCpu.so" "$LIB/libQairtCpu.so" $DEV/lib/ >/dev/null ;;
    gpu) adb push "$LIB/libQnnGpu.so" "$LIB/libQairtGpu.so" $DEV/lib/ >/dev/null ;;
    *) echo "backend must be htp, cpu or gpu" >&2; exit 2 ;;
  esac
fi

COMMAND=$(python3 -c '
import shlex, sys
print(shlex.join(["./phonon", "--bundle", "bundle", "--lang", sys.argv[1],
                  "--soc-model", sys.argv[2], "--backend", sys.argv[3], "--lib-dir", sys.argv[4],
                  "--text", sys.argv[5], "--out", "out.wav"]))' "$LANGUAGE" "$SOC_MODEL" "$BACKEND" "$DEV/lib" "$TEXT")
adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV/lib ADSP_LIBRARY_PATH='$DEV/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/dsp' $COMMAND"
adb pull $DEV/out.wav ./out.wav >/dev/null
echo "pulled out.wav"
