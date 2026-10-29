# Phonon on Android with QNN

An Android library around the optimized QNN engine. It streams audio on Snapdragon's HTP,
with shared inference buffers and reused model state. Android 12 or newer, ARM64 only.
There is no automatic CPU fallback.

The package is not published yet. Its proposed Maven coordinates are `ai.gradium:ptts`.
Model weights are separate from the library and retain their own license.

## Build the package

Maintainers need JDK 17 or newer, Android SDK platform 36, NDK 29.0.14206865, CMake 3.31.6,
Rust, cargo-ndk and the QAIRT 2.50.0 SDK headers. Consumers will only need the Maven package.
The matching Qualcomm runtime comes from `com.qualcomm.qti:qnn-runtime:2.50.0` on Maven Central;
do not copy a different version into the app.
CMake downloads the tested nlohmann/json 3.11.3 release and verifies its SHA-256 when configuring
a new source build. The JSON code is compiled into the native library; AAR consumers need no separate
JSON dependency. Its MIT license is included in the AAR.

```sh
export ANDROID_HOME=/path/to/android-sdk
export QAIRT_ROOT=/path/to/qairt/2.50.0.260828
cargo install cargo-ndk --locked
cd qnn/android
./gradlew :ptts:assembleRelease
```

The output is `ptts/build/outputs/aar/ptts-release.aar`. Maven publishing also produces the
POM that carries the Qualcomm runtime dependency. A standalone AAR file does not carry that
transitive dependency, so use the local Maven repository when testing distribution:

```sh
./gradlew :ptts:publishReleasePublicationToVerificationRepository
./gradlew :smoke:assembleDebug :smoke:assembleRelease \
  -PpttsRepository="$PWD/ptts/build/repository" -PpttsVersion=<workspace-version>
```

This repository only publishes locally. Maven Central access and release automation are
separate release tasks. No private model files, QAIRT headers or vendor binaries are committed.

## Use it in an application

The intended dependency after publication is:

```kotlin
implementation("ai.gradium:ptts:<version>")
```

Enable native extraction in the app's Gradle file.
QNN's DSP loader needs actual files, including the matching Hexagon skel libraries:

```kotlin
android {
    packaging { jniLibs { useLegacyPackaging = true } }
}
```

The library manifest declares optional access to `libcdsprpc.so`. The loader prepends the app's
native library directory to `ADSP_LIBRARY_PATH`, preserving existing entries. Devices need working
QNN HTP support from their firmware. An unsupported device or model produces an error.

Load once on a worker thread, then reuse the model:

```kotlin
val tts = PhononTTS.load(context, File(modelDirectory), lang = "en")
tts.setVoice(tts.voices.first())
val sampleRate = tts.sampleRate
val result = tts.speak("Hello from Phonon.") { pcm ->
    // Deliver mono float PCM to AudioTrack at sampleRate.
    audioSink.write(pcm)
    true // Return false to stop.
}
// tts.stop() can also cancel from another thread.
// Wait for speak to finish before closing.
tts.close()
```

Supported normalization choices are `en`, `fr`, `de`, `es`, `pt`, `ar`, `hi`, `ko`, `pl`, `sv`,
`fi`, `no`, `it`, `nl`, `ja`, `zh`, `da`, `tr` and `none`. The language is
required. Callback exceptions propagate, and the model remains reusable after cancellation
or callback failure. Concurrent generation and closing during generation are rejected.

## Model bundles

Each [GitHub Release](https://github.com/gradium-ai/xn-ptts/releases) has a QNN bundle for
Phonon 31m, the default model, as `phonon-31m-en-qnn.zip`. It holds context binaries for the
SM8650, SM8750, SM8850 and QCS8275, each checked on that SoC, and the English voices of the
voice catalog. From 0.5.0 it also holds one for the SC8380XP (Snapdragon X Elite), for
`qnn/runner` embedded in a Windows app. The bundle has no Windows text library, so set
`Options.text_library` to a `ptts_text.dll` built from `qnn/text-ffi`. The context binaries
were built with QAIRT 2.50 and do not load on an older QNN runtime. For another model, such
as one of the gated Phonon 40m models, build a bundle with the export tools below. They
compile on Qualcomm AI Hub, so they need an AI Hub account.

A bundle holds `metadata.json`, a tokenizer, BOS data, voices and compiled context binaries.
These are exported model assets, not raw safetensors or GGUF checkpoints.
Model export and diagnostic tools remain on the [feat/phonon-qnn branch](https://github.com/gradium-ai/xn-ptts/tree/feat/phonon-qnn/qnn/export).
They are not needed to build or use the AAR with an existing bundle. Bundle metadata must
declare the exact tested SoC for each compiled context binary.

Every `runtime.context_binaries` entry needs an explicit `soc_models` list:

```json
{
  "Galaxy S25": {"file": "phonon.bin", "soc_models": ["SM8750"]}
}
```

The Android loader matches `Build.SOC_MODEL` exactly, after converting it to uppercase. Only
list targets whose correctness and performance have been verified. Existing test bundles need
this metadata added before using the Android library. The app loads its packaged Rust text
library, even when an older bundle includes a copy. Downloaded models cannot supply executable
code to the app.

### Use a bundle

Extract the complete folder and copy it into app storage before passing that directory to
`PhononTTS.load`. Keep `metadata.json`, its context binaries, tokenizer, BOS data and voices
together. A raw checkpoint with `config.json` and safetensors or GGUF weights must be exported
and compiled first; this API cannot load it directly.

Bundles are built for QAIRT 2.50.0. Adding another SoC to a bundle's metadata does not make a
compiled binary compatible with that device; it needs compiling and testing for that target.
The AAR supplies the Rust text library, so a bundle's copy is not used.

Bundles may live in an app's private storage or be copied from application assets. The API
accepts a local directory.

## Verify it

```sh
./gradlew :ptts:testDebugUnitTest :smoke:assembleDebug :smoke:assembleRelease
```

Model-free CI checks the Rust text frontend with its locked dependencies, model selection,
and the Kotlin lifecycle and callback tests without proprietary SDK headers:

```sh
./gradlew :ptts:testDebugUnitTest -PpttsUnitTestsOnly=true
cargo check --manifest-path ../text-ffi/Cargo.toml --locked
cmake -S ../runner/tests -B build/qnn-bundle-tests && cmake --build build/qnn-bundle-tests \
  && ctest --test-dir build/qnn-bundle-tests --output-on-failure
```

That flag refuses packaging tasks, so it cannot accidentally produce an AAR without native code.

The smoke app exercises normal Android application loading and streaming, including a minified
release build. Push a compatible bundle to its external files directory and launch the app:

```sh
adb install -r smoke/build/outputs/apk/debug/smoke-debug.apk
adb shell mkdir -p /sdcard/Android/data/ai.gradium.phonon.smoke/files/model
adb push /path/to/bundle/. /sdcard/Android/data/ai.gradium.phonon.smoke/files/model/
adb shell am start -n ai.gradium.phonon.smoke/.MainActivity
```

Tap Speak. Repeat using `smoke/build/outputs/apk/release/smoke-release.apk` to check R8 and native
loading. A model-free JNI probe can also be launched in either build:

```sh
adb shell am start -n ai.gradium.phonon.smoke/.MainActivity --ez probe true
adb logcat -d -s PhononSmoke:I
```

A successful probe prints `JNI_PROBE_OK`. It verifies native loading, not NPU execution.
Automated installed-app checks are available with:

```sh
./gradlew :smoke:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.modelDir=/sdcard/Android/data/ai.gradium.phonon.smoke/files/model \
  -Pandroid.testInstrumentationRunnerArguments.lang=en
```

Without `modelDir`, the real-model test is skipped and native packaging/JNI loading is still
checked. A real Snapdragon phone is required to verify NPU execution and performance. Desktop
unit tests and APK builds do not establish that the engine works inside an installed app.
