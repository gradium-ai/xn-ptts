# Phonon on Android

For supported Snapdragon NPUs, the [QNN AAR preview](../qnn/android/README.md) provides a
Kotlin API and compiled model support on Android 12+ ARM64. Build it from this repository;
the Maven package is not published yet. This guide covers the CPU library.

Phonon runs on Android on the CPU as a C library: `libptts_ffi.so` and its header,
[`ptts.h`](../ptts-ffi/include/ptts.h). Anything that can call C can use it: Kotlin and Java, C++
through the NDK, Flutter, Unity, .NET and others. The iOS package exports the same calls, so one
binding can serve both platforms.

For native Android apps there is a ready-made Kotlin wrapper, [`PhononTTS.kt`](PhononTTS.kt).
There is no Gradle package: you build the library and copy what you need into your app.

## 1. Build the library

You need Rust, [cargo-ndk](https://github.com/bbqsrc/cargo-ndk) and the Android NDK.

```sh
cargo install cargo-ndk
export ANDROID_NDK_HOME=/path/to/ndk
./android/build.sh
```

This writes `android/jniLibs/arm64-v8a/libptts_ffi.so` for phones and
`android/jniLibs/x86_64/libptts_ffi.so` for the emulator on an x86 machine. Copy the `jniLibs`
folder to your app's `src/main/`. Only these two ABIs are built: there is no 32-bit
(`armeabi-v7a`) library.

The arm64 build requires the dotprod and fp16 instructions (ARMv8.2), which nearly every phone
since 2019 has. The kernels are chosen at compile time, so the library crashes on a phone without
them. To support older phones, build with `ARM64_FEATURES= ./android/build.sh`. That build runs
everywhere but is slower.

## 2. Get a model

The library loads a checkpoint folder:

| File | What it is |
|---|---|
| `tokenizer.json` | The checkpoint's tokenizer. |
| `model.q8.gguf` or `model.safetensors` | The weights: q8_0 GGUF, or f32 safetensors. GGUF is used when both are there. |
| `config.json` | Required. The checkpoint's own model configuration; there is no fallback config. |
| `voices/<name>.safetensors` or `embeddings/<name>.safetensors` | The voices, each named after its file. |
| `default-voice.safetensors` | Optional. Listed as the voice `default`, and spoken when no voice is set. |

Download the folder into the app's files directory on first run, rather than shipping it in the
APK.

## 3. The interface

[`ptts.h`](../ptts-ffi/include/ptts.h) has eight calls:

| Call | What it does |
|---|---|
| `ptts_new(dir, PTTS_UNIT_CPU, lang)` | Load the model folder. This is the slow call. |
| `ptts_voices(h)` | The voice names, as one block of NUL-separated strings. |
| `ptts_set_voice(h, name)` | Speak in one of them from now on. |
| `ptts_languages(h)` | The languages the model was trained for, in the same format. |
| `ptts_set_lang(h, lang)` | Normalize text for another language from now on, without loading the model again. |
| `ptts_speak(h, text, callback, user, &result)` | Speak `text`, passing the audio to `callback` as it is made. |
| `ptts_last_error(h)` | Why the last call failed. Pass `NULL` after a failed `ptts_new`. |
| `ptts_free(h)` | Free the model. Nothing else does. |

Every binding follows the same rules:

- **Load once.** Keep one handle for the life of the app rather than one per sentence.
- **Calls block.** Make them off the main thread, and from one thread at a time.
- **The audio is 24 kHz mono `float` in [-1, 1].** The callback runs on the thread that called
  `ptts_speak`, with one or more frames at a time. Return `false` from it to stop early. It must not
  call back into the same handle. Generation runs on its own threads with bounded buffers. A callback that waits for
  playback pauses generation once those buffers fill.
- **`lang` is required.** `en`, `fr`, `de`, `es` and `pt` normalize numbers and symbols for that
  language, and `none` reads the text as written.
- **Set the thread count** with the `RAYON_NUM_THREADS` environment variable before the first
  `ptts_new`. The default is one thread per core, which is too many on phones that mix fast and
  slow cores. Measure on your target phones.

## 4. Call it

### Kotlin or Java

Copy [`PhononTTS.kt`](PhononTTS.kt) into your app and change its `package` line. Java code can use
it too, as long as the project builds Kotlin. It calls the library through
[JNA](https://github.com/java-native-access/jna):

```kotlin
dependencies {
    implementation("net.java.dev.jna:jna:5.17.0@aar")
}
```

If your release build uses R8 (`isMinifyEnabled = true`), add these lines to
`proguard-rules.pro`. Without them, the release build fails when it loads the library.

```
-keep class com.sun.jna.** { *; }
-keep class * implements com.sun.jna.** { *; }
-dontwarn java.awt.**
```

Then:

```kotlin
android.system.Os.setenv("RAYON_NUM_THREADS", "2", true)

thread {
    val track = AudioTrack.Builder()
        .setAudioFormat(
            AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(24000)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build(),
        )
        .setTransferMode(AudioTrack.MODE_STREAM)
        .build()
    track.play()
    PhononTTS(File(filesDir, "model").path, "en").use { tts ->
        tts.setVoice(tts.voices.first())
        tts.speak("Hello from Phonon.") { pcm ->
            track.write(pcm, 0, pcm.size, AudioTrack.WRITE_BLOCKING)
            true // false stops early
        }
    }
    track.release()
}
```

The wrapper turns failures into `IllegalStateException`, and rethrows from `speak` whatever the
callback throws. This example loads the model for one sentence only to stay short; an app should
keep its `PhononTTS` open.

### C or C++ with the NDK

Include `ptts.h` and link `libptts_ffi.so`, for example as an imported library in your
`CMakeLists.txt`:

```c
#include "ptts.h"

static bool on_audio(const float *pcm, size_t n, void *user) {
    // Queue `n` samples for playback, e.g. to an AAudio or Oboe stream.
    return true; // false stops early
}

setenv("RAYON_NUM_THREADS", "2", 1);
PttsHandle *h = ptts_new(model_dir, PTTS_UNIT_CPU, "en");
if (!h) {
    // ptts_last_error(NULL) says why.
}
PttsResult result;
if (!ptts_speak(h, "Hello from Phonon.", on_audio, NULL, &result)) {
    // ptts_last_error(h) says why.
}
ptts_free(h);
```

`ptts_speak`, `ptts_set_voice` and `ptts_set_lang` return `false` on failure. Strings from
`ptts_voices`, `ptts_languages` and `ptts_last_error` belong to the library: copy them if you keep
them.

### Flutter, Unity, .NET and other frameworks

Load `libptts_ffi.so` with the framework's usual way of calling C, and declare the calls from
`ptts.h`. In Flutter that is `dart:ffi`; in C#, `[DllImport("ptts_ffi")]`; in Unity, a native
plugin under `Plugins/Android/arm64-v8a`. Three details matter in any language:

- `ptts_speak`, `ptts_set_voice`, `ptts_set_lang` and the callback use a C `bool`, which is one byte.
- `size_t` is 64 bits on both ABIs the library is built for.
- The callback runs on the thread that called `ptts_speak`, so call it from a thread your runtime
  can run callbacks on.

These routes have not been tested here yet; the Kotlin wrapper and the C interface have.

### No native code

To run Phonon in a WebView instead, use the [`phonon-tts`](../ptts-wasm/js/README.md) npm package,
which runs the same model as WebAssembly.
