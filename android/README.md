# Phonon on Android

For supported Snapdragon NPUs, the [QNN AAR](../qnn/android/README.md) provides a
Kotlin API and compiled model support on Android 12+ ARM64. Build it from this repository;
the Maven package is not published yet. This guide covers the CPU library.

Phonon runs on Android on the CPU as a C library: `libptts_ffi.so` and its header,
[`ptts.h`](../ptts-ffi/include/ptts.h). Anything that can call C can use it: Kotlin and Java, C++
through the NDK, Flutter, Unity, .NET and others. The iOS package exports the same calls, so one
binding can serve both platforms. The [C guide](../ptts-ffi/README.md) describes the interface;
this guide covers what is particular to Android.

For native Android apps there is a ready-made Kotlin wrapper, [`PhononTTS.kt`](PhononTTS.kt).
There is no Gradle package: you build the library and copy what you need into your app.

To use Phonon as the phone's system voice, for TalkBack and any app that speaks through Android's
`TextToSpeech`, build the engine app in [`tts-engine/`](tts-engine/README.md) instead.

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
(`armeabi-v7a`) library. JNA ships more ABIs, so limit your app to these two with
`ndk { abiFilters += listOf("arm64-v8a", "x86_64") }` in `defaultConfig`; otherwise the app also
installs on 32-bit phones, where Phonon fails to load.

The arm64 build requires the dotprod and fp16 instructions (ARMv8.2), which nearly every phone
since 2019 has. The kernels are chosen at compile time, so the library crashes on a phone without
them. To support older phones, build with `ARM64_FEATURES= ./android/build.sh`. That build runs
everywhere but is slower.

## 2. Get a model

The Phonon models are on Hugging Face, [listed in the README](../README.md#model-setup). The
library loads a checkpoint folder: the [C guide](../ptts-ffi/README.md#2-get-a-model) lists its
files and how to download them.

Download the folder into the app's files directory on first run, rather than shipping it in the
APK. A gated model needs a Hugging Face token to download, so host its files yourself, under the
model's license.

## 3. The interface

[`ptts.h`](../ptts-ffi/include/ptts.h) has eight calls. The
[C guide](../ptts-ffi/README.md#3-the-interface) lists them, with the rules every binding follows.
On Android, also set the thread count with the `RAYON_NUM_THREADS` environment variable before the
first `ptts_new`. The default is one thread per core, which is too many on phones that mix fast
and slow cores. Measure on your target phones.

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
`CMakeLists.txt`. The [C guide](../ptts-ffi/README.md#4-call-it) shows the calls. Queue the audio
the callback receives to an AAudio or Oboe stream, and call `setenv("RAYON_NUM_THREADS", "2", 1)`
before `ptts_new`.

### Flutter, Unity, .NET and other frameworks

Load `libptts_ffi.so` with the framework's usual way of calling C, and declare the calls from
`ptts.h`. The [C guide](../ptts-ffi/README.md#other-languages) has the details that matter in any
language. In Unity, the library goes in a native plugin under `Plugins/Android/arm64-v8a`.

These routes have not been tested here yet; the Kotlin wrapper and the C interface have.

### No native code

To run Phonon in a WebView instead, use the [`phonon-tts`](../ptts-wasm/js/README.md) npm package,
which runs the same model as WebAssembly.
