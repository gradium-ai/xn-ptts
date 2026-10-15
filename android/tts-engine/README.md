# Phonon as an Android text-to-speech engine

This app makes Phonon a system text-to-speech engine. Once it is installed, it shows up as a
preferred engine in the system's text-to-speech settings (under Accessibility, or System >
Languages, depending on the phone). With a model imported, every app that speaks through Android's
`TextToSpeech` can use it: TalkBack, e-book readers, navigation and so on.

It runs on the CPU through the same `libptts_ffi.so` and [`PhononTTS.kt`](../PhononTTS.kt) that
[the Android guide](../README.md) describes. The build compiles `PhononTTS.kt` from there, so
there is one copy of it.

## Build

You need what `android/build.sh` needs (Rust, cargo-ndk, the NDK), plus JDK 17 or newer and the
Android SDK with platform 36.

```sh
export ANDROID_NDK_HOME=/path/to/ndk
./android/build.sh                 # the native library, into android/jniLibs
cd android/tts-engine
export ANDROID_HOME=/path/to/sdk
./gradlew assembleDebug            # build/outputs/apk/debug/phonon-tts-engine-debug.apk
./gradlew assembleRelease          # minified, signed with the debug key; use your own to publish
adb install -r build/outputs/apk/debug/phonon-tts-engine-debug.apk
```

The APK carries `arm64-v8a` for phones and `x86_64` for emulators on x86 machines.

## Install a model

There is no model in the APK. Open the Phonon TTS app, tap "Import a checkpoint folder", and pick
a folder holding the files [the Android guide lists](../README.md#2-get-a-model): `config.json`,
`tokenizer.json`, `model.q8.gguf` (or `model.safetensors`), and optionally `voices/` and
`default-voice.safetensors`. The app copies them into its own storage and loads the model. Android
does not let an app pick the top of `Download` itself, so put the files in a folder inside it.

Then choose Phonon TTS as the preferred engine in the system settings (the app has a button for
it), or from a computer:

```sh
adb shell settings put secure tts_default_synth ai.gradium.phonon.tts
```

`Model.install` is the one way a checkpoint gets in. A download from a model hub would be a second
caller of it, filling the same folder.

## How it answers Android

- **Languages.** Android asks in ISO 639 codes. `eng`, `fra`, `deu`, `spa` and `por` map to the
  runtime's `en`, `fr`, `de`, `es` and `pt`, and each request's text is normalized for its own
  language (`PhononTTS.setLanguage`, which is `ptts_set_lang`). A checkpoint whose config lists
  `languages` offers only those; one that lists none offers all five. Countries are accepted but
  make no difference. Without a model, every language reports missing data.
- **Voices.** Each checkpoint voice is offered in each language, named `<language>-<voice>`, such
  as `en-default`. A request without a voice gets the `default` voice, or the first one.
- **Streaming.** Audio goes to Android as it is generated, as 24 kHz 16-bit mono, in pieces no
  larger than Android's buffer.
- **Stopping.** When Android stops a request, the engine answers the library's next piece of
  audio with false, which ends generation there.
- **Leading silence.** Each utterance starts with a stretch of near-silence before the speech.
  `LeadingSilence` holds audio back until a 10 ms window reaches -40 dBFS, then plays from 20 ms
  before it, so the first sound is heard sooner. It drops at most 300 ms, so a quiet start still
  plays, and only touches the start of an utterance.
- **Threads.** The engine sets `RAYON_NUM_THREADS` to 2 before loading, as the Android guide
  suggests. Measure on your target phones before changing it.

The model loads once, off the main thread, and stays loaded while the service runs.

## Not done yet

- **Speech rate and pitch** are ignored, so the system sliders do nothing. Rate could come from
  the `duration_delta` conditioner once a checkpoint has one, which needs the conditions set per
  request rather than once at load, through the C interface. Pitch has no model control; it
  would take signal processing on the audio, as would rate on a checkpoint without that
  conditioner.
- **Downloading a model** from a configurable repository. See `Model.install` above.
- **The Snapdragon NPU.** The [QNN AAR](../../qnn/android/README.md) has its own `PhononTTS` with a
  close interface (load, voices, speak with a callback, stop). The engine could pick it when the
  phone has a supported Snapdragon and an exported bundle is installed, and fall back to this CPU
  path otherwise. The two classes share a name and package, so one would need renaming first.

## Test

`src/test` checks `LeadingSilence` on the JVM:

```sh
./gradlew testDebugUnitTest
```

`src/androidTest` drives the installed engine through `TextToSpeech`, as any app would. It needs a
device or emulator; with a model imported, it writes WAV files for English, French and every
English voice, and checks that `QUEUE_FLUSH` cuts a long utterance short. Without one, it checks
that every language reports missing data and synthesis fails cleanly.

```sh
./gradlew connectedDebugAndroidTest
adb shell run-as ai.gradium.phonon.tts ls files   # the WAVs it wrote
```

## Files

| File | What it is |
|---|---|
| `PhononService.kt` | The `TextToSpeechService`: languages, voices, synthesis and stop. |
| `Model.kt` | The installed checkpoint and the loaded model, shared by the service and the settings. |
| `LeadingSilence.kt` | Drops the near-silence each utterance starts with. |
| `SettingsActivity.kt` | Status, folder import, and a link to the system settings. |
| `DataActivity.kt` | Answers the system settings' data check and sample text requests. |
