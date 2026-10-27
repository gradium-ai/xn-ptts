# ptts for Swift

Phonon text to speech for your own iOS or macOS app. The model runs on the device, with its
transformer on the Apple Neural Engine, and audio streams as it is generated: on an iPhone 16 Pro
speech starts in under 40 ms and is produced 12 times faster than it plays.

```swift
import Foundation
import PhononTTS

guard let bundled = Bundle.main.url(forResource: "Models", withExtension: nil) else {
    throw PhononError(description: "Add the exported model folder to your app as a folder reference named Models.")
}
let models = try PhononModels.install(bundled: bundled)
let tts = try await Phonon.load(models: models, language: .english)
let player = try PhononPlayer()
try await player.play(tts.stream("Hello! This is running entirely on the phone."))
```

It needs iOS 18 or macOS 15 on Apple silicon and Xcode 16. Mac Catalyst apps work too, on Apple
silicon. Installed models take about twice the size of the model folder, because Core ML keeps a
compiled copy beside each model, and a folder bundled in the app also stays in the app. Speaking
takes about 95 MB of memory. The simulator works, but without a Neural Engine it is several times
slower than a device. The compiled core is built for arm64 only: set Excluded Architectures to
`x86_64` in your app target, or a build for Any Mac, a Mac archive, a Mac Catalyst release build or
a Release build for the simulator fails to link.

Keep `tts` and `player` as properties of an object that stays alive, not as local constants in a function. `play` returns when generation finishes; releasing the player stops any audio still scheduled.

## Released package

The Swift package is named `ptts`; its module remains `PhononTTS`. Existing consumers selecting `.product(name: "PhononTTS", package: ...)` must select `ptts` as the product name; their `import PhononTTS` stays the same. Neither way below needs Rust, and neither contains model weights.

**From the repository.** In Xcode, choose *File, Add Package Dependencies*, enter `https://github.com/gradium-ai/xn-ptts.git`, choose a release version and add the `ptts` library to your app target. In a `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/gradium-ai/xn-ptts.git", from: "<version>"),
],
// and in your target:
dependencies: [.product(name: "ptts", package: "xn-ptts")],
```

Each release tag has a `Package.swift` at the repository root. It downloads that release's compiled core, `PhononCore.xcframework.zip`, and checks it against its exact checksum.

**From the release archive.** Each [GitHub Release](https://github.com/gradium-ai/xn-ptts/releases) also has `ptts-swift-<version>.zip`, the same package on its own. Extract it, add the folder to Xcode as a local package and select the `ptts` library. Swift Package Manager downloads the matching compiled core automatically.

You still need a Core ML model bundle. Each [GitHub Release](https://github.com/gradium-ai/xn-ptts/releases) has one for Phonon 31m, the default model, as `phonon-31m-en-coreml.zip`, with the English voices of the voice catalog. Unzip it, rename the folder `phonon-31m-en-coreml` to `Models`, and add it as described below. For another model, such as one of the gated Phonon 40m models, export a bundle yourself with `export_coreml`, as [step 1](#1-build-the-two-pieces-that-are-not-in-the-source) shows; with the released package, you can skip building the framework there. Bundle the folder with the app, or host it and download it with `PhononModels.download(from:progress:)`, as described below.

Source development uses this folder's `Package.swift` and the local framework built by `ios/build-xcframework.sh`. The root `Package.swift` always fetches a released framework, so on `main` it can lag behind these sources.

## 1. Build the two pieces that are not in the source

From the root of a checkout, with Rust installed:

```bash
./ios/build-xcframework.sh                                                          # PhononCore.xcframework
cargo run --release -p ptts --example export_coreml --features hub -- phonon-coreml   # the default model, as Core ML
```

The first is the compiled core the package wraps, written into `ios/PhononTTS` where the package
expects it. The second downloads the default model, Phonon 31m, from Hugging Face and exports it
as Core ML. For another model, add `--model`: a Hugging Face repo, such as
`--model hf://gradium/phonon-40m-fr`, or a local checkpoint folder with `config.json` and the
weights. The Phonon 40m models are gated: accept the terms on the model's Hugging Face page, then
set `HF_TOKEN` or run `hf auth login` before exporting.

The bundle gets the checkpoint's own voices. `--voices <folder>` takes the voice files in that
folder instead, each named after its file.

If the model takes conditions, such as `duration_delta`, set them here with
`--condition duration_delta=0.3` (repeatable): they are fixed in the exported folder, and those not
given take their defaults.

## 2. Add the package and the models

In Xcode, choose *File, Add Package Dependencies, Add Local*, select `ios/PhononTTS`, and add the
`ptts` library (module `PhononTTS`) to your app target. Then rename the model folder to `Models`
and add it to the target as a folder reference. The built app must contain one `Models` folder
with the `.mlpackage` files inside. In Xcode 16 and later, a synchronized folder also shows as a
blue folder, but Xcode compiles the models in it and copies its files loose. If Xcode adds the
folder that way, set its Build Rules to Apply Once to Folder in the File inspector.

Core ML compiles each model beside itself, so the models have to be copied out of the read-only
app bundle once. `PhononModels.install(bundled:)` does that, into Application Support and
excluded from backup, and does nothing when the installed copy is already current.

To keep the app small instead (the App Store asks before downloading an app over 200 MB on a
cellular connection), host the folder on any static server and download it on first run with
`PhononModels.download(from:progress:)`. Every file is checked against `bundle.json`, and an
interrupted download resumes.

## 3. Speak

Load once and keep the instance. The first load after an install or a model update compiles the
models for the device, about 10 s on an iPhone 16 Pro; later loads take about 0.6 s.
`Phonon.needsCompiling(_:)` tells you in advance which it will be.

| | |
|---|---|
| `tts.speak(text) { samples in ... }` | Calls the closure with each 80 ms chunk as it is generated, then returns `SpeechStats`. |
| `tts.stream(text)` | The same audio as an `AsyncThrowingStream<[Float], Error>`. |
| `tts.synthesize(text)` | All the samples at once. |
| `tts.voices`, `tts.setVoice(_:)` | The voices in the model folder, and switching between them (about 0.6 s). |
| `tts.languages` | The languages the model was trained for, as its config lists them. |
| `tts.setLanguage(_:)` | Normalize text for another language from the next utterance on, with no reload. Pair it with a voice of that language. |
| `PhononPlayer` | Plays chunks as they arrive. `enqueue(_:)` and `stop()` work from any thread. |

Audio is mono Float32 at 24 kHz. Every method can be called from any thread or task, and an
instance speaks one utterance at a time. Cancelling the task that called `speak` or consumes a
`stream` stops generation at the next chunk. Breaking out of a stream loop alone does not reliably
stop its producer; cancel the consuming task when interrupting speech. Long text is split into
sentences and spoken with no gap.

### Stop and speak again

Keep the playback task so your Stop button can cancel it:

```swift
let playback = Task {
    try await player.play(tts.stream("A longer piece of text."))
}
```

In the Stop button handler, stop both generation and audio already scheduled:

```swift
playback.cancel()
player.stop()
```

Use the same `tts` and `player` for the next request. The model serializes calls, so the next
utterance waits for the cancelled generation to finish its current operation. `player.stop()`
only clears scheduled playback; it does not cancel the task producing more audio.

`language` is required: numbers and symbols are spelled out before speaking, and how depends on
the language. `.none` speaks the text as written. `computeUnit: .cpu` is a slower fallback
whose audio differs slightly from the Neural Engine's, which computes in 16-bit floating point.

## Performance

On an iPhone 16 Pro, speaking a 2.3-second sentence:

| | Realtime factor | First audio |
|---|---|---|
| Neural Engine, a few seconds after the previous call | 12x | 37 ms |
| Neural Engine, calls back to back | 14.5x | 16 ms |
| CPU, a few seconds after the previous call | 10x | 55 ms |

The first two rows differ because iOS lowers the CPU's clock while an app is idle.

## Licensing

The code is MIT or Apache-2.0, like the rest of the repository. The Phonon model weights are not
part of this repository and have their own licenses: Phonon 31m is under Apache 2.0, and Phonon
40M and Phonon 40m multilingual are under CC BY-NC 4.0. See each model's page on Hugging Face.
