# Phonon from C

Phonon has a C interface: one header, [`include/ptts.h`](include/ptts.h), and one library,
`libptts_ffi`. Any language that can call C can use it: C and C++, Flutter through `dart:ffi`,
C# and Unity, Go, Zig and others. The [Swift package](../ios/PhononTTS/README.md) and the
[Android Kotlin wrapper](../android/README.md) are both built on it. If you write a Swift or
Kotlin app, start with those guides instead.

The same calls run on one of two engines, chosen by the platform you build for:

| Platform | Engine | `ptts_new` loads | Compute units |
|---|---|---|---|
| Linux, Windows, Android | CPU | a checkpoint folder | `PTTS_UNIT_CPU` |
| macOS, iOS | Core ML | a Core ML bundle | `PTTS_UNIT_ANE` (the Neural Engine) or `PTTS_UNIT_CPU` |

On Apple platforms the library runs Core ML only, so it cannot load a checkpoint folder there.

## 1. Build the library

You need [Rust](https://rustup.rs). From the root of a checkout:

```sh
cargo build --release -p ptts-ffi
```

This writes the library to `target/release/`:

| Platform | Shared library | Static library |
|---|---|---|
| Linux | `libptts_ffi.so` | `libptts_ffi.a` |
| macOS | `libptts_ffi.dylib` | `libptts_ffi.a` |
| Windows | `ptts_ffi.dll`, with its import library `ptts_ffi.dll.lib` | `ptts_ffi.lib` |

The shared library is the simpler one to link. The static one also needs the system libraries it
uses, which this lists:

```sh
cargo rustc --release -p ptts-ffi --crate-type staticlib -- --print native-static-libs
```

The repository's `.cargo/config.toml` builds for the CPU of the machine you build on. Such a
library can crash on an older CPU. To ship it to other machines, set a baseline instead.
`RUSTFLAGS` replaces that file's setting. The release builds use `x86-64-v3` on x86-64,
`generic` on arm64 Linux and `apple-m1` on Apple silicon:

```sh
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release -p ptts-ffi
```

Phones and Apple apps have their own build scripts:

- **Android:** [`android/build.sh`](../android/build.sh) builds `libptts_ffi.so` for phones and
  the emulator. See the [Android guide](../android/README.md#1-build-the-library).
- **iOS and macOS apps:** [`ios/build-xcframework.sh`](../ios/build-xcframework.sh) builds
  `PhononCore.xcframework`: the static library and `ptts.h` for iPhone, the simulator,
  Apple-silicon Macs and Mac Catalyst. Each
  [release](https://github.com/gradium-ai/xn-ptts/releases) has it prebuilt as
  `PhononCore.xcframework.zip`, so an Xcode project needs no Rust.

## 2. Get a model

The library never downloads. You give `ptts_new` a folder.

**Linux, Windows and Android** load a checkpoint folder, laid out as the model's Hugging Face
repo is:

| File | What it is |
|---|---|
| `config.json` | Required. The checkpoint's own model configuration; there is no fallback config. |
| `tokenizer.json` | The checkpoint's tokenizer. |
| `model.q8.gguf` or `model.safetensors` | The weights: q8_0 GGUF, or f32 safetensors. GGUF is used when both are there. |
| `voices/<name>.safetensors` or `embeddings/<name>.safetensors` | The voices, each named after its file. |
| `default-voice.safetensors` | Optional. Listed as the voice `default`, and spoken when no voice is set. |

With the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), this
downloads the files the library needs from the default model's repo:

```sh
hf download gradium/phonon-31m-en config.json tokenizer.json model.q8.gguf default-voice.safetensors --local-dir phonon-31m-en
```

The [README](../README.md#model-setup) lists the other models. The Phonon 40m models are gated:
accept the terms on the model's Hugging Face page before you download them. To add a voice, put
its file in `voices/`. It can come from the [voice catalog](../README.md#voices) or from Gradium
Studio.

**macOS and iOS** load a Core ML bundle. Each
[release](https://github.com/gradium-ai/xn-ptts/releases) has one for the default model, as
`phonon-31m-en-coreml.zip`. For another model, export a bundle with `export_coreml`, as the
[Swift guide](../ios/PhononTTS/README.md#1-build-the-two-pieces-that-are-not-in-the-source)
shows. Core ML compiles each model beside itself the first time it loads, so the folder must be
writable.

## 3. The interface

[`ptts.h`](include/ptts.h) has eight calls:

| Call | What it does |
|---|---|
| `ptts_new(dir, unit, lang)` | Load the model folder. This is the slow call. |
| `ptts_voices(h)` | The voice names, as one block of NUL-separated strings. |
| `ptts_set_voice(h, name)` | Speak in one of them from now on. |
| `ptts_languages(h)` | The languages the model was trained for, in the same format. |
| `ptts_set_language(h, lang)` | Normalize text as `lang` from now on, with no reload. |
| `ptts_speak(h, text, callback, user, &result)` | Speak `text`, passing the audio to `callback` as it is made. |
| `ptts_last_error(h)` | Why the last call failed. Pass `NULL` after a failed `ptts_new`. |
| `ptts_free(h)` | Free the model. Nothing else does. |

Every binding follows the same rules:

- **Load once.** Keep one handle for the life of the app rather than one per sentence.
- **Calls block.** Make them off the UI thread, and from one thread at a time.
- **The audio is 24 kHz mono `float` in [-1, 1].** The callback runs on the thread that called
  `ptts_speak`, with one or more frames at a time. Return `false` from it to stop early. It must
  not call back into the same handle. Generation runs on its own threads with bounded buffers, so
  a callback that waits for playback pauses generation once those buffers fill.
- **`lang` is required.** `en`, `fr`, `de`, `es` and `pt` normalize numbers and symbols for that
  language, and `none` reads the text as written.
- **Errors.** `ptts_new` returns `NULL` on failure, and `ptts_speak`, `ptts_set_voice` and
  `ptts_set_language` return `false`. Then `ptts_last_error` says why.
- **Strings from the library are borrowed.** Those from `ptts_voices`, `ptts_languages` and
  `ptts_last_error` belong to the library: copy them if you keep them.
- **Set the thread count** of the CPU engine with the `RAYON_NUM_THREADS` environment variable,
  before the first `ptts_new`. The default is one thread per core, which is usually more than one
  stream of speech can use. Measure on your target machines.

## 4. Call it

```c
#include "ptts.h"

static bool on_audio(const float *pcm, size_t n, void *user) {
    // Queue `n` samples for playback.
    return true; // false stops early
}

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

[`examples/speak.c`](examples/speak.c) is a complete program. It loads a model, picks a voice and
writes the speech to a WAV file as it is made. On Linux or macOS, build the library, then the
program, from the root of the checkout:

```sh
cargo build --release -p ptts-ffi
cc ptts-ffi/examples/speak.c -I ptts-ffi/include -L target/release -lptts_ffi \
    -Wl,-rpath,"$PWD/target/release" -o speak
./speak phonon-31m-en en "Hello from Phonon." speech.wav
```

On macOS, pass the Core ML bundle instead of the checkpoint folder. On Windows, link
`ptts_ffi.dll.lib` and put `ptts_ffi.dll` beside the program.

## Other languages

Load the library with your language's usual way of calling C, and declare the calls from
`ptts.h`. In Flutter that is `dart:ffi`; in C#, `[DllImport("ptts_ffi")]`; in Unity, a native
plugin; in Go, cgo. Three details matter in any language:

- `ptts_speak`, `ptts_set_voice`, `ptts_set_language` and the callback use a C `bool`, which is
  one byte. JNA's `Boolean` and C#'s default `bool` are four bytes, and misread it.
- `size_t`, in `PttsResult` and the callback, is the size of a pointer. Map it to an unsigned
  pointer-sized integer, not a 32-bit one.
- The callback runs on the thread that called `ptts_speak`, so call it from a thread your runtime
  can run callbacks on.

The Swift package, the Kotlin wrapper and C are tested here. The other languages are not yet.
