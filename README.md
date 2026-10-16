# Phonon

**Streaming speech, on your device.**

Phonon brings natural text-to-speech to phones, laptops, browsers, and local services. It is built for offline assistants, accessibility tools, and interactive experiences that need speech without a cloud round trip. The runtime is written in Rust and needs no PyTorch.

[![Rust CI](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml/badge.svg?branch=main)](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml)
[![Python wheels](https://github.com/gradium-ai/xn-ptts/actions/workflows/maturin-pub.yml/badge.svg?branch=main)](https://github.com/gradium-ai/xn-ptts/actions/workflows/maturin-pub.yml)
[![Browser package](https://github.com/gradium-ai/xn-ptts/actions/workflows/npm-publish.yml/badge.svg?branch=main)](https://github.com/gradium-ai/xn-ptts/actions/workflows/npm-publish.yml)
[![Native packages](https://github.com/gradium-ai/xn-ptts/actions/workflows/cli-release.yml/badge.svg?branch=main)](https://github.com/gradium-ai/xn-ptts/actions/workflows/cli-release.yml)
[![Docker builds](https://github.com/gradium-ai/xn-ptts/actions/workflows/docker.yml/badge.svg?branch=main)](https://github.com/gradium-ai/xn-ptts/actions/workflows/docker.yml)

[![Code license: MIT OR Apache-2.0](https://img.shields.io/badge/code-MIT%20OR%20Apache--2.0-blue)](#license)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/T85yt8kD33)

[![Published Rust version](https://img.shields.io/crates/v/ptts?label=crates.io)](https://crates.io/crates/ptts)
[![Published Python version](https://img.shields.io/pypi/v/ptts?label=PyPI)](https://pypi.org/project/ptts/)
[![Published npm version](https://img.shields.io/npm/v/phonon-tts?label=npm)](https://www.npmjs.com/package/phonon-tts)
[![Native downloads](https://img.shields.io/github/v/release/gradium-ai/xn-ptts?include_prereleases&label=native%20downloads)](https://github.com/gradium-ai/xn-ptts/releases)
[![Swift package](https://img.shields.io/badge/Swift-download-orange)](#swift)
[![Android QNN package](https://img.shields.io/badge/Android-QNN%20source%20preview-blue)](#android)

[Try it](#try-it) · [What you get](#what-you-get) · [Choose an integration](#choose-an-integration) · [Quickstart](#quickstart) · [Guides](#guides)

> **Runtime preview available:** Python, Rust, browser packages, and native downloads are published. See the [runtime preview release](https://github.com/gradium-ai/xn-ptts/releases/tag/v0.4.0-rc.2) for installation instructions. The unversioned registry commands below target the upcoming stable release and currently select older packages. Phonon model files are available separately from the runtime packages. Docker public access is being finalized.

## Try it

With [uv](https://docs.astral.sh/uv/), one line writes a WAV:

```sh
uvx ptts --lang en "Hello from Phonon." -o speech.wav
```

The first run downloads the default model and caches it. To run the runtime preview now, use `uvx --prerelease allow ptts`. See [Model setup](#model-setup) to use another model.

## What you get

- **Small enough to ship:** Choose a 31M, 56M or 98M parameter model. Each can ship inside a mobile app or load in a browser tab.
- **Five languages in one model:** Speak English, French, German, Spanish, and Portuguese, with more languages planned.
- **Fast across devices:** A custom inference stack runs in browsers and on phones, laptops, and embedded hardware, with Core ML on Apple devices, a QNN preview for supported Snapdragon NPUs, and GPU support. Multiple backends deliver faster inference than Kokoro and llama.cpp-based runtimes on the same hardware.
- **Voices ready to use or make your own:** Choose from ready-to-use voices, or design or clone your own and bring it to your device with [two scripts](#voices).
- **Control the delivery:** Adjust speaking speed and start playing streamed audio before the full utterance is ready.

## Performance

Three model sizes, one inference stack. Each cell is how many seconds of audio Phonon makes per
second of compute (higher is faster; above 1x is faster than real time) and the time to first
byte (TTFB). CPU rows use a single core; the Metal and Neural Engine rows run on the accelerator.

| Device | 31M | 56M | 98M |
|---|---|---|---|
| MacBook Air M5, CPU, 1 core | 26x · 5 ms | 22x · 6 ms | 17x · 11 ms |
| MacBook Air M5, Metal GPU | 29x · 4 ms | 25x · 5 ms | 20x · 10 ms |
| iPhone 16 Pro, Neural Engine | 22x · 9 ms | 19x · 13 ms | 15x · 17 ms |
| Raspberry Pi 5, CPU, 1 core | 2.3x · 68 ms | 2.2x · 92 ms | 1.8x · 191 ms |

- Every size runs faster than real time on a single CPU core, from a MacBook to a Raspberry Pi 5.
- TTFB stays under 20 ms on Apple devices, because Phonon streams frame by frame instead of waiting
  for the whole sentence.
- On one MacBook Air core, the 56M model runs at 22x real time with a 6 ms TTFB. Kokoro (82M), run
  through its own PyTorch package on the same core, runs at 7.3x with a 776 ms TTFB. Kokoro does not
  stream: it makes the whole sentence before returning any audio, so its TTFB is the time for the
  full sentence. Streaming frame by frame is what keeps Phonon's TTFB in milliseconds.

## Choose an integration

Each goal links to its quickstart below.

| You want to… | Package | Get it | Details |
|---|---|---|---|
| [Generate a speech file](#command-line) | Desktop `ptts` command | [GitHub Releases](https://github.com/gradium-ai/xn-ptts/releases) or [Homebrew tap](https://github.com/gradium-ai/homebrew-tap) | Linux x64/ARM64, Mac Apple silicon/Intel, and Windows x64 archives, with checksums. No Python or Rust needed. [CLI guide](docs/cli.md). |
| [Add speech to a script or backend](#python) | Python `ptts` | [PyPI](https://pypi.org/project/ptts/) | Wheels for supported platforms; CPython 3.9+. [Python guide](ptts-pyo3/README.md). |
| [Run speech in a web app](#browser) | `phonon-tts` | [npm](https://www.npmjs.com/package/phonon-tts) | Worker, single-thread and threaded Wasm builds, and TypeScript declarations. [Browser guide](ptts-wasm/js/README.md). |
| [Add speech to an iPhone or Mac app](#swift) | Swift `ptts` | [GitHub Releases](https://github.com/gradium-ai/xn-ptts/releases) | `ptts-swift-<version>.zip` and its matching compiled framework. Needs a prepared Core ML bundle. [Swift guide](ios/PhononTTS/README.md). |
| [Add speech to an Android app](#android) | Android `ptts` (QNN preview) | [QNN Android guide](qnn/android/README.md) | Kotlin API and AAR for supported Snapdragon NPUs. Not published to Maven Central yet. |
| | Android CPU library | [Build guide](android/README.md#1-build-the-library) | `libptts_ffi.so` and a Kotlin wrapper, built from source with Rust and the Android NDK. |
| [Give Home Assistant's voice assistant a local voice](ptts-openai-server/README.md#home-assistant) | Home Assistant app | [phonon-home-assistant](https://github.com/gradium-ai/phonon-home-assistant) | App for Home Assistant OS that runs the server image and connects it to your voice assistants. No custom integration. |
| [Connect an existing app or self-host](#docker-and-openai-compatible-api) | `ptts-openai-server` | `ghcr.io/gradium-ai/ptts-openai-server:<version>` | CPU image for amd64 and arm64; it downloads the default model on first start, or mounts your own. [Server guide](ptts-openai-server/README.md). |
| [Embed the runtime in Rust](#rust) | Rust `ptts` | [crates.io](https://crates.io/crates/ptts) | Library API; add the `cli` feature to install the command. [Rust guide](ptts/README.md). |

Phonon is the product and model identity. The Rust, Python, Swift, and planned Android Maven packages are named **`ptts`**; the browser npm package is **`phonon-tts`**. This repository remains `xn-ptts`. Swift apps use `import PhononTTS`.

Desktop downloads and Python wheels do not need a Rust compiler. Older x86 CPUs may need a source build: x86 wheels and desktop downloads target x86-64-v3. See the platform requirements in the [CLI guide](docs/cli.md) and [Python guide](ptts-pyo3/README.md).

## Model setup

Without a model argument, the command line, Python, the browser package, the servers and Rust download the default model from Hugging Face on first use and cache it. The default is currently [`hf://davencyw/pocket-tts-ptts`](https://huggingface.co/davencyw/pocket-tts-ptts), a test copy of Kyutai's [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) (English, CC-BY-4.0), pinned to one commit. Swift and Android load a model bundle you add to the app.

To use another model, pass a Hugging Face repo as `hf://OWNER/MODEL`, adding a commit as `hf://OWNER/MODEL@COMMIT_SHA` for a repeatable version, or a local checkpoint folder. Private repos require `HF_TOKEN` or a saved HF login. Local folders require no Hub access.

A checkpoint folder looks like this:

```text
model/
  config.json
  tokenizer.json
  model.q8.gguf
  default-voice.safetensors
```

Voice files may instead live in `voices/` or `embeddings/`. Use the files and voice names your checkpoint supplies; do not substitute another model's tokenizer or config. The command line, Python and the servers load f32 weights unless given `--quant q8` or `quant="q8"`, which runs faster on the CPU; the browser loads q8 unless given `quant: 'f32'`.

## Voices

Each checkpoint comes with its own voices, and there are ready-made voices in every supported language. `--list-voices` lists them, and `--voice NAME` speaks with one. Only the voices you use are downloaded, the first time you use them:

```sh
ptts --list-voices
ptts --lang pt --voice Kauã "Olá do Phonon." -o speech.wav
```

A name matches in any case and with or without accents, so `--voice kaua` works too. In Python, pass the name as `voice=` to `TTS` or to any speech method, and `ptts.catalog_voices()` lists the voices.

You can also design a voice from a text description in [Gradium Studio](https://studio.gradium.ai/), or clone one from a short recording. Two scripts download these voices as voice files. They need [uv](https://docs.astral.sh/uv/) and a Gradium API key, which you can create in Gradium Studio. The voice files work with every Phonon checkpoint.

```sh
export GRADIUM_API_KEY=...

# List your voices, with their IDs
uv run scripts/download-voice.py --list

# Download a voice you designed or cloned, by ID
uv run scripts/download-voice.py VOICE_ID -o voice.safetensors

# Clone a voice from a recording; 10 seconds of clear speech are used
uv run scripts/clone-voice.py speaker.wav --lang en -o voice.safetensors

ptts --model hf://OWNER/MODEL --lang en --voice voice.safetensors "Hello from Phonon." -o speech.wav
```

To use a voice by name in Python and the servers too, save it in the checkpoint folder's `voices/`: `-o model/voices/NAME.safetensors` adds the voice `NAME`. A cloned voice is also saved in your Gradium account, so `download-voice.py` can fetch it again by the ID `clone-voice.py` prints. Clone a voice only with the speaker's consent.

## Quickstart

These examples use the default model. The install commands below target the upcoming stable release; to install the runtime preview now, follow the [release instructions](https://github.com/gradium-ai/xn-ptts/releases/tag/v0.4.0-rc.2) or the [source setup](docs/development.md).

### Command line

<details>
<summary>Show command line quickstart</summary>

[uv](https://docs.astral.sh/uv/) can run the Python command in an isolated environment:

```sh
uvx ptts --lang en "Hello from Phonon." -o speech.wav
```

`uvx` manages the Python environment and dependencies.

On Mac or Linux, install the prebuilt desktop command from Gradium's Homebrew tap. It currently installs the `0.4.0-rc.2` runtime preview:

```sh
brew install gradium-ai/tap/ptts
```

You can also extract a [desktop archive](docs/cli.md) and put `ptts` on your `PATH`. Neither path needs Python or Rust. Generate a WAV:

```sh
ptts --lang en "Hello from Phonon." -o speech.wav
```

The output is a mono 24 kHz WAV. Open it in your audio player.

If you prefer Cargo, install the command with `cargo install ptts --locked --features cli`.

To use another model, pass a local folder or a Hugging Face repo, and `--quant q8` for faster q8 weights on the CPU:

```sh
ptts --model hf://OWNER/MODEL@COMMIT_SHA --lang en --quant q8 \
  "Hello from Phonon." -o speech.wav
```

Use `--voice NAME` to choose a voice, and `--list-voices` to see the [ready-made ones](#voices); omitting it uses the model's default selection. Run `ptts --help` for the remaining options. [CLI guide →](docs/cli.md)

</details>

### Python

<details>
<summary>Show Python quickstart</summary>

Install the Python package into your environment:

```sh
python -m pip install ptts
```

Then save speech in four lines:

```python
import ptts

tts = ptts.TTS(lang="en")
tts.save("speech.wav", "Hello from Phonon.")
```

Reuse `tts`. `tts.synth(text)` returns a float32 NumPy array; `tts.stream(text)` yields audio chunks:

```python
with tts.stream("Speech can play while the rest is being generated.") as audio:
    for pcm in audio:
        # Send each chunk to your player at audio.sample_rate.
        print(pcm.shape)
```

Leaving the `with` block stops generation and releases its workers. To use another model, pass a local folder or `"hf://OWNER/MODEL@COMMIT_SHA"` as the first argument.

The wheel also provides a command:

```sh
python -m ptts --lang en "Hello from Phonon." -o speech.wav
```

Rust and Python both install a command named `ptts`, with different flags. `python -m ptts` explicitly selects Python. [Python guide →](ptts-pyo3/README.md)

</details>

### Browser

<details>
<summary>Show browser quickstart</summary>

Install the browser package into your web app:

```sh
npm install phonon-tts
```

Then generate a WAV:

```js
import { PhononTTS } from 'phonon-tts';

const tts = await PhononTTS.load({ lang: 'en' });
const wav = await tts.synthWav('Hello from Phonon.'); // WAV Blob
```

`model` picks another one: a public Hugging Face repo as `'hf://OWNER/MODEL@COMMIT_SHA'`, or the URLs of files you serve with your app.

Use the returned `Blob` for a download link or your audio player. `tts.stream(text)` yields mono `Float32Array` chunks for streaming playback; the [browser guide](ptts-wasm/js/README.md#streaming) includes a complete Web Audio example. Call `tts.dispose()` when finished with the model.

The first load downloads and caches the files. `onProgress` reports download progress. Serve over HTTPS or localhost for caching, and use a browser with WebAssembly SIMD. The package runs in browsers, not Node.js. CPU is the default. Threading and WebGPU setup are covered in the [browser guide →](ptts-wasm/js/README.md).

</details>

### Swift

<details>
<summary>Show Swift quickstart</summary>

Extract `ptts-swift-<version>.zip`, add the folder to Xcode as a local package, and select the **`ptts`** library product. Swift Package Manager downloads the matching compiled framework. This path needs no Rust build.

Add the prepared Core ML model bundle to your app as a folder reference named `Models`:

```swift
import Foundation
import PhononTTS

guard let bundled = Bundle.main.url(forResource: "Models", withExtension: nil) else {
    throw PhononError(description: "Add the model folder to your app as Models.")
}
let models = try PhononModels.install(bundled: bundled)
let tts = try await Phonon.load(models: models, language: .english)
let player = try PhononPlayer()
try await player.play(tts.stream("Hello from Phonon."))
```

Keep `tts` and `player` in your app's state while audio plays. You can also download models on first run. Requires iOS 18+ or macOS 15+ on Apple silicon. The public Core ML model bundle is still being prepared. [Swift guide and Stop example →](ios/PhononTTS/README.md)

</details>

### Android

<details>
<summary>Show Android integration options</summary>

**Snapdragon NPU:** The [QNN Android preview](qnn/android/README.md) provides a Kotlin API and an AAR around the optimized QNN engine. It requires Android 12+, ARM64, working QNN HTP support and a model bundle compiled for the phone's SoC. There is no automatic CPU fallback. Build the AAR from this repository using its build guide. The planned Maven package is `ai.gradium:ptts`; it is not published yet.

Load a supplied compiled model folder on a worker thread and stream audio to your player:

```kotlin
import ai.gradium.phonon.PhononTTS

val tts = PhononTTS.load(context, modelDirectory, lang = "en")
val sampleRate = tts.sampleRate
tts.speak("Hello from Phonon.") { pcm ->
    audioSink.write(pcm) // Mono float PCM at sampleRate.
    true // Return false to stop; tts.stop() also works from another thread.
}
// Reuse tts. Stop and wait for speech to finish before calling tts.close().
```

The preview guide covers native library extraction, model bundles and the runnable Speak/Stop example.

**CPU:** The existing [Android guide](android/README.md) provides a Kotlin wrapper and C API. From the repository root:

```sh
cargo install cargo-ndk
export ANDROID_NDK_HOME=/path/to/ndk
./android/build.sh
```

Copy `android/jniLibs` into your app's `src/main/`, then add the Kotlin wrapper and JNA dependency as described in the guide. Supply a checkpoint with its own config, tokenizer, weights and voices. Load `PhononTTS(modelDir, "en")` on a worker thread and reuse it. The guide includes AudioTrack playback, CPU requirements and callback cancellation.

</details>

### Docker and OpenAI-compatible API

<details>
<summary>Show Docker and API quickstart</summary>

Choose an image version from [GitHub Releases](https://github.com/gradium-ai/xn-ptts/releases) and replace `<release-version>` below, without the leading `v`. Public GHCR access is being finalized; until then use an image you can access or the [source setup](docs/development.md#docker). The Docker commands below use Bash or another POSIX shell. Start the speech server, which downloads the default model on first start:

```sh
PTTS_IMAGE="ghcr.io/gradium-ai/ptts-openai-server:<release-version>"
docker run --rm -p 127.0.0.1:8880:8880 -e PTTS_LANG=en -e PTTS_QUANT=q8 "$PTTS_IMAGE"
```

In a second terminal:

```sh
curl http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Phonon.","voice":"default","response_format":"wav"}' \
  -o speech.wav
```

For applications with an OpenAI-compatible TTS setting, use `http://localhost:8880/v1` as the base URL and a voice the model has (`GET /v1/audio/voices` lists them). The [server guide](ptts-openai-server/README.md) includes client setup recipes, supported formats, and limits.

For a download cache that survives container replacement, use the [Hub Compose file](ptts-openai-server/compose.hub.yaml). Set `PTTS_MODEL` to serve another model: `hf://OWNER/MODEL@COMMIT_SHA`, or a folder mounted into the container. The image contains no model weights. Deployment and access controls are in the [server guide →](ptts-openai-server/README.md).

</details>

### Rust

<details>
<summary>Show Rust quickstart</summary>

Add `ptts` with tokenizer support to your Rust project, or use a [path dependency](docs/development.md#rust) to develop from source:

```sh
cargo add ptts --features hf,hub
```

```rust
use ptts::checkpoint::{Checkpoint, DEFAULT_MODEL, ResolveOptions};
use ptts::preprocess::Lang;
use ptts::synth::{DeviceKind, Quant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint =
        Checkpoint::locate(DEFAULT_MODEL, ResolveOptions { quant: Quant::Q80, weights: None })?;
    let mut tts = checkpoint.builder(Lang::En).device(DeviceKind::Cpu).build()?;
    checkpoint.register_voices(&mut tts);
    let pcm = tts.say("Hello from Phonon.")?;
    ptts::wav::write_wav_file("speech.wav", &pcm, tts.sample_rate())?;
    Ok(())
}
```

Build with `--release` and reuse the model. `tts.stream(text)?` yields audio chunks for your own playback. `Checkpoint::locate` also takes a local folder or `hf://OWNER/MODEL@COMMIT_SHA`. Without the `hub` feature the library never downloads, and reads local folders only. [Rust API and small example →](ptts/README.md)

</details>

## Guides

Each integration's guide is linked from its row in [Choose an integration](#choose-an-integration) and from its quickstart. Two more:

- [WebSocket server](ptts-ws-server/README.md): streaming text and audio over one connection.
- [Development](docs/development.md): build packages, run the local demo, and develop from source.

### Questions and feedback

Report bugs or request integrations through [GitHub Issues](https://github.com/gradium-ai/xn-ptts/issues). Include your runtime version, device/OS, model revision, weight format, normalization language, and a small reproduction.

## Acknowledgements

Phonon builds on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), developed by Kyutai, and uses the [xn](https://github.com/LaurentMazare/xn) Rust tensor runtime. Gradium's Phonon checkpoints are the primary target. Compatible Pocket TTS checkpoints use the same explicit loading interface when they supply their own config, tokenizer JSON, weights, and voices.

## License

The code is **MIT OR Apache-2.0**, at your option: [MIT](LICENSE-MIT) · [Apache-2.0](LICENSE-APACHE).

Model weights and voices are distributed separately and have their own licenses. Check the model license before redistribution.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this repository is dual licensed as above, without additional terms or conditions.
