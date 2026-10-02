# Phonon

Phonon is Gradium's on-device text-to-speech runtime, written in Rust, with Python bindings. It builds on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), developed by Kyutai. This preview pairs the code in this repository with a model package supplied by Gradium; the model is not in this repository.

[![Rust CI](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml)

## 1. Set up

You need [Rust](https://rustup.rs) for every path, and [uv](https://docs.astral.sh/uv/) for Python.

Point `MODEL_DIR` at the model folder, the one holding `config.json`, `model.q8.gguf`, `tokenizer.json` and `voices/`:

```bash
export MODEL_DIR=/path/to/model
```

## 2. Run it

With Rust, from the repository root (the first build takes a few minutes):

```bash
cargo run --release -p ptts --example ptts --features hf,audio -- \
  --lang en --dir "$MODEL_DIR" --quant q8 "Hello world" -o out.wav
```

With Python, from the repository root (the first run builds the package, a few minutes):

```bash
uv run --project ptts-pyo3 --locked ptts --lang en \
  --model "$MODEL_DIR/config.json" --quant q8 "Hello world" -o out.wav
```

`--quant q8` runs the model in q8, the format `model.q8.gguf` is stored in. Without it the weights are expanded to f32, which is slower and uses more memory; the Rust and Python examples below set q8 too. `--lang` is required. It picks how numbers, symbols and abbreviations are spelled out before synthesis: `en`, `fr`, `de`, `es` or `pt`, or `none` to use the text as written.

When no voice is specified, the Rust, Python and Swift frontends select the first voice by name, `Freya` in this package. For a fixed choice, pass `--voice Freya` to either CLI, `voice="Freya"` to Python, or call `tts.setVoice("Freya")` in Swift.

## 3. Use it from Rust

Add the crate from your checkout as a path dependency:

```toml
[dependencies]
ptts = { path = "/path/to/xn-ptts/ptts", features = ["hf"] }
serde_json = "1"
```

```rust
use std::{env, fs, path::PathBuf};
use ptts::preprocess::{Lang, Normalize};
use ptts::synth::{Quant, Synth};
use ptts::tts_model::TTSConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(env::var("MODEL_DIR")?);
    let config: TTSConfig = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
    let tts = Synth::builder(config, dir.join("model.q8.gguf"), Normalize::for_lang(Lang::En))
        .tokenizer_file(dir.join("tokenizer.json"))
        .quant(Quant::Q80)
        .add_voice("Freya", dir.join("voices/Freya.safetensors"))
        .build()?;

    let pcm = tts.say("Hello world")?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate() as u32)?;
    Ok(())
}
```

Load the model once and reuse it. `tts.say` returns the whole waveform as mono `f32` samples at `tts.sample_rate()`. `tts.stream(text)?` is an iterator of `Result<Vec<f32>>` chunks, yielded as they are generated, for playback that starts before the sentence is finished. Replace `Freya` with another supplied voice name and filename to select it. Build with `--release`: a debug build is far too slow for realtime.

## 4. Use it from Python

Install the package from your checkout into your project. This compiles the Rust code, so it needs Rust installed:

```bash
uv add /path/to/xn-ptts/ptts-pyo3      # or: pip install /path/to/xn-ptts/ptts-pyo3
```

```python
import os
import ptts

model = os.environ["MODEL_DIR"]
tts = ptts.TTS(lang="en", config=f"{model}/config.json", quant="q8")

tts.save("out.wav", "Hello world")    # write a 16-bit WAV
pcm = tts.synth("Hello world")        # float32 NumPy array at tts.sample_rate
with tts.stream("A longer sentence, played as it is generated.") as audio:
    for chunk in audio:
        ...                           # each chunk is a float32 NumPy array
```

Load the model once and reuse it. The [Python README](ptts-pyo3/README.md) covers voices and the remaining options.

## 5. Use it in an iOS or macOS app

The `PhononTTS` Swift package runs the model on the device through Core ML, with its transformer on the Apple Neural Engine: about 12 times faster than realtime on an iPhone 16 Pro, with first audio in under 40 ms. It needs iOS 18 or macOS 15, and Xcode.

Build the package's compiled core and convert the model to Core ML, both from the repository root:

```bash
./ios/build-xcframework.sh
cargo run --release -p ptts --example export_coreml -- --dir "$MODEL_DIR" phonon-coreml
```

Then add `ios/PhononTTS` to your Xcode project as a local package, add the `phonon-coreml` folder to your app as a folder reference named `Models`, and speak:

```swift
import PhononTTS

let models = try PhononModels.install(bundled: Bundle.main.url(forResource: "Models", withExtension: nil)!)
let tts = try await Phonon.load(models: models, language: .english)
try await PhononPlayer().play(tts.stream("Hello world"))
```

The [package README](ios/PhononTTS/README.md) covers downloading the models instead of bundling them, voices, and the rest of the API.

## 6. Use it in the browser

The `phonon-tts` JavaScript package runs the model in the page, compiled to WebAssembly, in a Web Worker: on the GPU through WebGPU when the browser offers it, and on the CPU otherwise. Build it from the repository, which needs Rust with the `wasm32-unknown-unknown` target, [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/), Node 22.7 or later, [binaryen](https://github.com/WebAssembly/binaryen/releases) 124 or later, and a pinned nightly toolchain for the package's multithreaded build, which `make threads-toolchain` installs:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
brew install binaryen           # or a release from GitHub: distribution packages are often older than 124

cd ptts-wasm
make threads-toolchain          # once
make build                      # the package, in ptts-wasm/pkg
cd pkg && npm pack              # and as a tarball, phonon-tts-<version>.tgz
```

Install the tarball into your web app, and serve the model folder with the app's static files, here under `/model/`:

```bash
npm install /path/to/xn-ptts/ptts-wasm/pkg/phonon-tts-*.tgz
```

Install the tarball rather than the `pkg` folder: npm links a folder instead of copying it, and Vite's dev server refuses to serve files from outside the app.

```js
import { PhononTTS } from 'phonon-tts';

const tts = await PhononTTS.load({
  lang: 'en',
  model: {
    weights: { q8: '/model/model.q8.gguf' },
    tokenizer: '/model/tokenizer.json',
    config: '/model/config.json',
    voices: {
      Freya: '/model/voices/Freya.safetensors',
      Harper: '/model/voices/Harper.safetensors',
      Sterling: '/model/voices/Sterling.safetensors',
      Toby: '/model/voices/Toby.safetensors',
    },
    defaultVoice: 'Freya',
  },
});

for await (const pcm of tts.stream('Hello from the browser.')) {
  // mono Float32Array chunks of 80 ms at tts.sampleRate, as they are generated
}
const wav = await tts.synthWav('Hello world');   // or a whole WAV Blob
```

Load the model once and reuse it. The first load downloads the model files and keeps them in the browser's Cache API, which needs the page served over `https://` or from `localhost`. Files are cached by URL, so when you replace the model, serve it under a new path (say `/model-v2/`) or call `clearCache()` first; otherwise the browser keeps using the old files. The browser needs WebAssembly Relaxed SIMD; this was tested in current Chrome. Bundlers such as Vite pick up the package's worker and wasm with no configuration. The [package README](ptts-wasm/js/README.md) covers streaming playback, voices and the remaining options.

`tts.device` says whether it runs on `'webgpu'` or `'cpu'`. On the CPU, generation runs on 3 threads when the page is served with these two headers, and on one thread otherwise. Pass `threads` to `load` to choose another number:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

With them, the page can only load cross-origin files that opt in through CORS, which matters if the model is served from another origin. `tts.threads` says how many threads it got, and `tts.threadsReason` why.
