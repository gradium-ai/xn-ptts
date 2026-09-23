<!--
  Assets still to supply: search this file for "TODO(assets)". The full checklist is at the bottom.
  The repo name gradium-ai/xn-ptts is used throughout. If the repo moves, rename it in one pass.
-->

<div align="center">

# Phonon

**Gradium's on-device text-to-speech. It turns text into 24 kHz speech with the Pocket TTS model, in one Rust runtime. That runtime ships as a Python package, a command-line tool, a WebSocket server and a browser build, all from the same source tree. No PyTorch anywhere.**

Built on [Pocket TTS](https://huggingface.co/kyutai/pocket-tts), the model by [Kyutai](https://kyutai.org). The runtime, the bindings, the server and the browser build are by [Gradium](https://gradium.ai). They ship under one name everywhere: **`ptts`** on PyPI, on crates.io, and at the command line.

[![Rust CI](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ptts)](https://pypi.org/project/ptts/)
[![crates.io](https://img.shields.io/crates/v/ptts)](https://crates.io/crates/ptts)
[![Python 3.9 and up](https://img.shields.io/pypi/pyversions/ptts)](https://pypi.org/project/ptts/)
[![Licence: MIT OR Apache-2.0](https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-blue)](#licence)

<!-- TODO(assets): hero video or GIF, 20 to 30 seconds. Type a sentence, hear it, show the real-time factor. -->
<!-- <video src="docs/assets/hero.mp4" width="720" autoplay loop muted></video> -->
> 🎬 *Demo video coming here.* Until then, [try it in your browser](https://laurentmazare.github.io/pocket-tts) <!-- TODO(assets): move the hosted demo under the org and update this link -->

</div>

---

## Thirty seconds to speech

```bash
uvx ptts --lang en "Hello world" -o out.wav     # nothing to install
```

```bash
pip install ptts                                 # numpy is the only dependency
```

```python
import ptts

tts = ptts.TTS(lang="en")
tts.save("out.wav", "Hello world")
```

The first run downloads the checkpoint (about 240 MB) from the Hugging Face Hub. The published checkpoint is gated. Accept its terms once on [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts). Then run `huggingface-cli login`, or set `HF_TOKEN`. See [Models, voices and languages](#models-voices-and-languages).

## What you get

- **24 kHz speech** from a flow-matching language model and the Mimi neural codec. Audio is streamed frame by frame as it is generated.
- **Voice cloning from about 10 seconds of audio.** You only need the audio. No transcript.
- **Six languages** in the published checkpoints: English, French, German, Italian, Portuguese and Spanish, with 26 voices each. See [Models](#models-voices-and-languages).
- **Text normalization** for en, fr, de, es and pt. Numbers, currency, dates and symbols are read the way a speaker of that language would say them.
- **Quantized weights** in ten GGML formats, from `q8_0` down to `q4k`. Make them with [`quantize`](ptts/examples/quantize.rs) and load them with one flag.
- **Backends**: CPU everywhere, plus Apple Accelerate, Metal, CUDA, Vulkan and WebGPU. Pick them at build time with Cargo features.
- **Errors you can catch.** A bad argument, a missing voice, a gated repo and running out of memory raise different exception classes, not the same string.

<!-- TODO(assets): a short "hear it" block: one sample per bundled voice, hosted. GitHub cannot play audio inline, so link to a page or the model card. -->

## Python

### Install

```bash
pip install ptts
```

One `abi3` wheel per platform works for every CPython from 3.9 on. Platforms: Linux x86_64 and aarch64 (glibc and musl), Windows x64 and ARM64, macOS Apple Silicon and Intel. Anything else installs from the source distribution. That is built and tested on every release.

### Use

```python
import ptts

tts = ptts.TTS(lang="en")                     # lang is required: spoken forms differ per language
print(tts.voices)                             # ['alba', 'azelma', 'cosette', ...]

pcm = tts.synth("Hello", voice="marius")      # 1-D float32 numpy array at tts.sample_rate
seconds = tts.save("out.wav", "Hello")        # straight to a mono 16-bit WAV

for chunk in tts.stream("A longer piece of text."):
    play(chunk)                               # audio as the decoder produces it
```

Streaming can be cancelled. Leave the block and the worker threads stop:

```python
with tts.stream(text) as audio:
    for chunk in audio:
        if user_interrupted():
            break
```

Ctrl-C works during a generation, not only between generations.

### Voices

Bundled voices come with the checkpoint. To clone a voice, pass about ten seconds of speech as float32 PCM at `tts.voice_prompt_sample_rate`. No transcript is needed:

```python
tts.clone_voice("me", my_pcm)                 # needs a checkpoint with a speaker encoder:
tts.save("out.wav", "Now in my voice.", voice="me")   #   check tts.supports_voice_cloning
```

A cloned voice is prepared once and kept for the life of the `TTS` object. Later calls reuse it.

### Checkpoints and backends

```python
ptts.TTS(lang="fr", config="model/config.json")   # a local checkpoint directory
ptts.TTS(lang="en", device="cuda")                # see ptts.available_devices()
ptts.TTS(lang="en", quant="q8_0")                 # smaller and faster on CPU; see ptts.available_quants()
```

Quantized weights are CPU-only.

### Command line

The wheel installs a `ptts` command. `python -m ptts` runs the same thing.

```bash
ptts --lang en "hello world" -o out.wav
ptts --lang en --list-voices
ptts --lang fr "bonjour" -v marius -q q8_0 -o out.wav
```

<details>
<summary>All flags</summary>

| Flag | |
|---|---|
| `--lang`, `-l` | **required**. One of `en`, `fr`, `de`, `es`, `pt`, or `none` to skip normalization |
| `-o`, `--output` | output WAV path (default `out.wav`) |
| `-v`, `--voice` | voice name. Defaults to the checkpoint's own |
| `-m`, `--model` | Hugging Face repo id, or path to a local `config.json` |
| `-d`, `--device` | `auto`, `cpu`, `cuda`, `vulkan`, `metal` |
| `-q`, `--quant` | weight format, for example `q8_0` |
| `-t`, `--temperature`, `-s`, `--seed` | sampling |
| `--threads` | CPU threads for tensor ops |
| `--list-voices`, `--build-info`, `--version` | |

</details>

### Errors

| Exception | Cause |
|---|---|
| `ValueError` | a bad argument: an unknown weight format, a mis-shaped array, a negative temperature |
| `LookupError` | an unknown voice, or a checkpoint file that is not there |
| `NotImplementedError` | a backend this wheel was not built with, or cloning on a checkpoint without a speaker encoder |
| `OSError` | the Hub could not be reached, or a file could not be read |
| `RuntimeError` | anything else |

### Types

The package ships `py.typed` and complete type stubs. Editors and `mypy --strict` see the whole API. Full reference: [`ptts-pyo3/README.md`](ptts-pyo3/README.md).

## Rust

```bash
cargo add ptts --features hf,audio
```

`ptts` reads the files you give it. It never downloads anything itself, so the library works offline and in embedded settings. It needs a checkpoint directory like this:

```
model/
├── model.safetensors        # or model.q8.gguf
├── tokenizer.json
└── voices/alba.safetensors  # one file per voice
```

```rust
use ptts::preprocess::{Lang, Normalize};
use ptts::synth::Synth;
use ptts::tts_model::TTSConfig;

let tts = Synth::builder(TTSConfig::v202601(0.5), "model/model.safetensors", Normalize::For(Lang::En))
    .tokenizer_file("model/tokenizer.json")
    .add_voice("alba", "model/voices/alba.safetensors")
    .build()?;

let pcm = tts.say("Hello world")?;
ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate() as u32)?;
```

Audio arrives in chunks from `stream`. `say` is built on it:

```rust
for chunk in tts.stream("A longer piece of text.")? {
    let pcm: Vec<f32> = chunk?;
}
```

A server that answers many requests for one voice uses a `Session`. It sets the KV budget once:

```rust
let session = tts.session(&ptts::synth::SpeechOptions::default().voice("alba"), 1024)?;
for line in lines {
    let pcm = session.say(line)?;
}
```

Cloning from audio needs the `audio` feature and a checkpoint with a speaker encoder:

```rust
use std::path::Path;

let pcm = ptts::audio::load_mono_at(Path::new("me.wav"), tts.voice_prompt_sample_rate())?;
tts.add_voice_from_pcm("me", &pcm)?;
```

Some callers must drive the loop themselves, for example a browser stepping from an event loop with no threads to spawn. They use [`TTSModel`](ptts/src/tts_model.rs) directly. `Synth` is built from those lower-level pieces and does not replace them. Full API on [docs.rs](https://docs.rs/ptts).

### Cargo features

| Feature | |
|---|---|
| `hf` | the tokenizer (`tokenizer.json`). Needed to speak at all |
| `audio` | decode and resample audio files, for voice cloning |
| `accelerate` | Apple Accelerate for CPU matmuls |
| `metal`, `cuda`, `vulkan`, `webgpu` | GPU backends. Quantized weights stay CPU-only |

`--all-features` does not build. The `cuda` feature needs `nvcc`.

### The CLI example

```bash
cargo run --release --example pocket_tts --features hf,audio -- --lang en "hello world" -o out.wav
```

It downloads from the Hub on first run. `--voice` takes a bundled voice id, a voice `.safetensors` file, or an audio file of about 10 seconds to clone. `--repo <id>` picks another Hub repo with the same layout. `kyutai/pocket-tts-without-voice-cloning` is the ungated one. `--dir <path>` loads a local checkpoint. `--weights model.q8.gguf --quant q8` loads pre-quantized weights. `--device` picks the backend. `--lang none` skips normalization.

## Server

`ptts-ws-server` streams audio over a WebSocket. Each connection gets its own voice-conditioned session.

```bash
cargo run --release -p ptts-ws-server -- --lang en --addr 0.0.0.0:8080
```

`--voice-dir <dir>` loads extra voices. `--max-seq-len` (default 4096) sets each session's KV budget. `--quant`, `--cuda`, `--vulkan` and `--metal` work as in the library. The wire protocol is the enum in [`protocol.rs`](ptts-ws-server/src/protocol.rs). Audio is Opus-encoded, so building needs `libopus` (`apt install libopus-dev` or `brew install opus`).

<!-- TODO: add a Dockerfile and a short example client, then link them here. -->

## Browser

The same crate compiled to WebAssembly, tokenizer included. [Try it](https://laurentmazare.github.io/pocket-tts) <!-- TODO(assets): org-hosted demo URL -->. The page downloads the weights once and caches them.

```bash
cd ptts-wasm && make build     # wasm-pack build --target web --release
python3 -m http.server -d pkg 8080
```

<!-- TODO(assets): screenshot or GIF of the browser demo. -->
Details in [`ptts-wasm/README.md`](ptts-wasm/README.md).

## Models, voices and languages

Phonon runs the [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) checkpoints published by Kyutai. The repo holds several:

| Path in the repo | What it is |
|---|---|
| `tts_b6369a24.safetensors` + `embeddings/` | the original English checkpoint, 8 voices |
| `languages/{english,french,german,italian,portuguese,spanish}/` | one checkpoint per language, **26 voices each** |
| `languages/*_24l/` | 24-layer versions of the above. Larger and higher quality. They need their own `config.json` |

Bundled voices in the original checkpoint: `alba`, `azelma`, `cosette`, `eponine`, `fantine`, `javert`, `jean`, `marius`.

<!-- TODO(assets): per-voice audio samples, and a sample per language. -->

**Getting the weights.** `kyutai/pocket-tts` is gated. Accept its terms on the model card, then run `huggingface-cli login` or export `HF_TOKEN`. [`kyutai/pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning) has the same layout and is not gated.

**Normalization** covers `en`, `fr`, `de`, `es` and `pt`. The model speaks Italian, but there is no Italian normalizer yet. Pass `--lang none` to give the text to the tokenizer as written.

**Your own checkpoint.** Anything with the layout above loads with `--dir`, `config=` or `Synth::builder`. <!-- TODO: Gradium's own Phonon checkpoint, if and when it is public: repo id, what it adds over the Kyutai ones, licence. -->

## Performance

<!-- TODO(assets): fill from `bench` runs. Report RTF and peak RSS, f32 and q8_0 at least, and say
     which threads and backends. Suggested rows: Apple M-series, x86-64 laptop, Raspberry Pi 5. -->

| Device | Weights | Threads | Real-time factor | Peak RSS |
|---|---|---|---|---|
| *Apple M-series* | f32 / q8_0 | | *TODO* | *TODO* |
| *x86-64 laptop* | f32 / q8_0 | | *TODO* | *TODO* |
| *Raspberry Pi 5* | q8_0 / q4k | | *TODO* | *TODO* |

Reproduce with the benchmark harness. It reports time to first audio, per-frame time and RTF over `--iters` runs, and leaves out the one-off model load:

```bash
cargo run --release --features hf,accelerate --example bench -- \
  --lang en --model model/model.q8.gguf --config model/config.json --quant q8 \
  --voice model/voices/alba.safetensors --threads 8 --iters 20
```

## How it compares

Facts as of September 2026, from each project's README and PyPI metadata. Corrections welcome.

| | **Phonon (`ptts`)** | [Kokoro](https://github.com/hexgrad/kokoro) | [NeuTTS](https://github.com/neuphonic/neutts) |
|---|---|---|---|
| Runtime | **Rust. No Python needed** | Python + PyTorch | Python + PyTorch |
| Voice cloning | **about 10 s of audio, no transcript** | no (fixed voice packs) | reference audio **plus its transcript** |
| Voices / languages | 26 per language, 6 languages | 54 voices, 9 languages | per model |
| Quantization | 10 GGML formats, first-party | community ONNX | GGUF via llama-cpp-python |
| Browser / server / CLI | **all first-party, one code base** | community ports | no |
| Watermarking | no | no | Perth, on by default |
| Code licence | **MIT OR Apache-2.0** | Apache-2.0 | Apache-2.0 (Air). Bespoke licence (Nano, 2E) |
| Model licence | CC-BY-4.0, gated | Apache-2.0 | per model |

Kokoro and NeuTTS are good. Where they lead, in voice and language breadth, model sizes and watermarking, the answer is a better model, and this runtime does not try to be that. What it offers is deployment: one implementation you can put in a wheel, a binary, a browser tab and a phone, from one source tree.

## Responsible use

The published weights are CC-BY-4.0 with an acceptable-use agreement. You accept it on the model card. In short: do not impersonate or clone anyone without their explicit, lawful consent. Do not make deceptive or fraudulent content. Do not pass generated audio off as a real recording. This runtime does not watermark its output. If you build cloning into a product, the consent flow is yours to design.

## Repository layout

| Crate | |
|---|---|
| [`ptts/`](ptts/) | the library. `synth` is the one-call API, `tts_model` the lower-level pieces. Examples: [`say`](ptts/examples/say.rs), [`pocket_tts`](ptts/examples/pocket_tts.rs), [`bench`](ptts/examples/bench.rs), [`create_voice`](ptts/examples/create_voice.rs), [`quantize`](ptts/examples/quantize.rs) |
| [`ptts-pyo3/`](ptts-pyo3/) | the Python package, built with maturin |
| [`ptts-ws-server/`](ptts-ws-server/) | the WebSocket server |
| [`ptts-wasm/`](ptts-wasm/) | the browser build and demo page |

## Building and contributing

```bash
cargo test --workspace                       # ptts-ws-server needs libopus
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test -p ptts --features hf,audio       # the optional features have their own tests
```

CI runs all of that on stable and nightly, on Linux, macOS and Windows, for every feature combination, for the docs as docs.rs builds them, and for the `wasm32` target. [`rust-ci.yml`](.github/workflows/rust-ci.yml) is the source of truth. Python wheels come from [`maturin-pub.yml`](.github/workflows/maturin-pub.yml). Working notes for the code base are in [`CLAUDE.md`](CLAUDE.md).

<!-- TODO: add CONTRIBUTING.md, issue templates and a CODE_OF_CONDUCT.md, then link them here. -->

## Licence

Two things, two licences:

- **Phonon**, meaning the runtime, the bindings, the server and the browser build, is dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
- **The model weights** are published by Kyutai under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) with an acceptable-use agreement. See the [model card](https://huggingface.co/kyutai/pocket-tts). Redistributing them requires attribution.

## Acknowledgements

Phonon exists because of [Pocket TTS](https://huggingface.co/kyutai/pocket-tts), the model by [Kyutai](https://kyutai.org). The weights are theirs. Their [reference implementation](https://github.com/kyutai-labs/pocket-tts) in PyTorch is the place to study or fine-tune the model. This repository is the runtime for that model, not a replacement for it. Phonon is built by [Gradium](https://gradium.ai) on the [`xn`](https://github.com/gradium-ai/xn) tensor library. It began as [Laurent Mazare](https://github.com/LaurentMazare)'s `xn-ptts`.

<!--
  ================ ASSET / TODO CHECKLIST (delete when done) ================
  [ ] Hero video or GIF (20 to 30 seconds), top of file
  [ ] Hosted browser demo under the org. Replace both laurentmazare.github.io links
  [ ] Audio samples: one per bundled voice (8), one per language (6). Hosted page or model card
  [ ] Browser demo screenshot or GIF
  [ ] Performance table: RTF and peak RSS from `bench`, 3 or more devices, f32 and q8_0
  [ ] Repo or org name, if the repository is renamed at launch (search "gradium-ai/xn-ptts")
  [ ] Confirm the crates.io and PyPI names and badges once 0.3.x is published
  [ ] "Your own checkpoint" section: Gradium's Phonon checkpoint, if it goes public
  [ ] Dockerfile and example WebSocket client. CONTRIBUTING.md, templates, code of conduct
  [ ] Social preview image (GitHub, Settings, Social preview)
  ============================================================================
-->
