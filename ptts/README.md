# ptts

Phonon is Gradium's on-device text-to-speech runtime. The `ptts` crate provides its Rust API and an optional `ptts` command. The code is licensed under MIT OR Apache-2.0. The models have their own licenses: Phonon 31m is under Apache 2.0, and Phonon 40m and Phonon 40m multilingual are under CC BY-NC 4.0. Voices have their own terms.

Each checkpoint supplies its own `config.json`, `tokenizer.json`, weights, and voice assets. The command and `Checkpoint::locate` with `ptts::checkpoint::DEFAULT_MODEL` use the default model, Phonon 31m, downloaded from Hugging Face on first use. The library downloads only with the `hub` feature, when given an `hf://` source.

See the [CLI guide](https://github.com/gradium-ai/xn-ptts/blob/main/docs/cli.md) for desktop downloads and platform requirements.

## Command line

Installing the command requires `--features cli`; without it, this crate builds only the library.

Install the command from crates.io:

```sh
cargo install ptts --locked --features cli
ptts --lang en "Hello world" -o out.wav
```

Or install the prebuilt CPU command with `cargo binstall ptts`. It uses the desktop download for your platform, with the same requirements.

Without `--model` the command uses the default model. `--model /path/to/model` loads a local checkpoint, `--model hf://<owner>/<model>` another one from Hugging Face, such as `hf://gradium/phonon-40m-fr`, and `--model hf://<owner>/<model>@<commit>` pins all its files to that revision. The Phonon 40m models are gated: accept the terms on the model's Hugging Face page first, and approval is automatic. For gated and private models, set `HF_TOKEN` or log in with `hf auth login`. `ptts --help` lists all options.

`--lang` is required: `en`, `fr`, `de`, `es`, `pt`, or `none` to disable normalization. Choose it for the language of your input; the checkpoint determines which spoken languages it supports. Phonon 31m speaks English. Phonon 40m comes as one model per language, and Phonon 40m multilingual speaks English, French, German, Spanish and Portuguese. `--quant q8` keeps q8 GGUF weights quantized on CPU; omit it for f32 weights.

The Python package also installs a command named `ptts`, with the same `--model`. When both are installed, `PATH` order selects the command. Use `python -m ptts` with your Python environment's interpreter to invoke Python explicitly.

`examples/say.rs` is a short library example. From a checkout, run `cargo run --release -p ptts --example say --features hf,hub -- "Hello world"`.

## Rust API

Add the library to your project:

```sh
cargo add ptts --features hf
```

Enable `hf` for the Hugging Face JSON tokenizer. Add `hub` to load `hf://` models, including `DEFAULT_MODEL`, with `Checkpoint::locate`. The library does not need the `cli` feature.

```rust,no_run
use ptts::checkpoint::{Checkpoint, ResolveOptions};
use ptts::preprocess::Lang;
use ptts::synth::{DeviceKind, Quant};

fn main() -> ptts::Result<()> {
    let checkpoint = Checkpoint::resolve(
        "/path/to/model",
        ResolveOptions { quant: Quant::Q80, weights: None },
    )?;
    let mut tts = checkpoint.builder(Lang::En).device(DeviceKind::Cpu).build()?;
    checkpoint.register_voices(&mut tts);
    let pcm = tts.say("Hello world")?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate())
}
```

Reuse the loaded model for later requests. `tts.stream(text)?` yields bounded PCM chunks for streaming playback. Build consumers with `--release` for synthesis. See the [repository README](https://github.com/gradium-ai/xn-ptts#choose-an-integration) for other integrations.

## Voices and generation

`tts.voices()` lists registered voices, and `tts.languages()` the languages the checkpoint's config says it was trained for. When a request selects none, the builder's configured voice is used, then `default`, then the first registered voice by name. With no registered voice, synthesis is unconditioned.

Choose a registered voice per request with `SpeechOptions::voice`:

```rust,no_run
use ptts::synth::SpeechOptions;

// Reuse the model loaded above and choose one of tts.voices().
let options = SpeechOptions::default().voice("VOICE_NAME");
let pcm = tts.say_with("Hello world", &options)?;
```

The same options let you set `temperature`, `seed`, and `max_tokens_per_chunk`. Options left unset use the builder's settings.

Long text is grouped at sentence boundaries, usually aiming for 50 text tokens per chunk. A sentence may exceed that target. Sentences over 200 tokens are split at a sentence or clause mark when possible, otherwise at a word boundary. A single word over 200 tokens, such as a long hash or digit run, is cut between its characters. Each piece starts from the selected voice prompt.

## Repeated requests

For repeated requests in one voice, `tts.session_default(&options)?` keeps a primed session with a budget calculated from the selected voice and configured chunk target. Use `tts.session(&options, max_seq_len)?` to set the KV budget yourself. Longer sentences are split to fit that session's budget.

```rust,no_run
let session = tts.session_default(&options)?;
let pcm = session.say("Hello again")?;
```
