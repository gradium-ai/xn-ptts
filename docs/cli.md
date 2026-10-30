# Phonon command line

The `ptts` command generates a WAV from a Phonon checkpoint: the default model, or the one `--model` names. Models and voices are not part of the command and have their own licenses.

## Desktop downloads

Each [GitHub Release](https://github.com/gradium-ai/xn-ptts/releases) has these archives, with `SHA256SUMS` for verification.

| Platform | Archive target | Requirements |
|---|---|---|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | glibc 2.35 or later; x86-64-v3 CPU |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | glibc 2.35 or later; ARM64 CPU |
| Mac Apple silicon | `aarch64-apple-darwin` | macOS 15 or later |
| Mac Intel | `x86_64-apple-darwin` | macOS 15 or later; x86-64-v3 CPU |
| Windows x64 | `x86_64-pc-windows-msvc` | Windows 10 or later; x86-64-v3 CPU |

Mac downloads target macOS 15, the oldest Mac runner used to check them. Older macOS versions require a source build and their own verification.

x86-64-v3 includes AVX2, FMA, and F16C. Older x86 CPUs need a source build suited to their CPU. The Apple silicon download also includes Metal: with f32 weights, `--device auto` runs on the GPU and `--device cpu` keeps it on the CPU. Quantized weights such as `--quant q8` always run on the CPU. The other downloads run on the CPU only. For CUDA or Vulkan, build from source with that backend feature.

Extract the archive. On Linux or Mac, put the extracted `ptts` file in a directory on your `PATH`, for example `~/.local/bin`. On Windows, run `ptts.exe` in PowerShell, or add its folder to `PATH`. Rust and Python are not required for these downloads. Mac may prompt you to allow the downloaded executable in System Settings because it is not notarized.

For Linux or Mac, download `SHA256SUMS` alongside the archive and verify that archive's entry:

```sh
# Linux
sha256sum --ignore-missing --check SHA256SUMS
# Mac: use the checksum entry for the archive you downloaded
shasum -a 256 ptts-<version>-<target>.tar.gz
```

On Windows, compare `Get-FileHash .\ptts-<version>-x86_64-pc-windows-msvc.zip -Algorithm SHA256` with its entry in `SHA256SUMS`.

With [cargo-binstall](https://github.com/cargo-bins/cargo-binstall), `cargo binstall ptts` installs the archive for your platform. The requirements above still apply, since binstall cannot check them. It does not check `SHA256SUMS`. Where no archive exists for your target, `cargo binstall ptts --locked --features cli` builds from source instead; without `--features cli` that build finds no command to install.

## Homebrew

On Mac or Linux, install the prebuilt command from [Gradium's Homebrew tap](https://github.com/gradium-ai/homebrew-tap):

```sh
brew install gradium-ai/tap/ptts
```

Homebrew adds the tap automatically. The formula pins each release archive by SHA-256. It does not include a model: `ptts` downloads the default model on first use, or loads the one `--model` names.

After a new formula version is published, run `brew update` and `brew upgrade gradium-ai/tap/ptts`. The platform and CPU requirements above still apply.

## Generate speech

```sh
ptts --lang en "Hello world" -o out.wav
```

The first run downloads the default model, Phonon 31m, from Hugging Face and caches it. To use a local checkpoint instead:

```sh
ptts --model /path/to/model --lang en --quant q8 "Hello world" -o out.wav
```

The folder must contain `config.json`, `tokenizer.json`, weights, and any required voice assets. Use q8 for a q8 GGUF checkpoint, or omit `--quant` for f32 weights. `--lang` selects how the runtime reads numbers and symbols: `en`, `fr`, `de`, `es`, `pt`, `ar`, `hi`, `ko`, `pl`, `sv`, `fi`, `no`, `it`, `nl`, `ja`, `zh`, `da`, `tr`, or `none` to keep text unchanged. Numbers, amounts and addresses are read out in the first five only. Choose it for the language of your input; the checkpoint determines which spoken languages it supports. The default model speaks English. For French, German, Spanish or Portuguese, use a Phonon 40m model, [listed in the README](https://github.com/gradium-ai/xn-ptts#model-setup).

You can also use a Hugging Face repo, such as Phonon 40m in French:

```sh
ptts --model hf://gradium/phonon-40m-fr --lang fr --quant q8 "Bonjour tout le monde" -o out.wav
```

To pin all files to one revision, add it after `@`, as in `hf://gradium/phonon-40m-fr@<commit>`. Without one, the repo's current main revision is used. The Phonon 40m models are gated: accept the terms on the model's Hugging Face page first, and approval is automatic. Gated and private models require `HF_TOKEN` or a saved Hugging Face login (`hf auth login`). Cached files are reused. Without a commit after `@`, each run asks the Hub for the current revision, so a gated model needs the token every time; with a commit, cached files load without one. With `HF_HUB_OFFLINE=1`, files come only from the cache and the Hub is never asked; the repo cannot be listed then, so voices in its `voices/` or `embeddings/` folder are not found, while catalog voices already downloaded are. Local folders require no download.

Use `--voice <name>` for a bundled voice or one of a voice catalog, or `--voice /path/to/voice.safetensors` for a voice file. `--voice` also takes a short audio file to clone, but only with a checkpoint that has a speaker encoder, and the Phonon models on Hugging Face do not have one. To clone a voice for them, use `scripts/clone-voice.py`, described in the [README](https://github.com/gradium-ai/xn-ptts#voices). With no voice argument, the checkpoint's default voice is used. A name the checkpoint does not have is looked up in the voice catalog, by name in any case and with or without accents, or by its voice ID; only that voice is downloaded, on first use. The catalog is [`gradium/phonon-voices`](https://huggingface.co/gradium/phonon-voices) unless `--voice-catalog` names another: an `hf://owner/name[@revision]` repo, or a local `voices.jsonl` or folder holding one, with one `{"name", "filename"}` object per line. `ptts --list-voices` lists the catalog, with each voice's language. `ptts --help` lists the remaining options, including `--threads` for CPU tuning. `ptts --version` reports the runtime version.

To change the speaking rate, set the `duration_delta` condition: `--condition duration_delta=-0.3` speaks faster, `--condition duration_delta=0.3` slower, and 0 is the default. Small values work best; large ones make speech less reliable.

## Build from source

From the repository root:

```sh
cargo install --path ptts --locked --features cli
```

The `cli` feature includes tokenizer support and audio-file decoding. Optional backend features, such as `metal` and `cuda`, can be added to the installation command. `say` remains a small Rust library example. There is no longer a `ptts` example target; contributors can run the command with `cargo run --release -p ptts --features cli --bin ptts -- ...`.
