# Build and develop Phonon

The root [README](../README.md) covers the consumer integrations. This guide is for building the current checkout, testing a checkpoint, and preparing release packages.

## Model setup

Set `MODEL_DIR` to the absolute path of a checkpoint folder with its own config, tokenizer JSON, weights, and voice assets. The examples use q8 GGUF weights. `--lang` selects how the runtime reads numbers and symbols: `en`, `fr`, `de`, `es`, `pt`, or `none` to keep text unchanged. Choose it for the language of your input; the checkpoint determines which spoken languages it supports. See [model setup](../README.md#model-setup).

Clone the repository and run the following commands from its root:

```sh
git clone https://github.com/gradium-ai/xn-ptts.git
cd xn-ptts
```

## Command line

Install [Rust](https://rustup.rs), then:

```sh
cargo install --path ptts --locked --features cli
ptts --model "$MODEL_DIR" --lang en --quant q8 "Hello from Phonon." -o speech.wav
```

For an edit/build/run loop:

```sh
cargo run --release -p ptts --features cli --bin ptts -- \
  --model "$MODEL_DIR" --lang en --quant q8 "Hello from Phonon." -o speech.wav
```

The short `say` library example is [`ptts/examples/say.rs`](../ptts/examples/say.rs). The full CLI is the `ptts` binary. See [CLI options](cli.md).

## Python

Install Rust and [uv](https://docs.astral.sh/uv/), then run from the repository root:

```sh
uv run --project ptts-pyo3 --locked python -m ptts \
  --model "$MODEL_DIR" --lang en --quant q8 "Hello from Phonon." -o speech.wav
```

For another project, use `uv add /path/to/xn-ptts/ptts-pyo3` or `python -m pip install /path/to/xn-ptts/ptts-pyo3`. These are source installs and compile Rust. To install a built wheel instead:

```sh
python -m pip install "/path/to/ptts-VERSION-PLATFORM.whl"
```

Replace the example path with the actual wheel filename. [Python guide](../ptts-pyo3/README.md).

## Browser

Requires Rust with the Wasm target, Node 22.7+, [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/), binaryen 124+ (`wasm-opt` on `PATH`), and [cargo-about](https://github.com/EmbarkStudios/cargo-about) for the package's third-party licence notices. The threaded build also uses a pinned nightly toolchain installed by the Makefile.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
# Install binaryen 124+ from your package manager or its GitHub releases.
cd ptts-wasm
make threads-toolchain
make build
(cd pkg && npm pack)
```

Install the resulting tarball in your web app, using its actual filename:

```sh
npm install "/path/to/phonon-tts-VERSION.tgz"
```

Use the tarball rather than a folder dependency so npm copies the files into the consumer project. Model files are served separately; the package contains no weights.

To run the demo with the default model, or with your local checkpoint:

```sh
make serve
make serve MODEL_DIR="$MODEL_DIR"
```

Open `http://localhost:8080`. The server adds the headers required for Wasm threads. `make profiling` builds the single-thread variant without wasm-opt. [Browser guide](../ptts-wasm/js/README.md).

## Swift

On a Mac with Xcode 16 and the required Rust targets, build the core and export your checkpoint:

```sh
./ios/build-xcframework.sh
cargo run --release -p ptts --example export_coreml --features hub -- phonon-coreml
```

Without `--model` it exports the default model; `--model "$MODEL_DIR"` exports a local checkpoint. Add `ios/PhononTTS` to Xcode as a local package, select the `ptts` product, and add the exported `phonon-coreml` folder as a folder reference named `Models`. The module remains `PhononTTS`. Export conditions must match the checkpoint. [Swift guide](../ios/PhononTTS/README.md).

For ordinary app integration, apps add the repository itself as a package: the root `Package.swift` at each release tag downloads that release's compiled core. Custom checkpoint exports remain available. Build against local changes with `ios/PhononTTS`, not the root package, which always fetches a released framework.

## Docker

From the repository root:

```sh
docker build -f ptts-openai-server/Dockerfile -t ptts-openai-server:local .
docker run --rm -p 127.0.0.1:8880:8880 -e PTTS_LANG=en -e PTTS_QUANT=q8 \
  ptts-openai-server:local
```

The image contains code, not weights: it downloads the default model on first start. Mounting a model folder, persistent cache volumes, formats, and admission limits are documented in the [server guide](../ptts-openai-server/README.md).

## Rust

Use a path dependency while working from this checkout:

```toml
[dependencies]
ptts = { path = "/absolute/path/to/xn-ptts/ptts", features = ["hf"] }
```

Run the consumer with `cargo run --release`. The `hf` feature enables the JSON tokenizer; `audio` adds audio-file decoding for voice creation. `hub` downloads `hf://` models; without it, the library reads local folders only. [Rust guide](../ptts/README.md).

## Checks

CI is defined in [the Rust workflow](../.github/workflows/rust-ci.yml). Start with the checks for the code you changed:

```sh
cargo fmt --all -- --check
cargo test -p ptts --features cli --all-targets
cargo clippy -p ptts --features cli --all-targets -- -D warnings
node --test ptts-wasm/js/test/*.test.mjs
```

Do not use `--all-features`: CUDA requires its own SDK. The servers require libopus, and the HTTP server also uses LAME; their guides list the system packages. Release workflows set explicit CPU baselines rather than the checkout's `target-cpu=native` flags.

Installed wheel tests run without weights by default. Set `PTTS_TEST_MODEL` to a local checkpoint folder or to `hf://OWNER/MODEL@COMMIT_SHA` to include the marked checkpoint tests. Keep private model files and credentials out of public artifacts and shared caches.

## Releasing the Swift package

Swift Package Manager and the [Swift Package Index](https://swiftpackageindex.com) read the root `Package.swift` from a release tag. That manifest must already name the release's framework URL and its exact checksum when the tag is pushed. Two builds of the same commit do not give the same archive, so the framework is built once, on `main`, and the tag reuses it:

1. Merge the version bump to `main`. The native packages workflow runs on that push and builds the framework. Wait for it to succeed.
2. With `gh` signed in, run `python3 scripts/package-swift.py pin` on an up-to-date `main`. It finds that run, downloads its framework and writes the root `Package.swift` with the new version's URL and checksum.
3. Commit only `Package.swift` and land it on `main`.
4. Tag that commit `v<version>` and push the tag.
5. The tag run does not rebuild the framework. It downloads the pinned one from the main run that built the same inputs, checks its checksum, and attaches it to the draft release with the rest.
6. Attach the Phonon 31m bundles to the draft release with `gh release upload v<version>`: `phonon-31m-en-coreml.zip`, made with `export_coreml`, and `phonon-31m-en-qnn.zip`, made with the tools in `qnn/export` on the feat/phonon-qnn branch.
7. Publish the release. The framework URL works only from then on, so publish promptly: the Swift Package Index builds new tags soon after they appear. If its build ran before the release was public, start it again from the package's page on the index.

If the tag is pushed before step 2, the tag run fails and says so. The root `Package.swift` is generated from `ios/PhononTTS/Package.swift`: after changing that file, run `python3 scripts/package-swift.py check --write`. CI fails while they disagree.

## Updating the Homebrew formula

The native release workflow generates `ptts.rb` from the four Mac and Linux archives it uploads. After publishing a release, copy that generated file into [`Formula/ptts.rb` in Gradium's tap](https://github.com/gradium-ai/homebrew-tap/blob/main/Formula/ptts.rb) and merge the change there. Check `ruby -c Formula/ptts.rb` and verify each SHA-256 against its published archive before merging. Tap users receive the new version through `brew update` and `brew upgrade`.

`cargo binstall ptts` downloads from the release, and a draft release's archives are not public. Publish the GitHub release before, or together with, `cargo publish -p ptts`.
