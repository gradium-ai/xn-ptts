# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

Cargo workspace (resolver "3", edition 2024) with seven members:

- `ptts/`: core TTS library, depending on the `xn` tensor/nn crate. The full CLI lives in `src/bin/ptts/main.rs`, behind the optional `cli` feature. Install it with `cargo install --path ptts --locked --features cli`. Examples under `ptts/examples/` include `say` (the shortest library call) and `bench`, which require `hf`; `create_voice`, which requires `audio`; and `quantize`, which requires neither. `export_coreml` exports a checkpoint for Apple. `dump_reference`, which requires `hf`, writes one unquantized CPU generation out step by step (prompt rows, noise, latents, EOS logits, audio), for checking another implementation of the model against ptts. The CLI and examples share `src/bin/ptts/model_helpers.rs`. `autoexamples = false` and `autobins = false` keep every target explicit in `Cargo.toml`.
- `ptts-pyo3/` — PyO3 bindings exposing `TTS` to Python. Built with maturin in a mixed layout: `python/ptts/` is the package (`__init__.py`, `__init__.pyi` stubs, `py.typed`, `__main__.py`) and the cdylib lands inside it as `ptts._ptts`, so a pure-Rust layout's lack of anywhere to put `py.typed` is not a problem. `tests/` is a pytest suite that needs no weights except where marked `checkpoint`; run it against a built wheel, not the source tree. Has its own `pyproject.toml` and `uv.lock`.
- `ptts-wasm/` — browser build via `wasm-bindgen` / `wasm-pack`, published to npm as `phonon-tts`. `src/lib.rs` is the raw frame-at-a-time `Model`; `js/` is the package's public API around it (`PhononTTS`, which runs the model in a worker, downloads and caches the files, and speaks by voice name), with its own `package.json`, `README.md` and node tests. `www/index.html` is a demo page built on the package.
- `ptts-ws-server/` — WebSocket streaming server (`axum` + `kaudio`). Needs a system libopus through `kaudio` → `libopus_sys`, which is why CI installs it on Linux and macOS and skips this crate on Windows.
- `ptts-openai-server/`: the OpenAI-compatible `POST /v1/audio/speech` (`src/api.rs`) with `GET /v1/models`, `/v1/audio/voices` and `/health`, and, with `--wyoming-addr`, the Wyoming protocol Home Assistant uses (`src/wyoming.rs`; the Home Assistant app that runs it lives in `gradium-ai/phonon-home-assistant`). Its model loading is a copy of `ptts-ws-server`'s, kept separate on purpose. Needs a system libopus like `ptts-ws-server`, and a system libmp3lame for MP3 (`src/mp3.rs`, linked dynamically since LAME is LGPL; `build.rs` finds it); CI skips it on Windows too. Every flag also reads a `PTTS_*` environment variable, which is how its Docker image is configured: `Dockerfile` (built from the repository root; `Dockerfile.dockerignore` keeps `.cargo/config.toml` and local weights out of the context) carries no weights: supply a mounted model folder or an `hf://` repo through `PTTS_MODEL`. `.github/workflows/docker.yml` publishes it to `ghcr.io/gradium-ai/ptts-openai-server` for amd64 and arm64, each built on a native runner. The amd64 image holds an x86-64-v3 and an x86-64-v2 build, and `select-cpu.sh` starts the one the CPU can run.
- `ptts-coreml/` — CoreML backend, Apple only: the flow LM and Mimi emitted from Rust as ML Program graphs (`mil.rs`, `package.rs`, `blob.rs`, `phonon/flow_lm.rs`, `phonon/mimi.rs`), exported once per checkpoint by `ptts/examples/export_coreml.rs` (sizes and architecture from the checkpoint's config; an LSD flow head must take a single step), and driven by `phonon/driver.rs`. The flow LM runs on the Neural Engine, which needs fully static shapes, no CoreML `state` and fp16; its KV cache is a host-managed ring in IOSurface buffers. Mimi stays f32 on the CPU, decoded on a worker thread overlapped with the next flow step. The part of the Core ML protobuf schema it writes is hand-written as `prost` messages in `src/proto.rs`, so there is no codegen or `protoc` in the build.
- `ptts-ffi/` — the C interface (`include/ptts.h`), one engine per platform behind it: on Apple, `src/coreml.rs` over `ptts-coreml`, loading a bundle `export_coreml` wrote; everywhere else, `src/cpu.rs` over `ptts::synth::Synth` on the CPU, loading a checkpoint folder. Both use `ptts` for text preparation, normalization and the tokenizer. Its `README.md` is the C guide, and `examples/speak.c` the C example CI compiles. On Apple it is what `ios/PhononTTS/`, the Swift package apps integrate (its README is the user guide), wraps, as `PhononCore.xcframework` built by `ios/build-xcframework.sh`. The root `Package.swift` is that package for apps adding the repository by URL (and the Swift Package Index, configured by `.spi.yml`): generated from `ios/PhononTTS/Package.swift` by `scripts/package-swift.py`, it pins the last release's framework by URL and checksum, so it is set by `package-swift.py pin` before each release tag (`docs/development.md`) and is never built against local changes. On Android, `android/build.sh` builds it as `libptts_ffi.so` with cargo-ndk, and apps call it from `android/PhononTTS.kt` through JNA; `android/README.md` is that guide. `android/tts-engine/` is a small Gradle app over the same two that makes Phonon an Android system text-to-speech engine (`TextToSpeechService`); it compiles `PhononTTS.kt` from `android/` rather than copying it, and CI does not build it. Neither platform has an app for measuring: that needs a local app.

Outside the workspace, `qnn/` runs Phonon on Qualcomm NPUs through QNN: `runner/` is the C++ engine and a `phonon` CLI, `text-ffi/` is a crate with its own `[workspace]` and `Cargo.lock` that gives the runner ptts's text preparation over C, and `android/` is the `ai.gradium:ptts` Android library over the runner. Building the engine needs Qualcomm's QAIRT SDK; `.github/workflows/android-qnn.yml` checks the parts that build without it, outside the `CI` gate.

Also outside the workspace, `ptts-litert/` is a Python package (its own `pyproject.toml` and `uv.lock`) that exports a checkpoint to LiteRT for Android NPUs, compiles it per SoC, and checks it against `dump_reference` output; its README covers the bundle and the graphs. It is a PyTorch reimplementation of the model, because LiteRT's converter and the vendors' NPU compilers take PyTorch or `.tflite` input; `check` is what keeps it equal to `ptts`. There is no on-device runtime for it yet.

Shared dependency versions (notably `xn`) and the workspace version live in the top-level `Cargo.toml`. Bumping the release version means editing `workspace.package.version` and the `ptts` workspace dep, then refreshing `Cargo.lock` and `qnn/text-ffi/Cargo.lock` (CI builds the latter with `--locked`).

## Build / test / lint

CI (`.github/workflows/rust-ci.yml`) is the source of truth. Twelve jobs, gated behind one
required check called `CI`:

| Job | What it covers |
|---|---|
| `fmt` | `cargo fmt --all -- --check` (rustfmt.toml: `use_small_heuristics = "Max"`, edition 2024) |
| `clippy` | whole workspace `--all-targets -D warnings`, then `ptts` with `cli`, then `ptts-ffi` for `aarch64-linux-android` |
| `test` | stable + nightly × Linux/macOS/Windows; default features, then `cli`, then doctests; `metal`, `accelerate` and `kai` type-checked on the macOS leg; the C example built and run against `ptts-ffi` on stable Linux and macOS |
| `features` | every combination of `hf`/`audio`, plus `hub`, `cli`, `vulkan` and `webgpu` |
| `docs` | `cargo doc` on nightly with `--cfg docsrs` exactly as docs.rs builds it, then again on stable |
| `wasm` | `ptts-wasm` for `wasm32-unknown-unknown` with the SIMD flags real builds use, with and without `webgpu`; the `phonon-tts` JS wrapper's node tests; and `make build` with binaryen 124 |
| `coreml` | macOS only: clippy on `ptts-coreml` and `ptts-ffi` for macOS and iOS, then `ios/build-xcframework.sh` and `swift build` of the `PhononTTS` package, and the root `Package.swift` checked against it |
| `changes` | whether the three speech jobs run: only when the change touches `ptts*/`, `Cargo.toml`, `Cargo.lock`, `rust-ci.yml`, `setup-rust` or `scripts/check-speech.py`, and on every manual run. `CI` accepts them skipped only then |
| `speech` | the default model, downloaded from the Hub, speaking `SPEECH_TEXT` on Linux x64/arm64, macOS and Windows: the release-flag command on the CPU with f32, q8 and q4k weights, Vulkan (Mesa's lavapipe) and Metal, and `speak.c` through `ptts-ffi`'s CPU engine on Linux and Core ML on macOS |
| `speech-browser` | the package `wasm` built, in headless Chrome (`ptts-wasm/scripts/speak-in-chrome.mjs`): threaded, single-threaded and WebGPU on SwiftShader |
| `speech-check` | `scripts/check-speech.py`: Whisper transcribes every WAV the two jobs above wrote, which must read back as `SPEECH_TEXT` |

Other workflows also run on pull requests: `maturin-pub.yml` (the wheels), `cli-release.yml` (CLI
archives, the Swift package and the Homebrew formula), `npm-publish.yml`, `docker.yml`,
`android-qnn.yml` and `diff-aware-pr-reviewer.yml`. Only `rust-ci.yml` feeds the `CI` check.

`.github/actions/setup-rust` is a composite action holding the parts every job shares: the
toolchain, the cache, and the platform quirks below.

Three things worth knowing before editing it:

- **`--all-features` never works.** It turns on `cuda`, whose `cudarc` build script shells out to
  `nvcc`. Feature sets are always named explicitly, including in `[package.metadata.docs.rs]`.
- **The two servers need a system libopus, and `ptts-openai-server` also libmp3lame** (libopus
  through `kaudio` → `libopus_sys`; LAME for MP3, `libmp3lame-dev` / `brew install lame`). CI
  installs both on Linux and macOS; Windows has no one-line equivalent, so both crates are
  excluded there and only there, through `$WS_EXCLUDE`. The `vulkan` feature likewise needs `glslc`, installed in the
  `features` job.
- **CI deletes `.cargo/config.toml`** because it pins `target-cpu=native`, which breaks portable
  dependency builds. If you reproduce a CI failure locally, do the same (`rm -f
  .cargo/config.toml`) — otherwise keep the file in place for fast local builds. The `wasm` job
  re-sets that file's SIMD flags itself.

Cargo features that gate optional functionality:

- `ptts`: `hf` enables `ptts::tok` for the JSON tokenizer; `audio` enables audio-file decoding and resampling for voice cloning. Both are optional for the library. `hub` enables downloading `hf://` sources in `Checkpoint::locate` (`ptts/src/hub.rs`, the blocking `hf-hub` client); without it the library never downloads. `cli` enables all three plus argument parsing, tracing, and Unix memory reporting for the command. The two servers and `ptts-pyo3` turn on `hub` too. Backend features include `cuda`, `accelerate`, and `kai` (see below).
- `ptts-pyo3`: `cuda`, `vulkan`, `metal`, `webgpu`, `accelerate`, `kai` (each forwards to both `xn/*` and `ptts/*`).

`kai` runs the `q8_0` transformer linears through Arm KleidiAI's SME2 kernels, which `xn`
vendors and compiles itself, so it needs no setup. It only does anything on a CPU with SME2
(Apple M4 and later, Arm Cortex-X925 and later); elsewhere the weights keep xn's own layouts.
It makes prompt prefill and voice conditioning faster, not decode, and it requantizes the
weights to one scale per row, which moves the output slightly. So with `kai` on, the same
binary gives slightly different audio on an SME2 CPU than on any other: anything that
compares outputs should set `XN_KAI=0`, which turns it off at run time, or allow a tolerance.

Run the CLI:

```
cargo run --release -p ptts --bin ptts --features cli -- --lang en "hello world" -o out.wav
```

A model is named by one string, a local folder (or `config.json`) or `hf://OWNER/MODEL[@REVISION]`: `--model` for the CLI, the examples and both servers (`PTTS_MODEL` for the OpenAI one), `model=` for Python, `model` for `PhononTTS.load`. Left out, every one of them loads `ptts::checkpoint::DEFAULT_MODEL`, a public Hub repo pinned to a commit (Phonon 31m, `gradium/phonon-31m-en`). The browser package keeps a copy in `ptts-wasm/js/hub.js` that its tests hold to the Rust one, so switching the default means editing both. `bench` and `ptts-ffi` (iOS, Android) still take explicit local files. Only the `hf://` prefix marks a Hub repo; a bare `owner/name` is a local path, and its not-found error suggests the prefix. The OpenAI server's `--config`/`PTTS_CONFIG`/`PTTS_REVISION` and Python's `config=`/`revision=` are deprecated aliases that still take a bare repo ID (`checkpoint::legacy_source`); the CLI keeps `--dir` and the WebSocket server `--config` as aliases of `--model`. Each checkpoint supplies `config.json`, `tokenizer.json`, weights, and optional voice assets. There is no built-in model config: the default is an ordinary checkpoint repo. Resolution, local or Hub, lives in `ptts/src/checkpoint.rs` and `ptts/src/hub.rs`; the servers call it through `spawn_blocking`, since the blocking Hub client runs its own runtime. `--weights` selects a weights filename inside the supplied folder or HF repo, for example `--weights model.q8.gguf --quant q8`. `--voice` accepts a checkpoint voice name, a voice of the voice catalog, a voice safetensors file, or a short audio sample. The catalog (`ptts/src/voice_catalog.rs`) is a `voices.jsonl` in an `hf://` repo or local folder; `--voice-catalog` and Python's `voice_catalog=` name it, and both default to `voice_catalog::DEFAULT_VOICE_CATALOG`, `gradium/phonon-voices` pinned to a commit. Both servers take `--voice-catalog` too (`PTTS_VOICE_CATALOG` for the OpenAI one), with the same default, but download and register the catalog's voices in their `--lang` at startup, since a shared `Synth` cannot add voices later; an empty value turns the catalog off. `VoiceCatalog::register` looks a name the checkpoint lacks up there, in any case and with or without accents, or by `voice_id`, and downloads only that voice; `--list-voices` needs only the JSONL, and a catalog. The CLI and Python use it; the servers list the catalog at startup through `VoiceCatalog` too, and fetch every voice in their language. `--tokenizer` supplies a tokenizer JSON from another path. `--device auto|cpu|cuda|vulkan|metal` selects the backend. `--lang en|fr|de|es|pt|none` is required.

The Python package also installs a `ptts` command, with the same `--model`. If both commands are installed, `PATH` order selects which one runs. Use `python -m ptts` with the Python environment's interpreter to select its CLI explicitly.

`say` is the short library example, for checking that the library works:

```
cargo run --release --example say --features hf,hub -- "hello world"
cargo run --release --example say --features hf -- --model "$MODEL_DIR" "hello world"
```

Benchmark a local model:

```
cargo run --release --features hf,accelerate --example bench -- \
  --model model/model.q8.gguf --config model/config.json --quant q8 \
  --voice voices/freya.safetensors --threads 8 --iters 20
```

`bench` takes explicit paths and a precomputed voice embedding, never downloads, and reports time-to-first-audio, per-frame time, total generate time and RTF (generate time over audio duration, lower is better) over `--iters` runs, excluding the one-off model load and voice conditioning. Each run is a `Synth::stream` call, so these are the numbers every frontend gets: the flow LM and Mimi on two threads, Mimi decoding whatever frames have queued in one call. `--breakdown` instead runs both on the calling thread, one frame at a time, and splits each frame into flow LM sampling and Mimi decoding; use it to attribute time to a stage (kernel work), not to quote end-to-end numbers, since its RTF reads about 20% higher and its time to first audio lower than the real path. On a GPU the two modes also differ in precision (`Synth` runs bf16 on CUDA and Metal, `--breakdown` the f32 of the weights), so compare them only on the CPU. `--threads` defaults to xn's one-per-logical-core, usually too many for a single autoregressive stream. For profiling rather than measuring, `ptts --chrome-tracing` writes a Chrome trace for https://ui.perfetto.dev. For a sampling profiler (samply, Instruments, perf), build with `--profile profiling` instead of `--release`: the release profile carries no debug info, and `profiling` is release plus symbols. Binaries then land in `target/profiling/`. The same goes for a crash that only happens in release: a `--release` backtrace has function names but no file and line numbers.

## WASM build

From `ptts-wasm/`:

```
make build        # the phonon-tts npm package in pkg/: wasm-pack output in pkg/wasm/ and pkg/wasm-threads/, plus js/
make profiling    # same but --profiling (no wasm-opt)
make demo         # pkg/ copied to site/phonon-tts/ and www/index.html; MODEL_DIR=/path also links that folder as site/model/
make serve        # make demo, then serve site/ on :8080, cross-origin isolated; takes MODEL_DIR too
make test         # node --test js/test/*.test.mjs -- the wrapper's logic, no browser or model needed
```

Requires `wasm-pack` 0.12 or later (`cargo install wasm-pack`), node 22.7 or later, and binaryen's `wasm-opt` 124 or later on `PATH`: wasm-pack otherwise downloads binaryen 117, and releases up to 123 abort on this module. It also needs `cargo-about` (`cargo install cargo-about --locked --features cli`), which writes the crates' licence notices into `pkg/THIRD_PARTY_LICENSES.md` from `about.toml` and `about.hbs`; a crate under a licence `about.toml` does not accept fails the build. The threaded build (`pkg/wasm-threads/`, the `threads` feature) also needs the nightly pinned in the Makefile with `rust-src`, since wasm threads need std rebuilt with atomics: `make threads-toolchain` installs it. `js/worker.js` loads that build only on a cross-origin isolated page, and the single-threaded one otherwise; `js/threads.js` picks the thread count, and `make serve` serves the demo with the isolation headers (`scripts/serve.mjs`). Both builds have the `webgpu` feature: `src/lib.rs` has one engine generic over the device, with only the readback differing, and `js/device.js` keeps the CPU as the default: `device: 'webgpu'` opts into WebGPU, and `'auto'` takes it only when the browser offers a hardware adapter and the weights are q8 GGUF. WebGPU is opt in because it is not faster than the CPU on every device, phones in particular. `scripts/pack.mjs` assembles the package and stamps its version from `workspace.package.version`, so `js/package.json` deliberately has no `version`; it also derives what to copy from that file's `files` list. It deletes the `.gitignore` wasm-pack writes into `pkg/wasm/`: npm reads a subdirectory `.gitignore` as that directory's `.npmignore`, which would silently publish a package without its wasm. `make demo` and `make serve` take an optional `MODEL_DIR`, a model folder, and without it the page loads the package's default model: it is linked into `site/model/`, and `scripts/demo-model.mjs` writes `site/model.json` describing its weights and voices, since a static server cannot list a directory for the page. Wasm SIMD flags (`+simd128,+relaxed-simd`) and `getrandom_backend="wasm_js"` come from `.cargo/config.toml`. `make build` also builds both modules without `relaxed-simd`, for browsers that lack it (Safari, and so every browser on iOS); `scripts/pack.mjs` checks wasm-bindgen generated the same JS for both and keeps only that `.wasm`, as `phonon_tts_simd_bg.wasm` beside the Relaxed one, and `js/worker.js` picks between them with the probe in `js/simd.js`.

The browser takes a model as a `ModelSpec` with config, tokenizer, weights, and voice URLs, or a public repo as `hf://owner/name[@revision]` (`DEFAULT_MODEL` when left out), which `js/hub.js` turns into a `ModelSpec` with one Hub API request, every URL pinned to the resolved commit and the files picked by the native rules. Use revision-pinned HF URLs or versioned local paths in a `ModelSpec` because files are cached by URL. `.github/workflows/npm-publish.yml` builds the package on PRs that touch it and publishes it on a `v*` tag through npm trusted publishing (OIDC, no token).

## Python build

From the repo root:

```
maturin develop --manifest-path ptts-pyo3/Cargo.toml          # local install
maturin build --release --manifest-path ptts-pyo3/Cargo.toml  # produce wheel
cd ptts-pyo3 && python -m pytest -m 'not checkpoint'          # against an installed wheel
ptts --lang en "hello world" -o out.wav  # the console script, with the default model
python -m ptts --model "$MODEL_DIR" --lang en "hello world" -o out.wav  # the same `main`, a local model
```

Run the tests from `ptts-pyo3/`, so pytest reads the `testpaths` and `markers` in its
pyproject.toml, and so `import ptts` finds the installed wheel rather than `python/`. Each
wheel job in CI does the same, through `.github/actions/test-wheel`; musllinux is skipped
because a musl wheel will not install on the glibc runner.

The package is a mixed maturin layout: `python/ptts/` is the package and the cdylib lands in
it as `ptts._ptts`. Renaming or adding anything on the Python surface means editing
`python/ptts/__init__.py`, `python/ptts/__init__.pyi` and the `#[pymodule]` list together;
`test_the_stubs_cover_everything_the_extension_exports` catches the stub half of that and
`test_all_covers_everything_the_extension_exports` the re-export half, by diffing
`dir(ptts._ptts)` against `__all__`. `[project.scripts]` installs the `ptts` command, which is
what `uvx ptts` and `pipx run ptts` run; `python/ptts/__main__.py` is the whole of it. `--lang`
is required there as it is on the `ptts` command and `ptts-ws-server`, but checked by hand rather
than by argparse, so that `--build-info` still works without one.

`pyo3` is built with `abi3-py39`, so one wheel per platform serves every CPython from 3.9 on
and a new CPython release needs no rebuild. abi3 does not load on free-threaded CPython and
PyPy needs its own ABI; both fall back to the sdist, which `sdist-fallback` compiles and tests
on `3.14t` and `pypy3.11`. It is deliberately outside `release`'s `needs`: those users compile
either way, so blocking everyone else's wheels would not help them. Wheels are built `--strip`:
on Linux debug info lands inside the `.so`, which is what made the published 0.2.2 Linux wheels
41 MB against 3.9 MB for macOS and Windows. `[profile.release]` no longer carries debug info,
so `--strip` is now a guard rather than a fix.
`pyproject.toml` deliberately has no `features` key under `[tool.maturin]`: a `--features` on
the maturin command line replaces that list rather than adding to it, so `pyo3/extension-module`
lives in `ptts-pyo3/Cargo.toml` where the macOS jobs' `--features` (accelerate, plus kai and
metal on Apple silicon; the CLI binaries get the same) cannot drop it.

Release wheels are produced by `.github/workflows/maturin-pub.yml`: manylinux and musllinux
on x86_64 and aarch64, Windows on x64 and aarch64, macOS on aarch64 and x86_64, and an sdist.
PyPI accepts the upload through a trusted publisher pinned to the repository *and to that
file's name*, so renaming the file breaks releasing until PyPI is updated; a `v*` tag is what
triggers it. It started as `maturin generate-ci github -m ptts-pyo3/Cargo.toml` output and
has diverged; the header comment lists what a regeneration would undo. Chief among them:
every job deletes `.cargo/config.toml` and sets `RUSTFLAGS` itself, because `target-cpu=native`
in a published wheel means whatever CPU the runner had. Published Linux and Windows x86_64
wheels target `x86-64-v3` — `xn` selects its quantized kernels with `cfg!(target_feature = "avx")`
at compile time, so a lower baseline silently costs every `q8_0` path its AVX kernels. The macOS
x86_64 wheel is the exception: it keeps `.cargo/config.toml`'s `-avx,-avx2`.

## Architecture

The library implements Phonon: text → tokens → flow-matching language model produces Mimi codec latents → Mimi decoder produces 24 kHz PCM audio.

`ptts/src/lib.rs` exposes a single `Tokenizer` trait (`encode` / `decode`) so each binding plugs in its own implementation:

- the `ptts` command, `say` / `bench` examples, `ptts-pyo3` and the two servers: `ptts::tok::Tok` (the `hf` feature), a Hugging Face `tokenizers` wrapper. The examples find the file beside the weights and pass it to `SynthBuilder::tokenizer_file`.
- `ptts-wasm`: the same `ptts::tok::Tok`, built from the `tokenizer.json` the `phonon-tts` worker fetches and handed to `Model::new`; the browser passes text, not token ids.

Every frontend loads a `tokenizer.json` and nothing else, and none is bundled or defaulted to: each checkpoint has its own vocabulary, and loading the wrong one yields plausible audio from the wrong ids, so `Tok::open` refuses to guess. `ptts --tokenizer <path>` and `bench --tokenizer <path>` override where the examples look; otherwise they, `ptts-pyo3` and the two servers all expect `tokenizer.json` in the HF repo or beside the config.

Top-level orchestrator is `tts_model::TTSModel<Q>`, generic over a backend-quantization parameter `Q: BackendQ` from `xn`. It owns:

- `flow_lm: FlowLM<Q>` — token-conditioned flow-matching transformer that emits Mimi latents (`flow_lm.rs`, `transformer.rs`, `rope.rs`, `mlp.rs`, `layer_scale.rs`, `conditioners.rs`).
- `mimi: MimiDecoder<Unquantized<f32, Q::B>>` — neural audio codec decoder (`mimi.rs`, `seanet.rs`, `conv.rs`, `resample.rs`, `dummy_quantizer.rs`). The encoder side (`MimiEncoder` / `MimiEnc`) is used only for voice-prompt embedding from a 10s audio sample.

`synth::Synth` sits on top of all of it: `synth::SynthBuilder::new(config, weights)` loads a
checkpoint whose files the caller has already located and registers voices,
`plan` supplies the frame/KV budgets and the EOS policy, and `Synth::say` / `Synth::stream`
run the flow LM and the Mimi decoder on two threads. `Synth` erases the `Q` parameter behind a trait object (`Box<dyn SynthApi>`, which it
dereferences to) so a CLI flag can pick the weight format; the generic `SynthOf<Q>` is
private. `ptts-wasm` still drives `TTSModel` directly.

Generation is streaming and stateful: callers `init_flow_lm_state(batch, seq_len)`, then `prompt_text*` / `prompt_audio` to seed the state, then step-decode latents and feed them into `MimiDecoderState`. `lsd_decode_steps` controls flow-matching solver steps; `eos_threshold` controls termination. Every frontend reads the supplied checkpoint's own model config.

A config can list `conditioners` (`lut` or `continuous`), which are summed into one vector added to every generated frame's input. Their values are given by name: `--condition NAME=VALUE` on the CLI and examples (`bench`, `export_coreml`) and both servers (`PTTS_CONDITION` for the OpenAI one), `conditions=` on `ptts-pyo3`, `conditions` on `PhononTTS.load`. Those not given take their defaults (`num_speakers` 1, `padding_bonus` and `duration_delta` 0); one with no default is an error. A config's baked-in `voices` each carry their own values, which win over those given. `loader::load_conditions` computes the vector for frontends that keep it themselves: the Core ML export fixes it in the bundle, in `host.safetensors` and, for a baked-in voice, in its voice file. The browser build refuses checkpoints with baked-in voices.

Text normalization (`ptts/src/preprocess.rs`) is mandatory to choose and has no default. `preprocess::Normalize` is either `For(lang)` or `Off`, and it is a required third argument to `SynthBuilder::new`, a required `--lang` flag on the `ptts` command, `bench` example, and both servers, a required keyword-only `lang=` on `ptts-pyo3`, and a required `lang` argument to the `ptts-wasm` `Model` constructor and to `PhononTTS.load` in `phonon-tts`. The reason it is not defaulted rather than defaulted to English: normalization makes the model noticeably better, but the spoken forms of `@`, `+` and `=` are per-language, so normalizing German as English says "at" where it should say "ät" -- guessing is worse than doing nothing. `Normalize::Off` (`--lang none`, `lang="none"`) hands text to the tokenizer as written. A language the checkpoint's config does not list in `languages` is warned about (`Normalize::language_warning`: a `tracing` warning from `SynthBuilder::build`, a `UserWarning` in Python, `console.warn` in the browser), not refused.

`Normalize::apply` is the one implementation, and `plan::chunks` runs it before `prepare_text_prompt`. `Synth::normalization` / `Session::normalization` hand it to callers that need the normalized text themselves (the ws-server's empty-request check); `ptts-wasm`, which has no `Synth`, builds its own `Normalize` and calls `plan::chunks`.

After the character pass, each word goes through the rewrite rules (`ptts/src/preprocess/rewrite.rs`, one module per rule under `rewrite/`): `numbers`, `currency`, `dashed-digits`, `emails`, `urls`, `abbreviations` and `elongations` (both English only), and the opt-in `phones`, `times` and `dates`. The first rule in the `RULES` table that claims a word wins. Before that, `rewrite_text` lets a few readings that need their neighbours claim several words: a currency amount followed by a scale word ("$2 million"), runs of digit groups ("+44 20 7946 0958"), a code after "ending in", and "Lt Col". A row's `default` flag decides whether `Rules::DEFAULT`, which `Normalize::for_lang` and every frontend's `--rewrites default` use, runs it. The readings match the serving stack's, quirks included (`$1` reads "1 dollars"), apart from a few fixes noted where they are made: the top-level domain table, capitalized domains, numbers tried before times, a spoken `+` in phone numbers and "-$500", and those from the hard-sentence evaluation: scaled amounts read "2 million dollars" rather than "2 dollars million", digit groups and leading-zero numbers read digit by digit, two-digit and year ranges read "20 to 25" and "1445 to 1450" (German ranges from one digit too, "8 bis 10"), French thousands grouped with a space ("40 000") are joined into one number unless the groups run on into a code, and Spanish "1.er" and "3.er" read "primer" and "tercer". Change a reading only on purpose. The character pass keeps `@` and `+` in the text because the email and phone rules read them, and spells them out afterwards in the words no rule claimed; a slash between two words there ("and/or") becomes a space, except around units ("mg/kg") and single letters ("c/o", "km/h"). It also folds the accented letters the checkpoints' tokenizer has no piece for to their base letter (`Lang::fold`), since the tokenizer would hand them to the model as raw bytes, which it reads as noise; subscripts become digits, superscripts powers, and Greek letters their names. `Lang::fold` matches every language explicitly, so adding one means checking the table against its letters and tokenizer.

Quantization story: only `flow_lm.transformer.layers.*.{linear1,linear2,self_attn.in_proj,self_attn.out_proj}.weight` get GGML-quantized (see `examples/quantize.rs`); Mimi stays in `Unquantized<f32>`. The Mimi quantizer codebook tensors (`mimi.quantizer.*` except `output_proj`) are excluded from output GGUFs since the runtime uses `dummy_quantizer.rs`.

Prefix sinks: `prefix_sinks` in `config.json` lists the `speaker_wavs` and
`transcript_in_segment` conditionings that get a learnt token in front of their frames, or a
`sink_dropped` one when they are empty (`dropped_token`). A checkpoint with sinks must set
`cfg_null_audio_empty: true`.

Smaller Pocket TTS checkpoints use optional `flow_lm` keys: `flow_type` (`lsd`, the default;
`drifting`, whose one flow-net evaluation is the latent; `pointwise`, noise plus that
evaluation; the last two take no time conditions), `z_dim` (a `z_proj` in front of the flow net
and the EOS head), and `attn_dim` / `num_kv_heads` (attention width and grouped-query k/v heads,
repeated to `num_heads` in the KV cache). A low-rank text table is `lut_dim` wide with a
`conditioner.output_proj`. Their small Mimi decoder is `mimi`, and the released encoder that
clones voices is `speaker_mimi` with `prefix: "mimi"` and `inner_dim`, the width of its latents.
The Core ML export supports them too: its graph repeats the grouped k/v heads, as ptts's cache
holds them, and it writes the low-rank text table already projected.

## Conventions to be aware of

- `target-cpu=native` is on by default for host builds and `apple-m1` on the macOS CI release lane; do not assume binaries are portable.
- macOS x86_64 disables AVX/AVX2 (`.cargo/config.toml`), keep that in mind when benchmarking.
- The single workspace version (`workspace.package.version`) is shared by every workspace crate and the `ptts` workspace dep — update them together.
