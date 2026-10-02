# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

Cargo workspace (resolver "3", edition 2024) with six members:

- `ptts/` — core TTS library. Pure Rust, depends on the `xn` tensor/nn crate. Examples live under `ptts/examples/`: `say` (shortest end-to-end call) and `bench` (benchmark harness) require the `hf` feature for the tokenizer, `ptts` (full CLI) requires `hf` and `audio`, `create_voice` (voice embeddings from audio samples) requires `audio`, and `quantize` (safetensors → GGUF converter that selectively quantizes `flow_lm.transformer.layers.*` weights) requires nothing. `model_helpers.rs` is not an example — it is a shared module each example pulls in with `#[path = "..."] mod`, so `autoexamples = false` and every example is listed explicitly in `Cargo.toml`.
- `ptts-pyo3/` — PyO3 bindings exposing `TTS` to Python. Built with maturin in a mixed layout: `python/ptts/` is the package (`__init__.py`, `__init__.pyi` stubs, `py.typed`, `__main__.py`) and the cdylib lands inside it as `ptts._ptts`, so a pure-Rust layout's lack of anywhere to put `py.typed` is not a problem. `tests/` is a pytest suite that needs no weights except where marked `checkpoint`; run it against a built wheel, not the source tree. Has its own `pyproject.toml` and `uv.lock`.
- `ptts-wasm/` — browser build via `wasm-bindgen` / `wasm-pack`, published to npm as `phonon-tts`. `src/lib.rs` is the raw frame-at-a-time `Model`; `js/` is the package's public API around it (`PhononTTS`, which runs the model in a worker, downloads and caches the files, and speaks by voice name), with its own `package.json`, `README.md` and node tests. `www/index.html` is a demo page built on the package.
- `ptts-ws-server/` — WebSocket streaming server (`axum` + `kaudio`). Needs a system libopus through `kaudio` → `libopus_sys`, which is why CI installs it on Linux and macOS and skips this crate on Windows.
- `ptts-coreml/` — CoreML backend, Apple only: the flow LM and Mimi emitted from Rust as ML Program graphs (`mil.rs`, `package.rs`, `blob.rs`, `phonon/flow_lm.rs`, `phonon/mimi.rs`), exported once per checkpoint by `ptts/examples/export_coreml.rs` (sizes from the checkpoint's config, so any single-flow-step Phonon checkpoint works), and driven by `phonon/driver.rs`. The flow LM runs on the Neural Engine, which needs fully static shapes, no CoreML `state` and fp16; its KV cache is a host-managed ring in IOSurface buffers. Mimi stays f32 on the CPU, decoded on a worker thread overlapped with the next flow step. The part of the Core ML protobuf schema it writes is hand-written as `prost` messages in `src/proto.rs`, so there is no codegen or `protoc` in the build.
- `ptts-coreml-ffi/` — C interface over `ptts-coreml` (empty on non-Apple targets), using `ptts` for text preparation, normalization and the tokenizer. It is what `ios/PhononTTS/`, the Swift package apps integrate (its README is the user guide), wraps, as `PhononCore.xcframework` built by `ios/build-xcframework.sh`. There is no app in the repo: measuring on a device needs a local app on the package.

Shared dependency versions (notably `xn`) and the workspace version live in the top-level `Cargo.toml`. Bumping the release version means editing `workspace.package.version` and the `ptts` workspace dep.

## Build / test / lint

CI (`.github/workflows/rust-ci.yml`) is the source of truth. Eight jobs, gated behind one
required check called `CI`:

| Job | What it covers |
|---|---|
| `fmt` | `cargo fmt --all -- --check` (rustfmt.toml: `use_small_heuristics = "Max"`, edition 2024) |
| `clippy` | whole workspace `--all-targets -D warnings`, then `ptts` with `hf,audio` |
| `test` | stable + nightly × Linux/macOS/Windows; default features, then `hf,audio`, then doctests; `metal` and `accelerate` type-checked on the macOS leg |
| `features` | every combination of `hf`/`audio`, plus `vulkan` and `webgpu` |
| `docs` | `cargo doc` on nightly with `--cfg docsrs` exactly as docs.rs builds it, then again on stable |
| `wasm` | `ptts-wasm` for `wasm32-unknown-unknown` with the SIMD flags real builds use, with and without `webgpu`; the `phonon-tts` JS wrapper's node tests; and `make build` with binaryen 124 |
| `coreml` | macOS only: clippy on `ptts-coreml` and `ptts-coreml-ffi` for macOS and iOS, then `ios/build-xcframework.sh` and `swift build` of the `PhononTTS` package |

`.github/actions/setup-rust` is a composite action holding the parts every job shares: the
toolchain, the cache, and the platform quirks below.

Three things worth knowing before editing it:

- **`--all-features` never works.** It turns on `cuda`, whose `cudarc` build script shells out to
  `nvcc`. Feature sets are always named explicitly, including in `[package.metadata.docs.rs]`.
- **`ptts-ws-server` needs a system libopus** (through `kaudio` → `libopus_sys`). CI installs it
  on Linux and macOS; Windows has no one-line equivalent, so the crate is excluded there and only
  there, through `$WS_EXCLUDE`. The `vulkan` feature likewise needs `glslc`, installed in the
  `features` job.
- **CI deletes `.cargo/config.toml`** because it pins `target-cpu=native`, which breaks portable
  dependency builds. If you reproduce a CI failure locally, do the same (`rm -f
  .cargo/config.toml`) — otherwise keep the file in place for fast local builds. The `wasm` job
  re-sets that file's SIMD flags itself.

Cargo features that gate optional functionality:

- `ptts`: `hf` (Hugging Face `tokenizers`, i.e. `ptts::tok`, required by the `say`, `ptts` and `bench` examples), `audio` (`ptts::audio`, decoding and resampling audio files for voice cloning — pulls in `symphonia` and `rubato`, so it is off by default and out of the wasm build; required by `ptts` and `create_voice`), `cuda`, `accelerate`. The library never downloads anything, so there is no hub feature: `hf-hub` is a dev-dependency used by the examples.
- `ptts-pyo3`: `cuda`, `accelerate` (each forwards to both `xn/*` and `ptts/*`).

Run the CLI example:

```
cargo run --release --example ptts --features hf,audio -- "hello world" -o out.wav
```

It downloads weights from the `kyutai/pocket-tts` HuggingFace repo on first run. Which files that means — the repo id, the weight and tokenizer file names, the bundled voice list, the config to assume when a directory ships none — lives in `ptts/examples/model_helpers.rs`, not in the library: it changes with each published checkpoint, and `ptts` only reads the files it is handed. Built-in voice IDs: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`. `--voice` also accepts a path to a 10s audio file or a voice safetensors: either a precomputed `emb` or the training pipeline's `speaker_wavs` latents, which `ptts::loader::load_voice_emb` runs through the checkpoint's speaker projection. `--repo <id>` downloads from another Hub repo with the same layout (`config.json`, weights, tokenizer, optional `embeddings/*.safetensors` voices and an optional `default-voice.safetensors`, which is picked when no `--voice` is given); `--weights <file>` names the weights file inside the repo or directory so only that one is downloaded (`--weights model.q8.gguf --quant q8` for the pre-quantized weights); `--tokenizer <file>` points at a `tokenizer.json` outside the checkpoint, for a repo that ships only a SentencePiece `tokenizer.model`; `--dir` loads a local checkpoint instead of downloading; `--device auto|cpu|cuda|vulkan|metal` picks the backend; `--lang en|fr|de|es|pt|none` picks the text-normalization language and is **required**.

`say` is the same thing in fifteen lines, for checking that the library works:

```
cargo run --release --example say --features hf -- "hello world"
```

Benchmark a local model:

```
cargo run --release --features hf,accelerate --example bench -- \
  --model model/model.q8.gguf --config model/config.json --quant q8 \
  --voice voices/freya.safetensors --threads 8 --iters 20
```

`bench` takes explicit paths and a precomputed voice embedding, never downloads, and reports time-to-first-audio, per-frame time, total generate time and RTF over `--iters` runs, excluding the one-off model load and voice conditioning. It decodes each frame on the generating thread rather than overlapping Mimi with the next frame's sampling as the `ptts` example does, so its RTF (generate time over audio duration, lower is better) reads higher than that example for the same weights — don't compare the two directly. `--threads` defaults to xn's one-per-logical-core, usually too many for a single autoregressive stream. For profiling rather than measuring, `ptts --chrome-tracing` writes a Chrome trace for https://ui.perfetto.dev.

## WASM build

From `ptts-wasm/`:

```
make build        # the phonon-tts npm package in pkg/: wasm-pack output in pkg/wasm/ and pkg/wasm-threads/, plus js/
make profiling    # same but --profiling (no wasm-opt)
make demo MODEL_DIR=/path/to/model   # pkg/ copied to site/phonon-tts/, www/index.html, the model folder as site/model/
make serve MODEL_DIR=/path/to/model  # make demo, then serve site/ on :8080, cross-origin isolated
make test         # node --test js/test/*.test.mjs -- the wrapper's logic, no browser or model needed
```

Requires `wasm-pack` 0.12 or later (`cargo install wasm-pack`), node 22.7 or later, and binaryen's `wasm-opt` 124 or later on `PATH`: wasm-pack otherwise downloads binaryen 117, and releases up to 123 abort on this module. The threaded build (`pkg/wasm-threads/`, the `threads` feature) also needs the nightly pinned in the Makefile with `rust-src`, since wasm threads need std rebuilt with atomics: `make threads-toolchain` installs it. `js/worker.js` loads that build only on a cross-origin isolated page, and the single-threaded one otherwise; `js/threads.js` picks the thread count, and `make serve` serves the demo with the isolation headers (`scripts/serve.mjs`). Both builds have the `webgpu` feature: `src/lib.rs` has one engine generic over the device, with only the readback differing, and `js/device.js` picks WebGPU when the browser offers a hardware adapter and the weights are q8 GGUF. `scripts/pack.mjs` assembles the package and stamps its version from `workspace.package.version`, so `js/package.json` deliberately has no `version`; it also derives what to copy from that file's `files` list. It deletes the `.gitignore` wasm-pack writes into `pkg/wasm/`: npm reads a subdirectory `.gitignore` as that directory's `.npmignore`, which would silently publish a package without its wasm. `make demo` and `make serve` take `MODEL_DIR`, a model folder: it is linked into `site/model/`, and `scripts/demo-model.mjs` writes `site/model.json` describing its weights and voices, since a static server cannot list a directory for the page. Wasm SIMD flags (`+simd128,+relaxed-simd`) and `getrandom_backend="wasm_js"` come from `.cargo/config.toml`. `relaxed-simd` is required rather than an optimization: `xn`'s quantized kernels call `f32x4_relaxed_madd` unconditionally, so browsers without Relaxed SIMD cannot compile the module at all.

Kyutai's published checkpoint URLs, pinned to HF revisions, are in `js/models.js`. Files are cached by URL, so bump those revisions together with the package version.

## Python build

From the repo root:

```
maturin develop --manifest-path ptts-pyo3/Cargo.toml          # local install
maturin build --release --manifest-path ptts-pyo3/Cargo.toml  # produce wheel
cd ptts-pyo3 && python -m pytest -m 'not checkpoint'          # against an installed wheel
ptts --lang en "hello world" -o out.wav                       # the console script
python -m ptts --lang en "hello world" -o out.wav             # the same `main`
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
is required there as it is on the `ptts` example and `ptts-ws-server`, but checked by hand rather
than by argparse, so that `--build-info` still works without one.

`pyo3` is built with `abi3-py39`, so one wheel per platform serves every CPython from 3.9 on
and a new CPython release needs no rebuild. abi3 does not load on free-threaded CPython and
PyPy needs its own ABI; both fall back to the sdist, which is why a CI job builds and tests it.
`pyproject.toml` deliberately has no `features` key under `[tool.maturin]`: a `--features` on
the maturin command line replaces that list rather than adding to it, so `pyo3/extension-module`
lives in `ptts-pyo3/Cargo.toml` where the macOS job's `--features accelerate` cannot drop it.

Release wheels are produced by `.github/workflows/maturin-pub.yml`: manylinux and musllinux
on x86_64 and aarch64, Windows on x64 and aarch64, macOS on aarch64 and x86_64, and an sdist.
PyPI accepts the upload through a trusted publisher pinned to the repository *and to that
file's name*, so renaming the file breaks releasing until PyPI is updated; a `v*` tag is what
triggers it. It started as `maturin generate-ci github -m ptts-pyo3/Cargo.toml` output and
has diverged; the header comment lists what a regeneration would undo. Chief among them:
every job deletes `.cargo/config.toml` and sets `RUSTFLAGS` itself, because `target-cpu=native`
in a published wheel means whatever CPU the runner had. Published x86_64 wheels target
`x86-64-v3` — `xn` selects its quantized kernels with `cfg!(target_feature = "avx")` at
compile time, so a lower baseline silently costs every `q8_0` path its AVX kernels.

## Architecture

The library implements Phonon: text → tokens → flow-matching language model produces Mimi codec latents → Mimi decoder produces 24 kHz PCM audio.

`ptts/src/lib.rs` exposes a single `Tokenizer` trait (`encode` / `decode`) so each binding plugs in its own implementation:

- `say` / `ptts` / `bench` examples, `ptts-pyo3` and `ptts-ws-server`: `ptts::tok::Tok` (the `hf` feature), a Hugging Face `tokenizers` wrapper. The examples find the file beside the weights and pass it to `SynthBuilder::tokenizer_file`.
- `ptts-wasm`: the same `ptts::tok::Tok`, built from the `tokenizer.json` the `phonon-tts` worker fetches and handed to `Model::new`; the browser passes text, not token ids.

Every frontend loads a `tokenizer.json` and nothing else, and none is bundled or defaulted to: each checkpoint has its own vocabulary, and loading the wrong one yields plausible audio from the wrong ids, so `Tok::open` refuses to guess. `ptts --tokenizer <path>` and `bench --tokenizer <path>` override where the examples look; otherwise they, `ptts-pyo3` and `ptts-ws-server` all expect `tokenizer.json` in the HF repo or beside the config. A checkpoint that carries only a `tokenizer.model` needs converting once with `scripts/convert-tokenizer.py`, which writes the equivalent json.

Top-level orchestrator is `tts_model::TTSModel<Q>`, generic over a backend-quantization parameter `Q: BackendQ` from `xn`. It owns:

- `flow_lm: FlowLM<Q>` — token-conditioned flow-matching transformer that emits Mimi latents (`flow_lm.rs`, `transformer.rs`, `rope.rs`, `mlp.rs`, `layer_scale.rs`, `conditioners.rs`).
- `mimi: MimiDecoder<Unquantized<f32, Q::B>>` — neural audio codec decoder (`mimi.rs`, `seanet.rs`, `conv.rs`, `resample.rs`, `dummy_quantizer.rs`). The encoder side (`MimiEncoder` / `MimiEnc`) is used only for voice-prompt embedding from a 10s audio sample.

`synth::Synth` sits on top of all of it: `synth::SynthBuilder::new(config, weights)` loads a
checkpoint whose files the caller has already located and registers voices,
`plan` supplies the frame/KV budgets and the EOS policy, and `Synth::say` / `Synth::stream`
run the flow LM and the Mimi decoder on two threads. `Synth` erases the `Q` parameter behind
an enum so a CLI flag can pick the weight format; `SynthBuilder::load::<Q>` skips that for
callers who want it fixed at compile time. `ptts-pyo3`, `ptts-wasm` and `ptts-ws-server`
still drive `TTSModel` directly.

Generation is streaming and stateful: callers `init_flow_lm_state(batch, seq_len)`, then `prompt_text*` / `prompt_audio` to seed the state, then step-decode latents and feed them into `MimiDecoderState`. `lsd_decode_steps` controls flow-matching solver steps; `eos_threshold` controls termination. The default `TTSConfig::v202601` configuration is the canonical one consumed by all three frontends.

Text normalization (`ptts/src/preprocess.rs`) is mandatory to choose and has no default. `preprocess::Normalize` is either `For(lang)` or `Off`, and it is a required third argument to `SynthBuilder::new`, a required `--lang` flag on the `ptts` and `bench` examples, and `ptts-ws-server`, a required keyword-only `lang=` on `ptts-pyo3`, and a required `lang` argument to the `ptts-wasm` `Model` constructor and to `PhononTTS.load` in `phonon-tts`. The reason it is not defaulted rather than defaulted to English: normalization makes the model noticeably better, but the spoken forms of `@`, `+` and `=` are per-language, so normalizing German as English says "at" where it should say "ät" -- guessing is worse than doing nothing. `Normalize::Off` (`--lang none`, `lang="none"`) hands text to the tokenizer as written.

`Normalize::apply` is the one implementation, and it has to run before `prepare_text_prompt`, whose leading-space padding of short text it would otherwise collapse. `Synth::normalization` / `Session::normalization` hand it to callers that tokenize by hand (`ptts-ws-server`, `ptts-wasm`) rather than going through `say`/`stream`.

Quantization story: only `flow_lm.transformer.layers.*.{linear1,linear2,self_attn.in_proj,self_attn.out_proj}.weight` get GGML-quantized (see `examples/quantize.rs`); Mimi stays in `Unquantized<f32>`. The Mimi quantizer codebook tensors (`mimi.quantizer.*` except `output_proj`) are excluded from output GGUFs since the runtime uses `dummy_quantizer.rs`.

## Conventions to be aware of

- `target-cpu=native` is on by default for host builds and `apple-m1` on the macOS CI release lane; do not assume binaries are portable.
- macOS x86_64 disables AVX/AVX2 (`.cargo/config.toml`), keep that in mind when benchmarking.
- The single workspace version (`workspace.package.version`) is shared by all three crates and the `ptts` workspace dep — update them together.
