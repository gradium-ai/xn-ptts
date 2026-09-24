# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

Cargo workspace (resolver "3", edition 2024) with four members:

- `ptts/` — core TTS library. Pure Rust, depends on the `xn` tensor/nn crate. Examples live under `ptts/examples/`: `say` (shortest end-to-end call) and `bench` (benchmark harness) require the `hf` feature for the tokenizer, `pocket_tts` (full CLI) requires `hf` and `audio`, `create_voice` (voice embeddings from audio samples) requires `audio`, and `quantize` (safetensors → GGUF converter that selectively quantizes `flow_lm.transformer.layers.*` weights) requires nothing. `model_helpers.rs` is not an example — it is a shared module each example pulls in with `#[path = "..."] mod`, so `autoexamples = false` and every example is listed explicitly in `Cargo.toml`.
- `ptts-pyo3/` — PyO3 bindings exposing `TTSModel` to Python. Built with maturin; the cdylib is named `ptts`. Has its own `pyproject.toml` and `uv.lock`.
- `ptts-wasm/` — browser build via `wasm-bindgen` / `wasm-pack`, published to npm as `phonon-tts`. `src/lib.rs` is the raw frame-at-a-time `Model`; `js/` is the package's public API around it (`PhononTTS`, which runs the model in a worker, downloads and caches the files, and speaks by voice name), with its own `package.json`, `README.md` and node tests. `www/index.html` is a demo page built on the package.
- `ptts-ws-server/` — WebSocket streaming server (`axum` + `kaudio`). Needs a system libopus through `kaudio` → `libopus_sys`, which is why CI installs it on Linux and macOS and skips this crate on Windows.

Shared dependency versions (notably `xn`) and the workspace version live in the top-level `Cargo.toml`. Bumping the release version means editing `workspace.package.version` and the `ptts` workspace dep.

## Build / test / lint

CI (`.github/workflows/rust-ci.yml`) is the source of truth. Seven jobs, gated behind one
required check called `CI`:

| Job | What it covers |
|---|---|
| `fmt` | `cargo fmt --all -- --check` (rustfmt.toml: `use_small_heuristics = "Max"`, edition 2024) |
| `clippy` | whole workspace `--all-targets -D warnings`, then `ptts` with `hf,audio` |
| `test` | stable + nightly × Linux/macOS/Windows; default features, then `hf,audio`, then doctests; `metal` and `accelerate` type-checked on the macOS leg |
| `features` | every combination of `hf`/`audio`, plus `vulkan` and `webgpu` |
| `docs` | `cargo doc` on nightly with `--cfg docsrs` exactly as docs.rs builds it, then again on stable |
| `wasm` | `ptts-wasm` for `wasm32-unknown-unknown` with the SIMD flags real builds use, and the `phonon-tts` JS wrapper's node tests |

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

- `ptts`: `hf` (Hugging Face `tokenizers`, i.e. `ptts::tok`, required by the `say`, `pocket_tts` and `bench` examples), `audio` (`ptts::audio`, decoding and resampling audio files for voice cloning — pulls in `symphonia` and `rubato`, so it is off by default and out of the wasm build; required by `pocket_tts` and `create_voice`), `cuda`, `accelerate`. The library never downloads anything, so there is no hub feature: `hf-hub` is a dev-dependency used by the examples.
- `ptts-pyo3`: `cuda`, `accelerate` (each forwards to both `xn/*` and `ptts/*`).

Run the CLI example:

```
cargo run --release --example pocket_tts --features hf,audio -- "hello world" -o out.wav
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

`bench` takes explicit paths and a precomputed voice embedding, never downloads, and reports time-to-first-audio, per-frame time, total generate time and RTF over `--iters` runs, excluding the one-off model load and voice conditioning. It decodes each frame on the generating thread rather than overlapping Mimi with the next frame's sampling as `pocket_tts` does, so its RTF reads lower than `pocket_tts` for the same weights — don't compare the two directly. `--threads` defaults to xn's one-per-logical-core, usually too many for a single autoregressive stream. For profiling rather than measuring, `pocket_tts --chrome-tracing` writes a Chrome trace for https://ui.perfetto.dev.

## WASM build

From `ptts-wasm/`:

```
make build        # the phonon-tts npm package in pkg/: wasm-pack output in pkg/wasm/, plus js/
make profiling    # same but --profiling (no wasm-opt)
make demo         # pkg/ copied to site/phonon-tts/, plus www/index.html
make serve        # make demo, then serve site/ on :8080
make test         # node --test js/test/ -- the wrapper's logic, no browser or model needed
```

Requires `wasm-pack` (`cargo install wasm-pack`) and node. `scripts/pack.mjs` assembles the package and stamps its version from `workspace.package.version`, so `js/package.json` deliberately has no `version`. It also deletes the `.gitignore` wasm-pack writes into `pkg/wasm/`: npm reads a subdirectory `.gitignore` as that directory's `.npmignore`, which would silently publish a package without its wasm. The demo downloads the q8 weights (~146 MB) from HuggingFace once and keeps them in the Cache API. Wasm SIMD flags (`+simd128,+relaxed-simd`) and `getrandom_backend="wasm_js"` come from `.cargo/config.toml`. `relaxed-simd` is required rather than an optimization: `xn`'s quantized kernels call `f32x4_relaxed_madd` unconditionally, so browsers without Relaxed SIMD cannot compile the module at all.

The default checkpoint's URLs, pinned to HF revisions, are in `js/models.js`. Files are cached by URL, so bump those revisions together with the package version. `.github/workflows/npm-publish.yml` builds the package on PRs that touch it and publishes it on a `v*` tag through npm trusted publishing (OIDC, no token).

## Python build

From the repo root:

```
maturin develop --manifest-path ptts-pyo3/Cargo.toml          # local install
maturin build --release --manifest-path ptts-pyo3/Cargo.toml  # produce wheel
```

Release wheels are produced by `.github/workflows/maturin-pub.yml` (Linux x86_64 manylinux + musllinux, Windows x64, macOS aarch64, sdist), and a `v*` tag is what publishes them. It is autogenerated — regenerate with `maturin generate-ci github -m ptts-pyo3/Cargo.toml` rather than hand-editing, but one edit is hand-added and a regeneration drops it: every job deletes `.cargo/config.toml` and sets `RUSTFLAGS` itself, because `target-cpu=native` in a published wheel means whatever CPU the runner had. Published x86_64 wheels target `x86-64-v3` — `xn` selects its quantized kernels with `cfg!(target_feature = "avx")` at compile time, so a lower baseline silently costs every `q8_0` path its AVX kernels.

`pyproject.toml` deliberately has no `features` key under `[tool.maturin]`: a `--features` on the maturin command line replaces that list rather than adding to it, so `pyo3/extension-module` lives in `ptts-pyo3/Cargo.toml` where the macOS job's `--features accelerate` cannot drop it.

## Architecture

The library implements Pocket TTS: text → tokens → flow-matching language model produces Mimi codec latents → Mimi decoder produces 24 kHz PCM audio.

`ptts/src/lib.rs` exposes a single `Tokenizer` trait (`encode` / `decode`) so each binding plugs in its own implementation:

- `say` / `pocket_tts` / `bench` examples, `ptts-pyo3` and `ptts-ws-server`: `ptts::tok::Tok` (the `hf` feature), a Hugging Face `tokenizers` wrapper. The examples find the file beside the weights and pass it to `SynthBuilder::tokenizer_file`.
- `ptts-wasm`: the same `ptts::tok::Tok`, built from the `tokenizer.json` the `phonon-tts` worker fetches and handed to `Model::new`; the browser passes text, not token ids.

Every frontend loads a `tokenizer.json` and nothing else, and none is bundled or defaulted to: each checkpoint has its own vocabulary, and loading the wrong one yields plausible audio from the wrong ids, so `Tok::open` refuses to guess. `pocket_tts --tokenizer <path>` and `bench --tokenizer <path>` override where the examples look; otherwise they, `ptts-pyo3` and `ptts-ws-server` all expect `tokenizer.json` in the HF repo or beside the config. A checkpoint that carries only a `tokenizer.model` needs converting once with `scripts/convert-tokenizer.py`, which writes the equivalent json.

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

Text normalization (`ptts/src/preprocess.rs`) is mandatory to choose and has no default. `preprocess::Normalize` is either `For(lang)` or `Off`, and it is a required third argument to `SynthBuilder::new`, a required `--lang` flag on `pocket_tts`, `bench` and `ptts-ws-server`, a required keyword-only `lang=` on `ptts-pyo3`, and a required `lang` argument to the `ptts-wasm` `Model` constructor and to `PhononTTS.load` in `phonon-tts`. The reason it is not defaulted rather than defaulted to English: normalization makes the model noticeably better, but the spoken forms of `@`, `+` and `=` are per-language, so normalizing German as English says "at" where it should say "ät" -- guessing is worse than doing nothing. `Normalize::Off` (`--lang none`, `lang="none"`) hands text to the tokenizer as written.

`Normalize::apply` is the one implementation, and it has to run before `prepare_text_prompt`, whose leading-space padding of short text it would otherwise collapse. `Synth::normalization` / `Session::normalization` hand it to callers that tokenize by hand (`ptts-ws-server`, `ptts-wasm`) rather than going through `say`/`stream`.

Quantization story: only `flow_lm.transformer.layers.*.{linear1,linear2,self_attn.in_proj,self_attn.out_proj}.weight` get GGML-quantized (see `examples/quantize.rs`); Mimi stays in `Unquantized<f32>`. The Mimi quantizer codebook tensors (`mimi.quantizer.*` except `output_proj`) are excluded from output GGUFs since the runtime uses `dummy_quantizer.rs`.

## Conventions to be aware of

- `target-cpu=native` is on by default for host builds and `apple-m1` on the macOS CI release lane; do not assume binaries are portable.
- macOS x86_64 disables AVX/AVX2 (`.cargo/config.toml`), keep that in mind when benchmarking.
- The single workspace version (`workspace.package.version`) is shared by all three crates and the `ptts` workspace dep — update them together.
