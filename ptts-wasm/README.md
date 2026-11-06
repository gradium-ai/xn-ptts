# ptts-wasm

The browser build of [Phonon](../ptts/), published to npm as [`phonon-tts`](https://www.npmjs.com/package/phonon-tts). This README is about building and changing it. For using the package, see [`js/README.md`](js/README.md), which is also the README on npm.

## Layout

- `src/lib.rs`: the raw `wasm-bindgen` surface, CPU only. `Model` runs the whole model, taking bytes that are already fetched and generating one 80 ms frame per call, because the worker it runs in must yield to its event loop between calls to hear a cancel. With the `threads` feature it also exports `initThreadPool` and `start_cpu_pool`, which split the work inside a frame across Web Workers. Text is normalized, split into sentence-aligned chunks and tokenized in Rust, with the same rules as `ptts::synth`. Voices can be `emb` embeddings or `speaker_wavs` latents, which are run through the model once when they are added, or the precomputed KV caches of `embeddings_v2/`. `Preparer` is the same text and voice handling without the model, for the WebGPU engine: it holds the tokenizer, the config and the few tensors of the conditioners and the speaker projection, and hands out token ids, voice prompt rows and the conditioner sum.
- `js/gpu/`: the WebGPU engine, in JS and WGSL. It reads the q8 GGUF itself and runs the flow LM and the Mimi decoder on the device, with nothing read back but the audio; `engine.js` starts with how it fits together (dispatch lists, the state buffer, the submit pipeline, subgroups), and `support.js` says which checkpoints it runs. The weights never enter wasm memory.
- `js/`: the package's public API. `index.js` exports `PhononTTS`, which runs the model in a worker (`worker.js`), downloads and caches its files (`fetch.js`, via the Cache API), and turns requests into async iterators. The worker runs either the CPU `Model` (the threaded build on a cross-origin isolated page and the single-threaded one elsewhere, `threads.js` picking how many threads) or the WebGPU engine with the single-threaded build's `Preparer` (`device.js` picks). `index.d.ts` holds the types. `test/` holds node tests for the wrapper's own logic and the engine's host-side logic.
- `scripts/pack.mjs`: assembles the npm package around the wasm-pack outputs: `pkg/wasm/` and `pkg/wasm-threads/`, plus the `.wasm` of `pkg/wasm-simd/` and `pkg/wasm-threads-simd/`, the builds without Relaxed SIMD, kept as `phonon_tts_simd_bg.wasm`. It also patches `wasm-bindgen-rayon`'s worker helper, whose bare `'../../..'` import resolves for neither a bundler nor a browser here.
- `scripts/serve.mjs`: serves the demo with the headers that make it cross-origin isolated.
- `www/index.html`: the demo page, built on the package the way a consumer would use it.

### Driving `src/lib.rs` directly

`js/worker.js` is the only caller. On the CPU this is the loop it runs. Text is split in Rust, so prompting is per chunk and generation is per frame, which makes it two levels deep:

```js
// device must be 'cpu': WebGPU runs in the package's own engine (js/gpu/).
const model = await Model.load(modelWeights, tokenizerJson, configJson, quant, lang, rewrites, conditions, 'cpu');
const voiceIndex = model.add_voice(voiceBytes);

// Splits and tokenizes. Runs no model, and returns the number of chunks.
model.start_generation(voiceIndex, text, temperature, seed);

while (true) {
  // Prompts the next chunk's text, or returns undefined once every chunk is done.
  if (model.next_chunk() === undefined) break;

  while (true) {
    // One 80 ms frame of mono PCM at model.sample_rate(), model.frame_size() samples, or
    // undefined at the end of the chunk.
    const pcm = await model.generation_step();
    if (!pcm) break;
    // ... play or buffer pcm ...
  }
}
```

`stop_generation()` drops a generation in progress. An error thrown by `next_chunk` or `generation_step` also drops it, so a caller that swallows one cannot carry on and silently lose a sentence -- every later call reports the end instead. One noise source covers every chunk, so `seed` fixes the whole utterance. Calls must not overlap: a call that arrives while `generation_step` holds the model is refused. `stop_generation` is the exception: it takes effect when the pending step returns. See the rustdoc on `Model::load` for `quant`, `lang`, `conditions` and what a supplied `config.json` does not change.

This API changed in 0.4: `Model` used to be built with `new Model(...)`, with no `device`, and `generation_step` returned one frame directly rather than a promise. Since the WebGPU engine moved out of this module, `device: 'webgpu'` is an error here and `flush()` resolves at once.

The WebGPU engine's side is `new Preparer(tokenizerJson, configJson, tensors, lang, rewrites, conditions)`, where `tensors` is a safetensors file of the GGUF's `flow_lm.condition_provider.conditioners.*` tensors: `plan(text)` gives the chunks (`{ tokens, framesAfterEos, frameBudget }`), `voice(bytes)` a voice's prompt rows or KV cache, and `conditions()` the vector every frame's input adds. `js/worker.js` shows it driving `js/gpu/engine.js`.

## Build

Needs [wasm-pack](https://github.com/drager/wasm-pack) 0.12 or later, for `--no-pack` (`cargo install wasm-pack`), node 22.7 or later, for module-syntax detection on the `.js` files under `js/`, and [binaryen](https://github.com/WebAssembly/binaryen/releases)'s `wasm-opt` 124 or later on `PATH` (`brew install binaryen`). The `wasm-opt` that wasm-pack downloads by itself is too old for this module and aborts; `make profiling` skips it. The threaded build also needs the nightly toolchain pinned in the Makefile, with `rust-src`, since wasm threads need std rebuilt with atomics. [cargo-about](https://github.com/EmbarkStudios/cargo-about) writes the package's `THIRD_PARTY_LICENSES.md`, the licence notices of the crates compiled into it (`cargo install cargo-about --locked --features cli`); `about.toml` lists the licences it accepts.

```bash
make threads-toolchain                   # once: the pinned nightly, with rust-src
make build                               # the npm package, in pkg/: both builds and the WebGPU engine
make test                                # the wrapper's tests: no browser, no model
make serve                               # build, then serve the demo from site/ on http://localhost:8080
```

`make profiling` builds only the single-threaded module, without wasm-opt, keeping names for the browser profiler. `make serve` sends the cross-origin isolation headers, so the demo runs on the threaded build and shows how many threads it got.

The demo loads the package's default model. `make serve MODEL_DIR=/path/to/model` serves a local one instead: `MODEL_DIR` is a model folder holding `config.json`, `tokenizer.json`, `model.q8.gguf` or `model.safetensors`, and its voices under `voices/` or `embeddings/` or as `default-voice.safetensors`. `make demo` links it into `site/model/` and writes `site/model.json` describing what is in it, since the page cannot list a directory over HTTP. The page offers the weight formats the folder has, downloads them the first time, then loads them from the browser's cache.

The package version is not in `js/package.json`. `pack.mjs` stamps it from `workspace.package.version` in the top-level `Cargo.toml`, so npm, PyPI and crates.io stay on one version.

## Publishing

`.github/workflows/npm-publish.yml` builds the package on every PR that touches it. It publishes on a `v*` tag, the same tag that publishes the Python wheels. It uses npm trusted publishing, so no token is stored here.

Trusted publishing is enabled for `phonon-tts` on npmjs.com, pointed at this repository and at `npm-publish.yml`. Renaming that workflow file breaks publishing until the setting on npmjs.com is updated to match.

Supply the checkpoint URLs through `ModelSpec`, or name a public Hugging Face repo as `hf://owner/name@revision`. Use pinned HF revisions or versioned local paths because files are cached by URL; an `hf://` model is pinned to the commit its revision points at. Every checkpoint must supply its config and tokenizer JSON.

## Known limits

- The module needs WebAssembly SIMD. Each build ships twice, with and without Relaxed SIMD, and `js/worker.js` loads the Relaxed one only where the browser has it (`js/simd.js`). Safari has no Relaxed SIMD, so neither has any browser on iOS; there the CPU runs slower.
- No voice cloning: the Mimi encoder is not in the browser build.
- Threads need a cross-origin isolated page. Elsewhere generation runs on one thread.
- WebGPU needs `q8` weights in a GGUF file: quantizing `f32` weights on the GPU would read each one back to the host.
- The WebGPU engine runs the published Phonon checkpoints (`gradium/phonon-31m-en` and the `gradium/phonon-40m-*` models): a 512-wide flow LM of any depth with a one-step LSD flow head and the exact GELU, and their Mimi decoder. Other configs (more LSD steps, the drifting or pointwise heads, grouped-query attention, tanh GELU, `z_dim`, a low-rank text table, prefix sinks, other Mimi shapes) run on the CPU; `js/gpu/support.js` lists the checks. Under `device: 'auto'` such a checkpoint goes to the CPU with a warning, under `'webgpu'` the load fails with the reason.
- The WebGPU engine's fast kernels use 32-wide subgroups; on adapters without them (or with another width, such as Intel, AMD's wave64 or SwiftShader) it runs equivalent shared-memory kernels.
