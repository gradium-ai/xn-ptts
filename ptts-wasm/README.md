# ptts-wasm

The browser build of [Phonon](../ptts/), published to npm as [`phonon-tts`](https://www.npmjs.com/package/phonon-tts). This README is about building and changing it. For using the package, see [`js/README.md`](js/README.md), which is also the README on npm.

## Layout

- `src/lib.rs`: the raw `wasm-bindgen` surface. It takes bytes that are already fetched and generates a few 80 ms frames per call, because the worker it runs in must yield to its event loop between calls to hear a cancel. One engine, generic over the device, serves the CPU and, with the `webgpu` feature, WebGPU: loading, voices, normalization, chunking and the end-of-speech rule are shared, and only reading a result back differs. A call makes one frame on the CPU and up to four on WebGPU, which reads them back in one round trip. With the `threads` feature it also exports `initThreadPool` and `start_cpu_pool`, which split the work inside a frame across Web Workers. Text is normalized, split into sentence-aligned chunks and tokenized in Rust, with the same rules as `ptts::synth`. Voices can be `emb` embeddings, which are run through the model once when they are added, or the precomputed KV caches of `embeddings_v2/`.
- `js/`: the package's public API. `index.js` exports `PhononTTS`, which runs the model in a worker (`worker.js`), downloads and caches its files (`fetch.js`, via the Cache API), and turns requests into async iterators. The worker loads the threaded build on a cross-origin isolated page and the single-threaded one elsewhere, and `threads.js` picks how many threads. `index.d.ts` holds the types. `test/` holds node tests for the wrapper's own logic.
- `scripts/pack.mjs`: assembles the npm package around the wasm-pack outputs: `pkg/wasm/` and `pkg/wasm-threads/`, plus the `.wasm` of `pkg/wasm-simd/` and `pkg/wasm-threads-simd/`, the builds without Relaxed SIMD, kept as `phonon_tts_simd_bg.wasm`. It also patches `wasm-bindgen-rayon`'s worker helper, whose bare `'../../..'` import resolves for neither a bundler nor a browser here.
- `scripts/serve.mjs`: serves the demo with the headers that make it cross-origin isolated.
- `www/index.html`: the demo page, built on the package the way a consumer would use it.

### Driving `src/lib.rs` directly

`js/worker.js` is the only caller, and this is the loop it runs, on either device. Text is split in Rust, so prompting is per chunk and generation is per step, which makes it two levels deep:

```js
// device is 'cpu' or 'webgpu'; WebGPU needs q8 weights in a GGUF file.
const model = await Model.load(modelWeights, tokenizerJson, configJson, quant, lang, rewrites, conditions, device);
const voiceIndex = model.add_voice(voiceBytes);
// On WebGPU the voice's prompt is only recorded; this runs it now rather than in the first step.
await model.flush();

// Splits and tokenizes. Runs no model, and returns the number of chunks.
model.start_generation(voiceIndex, text, temperature, seed);

while (true) {
  // Prompts the next chunk's text, or returns undefined once every chunk is done.
  if (model.next_chunk() === undefined) break;

  while (true) {
    // Whole 80 ms frames of mono PCM at model.sample_rate(), model.frame_size() samples
    // each, or undefined at the end of the chunk. One frame on the CPU, up to four on WebGPU.
    const pcm = await model.generation_step();
    if (!pcm) break;
    // ... play or buffer pcm ...
  }
}
```

`stop_generation()` drops a generation in progress. An error thrown by `next_chunk` or `generation_step` also drops it, so a caller that swallows one cannot carry on and silently lose a sentence -- every later call reports the end instead. One noise source covers every chunk, so `seed` fixes the whole utterance. Calls must not overlap: `generation_step` holds the model while it waits for the GPU, and a call that arrives meanwhile is refused. `stop_generation` is the exception: it takes effect when the pending step returns. This API changed in 0.4: `Model` used to be built with `new Model(...)`, with no `device`, and `generation_step` returned one frame directly rather than a promise of one or more. See the rustdoc on `Model::load` for `quant`, `lang`, `conditions`, `device` and what a supplied `config.json` does not change.

## Build

Needs [wasm-pack](https://github.com/drager/wasm-pack) 0.12 or later, for `--no-pack` (`cargo install wasm-pack`), node 22.7 or later, for module-syntax detection on the `.js` files under `js/`, and [binaryen](https://github.com/WebAssembly/binaryen/releases)'s `wasm-opt` 124 or later on `PATH` (`brew install binaryen`). The `wasm-opt` that wasm-pack downloads by itself is too old for this module and aborts; `make profiling` skips it. The threaded build also needs the nightly toolchain pinned in the Makefile, with `rust-src`, since wasm threads need std rebuilt with atomics. [cargo-about](https://github.com/EmbarkStudios/cargo-about) writes the package's `THIRD_PARTY_LICENSES.md`, the licence notices of the crates compiled into it (`cargo install cargo-about --locked --features cli`); `about.toml` lists the licences it accepts.

```bash
make threads-toolchain                   # once: the pinned nightly, with rust-src
make build                               # the npm package, in pkg/: both builds
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
