# ptts-wasm

The browser build of [Phonon](../ptts/), published to npm as [`phonon-tts`](https://www.npmjs.com/package/phonon-tts). This README is about building and changing it. For using the package, see [`js/README.md`](js/README.md), which is also the README on npm.

## Layout

- `src/lib.rs`: the raw `wasm-bindgen` surface. It takes bytes that are already fetched and generates a few 80 ms frames per call, because the worker it runs in must yield to its event loop between calls to hear a cancel. One engine, generic over the device, serves the CPU and, with the `webgpu` feature, WebGPU: loading, voices, normalization, chunking and the end-of-speech rule are shared, and only reading a result back differs. A call makes one frame on the CPU and eight on WebGPU, which reads them back in one round trip. With the `threads` feature it also exports `init_thread_pool` and `start_cpu_pool`, which split the work inside a frame across Web Workers. Text is normalized, split into sentence-aligned chunks and tokenized in Rust, with the same rules as `ptts::synth`. Voices can be `emb` embeddings, which are run through the model once when they are added, or the precomputed KV caches of `embeddings_v2/`.
- `js/`: the package's public API. `index.js` exports `PhononTTS`, which runs the model in a worker (`worker.js`), downloads and caches its files (`fetch.js`, via the Cache API), and turns requests into async iterators. The worker loads the threaded build on a cross-origin isolated page and the single-threaded one elsewhere, and `threads.js` picks how many threads. `models.js` holds the pinned URLs of Kyutai's published Pocket TTS checkpoint. `index.d.ts` holds the types. `test/` holds node tests for the wrapper's own logic.
- `scripts/pack.mjs`: assembles the npm package around the two wasm-pack outputs, `pkg/wasm/` and `pkg/wasm-threads/`. It also patches `wasm-bindgen-rayon`'s worker helper, whose bare `'../../..'` import resolves for neither a bundler nor a browser here.
- `scripts/serve.mjs`: serves the demo with the headers that make it cross-origin isolated.
- `www/index.html`: the demo page, built on the package the way a consumer would use it.

### Driving `src/lib.rs` directly

`js/worker.js` is the only caller, and this is the loop it runs, on either device. Text is split in Rust, so prompting is per chunk and generation is per step, which makes it two levels deep:

```js
// device is 'cpu' or 'webgpu'; WebGPU needs q8 weights in a GGUF file.
const model = await Model.load(modelWeights, tokenizerJson, configJson, quant, lang, rewrites, device);
const voiceIndex = model.add_voice(voiceBytes);

// Splits and tokenizes. Runs no model, and returns the number of chunks.
model.start_generation(voiceIndex, text, temperature, seed);

while (true) {
  // Prompts the next chunk's text, or returns undefined once every chunk is done.
  if (model.next_chunk() === undefined) break;

  while (true) {
    // Whole 80 ms frames of mono PCM at model.sample_rate(), model.frame_size() samples
    // each, or undefined at the end of the chunk. One frame on the CPU, up to eight on WebGPU.
    const pcm = await model.generation_step();
    if (!pcm) break;
    // ... play or buffer pcm ...
  }
}
```

`stop_generation()` drops a generation in progress. An error thrown by `next_chunk` or `generation_step` also drops it, so a caller that swallows one cannot carry on and silently lose a sentence -- every later call reports the end instead. One noise source covers every chunk, so `seed` fixes the whole utterance. Calls must not overlap: `generation_step` holds the model while it waits for the GPU, and a call that arrives meanwhile is refused. See the rustdoc on `Model::load` for `quant`, `lang`, `device` and what a supplied `config.json` does not change.

## Build

Needs [wasm-pack](https://github.com/drager/wasm-pack) 0.12 or later, for `--no-pack` (`cargo install wasm-pack`), node 22.7 or later, for module-syntax detection on the `.js` files under `js/`, and [binaryen](https://github.com/WebAssembly/binaryen/releases)'s `wasm-opt` 124 or later on `PATH` (`brew install binaryen`). The `wasm-opt` that wasm-pack downloads by itself is too old for this module and aborts; `make profiling` skips it. The threaded build also needs the nightly toolchain pinned in the Makefile, with `rust-src`, since wasm threads need std rebuilt with atomics.

```bash
make threads-toolchain                   # once: the pinned nightly, with rust-src
make build                               # the npm package, in pkg/: both builds
make test                                # the wrapper's tests: no browser, no model
make serve MODEL_DIR=/path/to/model      # build, then serve the demo from site/ on http://localhost:8080
```

`make profiling` builds only the single-threaded module, without wasm-opt, keeping names for the browser profiler. `make serve` sends the cross-origin isolation headers, so the demo runs on the threaded build and shows how many threads it got.

`MODEL_DIR` is a model folder holding `tokenizer.json`, `model.q8.gguf` or `model.safetensors`, an optional `config.json`, and voices under `voices/`. `make demo` links it into `site/model/` and writes `site/model.json` describing what is in it, since the page cannot list a directory over HTTP. The page offers the weight formats the folder has, downloads them the first time, then loads them from the browser's cache.

The package version is not in `js/package.json`. `pack.mjs` stamps it from `workspace.package.version` in the top-level `Cargo.toml`, so npm, PyPI and crates.io stay on one version.

## Before a release

Check the checkpoint URLs in `js/models.js`. They are pinned to Hugging Face revisions, and the files are cached by URL, so changing a revision makes every user of them download again.

## Known limits

- The module needs WebAssembly Relaxed SIMD. `xn`'s quantized kernels call `f32x4_relaxed_madd` unconditionally, so a browser without it cannot compile the module, even for f32 weights.
- No voice cloning: the Mimi encoder is not in the browser build.
- Threads need a cross-origin isolated page. Elsewhere generation runs on one thread.
- WebGPU needs `q8` weights in a GGUF file: quantizing `f32` weights on the GPU would read each one back to the host.
