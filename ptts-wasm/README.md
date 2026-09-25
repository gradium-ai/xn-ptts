# ptts-wasm

The browser build of [Pocket TTS](../ptts/), published to npm as [`phonon-tts`](https://www.npmjs.com/package/phonon-tts). This README is about building and changing it. For using the package, see [`js/README.md`](js/README.md), which is also the README on npm.

## Layout

- `src/lib.rs`: the raw `wasm-bindgen` surface. It takes bytes that are already fetched and generates one 80 ms frame per call, because the browser has no threads to hand generation to. Text is normalized, split into sentence-aligned chunks and tokenized in Rust, with the same rules as `ptts::synth`. Voices can be `emb` embeddings, which are run through the model once when they are added, or the precomputed KV caches of `embeddings_v2/`.
- `js/`: the package's public API. `index.js` exports `PhononTTS`, which runs the model in a worker (`worker.js`), downloads and caches its files (`fetch.js`, via the Cache API), and turns requests into async iterators. `models.js` says where the default checkpoint lives. `index.d.ts` holds the types. `test/` holds node tests for the wrapper's own logic.
- `scripts/pack.mjs`: assembles the npm package around the wasm-pack output.
- `www/index.html`: the demo page, built on the package the way a consumer would use it.

### Driving `src/lib.rs` directly

`js/worker.js` is the only caller, and this is the loop it runs. Text is split in Rust, so prompting is per chunk and generation is per frame, which makes it two levels deep:

```js
const model = new Model(modelWeights, tokenizerJson, configJson, quant, lang, rewrites);
const voiceIndex = model.add_voice(voiceBytes);

// Splits and tokenizes. Runs no model, and returns the number of chunks.
model.start_generation(voiceIndex, text, temperature, seed);

while (true) {
  // Prompts the next chunk's text, or returns undefined once every chunk is done.
  if (model.next_chunk() === undefined) break;

  while (true) {
    // 80ms of mono PCM at model.sample_rate(), or undefined at the end of the chunk.
    const pcm = model.generation_step();
    if (!pcm) break;
    // ... play or buffer pcm ...
  }
}
```

`stop_generation()` drops a generation in progress. An error thrown by `next_chunk` or `generation_step` also drops it, so a caller that swallows one cannot carry on and silently lose a sentence -- every later call reports the end instead. One noise source covers every chunk, so `seed` fixes the whole utterance. See the rustdoc on `Model::new` for `quant`, `lang` and what a supplied `config.json` does not change.

## Build

Needs [wasm-pack](https://github.com/drager/wasm-pack) (`cargo install wasm-pack`) and node.

```bash
make build    # the npm package, in pkg/
make test     # the wrapper's tests: no browser, no model
make serve    # build, then serve the demo from site/ on http://localhost:8080
```

The page downloads the q8 weights (about 146 MB) from Hugging Face the first time, then loads them from the browser's cache.

The package version is not in `js/package.json`. `pack.mjs` stamps it from `workspace.package.version` in the top-level `Cargo.toml`, so npm, PyPI and crates.io stay on one version.

## Publishing

`.github/workflows/npm-publish.yml` builds the package on every PR that touches it. It publishes on a `v*` tag, the same tag that publishes the Python wheels. It uses npm trusted publishing, which has to be enabled once for `phonon-tts` on npmjs.com.

Before a release, check the default checkpoint in `js/models.js`. Its URLs are pinned to Hugging Face revisions, and the files are cached by URL, so changing a revision makes every user download again.

## Known limits

- The module needs WebAssembly Relaxed SIMD. `xn`'s quantized kernels call `f32x4_relaxed_madd` unconditionally, so a browser without it cannot compile the module, even for f32 weights.
- No voice cloning: the Mimi encoder is not in the browser build.
- The `webgpu` feature does not compile for `wasm32`. See the note in `.github/workflows/rust-ci.yml`.
