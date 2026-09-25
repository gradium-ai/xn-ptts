# wasm-pocket-tts

WebAssembly build of [Pocket TTS](../ptts/) — run text-to-speech directly in the browser.

Try it online [here](https://laurentmazare.github.io/pocket-tts).

## Prerequisites

Install [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

```bash
cargo install wasm-pack
```

## Build

From the `ptts-wasm/` directory:

```bash
make build
```

This runs `wasm-pack build` and copies `www/` into `pkg/`.

## Run

Serve the `pkg/` directory with any HTTP server, for example:

```bash
cd ptts-wasm/pkg
python3 -m http.server 8080
```

Then open http://localhost:8080 in your browser. The page will download the
model weights from HuggingFace on first use (~240 MB) and cache them for subsequent generations.

## The JS API

A browser has no threads to hand generation to, so the module generates one frame per call
and the caller yields to the event loop between them. Long text is split into
sentence-aligned chunks, and each chunk is prompted before its frames are generated, which
makes the loop two levels deep:

```js
const model = new Model(modelWeights, tokenizerJson, configJson, quant, lang, rewrites);
const voiceIndex = model.add_voice(voiceBytes);

// Splits and tokenizes. Runs no model, and returns the number of chunks.
model.start_generation(voiceIndex, text, temperature, seed);

while (true) {
  // Prompts the next chunk's text, or returns undefined once every chunk is done.
  const numTokens = model.next_chunk();
  if (numTokens === undefined) break;

  while (true) {
    // 80ms of mono PCM at model.sample_rate(), or undefined at the end of the chunk.
    const pcm = model.generation_step();
    if (!pcm) break;
    // ... play or buffer pcm ...
  }
}
```

`model.stop_generation()` drops a generation in progress. An error thrown by `next_chunk` or
`generation_step` also drops it, so a caller that swallows one cannot carry on and silently
lose a sentence -- every later call reports the end instead.

`configJson` is a checkpoint's `config.json`, or `undefined` for the original Pocket TTS
architecture; `rewrites` may be omitted. See `Model::new` in `src/lib.rs` for `quant`, `lang`
and what a supplied config does not change.

### Incompatible with earlier builds

`Model::new` and `start_generation` both took fewer arguments before, and the old positional
calls now bind the wrong parameters at runtime rather than failing to build:

| Before | Now |
| --- | --- |
| `new Model(weights, tokenizer, quant, lang, rewrites)` | `new Model(weights, tokenizer, configJson, quant, lang, rewrites)` |
| `start_generation(voice, text, temperature)` -> token count | `start_generation(voice, text, temperature, seed)` -> chunk count |
| `generation_step()` until `undefined` | `next_chunk()` per chunk, `generation_step()` within it |

The seed was a hardcoded 42 and is now the caller's; one noise source covers every chunk, so
a seed fixes the whole utterance.

## Todo

- Voice cloning.
