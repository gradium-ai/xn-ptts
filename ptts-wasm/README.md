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

## Todo

- Voice cloning.
