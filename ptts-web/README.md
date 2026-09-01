# ptts-web

Pocket TTS running **entirely in the browser** on xn's WebGPU backend. The
flow-matching LM and the Mimi decoder are WGSL compute shaders; no server does
any inference, it only serves files.

This is a static site: `pkg/` after a build is the whole deployable artifact.

## Build and run

```bash
cargo install wasm-bindgen-cli   # once, must match the wasm-bindgen dep
make build                       # -> pkg/
make serve                       # pkg/ at :8788, plus the local model dir
```

Then open <http://127.0.0.1:8788>. WebGPU needs a secure context; `localhost`
counts as one, so no TLS is needed locally.

`server.js` also maps `/model` and `/voices` to `../../phonon-inference/{model,voices}`
(override with `PTTS_MODEL_DIR` / `PTTS_VOICE_DIR`) so the page can be driven
without downloading weights from HuggingFace every time. The other source in the
picker is the public `kyutai/pocket-tts-without-voice-cloning` repo, which is what
a real deployment would use.

## Compute dtypes

| dtype | weights file | needs | notes |
| --- | --- | --- | --- |
| `f32` | `model.safetensors` | — | baseline |
| `f16` | `model.safetensors` | adapter reports WGSL `shader-f16` | fastest; errors rather than silently downgrading |
| `q8` | `model.q8.gguf` | — | `q8_0` weights, f32 activations |
| `q8f16` | `model.q8.gguf` | `shader-f16` | `q8_0` weights, f16 activations |

The dtype dictates the container. f32 and f16 need dense weights; the q8 dtypes
need blocks that are *already* quantized, because quantizing dense weights means
reading each one back to the host (`Q8Tensor::quantize`), and a browser cannot
block on a readback. Asking for a q8 dtype without a GGUF fails with that
explanation rather than hanging.

## Threads: one, and not configurable

There is no threads setting, because there is nothing for it to do. xn's threading
lives entirely in its CPU backend -- the WebGPU backend contains no rayon call --
and the wasm32 rustflags carry no `+atomics`, so `std::thread` cannot spawn and
`num_cpus` reports one core. The page states the fact in a badge
(`1 thread · 1 cpu · all compute on the GPU`) rather than offering a knob that
silently changes nothing.

There is deliberately no setter binding either: `xn::set_num_threads` writes
`RAYON_NUM_THREADS`, and `std::env::set_var` is unsupported on
`wasm32-unknown-unknown`, so calling it traps the module. That is worth guarding
in xn for any other wasm caller, but nothing here calls it.

## Headless check

The page drives itself when given `?auto=1`, and POSTs its result to `/report`:

```bash
PTTS_EXIT_ON_REPORT=1 node server.js &
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --enable-unsafe-webgpu --no-first-run --no-sandbox \
  --user-data-dir=/tmp/ptts-web-cd \
  "http://127.0.0.1:8788/?auto=1&dtype=f16&iters=3"
```

The server prints the JSON report and exits. Parameters: `dtype`, `iters`, `text`,
`base`. The report carries per-run timings plus rms/peak/non-finite counts over
the produced audio, so a run that "succeeds" while emitting silence is visible.
It also counts waveform pixels while frames are arriving, which is what shows the
waveform is drawn during the run rather than at the end of it.
Any failure — no `navigator.gpu`, a load error, a timeout — still reports, with
the checkpoints it reached, so a hang says where it stopped.

## How it avoids blocking

A browser delivers GPU completion through the event loop, so xn's blocking
readback would deadlock rather than merely stall. Two things make the model run
anyway:

* **Ops never read back.** They only record into xn's batch, so the whole op set
  goes through the ordinary synchronous `Backend` trait. Only readbacks are
  awaited, via `Device::tensor_to_vec`.
* **The step API returns tensors, not decisions.** `TTSModel::generate_step`
  returns `is_eos: bool`, which forces a readback mid-step, and the older latent
  API signals "first step" with a NaN-filled tensor that has to be read back to be
  noticed. `generate_step_parts` instead takes a `StepInput` and returns the raw
  eos logit, so a frame records its sampling *and* its Mimi decode before anything
  is read. The frame's two readbacks then resolve on one flush.

That second point helps the native WebGPU backend too, which is why
`ptts-ws-server` now uses the same API -- though by less than the browser case
might suggest. Measured there, same text and 3 runs each: 3.80x -> 4.13x RTF and
18.1 ms -> 16.6 ms per frame, so about 9%. The readbacks it removes are small; the
per-frame cost is dominated by dispatch count, not by round trips.

The third readback, the one that blocked q8, was at load: `Q8Tensor::quantize`
pulled every weight back to quantize it on the host. `Q8Tensor::from_q8_0` now
takes the blocks straight from a GGUF instead, so nothing but the upload touches
the device. That is an xn change, and it made native loads about 2x faster as
well.

## Waveform

The canvas is laid out from the frame budget at `gen_start` and each frame is
drawn into its own slice as it arrives, so nothing is redrawn on the generation's
hot path. eos usually ends a run short of the budget, so the waveform is redrawn
once at the end against the length actually produced.

Frames are queued and flushed from `requestAnimationFrame` rather than painted
straight out of the worker's message handler. They arrive ~13 ms apart against a
16.7 ms refresh, and drawing from the handler only guarantees the canvas *bitmap*
is updated -- whether that reaches the screen before the run ends is up to the
compositor, and in practice it did not. Note that reading the canvas back
(`getImageData`) cannot tell the two apart: it sees the bitmap either way. The
check that can is `scripts/shots.mjs`, which drives the page over CDP and
compares composited screenshots.

## Measured

Apple M5, Chrome headless, 94 frames (7.5 s of audio), 3 runs, phonon model:

| dtype | container | RTF (mean of 4) | TTFA | frame p50 | load | download |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | safetensors | 6.21x | 115 ms | 11.5 ms | 432 ms | 317 MB |
| f16 | safetensors | 8.18x | 100 ms | 8.6 ms | 213 ms | 317 MB |
| q8 | gguf | 8.40x | 111 ms | 8.4 ms | 176 ms | 136 MB |
| **q8f16** | gguf | **8.56x** | 111 ms | 8.0 ms | 194 ms | 136 MB |

All four produce the same audio (rms 0.064-0.066, peak ~0.57, no non-finite
samples), so neither f16 nor q8 is silently degrading.

**Use `q8f16`.** It is at least as fast as f16 and less than half the download.
Do not read too much into the f16-vs-q8 ordering, though: an earlier sweep on the
same machine put q8 at 7.11x and q8f16 at 7.89x, below f16, and q8's run-to-run
spread has been as wide as 4.69x-8.55x. Something about the quantized path is
sensitive to state this benchmark does not control -- see the end-to-end q8
regression documented at the top of xn's `webgpu_backend/quantization.rs`, where
quantized layers slow down unrelated f32 work in later submits. Treat f16 and q8
as roughly equal on throughput and pick on size.

What q8 unambiguously buys: 136 MB instead of 317 MB, ~90 MB of weight VRAM
instead of ~340 MB, and the fastest load. For a page a stranger opens, that
matters more than a throughput tie.

Browsers mask the adapter name, so the device reports as
`WebGPU (Other BrowserWebGpu)` rather than naming the GPU.

These are **not** comparable to `ptts-ws-server`'s numbers on the same machine
(4.13x f32, 5.20x f16). That server encodes each frame to s16, base64s it, wraps
it in JSON and pushes it through a websocket inside the measured loop; this page
hands a `Float32Array` straight to WebAudio. The two measure different pipelines,
not two WebGPU implementations.

To run it:

```bash
node server.js &
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --enable-unsafe-webgpu --no-first-run --no-sandbox \
  --remote-debugging-port=9222 --window-size=1200,900 \
  --user-data-dir=/tmp/ptts-shots-cd about:blank &
WS=$(curl -s http://127.0.0.1:9222/json | python3 -c \
  "import json,sys; print([t for t in json.load(sys.stdin) if t['type']=='page'][0]['webSocketDebuggerUrl'])")
node scripts/shots.mjs "$WS" "http://127.0.0.1:8788/?auto=1&dtype=f16&iters=3"
```

## Caching

`server.js` sends `no-cache` for everything in `pkg/` and a day of caching only
for `/model` and `/voices`. Caching the app shell means a rebuild is invisible to
a browser that already has the page, which looks exactly like a change that did
not work.
