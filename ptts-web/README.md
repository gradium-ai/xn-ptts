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

| dtype | needs | notes |
| --- | --- | --- |
| `f32` | — | baseline |
| `f16` | adapter reports WGSL `shader-f16` | errors rather than silently downgrading |
| `q8` | — | `q8_0` weights, f32 activations |
| `q8f16` | `shader-f16` | `q8_0` weights, f16 activations |

The q8 paths quantize from the f32 safetensors at load rather than reading a
pre-quantized GGUF, so they download the same file `f32` does and pay a
quantization pass on the GPU when the model is built. GGUF input is rejected with
a message saying so.

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

## Waveform

The canvas is laid out from the frame budget at `gen_start` and each frame is
drawn into its own slice as it arrives, so nothing is redrawn on the generation's
hot path. eos usually ends a run short of the budget, so the waveform is redrawn
once at the end against the length actually produced.

## Measured

Apple M5, Chrome headless, 94 frames (7.5 s of audio), 3 runs, phonon model:

| dtype | RTF | TTFA | frame p50 |
| --- | --- | --- | --- |
| f32 | 6.53x | 109 ms | 11.1 ms |
| f16 | 7.78x | 141 ms | 8.7 ms |

Both produce the same audio (rms 0.065, peak ~0.58, no non-finite samples), so
f16 is not silently degrading.

Browsers mask the adapter name, so the device reports as
`WebGPU (Other BrowserWebGpu)` rather than naming the GPU.

These are **not** comparable to `ptts-ws-server`'s numbers on the same machine
(4.13x f32, 5.20x f16). That server encodes each frame to s16, base64s it, wraps
it in JSON and pushes it through a websocket inside the measured loop; this page
hands a `Float32Array` straight to WebAudio. The two measure different pipelines,
not two WebGPU implementations.
