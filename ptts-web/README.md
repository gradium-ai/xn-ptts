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

## Why a server at all?

Nothing here is served *by* a server in the sense of doing work: inference is
entirely in the browser, and the server only hands over files. Any static file
host does, with no configuration:

```bash
cd pkg && python3 -m http.server 8790     # verified: full q8 run, start to finish
```

`server.js` exists for two conveniences, not because the app needs it: it maps
`/model` and `/voices` to a local directory so a run does not pull weights over
the network, and it serves https so a phone can reach it (WebGPU needs a secure
context). For a real deployment, put `pkg/` on GitHub Pages or any object store
and pick the HuggingFace source in the page -- then nothing is self-hosted at all.

What does **not** work is opening `pkg/index.html` from disk. A `file://` page has
an opaque origin (`null`), and CORS blocks everything the app is built from:

```
Access to fetch at 'file:///build.txt' from origin 'null' has been blocked by CORS
policy: Cross origin requests are only supported for protocol schemes: chrome,
chrome-extension, chrome-untrusted, data, http, https, isolated-app.
```

The same rule stops the ES module imports, the module worker, and the streaming
instantiation of the wasm. So it needs an `http://` or `https://` origin -- but
only an origin, not a backend.

## On a phone

```bash
make phone
```

That builds, generates a self-signed cert, and serves. The console prints the URL
to open, e.g. `https://192.168.68.55:8789`. Your phone has to be on the same wifi.

It must be **https**, and that is not a detail: WebGPU only runs in a secure
context. `localhost` counts as one automatically, but a LAN address over plain
http does not, so `http://192.168.x.x:8788` gives `navigator.gpu === undefined`
and the page cannot start at all. The self-signed cert is the cheapest way to get
a secure context on a LAN -- the phone shows a warning once ("Show Details" ->
"visit this website" on iOS Safari, "Advanced" -> "Proceed" on Android Chrome).

Pick **q8_0** on a phone: 136 MB rather than 317 MB to pull over wifi, and
correspondingly less GPU memory for the weights. Needs iOS 18+ / Safari 18+, or
Chrome 121+ on Android; older versions have no WebGPU.

### "WebGPU is exposed but the browser offers no adapter"

`navigator.gpu` existing does not mean there is an adapter. Android Chrome
blocklists WebGPU on many GPU and driver pairs, and when it does the API stays
exposed while every `requestAdapter()` returns null -- including
`forceFallbackAdapter`, and identically on the main thread and in a worker. That
is not fixable from the page; the device has to allow it.

On the phone: open `chrome://gpu` and read the WebGPU line, which states the
reason outright. Then try `chrome://flags/#enable-unsafe-webgpu` and
`chrome://flags/#enable-vulkan` set to Enabled (WebGPU on Android runs on
Vulkan), and fully relaunch Chrome. Battery saver can also disable it.

`/diag.html` probes all four request forms on both the main thread and in a
worker and prints a JSON summary to paste elsewhere. It is plain JS with no wasm,
so it answers in a second rather than after a 136 MB download. The main page runs
the same check before offering to load anything.

### If it hangs instead of erroring

macOS stealth mode (on by default with the firewall) silently drops packets to
ports with nothing listening, rather than refusing them. So a server that is not
running looks identical to a slow one: the other device just spins. Check in this
order:

1. `lsof -nP -iTCP -sTCP:LISTEN | grep 8789` -- is it actually up? Backgrounded
   shells kill it more often than you would expect; `make phone` in its own
   terminal is the reliable way.
2. Are you on the LAN address? The console also prints a `100.x` tailnet address,
   and a device without Tailscale has no route to it, so it hangs forever with no
   error. Use the `192.168.x` one.
3. `/usr/libexec/ApplicationFirewall/socketfilterfw --listapps | grep -A1 node`
   -- node needs "Allow incoming connections".
4. Some routers isolate wireless clients from each other ("AP isolation" /
   "client isolation"), which blocks this entirely.

`certs/` is gitignored -- it holds a private key, and the cert only covers the
addresses this machine had when it was generated. Re-run `make cert` after
changing networks.

If your phone is on the same Tailscale tailnet, `tailscale serve https:443 /
http://127.0.0.1:8788` gives a real certificate and no warning at all, but it
needs HTTPS enabled for the tailnet and the Tailscale app on the phone.

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

The fill follows the **audio clock**, not frame arrival. Generation runs several
times faster than realtime -- ~1.2 s of compute for 7.5 s of speech -- so a
waveform drawn as frames arrive is complete about six seconds before the listener
has heard any of it, which is indistinguishable from not animating at all.

So there are two layers. Generated audio is drawn dim as it arrives, which shows
generation progress; the portion actually heard is drawn bright over it, advancing
with `AudioContext.currentTime`, with a playhead line between them. `Player.played()`
is the clock for streaming playback, `playAll` returns one for "after run", and
with playback off there is no clock and the waveform simply completes with
generation.

The whole canvas is repainted from `requestAnimationFrame` off a single contiguous
sample buffer, so painting is tied to the display rather than to whenever a worker
message lands.

### Checking it

`scripts/shots.mjs` drives the page over CDP and samples the canvas while it runs.
Verdicts come from counting waveform pixels, not from screenshots: this headless
setup stops producing composited frames once the DOM settles, so screenshots go
byte-identical even while the canvas provably changes. The script prints both and
says which one is the verdict.

```bash
node server.js &
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --enable-unsafe-webgpu --no-first-run --no-sandbox \
  --autoplay-policy=no-user-gesture-required \
  --remote-debugging-port=9222 --window-size=1200,900 \
  --user-data-dir=/tmp/ptts-shots-cd about:blank &
WS=$(curl -s http://127.0.0.1:9222/json | python3 -c \
  "import json,sys; print([t for t in json.load(sys.stdin) if t['type']=='page'][0]['webSocketDebuggerUrl'])")
node scripts/shots.mjs "$WS" "http://127.0.0.1:8788/?auto=1&dtype=f16&iters=1&play=stream"
```

Measured, f16, 7.5 s of audio: `play=stream` fills 849 -> 6644 bright px over the
7.35 s of playback with dim reaching 0; `play=end` fills 0 -> 6643; `play=off`
completes with generation at 6734 bright px.

## Caching

`server.js` sends `no-cache` for everything in `pkg/` and a day of caching only
for `/model` and `/voices`. Caching the app shell means a rebuild is invisible to
a browser that already has the page, which looks exactly like a change that did
not work.
