# ptts-web

Three tabs, all inference in the browser on xn's WebGPU backend:

| Tab | What it is |
| --- | --- |
| **Agent** | ASR -> LLM -> TTS. Always listening; talk and it talks back. Both models' quantization is picked here. |
| **TTS** | Pocket TTS on its own, with the timings below. |
| **ASR** | Streaming ASR on its own, with per-frame timings. |

ASR and TTS run in **separate workers**, so each gets its own thread and its own
WebGPU device. Only the LLM leg leaves the machine: it is an OpenRouter call,
proxied by `server.js` so the API key stays out of the page.

```bash
export OPENROUTER_API_KEY=...        # the Agent tab needs this; the other two do not
make serve
```

The LLM is `liquid/lfm-2.5-2.6b:free` by default (`PTTS_LLM_MODEL` overrides it).
Being free, it shares an upstream pool and returns 429 often; the proxy honours
`Retry-After` and retries up to three times, and the page shows the elapsed wait
rather than looking hung. A turn that waits 40 s for the LLM is the free tier,
not the models.

Measured in headless Chrome on an Apple M5, one turn end to end:

| Leg | Time |
| --- | --- |
| ASR (5.4 s of speech) | 2.0 s, 2.71x realtime |
| LLM (free tier, when not rate-limited) | 1.0 s |
| TTS (4.9 s of speech, q8_0/f16) | 0.8 s, 6.49x realtime |
| **Whole turn** | **4.1 s** |

That is with the turn closed by the silence detector, not a button. A rate-limited
LLM turn measured 43 s instead of 1 s, which is the free tier rather than
anything local.

## What streams

Both legs are rendered as they happen, not at the end of the turn.

* **Transcription.** The ASR posts each word as the model closes it, and the
  Agent tab shows the utterance so far in its own bubble, replaced by the final
  text when the turn closes. An abandoned turn removes it.
* **The reply.** The bubble starts empty and its words are revealed against
  `AudioContext.currentTime`, so the text tracks what is being said rather than
  appearing whole before the audio starts. Same idea as the TTS tab's waveform.

That reveal loop also decides when to listen again. `gen_done` means generation
finished, and generation runs several times faster than realtime, so resuming
there would put the microphone back on while the agent is still talking. It
resumes when the audio has actually finished instead.

## Turn-taking

The Agent tab has no push-to-talk. The microphone stays open once started, and
each 80 ms frame's RMS decides where an utterance ends: speech has to be heard
for 3 frames before one opens, and ~720 ms of silence closes it and sends the
turn. Three points that are easy to get wrong and are handled:

* **Pre-roll.** The gate needs a few frames to trip, so the frames just before it
  are held and fed in when it does. Without them "Hello, this is a test" arrives
  as "this is a test" -- which is exactly what happened before it was added.
* **False starts.** A cough trips the gate without becoming speech. If the
  hangover expires before `minSpeech` frames are heard, the utterance is dropped
  and the ASR state reset, rather than the session waiting forever for a turn.
* **The agent's own voice.** Frames arriving while it is transcribing or
  speaking are dropped, so it does not transcribe itself. That also means no
  barge-in: it cannot be interrupted mid-sentence.

The thresholds are in the `VAD` object at the top of the agent section, in frames
rather than milliseconds since the frame is the model's unit.

## Weights

`server.js` maps `/model` and `/voices` to the local TTS checkpoint and `/asr` to
the ASR one (`gr4d/asr-23b5a198.500` as `huggingface-cli` leaves it; override with
`PTTS_ASR_DIR`). The ASR is ~960 MB across two safetensors, the TTS ~317 MB.

## ASR on WebGPU: what was slow

The ASR is a Mimi encoder feeding a causal LM, one 80 ms frame at a time. Two
values per frame have to reach the host -- the codebook indices the LM consumes
and the token it sampled -- so `ptts::asr` splits the step in two and the browser
awaits each, the same shape the TTS path uses.

Native WebGPU first ran this at **0.49x realtime**. Almost all of it was
`Tensor::stack` on the per-codebook index tensors: WGSL has no 64-bit integer, so
xn's WebGPU backend computes in float dtypes only and every i64 op falls back to
a host round trip. Stacking cost one such trip per codebook per frame, ~70 in
all, each forcing a flush.

`asr_quantizer` now returns the indices unstacked and the loop stays on-device
(`decode` consumes the index tensor directly), which took native to 0.80x. The
browser reaches **2.71x** on the same code, because awaiting a readback avoids
the blocking `poll(Wait)` that costs ~2.7 ms whether or not the work is done.

The remaining native gap is that floor, 33 flushes per frame. Fixing it properly
means giving the WebGPU backend an i64 `copy2d` -- which needs no arithmetic,
only data movement, so it could dispatch the u32 kernel over doubled strides.
That is a change in xn, not here.


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

## Proving a page really runs on the GPU

"Uses WebGPU" is easy to claim and easy to get wrong in both directions, so
`scripts/gpu-calls.mjs` counts the calls. It drives a page over CDP and wraps the
WebGPU entry points, reporting adapters, devices, shader modules, pipelines,
queue submissions and workgroup dispatches per target.

```bash
node scripts/gpu-calls.mjs "$(curl -s http://127.0.0.1:9233/json/version \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["webSocketDebuggerUrl"])')" \
  https://example.com/some-page/
```

Two things it has to get right, both of which produced confident wrong answers
first:

* **Attach to the worker.** A model normally runs in a dedicated worker, and those
  attach under their *page's* session, not the browser's. Watching only the
  browser-level targets reports zeros from contexts that never touch the GPU.
* **Wrap instances, not prototypes.** With `waitForDebuggerOnStart` the worker is
  paused before its globals are populated, so `GPUAdapter.prototype` does not
  exist yet and patching it silently no-ops -- which looks exactly like a page
  that never used the GPU. Wrapping what `requestAdapter` returns cannot miss,
  since every device, queue, encoder and pass descends from it.

Sanity check on the output: dispatches should be in the tens of thousands for an
utterance, and dispatches/submit should land near xn's batch size (~165-190).
Single-digit numbers mean the counters are watching the wrong context.

## Caching

`server.js` sends `no-cache` for everything in `pkg/` and a day of caching only
for `/model` and `/voices`. Caching the app shell means a rebuild is invisible to
a browser that already has the page, which looks exactly like a change that did
not work.
