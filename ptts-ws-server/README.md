# ptts-ws-server

Streaming Pocket TTS over a websocket, plus a small web app on `/` for driving it
and measuring it.

## The web app

`ptts-ws-server` serves a single-page app at `/` (embedded in the binary, source in
[`www/index.html`](www/index.html)). It reads `/api/info` for the backend label,
device name and voice list, then drives the same `/speech/tts` websocket any other
client would use: one `setup` (which conditions the voice), then a `text` + `flush`
round-trip per iteration.

Per run it reports **RTF** (audio produced per unit of wall time; above 1.0 is
faster than realtime), **TTFA** (delay before the first audio chunk), total time,
and the per-frame interval distribution — the cadence a streaming client sees.
Runs accumulate in a history table, aggregated per backend so a table spanning a
server restart does not average two different backends together.

The timings come from the server (`TtsReply::Stats`), measured from the moment the
text is flushed to the last encoded chunk. Model load and voice conditioning happen
once at setup and are excluded, since a server pays them once and then serves many
requests.

## Running

```bash
# WebGPU (wgpu -> Metal / Vulkan / DX12), f32
cargo run --release -p ptts-ws-server --features webgpu -- \
  --webgpu \
  --config      ../phonon-inference/model/config.json \
  --model       ../phonon-inference/model/model.safetensors \
  --voice-dir   ../phonon-inference/voices \
  --addr        127.0.0.1:8080
```

Then open <http://127.0.0.1:8080>.

### Backend flags

| Flag | Feature needed | Notes |
| --- | --- | --- |
| *(none)* | — | CPU, f32 |
| `--quant q8` | — | CPU, q8_0 weights. Also `q8_1`, `q8k`, `q6k`, `q5_0/1/k`, `q4_0/1/k` |
| `--webgpu` | `webgpu` | wgpu, f32 |
| `--webgpu --webgpu-dtype f16` | `webgpu` | needs adapter `shader-f16`; errors rather than silently downgrading |
| `--webgpu --quant q8` | `webgpu` | q8_0 weights on the GPU, quantized at load from the f32 safetensors |
| `--metal` | `metal` | f32 |
| `--vulkan` | `vulkan` | f32 |
| `--cuda` | `cuda` | bf16 |

`--model` overrides the `model.safetensors` / `model.q8.gguf` lookup next to
`--config`. The WebGPU q8_0 path quantizes from f32 weights at load, so point it at
the safetensors, not the gguf.

To compare backends, restart the server with a different flag and generate again —
the history table labels each row with the backend that produced it.

Set `XN_WEBGPU_PROFILE=1` to get xn's own per-kernel profile on stderr at exit.

## Websocket protocol

`/speech/tts`, JSON text frames. Client sends `setup` (`output_format`: `pcm`,
`wav`, `opus`, `pcm_<rate>`, `ulaw_8000`, `alaw_8000`), then `text`, then `flush`
or `end_of_stream`. Server replies `ready`, `audio` (base64), `stats`, `flushed`,
`end_of_stream`, or `error`. See [`src/protocol.rs`](src/protocol.rs).
