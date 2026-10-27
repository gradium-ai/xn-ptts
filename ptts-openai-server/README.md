# ptts-openai-server

A speech server for Phonon compatible with OpenAI's text-to-speech API: `POST /v1/audio/speech`, so any client with an "OpenAI TTS" setting and a custom base URL works unchanged. It can also serve the [Wyoming protocol](#home-assistant), which Home Assistant uses for local voice services. For streaming text in and audio out over one WebSocket connection, use [`ptts-ws-server`](../ptts-ws-server) instead.

## Docker

On first start the server downloads the default model and the catalog voices in its language, and logs nothing more until they are loaded:

```bash
docker run -p 8880:8880 -e PTTS_LANG=en -e PTTS_QUANT=q8 ghcr.io/gradium-ai/ptts-openai-server
```

To serve a local checkpoint instead, mount its folder and point `PTTS_MODEL` at it:

```bash
docker run -p 8880:8880 -v "$MODEL_DIR:/models:ro" \
  -e PTTS_MODEL=/models -e PTTS_LANG=en -e PTTS_QUANT=q8 \
  ghcr.io/gradium-ai/ptts-openai-server
```

The image contains the server and no model weights. It is built for `linux/amd64` and `linux/arm64`. On an x86 CPU without AVX2 it runs a slower build made for such CPUs, and says so when it starts. Every model supplies its own `config.json` and `tokenizer.json`.

To serve another model from HF, set `PTTS_MODEL=hf://OWNER/MODEL@COMMIT_SHA`, for example `PTTS_MODEL=hf://gradium/phonon-40m-fr` with `PTTS_LANG=fr`. Set `HF_TOKEN` for a gated or private repo. The Phonon 40m models are gated: accept the terms on the model's Hugging Face page first. The image uses `/home/phonon/.cache/huggingface` as its writable `HF_HOME`. The Hub Compose file mounts a named volume there, so downloads survive container replacement.

The Hub Compose file serves the default model, or the one `PTTS_MODEL` names:

```bash
export PTTS_LANG=en
# Optional: export PTTS_MODEL=hf://OWNER/MODEL@COMMIT_SHA, and HF_TOKEN for a gated or private repo.
docker compose -f ptts-openai-server/compose.hub.yaml up
```

Without `@COMMIT_SHA` the repo's `main` is used, so the cache retains files but does not freeze the checkpoint version. Set a commit for a fixed checkpoint.

The cache remains after `docker compose down`; `down --volumes` deletes it. Use `PTTS_IMAGE=ghcr.io/gradium-ai/ptts-openai-server:<version>` to select a release in either Compose file. First startup downloads and loads the checkpoint; follow progress with `docker compose -f ptts-openai-server/compose.hub.yaml logs -f`.

Every flag below has an environment variable, so the image is configured with `-e`. Additional voices can be mounted with `-v ./voices:/voices:ro -e PTTS_VOICE_DIR=/voices`. The server also serves the voices of a voice catalog that are in its language, `PTTS_LANG`. The catalog is [`gradium/phonon-voices`](https://huggingface.co/gradium/phonon-voices), downloaded on first start and cached, unless `PTTS_VOICE_CATALOG` names another: an `hf://owner/name[@revision]` repo, or a local JSONL file with one `{"name", "filename"}` object per line and `filename` relative to the catalog, such as `/voices/voices.jsonl`. An empty `PTTS_VOICE_CATALOG` serves only the checkpoint's voices. [compose.yaml](compose.yaml) mounts the folder named by `MODEL_DIR`:

```bash
docker compose -f ptts-openai-server/compose.yaml up
```

To build the image yourself, from the repository root: `docker build -f ptts-openai-server/Dockerfile -t ptts-openai-server .`

## Build and run

It needs two system libraries: libopus, and LAME 3.99 or later for MP3.

```bash
sudo apt-get install libopus-dev libmp3lame-dev pkg-config   # Debian, Ubuntu
brew install opus lame pkg-config                            # macOS

cargo run --release -p ptts-openai-server -- --model "$MODEL_DIR" --quant q8 --lang en
```

`--model` (`PTTS_MODEL`) names a checkpoint folder (or a `config.json` in one), or a Hugging Face repo as `hf://owner/name` or `hf://owner/name@revision`. Without it, the server loads the default model. The older `PTTS_CONFIG` and `PTTS_REVISION` still work but are deprecated.

It listens on `0.0.0.0:8880` (`--addr` to change it). There is no authentication, and each request can ask for up to 4096 characters of speech. On a machine others can reach, bind to `--addr 127.0.0.1:8880` or put it behind a proxy that checks access. `--lang` is required: it picks how numbers and symbols are spelled out. `--device auto` uses the GPU backend the binary was built with, if any; quantized weights such as `--quant q8` run on the CPU only. A checkpoint whose config lists conditioners, such as `duration_delta`, takes their values with `--condition duration_delta=0.3` (repeatable, or `PTTS_CONDITION=duration_delta=0.3,num_speakers=1` in the environment); those not given take their defaults. `--help` lists the rest.

By default, one speech generation runs at a time. `--max-concurrent-requests N` (or `PTTS_MAX_CONCURRENT_REQUESTS=N`) sets a positive limit. Excess speech requests immediately receive HTTP 429 with OpenAI error code `server_busy`; retry when an active generation finishes. Slots are released after generation and worker cleanup, including after a disconnect or failure. Health, model, and voice endpoints remain available while generation is busy.

## The OpenAI-compatible API

```bash
curl http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "Hello from Phonon.", "voice": "default"}' \
  -o hello.mp3
```

Or with the OpenAI SDK, pointed at the server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="unused")
client.audio.speech.create(model="tts-1", voice="default", input="Hello.").write_to_file("hello.mp3")
```

| Field | What the server does |
|---|---|
| `input` | Required, up to 4096 characters. Long text is split at sentence ends. |
| `voice` | A voice of this server, in any case (`GET /v1/audio/voices` lists them), or `default`, which is also what a missing `voice` means. Any other name, such as OpenAI's own `alloy`, is a 400 unless it is set as an [alias](#openai-clients). |
| `response_format` | `mp3` (the default), `opus`, `wav` or `pcm` (headerless 16-bit mono at 24 kHz). `aac` and `flac` are a 400. |
| `speed` | Only `1.0`: the model has no rate control yet, so other values are a 400. With `--ignore-speed`, any value from `0.25` to `4.0` is accepted and ignored. Values outside that range are always a 400. |
| `model`, `instructions`, `stream_format` | Accepted and ignored. The reply is always the audio bytes, streamed as they are generated. |

Errors use OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`, so client SDKs report them as usual. `GET /v1/models` names the loaded checkpoint and lists the `languages` its config declares. `GET /v1/audio/voices` answers `{"voices": [...], "aliases": {"alias": "voice"}}`: aliases are not in `voices`, so a client that offers that list offers only real voices. `GET /health` answers `{"status": "ok"}`. There is no authentication: an `Authorization` header is ignored. A failure after the audio has started can only cut the response short. Audio buffering is bounded through the model workers and HTTP response. A slow reader pauses generation; closing the response cancels generation and releases its workers. A client that stops reading for 30 seconds has its response ended and its slot freed. A streamed `wav` has no length in its header, which a few strict parsers reject; `mp3` or `pcm` suit those.

## OpenAI clients

Some clients only offer OpenAI's own voice names, such as `alloy` or `nova`, and some always send a `speed`. By default the server rejects both, so a mistake is not hidden. Three settings make such clients work, and all are off unless set:

- `--openai-voices VOICE` (`PTTS_OPENAI_VOICES`) accepts every OpenAI voice name as `VOICE`, a voice of this server or `default`. A checkpoint voice with the same name as an OpenAI voice keeps its own name.
- `--voice-alias ALIAS=VOICE` (`PTTS_VOICE_ALIAS`, comma-separated) adds one name; repeatable. It wins over `--openai-voices` for the same name. An alias cannot be the name of a voice, and a voice it names must exist, or the server does not start.
- `--ignore-speed` (`PTTS_IGNORE_SPEED=true`) accepts a `speed` from `0.25` to `4.0` and speaks at the normal rate.

Aliases match in any case, like voice names. A name that is neither a voice nor an alias is still a 400. With Docker:

```bash
docker run -p 8880:8880 -v "$MODEL_DIR:/models:ro" \
  -e PTTS_MODEL=/models -e PTTS_LANG=en -e PTTS_QUANT=q8 \
  -e PTTS_OPENAI_VOICES=default -e PTTS_VOICE_ALIAS=nova=VOICE \
  -e PTTS_IGNORE_SPEED=true \
  ghcr.io/gradium-ai/ptts-openai-server
```

Here every OpenAI voice is the default voice, except `nova`, which is `VOICE`, one of `GET /v1/audio/voices`.

## Clients

- **Open WebUI:** Admin Settings, Audio, Text-to-Speech engine `OpenAI`, API base URL `http://<host>:8880/v1`, any API key. Set the voice to `default` or a voice from `/v1/audio/voices`.
- **Home Assistant:** use the [Wyoming protocol](#home-assistant), which needs no custom integration. The [sfortis/openai_tts](https://github.com/sfortis/openai_tts) integration also works: Custom endpoint, URL `http://<host>:8880/v1/audio/speech`, model `tts-1`. The voice list is read from `/v1/audio/voices`.
- **LiveKit Agents:** `openai.TTS(base_url="http://<host>:8880/v1", api_key="unused", model="tts-1", voice="default")`.
- **SillyTavern:** TTS provider "OpenAI Compatible", endpoint `http://<host>:8880/v1/audio/speech`, voices `default` or one from `/v1/audio/voices`. Leave the speed at 1, or start the server with `--ignore-speed`.

## Home Assistant

Home Assistant talks to local speech services over the Wyoming protocol. Set `--wyoming-addr` (or `PTTS_WYOMING_ADDR`) and the server answers it on that address as well as serving the HTTP API:

```bash
docker run -p 10200:10200 -e PTTS_LANG=en -e PTTS_QUANT=q8 \
  -e PTTS_WYOMING_ADDR=0.0.0.0:10200 \
  ghcr.io/gradium-ai/ptts-openai-server
```

Then, in Home Assistant, add the **Wyoming Protocol** integration with this machine's address and port `10200`. Phonon appears as a text-to-speech engine that you can pick in a voice assistant. On Home Assistant OS, the [Phonon app](https://github.com/gradium-ai/phonon-home-assistant) runs the server for you, and Home Assistant finds it on its own.

- **Language.** Every voice is listed in the `--lang` language, which must not be `none`: Home Assistant names a voice in each request but never a language. For a second language, run a second server.
- **Streaming.** Home Assistant sends an assistant's reply while it is still being written. The server speaks it a sentence at a time, so speech starts with the first sentence, not after the last.
- **Busy server.** Wyoming requests share the HTTP API's `--max-concurrent-requests` slots. They wait for a free slot instead of failing.
- **Voices.** Home Assistant offers the voices the server lists, so [aliases](#openai-clients) apply to the HTTP API only.
- **Access.** Like the HTTP API, the Wyoming port has no authentication. Publish it only on a network you trust.
