# Phonon WebSocket server

Run it with the default model, downloaded on first start:

```bash
cargo run --release -p ptts-ws-server -- --quant q8 --lang en
```

`--model` loads another one: a checkpoint folder, or a Hugging Face repo as `hf://owner/name`, such as `--model hf://gradium/phonon-40m-fr` with `--lang fr`. The Phonon 40m models are gated: accept the terms on the model's Hugging Face page, then set `HF_TOKEN` or run `hf auth login`.

The server listens on `0.0.0.0:8080`, with WebSocket connections at `/speech/tts`. `--addr` changes the address. The checkpoint supplies its config, tokenizer, weights, and voices; `--model hf://owner/name@revision` selects a Hub revision. The server also offers the voices in its `--lang` from the voice catalog [`gradium/phonon-voices`](https://huggingface.co/gradium/phonon-voices), downloaded on first start, or from the one `--voice-catalog` names: an `hf://` repo or a local `voices.jsonl`. `--voice-catalog ''` offers only the checkpoint's voices. Building requires libopus (`apt-get install libopus-dev` or `brew install opus`).

By default, one WebSocket session is accepted at a time. `--max-concurrent-requests N` sets a positive limit. Excess connections receive HTTP 429 before the WebSocket upgrade, with a message asking the client to retry after an active session closes.

A session retains its primed voice and KV cache between messages. Its slot is held for the entire connection, and released after the session and generation workers stop. Close connections when finished so other clients can connect. A session that sends no message for five minutes while no audio is being generated gets an error with code 408 and is closed. A client that stops reading for 30 seconds is disconnected. The limit applies to connections, including those awaiting their first setup message.

## Protocol

Messages are JSON text frames.

1. Send `{"type": "setup", "output_format": "pcm", "voice": "Zoey"}` first. `output_format` is `pcm` (16-bit mono at the model's rate), `pcm_8000`, `pcm_16000`, `pcm_22050`, `pcm_24000`, `pcm_44100`, `pcm_48000`, `ulaw_8000` (or `mulaw_8000`), `alaw_8000`, `wav` or `opus`. Leave out `voice` for the default voice. The server answers `{"type": "ready", "sample_rate": ..., "frame_size": ..., ...}`, or an `error`.
2. Send text with `{"type": "text", "text": "..."}`. Text is buffered, up to 4096 characters.
3. `{"type": "flush", "flush_id": 1}` speaks the buffered text. The server answers with `audio` messages, then `{"type": "flushed", "flush_id": 1}`. The session stays open.
4. `{"type": "end_of_stream"}` speaks the rest, answers `{"type": "end_of_stream"}` and ends the session.

Audio arrives as `{"type": "audio", "audio": "<base64>", "start_s": ..., "stop_s": ..., "stream_id": ...}`. Errors are `{"type": "error", "message": ..., "code": ...}`, with codes 400, 404, 408, 500 and 501.
