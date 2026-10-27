# ptts for Python

Python bindings for the [Phonon Rust runtime](https://github.com/gradium-ai/xn-ptts).

To try it with [uv](https://docs.astral.sh/uv/), run:

```bash
uvx ptts --lang en "Hello world" -o out.wav
```

`uvx` handles the isolated Python environment and package dependencies. The first run downloads the default model, Phonon 31m, and caches it. `--model` accepts a local checkpoint directory or a Hugging Face repo as `hf://owner/name`, such as `hf://gradium/phonon-40m-fr` with `--lang fr`.

To use the Python API in your own environment, install the package:

```bash
pip install ptts
```

The wheel contains no model weights. Without a model argument, `TTS` downloads the default model on first use and caches it:

```python
import ptts

tts = ptts.TTS(lang="en")
print(tts.voices)
tts.save("out.wav", "Hello world")
```

For a source install, use `uv add /path/to/xn-ptts/ptts-pyo3` or `pip install /path/to/xn-ptts/ptts-pyo3`. Source builds need Rust installed; supported CPython wheels do not. See the [development guide](https://github.com/gradium-ai/xn-ptts/blob/main/docs/development.md) for repository build commands.

`model`, the first argument, names a local directory, a `config.json`, or a Hugging Face repo as `"hf://owner/name"` or `"hf://owner/name@revision"`. Left out, it is `ptts.DEFAULT_MODEL`. Local paths download nothing. Gated repos, such as the Phonon 40m models, need their terms accepted on the model's Hugging Face page, then `HF_TOKEN` set or `hf auth login`. Rust examples, Python, and both servers use the [shared checkpoint resolver](https://github.com/gradium-ai/xn-ptts/blob/main/ptts/src/checkpoint.rs). Every checkpoint supplies its own `config.json` and `tokenizer.json`. q8 prefers `model.q8.gguf` and q4k `model.q4k.gguf`; other formats prefer `model.safetensors`. Voices come from `voices/` or `embeddings/`, plus `default-voice.safetensors` as `default`. Hub downloads use the revision after `@` for all files. `config=` and `revision=` still work but are deprecated.

`lang` is required: `en`, `fr`, `de`, `es` or `pt` picks how numbers, symbols and abbreviations are spelled out; `none` uses the text as written. `ptts.normalize("It costs $5.", lang="en")` shows that reading, "It costs 5 dollars.", without loading a model, and takes the same `rewrites`.

## Speech and voices

Reuse `tts` across requests. Use `synth` for an array, `save` for a WAV, or `stream` for chunks as they are decoded:

```python
voice = tts.voices[0]
pcm = tts.synth("Hello", voice=voice)  # 1-D float32 NumPy array
seconds = tts.save("out.wav", "Hello", voice=voice)
with tts.stream("A longer sentence.", voice=voice) as audio:
    for chunk in audio:
        print(chunk.shape)  # process each PCM chunk as it arrives
```

`tts.sample_rate` is the PCM sample rate; `save` writes a mono 16-bit WAV and returns its duration. Leaving the `with` block stops a stream early and waits for its workers to finish. Streaming keeps bounded audio and latent buffers: leaving a stream unread pauses generation once they fill, and consuming chunks lets it resume.

`tts.voices` lists the voices that were found, and `tts.languages` the languages the checkpoint's config says it was trained for. When no voice is given, the checkpoint's configured default is used, then `default` if present, then the first by name. Pass `voice="name"` to any speech method to select one. A name the checkpoint does not have is looked up in the voice catalog, by name in any case and with or without accents, or by its voice ID. Only that voice is downloaded, on first use, and then joins `tts.voices`. The catalog is `ptts.DEFAULT_VOICE_CATALOG`, [`gradium/phonon-voices`](https://huggingface.co/gradium/phonon-voices), unless `voice_catalog=` names another: an `hf://owner/name[@revision]` repo, or a local `voices.jsonl` or a folder holding one. `tts.add_voice("NAME", "voice.safetensors")` adds a voice file, such as one from the Gradium Studio scripts. `ptts.catalog_voices()` lists it without loading a model, and `ptts.catalog_voices(catalog)` another one. To change the speaking rate, pass `conditions={"duration_delta": -0.3}` to speak faster, or a positive value to speak slower.

## Command line

The package also provides the `ptts` command:

```bash
python -m ptts --lang en "Hello world" -o out.wav
```

`--model` loads another model, as `TTS`'s first argument does. `--voice` selects a voice of the checkpoint or of `--voice-catalog`, and `--list-voices` prints the catalog's voices. Run with `--help` for the remaining options. The package ships type stubs and `py.typed`.

## Wheels

Linux and Windows x86_64 wheels target x86-64-v3 (AVX2, FMA and F16C); older x86 CPUs need a source build. The macOS x86_64 wheel does not need AVX.

Released wheels are `cp39-abi3`, so one per platform covers every CPython from 3.9 on, and a
new CPython release needs no new wheel. Free-threaded CPython and PyPy cannot load an abi3
extension, so those two install from the sdist and compile the Rust runtime, which needs a
Rust toolchain and takes a few minutes.

## License

MIT or Apache-2.0, at your option. The model weights are not part of the package and come with their own licenses: Phonon 31m is under Apache 2.0, and Phonon 40m and Phonon 40m multilingual are under CC BY-NC 4.0. See each model's page on Hugging Face.
