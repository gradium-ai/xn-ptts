# ptts for Python

Python bindings for the [Phonon Rust runtime](https://github.com/gradium-ai/xn-ptts).

To try a q8 checkpoint with [uv](https://docs.astral.sh/uv/), run:

```bash
uvx ptts --model /path/to/model --lang en --quant q8 "Hello world" -o out.wav
```

`uvx` handles the isolated Python environment and package dependencies. Until the stable release, use `uvx --prerelease allow ptts` to run the preview. Model files are supplied separately; `--model` accepts a local checkpoint directory or a Hugging Face repo as `hf://owner/name`.

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

`model`, the first argument, names a local directory, a `config.json`, or a Hugging Face repo as `"hf://owner/name"` or `"hf://owner/name@revision"`. Left out, it is `ptts.DEFAULT_MODEL`. Local paths download nothing. Rust examples, Python, and both servers use the [shared checkpoint resolver](https://github.com/gradium-ai/xn-ptts/blob/main/ptts/src/checkpoint.rs). Every checkpoint supplies its own `config.json` and `tokenizer.json`. q8 prefers `model.q8.gguf`; other formats prefer `model.safetensors`. Voices come from `voices/` or `embeddings/`, plus `default-voice.safetensors` as `default`. Hub downloads use the revision after `@` for all files. `config=` and `revision=` still work but are deprecated.

`lang` is required: `en`, `fr`, `de`, `es` or `pt` picks how numbers, symbols and abbreviations are spelled out; `none` uses the text as written.

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

`tts.voices` lists the voices that were found, and `tts.languages` the languages the checkpoint's config says it was trained for. When no voice is given, the checkpoint's configured default is used, then `default` if present, then the first by name. Pass `voice="name"` to any speech method to select one. A name the checkpoint does not have is looked up in the voice catalog, `ptts.DEFAULT_VOICES` unless `voice_catalog=` names another, by name in any case and with or without accents, or by its voice ID. Only that voice is downloaded, on first use, and then joins `tts.voices`. `ptts.catalog_voices()` lists the catalog without loading a model.

## Command line

The package also provides the `ptts` command:

```bash
python -m ptts --lang en "Hello world" -o out.wav
```

`--model` loads another model, as `TTS`'s first argument does. `--voice` selects a voice of the checkpoint or the catalog, and `--list-voices` prints the catalog's voices. Run with `--help` for the remaining options. The package ships type stubs and `py.typed`.

## Wheels

Released wheels are `cp39-abi3`, so one per platform covers every CPython from 3.9 on, and a
new CPython release needs no new wheel. Free-threaded CPython and PyPy cannot load an abi3
extension, so those two install from the sdist and compile the Rust runtime, which needs a
Rust toolchain and takes a few minutes.

## License

MIT or Apache-2.0, at your option. The model weights are not part of the package and come with their own license.
