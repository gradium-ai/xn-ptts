# ptts

Text to 24 kHz speech, on device. Python bindings for [Pocket TTS][repo], a Rust runtime for
the [`kyutai/pocket-tts`][model] model.

```bash
uvx ptts --lang en "Hello world" -o out.wav   # nothing to install
pip install ptts                      # or keep it around
```

```python
import ptts

tts = ptts.TTS(lang="en")
tts.save("out.wav", "Hello world")
```

## No PyTorch

The entire runtime — the flow-matching language model, the Mimi codec, the tokenizer, the
resampler — is a Rust extension inside this wheel. There is nothing to install alongside it.

| | `ptts` | others |
|---|---|---|
| Runtime dependencies | `numpy` | `torch`, `transformers`, and a phonemizer binary |
| Install size | a wheel and a checkpoint | ~2 GB before the checkpoint |
| Python versions | 3.9+ | usually capped two releases back |
| System packages | none | `espeak-ng` or `phonemizer`, per platform |

## Using it

```python
import ptts

tts = ptts.TTS(lang="en")                 # downloads the checkpoint on first use
print(tts.voices)                         # ['alba', 'azelma', 'cosette', ...]

pcm = tts.synth("Hello", voice="marius")  # 1-D float32 numpy array
seconds = tts.save("out.wav", "Hello")    # straight to a mono 16-bit WAV

for chunk in tts.stream("A longer piece of text."):
    play(chunk)                           # audio as the decoder produces it
```

Streaming is cancellable — leave the block and the worker threads stop:

```python
with tts.stream(text) as audio:
    for chunk in audio:
        if user_interrupted():
            break
```

Ctrl-C works during a generation, not only between them.

### Language

`lang` is required and keyword-only. Text is normalized before it is tokenized, and the spoken
forms of `@`, `+` and `=` differ per language, so there is nothing safe to default to:

```python
ptts.TTS(lang="en")     # en, fr, de, es, pt
ptts.TTS(lang="none")   # hand the text to the tokenizer as written
```

### Voices

Bundled voices come with the checkpoint. To clone one, pass about ten seconds of speech as
float32 PCM at `tts.voice_prompt_sample_rate` — no transcript needed:

```python
tts.clone_voice("me", my_pcm)
tts.save("out.wav", "Now in my voice.", voice="me")
```

### Choosing a checkpoint and a backend

```python
ptts.TTS(lang="en", config="kyutai/pocket-tts")   # a Hugging Face repo id
ptts.TTS(lang="en", config="model/config.json")   # a local checkpoint
ptts.TTS(lang="en", device="cuda")                # see ptts.available_devices()
ptts.TTS(lang="en", quant="q8_0")                 # smaller and faster on CPU
```

Quantized weights are CPU-only.

## Command line

The wheel installs a `ptts` command, so `uvx ptts` and `pipx run ptts` need no install step.
`python -m ptts` runs the same thing.

```bash
ptts --lang en "hello world" -o out.wav
ptts --lang en --list-voices
ptts --lang fr "bonjour" -v marius -q q8_0 -o out.wav
ptts --help
```

The first run downloads the checkpoint into the Hugging Face cache; later ones do not.

## Errors

Failures raise the exception their class calls for, so `except` can be specific:

| Exception | Cause |
|---|---|
| `ValueError` | a bad argument — an unknown weight format, a mis-shaped array |
| `LookupError` | an unknown voice, or a checkpoint file that is not there |
| `NotImplementedError` | a backend this wheel was not built with, or cloning on a checkpoint without a speaker encoder |
| `PermissionError` | a gated Hugging Face repo — the message says how to authenticate |
| `RuntimeError` | anything else |

## Types

The package ships `py.typed` and complete stubs, so editors and `mypy` see the full API.

## Licence

The Python package and the Rust runtime are MIT OR Apache-2.0. The model weights are published
separately by Kyutai under CC-BY-4.0 with an acceptable-use agreement; see the [model card][model].

[repo]: https://github.com/gradium-ai/xn-ptts
[model]: https://huggingface.co/kyutai/pocket-tts
