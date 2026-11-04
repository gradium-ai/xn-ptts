# graphon

Streaming speech-to-text on `xn`: a Gemimi encoder (Mimi's SEANet encoder and encoder
transformer, with a latent head instead of the downsampling) feeds one 2048-wide latent frame per
80 ms step into a Qwen3-style text decoder (the trunk), which emits one text token per step a
fixed delay behind the audio.

CPU only for now, one stream at a time.

## Export a checkpoint

On a machine with the training checkpoints and a Python with `torch`, `safetensors`, `omegaconf`
and `sentencepiece`:

```
python scripts/export_audium.py --student <xp>/checkpoint.th \
    --codec <gemimi>/checkpoint.th --tokenizer <tokenizer>.model --out <export> \
    [--condition is_start=True]
```

The student reads the codec as a `self_wav` conditioner. A frozen codec comes from `--codec`; a
finetuned one from the student's checkpoint, except for SEANet convs it did not train, which
still come from `--codec`. The export folds the codec's weight norms, keeps only its encoder side,
and precomputes the two constant input vectors (the initial audio token's embeddings and the
summed conditioners). A summed conditioner not given with `--condition` reads as dropped, except
`delay`, which defaults to the run's audio delay. See the script's docstring.

An export is a folder of `config.json`, `model.safetensors` (the trunk), `gemimi.safetensors`
(the encoder, float32) and `tokenizer.json`.

## Transcribe

```
cargo run -r -p graphon --features accelerate --example transcribe -- \
    --model <export> [--quant q8_0] a.wav b.flac
```

`--quant q8_0` quantizes the transformers' linears; the SEANet convs always stay float32.
`--tokens` also prints every step's token.

## The token stream

`Stream::push` takes audio and returns one token per complete 80 ms step; `Stream::flush` feeds
the silence that brings out the last words. Besides text pieces the stream holds two control
tokens (`Stream::is_text` is false for both):

- `text.padding_id`: nothing new this step.
- `text.end_of_padding_id`: a word starts; its pieces follow.

A word's marker comes `delay_steps` steps after the word starts in the audio, so its start time
is `(step - delay_steps) * 80 ms`. Words have no end marker, and the model has no end-of-turn
token: a run of padding after the last word is the only sign of silence.

## Benchmark

```
cargo run -r -p graphon --features accelerate --example bench -- \
    --model <export> --threads 1 --quant q8_0 --encoder-quant q8_0
```

It reports the time per step of each stage (SEANet, encoder transformer, latent head, trunk)
against the 80 ms a step has in real time, the time to flush the last words, and the real-time
factor. `--encoder-chunk N` encodes N steps per encoder call, as a caller holding buffered audio
could.

## Test

```
GRAPHON_MODEL=<export> cargo test -r -p graphon
```

checks the ring-buffer encoder transformer against ptts's Mimi transformer on the export's
weights. Without `GRAPHON_MODEL` it is skipped.
