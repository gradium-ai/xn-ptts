# Opt-in packed Mimi CPU operators

Mimi reuses constant f32 weights for many small streaming products. This path
retains a packed copy of eight transformer matrices and optionally the initial
waveform convolution, using xn's owned f32 SME2 operator. It borrows CPU input
storage, writes directly to fresh output storage and reuses worker-local scratch.

## Backend prerequisite

This branch depends on [xn #134](https://github.com/gradium-ai/xn/pull/134).
The workspace and QNN wrapper temporarily pin the public fork revision containing
that API. **Before merging, replace the fork pin with an accepted upstream xn
revision or release.** The Phonon integration can be reviewed while that API is
reviewed, but is not independently ready to merge with the temporary dependency.

## Enable and reproduce

Build on a CPU supporting SME2 (measured on Apple M4 Pro):

```sh
cargo build --locked -p ptts --release --features hf,accelerate,kai --example bench
```

Set `MODEL` to a local Phonon checkpoint directory containing `model.q8.gguf`,
`config.json`, `tokenizer.json` and `default-voice.safetensors`. A warm baseline:

```sh
env -u PTTS_EXPERIMENT_PACKED_MIMI -u PTTS_EXPERIMENT_PACKED_CONV \
  XN_KAI=1 target/release/examples/bench \
  --model "$MODEL/model.q8.gguf" --config "$MODEL/config.json" \
  --voice "$MODEL/default-voice.safetensors" --quant q8 --cpu --threads 4 \
  --lang en --seed 42 --warmup 5 --iters 20 \
  --input 'The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.' \
  --metrics-json baseline.json
```

Repeat the same command in a fresh process with
`PTTS_EXPERIMENT_PACKED_MIMI=1` and
`PTTS_EXPERIMENT_PACKED_CONV=initial-small`, saving to a different JSON file.
To measure linears alone, set only `PTTS_EXPERIMENT_PACKED_MIMI=1`.
These environment settings are read when model weights are loaded.

Both flags are off by default. With no `kai`, no supported CPU, non-f32 tensors,
or `XN_KAI=0`, loading retains the ordinary operators. The convolution selection
is limited to the checkpoint path ending in `decoder.model.0.conv`, groups=1,
and at least 32 output channels. At runtime it uses the packed operator only for
output lengths at most 16; larger queued batches use the original convolution.
The rejected broad/all-convolution experiment is not part of this branch.

TTFA starts before `Synth::stream_with` and ends when its first nonempty PCM
chunk reaches the caller. Model loading and initial voice conditioning are
reported separately. JSON arrivals contain `[elapsed_ms, chunk_samples]` pairs,
so the complete first-frame delivery boundary can be checked. This does not
measure HTTP transport, device playback or server queueing.

For numerical comparisons, use separate runs with `--warmup 2 --iters 1
--pcm-output baseline.f32`, then the enabled candidate. Files contain mono
little-endian float32 PCM. Capture and timing export are mutually exclusive:
copying the audio is not part of the timing experiment. PCM files are written
after generation. Use the same voice, input, seed, temperature and conversion
when making playable WAVs.

## Evidence and tradeoffs

The [focused PR validation](https://github.com/rudrakshkarpe/xn-ptts/blob/4140cdc9b9efb17dec80c8d0989eae48ab9de7ff/benchmarks/mimi-backend/upstream-pr-m4-pro-2026-11-01/README.md)
contains 540 measured requests and 135 warmups on this branch's implementation.
Median full-frame TTFA improves 5.6–7.6% across three texts. Twenty paired
candidate waveform comparisons preserve length with maximum sample difference
1.684e-6. Graphs, raw arrivals, trial statistics, commands, asset hashes and paired
WAV samples are included. The earlier
[research report](https://github.com/rudrakshkarpe/xn-ptts/blob/2b365e38b63e5deaf4e02a95d69e6387fb8d7848/benchmarks/mimi-backend/m4-pro-2026-11-01/README.md)
contains component profiles, convolution ablations, concurrent-stream measurements,
waveform comparisons, memory probes, graphs and audio samples. That research
baseline also contains a separate prefill optimization; do not add its gains to
this branch's measurements or describe its numbers as this PR's exact timings.

The selected research candidate retains 31.04 MiB of packed weights, with median
peak RSS 351.16 to 386.69 MiB and model loading 42.1 to 46.4 ms. Original weights
remain for fallback. Scratch capacity follows the largest input seen by a decoder
worker and is released when that worker exits. These are measurements from the
linked research build; this PR does not claim a new memory measurement.

Reduction order changes, so output is not promised bit-identical. Numerical
agreement is not a MOS, WER or blinded listening score. Prior long-text tail
latency was mixed and occasional scheduling stalls remain unresolved. There is
no default enablement, new assembly kernel, OS priority policy or change to the
size or timing policy of emitted chunks.

## Validation

```sh
cargo test --locked -p ptts --release --features hf,accelerate,kai
cargo test --locked -p ptts --release --no-default-features packed
XN_KAI=0 cargo test --locked -p ptts --release --features hf,accelerate,kai packed
```

Native operator tests compare odd dimensions and changing row counts with the
ordinary backend; convolution tests cover stride, dilation and batching. A
streaming test crosses the 16-step dispatch boundary in both directions and
checks bias, subsequent output and retained causal state. On unsupported hardware,
these tests validate fallback construction but cannot exercise SME2 arithmetic.
