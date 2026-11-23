# ptts-litert

Export a ptts checkpoint to [LiteRT](https://ai.google.dev/edge/litert), compile it ahead of time for Android NPUs, and check it against ptts.

LiteRT has NPU compilers for several vendors, so one exported model reaches Qualcomm and MediaTek NPUs today. This package writes the model. The runtime that speaks with it on a phone is not here yet.

## Commands

```
cd ptts-litert
uv run --extra export ptts-litert export out/litert --dir path/to/checkpoint
uv run ptts-litert check out/litert ref/hello
uv run --extra compile ptts-litert compile out/litert qualcomm:SM8750 mediatek:MT6991
```

`export` and `compile` need different LiteRT releases, so they are separate extras and cannot be installed together.

### export

Reads a checkpoint directory (`config.json`, safetensors weights, `tokenizer.json`, voices) and writes a bundle with the same layout as the Core ML export (`ptts/examples/export_coreml.rs`):

| File | What it holds |
|---|---|
| `model.tflite` | The model, as five signatures (below). |
| `host.safetensors` | What the host applies itself: the text embedding table, `flow_lm.input_linear.weight`, `flow_lm.bos_emb`, and the summed conditions (`flow_lm.conditions`). |
| `voices/<name>.safetensors` | Each voice as the embedding the flow LM is prompted with, `emb` [1, T, d]. |
| `tokenizer.json` | The checkpoint's tokenizer. |
| `bundle.json` | Sizes the runtime needs (KV cache slots, prefill rows, frame budget), the EOS threshold, the temperature, the voices, and every file's size and SHA-256. |

`--voices` takes the voices from a directory instead of the checkpoint. `--condition NAME=VALUE` sets a conditioner, as on the other frontends. `--max-tokens` is the number of rows one prefill call takes; together with the longest voice and the frames that much text may generate, it sizes the KV cache.

Checkpoints with baked-in voices are refused for now, as in the browser build.

### check

Speaks the text of each `dump_reference` output with the bundle on LiteRT's CPU runtime and compares every stage with what ptts computed: the prompt rows, the conditions, each step's input, the latents, the EOS logits and frame, and the audio. Write a dump with:

```
cargo run --release -p ptts --example dump_reference --features hf -- \
  --dir path/to/checkpoint --voice path/to/voice.safetensors --lang en \
  --seed 0 --out ref/hello "Hello world."
```

The check replays the dump's own noise. `--teacher` feeds each step ptts's previous latent rather than the bundle's own. Without it, tiny differences grow over a long run, because each step samples from the last; with it, the error per step is what is left.

### compile

Compiles `model.tflite` for each `VENDOR:SOC` into `npu/<Vendor>_<SoC>.tflite` and records it in `bundle.json` under `npu`. Only the three model signatures are offered to the NPU; the two table signatures stay on the CPU. The report says, per signature, how many ops reached the NPU and in how many pieces. One piece per signature is the goal: every boundary is a round trip between the CPU and the NPU.

The vendor compilers run on Linux x86_64 only. Installing the `compile` extra downloads the vendors' SDKs; the Qualcomm one unpacks to several gigabytes. Google Tensor's compiler is not public.

A compiled model must run with the same LiteRT release it was compiled with.

## The model

| Signature | Inputs | Outputs | Runs on |
|---|---|---|---|
| `prefill` | `x` [1, P, d], `kv`, `cos`, `sin`, `mask` | `kv_new` [2L, H, P, D] | NPU |
| `flow_step` | `emb` [1, 1, d], `noise` [1, ldim], `kv`, `cos`, `sin`, `mask` | `eos`, `latent`, `kv_new` [2L, H, 1, D] | NPU |
| `mimi_step` | `latent`, `cos`, `sin`, `mask`, `mimi_s00` ... `mimi_kv` | `audio` [1, 1, samples], `mimi_s00_out` ... `mimi_kv_out` | NPU or CPU |
| `prefill_tables` | `pos` | `cos`, `sin`, `mask` for `prefill` | CPU |
| `step_tables` | `pos`, `frame` | `cos`, `sin`, `mask` for `flow_step`, and `mimi_cos`, `mimi_sin`, `mimi_mask` for `mimi_step` | CPU |

`kv` is the flow LM's whole KV cache, [2L, H, slots, D]. The host owns it and writes each call's new rows into it. `runner.py` is the host loop, written plainly; an on-device runtime does the same:

1. The voice rows, then the text rows (looked up in the embedding table), go through `prefill` P rows a call, padded. Only the real rows' keys and values are kept.
2. Each step feeds `input_linear(previous latent) + conditions` to `flow_step`, with `bos_emb` in place of the latent at the first step. Its latent goes to `mimi_step`, which returns a frame of audio and Mimi's next states.
3. An EOS logit above the threshold ends the text: that frame and a few more are kept, as in `plan::EosPolicy`.

The graphs are shaped by what NPU compilers accept. All shapes are static, and no graph keeps state. The NPU signatures take and return only floats: positions become RoPE tables and masks in the CPU signatures, since some compilers reject gathers and integer compares. Transposes stay at rank 4, masks have the rank of the attention scores, and Mimi's ELU is written with tanh instead of exp; each of those is something one vendor's compiler rejects. Mimi's transposed convolutions are written as a matmul and an overlap-add, with the same values.

The conditions are applied by the host, not baked into the graph, so one compiled model serves any condition values.
