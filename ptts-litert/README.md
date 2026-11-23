# ptts-litert

Export a ptts checkpoint to [LiteRT](https://ai.google.dev/edge/litert), compile it ahead of time for Android NPUs, and check it against ptts.

LiteRT has NPU compilers for several vendors, so one exported model reaches Qualcomm and MediaTek NPUs today. This package writes the model. The C library in `ptts-ffi`, built with its `litert` feature, speaks with it: see [Run it](#run-it).

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
| `bundle.json` | Sizes the runtime needs (KV cache slots, prefill rows, frame budget), the EOS threshold, the temperature, the voices, the languages, and every file's size and SHA-256. |

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

## Run it

`ptts-ffi`, the C interface in [`include/ptts.h`](../ptts-ffi/include/ptts.h), runs a bundle when it is built with its `litert` feature. `ptts_new` on a folder that holds a `model.tflite` loads it through LiteRT. Everything else works as with the other engines ([C guide](../ptts-ffi/README.md), [Android guide](../android/README.md)): voices, languages, temperature, seed, and streaming audio to a callback. Text normalization, the tokenizer and the host loop under [The model](#the-model) come from ptts. LiteRT runs the five signatures.

```sh
cargo build --release -p ptts-ffi --features litert   # this machine, to try it
FEATURES=litert ./android/build.sh                     # Android
```

The library does not link against LiteRT. `ptts_new` loads `libLiteRt.so` (`libLiteRt.dylib` on macOS) wherever the dynamic linker finds it, or from the folder in `PTTS_LITERT_LIB_DIR`. LiteRT's accelerator and NPU libraries are looked for in the same folder. Use LiteRT 2.3.0: a model compiled for an NPU runs only with the LiteRT release that compiled it.

On a desktop, the `ai-edge-litert==2.3.0` Python wheel contains `libLiteRt`. With it, `ptts-ffi/examples/speak.c` speaks a bundle:

```sh
export PTTS_LITERT_LIB_DIR=$(python -c 'import ai_edge_litert, os; print(os.path.dirname(ai_edge_litert.__file__))')
PTTS_UNIT=cpu ./speak out/litert en "Hello from Phonon." speech.wav
```

### Where it runs

| `unit` | What runs where | The app also ships |
|---|---|---|
| `PTTS_UNIT_CPU` | Everything on the CPU, through XNNPACK. | `libLiteRt.so` |
| `PTTS_UNIT_GPU` | The model on the GPU where LiteRT supports its ops, the rest on the CPU. | `libLiteRtClGlAccelerator.so` |
| `PTTS_UNIT_NPU` | The three model signatures on the NPU, the two table signatures on the CPU. | The vendor's dispatch library, and for Qualcomm the QNN runtime |

For `PTTS_UNIT_NPU` on Android, the engine reads the SoC from the `ro.soc.manufacturer` and `ro.soc.model` properties and loads the model `compile` wrote for it, `npu/Qualcomm_SM8650.tflite` for example. `PTTS_LITERT_MODEL` names another file in the bundle instead. If the bundle has no model for the SoC, the engine gives LiteRT `model.tflite`. LiteRT then compiles it on the phone if the app ships the vendor's compiler plugin, and runs it on the CPU if not. Only the file for the phone's SoC needs to be on the phone, so an app can download that one file and leave out the rest of `npu/`.

`PTTS_LITERT_THREADS` sets how many CPU threads LiteRT uses. The default is the number of cores, at most 4.

### In an Android app

LiteRT's libraries come from Google's Maven repository, and the QNN runtime comes from Maven Central:

```kotlin
android {
    // The NPU libraries are loaded from the app's native library folder, so they have to be
    // extracted there rather than left inside the APK.
    packaging { jniLibs { useLegacyPackaging = true } }
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:2.3.0")                      // libLiteRt.so
    implementation("com.google.ai.edge.litert:litert-gpu:2.3.0")                  // PTTS_UNIT_GPU
    implementation("com.google.ai.edge.litert:litert-npu-runtime-qualcomm:2.3.0") // Snapdragon NPUs
    implementation("com.qualcomm.qti:qnn-runtime:2.50.0")                         // QNN, which the Qualcomm runtime loads
    implementation("com.google.ai.edge.litert:litert-npu-runtime-mediatek:2.3.0") // Dimensity NPUs
}
```

`qnn-runtime` 2.50.0 is the QNN release that LiteRT 2.3.0's Qualcomm compiler uses. It adds about 70 MB to the APK, most of it for Hexagon versions a given phone does not have. The Hexagon side of QNN finds its libraries through `ADSP_LIBRARY_PATH`. Set it to the native library folder before the first `ptts_new`:

```kotlin
android.system.Os.setenv("ADSP_LIBRARY_PATH", applicationInfo.nativeLibraryDir, true)
val tts = PhononTTS(File(filesDir, "phonon-litert").path, "en", PhononTTS.UNIT_NPU)
```

### What has been checked

- On macOS (CPU and Metal GPU) and on the Android emulator's CPU, the engine speaks the default model. On the CPU, with the same seed, its audio matches ptts's: same length and about 70 dB SNR. Whisper reads back the text in every case.
- `compile` puts all three model signatures on the NPU in one piece each for Qualcomm SM8550, SM8650, SM8750 and SM8850, and MediaTek MT6989 and MT6991.
- On a device without an NPU (the emulator), `PTTS_UNIT_NPU` falls back to the CPU and still speaks.
- Not yet run on a phone's NPU.

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
