# QNN runner on Windows ARM64

The shared C++ engine in [`../runner`](../runner) can run Phonon's prefill and step graphs
on a Snapdragon X Elite's Hexagon NPU through QNN HTP. It uses the existing
Rust frontend in [`../text-ffi`](../text-ffi) for text preparation and tokenization.
This is a native Windows build; it does not run the Android AAR.

## Build

Install Git, Rust for `aarch64-pc-windows-msvc`, CMake, Visual Studio C++ ARM64
build tools, and a QAIRT SDK with Windows ARM64 headers/runtime. From the repository
root in PowerShell:

```powershell
.\qnn\windows\build.ps1 -QairtRoot 'C:\Qualcomm\AIStack\QAIRT\2.45.0.260326'
```

This builds `phonon.exe`, the opt-in `phonon_benchmark.exe`, and `ptts_text.dll`,
and runs the model-free C++ tests. CMake's default build still builds only the
existing CLI; `-DPHONON_BUILD_BENCHMARK=ON` enables the benchmark.

## Model bundle and runtime

Obtain or export a complete bundle for the exact target. The Android release
bundle's phone contexts do not establish compatibility with X Elite. Export tools
remain on the [`feat/phonon-qnn` branch](https://github.com/gradium-ai/xn-ptts/tree/feat/phonon-qnn/qnn/export);
they are not copied into this tree. Both 31M and gated 40M checkpoints can be
exported; respect their respective model licenses and access requirements.

The bundle needs its own tokenizer, BOS data, precomputed voice caches and a
compiled context holding both graphs. For a context actually compiled and tested
for SC8380XP, its metadata includes:

```json
{
  "runtime": {
    "context_binaries": {
      "Snapdragon X Elite CRD": {
        "file": "phonon-sc8380xp.bin",
        "soc_models": ["SC8380XP"]
      }
    }
  },
  "text": {
    "tokenizer": "tokenizer.json",
    "library": {"windows-arm64": "ptts_text.dll"}
  }
}
```

These are fragments of a full bundle's metadata, not a replacement for its
generation/voice settings. Merely adding a SoC name does not convert a binary.
Pass `--text-library` to the benchmark to use the trusted DLL you just built,
independently of the bundle's library entry.

Use matching QAIRT versions for compilation, **linking**, headers and runtime.
Passing `--qairt_version 2.45` to an AI Hub compile job does not automatically pin
the link job. Pass it explicitly to the link job as well. A context linked with
a newer service default failed on the 2.45 runtime with QNN error 5000, "Using
newer context binary on old SDK". Relinking the compiled graphs with 2.45 resolved it.

## Generate speech and verify HTP execution

Replace the bundle and output paths with your own. Create the output folder first.

```powershell
$qairt = 'C:\Qualcomm\AIStack\QAIRT\2.45.0.260326'
$lib = "$qairt\lib\aarch64-windows-msvc"
$build = (Resolve-Path '.\qnn\windows\build\Release').Path
$bundle = 'C:\models\phonon-31m-en-xelite'
$out = 'C:\phonon-results'
New-Item -ItemType Directory -Force $out | Out-Null
$env:PATH = "$lib;$build;$env:PATH"
$env:ADSP_LIBRARY_PATH = "$qairt\lib\hexagon-v73\unsigned"

& "$build\phonon_benchmark.exe" --bundle $bundle --lib-dir $lib `
  --text-library "$build\ptts_text.dll" --soc-model SC8380XP --lang en `
  --text 'Hello from Phonon. This speech is generated locally on the Snapdragon X Elite.' `
  --repeats 6 --out "$out\phonon"
if ($LASTEXITCODE) { throw 'Strict HTP verification failed.' }

(New-Object System.Media.SoundPlayer "$out\phonon-0.wav").PlaySync()
```

The benchmark always requests HTP and enables QNN profiling. It rejects a run
unless **both** graphs report positive accelerator time and nonzero call counts.
It does not accept CPU/GPU selection and has no automatic model fallback. A missing
HTP library, incompatible context or unsupported target fails. The general-purpose
`phonon` CLI retains its existing explicit backend options.

Each run writes a WAV and a JSON row; the final JSON report records load time and
all runs. First PCM latency is generation start to the first streaming callback,
not speaker latency. RTF is synthesis time divided by generated audio duration.
Load and WAV writing are excluded from these generation metrics. Normalization,
tokenization, sampling and cache management remain host CPU work. Windows uses
ordinary host buffers; Linux/Android rpcmem zero-copy is not claimed here.

## Local hardware measurements

On an X1E80100 laptop with Windows ARM64 and QAIRT 2.45.0.260326, using the text
above, the default voice and temperature 0.3:

| Model | Load (ms) | First-generation PCM (ms) | Warm PCM median (ms) | Warm RTF median |
|---|---:|---:|---:|---:|
| Phonon 31M | 378.4 | 10.60 | 9.56 | 0.0538 |
| Phonon 40M | 399.2 | 16.86 | 11.35 | 0.0633 |

Each model was loaded once and generated six utterances sequentially. Warm medians
use runs 1–5. Both graphs had positive accelerator times on every request. Four
additional long-text utterances exercised multiple text chunks. All 16 WAVs were
nonempty, nonsilent 24 kHz mono PCM16; playback was also checked by the tester.
These small local samples include profiling overhead and are not general performance
guarantees. Other running applications were not stopped. No waveform parity or
formal speech-quality assessment is claimed.

Model-free bundle/path and MSVC FP16 tests pass, including all 63,488 finite half
representations and rounding/special-value cases. Rust frontend parity tests pass
with `PTTS_TOKENIZER` pointing to the checkpoint's tokenizer. CPU/GPU argument,
unsupported-SoC and missing-HTP-library failure cases return nonzero without audio.
Linux/Android runtime regression tests were not run on this Windows laptop; their
existing CI contracts remain applicable.

To reproduce the failure checks with the same local paths as above:

```powershell
python .\qnn\runner\tests\strict_htp.py --runner "$build\phonon_benchmark.exe" `
  --bundle $bundle --lib-dir $lib --text-library "$build\ptts_text.dll"
```
