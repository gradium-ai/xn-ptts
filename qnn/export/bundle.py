"""Assemble a Phonon bundle: model, text front end, voices and metadata.json.

    python bundle.py --out bundle --binary out/s25=SM8750 [--binary out/s24=SM8650 ...]
        [--cpu out/cpu] [--name phonon-31m-en --display-name "Phonon 31M"]
        [--catalog voices.jsonl] [--source hf://owner/name@commit]

The layout qnn/runner reads:

    metadata.json      model, voices, graph interface, generation settings
    phonon-<soc>.bin   QNN HTP context binaries with the `prefill` and `step` graphs, one per
                       SoC; the runtime loads the one whose `soc_models` lists the device's
    cpu/prefill.dlc    with --cpu: the graphs for QNN's CPU backend (fp32 I/O), which
    cpu/step.dlc       qnn/runner's --backend cpu loads; the Android library does not
    tokenizer.json     the checkpoint's own tokenizer
    lib/<platform>/libptts_text.so
                       ptts's text front end (normalization, chunking, tokenization)
    bos.bin            fp16 [1, 1, d_model]: step's `emb` for the first frame (step adds
                       the frame bias itself)
    voices/<name>.bin  fp16 [2L, H, n, D]: the voice's keys and values, for slots 0..n-1

Each --binary is a package.py output directory (package.json and its context binary) and
the SoC it was compiled for. The checkpoint and voices come from PHONON_CHECKPOINT and
PHONON_VOICES (see load.py); --catalog gives the voices' languages and descriptions. Needs
the builds of qnn/text-ffi (its build.sh) in qnn/text-ffi/dist/, and for --cpu the DLCs
package_cpu.py writes.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import qai_hub as hub
import torch

from host import CACHE, PREFILL, TEMPERATURE, voice_cache
from load import CHECKPOINT, VOICES, load, voice_embedding

TEXT_FFI = Path(__file__).resolve().parent.parent / "text-ffi/dist"  # qnn/text-ffi/build.sh
LANGUAGE_NAMES = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese"}
# What a summed conditioner is baked in at when nothing sets it (`summed_conditions`).
CONDITION_DEFAULTS = {"num_speakers": "1", "duration_delta": "0.0", "padding_bonus": "0.0"}


def context_binary(work: Path) -> Path:
    for name in ("phonon.bin", "model.bin"):
        if (work / name).exists():
            return work / name
    found = sorted(work.glob("*.bin"))
    if not found:
        raise SystemExit(f"no context binary in {work}: run package.py")
    return found[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bundle")
    ap.add_argument("--binary", action="append", required=True, metavar="DIR=SOC",
                    help="a package.py output directory and the SoC it was compiled for, e.g. out=SM8750; repeatable")
    ap.add_argument("--cpu", help="the package_cpu.py output directory, to include the CPU DLCs")
    ap.add_argument("--name", default="phonon")
    ap.add_argument("--display-name", default="Phonon")
    ap.add_argument("--catalog", help="a voices.jsonl giving each voice's language and description")
    ap.add_argument("--source", help="where the checkpoint came from, e.g. hf://owner/name@commit")
    args = ap.parse_args()
    out = Path(args.out)
    for d in ("voices", "lib/android-arm64", "lib/linux-arm64") + (("cpu",) if args.cpu else ()):
        (out / d).mkdir(parents=True, exist_ok=True)

    phonon = load(io_half=True)
    config = phonon.config
    languages = config.get("languages") or ["en"]
    binaries, graph_io = {}, {}
    for spec in args.binary:
        work, soc = spec.rsplit("=", 1)
        jobs = json.loads((Path(work) / "package.json").read_text())
        file = f"phonon-{soc.lower()}.bin"
        shutil.copy(context_binary(Path(work)), out / file)
        binaries[jobs["device"]] = {"file": file, "soc_models": [soc.upper()], "ai_hub_model": jobs.get("model")}
        if not graph_io:
            for name, jid in jobs["compile"].items():
                specs = hub.get_job(jid).get_target_shapes()
                graph_io[name] = {"inputs": [{"name": k, "shape": list(s), "dtype": d} for k, (s, d) in specs.items()]}
    for f in ("prefill.dlc", "step.dlc") if args.cpu else ():
        shutil.copy(Path(args.cpu) / f, out / "cpu" / f)
    shutil.copy(CHECKPOINT / "tokenizer.json", out / "tokenizer.json")
    shutil.copy(TEXT_FFI / "android-arm64-v8a/libptts_text.so", out / "lib/android-arm64/libptts_text.so")
    shutil.copy(TEXT_FFI / "linux-arm64/libptts_text.so", out / "lib/linux-arm64/libptts_text.so")
    phonon.bos_emb.half().numpy().tofile(out / "bos.bin")

    catalog = {}
    if args.catalog:
        for line in Path(args.catalog).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                catalog[row["name"]] = row
    names = {p.stem for p in VOICES.glob("*.safetensors")}
    if (CHECKPOINT / "default-voice.safetensors").exists():
        names.add("default")
    voices = []
    for name in sorted(names):
        emb = voice_embedding(phonon.weights, name)
        kv, n = voice_cache(phonon, emb, torch.float16)
        np.ascontiguousarray(kv[:, :, :n].numpy()).tofile(out / "voices" / f"{name}.bin")
        row = catalog.get(name, {})
        language = row.get("language") or languages[0]
        voices.append({"name": name, "display_name": name if row else name.capitalize(), "language": language,
                       "language_name": LANGUAGE_NAMES.get(language, language),
                       "sample_rate": config["mimi"]["sample_rate"], "file": f"voices/{name}.bin",
                       "length": n, "description": row.get("description") or f"Phonon voice '{name}'"})
        print("voice", name, n, flush=True)

    fl, mc = config["flow_lm"], config["mimi"]
    frame_rate = mc["frame_rate"]
    model_id = config.get("model_id") or {}
    conditions = {c["name"]: CONDITION_DEFAULTS.get(c["name"]) for c in config.get("conditioners", [])}
    metadata = {
        "name": args.name,
        "display_name": args.display_name,
        "version": "0.1.0",
        "description": f"{args.display_name} for the Snapdragon NPU: a flow-matching language model generating "
                       "Mimi latents, decoded to 24 kHz audio, one 80 ms frame per step.",
        "model_type": "phonon",
        "languages": languages,
        "voices": voices,
        "runtime": {
            "language": languages[0],
            "qnn_version": {"major": 2, "minor": 50, "patch": 0},
            "qairt_build": "2.50.0.260828",
            "arch_bit": 64,
            "precision": "fp16",
            "context_binaries": binaries,
            **({"dlcs": {"prefill": "cpu/prefill.dlc", "step": "cpu/step.dlc"},
                "dlc_io": "float32: the CPU backend cannot run fp16 casts; a runner converts from its fp16 buffers"}
               if args.cpu else {}),
        },
        "graphs": graph_io,
        "generation": {
            "sample_rate": mc["sample_rate"],
            "frame_samples": int(mc["sample_rate"] / frame_rate),
            "frame_rate": frame_rate,
            "cache_slots": CACHE,
            "prefill_tokens": PREFILL,
            "max_tokens_per_chunk": 50,
            "temperature": TEMPERATURE,
            "noise_std": math.sqrt(TEMPERATURE),
            "eos_threshold": config["eos_threshold"],
            "min_frames_before_eos": 0,
            # Frames played from the EOS frame on, that one included (ptts: the EOS frame plus 3 or 1).
            "frames_after_eos": {"words_at_most_4": 4, "otherwise": 2},
            "max_frames": "ceil((tokens / 3 + 2) * frame_rate)",
            "bos": {"file": "bos.bin", "shape": [1, 1, fl["d_model"]], "dtype": "float16"},
            "conditions": {**conditions, "note": "baked into step as a per-frame bias"},
            "layout": {"kv_cache": "[2 * layers, heads, slots, head_dim]; layer l keys at 2l, values at 2l + 1",
                       "layers": fl["num_layers"], "heads": fl["num_heads"],
                       "head_dim": phonon.dims.head_dim},
        },
        "text": {
            "frontend": "ptts_text",
            "library": {"android-arm64": "lib/android-arm64/libptts_text.so",
                        "linux-arm64": "lib/linux-arm64/libptts_text.so"},
            "lang": languages[0],
            "tokenizer": "tokenizer.json",
        },
        "source": {"model": args.source or "Phonon", "sig": model_id.get("sig"), "epoch": model_id.get("epoch")},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"bundle {out}: {size / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
