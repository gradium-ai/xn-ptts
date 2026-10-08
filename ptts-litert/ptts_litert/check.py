"""Check a bundle against `ptts` itself: the output of `ptts/examples/dump_reference.rs`.

Speaks each dump's tokens with its voice and its noise through the bundle on LiteRT's CPU
runtime, and compares every stage: the prompt rows, the conditions, each step's input, the
latents, the EOS logits and frame, and the audio. With `--teacher`, each step is fed the
dump's previous latent rather than the bundle's own, which keeps a run's per-step error from
compounding.
"""

import json
import math
import wave
from pathlib import Path

import numpy as np

from .runner import Bundle


def read(case: Path, name: str, manifest: dict) -> np.ndarray:
    return np.fromfile(case / name, dtype="<f4").reshape(manifest["files"][name]["shape"])


def rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def snr_db(a, b) -> float:
    return 10 * math.log10(float((b**2).sum() / max(((a - b) ** 2).sum(), 1e-30)))


def write_wav(path: Path, audio: np.ndarray, sample_rate: int):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def check_case(bundle: Bundle, case: Path, teacher: bool, out: Path | None):
    m = json.loads((case / "manifest.json").read_text())
    ref = {
        k: read(case, f"{k}.bin", m)
        for k in ["prefix_embeddings", "frame_bias", "step_inputs", "latents", "eos_logits", "audio", "noise"]
    }
    tokens = m["text"]["token_ids"]
    v = m["voice"]["frames"]
    name = m["voice"]["name"]
    print(
        f"== {case.name}: {len(tokens)} tokens, voice {name}, {m['frames']} frames{', teacher-forced' if teacher else ''}"
    )

    if name in bundle.meta["voices"]:
        voice = bundle.voice(name)
        print(f"  voice rows         rel err {rel(voice, ref['prefix_embeddings'][:v]):.2e}")
    else:
        voice = ref["prefix_embeddings"][:v]
        print(f"  voice rows         {name} is not in the bundle: using the dump's")
    print(f"  text rows          rel err {rel(bundle.text_rows(tokens), ref['prefix_embeddings'][v:]):.2e}")
    print(f"  conditions         rel err {rel(bundle.host['flow_lm.conditions'], ref['frame_bias']):.2e}")

    eos = m["eos"]
    got = bundle.generate(
        voice,
        tokens,
        ref["noise"],
        eos["frames_after_eos"],
        eos["max_frames_used"],
        teacher=ref["latents"] if teacher else None,
    )
    n = min(len(got["latents"]), len(ref["latents"]))
    per = [rel(got["latents"][i], ref["latents"][i]) for i in range(n)]
    print(
        f"  frames             {len(got['latents'])} (ptts {len(ref['latents'])}); EOS frame {got['eos_step']} (ptts {eos['first_eos_frame']})"
    )
    print(f"  step inputs        rel err {rel(got['step_inputs'][:n], ref['step_inputs'][:n]):.2e}")
    print(
        f"  latents            rel err {rel(got['latents'][:n], ref['latents'][:n]):.2e} (per frame median {np.median(per):.2e}, max {max(per):.2e})"
    )
    print(f"  EOS logits         max |diff| {np.abs(got['eos_logits'][:n] - ref['eos_logits'][:n]).max():.2e}")
    s = min(len(got["audio"]), len(ref["audio"]))
    print(f"  audio              SNR {snr_db(got['audio'][:s], ref['audio'][:s]):.1f} dB")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        write_wav(out / f"{case.name}.wav", got["audio"], bundle.meta["sample_rate"])
    return got["eos_step"] == eos["first_eos_frame"]


def check(args):
    bundle = Bundle(args.bundle, threads=args.threads)
    ok = True
    for case in args.dumps:
        ok &= check_case(bundle, case, args.teacher, args.audio_out)
    if not ok:
        raise SystemExit("the EOS frame differs from ptts's in at least one dump")
