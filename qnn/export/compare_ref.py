"""Compare the graphs against ptts's own dumps in ref/<case>/ (written by xn-ptts's dump_reference).

    python compare_ref.py [case ...] [--half] [--overlap-add]

Feeds the dumped token ids, voice and noise through the graphs via host.run and
compares the prefix embeddings, frame bias, per-step inputs, latents, EOS logits
and step, and audio. Writes ref_<case>[_half].wav for Whisper.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch

from check import graph_call, rel, snr_db
from host import run, voice_cache
from load import load, voice_embedding

REF = Path(__file__).parent / "ref"


def read(case: Path, name: str, manifest: dict) -> torch.Tensor:
    shape = manifest["files"][name]["shape"]
    return torch.from_numpy(np.fromfile(case / name, dtype="<f4").reshape(shape).copy())


@torch.no_grad()
def compare(ph, case: Path, dtype, tag: str, teacher: bool = False):
    m = json.loads((case / "manifest.json").read_text())
    W = ph.weights
    tokens = torch.tensor([m["text"]["token_ids"]], dtype=torch.int32)
    emb = voice_embedding(W, m["voice"]["path"])
    noise = read(case, "noise.bin", m)
    text_rows = ph.graphs["prefill"].embed[tokens[0].long()]
    prefix = torch.cat([emb[0], text_rows])
    ref = {k: read(case, f"{k}.bin", m) for k in ["prefix_embeddings", "frame_bias", "step_inputs", "latents", "eos_logits", "audio"]}

    print(f"== {case.name} [{tag}{', teacher-forced' if teacher else ''}]: {m['text']['n_tokens']} tokens, voice {m['voice']['name']}, {m['frames']} frames")
    print(f"  prefix embeddings rel err {rel(prefix, ref['prefix_embeddings']):.2e}   frame bias max |diff| {float((ph.frame_bias - ref['frame_bias']).abs().max()):.2e}")

    kv, pos = voice_cache(ph, emb, dtype)
    max_frames = min(m["eos"]["max_frames_used"], len(noise))
    out = run(graph_call(ph.graphs), ph.dims, kv, pos, tokens, ph.bos_emb, list(noise[:, None]), max_frames,
              m["eos"]["frames_after_eos"], m["eos"]["threshold"], io_dtype=dtype,
              teacher=ref["latents"] if teacher else None, input_linear=ph.graphs["step"].head.input_linear)
    n = min(len(out["latents"]), len(ref["latents"]))
    step_inputs = out["step_inputs"][:n] + ph.frame_bias  # the dump includes the frame bias
    lat = out["latents"][:n]
    per = torch.tensor([rel(lat[i], ref["latents"][i]) for i in range(n)])
    print(f"  frames graphs {len(out['latents'])} ref {len(ref['latents'])}; EOS step graphs {out['eos_step']} ref {m['eos']['first_eos_frame']}")
    print(f"  step inputs rel err {rel(step_inputs, ref['step_inputs'][:n]):.2e}")
    print(f"  latents rel err {rel(lat, ref['latents'][:n]):.2e} (per frame max {float(per.max()):.2e}, median {float(per.median()):.2e})")
    print(f"  eos logits max |diff| {float((out['eos_logits'][:n] - ref['eos_logits'][:n]).abs().max()):.2e}")
    s = min(len(out["audio"]), len(ref["audio"]))
    print(f"  audio {s} samples, SNR {snr_db(out['audio'][:s], ref['audio'][:s]):.1f} dB")
    scipy.io.wavfile.write(f"ref_{case.name}_{tag}.wav", 24000, out["audio"].numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--half", action="store_true", help="fp16 graph I/O and an fp16 voice cache (the NPU variant)")
    ap.add_argument("--teacher", action="store_true", help="feed back ptts's latents, not the graphs' own")
    ap.add_argument("--overlap-add", action="store_true", help="transposed convs as overlap-add (the CPU variant)")
    args = ap.parse_args()
    ph = load(io_half=args.half, overlap_add=args.overlap_add)
    dtype = torch.float16 if args.half else torch.float32
    tag = ("fp16" if args.half else "fp32") + ("_oa" if args.overlap_add else "") + ("_tf" if args.teacher else "")
    cases = [REF / c for c in args.cases] or sorted(p.parent for p in REF.glob("*/manifest.json"))
    for case in cases:
        compare(ph, case, dtype, tag, args.teacher)


if __name__ == "__main__":
    main()
