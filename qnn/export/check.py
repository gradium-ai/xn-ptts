"""Check the graphs against reference.py (a literal port of ptts), with the same noise.

    python check.py [--voice Freya] [--text "..."] [--half] [--overlap-add]

Runs the reference model, then the graphs through `host.run`, and compares the
step inputs, latents, EOS logits and audio. Without --half the graphs have fp32
I/O and an fp32 voice cache, so this checks the rewrite; with --half everything
crossing the graph boundary is fp16, which shows what that storage costs.
"""

import argparse
import math

import scipy.io.wavfile
import torch
from tokenizers import Tokenizer

from host import frame_budget, noise_source, prepare_text_prompt, run, voice_cache
from load import CHECKPOINT, load, voice_embedding
import reference

TEXT = "Hello world. I am Phonon, running on a Snapdragon."


def rel(a, b):
    return float((a - b).norm() / b.norm())


def snr_db(a, b):
    return 10 * math.log10(float(b.pow(2).sum() / (a - b).pow(2).sum().clamp_min(1e-30)))


def tokenize(text):
    tok = Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
    prepared, frames_after_eos = prepare_text_prompt(text)
    return torch.tensor([tok.encode(prepared, add_special_tokens=False).ids], dtype=torch.int32), frames_after_eos


def graph_call(graphs):
    def call(name, *inputs):
        out = graphs[name](*inputs)
        return out if isinstance(out, tuple) else (out,)
    return call


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Freya")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--half", action="store_true", help="fp16 graph I/O (the NPU variant)")
    ap.add_argument("--overlap-add", action="store_true", help="transposed convs as overlap-add (the CPU variant)")
    ap.add_argument("--frames", type=int, default=0, help="fixed frame count, ignoring EOS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="check")
    args = ap.parse_args()

    ph = load(io_half=args.half, overlap_add=args.overlap_add)
    emb = voice_embedding(ph.weights, args.voice)
    tokens, frames_after_eos = tokenize(args.text)
    frames = args.frames or frame_budget(tokens.shape[1], ph.dims.frame_rate)
    nxt = noise_source(ph.dims.ldim, seed=args.seed)
    noises = [nxt() for _ in range(frames)]
    print("tokens", tokens.shape[1], "voice frames", emb.shape[1], "max frames", frames)

    stop = not args.frames
    ref = reference.generate(ph.weights, ph.config, emb, tokens, noises, frames, frames_after_eos, stop_at_eos=stop)

    dtype = torch.float16 if args.half else torch.float32
    kv, pos = voice_cache(ph, emb, dtype)
    out = run(graph_call(ph.graphs), ph.dims, kv, pos, tokens, ph.bos_emb, noises, frames, frames_after_eos,
              io_dtype=dtype, stop_at_eos=stop)

    n = min(len(ref["latents"]), len(out["latents"]))
    print("frames ref", len(ref["latents"]), "graphs", len(out["latents"]), "EOS step ref", ref["eos_step"], "graphs", out["eos_step"])
    print("step inputs rel err", rel(out["step_inputs"][:n], ref["step_inputs"][:n]))
    print("latents rel err", rel(out["latents"][:n], ref["latents"][:n]))
    print("per-frame latent rel err (every 10th)", ["%.1e" % rel(out["latents"][i], ref["latents"][i]) for i in range(0, n, 10)])
    print("eos logits max abs diff", float((out["eos_logits"][:n] - ref["eos_logits"][:n]).abs().max()))
    m = min(len(ref["audio"]), len(out["audio"]))
    print("audio", m, "samples, SNR dB", snr_db(out["audio"][:m], ref["audio"][:m]))
    scipy.io.wavfile.write(f"{args.out}_graphs.wav", 24000, out["audio"].numpy())
    scipy.io.wavfile.write(f"{args.out}_ref.wav", 24000, ref["audio"].numpy())


if __name__ == "__main__":
    main()
