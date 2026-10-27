"""torch.export every graph, in both variants, and check the exported programs match eager.

    python export_check.py [--voice Freya]

Inputs are real: recorded from a short generation (the last prefill and step
calls, and a voice_prefill on a real voice). The NPU variant has fp16 I/O; the
CPU variant fp32 I/O and transposed convs as overlap-add.
"""

import argparse

import torch

from check import graph_call, tokenize
from graphs import VoicePrefill
from host import int32, noise_source, run, voice_cache, CACHE
from load import load, voice_embedding


@torch.no_grad()
def check(io_half: bool, voice: str):
    ph = load(io_half=io_half)
    dtype = torch.float16 if io_half else torch.float32
    emb = voice_embedding(ph.weights, voice)
    ph.graphs["voice_prefill"] = VoicePrefill(ph.backbone, ph.tables, emb.shape[1], io_half).eval()
    tokens, _ = tokenize("Hello world.")
    kv, pos = voice_cache(ph, emb, dtype)
    nxt = noise_source(ph.dims.ldim)
    noises = [nxt() for _ in range(4)]
    rec = {}
    run(graph_call(ph.graphs), ph.dims, kv, pos, tokens, ph.bos_emb, noises, 4, io_dtype=dtype, record=rec, stop_at_eos=False)
    d = ph.dims
    rec["voice_prefill"] = [[emb.to(dtype), torch.zeros(2 * d.layers, d.heads, CACHE, d.head_dim, dtype=dtype), int32(0)]]
    variant = "npu (fp16 I/O)" if io_half else "cpu (fp32 I/O, overlap-add)"
    programs = {}
    for name in ["prefill", "step", "voice_prefill"]:
        ins = tuple(rec[name][-1])
        ep = torch.export.export(ph.graphs[name], ins)
        programs[name] = ep
        a = graph_call(ph.graphs)(name, *ins)
        b = ep.module()(*ins)
        b = b if isinstance(b, tuple) else (b,)
        diff = max(float((x.float() - y.float()).abs().max()) for x, y in zip(a, b))
        dtypes = sorted({str(t.dtype) for t in (*ins, *b) if t.is_floating_point()})
        print(f"{variant} {name}: {len(ins)} inputs, {len(b)} outputs, float I/O {dtypes}, max |exported - eager| {diff:.3g}")
    return programs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Freya")
    args = ap.parse_args()
    check(True, args.voice)
    check(False, args.voice)
