"""Replay one sentence's graph inputs on a device and compare the outputs with the Mac's.

    python device_check.py [--case freya_hello]

Runs the fp16-I/O graphs on the Mac with ptts's dumped token ids and noise
(ref/<case>/), recording every call's inputs, then sends the same inputs through
the linked context binary on the device (one inference job per graph). Each call
is fed the Mac's inputs, so the comparison is per call, without the drift a free
run would add. The device's audio chunks, joined, go to out/device.wav, next to
out/mac.wav.

Needs out/package.json from package.py. Inference job ids go to out/device_check.json.
"""

import argparse
import inspect
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import qai_hub as hub
import scipy.io.wavfile
import torch

from check import graph_call
from compare_ref import REF, read
from host import run, voice_cache
from load import load, voice_embedding

OUT = Path(os.environ.get("PHONON_OUT", Path(__file__).parent / "out"))
OUTPUT_NAMES = {
    "prefill": ["kv_new"],
    "step": ["audio", "eos", "next_emb", "kv_new", "latent"],  # then the Mimi states
}


def snr_db(a, b):
    return 10 * math.log10(float((b**2).sum() / max(((a - b) ** 2).sum(), 1e-30)))


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def output_order(keys):
    """Output names in graph order: by trailing number when they all have one (output_10 after output_2)."""
    keys = list(keys)
    nums = [re.search(r"(\d+)$", k) for k in keys]
    return sorted(keys, key=lambda k: int(re.search(r"(\d+)$", k).group(1))) if all(nums) else keys


def input_names(module, count):
    """The exported input names: forward's parameters, with *args expanded as args_0, args_1, ..."""
    names, var = [], None
    for p in inspect.signature(module.forward).parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            var = p.name
        else:
            names.append(p.name)
    return names + [f"{var}_{i}" for i in range(count - len(names))]


def as_tuple(out):
    return out if isinstance(out, tuple) else (out,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="freya_hello")
    ap.add_argument("--graphs", default="prefill,step")
    args = ap.parse_args()
    jobs = json.loads((OUT / "package.json").read_text())
    device = hub.Device(jobs["device"])
    model = hub.get_model(jobs["model"])

    ph = load(io_half=True)
    case = REF / args.case
    m = json.loads((case / "manifest.json").read_text())
    tokens = torch.tensor([m["text"]["token_ids"]], dtype=torch.int32)
    n_tokens = tokens.shape[1]
    noise = read(case, "noise.bin", m)
    emb = voice_embedding(ph.weights, m["voice"]["path"])
    kv, pos = voice_cache(ph, emb, torch.float16)
    ref_audio = read(case, "audio.bin", m).numpy()

    record = {}
    with torch.no_grad():
        out = run(graph_call(ph.graphs), ph.dims, kv, pos, tokens, ph.bos_emb, list(noise[:, None]),
                  min(m["eos"]["max_frames_used"], len(noise)), m["eos"]["frames_after_eos"], m["eos"]["threshold"],
                  io_dtype=torch.float16, record=record)
        mac = {n: [[t.float().numpy() for t in as_tuple(ph.graphs[n](*ins))] for ins in calls] for n, calls in record.items()}
    mac_audio = out["audio"].numpy()
    print(f"{n_tokens} tokens, {len(record['step'])} steps, EOS step {out['eos_step']} (ptts {m['eos']['first_eos_frame']}), "
          f"{mac_audio.size / 24000:.2f} s of audio; Mac fp16 vs ptts audio SNR "
          f"{snr_db(mac_audio[:ref_audio.size], ref_audio[:mac_audio.size]):.1f} dB", flush=True)
    scipy.io.wavfile.write(OUT / "mac.wav", 24000, mac_audio)

    saved = OUT / "device_check.json"
    inference_jobs = json.loads(saved.read_text()) if saved.exists() else {}
    device_out, device_ms = {}, {}
    for name in args.graphs.split(","):
        calls = record[name]
        if name in inference_jobs:
            job = hub.get_job(inference_jobs[name])
        else:
            # Matched by name, in the compiled order and dtypes: the compiler may
            # reorder inputs, and would widen an fp16 one to fp32.
            specs = hub.get_job(jobs["compile"][name]).get_target_shapes()
            names = input_names(ph.graphs[name], len(calls[0]))
            inputs = {k: [c[names.index(k)].numpy().astype(dtype) for c in calls] for k, (_, dtype) in specs.items()}
            job = hub.submit_inference_job(
                model=model, device=device, inputs=inputs, name=f"phonon {name} check",
                options=f"--qnn_options context_enable_graphs={name}",
            )
            print(name, "inference", job.job_id, job.url, flush=True)
            inference_jobs[name] = job.job_id
            saved.write_text(json.dumps(inference_jobs, indent=2))
        status = job.wait()
        print(name, "inference", status.code, status.message or "", flush=True)
        if not status.success:
            return
        summary = job.download_profile()["execution_summary"]
        device_ms[name] = summary["estimated_inference_time"] / 1000
        print(f"{name}: {device_ms[name]:.2f} ms on device, peak memory "
              f"{summary.get('estimated_inference_peak_memory', 0) / 2**20:.0f} MiB", flush=True)
        device_out[name] = job.download_output_data()

    # Outputs come back keyed by name; match them to the Mac's by position in the graph's output order.
    print("\ngraph  output        shape              rel err median   max")
    for name, outs in device_out.items():
        keys = output_order(outs.keys())
        labels = OUTPUT_NAMES[name] + [f"mimi_state_{i}" for i in range(len(keys) - len(OUTPUT_NAMES[name]))]
        for i, (k, label) in enumerate(zip(keys, labels)):
            pairs = [(d.astype(np.float32), mm[i]) for d, mm in zip(outs[k], mac[name])]
            if name == "prefill":  # only the chunk's real tokens are kept; the padding rows are discarded
                pairs = [(d[:, :, :n_tokens], mm[:, :, :n_tokens]) for d, mm in pairs]
            errs = [rel(d, mm) for d, mm in pairs]
            print(f"{name:6s} {label:13s} {str(outs[k][0].shape):18s} {np.median(errs):.2e}       {max(errs):.2e}")
        if name == "step":
            eos_key = keys[1]
            diffs = [abs(float(d.reshape(-1)[0]) - float(mm[1].reshape(-1)[0])) for d, mm in zip(outs[eos_key], mac[name])]
            dev_eos = [float(d.reshape(-1)[0]) for d in outs[eos_key]]
            first = next((i for i, e in enumerate(dev_eos) if e > m["eos"]["threshold"]), None)
            print(f"step   eos logit max |diff| {max(diffs):.3f}; device's first EOS frame {first} (Mac {out['eos_step']})")

    if "step" in device_out:
        step = device_out["step"]
        audio_key = output_order(step.keys())[0]
        dev_audio = np.concatenate([a.reshape(-1).astype(np.float32) for a in step[audio_key]])
        scipy.io.wavfile.write(OUT / "device.wav", 24000, dev_audio)
        s = min(dev_audio.size, ref_audio.size)
        print(f"audio: device vs Mac SNR {snr_db(dev_audio, mac_audio):.1f} dB, device vs ptts {snr_db(dev_audio[:s], ref_audio[:s]):.1f} dB; "
              f"wrote {OUT / 'device.wav'} and {OUT / 'mac.wav'}")


if __name__ == "__main__":
    main()
