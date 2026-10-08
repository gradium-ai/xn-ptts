"""Export a checkpoint as a LiteRT bundle: one `.tflite` and what the host needs beside it.

The bundle has the Core ML export's layout (`ptts/examples/export_coreml.rs`):

- `model.tflite`: five signatures. `prefill`, `flow_step` and `mimi_step` are the model, for
  an NPU; `prefill_tables` and `step_tables` turn positions into RoPE tables and masks, on the
  CPU. See `model.py` and the README for their inputs and outputs.
- `host.safetensors`: what the host applies itself. The text embedding table
  (`flow_lm.conditioner.embed.weight`), `flow_lm.input_linear.weight`, `flow_lm.bos_emb`, and
  the summed conditions (`flow_lm.conditions`).
- `voices/<name>.safetensors`: each voice as the embedding the flow LM is prompted with, `emb`.
- `tokenizer.json`, and `bundle.json` describing it all with every file's size and SHA-256.
"""

import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from . import model as M
from .checkpoint import Checkpoint, frame_budget


def build(ck: Checkpoint, prefill_len: int, ctx: int) -> tuple[dict, dict]:
    """The signatures as modules, and sample inputs for each."""
    W, cfg = ck.weights, ck.config
    fl, mc = cfg["flow_lm"], cfg["mimi"]
    heads = fl["num_heads"]
    head_dim = fl["d_model"] // heads
    backbone = M.Transformer(W, "flow_lm.transformer", fl["num_layers"], heads)
    mimi = M.Mimi(W, mc)
    prefill_tables = M.FlowTables(ctx, prefill_len, head_dim, fl["max_period"])
    step_tables = M.FlowTables(ctx, 1, head_dim, fl["max_period"])
    mimi_tables = M.MimiTables(mimi.window, mimi.steps, mimi.head_dim, mc["transformer_max_period"])
    modules = {
        "prefill": M.Prefill(backbone),
        "flow_step": M.FlowStep(W, cfg, backbone),
        "mimi_step": M.MimiStep(mimi),
        "prefill_tables": M.PrefillTables(prefill_tables),
        "step_tables": M.StepTables(step_tables, mimi_tables),
    }

    pos = torch.tensor([0], dtype=torch.int32)
    with torch.no_grad():
        pcos, psin, pmask = prefill_tables(pos)
        cos, sin, mask = step_tables(pos)
        mcos, msin, mmask = mimi_tables(pos)
    kv = torch.zeros(2 * fl["num_layers"], heads, ctx, head_dim)
    states = [torch.zeros(s) for s in mimi.state_shapes()]
    samples = {
        "prefill": {"x": torch.zeros(1, prefill_len, fl["d_model"]), "kv": kv, "cos": pcos, "sin": psin, "mask": pmask},
        "flow_step": {
            "emb": torch.zeros(1, 1, fl["d_model"]),
            "noise": torch.zeros(1, fl["ldim"]),
            "kv": kv.clone(),
            "cos": cos,
            "sin": sin,
            "mask": mask,
        },
        "mimi_step": {
            "latent": torch.zeros(1, fl["ldim"]),
            "cos": mcos,
            "sin": msin,
            "mask": mmask,
            **dict(zip(M.state_names(len(states)), states)),
        },
        # Separate tensors: the converter merges inputs given the same tensor object.
        "prefill_tables": {"pos": pos},
        "step_tables": {"pos": pos.clone(), "frame": pos.clone()},
    }
    return {n: m.eval() for n, m in modules.items()}, samples


def convert(modules: dict, samples: dict, path: Path):
    import litert_torch

    conv = None
    for name, module in modules.items():
        conv = (litert_torch.signature if conv is None else conv.signature)(name, module, sample_kwargs=samples[name])
    with torch.no_grad():
        conv.convert().export(str(path))


def list_files(root: Path) -> list[dict]:
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "bundle.json":
            data = p.read_bytes()
            files.append(
                {"path": p.relative_to(root).as_posix(), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
    return files


def export(args):
    ck = Checkpoint.open(args.dir, args.weights)
    cfg = ck.config
    if cfg.get("voices"):
        raise SystemExit("checkpoints with baked-in voices are not supported yet")
    given = dict(c.split("=", 1) for c in args.condition)
    fl, mc = cfg["flow_lm"], cfg["mimi"]
    out: Path = args.out
    shutil.rmtree(out / "voices", ignore_errors=True)
    (out / "voices").mkdir(parents=True)

    voices = ck.voices(args.voices)
    if not voices:
        raise SystemExit("no voices: pass --voices <dir>")
    vlen = 0
    for name, path in voices:
        emb = ck.voice_emb(path)
        if emb.shape[2] != fl["d_model"]:
            raise SystemExit(f"voice {name} is {list(emb.shape)}, the model is {fl['d_model']}-wide")
        vlen = max(vlen, emb.shape[1])
        save_file({"emb": emb.contiguous()}, str(out / "voices" / f"{name}.safetensors"))

    # The KV cache holds the longest voice, one chunk of text and the frames it may generate.
    max_frames = frame_budget(args.max_tokens, mc["frame_rate"])
    ctx = vlen + args.max_tokens + max_frames
    print(f"{len(voices)} voices, prefill {args.max_tokens} rows, KV cache {ctx} slots")
    modules, samples = build(ck, args.max_tokens, ctx)
    convert(modules, samples, out / "model.tflite")

    W = ck.weights
    host = {
        n: W[n].contiguous()
        for n in ["flow_lm.conditioner.embed.weight", "flow_lm.input_linear.weight", "flow_lm.bos_emb"]
    }
    host["flow_lm.conditions"] = ck.conditions(given)
    save_file(host, str(out / "host.safetensors"))

    tokenizer = args.tokenizer or args.dir / "tokenizer.json"
    if tokenizer.suffix != ".json" or not tokenizer.is_file():
        raise SystemExit(
            f"no tokenizer.json at {tokenizer}: pass --tokenizer (scripts/convert-tokenizer.py converts a tokenizer.model)"
        )
    shutil.copy(tokenizer, out / "tokenizer.json")

    mimi = modules["mimi_step"].mimi
    files = list_files(out)
    meta = {
        "built": int(time.time()),
        "ctx": ctx,
        "prefill_len": args.max_tokens,
        "max_frames": max_frames,
        "mimi_window": mimi.window,
        "samples_per_frame": mimi.samples_per_frame,
        "sample_rate": mc["sample_rate"],
        "frame_rate": mc["frame_rate"],
        "eos_threshold": cfg["eos_threshold"],
        "temperature": args.temperature,
        "dims": {"d": fl["d_model"], "heads": fl["num_heads"], "layers": fl["num_layers"], "ldim": fl["ldim"]},
        "voices": [n for n, _ in voices],
        "conditions": given,
        "files": files,
    }
    (out / "bundle.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: {len(files)} files")
