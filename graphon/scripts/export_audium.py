"""Export an audium ASR student and its Gemimi codec to the files `graphon` loads.

    python export_audium.py --student xps/<sig>/checkpoint_<epoch>.th \
        --codec codecs/gemimi_24khz_150M_.../checkpoint.th \
        --tokenizer text_tokenizers/<name>.model --out graphon-<sig>

The student must use the codec as a `self_wav` encodec conditioner, summed into the trunk input
at every step, with the audio-token input masked. A frozen codec is read from `--codec`; a
finetuned one from the student's checkpoint, except for SEANet convs it did not train.
Writes into `--out`:

- `model.safetensors`: the trunk in the checkpoint's dtype, the self_wav head (output
  projection and LayerNorm), and two precomputed vectors: `initial_audio_embedding`, the sum of
  the audio embeddings of the initial token (the only audio input a masked model reads, on the
  first step), and `condition_sum`, the per-stream constant the fuser adds to every step.
- `gemimi.safetensors`: the codec's encoder side (SEANet with weight norms folded, encoder
  transformer, latent head) in float32.
- `tokenizer.json`: the SentencePiece pieces, for decoding.
- `config.json`.

`condition_sum` takes `--condition NAME=VALUE` for the summed conditioners. One left out reads as
dropped (its learnt padding), as the serving worker does, except `delay`, which defaults to the
interleaver's audio delay.
"""

import argparse
import json
import math
from pathlib import Path

import sentencepiece
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file


def load(path: str) -> tuple[dict, dict, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    weights = (state.get("fsdp_best_state") or state["best_state"])["model"]
    cfg = OmegaConf.to_container(state["xp.cfg"], resolve=True)
    return weights, cfg, state.get("exported_from") or {}


def sin_embedding(value: float, dim: int, max_period: float) -> torch.Tensor:
    """audiocraft.modules.transformer.create_sin_embedding for one position."""
    half = dim // 2
    adim = torch.arange(half, dtype=torch.float32)
    phase = torch.tensor(value, dtype=torch.float32) / (torch.tensor(max_period) ** (adim / (half - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)])


def condition_sum(weights: dict, cfg: dict, values: dict[str, str], dim: int) -> torch.Tensor:
    total = torch.zeros(dim)
    for name in cfg["fuser"]["sum"]:
        prefix = f"condition_provider.conditioners.{name}."
        conditioner = cfg["conditioners"][name]
        kind = conditioner["type"]
        params = conditioner[kind]
        value = values.get(name)
        if value is None:
            total += weights[prefix + "learnt_padding"].float().reshape(dim)
            continue
        if kind == "lut":
            possible = params["possible_values"]
            # WhiteSpaceTokenizer / NoopTokenizer: the value's index in `possible_values`.
            tokens = value.split() if params["tokenizer"] == "whitespace" else [value]
            emb = sum(weights[prefix + "embed.weight"][possible.index(t)].float() for t in tokens)
        elif kind == "continuous":
            scaled = float(value) * params["scale_factor"]
            emb = sin_embedding(scaled, params["dim"], params.get("max_period", 10000.0))
        else:
            raise ValueError(f"conditioner {name}: unsupported type {kind}")
        out = weights[prefix + "output_proj.weight"].float() @ emb
        if prefix + "output_proj.bias" in weights:
            out += weights[prefix + "output_proj.bias"].float()
        total += out
    return total


def fold_weight_norm(weights: dict) -> dict:
    out = {}
    for key, value in weights.items():
        if key.endswith(".weight_v"):
            stem = key[: -len("_v")]
            g = weights[stem + "_g"].float()
            v = value.float()
            norm = v.flatten(1).norm(dim=1).reshape(g.shape)
            out[stem] = (g * v / norm).contiguous()
        elif not key.endswith(".weight_g"):
            out[key] = value
    return out


def rename_in_proj(key: str) -> str:
    # Newer students keep the projections as one-entry lists: `in_projs.0` is `in_proj`.
    assert ".in_projs.1." not in key and ".out_projs.1." not in key, key
    key = key.replace("self_attn.in_projs.0.", "self_attn.in_proj.").replace("self_attn.out_projs.0.", "self_attn.out_proj.")
    return key.replace("self_attn.in_proj_weight", "self_attn.in_proj.weight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--condition", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    weights, cfg, _ = load(args.student)
    lm = cfg["transformer_lm"]
    dim = lm["dim"]
    sw = cfg["conditioners"]["self_wav"]
    assert sw["type"] == "encodec", "an encodec self_wav conditioner"
    finetuned = sw["encodec"]["finetune"]
    learn_conv = finetuned and sw["encodec"].get("learn_conv", True)
    assert sw["encodec"].get("layer_norm") is False and sw["encodec"].get("norm_output") == "layer_norm"
    assert cfg["fuser"]["streaming_sum"] == ["self_wav"] and cfg["tokens"]["mask_input_audio"]
    assert lm["positional_embedding"] == "rope_concat" and lm["gating"] == "silu" and lm["qk_norm"]
    assert lm.get("text_out_low_rank") is None, "a factored text head is not supported yet"

    values = dict(kv.split("=", 1) for kv in args.condition)
    values.setdefault("delay", str(cfg["interleaver"]["audio_delay"]))
    unknown = set(values) - set(cfg["fuser"]["sum"])
    assert not unknown, f"no summed conditioner named {unknown}"

    n_q = cfg["compression_model_n_q"]
    initial = sum(weights[f"emb.{i}.weight"][-1].float() for i in range(n_q))
    trunk = {}
    for key, value in weights.items():
        if key.startswith(("transformer.", "text_emb.", "text_linear.", "out_norm.")):
            trunk[rename_in_proj(key)] = value.contiguous()
        elif key.startswith("condition_provider.conditioners.self_wav.output_"):
            trunk[key[len("condition_provider.conditioners.") :]] = value.contiguous()
    trunk["initial_audio_embedding"] = initial.to(weights["text_linear.weight"].dtype)
    trunk["condition_sum"] = condition_sum(weights, cfg, values, dim).to(weights["text_linear.weight"].dtype)
    save_file(trunk, args.out / "model.safetensors")

    codec, codec_cfg, exported_from = load(args.codec)
    if finetuned:
        prefix = "condition_provider.conditioners.self_wav.compression_model."
        tuned = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        assert tuned.keys() == codec.keys(), "the student's codec has the codec file's layout"
        # SEANet convs it did not train come from the codec file, in its own precision.
        codec = {k: tuned[k] if learn_conv or not k.startswith("encoder.") else v for k, v in codec.items()}
    codec = fold_weight_norm(codec)
    # moshi's StreamingConv1d wraps a NormConv1d: `conv.conv.weight` is `conv.weight` in ptts.
    encoder = {
        rename_in_proj(k).replace(".conv.conv.", ".conv."): v.float().contiguous()
        for k, v in codec.items()
        if k.startswith(("encoder.", "encoder_transformer.", "downsample."))
    }
    save_file(encoder, args.out / "gemimi.safetensors")

    sp = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    pieces = []
    for i in range(sp.get_piece_size()):
        kind = "byte" if sp.is_byte(i) else "control" if sp.is_control(i) or sp.is_unknown(i) else "normal"
        pieces.append({"piece": sp.id_to_piece(i), "kind": kind})
    with open(args.out / "tokenizer.json", "w") as f:
        json.dump({"pieces": pieces}, f, ensure_ascii=False)

    seanet = codec_cfg["seanet"]
    # The encoder transformer's settings: the shared ones, then the encoder's own overrides.
    shared = codec_cfg["transformer"]
    enc = {**{k: v for k, v in shared.items() if not isinstance(v, dict)}, **(shared.get("encoder") or {})}
    num_layers = 1 + max(int(k.split(".")[3]) for k in encoder if k.startswith("encoder_transformer.transformer.layers."))
    width = encoder["encoder_transformer.input_proj.weight"].shape[0]
    stack = encoder["downsample.norm.weight"].shape[0] // width
    frame_rate = cfg["sample_rate"] / (math.prod(seanet["ratios"]) * stack)
    hidden = weights["transformer.layers.0.gating.linear_out.weight"].shape[1]
    config = {
        "sample_rate": cfg["sample_rate"],
        "frame_rate": frame_rate,
        "delay_steps": math.ceil(-cfg["interleaver"]["audio_delay"] * frame_rate),
        "text": {
            "card": lm["text_card"],
            "padding_id": lm["existing_text_padding_id"],
            "end_of_padding_id": lm["end_of_text_padding_id"],
        },
        "trunk": {
            "dim": dim,
            "num_heads": lm["num_heads"],
            "num_kv_heads": lm["num_heads"] // lm["kv_repeat"],
            "head_dim": lm["head_dim"],
            "num_layers": lm["num_layers"],
            "ffn_hidden": hidden,
            "context": lm["context"],
            "max_period": lm["max_period"],
            "norm_eps": 1e-5,
            "qk_norm_eps": lm["qk_norm_eps"],
            "text_emb_low_rank": lm.get("text_emb_low_rank"),
        },
        "gemimi": {
            "channels": 1,
            "dimension": seanet["dimension"],
            "n_filters": seanet["n_filters"],
            "n_residual_layers": seanet["n_residual_layers"],
            "ratios": seanet["ratios"],
            "kernel_size": seanet["kernel_size"],
            "last_kernel_size": seanet["last_kernel_size"],
            "residual_kernel_size": seanet["residual_kernel_size"],
            "dilation_base": seanet["dilation_base"],
            "compress": seanet["compress"],
            "transformer_d_model": width,
            "transformer_num_heads": enc["num_heads"],
            "transformer_num_layers": num_layers,
            "transformer_context": enc["context"],
            "transformer_max_period": enc.get("max_period", 10000.0),
            "transformer_dim_feedforward": encoder["encoder_transformer.transformer.layers.0.linear1.weight"].shape[0],
            "latent_stack": stack,
            "latent_dim": encoder["downsample.proj.weight"].shape[0],
        },
        "conditions": values,
        "exported_from": {
            "student": args.student,
            "codec": args.codec,
            "codec_finetuned": finetuned,
            "codec_overrides": exported_from.get("overrides"),
        },
    }
    with open(args.out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))
    print("seanet", json.dumps(seanet))
    print("encoder transformer", json.dumps(enc))


if __name__ == "__main__":
    main()
