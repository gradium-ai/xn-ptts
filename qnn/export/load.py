"""Load a checkpoint `ptts` runs straight from its safetensors and build the two graphs.

pocket-tts cannot load a Phonon checkpoint: it has no summed conditioners and uses
the tanh GELU, where Phonon (as `ptts` runs it) uses the exact one. So the model
is rebuilt here from the tensors, matching `xn-ptts/ptts/src` (see graphs.py),
which also covers Pocket TTS's smaller checkpoints in `ptts`'s layout.

Every weight is upcast to fp32, as `ptts` does when it loads the BF16 flow LM.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from graphs import Backbone, MimiStep, Prefill, Step, Tables, VoicePrefill, gelu_approximate, summed_conditions
from host import CACHE, PREFILL, Dims

# The checkpoint directory (config.json, model.safetensors, tokenizer.json,
# default-voice.safetensors) and the voice files, from the environment.
CHECKPOINT = Path(os.environ.get("PHONON_CHECKPOINT", Path(__file__).resolve().parents[2] / "model/phonon-7e71a02d.200"))
VOICES = Path(os.environ.get("PHONON_VOICES", CHECKPOINT / "voices"))


@dataclass
class Phonon:
    config: dict
    weights: dict  # name -> fp32 tensor, named as `ptts` loads them (`remap_key`)
    dims: Dims
    graphs: dict  # "prefill", "step"; "voice_prefill" too with voice_frames
    bos_emb: torch.Tensor  # [1, 1, d_model]: input_linear(bos_emb), before the frame bias
    frame_bias: torch.Tensor  # [d_model]: the summed conditioners, added to every frame's input
    backbone: Backbone
    tables: Tables


def remap_key(name: str) -> str | None:
    """`loader::remap_key`: a checkpoint's tensor name as `ptts` loads it, or None for one it drops."""
    if "flow.w_s_t" in name or "quantizer.vq" in name or "quantizer.logvar_proj" in name:
        return None
    name = name.replace("flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
                        "flow_lm.speaker_proj_weight")
    name = name.replace("flow_lm.condition_provider.conditioners.transcript_in_segment.", "flow_lm.conditioner.")
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.")
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.")
    return name.replace("mimi.model.", "mimi.")


def read_weights(path: Path) -> dict:
    out = {}
    with safe_open(str(path), "pt") as f:
        for k in f.keys():
            if k.startswith("speaker_mimi") or k.startswith("mimi.encoder") or k.startswith("mimi.quantizer.vq"):
                continue
            name = remap_key(k)
            if name is not None:
                out[name] = f.get_tensor(k).float()
    return out


@torch.no_grad()
def load(checkpoint: Path = CHECKPOINT, io_half: bool = True, conditions: dict | None = None,
         voice_frames: int = 0, overlap_add: bool | None = None) -> Phonon:
    """io_half=False is the QNN CPU-backend variant: fp32 I/O and transposed convs as overlap-add."""
    config = json.loads((checkpoint / "config.json").read_text())
    W = read_weights(checkpoint / config.get("weights_name", "model.safetensors"))
    fl, mc = config["flow_lm"], config["mimi"]
    attn_dim = fl.get("attn_dim") or fl["d_model"]
    assert attn_dim % fl["num_heads"] == 0

    mimi_step = MimiStep(W, mc, overlap_add=(not io_half) if overlap_add is None else overlap_add,
                         gelu=gelu_approximate(config, "mimi")).eval()
    dims = Dims(
        layers=fl["num_layers"],
        heads=fl["num_heads"],
        head_dim=attn_dim // fl["num_heads"],
        d_model=fl["d_model"],
        ldim=fl["ldim"],
        max_period=fl["max_period"],
        mimi_heads=mc["transformer_num_heads"],
        mimi_head_dim=mc["transformer_d_model"] // mc["transformer_num_heads"],
        mimi_window=mc["transformer_context"],
        mimi_steps=mimi_step.steps,
        mimi_max_period=mc["transformer_max_period"],
        mimi_states=mimi_step.state_shapes(),
        samples_per_frame=mimi_step.samples_per_frame,
        frame_rate=mc["frame_rate"],
        eos_threshold=config["eos_threshold"],
    )
    # One backbone and one set of tables, shared by the graphs so the link step stores them once.
    backbone = Backbone(W, "flow_lm.transformer", fl["num_layers"], fl["num_heads"], gelu_approximate(config, "flow_lm")).eval()
    tables = Tables(dims, CACHE, PREFILL)
    frame_bias = summed_conditions(W, config, conditions or {})
    graphs = {
        "prefill": Prefill(W, backbone, tables, PREFILL, io_half).eval(),
        "step": Step(W, config, backbone, tables, mimi_step, frame_bias, io_half).eval(),
    }
    if voice_frames:  # a voice_prefill graph for voice prompts of this many frames
        graphs["voice_prefill"] = VoicePrefill(backbone, tables, voice_frames, io_half).eval()
    bos = W["flow_lm.bos_emb"]
    bos_emb = (bos @ W["flow_lm.input_linear.weight"].T)[None, None].clone()
    return Phonon(config, W, dims, graphs, bos_emb, frame_bias, backbone, tables)


def voice_embedding(W: dict, name_or_path) -> torch.Tensor:
    """A voice file as the voice prompt [1, T, d_model] (the default voice for "default"), as
    `loader::load_voice_emb`: an `emb` [T, D] or [1, T, D] (or a file's only tensor) as it is,
    `speaker_wavs` latents [C, T] or [1, C, T] through the speaker projection."""
    p = Path(name_or_path)
    if not p.suffix:
        p = CHECKPOINT / "default-voice.safetensors" if str(name_or_path) == "default" else VOICES / f"{name_or_path}.safetensors"
    with safe_open(str(p), "pt") as f:
        names = list(f.keys())
        name = "emb" if "emb" in names else "speaker_wavs" if "speaker_wavs" in names else names[0]
        t = f.get_tensor(name).float()
    t = t if t.dim() == 3 else t[None]
    if name != "speaker_wavs":
        return t
    return t.transpose(1, 2) @ W["flow_lm.speaker_proj_weight"].T
