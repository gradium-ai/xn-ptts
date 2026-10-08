"""Reading a checkpoint the way `ptts` does: tensor names, voices and conditions.

Ports of `ptts::loader` (`remap_key`, `checkpoint_voices`, `load_voice_emb`) and
`ptts::conditioners` (`load_summed_conditions`), kept to what the export needs. `ptts-litert
check` compares the results with what `ptts` itself computed, through `dump_reference`.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_VOICE_FILE = "default-voice.safetensors"
VOICE_DIRS = ["voices", "embeddings"]
DEFAULT_CONDITIONS = {"num_speakers": "1", "duration_delta": "0.0", "padding_bonus": "0.0"}


def remap_key(name: str) -> str | None:
    """`loader::remap_key`: checkpoint names to the names `ptts` uses, or None for an unused tensor."""
    if "flow.w_s_t" in name or "quantizer.vq" in name or "quantizer.logvar_proj" in name:
        return None
    for old, new in [
        ("flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight", "flow_lm.speaker_proj_weight"),
        ("flow_lm.condition_provider.conditioners.transcript_in_segment.", "flow_lm.conditioner."),
        ("flow_lm.backbone.", "flow_lm.transformer."),
        ("flow_lm.flow.", "flow_lm.flow_net."),
        ("mimi.model.", "mimi."),
    ]:
        name = name.replace(old, new)
    return name


def unused(name: str) -> bool:
    """Tensors only the encoder side reads, which the export never needs."""
    return name.startswith(("mimi.encoder", "speaker_mimi", "mimi.downsample."))


@dataclass
class Checkpoint:
    dir: Path
    config: dict
    weights: dict  # ptts name -> f32 tensor

    @classmethod
    def open(cls, dir: Path, weights: str = "model.safetensors") -> "Checkpoint":
        config = json.loads((dir / "config.json").read_text())
        path = dir / weights
        if path.suffix != ".safetensors":
            raise SystemExit(f"{path}: the export reads safetensors weights, not quantized ones")
        W = {}
        with safe_open(str(path), "pt") as f:
            for k in f.keys():
                name = remap_key(k)
                if name is not None and not unused(name):
                    W[name] = f.get_tensor(k).float()
        return cls(dir, config, W)

    def voices(self, voices_dir: Path | None = None) -> list[tuple[str, Path]]:
        """`loader::voices_in` for a given directory, else `loader::checkpoint_voices`."""
        if voices_dir is not None:
            return sorted((p.stem, p) for p in voices_dir.glob("*.safetensors"))
        found = []
        for sub in VOICE_DIRS:
            found += sorted((p.stem, p) for p in (self.dir / sub).glob("*.safetensors"))
        if (self.dir / DEFAULT_VOICE_FILE).is_file():
            found.append(("default", self.dir / DEFAULT_VOICE_FILE))
        voices = {}
        for name, path in found:
            voices.setdefault(name, path)
        return list(voices.items())

    def voice_emb(self, path: Path) -> torch.Tensor:
        """`loader::load_voice_emb`: a voice file as the flow LM's prompt, [1, T, d_model].

        An `emb` tensor (or a file's single tensor under another name) is used as it is;
        `speaker_wavs` latents [C, T] go through the checkpoint's speaker projection.
        """
        with safe_open(str(path), "pt") as f:
            names = list(f.keys())
            name = "emb" if "emb" in names else "speaker_wavs" if "speaker_wavs" in names else names[0]
            t = f.get_tensor(name).float()
        if t.dim() == 2:
            t = t[None]
        if t.dim() != 3:
            raise SystemExit(
                f"{path}: voice tensor `{name}` has shape {list(t.shape)}, expected two or three dimensions"
            )
        if name != "speaker_wavs":
            return t
        proj = self.weights.get("flow_lm.speaker_proj_weight")
        if proj is None:
            raise SystemExit(f"{path} holds `speaker_wavs` latents, but the checkpoint has no speaker projection")
        if t.shape[1] != proj.shape[1]:
            raise SystemExit(f"{path}: {t.shape[1]} latent channels, the speaker projection takes {proj.shape[1]}")
        return t.transpose(1, 2) @ proj.T

    def conditions(self, values: dict[str, str]) -> torch.Tensor:
        """`conditioners::load_summed_conditions`: what the flow LM adds to every generated frame's input, [d_model].

        Zeros for a checkpoint without conditioners.
        """
        cfg = self.config
        d_model = cfg["flow_lm"]["d_model"]
        conditioners = cfg.get("conditioners") or []
        known = [c["name"] for c in conditioners]
        for name in values:
            if name not in known:
                raise SystemExit(f"no conditioner named {name!r} in this checkpoint")
        fuser = cfg.get("fuser")
        total = torch.zeros(d_model)
        for c in conditioners:
            name = c["name"]
            if fuser is not None and name not in fuser.get("sum", []):
                raise SystemExit(f"conditioner {name!r} is not in the fuser's sum, the only fusion supported")
            value = values.get(name, DEFAULT_CONDITIONS.get(name))
            if value is None:
                raise SystemExit(f"conditioner {name!r} has no default, a value has to be given for it")
            prefix = f"flow_lm.condition_provider.conditioners.{name}"
            if c["type"] == "lut":
                lut = c["lut"]
                if lut["tokenizer"] != "noop" or lut.get("possible_values") is None:
                    raise SystemExit(
                        f"conditioner {name!r}: only lut conditioners with a noop tokenizer and possible_values are supported"
                    )
                if value not in lut["possible_values"]:
                    raise SystemExit(f"{value!r} is not one of {name!r}'s values {lut['possible_values']}")
                emb = self.weights[f"{prefix}.embed.weight"][lut["possible_values"].index(value)]
            else:
                cc = c["continuous"]
                dim, half = cc["dim"], cc["dim"] // 2
                position = torch.tensor(cc["scale_factor"], dtype=torch.float32) * torch.tensor(float(value))
                max_period = torch.tensor(cc.get("max_period", 10000.0), dtype=torch.float32)
                phases = torch.stack([position / max_period ** (i / (half - 1)) for i in range(half)])
                emb = torch.cat([phases.cos(), phases.sin()])
                assert dim == emb.shape[0]
            if f"{prefix}.output_proj.weight" in self.weights:
                w = self.weights[f"{prefix}.output_proj.weight"]
                emb = emb @ w.T + self.weights.get(f"{prefix}.output_proj.bias", torch.zeros(w.shape[0]))
            total = total + emb
        return total


def frame_budget(num_tokens: int, frame_rate: float) -> int:
    """`plan::frame_budget`: the most frames a chunk of `num_tokens` may generate."""
    return math.ceil((num_tokens / 3.0 + 2.0) * frame_rate)
