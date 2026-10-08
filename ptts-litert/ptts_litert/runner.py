"""Speak one chunk of text with a bundle on LiteRT's CPU runtime: the host side of the model.

This is what an on-device runtime does around the graphs, written plainly:

1. The voice embedding, then the text embeddings, go through `prefill` `prefill_len` rows a
   call, padded; only the real rows' keys and values are written into the cache.
2. Each step feeds `input_linear(previous latent) + conditions` (`bos_emb` in place of the
   latent at the first step) to `flow_step`, writes its keys and values into the cache, and
   decodes its latent with `mimi_step`.
3. A logit above `eos_threshold` starts a countdown: that frame and `frames_after_eos` more
   are kept, then generation stops (`plan::EosPolicy`).

It needs numpy, safetensors and ai-edge-litert, not PyTorch.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def state_names(n: int) -> list[str]:
    return [f"mimi_s{i:02d}" for i in range(n - 1)] + ["mimi_kv"]


class Bundle:
    def __init__(self, dir: Path, model: Path | None = None, threads: int = 4):
        from ai_edge_litert.interpreter import Interpreter

        self.dir = dir
        self.meta = json.loads((dir / "bundle.json").read_text())
        self.host = load_file(str(dir / "host.safetensors"))
        it = Interpreter(model_path=str(model or dir / "model.tflite"), num_threads=threads)
        self.run = {n: it.get_signature_runner(n) for n in it.get_signature_list()}
        inputs = self.run["mimi_step"].get_input_details()
        self.mimi_states = [(n, inputs[n]["shape"]) for n in state_names(len(inputs) - 4)]
        self.kv_shape = self.run["flow_step"].get_input_details()["kv"]["shape"]

    def voice(self, name: str) -> np.ndarray:
        return load_file(str(self.dir / "voices" / f"{name}.safetensors"))["emb"][0]

    def text_rows(self, tokens) -> np.ndarray:
        return self.host["flow_lm.conditioner.embed.weight"][np.asarray(tokens)]

    def step_input(self, latent: np.ndarray) -> np.ndarray:
        """`input_linear(latent) + conditions`, [1, 1, d]."""
        emb = latent @ self.host["flow_lm.input_linear.weight"].T + self.host["flow_lm.conditions"]
        return emb.reshape(1, 1, -1).astype(np.float32)

    def prefill(self, kv: np.ndarray, pos: int, rows: np.ndarray) -> int:
        """Writes `rows` [n, d] into the cache from slot `pos`; returns the next free slot."""
        p = self.meta["prefill_len"]
        for start in range(0, len(rows), p):
            chunk = rows[start : start + p]
            x = np.zeros((1, p, rows.shape[1]), np.float32)
            x[0, : len(chunk)] = chunk
            t = self.run["prefill_tables"](pos=np.array([pos], np.int32))
            kv_new = self.run["prefill"](x=x, kv=kv, cos=t["cos"], sin=t["sin"], mask=t["mask"])["kv_new"]
            kv[:, :, pos : pos + len(chunk)] = kv_new[:, :, : len(chunk)]
            pos += len(chunk)
        return pos

    def generate(
        self,
        voice: np.ndarray,
        tokens,
        noises,
        frames_after_eos: int,
        max_frames: int | None = None,
        teacher: np.ndarray | None = None,
    ) -> dict:
        """`noises[i]` is frame i's flow starting point [ldim], already at the sampling temperature.

        With `teacher` [N, ldim], step i + 1 is fed `teacher[i]` instead of the model's own
        latent i, so per-step errors do not compound when comparing with another implementation.
        """
        meta = self.meta
        if max_frames is None:
            max_frames = meta["max_frames"]
        max_frames = min(max_frames, len(noises))
        kv = np.zeros(self.kv_shape, np.float32)
        pos = self.prefill(kv, 0, voice)
        pos = self.prefill(kv, pos, self.text_rows(tokens))
        if pos + max_frames > kv.shape[2]:
            raise ValueError(f"{pos} prompt rows and {max_frames} frames do not fit in the {kv.shape[2]}-slot cache")

        states = {n: np.zeros(s, np.float32) for n, s in self.mimi_states}
        latent = self.host["flow_lm.bos_emb"]
        out = {"latents": [], "eos_logits": [], "audio": [], "step_inputs": [], "eos_step": None}
        countdown = None
        for frame in range(max_frames):
            emb = self.step_input(latent)
            t = self.run["step_tables"](pos=np.array([pos], np.int32), frame=np.array([frame], np.int32))
            noise = np.asarray(noises[frame], np.float32).reshape(1, -1)
            f = self.run["flow_step"](emb=emb, noise=noise, kv=kv, cos=t["cos"], sin=t["sin"], mask=t["mask"])
            kv[:, :, pos : pos + 1] = f["kv_new"]
            pos += 1
            m = self.run["mimi_step"](
                latent=f["latent"], cos=t["mimi_cos"], sin=t["mimi_sin"], mask=t["mimi_mask"], **states
            )
            states = {n: m[f"{n}_out"] for n in states}
            eos = float(f["eos"].reshape(-1)[0])
            out["step_inputs"].append(emb.reshape(-1))
            out["latents"].append(f["latent"].reshape(-1))
            out["eos_logits"].append(eos)
            out["audio"].append(m["audio"].reshape(-1))
            latent = f["latent"].reshape(-1)
            if teacher is not None and frame < len(teacher):
                latent = teacher[frame]
            if countdown is None and eos > meta["eos_threshold"]:
                out["eos_step"], countdown = frame, frames_after_eos
            if countdown is not None:
                if countdown == 0:
                    break
                countdown -= 1
        for k in ("latents", "step_inputs", "audio"):
            out[k] = np.stack(out[k]) if k != "audio" else np.concatenate(out[k])
        out["eos_logits"] = np.array(out["eos_logits"])
        out["prefix_len"] = pos - len(out["latents"])
        return out
