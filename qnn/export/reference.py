"""A literal, stateful port of `ptts` (xn-ptts/ptts/src), to check the graph rewrite against.

Deliberately written the way the Rust is, not the way the graphs are: interleaved
RoPE (`rope_i`), growing KV caches, -inf masks built like `transformer.rs`, Mimi's
trimmed `KvCache`, convs that keep their state as the Rust does. It shares only
the weights, the flow net's time conditioning and the GELU choice with graphs.py.
"""

import math

import torch
from torch.nn import functional as F

from graphs import gelu_approximate, lsd_times, summed_conditions


def rope_i(x, offset, max_period):
    """x [B, T, H, D], interleaved pairs, positions offset..offset+T (`rope.rs`, `rope_i`)."""
    b, t, h, d = x.shape
    half = d // 2
    inv = 1.0 / torch.pow(torch.tensor(max_period, dtype=torch.float32), torch.arange(half, dtype=torch.float32) / half)
    ang = (torch.arange(t, dtype=torch.float32) + offset)[:, None] * inv[None]
    cos, sin = ang.cos()[None, :, None], ang.sin()[None, :, None]  # [1, T, 1, half]
    xr, xi = x[..., 0::2], x[..., 1::2]
    out = torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], -1)
    return out.reshape(b, t, h, d)


class TransformerRef:
    def __init__(self, W, prefix, layers, heads, max_period, layer_scale, context=None, gelu="none",
                 attn_dim=None, kv_heads=None):
        self.W, self.p, self.n, self.h, self.mp = W, prefix, layers, heads, max_period
        self.ls, self.context, self.gelu = layer_scale, context, gelu
        # Flow LM only (`Kind::FlowLm`): query and output width, and grouped k/v heads.
        self.attn_dim, self.kvh = attn_dim, kv_heads or heads
        # Flow LM: (k [B, T, H, D], v) growing; Mimi: (k [B, H, T, D], v) trimmed to context - 1.
        self.k = [None] * layers
        self.v = [None] * layers
        self.offset = 0  # FlowLm: current_end; Mimi: absolute_offset

    def forward(self, x):
        W = self.W
        b, t, e = x.shape
        a = self.attn_dim or e
        d = a // self.h
        kv_len = 0 if self.k[0] is None else self.k[0].shape[1 if self.context is None else 2]
        mask = None
        if t > 1:
            q_idx = torch.arange(t)[:, None] + kv_len
            k_idx = torch.arange(kv_len + t)[None]
            if self.context is None:
                ok = k_idx <= q_idx
            else:
                ok = (k_idx <= q_idx) & (q_idx < k_idx + self.context)
            mask = torch.where(ok, 0.0, float("-inf"))
        for i in range(self.n):
            p = f"{self.p}.layers.{i}"
            h = F.layer_norm(x, (e,), W[f"{p}.norm1.weight"], W[f"{p}.norm1.bias"], 1e-5)
            proj = h @ W[f"{p}.self_attn.in_proj.weight"].T
            if self.context is None:
                kd = self.kvh * d
                q = proj[..., :a].reshape(b, t, self.h, d)
                # Each k/v head serves h / kvh consecutive query heads; the cache holds them repeated.
                k, v = (proj[..., a + j * kd : a + (j + 1) * kd].reshape(b, t, self.kvh, d)
                        .repeat_interleave(self.h // self.kvh, 2) for j in range(2))
            else:
                pk = proj.reshape(b, t, 3, self.h, d)
                q, k, v = pk[:, :, 0], pk[:, :, 1], pk[:, :, 2]
            q = rope_i(q, self.offset, self.mp)
            k = rope_i(k, self.offset, self.mp)
            if self.context is None:
                self.k[i] = k if self.k[i] is None else torch.cat([self.k[i], k], 1)
                self.v[i] = v if self.v[i] is None else torch.cat([self.v[i], v], 1)
                K, V = self.k[i].transpose(1, 2), self.v[i].transpose(1, 2)
            else:
                k, v = k.transpose(1, 2), v.transpose(1, 2)
                K = k if self.k[i] is None else torch.cat([self.k[i], k], 2)
                V = v if self.v[i] is None else torch.cat([self.v[i], v], 2)
                start = max(K.shape[2] - (self.context - 1), 0)
                self.k[i], self.v[i] = K[:, :, start:], V[:, :, start:]
            qq = q.transpose(1, 2)
            att = qq @ K.transpose(-1, -2) / math.sqrt(d)
            if mask is not None:
                att = att + mask
            y = (att.softmax(-1) @ V).transpose(1, 2).reshape(b, t, a)
            y = y @ W[f"{p}.self_attn.out_proj.weight"].T
            if self.ls:
                y = y * W[f"{p}.layer_scale_1.scale"]
            x = x + y
            h = F.layer_norm(x, (e,), W[f"{p}.norm2.weight"], W[f"{p}.norm2.bias"], 1e-5)
            y = F.gelu(h @ W[f"{p}.linear1.weight"].T, approximate=self.gelu) @ W[f"{p}.linear2.weight"].T
            if self.ls:
                y = y * W[f"{p}.layer_scale_2.scale"]
            x = x + y
        self.offset += t
        return x


class ConvRef:
    def __init__(self, W, prefix):
        self.w, self.b = W[f"{prefix}.weight"], W.get(f"{prefix}.bias")
        self.prev = torch.zeros(1, self.w.shape[1], self.w.shape[2] - 1)

    def __call__(self, x):
        x = torch.cat([self.prev, x], 2)
        if self.prev.shape[2] > 0:
            self.prev = x[..., -self.prev.shape[2] :]
        return F.conv1d(x, self.w, self.b)


class ConvTrRef:
    def __init__(self, W, prefix, stride, groups=1):
        self.w, self.b, self.s, self.g = W[f"{prefix}.weight"], W.get(f"{prefix}.bias"), stride, groups
        self.partial = torch.zeros(1, self.w.shape[1] * groups, self.w.shape[2] - stride)

    def __call__(self, x):
        y = F.conv_transpose1d(x, self.w, self.b, stride=self.s, groups=self.g)
        pt = self.partial.shape[2]
        y = torch.cat([y[..., :pt] + self.partial, y[..., pt:]], 2)
        tail = y[..., -pt:]
        self.partial = tail - self.b[None, :, None] if self.b is not None else tail
        return y[..., :-pt]


class MimiRef:
    def __init__(self, W, mc, gelu="none"):
        self.W = W
        hop = math.prod(mc["ratios"])
        self.steps = int(mc["sample_rate"] / hop / mc["frame_rate"])
        self.up = ConvTrRef(W, "mimi.upsample.convtr.convtr", self.steps, groups=mc["dimension"])
        self.tr = TransformerRef(W, "mimi.decoder_transformer.transformer", mc["transformer_num_layers"],
                                 mc["transformer_num_heads"], mc["transformer_max_period"], True, mc["transformer_context"],
                                 gelu)
        # `ProjectedTransformer::load`: a projection only where the widths differ.
        dim, width = mc["dimension"], mc["transformer_d_model"]
        out = mc.get("decoder_dimension") or dim
        self.input_proj = W["mimi.decoder_transformer.input_proj.weight"] if width != dim else None
        self.output_proj = W["mimi.decoder_transformer.output_proj.0.weight"] if width != out else None
        p = "mimi.decoder.model"
        self.init = ConvRef(W, f"{p}.0.conv")
        self.stages = []
        idx = 1
        for r in mc["ratios"]:
            self.stages.append((ConvTrRef(W, f"{p}.{idx + 1}.convtr", r), ConvRef(W, f"{p}.{idx + 2}.block.1.conv"),
                                ConvRef(W, f"{p}.{idx + 2}.block.3.conv")))
            idx += 3
        self.final = ConvRef(W, f"{p}.{idx + 1}.conv")

    def decode(self, latent):
        """latent [1, ldim], normalized -> audio [samples]."""
        W = self.W
        x = latent * W["flow_lm.emb_std"] + W["flow_lm.emb_mean"]
        x = F.conv1d(x[:, :, None], W["mimi.quantizer.output_proj.weight"])
        x = self.up(x)
        x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = x @ self.input_proj.T
        x = self.tr.forward(x)
        if self.output_proj is not None:
            x = x @ self.output_proj.T
        x = x.transpose(1, 2)
        x = self.init(x)
        for ct, c1, c2 in self.stages:
            x = ct(F.elu(x))
            x = x + c2(F.elu(c1(F.elu(x))))
        return self.final(F.elu(x)).reshape(-1)


class FlowLMRef:
    def __init__(self, W, config):
        self.W, self.cfg = W, config
        fl = config["flow_lm"]
        self.tr = TransformerRef(W, "flow_lm.transformer", fl["num_layers"], fl["num_heads"], fl["max_period"], False,
                                 gelu=gelu_approximate(config, "flow_lm"), attn_dim=fl.get("attn_dim"),
                                 kv_heads=fl.get("num_kv_heads"))
        self.cond = summed_conditions(W, config, {})
        self.steps = config["lsd_decode_steps"]
        self.flow_type = fl.get("flow_type", "lsd")
        if self.flow_type == "lsd":
            self.times = lsd_times(W, "flow_lm.flow_net", self.steps, fl.get("time_rms_norm", "rms_var"))
        self.depth = fl["flow_depth"]
        self.z_dim = fl.get("z_dim")
        # `LUTConditioner::load`: a low-rank table is projected whole, once.
        self.text_table = W["flow_lm.conditioner.embed.weight"]
        if "flow_lm.conditioner.output_proj.weight" in W:
            self.text_table = self.text_table @ W["flow_lm.conditioner.output_proj.weight"].T

    def prompt(self, emb):
        """Text or voice embeddings [1, T, E], no frame bias (`run_backbone_and_increment`)."""
        self.tr.forward(emb)

    def text_embeddings(self, tokens):
        return self.text_table[tokens][None]

    def flow(self, c, x, time):
        W, p = self.W, "flow_lm.flow_net"
        lin = lambda x, n: x @ W[f"{p}.{n}.weight"].T + W[f"{p}.{n}.bias"]
        x = lin(x, "input_proj")
        y = lin(c, "cond_embed") if time is None else time + lin(c, "cond_embed")
        ch = x.shape[-1]
        for i in range(self.depth):
            a = lin(F.silu(y), f"res_blocks.{i}.adaLN_modulation.1")
            shift, scale, gate = a.split(ch, -1)
            h = F.layer_norm(x, (ch,), W[f"{p}.res_blocks.{i}.in_ln.weight"], W[f"{p}.res_blocks.{i}.in_ln.bias"], 1e-6)
            h = h * (1 + scale) + shift
            h = lin(F.silu(lin(h, f"res_blocks.{i}.mlp.0")), f"res_blocks.{i}.mlp.2")
            x = x + gate * h
        a = lin(F.silu(y), "final_layer.adaLN_modulation.1")
        shift, scale = a.split(ch, -1)
        x = F.layer_norm(x, (ch,), eps=1e-6) * (1 + scale) + shift
        return lin(x, "final_layer.linear")

    def step(self, prev_latent, noise):
        """prev_latent [1, ldim] or None (BOS) -> (latent [1, ldim], eos logit, step input [E])."""
        W = self.W
        seq = W["flow_lm.bos_emb"][None] if prev_latent is None else prev_latent
        inp = seq @ W["flow_lm.input_linear.weight"].T
        x = self.tr.forward((inp + self.cond)[:, None])
        h = F.layer_norm(x[:, -1], (x.shape[-1],), W["flow_lm.out_norm.weight"], W["flow_lm.out_norm.bias"], 1e-5)
        if self.z_dim:
            h = h @ W["flow_lm.z_proj.weight"].T
        eos = h @ W["flow_lm.out_eos.weight"].T + W["flow_lm.out_eos.bias"]
        if self.flow_type == "drifting":
            cur = self.flow(h, noise, None)
        elif self.flow_type == "pointwise":
            cur = noise + self.flow(h, noise, None)
        else:
            cur = noise
            for i in range(self.steps):
                cur = cur + self.flow(h, cur, self.times[i]) * (1.0 / self.steps)
        return cur, float(eos), inp.reshape(-1)


@torch.no_grad()
def generate(W, config, voice_emb, tokens, noises, max_frames, frames_after_eos=1, stop_at_eos=True):
    """`synth::run_backbone` + Mimi for one chunk. Returns the same dict as host.run."""
    lm = FlowLMRef(W, config)
    mimi = MimiRef(W, config["mimi"], gelu_approximate(config, "mimi"))
    lm.prompt(voice_emb)
    lm.prompt(lm.text_embeddings(torch.as_tensor(tokens).reshape(-1).long()))
    latents, audio, eos_logits, inputs = [], [], [], []
    prev, eos_step, countdown = None, None, None
    for frame in range(max_frames):
        latent, eos, inp = lm.step(prev, noises[frame])
        latents.append(latent)
        inputs.append(inp)
        eos_logits.append(eos)
        audio.append(mimi.decode(latent))
        if eos > config["eos_threshold"] and countdown is None:
            eos_step, countdown = frame, frames_after_eos
        if stop_at_eos and countdown is not None:
            if countdown == 0:
                break
            countdown -= 1
        prev = latent
    return {
        "latents": torch.cat(latents),
        "audio": torch.cat(audio),
        "eos_logits": torch.tensor(eos_logits),
        "eos_step": eos_step,
        "step_inputs": torch.stack(inputs),
    }
