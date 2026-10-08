"""The model as fixed-shape PyTorch graphs, written for LiteRT's converter and NPU compilers.

The same arithmetic as `ptts` (`ptts/src`), rebuilt from the checkpoint's tensors. What makes
it different is the shape of each graph, which NPU compilers need:

- Every shape is static. The prefill takes a fixed number of rows, padded by the host.
- No graph keeps state. The host owns both KV caches and passes them in; a graph returns only
  the new rows.
- The NPU graphs (`Prefill`, `FlowStep`, `MimiStep`) are float in, float out. Positions turn
  into RoPE tables and masks in two small CPU graphs (`PrefillTables`, `StepTables`): gathers
  and integer compares are what some NPU compilers reject.
- Transposes stay at rank 4 or below, masks have the attention scores' rank, and Mimi's ELU is
  written without `exp`. Each of those is something one NPU compiler or another rejects.

Each head's q/k channels are permuted from interleaved (re, im) pairs to rotate-half order.
That leaves the attention scores unchanged; the cached keys are in that order.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

# Added to masked-out attention scores. Small enough to stay finite in fp16 on an NPU.
NEG = -1e4


def head_perm(dim_per_head: int) -> torch.Tensor:
    """Channel order taking interleaved (re, im) pairs to rotate-half order."""
    return torch.cat([torch.arange(0, dim_per_head, 2), torch.arange(1, dim_per_head, 2)])


def inv_freq(dim_per_head: int, max_period: float) -> torch.Tensor:
    """`rope.rs`'s inverse frequencies, f32: 1 / max_period^(i / half)."""
    half = dim_per_head // 2
    return 1.0 / torch.pow(
        torch.tensor(max_period, dtype=torch.float32), torch.arange(half, dtype=torch.float32) / half
    )


def rope(positions: torch.Tensor, freqs: torch.Tensor):
    """cos and sin, each [len(positions), 2 * len(freqs)], in rotate-half order."""
    angles = positions.to(torch.float32)[:, None] * freqs[None, :]
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], -1)


def elu(x):
    """ELU without exp: for x <= 0, exp(x) - 1 = 2 tanh(x / 2) / (1 - tanh(x / 2)).

    As close to the exact ELU in f32 as `F.elu` is, and accurate near zero. Some NPUs have
    tanh but not exp.
    """
    t = torch.tanh(0.5 * torch.clamp(x, max=0.0))
    return F.relu(x) + 2 * t / (1 - t)


def linear(W, prefix, bias=True) -> nn.Linear:
    w = W[f"{prefix}.weight"]
    has_bias = bias and f"{prefix}.bias" in W
    m = nn.Linear(w.shape[1], w.shape[0], bias=has_bias)
    m.weight.data = w.clone()
    if has_bias:
        m.bias.data = W[f"{prefix}.bias"].clone()
    return m


def layer_norm(W, prefix, eps) -> nn.LayerNorm:
    w = W[f"{prefix}.weight"]
    m = nn.LayerNorm(w.shape[0], eps=eps)
    m.weight.data = w.clone()
    m.bias.data = W[f"{prefix}.bias"].clone()
    return m


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(scale.clone(), requires_grad=False)

    def forward(self, x):
        return x * self.scale


class Layer(nn.Module):
    """One `StreamingTransformerLayer` (`transformer.rs`), its cache passed in rather than held."""

    def __init__(self, W, prefix: str, heads: int):
        super().__init__()
        w = W[f"{prefix}.self_attn.in_proj.weight"]
        e = w.shape[1]
        self.h, self.d = heads, e // heads
        w = w.clone().view(3, self.h, self.d, e)
        perm = head_perm(self.d)
        w[0] = w[0][:, perm]
        w[1] = w[1][:, perm]
        self.in_proj = nn.Linear(e, 3 * e, bias=False)
        self.in_proj.weight.data = w.reshape(3 * e, e)
        self.out_proj = linear(W, f"{prefix}.self_attn.out_proj", bias=False)
        self.norm1 = layer_norm(W, f"{prefix}.norm1", 1e-5)
        self.norm2 = layer_norm(W, f"{prefix}.norm2", 1e-5)
        self.linear1 = linear(W, f"{prefix}.linear1", bias=False)
        self.linear2 = linear(W, f"{prefix}.linear2", bias=False)
        if f"{prefix}.layer_scale_1.scale" in W:
            self.ls1 = Scale(W[f"{prefix}.layer_scale_1.scale"])
            self.ls2 = Scale(W[f"{prefix}.layer_scale_2.scale"])
        else:
            self.ls1 = self.ls2 = nn.Identity()
        self.scale = self.d**-0.5

    def forward(self, x, k_cache, v_cache, cos, sin, mask):
        # x [1, S, E]; k_cache, v_cache [1, H, T, D]; cos, sin [S, D]; mask [1, S, T + S]
        s, e = x.shape[1], self.h * self.d
        qkv = self.in_proj(self.norm1(x))
        q, k, v = (qkv[..., i * e : (i + 1) * e].reshape(1, s, self.h, self.d).transpose(1, 2) for i in range(3))
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        keys = torch.cat([k_cache, k], 2)
        values = torch.cat([v_cache, v], 2)
        att = torch.softmax(q @ keys.transpose(-1, -2) * self.scale + mask, -1)
        y = (att @ values).transpose(1, 2).reshape(1, s, e)
        x = x + self.ls1(self.out_proj(y))
        y = self.linear2(F.gelu(self.linear1(self.norm2(x))))
        return x + self.ls2(y), k, v


class Transformer(nn.Module):
    def __init__(self, W, prefix: str, layers: int, heads: int):
        super().__init__()
        self.layers = nn.ModuleList(Layer(W, f"{prefix}.layers.{i}", heads) for i in range(layers))

    def forward(self, x, kv, cos, sin, mask):
        # kv [2L, H, T, D]: keys of layer l at 2l, values at 2l + 1.
        new = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, kv[2 * i : 2 * i + 1], kv[2 * i + 1 : 2 * i + 2], cos, sin, mask)
            new += [k, v]
        return x, torch.cat(new, 0)


# ---- flow net ----


def rms_norm(x, alpha, eps=1e-5):
    """`TimestepEmbedder`'s norm as `mlp.rs` computes it: x / sqrt(biased var + eps) * alpha * sqrt((n-1)/n)."""
    n = x.shape[-1]
    var = x.var(-1, unbiased=False, keepdim=True)
    return x / torch.sqrt(var + eps) * alpha * math.sqrt((n - 1) / n)


def timestep_embedding(W, prefix, t: float) -> torch.Tensor:
    freqs = W[f"{prefix}.freqs"]
    args = torch.tensor(t, dtype=torch.float32) * freqs
    emb = torch.cat([args.cos(), args.sin()])
    x = F.silu(emb @ W[f"{prefix}.mlp.0.weight"].T + W[f"{prefix}.mlp.0.bias"])
    x = x @ W[f"{prefix}.mlp.2.weight"].T + W[f"{prefix}.mlp.2.bias"]
    return rms_norm(x, W[f"{prefix}.mlp.3.alpha"])


class FlowNet(nn.Module):
    """`SimpleMLPAdaLN` (`mlp.rs`), with the time conditioning of each LSD step computed once.

    The steps are fixed, so their time embeddings are constants: the mean of the two time
    embedders at s = i / steps and t = (i + 1) / steps (`flow_lm::lsd_decode`).
    """

    def __init__(self, W, prefix: str, depth: int, steps: int):
        super().__init__()
        times = []
        for i in range(steps):
            s, t = i / steps, (i + 1) / steps
            times.append(
                (
                    timestep_embedding(W, f"{prefix}.time_embed.0", s)
                    + timestep_embedding(W, f"{prefix}.time_embed.1", t)
                )
                / 2
            )
        self.register_buffer("times", torch.stack(times))
        self.steps = steps
        self.cond_embed = linear(W, f"{prefix}.cond_embed")
        self.input_proj = linear(W, f"{prefix}.input_proj")
        self.in_ln = nn.ModuleList(layer_norm(W, f"{prefix}.res_blocks.{i}.in_ln", 1e-6) for i in range(depth))
        self.mlp1 = nn.ModuleList(linear(W, f"{prefix}.res_blocks.{i}.mlp.0") for i in range(depth))
        self.mlp2 = nn.ModuleList(linear(W, f"{prefix}.res_blocks.{i}.mlp.2") for i in range(depth))
        self.ada = nn.ModuleList(linear(W, f"{prefix}.res_blocks.{i}.adaLN_modulation.1") for i in range(depth))
        self.final_ada = linear(W, f"{prefix}.final_layer.adaLN_modulation.1")
        self.final_linear = linear(W, f"{prefix}.final_layer.linear")
        self.channels = self.input_proj.out_features

    def velocity(self, c, x, time):
        x = self.input_proj(x)
        sy = F.silu(time + self.cond_embed(c))
        ch = self.channels
        for ln, m1, m2, ada in zip(self.in_ln, self.mlp1, self.mlp2, self.ada):
            a = ada(sy)
            shift, scale, gate = a[..., :ch], a[..., ch : 2 * ch], a[..., 2 * ch :]
            x = x + gate * m2(F.silu(m1(ln(x) * (1 + scale) + shift)))
        a = self.final_ada(sy)
        shift, scale = a[..., :ch], a[..., ch:]
        x = F.layer_norm(x, (ch,), eps=1e-6) * (1 + scale) + shift
        return self.final_linear(x)

    def forward(self, c, noise):
        """LSD decode from x_0 = noise."""
        x = noise
        for i in range(self.steps):
            x = x + self.velocity(c, x, self.times[i]) * (1.0 / self.steps)
        return x


# ---- Mimi decoder ----


def conv1d(W, prefix) -> nn.Conv1d:
    w = W[f"{prefix}.weight"]
    m = nn.Conv1d(w.shape[1], w.shape[0], w.shape[2], bias=f"{prefix}.bias" in W)
    m.weight.data = w.clone()
    if m.bias is not None:
        m.bias.data = W[f"{prefix}.bias"].clone()
    return m


class ConvTr(nn.Module):
    """A transposed conv whose kernel is twice its stride, as a matmul and an overlap-add.

    The same values as `nn.ConvTranspose1d`, without the op: not every NPU compiler takes it.
    """

    def __init__(self, W, prefix, stride, depthwise=False):
        super().__init__()
        w = W[f"{prefix}.weight"]  # [Cin, Cout / groups, K]
        assert w.shape[-1] == 2 * stride, f"{prefix}: kernel {w.shape[-1]} is not twice the stride {stride}"
        assert not depthwise or w.shape[1] == 1
        self.register_buffer("weight", w.clone())
        self.register_buffer("bias", W[f"{prefix}.bias"].clone() if f"{prefix}.bias" in W else None)
        self.stride, self.depthwise = stride, depthwise
        self.out_channels = w.shape[0] if depthwise else w.shape[1]

    def forward(self, x):
        b, _, t = x.shape
        s = self.stride
        if self.depthwise:
            z = x[..., None] * self.weight[None, :, 0, None, :]  # [B, C, T, K]
        else:
            z = torch.einsum("bit,iok->botk", x, self.weight)
        zero = torch.zeros_like(z[:, :, :1, :s])
        y = torch.cat([z[..., :s], zero], 2) + torch.cat([zero, z[..., s:]], 2)  # [B, C, T + 1, S]
        y = y.reshape(b, y.shape[1], (t + 1) * s)
        return y if self.bias is None else y + self.bias[:, None]


def conv_step(conv: nn.Conv1d, prev, x):
    """A causal conv with its past input (kernel - 1 samples, stride 1) passed in."""
    x = torch.cat([prev, x], -1)
    return conv(x), x[..., -prev.shape[-1] :]


def convtr_step(convtr: ConvTr, partial, x):
    """`StreamingConvTranspose1d::forward`, with the partial tail passed in."""
    y = convtr(x)
    pt = partial.shape[-1]
    head = y[..., :pt] + partial
    tail = y[..., pt:]
    new_partial = tail[..., -pt:]
    if convtr.bias is not None:
        new_partial = new_partial - convtr.bias[:, None]
    return torch.cat([head, tail[..., :-pt]], -1), new_partial


class Mimi(nn.Module):
    """One normalized latent [1, ldim] -> audio [1, 1, samples_per_frame], every state threaded through.

    States, in `state_shapes()` order: the upsampler's partial frame, the first conv's past
    input, then per decoder stage the transposed conv's partial output and the residual
    block's conv past input, then the last conv's past input, and last the attention cache
    [2L, H, W, D] holding the last W keys and values, right-aligned.
    """

    def __init__(self, W, mc: dict):
        super().__init__()
        self.register_buffer("emb_std", W["flow_lm.emb_std"].clone())
        self.register_buffer("emb_mean", W["flow_lm.emb_mean"].clone())
        ow = W["mimi.quantizer.output_proj.weight"][..., 0]
        self.output_proj = nn.Linear(ow.shape[1], ow.shape[0], bias=False)
        self.output_proj.weight.data = ow.clone()
        hop = math.prod(mc["ratios"])
        enc_rate = mc["sample_rate"] / hop
        self.steps = int(enc_rate / mc["frame_rate"])
        self.samples_per_frame = self.steps * hop
        self.upsample = ConvTr(W, "mimi.upsample.convtr.convtr", self.steps, depthwise=True)
        heads = mc["transformer_num_heads"]
        self.transformer = Transformer(W, "mimi.decoder_transformer.transformer", mc["transformer_num_layers"], heads)
        self.window = mc["transformer_context"]
        assert mc["n_residual_layers"] == 1, "one residual block per decoder stage is supported"
        p = "mimi.decoder.model"
        self.init_conv = conv1d(W, f"{p}.0.conv")
        self.convtrs, self.res1, self.res2 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        idx = 1
        for r in mc["ratios"]:
            self.convtrs.append(ConvTr(W, f"{p}.{idx + 1}.convtr", r))
            self.res1.append(conv1d(W, f"{p}.{idx + 2}.block.1.conv"))
            self.res2.append(conv1d(W, f"{p}.{idx + 2}.block.3.conv"))
            idx += 3
        self.final_conv = conv1d(W, f"{p}.{idx + 1}.conv")
        assert all(c.kernel_size[0] == 1 for c in self.res2), "the residual blocks' second conv must be 1x1"

        dim = mc["dimension"]
        self.shapes = [(1, dim, self.steps), (1, dim, self.init_conv.kernel_size[0] - 1)]
        for ct, r1 in zip(self.convtrs, self.res1):
            self.shapes.append((1, ct.out_channels, ct.stride))
            self.shapes.append((1, r1.in_channels, r1.kernel_size[0] - 1))
        self.shapes.append((1, self.final_conv.in_channels, self.final_conv.kernel_size[0] - 1))
        layer = self.transformer.layers[0]
        self.head_dim = layer.d
        self.kv_shape = (2 * len(self.transformer.layers), layer.h, self.window, layer.d)

    def state_shapes(self):
        return self.shapes + [self.kv_shape]

    def forward(self, latent, cos, sin, mask, *states):
        states = list(states)
        kv = states.pop()
        x = self.output_proj(latent * self.emb_std + self.emb_mean)[:, :, None]  # [1, C, 1]
        x, up = convtr_step(self.upsample, states[0], x)
        new_states = [up]

        x, kv_new = self.transformer(x.transpose(1, 2), kv, cos, sin, mask)  # x [1, steps, C]
        kv_out = torch.cat([kv, kv_new], 2)[:, :, self.steps :]
        x = x.transpose(1, 2)

        it = iter(states[1:])
        x, s = conv_step(self.init_conv, next(it), x)
        new_states.append(s)
        for ct, r1, r2 in zip(self.convtrs, self.res1, self.res2):
            x, s = convtr_step(ct, next(it), elu(x))
            new_states.append(s)
            v, s = conv_step(r1, next(it), elu(x))
            new_states.append(s)
            x = x + r2(elu(v))
        x, s = conv_step(self.final_conv, next(it), elu(x))
        new_states.append(s)
        return (x, *new_states, kv_out)


# ---- the exported signatures ----


def state_names(n: int) -> list[str]:
    """Mimi's state inputs, in `Mimi.state_shapes()` order. Outputs add `_out`."""
    return [f"mimi_s{i:02d}" for i in range(n - 1)] + ["mimi_kv"]


class Prefill(nn.Module):
    """Rows of the prompt (voice or text embeddings) into the flow LM: their keys and values."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, kv, cos, sin, mask):
        _, kv_new = self.backbone(x, kv, cos, sin, mask)
        return {"kv_new": kv_new}


class FlowStep(nn.Module):
    """One flow LM step: the input embedding and the noise to a latent, an EOS logit and the new keys/values.

    `emb` is `input_linear` of the previous latent (of `bos_emb` at the first step) plus the
    summed conditions. The host computes it, so one graph serves any condition values.
    """

    def __init__(self, W, config, backbone):
        super().__init__()
        fl = config["flow_lm"]
        self.backbone = backbone
        self.out_norm = layer_norm(W, "flow_lm.out_norm", 1e-5)
        self.out_eos = linear(W, "flow_lm.out_eos")
        self.flow_net = FlowNet(W, "flow_lm.flow_net", fl["flow_depth"], config["lsd_decode_steps"])

    def forward(self, emb, noise, kv, cos, sin, mask):
        x, kv_new = self.backbone(emb, kv, cos, sin, mask)
        h = self.out_norm(x)[:, -1]
        return {"eos": self.out_eos(h), "latent": self.flow_net(h, noise), "kv_new": kv_new}


class MimiStep(nn.Module):
    """One latent to one frame of audio."""

    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi

    def forward(self, latent, cos, sin, mask, **states):
        audio, *new = self.mimi(latent, cos, sin, mask, *(states[n] for n in state_names(len(states))))
        return {"audio": audio, **{f"{n}_out": t for n, t in zip(state_names(len(new)), new)}}


class FlowTables(nn.Module):
    """RoPE tables and the mask for `rows` flow LM rows starting at cache slot `pos`."""

    def __init__(self, ctx: int, rows: int, head_dim: int, max_period: float):
        super().__init__()
        self.register_buffer("freqs", inv_freq(head_dim, max_period))
        self.register_buffer("offsets", torch.arange(rows, dtype=torch.int32))
        self.register_buffer("slots", torch.arange(ctx, dtype=torch.int32)[None, :])
        self.register_buffer("causal", torch.triu(torch.full((rows, rows), NEG), 1))

    def forward(self, pos):
        cos, sin = rope(pos + self.offsets, self.freqs)
        past = torch.where(self.slots < pos, 0.0, NEG).expand(self.causal.shape[0], -1)
        return cos, sin, torch.cat([past, self.causal], -1)[None]


class MimiTables(nn.Module):
    """RoPE tables and the mask for Mimi frame `frame`.

    Query i sits at column W + i and sees columns i..W + i, so itself and the W keys before it
    (`seq_idx - context <= key <= seq_idx`). The cache holds min(frame * steps, W) real
    entries, in its last slots.
    """

    def __init__(self, window: int, steps: int, head_dim: int, max_period: float):
        super().__init__()
        self.register_buffer("freqs", inv_freq(head_dim, max_period))
        self.register_buffer("offsets", torch.arange(steps, dtype=torch.int32))
        j = torch.arange(window + steps)[None, :]
        i = torch.arange(steps)[:, None]
        self.register_buffer("band", (j >= i) & (j <= window + i))
        self.register_buffer("cols", torch.arange(window + steps, dtype=torch.int32)[None, :])
        self.window, self.steps = window, steps

    def forward(self, frame):
        p0 = frame * self.steps
        cos, sin = rope(p0 + self.offsets, self.freqs)
        valid = self.band & (self.cols >= self.window - p0)
        return cos, sin, torch.where(valid, 0.0, NEG)[None]


class PrefillTables(nn.Module):
    def __init__(self, flow: FlowTables):
        super().__init__()
        self.flow = flow

    def forward(self, pos):
        cos, sin, mask = self.flow(pos)
        return {"cos": cos, "sin": sin, "mask": mask}


class StepTables(nn.Module):
    def __init__(self, flow: FlowTables, mimi: MimiTables):
        super().__init__()
        self.flow, self.mimi = flow, mimi

    def forward(self, pos, frame):
        cos, sin, mask = self.flow(pos)
        mcos, msin, mmask = self.mimi(frame)
        return {"cos": cos, "sin": sin, "mask": mask, "mimi_cos": mcos, "mimi_sin": msin, "mimi_mask": mmask}
