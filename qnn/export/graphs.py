"""Fixed-shape, stateless graphs over Phonon (and Pocket TTS's smaller checkpoints), for AI Hub.

The same design as the pocket-tts export this was first built for:

- Prefill: a chunk of text tokens through the flow LM backbone, returning the
  chunk's new keys and values.
- Step: one autoregressive step of the flow LM, then the Mimi decoder on the
  latent it produced: 80 ms of audio, the EOS logit, the next step's input
  embedding, the step's new keys and values, and the updated Mimi states.
- VoicePrefill (optional): a voice prompt's embeddings through the backbone, for
  when voices are not shipped as precomputed KV caches.

The host owns the flow LM's KV cache and passes the position. Masks and RoPE are
built in the graph from it, through precomputed tables. Each head's q/k channels
are permuted from interleaved (re, im) pairs to rotate-half order, which leaves
the attention scores unchanged; the cached keys are in that order.

What differs from pocket-tts, all read off `xn-ptts/ptts/src`:

- The weights are read straight from the checkpoint, no pocket-tts modules,
  under the names `ptts` gives them (`loader::remap_key`).
- GELU is the config's, in each transformer (`gelu_approximate`). QAIRT converts
  F.gelu to QNN's Gelu op whatever `approximate` says (and rejects torch.erf); on
  the S25 HTP that op is within 1.5e-4 of the exact GELU, fp16 level
  (diag_gelu.py), so it needs no rewrite.
- The flow LM transformer has no layer scale; Mimi's has.
- Every generated frame's input gets a constant bias: the summed conditioners
  (`num_speakers`, a lut, at its default "1"). Text and voice positions do not.
- The voice prompt is a voice file's `emb`, or its `speaker_wavs` latents
  through the speaker projection (`loader::load_voice_emb`).
- The flow net's time conditioning is a constant for a fixed number of LSD steps,
  so it is computed once at load (no sin/cos in the graph). `flow.w_s_t` is not
  used: `ptts` drops it on load (`loader::remap_key`), and its LSD decode feeds
  s and t straight to the two time embedders. The one-step flow types
  (`drifting`, `pointwise`) have no time conditioning.
- The optional `flow_lm` keys of the smaller checkpoints: `z_dim` (a `z_proj`
  after `out_norm`, which the EOS head and the flow net read), `attn_dim` and
  `num_kv_heads` (grouped-query attention; the k/v heads are repeated to
  `num_heads` at load, so the KV cache holds `num_heads` heads, as in `ptts`),
  and a low-rank text table, projected to `d_model` at load.
- Mimi's attention window: query i sees the cached slots after i, plus the new
  keys up to itself, so w keys (the Rust mask allows
  `key <= seq_idx < key + context`, pocket-tts's `delta < context`). The cache
  keeps w slots, so its oldest slot is never read.
- The EOS logit is compared against the config's `eos_threshold` on the host.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

NEG = -1e4


def head_perm(dim_per_head: int) -> torch.Tensor:
    """Channel order taking interleaved (re, im) pairs to rotate-half order."""
    return torch.cat([torch.arange(0, dim_per_head, 2), torch.arange(1, dim_per_head, 2)])


def rope_tables(positions: torch.Tensor, dim_per_head: int, max_period: float):
    """cos and sin, each [len(positions), dim_per_head], for rotate-half RoPE.

    Same arithmetic as `rope.rs`: f32 inverse frequencies 1 / max_period^(i / half).
    """
    half = dim_per_head // 2
    inv_freq = 1.0 / torch.pow(torch.tensor(max_period, dtype=torch.float32),
                               torch.arange(half, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], -1)


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


def erf_poly(z):
    """erf from exp and a rational polynomial (Abramowitz and Stegun 7.1.26), |error| < 1.5e-7."""
    a = z.abs()
    t = 1.0 / (1.0 + 0.3275911 * a)
    p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return torch.sign(z) * (1.0 - p * torch.exp(-a * a))


def gelu_variants():
    """GELU formulations, by name, for diag_gelu.py."""
    r = 1 / math.sqrt(2)
    return {
        "exact": F.gelu,
        "tanh": lambda x: F.gelu(x, approximate="tanh"),
        # torch.erf: "Converter does not support 'Erf' op type" (QAIRT 2.50)
        "poly": lambda x: 0.5 * x * (1 + erf_poly(x * r)),
        "sigmoid": lambda x: x * torch.sigmoid(1.702 * x),
    }


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(scale.clone(), requires_grad=False)

    def forward(self, x):
        return x * self.scale


def gelu_approximate(config: dict, section: str) -> str:
    """F.gelu's `approximate` for the transformer in `config[section]`: its `gelu`, or when that is
    absent tanh for a checkpoint without a `model_id` and erf otherwise (`TTSConfig::default_gelu`)."""
    gelu = config[section].get("gelu") or ("erf" if config.get("model_id") else "tanh")
    assert gelu in ("erf", "tanh"), f"unknown gelu {gelu!r}"
    return "none" if gelu == "erf" else "tanh"


class Layer(nn.Module):
    """One StreamingTransformerLayer (`transformer.rs`) with the cache passed in rather than held."""

    def __init__(self, W, prefix: str, heads: int, layer_scale: bool, gelu: str = "none"):
        super().__init__()
        self.gelu = gelu
        w = W[f"{prefix}.self_attn.in_proj.weight"]
        e = w.shape[1]
        a = W[f"{prefix}.self_attn.out_proj.weight"].shape[1]  # attn_dim: queries and attention output
        self.h, self.d = heads, a // heads
        # Each k/v head serves h / kv_heads consecutive query heads (torch's `repeat_interleave`):
        # they are repeated to h heads here, the layout ptts's KV cache holds them in.
        q, k, v = w.split([a, (w.shape[0] - a) // 2, (w.shape[0] - a) // 2])
        k, v = (t.view(-1, self.d, e).repeat_interleave(a // t.shape[0], 0) for t in (k, v))
        w = torch.stack([q.view(self.h, self.d, e), k, v])
        perm = head_perm(self.d)
        w[0] = w[0][:, perm]
        w[1] = w[1][:, perm]
        self.in_proj = nn.Linear(e, 3 * a, bias=False)
        self.in_proj.weight.data = w.reshape(3 * a, e)
        self.out_proj = linear(W, f"{prefix}.self_attn.out_proj", bias=False)
        self.norm1 = layer_norm(W, f"{prefix}.norm1", 1e-5)
        self.norm2 = layer_norm(W, f"{prefix}.norm2", 1e-5)
        self.linear1 = linear(W, f"{prefix}.linear1", bias=False)
        self.linear2 = linear(W, f"{prefix}.linear2", bias=False)
        if layer_scale:
            self.ls1 = Scale(W[f"{prefix}.layer_scale_1.scale"])
            self.ls2 = Scale(W[f"{prefix}.layer_scale_2.scale"])
        else:
            self.ls1 = self.ls2 = nn.Identity()
        self.scale = self.d**-0.5

    def forward(self, x, k_cache, v_cache, cos, sin, mask):
        # x [1, S, E]; k_cache, v_cache [1, H, T, D]; cos, sin [S, D]; mask [1, 1, S, T + S]
        s = x.shape[1]
        qkv = self.in_proj(self.norm1(x)).view(1, s, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [1, H, S, D]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        keys = torch.cat([k_cache, k], 2)
        values = torch.cat([v_cache, v], 2)
        att = torch.softmax(q @ keys.transpose(-1, -2) * self.scale + mask, -1)
        y = (att @ values).transpose(1, 2).reshape(1, s, self.h * self.d)
        x = x + self.ls1(self.out_proj(y))
        y = self.linear2(F.gelu(self.linear1(self.norm2(x)), approximate=self.gelu))
        return x + self.ls2(y), k, v


class Backbone(nn.Module):
    def __init__(self, W, prefix: str, layers: int, heads: int, gelu: str = "none"):
        super().__init__()
        self.layers = nn.ModuleList(Layer(W, f"{prefix}.layers.{i}", heads, False, gelu) for i in range(layers))

    def forward(self, x, kv_cache, cos, sin, mask):
        # kv_cache [2L, H, T, D]: keys of layer l at 2l, values at 2l + 1.
        new = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, kv_cache[2 * i : 2 * i + 1], kv_cache[2 * i + 1 : 2 * i + 2], cos, sin, mask)
            new += [k, v]
        return x, torch.cat(new, 0)


# ---- conditioners ----


def summed_conditions(W, config: dict, values: dict) -> torch.Tensor:
    """What every generated frame's input gets added, [d_model] (`conditioners::load_summed_conditions`).

    Only lut conditioners with a noop tokenizer are supported (all this checkpoint has).
    """
    d_model = config["flow_lm"]["d_model"]
    defaults = {"num_speakers": "1", "duration_delta": "0.0", "padding_bonus": "0.0"}
    fuser_sum = (config.get("fuser") or {}).get("sum", [])
    total = torch.zeros(d_model)
    for cfg in config.get("conditioners", []):
        name = cfg["name"]
        assert name in fuser_sum, f"conditioner {name} is not summed"
        value = values.get(name, defaults.get(name))
        assert value is not None, f"conditioner {name} needs a value"
        prefix = f"flow_lm.condition_provider.conditioners.{name}"
        if cfg["type"] == "lut":
            lut = cfg["lut"]
            assert lut["tokenizer"] == "noop"
            idx = lut["possible_values"].index(value)
            emb = W[f"{prefix}.embed.weight"][idx]
            if f"{prefix}.output_proj.weight" in W:
                emb = emb @ W[f"{prefix}.output_proj.weight"].T
        else:
            c = cfg["continuous"]
            half = c["dim"] // 2
            pos = c["scale_factor"] * float(value)
            phases = torch.tensor([pos / c.get("max_period", 10000.0) ** (i / (half - 1)) for i in range(half)])
            emb = torch.cat([phases.cos(), phases.sin()])
            if f"{prefix}.output_proj.weight" in W:
                w = W[f"{prefix}.output_proj.weight"]
                emb = emb @ w.T + W.get(f"{prefix}.output_proj.bias", torch.zeros(w.shape[0]))
        total = total + emb
    return total


# ---- flow net ----


def time_norm(x, alpha, kind: str, eps=1e-5):
    """`TimestepEmbedder`'s norm as `mlp.rs` computes it, the config's `flow_lm.time_rms_norm`:
    `rms` is x / sqrt(mean(x^2) + eps) * alpha, `rms_var` (the default) x / sqrt(unbiased var + eps) * alpha."""
    if kind == "rms":
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * alpha
    assert kind == "rms_var", f"unknown time_rms_norm {kind!r}"
    return x / torch.sqrt(x.var(-1, unbiased=True, keepdim=True) + eps) * alpha


def timestep_embedding(W, prefix, t: float, norm: str) -> torch.Tensor:
    freqs = W[f"{prefix}.freqs"]
    args = torch.tensor(t, dtype=torch.float32) * freqs
    emb = torch.cat([args.cos(), args.sin()])
    x = F.silu(emb @ W[f"{prefix}.mlp.0.weight"].T + W[f"{prefix}.mlp.0.bias"])
    x = x @ W[f"{prefix}.mlp.2.weight"].T + W[f"{prefix}.mlp.2.bias"]
    return time_norm(x, W[f"{prefix}.mlp.3.alpha"], norm)


def lsd_times(W, prefix, steps: int, norm: str) -> torch.Tensor:
    """The flow net's time conditioning for each LSD step, [steps, C]: the mean of the
    two time embedders at s = i / steps and t = (i + 1) / steps (`flow_lm::lsd_decode`)."""
    rows = []
    for i in range(steps):
        s, t = i / steps, (i + 1) / steps
        rows.append((timestep_embedding(W, f"{prefix}.time_embed.0", s, norm) + timestep_embedding(W, f"{prefix}.time_embed.1", t, norm)) / 2)
    return torch.stack(rows)


class FlowNet(nn.Module):
    """SimpleMLPAdaLN (`mlp.rs`) with its time conditioning precomputed, for the config's `flow_type`."""

    def __init__(self, W, prefix: str, depth: int, steps: int, norm: str = "rms_var", flow_type: str = "lsd"):
        super().__init__()
        assert flow_type in ("lsd", "drifting", "pointwise"), f"unknown flow_type {flow_type!r}"
        self.flow_type = flow_type
        if flow_type == "lsd":
            self.register_buffer("times", lsd_times(W, prefix, steps, norm))
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

    def velocity(self, c, x, time=None):
        x = self.input_proj(x)
        y = self.cond_embed(c) if time is None else time + self.cond_embed(c)
        sy = F.silu(y)
        ch = self.channels
        for ln, m1, m2, ada in zip(self.in_ln, self.mlp1, self.mlp2, self.ada):
            a = ada(sy)
            shift, scale, gate = a[..., :ch], a[..., ch : 2 * ch], a[..., 2 * ch :]
            h = ln(x) * (1 + scale) + shift
            x = x + gate * m2(F.silu(m1(h)))
        a = self.final_ada(sy)
        shift, scale = a[..., :ch], a[..., ch:]
        x = F.layer_norm(x, (ch,), eps=1e-6) * (1 + scale) + shift
        return self.final_linear(x)

    def forward(self, c, noise):
        """LSD decode from x_0 = noise; or one evaluation without time conditioning, which is the
        latent (`drifting`) or is added to the noise (`pointwise`), as `FlowLM::sample_from`."""
        if self.flow_type == "drifting":
            return self.velocity(c, noise)
        if self.flow_type == "pointwise":
            return noise + self.velocity(c, noise)
        x = noise
        for i in range(self.steps):
            x = x + self.velocity(c, x, self.times[i]) * (1.0 / self.steps)
        return x


class FlowHead(nn.Module):
    """Last hidden state and sampling noise -> latent, EOS logit, and the next step's input embedding.

    The latent is normalized (what the next step reads); MimiStep denormalizes it.
    The noise is N(0, temperature). next_emb is input_linear(latent), without the
    frame bias, which Step adds to whatever embedding comes in.
    """

    def __init__(self, W, config):
        super().__init__()
        fl = config["flow_lm"]
        self.out_norm = layer_norm(W, "flow_lm.out_norm", 1e-5)
        # With `z_dim`, the EOS head and the flow net read a bias-free projection of the normed output.
        self.z_proj = linear(W, "flow_lm.z_proj", bias=False) if fl.get("z_dim") else nn.Identity()
        self.out_eos = linear(W, "flow_lm.out_eos")
        self.flow_net = FlowNet(
            W, "flow_lm.flow_net", fl["flow_depth"], config["lsd_decode_steps"], fl.get("time_rms_norm", "rms_var"),
            fl.get("flow_type", "lsd"),
        )
        self.input_linear = linear(W, "flow_lm.input_linear", bias=False)

    def forward(self, x, noise):
        h = self.z_proj(self.out_norm(x)[:, -1])
        eos = self.out_eos(h)
        latent = self.flow_net(h, noise)
        return latent, eos, self.input_linear(latent)[:, None]


# ---- Mimi ----


def conv1d(W, prefix) -> nn.Conv1d:
    w = W[f"{prefix}.weight"]
    m = nn.Conv1d(w.shape[1], w.shape[0], w.shape[2], bias=f"{prefix}.bias" in W)
    m.weight.data = w.clone()
    if m.bias is not None:
        m.bias.data = W[f"{prefix}.bias"].clone()
    return m


def conv_tr1d(W, prefix, stride, groups=1) -> nn.ConvTranspose1d:
    w = W[f"{prefix}.weight"]  # [Cin, Cout / groups, K]
    has_bias = f"{prefix}.bias" in W
    m = nn.ConvTranspose1d(w.shape[0], w.shape[1] * groups, w.shape[2], stride=stride, groups=groups, bias=has_bias)
    m.weight.data = w.clone()
    if has_bias:
        m.bias.data = W[f"{prefix}.bias"].clone()
    return m


def conv_step(conv: nn.Conv1d, prev, x):
    """A causal conv with its past input (kernel - 1 samples, stride 1) passed in."""
    x = torch.cat([prev, x], -1)
    return conv(x), x[..., -prev.shape[-1] :]


def convtr_step(convtr: nn.ConvTranspose1d, partial, x, overlap_add: bool = False):
    """`StreamingConvTranspose1d::forward`, with the partial tail passed in."""
    y = convtr_overlap_add(convtr, x) if overlap_add else convtr(x)
    pt = partial.shape[-1]
    head = y[..., :pt] + partial
    tail = y[..., pt:]
    new_partial = tail[..., -pt:]
    if convtr.bias is not None:
        new_partial = new_partial - convtr.bias[:, None]
    return torch.cat([head, tail[..., :-pt]], -1), new_partial


def convtr_overlap_add(convtr: nn.ConvTranspose1d, x):
    """convtr(x) for a kernel of twice the stride, as a matmul and an overlap-add.

    QNN's CPU backend crashes on Mimi's transposed convolutions, so its graphs
    use this exact rewrite.
    """
    w = convtr.weight  # [Cin, Cout / groups, K]
    s = convtr.stride[0]
    assert w.shape[-1] == 2 * s
    b, cin, t = x.shape
    if convtr.groups == 1:
        z = torch.einsum("bit,iok->botk", x, w)  # [B, Cout, T, K]
    else:
        assert convtr.groups == cin and w.shape[1] == 1  # depthwise
        z = x[..., None] * w[None, :, 0, None, :]
    zero = torch.zeros_like(z[:, :, :1, :s])
    y = torch.cat([z[..., :s], zero], 2) + torch.cat([zero, z[..., s:]], 2)  # [B, C, T + 1, S]
    y = y.reshape(b, y.shape[1], (t + 1) * s)
    if convtr.bias is not None:
        y = y + convtr.bias[:, None]
    return y


class MimiStep(nn.Module):
    """One normalized latent [1, ldim] -> audio [1, 1, samples_per_frame], threading every state through.

    States, in `state_shapes()` order: the upsampler's partial frame, the init
    conv's past input, then per decoder stage the transposed conv's partial output
    and the residual block's k=3 conv past input, then the final conv's past
    input, and last the attention cache [2L, H, W, D] holding the last W keys/values,
    right-aligned (slot W - 1 is the newest).
    """

    def __init__(self, W, mc: dict, overlap_add: bool = False, gelu: str = "none"):
        super().__init__()
        self.overlap_add = overlap_add
        self.register_buffer("emb_std", W["flow_lm.emb_std"].clone())
        self.register_buffer("emb_mean", W["flow_lm.emb_mean"].clone())
        ow = W["mimi.quantizer.output_proj.weight"][..., 0]  # [dimension, ldim]
        self.output_proj = nn.Linear(ow.shape[1], ow.shape[0], bias=False)
        self.output_proj.weight.data = ow.clone()
        hop = math.prod(mc["ratios"])
        enc_rate = mc["sample_rate"] / hop
        self.steps = int(enc_rate / mc["frame_rate"])
        self.samples_per_frame = self.steps * hop
        assert abs(enc_rate - mc["frame_rate"]) > 0.01
        self.upsample = conv_tr1d(W, "mimi.upsample.convtr.convtr", self.steps, groups=mc["dimension"])
        # `ProjectedTransformer`: from `dimension` to `transformer_d_model` and on to the
        # decoder's `decoder_dimension`, each projection only where the widths differ.
        dim, width = mc["dimension"], mc["transformer_d_model"]
        p = "mimi.decoder_transformer"
        self.transformer_in = linear(W, f"{p}.input_proj", bias=False) if width != dim else nn.Identity()
        out = mc.get("decoder_dimension") or dim
        self.transformer_out = linear(W, f"{p}.output_proj.0", bias=False) if width != out else nn.Identity()
        self.layers = nn.ModuleList(
            Layer(W, f"{p}.transformer.layers.{i}", mc["transformer_num_heads"], True, gelu)
            for i in range(mc["transformer_num_layers"])
        )
        self.window = mc["transformer_context"]
        assert mc["n_residual_layers"] == 1
        p = "mimi.decoder.model"
        self.init_conv = conv1d(W, f"{p}.0.conv")
        self.convtrs = nn.ModuleList()
        self.res1 = nn.ModuleList()
        self.res2 = nn.ModuleList()
        idx = 1
        for r in mc["ratios"]:
            self.convtrs.append(conv_tr1d(W, f"{p}.{idx + 1}.convtr", r))
            self.res1.append(conv1d(W, f"{p}.{idx + 2}.block.1.conv"))
            self.res2.append(conv1d(W, f"{p}.{idx + 2}.block.3.conv"))
            idx += 3
        self.final_conv = conv1d(W, f"{p}.{idx + 1}.conv")
        assert all(c.kernel_size[0] == 1 for c in self.res2)

        self.shapes = [
            (1, dim, self.upsample.kernel_size[0] - self.steps),
            (1, self.init_conv.in_channels, self.init_conv.kernel_size[0] - 1),
        ]
        for ct, r1 in zip(self.convtrs, self.res1):
            self.shapes.append((1, ct.out_channels, ct.kernel_size[0] - ct.stride[0]))
            self.shapes.append((1, r1.in_channels, r1.kernel_size[0] - 1))
        self.shapes.append((1, self.final_conv.in_channels, self.final_conv.kernel_size[0] - 1))
        layer = self.layers[0]
        self.kv_shape = (2 * len(self.layers), layer.h, self.window, layer.d)

    def state_shapes(self):
        return self.shapes + [self.kv_shape]

    def forward(self, latent, cos, sin, mask, *states):
        states = list(states)
        kv = states.pop()
        x = self.output_proj(latent * self.emb_std + self.emb_mean)[:, :, None]  # [1, dimension, 1]
        x, up = convtr_step(self.upsample, states[0], x, self.overlap_add)
        new_states = [up]

        x = self.transformer_in(x.transpose(1, 2))  # [1, steps, E]
        new_kv = []
        for i, layer in enumerate(self.layers):
            k_cache, v_cache = kv[2 * i : 2 * i + 1], kv[2 * i + 1 : 2 * i + 2]
            x, k, v = layer(x, k_cache, v_cache, cos, sin, mask)
            new_kv += [torch.cat([k_cache, k], 2)[:, :, self.steps :], torch.cat([v_cache, v], 2)[:, :, self.steps :]]
        x = self.transformer_out(x).transpose(1, 2)

        it = iter(states[1:])
        x, s = conv_step(self.init_conv, next(it), x)
        new_states.append(s)
        for ct, r1, r2 in zip(self.convtrs, self.res1, self.res2):
            x, s = convtr_step(ct, next(it), F.elu(x), self.overlap_add)
            new_states.append(s)
            v, s = conv_step(r1, next(it), F.elu(x))
            new_states.append(s)
            x = x + r2(F.elu(v))
        x, s = conv_step(self.final_conv, next(it), F.elu(x))
        new_states.append(s)
        return (x, *new_states, torch.cat(new_kv, 0))


class Tables(nn.Module):
    """Position-derived constants: RoPE tables and mask building blocks."""

    def __init__(self, dims, cache: int, chunk: int):
        super().__init__()
        cos, sin = rope_tables(torch.arange(cache + chunk), dims.head_dim, dims.max_period)
        self.register_buffer("flow_cos", cos)
        self.register_buffer("flow_sin", sin)
        self.register_buffer("slots", torch.arange(cache, dtype=torch.int32)[None, :])
        w, s = dims.mimi_window, dims.mimi_steps
        cos, sin = rope_tables(torch.arange(cache * s), dims.mimi_head_dim, dims.mimi_max_period)
        self.register_buffer("mimi_cos", cos)
        self.register_buffer("mimi_sin", sin)
        j = torch.arange(w + s)[None, :]
        i = torch.arange(s)[:, None]
        # Query i sits at column w + i; it sees columns i + 1..w + i (`key <= seq_idx < key + context`).
        self.register_buffer("mimi_band", (j > i) & (j <= w + i))
        self.register_buffer("mimi_j", torch.arange(w + s, dtype=torch.int32)[None, :])
        self.window, self.steps = w, s
        self.cache = cache

    def flow(self, pos, n: int):
        """cos, sin [n, D] and mask [1, 1, n, cache + n] for n queries from position pos."""
        idx = pos + torch.arange(n, dtype=torch.int32)
        cos = self.flow_cos.index_select(0, idx)
        sin = self.flow_sin.index_select(0, idx)
        past = torch.where(self.slots < pos, 0.0, NEG).expand(n, self.cache)
        causal = torch.triu(torch.full((n, n), NEG), 1)
        return cos, sin, torch.cat([past, causal], -1)[None, None]

    def mimi(self, frame):
        """cos, sin [steps, D] and mask [1, 1, steps, window + steps] for Mimi frame `frame`."""
        p0 = frame * self.steps
        idx = p0 + torch.arange(self.steps, dtype=torch.int32)
        cos = self.mimi_cos.index_select(0, idx)
        sin = self.mimi_sin.index_select(0, idx)
        # The cache holds min(p0, w) real entries, in its last slots.
        valid = self.mimi_band & (self.mimi_j >= self.window - p0)
        return cos, sin, torch.where(valid, 0.0, NEG)[None, None]


class Prefill(nn.Module):
    """tokens [1, S] int32, kv_cache [2L, H, T, D], pos [1] int32 -> new keys/values [2L, H, S, D].

    Float I/O is fp16, or fp32 with io_half=False (QNN's CPU backend cannot run the casts).
    """

    def __init__(self, W, backbone, tables, chunk: int, io_half: bool = True):
        super().__init__()
        self.io_half = io_half
        # n_bins + 1 rows; the learnt padding row `ptts` appends is never looked up.
        # A low-rank table goes through its output_proj once, here, as `LUTConditioner::load` does.
        embed = W["flow_lm.conditioner.embed.weight"].clone()
        if "flow_lm.conditioner.output_proj.weight" in W:
            embed = embed @ W["flow_lm.conditioner.output_proj.weight"].T
        self.register_buffer("embed", embed)
        self.register_buffer("vocab", torch.arange(embed.shape[0], dtype=torch.int32))
        self.backbone = backbone
        self.tables = tables
        self.chunk = chunk

    def forward(self, tokens, kv_cache, pos):
        cos, sin, mask = self.tables.flow(pos, self.chunk)
        # One-hot matmul, not a gather: next to the RoPE table gathers, QAIRT 2.50
        # compiled a token gather that read each position's next token on the device.
        onehot = (tokens.view(-1, 1) == self.vocab[None]).float()
        x = (onehot @ self.embed)[None]
        _, kv_new = self.backbone(x, kv_cache.float(), cos, sin, mask)
        return kv_new.half() if self.io_half else kv_new


class VoicePrefill(nn.Module):
    """emb [1, V, E] (a voice prompt), kv_cache, pos -> new keys/values [2L, H, V, D].

    Only needed to condition on a voice on the device; shipped voices are
    precomputed KV caches (host.voice_cache).
    """

    def __init__(self, backbone, tables, frames: int, io_half: bool = True):
        super().__init__()
        self.io_half = io_half
        self.backbone = backbone
        self.tables = tables
        self.frames = frames

    def forward(self, emb, kv_cache, pos):
        cos, sin, mask = self.tables.flow(pos, self.frames)
        _, kv_new = self.backbone(emb.float(), kv_cache.float(), cos, sin, mask)
        return kv_new.half() if self.io_half else kv_new


class Step(nn.Module):
    """One frame: the flow LM step, then Mimi on the latent it produced.

    Inputs: emb [1, 1, E] (the previous step's next_emb, or input_linear(bos_emb)),
    noise [1, ldim], kv_cache, pos [1] int32 (cache slots filled), frame [1] int32
    (steps taken since the prompt), then the Mimi states.

    Outputs: audio [1, 1, samples], eos [1, 1] logit, next_emb, kv_new [2L, H, 1, D],
    latent [1, ldim], then the Mimi states.

    Every float input and output is fp16 (fp32 with io_half=False).
    """

    def __init__(self, W, config, backbone, tables, mimi_step, frame_bias, io_half: bool = True):
        super().__init__()
        self.io_half = io_half
        self.backbone = backbone
        self.tables = tables
        self.register_buffer("frame_bias", frame_bias.clone())
        self.head = FlowHead(W, config)
        self.mimi = mimi_step

    def forward(self, emb, noise, kv_cache, pos, frame, *mimi_states):
        cos, sin, mask = self.tables.flow(pos, 1)
        x, kv_new = self.backbone(emb.float() + self.frame_bias, kv_cache.float(), cos, sin, mask)
        latent, eos, next_emb = self.head(x, noise.float())
        cos, sin, mask = self.tables.mimi(frame)
        audio, *new_states = self.mimi(latent, cos, sin, mask, *(s.float() for s in mimi_states))
        outputs = (audio, eos, next_emb, kv_new, latent, *new_states)
        return tuple(t.half() for t in outputs) if self.io_half else outputs
