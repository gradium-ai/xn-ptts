//! The streaming transformer both the flow LM and Mimi are built on, with its KV caches.
//!
//! [`StreamingTransformerState`] holds one cache per layer. [`Kind`] picks which model a layer
//! belongs to, as the two lay out their attention differently.

use crate::layer_scale::LayerScale;
use crate::rope::RotaryEmbedding;
use xn::nn::{LayerNorm, Linear, var_builder::Path};
use xn::{Backend, BackendQ, Result, Tensor, TensorView, WithDTypeF};

/// State for StreamingMultiheadAttention.
#[derive(Debug, Clone)]
pub struct StreamingMHAState<T: WithDTypeF, B: Backend> {
    /// Key cache: shape [batch_size, sequence_length, num_heads, dim_per_head]
    pub k_cache: Tensor<T, B>,
    /// Value cache: shape [batch_size, sequence_length, num_heads, dim_per_head]
    pub v_cache: Tensor<T, B>,
    /// Current end position (number of tokens seen so far).
    pub current_end: usize,
}

impl<T: WithDTypeF, B: Backend> StreamingMHAState<T, B> {
    pub fn device(&self) -> &B {
        self.k_cache.device()
    }

    #[allow(clippy::type_complexity)]
    pub fn complete_kv(
        &mut self,
        k: &Tensor<T, B>,
        v: &Tensor<T, B>,
    ) -> Result<(TensorView<T, B>, TensorView<T, B>)> {
        let t = k.dim(1usize)?;

        self.k_cache.slice_set(k, 1usize, self.current_end)?;
        self.v_cache.slice_set(v, 1usize, self.current_end)?;

        self.current_end += t;
        let new_end = self.current_end;
        let keys = self.k_cache.narrow(1, 0..new_end)?;
        let values = self.v_cache.narrow(1, 0..new_end)?;
        Ok((keys, values))
    }

    pub fn materialize_causal_mask(&self, num_queries: usize) -> Result<Tensor<T, B>> {
        let num_keys = self.current_end + num_queries;
        // Upper-left triangular mask (causal)
        let mut data = Vec::with_capacity(num_queries * num_keys);
        for q in 0..num_queries {
            for k in 0..num_keys {
                if k <= q + self.current_end {
                    data.push(T::from_f32(0.0));
                } else {
                    data.push(T::from_f32(f32::NEG_INFINITY));
                }
            }
        }
        Tensor::from_vec(data, (num_queries, num_keys), self.k_cache.device())
    }
}

/// A single-token flow-LM step whose position lives on the device, so that its kernels take
/// no position argument and one recorded step replays at any position (a CUDA graph).
///
/// The keys and values are scattered to the position, the attention runs over the whole
/// cache with `mask` hiding the unfilled slots, and RoPE reads its row of a full-length table.
pub struct DeviceStep<T: WithDTypeF, B: Backend> {
    /// `[b, 1, h, d]`, every entry the position: the scatter index for the new key and value.
    kv_idx: Tensor<i64, B>,
    /// `[1, kv]`: `0` for the filled slots, the current one included, `-inf` for the others.
    mask: Tensor<T, B>,
    rope: RotaryEmbedding<T, B>,
}

impl<T: WithDTypeF, B: Backend> DeviceStep<T, B> {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    /// Records the per-step work: `pos` is `[1]`, `mask` the persistent `[1, kv]` mask (its
    /// slot at `pos` is opened here), `rope` the full-length table of the transformer.
    pub(crate) fn new(
        pos: &Tensor<i64, B>,
        mask: &Tensor<T, B>,
        rope: &RotaryEmbedding<T, B>,
        (b, h, d): (usize, usize, usize),
    ) -> Result<Self> {
        let kv_idx = pos.reshape((1, 1, 1, 1))?.broadcast_as((b, 1, h, d))?.contiguous()?;
        let zero = Tensor::zeros((1, 1), mask.device())?;
        mask.scatter_set(&pos.reshape((1, 1))?, &zero, 1)?;
        let rope = rope.rows(pos)?;
        Ok(Self { kv_idx, mask: mask.clone(), rope })
    }
}

/// Streaming multi-head attention (used by the flow LM transformer).
pub struct StreamingMultiheadAttention<Q: BackendQ> {
    in_proj: Q::LinearQ,
    out_proj: Q::LinearQ,
    pub embed_dim: usize,
    pub num_heads: usize,
    /// Width of the queries and of the attention output, see [`Kind::FlowLm`].
    pub attn_dim: usize,
    pub num_kv_heads: usize,
    device: Q::B,
}

impl<Q: BackendQ> StreamingMultiheadAttention<Q> {
    pub fn load(
        vb: &Path<Q::B>,
        embed_dim: usize,
        num_heads: usize,
        attn_dim: usize,
        num_kv_heads: usize,
    ) -> Result<Self> {
        let kv_dim = attn_dim / num_heads * num_kv_heads;
        let in_proj = Q::linear_load(vb.pp("in_proj"), embed_dim, attn_dim + 2 * kv_dim)?;
        let out_proj = Q::linear_load(vb.pp("out_proj"), attn_dim, embed_dim)?;
        let device = vb.device().clone();
        Ok(Self { in_proj, out_proj, embed_dim, num_heads, attn_dim, num_kv_heads, device })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<StreamingMHAState<Q::T, Q::B>> {
        let dim_per_head = self.attn_dim / self.num_heads;
        let shape = (batch_size, sequence_length, self.num_heads, dim_per_head);
        let dev = &self.device;
        let k_cache = Tensor::zeros(shape, dev)?;
        let v_cache = Tensor::zeros(shape, dev)?;
        Ok(StreamingMHAState { k_cache, v_cache, current_end: 0 })
    }

    #[tracing::instrument(name = "attn", skip_all)]
    pub(crate) fn forward(
        &self,
        query: &Tensor<Q::T, Q::B>,
        rope: &RotaryEmbedding<Q::T, Q::B>,
        state: &mut StreamingMHAState<Q::T, Q::B>,
        mask: Option<&Tensor<Q::T, Q::B>>,
        step: Option<&DeviceStep<Q::T, Q::B>>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;
        let (b, t, _) = query.dims3()?;
        let (h, kvh) = (self.num_heads, self.num_kv_heads);
        let d = self.attn_dim / h;
        let projected = self.in_proj.forward(query)?;
        // Split into q, k, v by narrowing on the last dimension
        let (ad, kd) = (self.attn_dim, kvh * d);
        let q = projected.narrow(2, 0..ad)?.contiguous()?.reshape((b, t, h, d))?;
        // Each k/v head serves h / kvh consecutive query heads (torch's `repeat_interleave`). The
        // cache holds them repeated, so attention below sees as many k/v heads as query heads.
        let heads = |x: Tensor<Q::T, Q::B>| {
            if kvh == h {
                return x.reshape((b, t, h, d));
            }
            x.reshape((b, t, kvh, 1, d))?
                .broadcast_as((b, t, kvh, h / kvh, d))?
                .contiguous()?
                .reshape((b, t, h, d))
        };
        let k = heads(projected.narrow(2, ad..ad + kd)?.contiguous()?)?;
        let v = heads(projected.narrow(2, ad + kd..ad + 2 * kd)?.contiguous()?)?;

        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        if let Some(step) = step {
            let (q, k) = step.rope.forward(&q, &k)?;
            state.k_cache.scatter_set(&step.kv_idx, &k, 1)?;
            state.v_cache.scatter_set(&step.kv_idx, &v, 1)?;
            let kv = state.k_cache.dim(1usize)?;
            let keys = state.k_cache.narrow(1, 0..kv)?;
            let values = state.v_cache.narrow(1, 0..kv)?;
            let x = q.sdpa_decode(&keys, &values, Some(&step.mask), inv_sqrt_d)?;
            return self.out_proj.forward(&x);
        }

        // Apply RoPE: q, k are [b, t, h, d]
        let (q, k) = rope.forward(&q, &k)?;
        let (k, v) = state.complete_kv(&k, &v)?;

        let x = if t == 1 {
            q.sdpa_decode(&k, &v, mask, inv_sqrt_d)?
        } else {
            // Transpose to [b, h, t, d] for attention
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;

            // Scaled dot-product attention
            let scale = Q::T::from_f32(inv_sqrt_d);
            let attn = q.matmul_t(&k)?.scale(scale)?;
            let attn = match mask {
                Some(m) => attn.broadcast_add(m)?,
                None => attn,
            };
            let attn = attn.softmax()?;
            let x = attn.matmul(&v)?;

            // Back to [b, t, h*d]
            x.transpose(1, 2)?.reshape((b, t, self.attn_dim))?.contiguous()?
        };
        self.out_proj.forward(&x)
    }
}
// ---- KV Cache ----

/// Simple append-based KV cache with optional context window trimming.
#[derive(Clone, Debug)]
pub struct KvCache<T: WithDTypeF, B: Backend> {
    k: Option<Tensor<T, B>>,
    v: Option<Tensor<T, B>>,
    context: usize,
    absolute_offset: usize,
}

impl<T: WithDTypeF, B: Backend> KvCache<T, B> {
    pub fn new(context: usize) -> Self {
        Self { k: None, v: None, context, absolute_offset: 0 }
    }

    pub fn current_seq_len(&self) -> Result<usize> {
        let l = match &self.k {
            Some(k) => k.dim(2)?, // k shape: [b, h, seq, d]
            None => 0,
        };
        Ok(l)
    }

    /// Append new k, v (shape [b, h, t, d]) and return full (k, v).
    /// Trims to context if exceeded.
    pub fn append(
        &mut self,
        new_k: &Tensor<T, B>,
        new_v: &Tensor<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, new_k], 2)?;
                let v = Tensor::cat(&[prev_v, new_v], 2)?;
                (k, v)
            }
            _ => (new_k.clone(), new_v.clone()),
        };

        let new_tokens = new_k.dim(2)?;
        self.absolute_offset += new_tokens;
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    pub fn trim(&mut self) -> Result<()> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(k), Some(v)) => (k, v),
            _ => return Ok(()),
        };
        let seq_len = k.dim(2)?;
        // Keep `context - 1` positions: with the next query, a window of `context` (pocket-tts's
        // `delta < context`).
        let keep = self.context.saturating_sub(1);
        if seq_len > keep {
            let trim = seq_len - keep;
            let k = k.narrow(2, trim..trim + keep)?.contiguous()?;
            let v = v.narrow(2, trim..trim + keep)?.contiguous()?;
            self.k = Some(k);
            self.v = Some(v);
        };
        Ok(())
    }
}

// ---- State types ----

#[derive(Clone, Debug)]
pub enum LayerAttentionState<T: WithDTypeF, B: Backend> {
    Mimi(KvCache<T, B>),
    FlowLm(StreamingMHAState<T, B>),
}

#[derive(Clone, Debug)]
pub struct StreamingTransformerState<T: WithDTypeF, B: Backend> {
    pub layer_states: Vec<LayerAttentionState<T, B>>,
}

impl<T: WithDTypeF, B: Backend> StreamingTransformerState<T, B> {
    /// Count `n` more filled positions in every flow-LM layer, after a step that wrote them
    /// without moving the host-side position (see [`DeviceStep`]).
    pub fn advance(&mut self, n: usize) {
        for layer in self.layer_states.iter_mut() {
            if let LayerAttentionState::FlowLm(mha) = layer {
                mha.current_end += n;
            }
        }
    }

    /// A fresh state with room for `seq_budget` positions, holding a copy of the filled prefix.
    ///
    /// Use this rather than `Clone`: cloning an `xn` tensor shares its storage, so two
    /// generations started from one clone would write into the same KV buffers. Mimi caches are
    /// fixed-context ring buffers and are cloned as-is.
    pub fn with_seq_budget(&self, seq_budget: usize) -> Result<Self> {
        let mut layer_states = Vec::with_capacity(self.layer_states.len());
        for layer in self.layer_states.iter() {
            let copied = match layer {
                LayerAttentionState::Mimi(kv) => LayerAttentionState::Mimi(kv.clone()),
                LayerAttentionState::FlowLm(mha) => {
                    let used = mha.current_end;
                    if used > seq_budget {
                        xn::bail!(
                            "cannot fit {used} filled KV positions into a budget of {seq_budget}"
                        )
                    }
                    let (b, _, h, d) = mha.k_cache.dims4()?;
                    let dev = mha.device();
                    let k_cache = Tensor::zeros((b, seq_budget, h, d), dev)?;
                    let v_cache = Tensor::zeros((b, seq_budget, h, d), dev)?;
                    if used > 0 {
                        k_cache.slice_set(
                            &mha.k_cache.narrow(1, 0..used)?.contiguous()?,
                            1usize,
                            0,
                        )?;
                        v_cache.slice_set(
                            &mha.v_cache.narrow(1, 0..used)?.contiguous()?,
                            1usize,
                            0,
                        )?;
                    }
                    LayerAttentionState::FlowLm(StreamingMHAState {
                        k_cache,
                        v_cache,
                        current_end: used,
                    })
                }
            };
            layer_states.push(copied);
        }
        Ok(Self { layer_states })
    }
}

// ---- MimiStreamingMultiheadAttention ----
// Uses KV cache with context window.

pub struct MimiStreamingMultiheadAttention<T: WithDTypeF, B: Backend> {
    in_proj: Linear<T, B>,
    out_proj: Linear<T, B>,
    embed_dim: usize,
    num_heads: usize,
    context: usize,
}

impl<T: WithDTypeF, B: Backend> MimiStreamingMultiheadAttention<T, B> {
    pub fn load(vb: &Path<B>, embed_dim: usize, num_heads: usize, context: usize) -> Result<Self> {
        let out_dim = 3 * embed_dim;
        let in_proj = Linear::load(vb.pp("in_proj"), embed_dim, out_dim)?;
        let out_proj = Linear::load(vb.pp("out_proj"), embed_dim, embed_dim)?;
        Ok(Self { in_proj, out_proj, embed_dim, num_heads, context })
    }

    pub fn init_state(&self) -> Result<KvCache<T, B>> {
        Ok(KvCache::new(self.context))
    }

    pub(crate) fn forward(
        &self,
        query: &Tensor<T, B>,
        rope: &RotaryEmbedding<T, B>,
        state: &mut KvCache<T, B>,
        mask: Option<&Tensor<T, B>>,
    ) -> Result<Tensor<T, B>> {
        let (b, t, _) = query.dims3()?;
        let d = self.embed_dim / self.num_heads;

        let projected = self.in_proj.forward(query)?;
        let packed = projected.reshape((b, t, 3, self.num_heads, d))?;
        let q = packed.narrow(2, 0..1)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let k = packed.narrow(2, 1..2)?.contiguous()?.reshape((b, t, self.num_heads, d))?;
        let v = packed.narrow(2, 2..3)?.contiguous()?.reshape((b, t, self.num_heads, d))?;

        // RoPE on [b, t, h, d]
        let (q, k) = rope.forward(&q, &k)?;

        // To [b, h, t, d]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // KV cache with context trimming
        let (k, v) = state.append(&k, &v)?;

        // Attention with causal mask
        let scale = T::from_f32(1.0 / (d as f32).sqrt());
        let attn = q.matmul_t(&k)?.scale(scale)?;
        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };
        let attn = attn.softmax()?;
        let x = attn.matmul(&v)?;

        state.trim()?;

        let x = x.transpose(1, 2)?.reshape((b, t, self.embed_dim))?;
        self.out_proj.forward(&x)
    }
}

// ---- StreamingTransformerLayer ----

enum AttentionKind<Q: BackendQ> {
    Mimi(MimiStreamingMultiheadAttention<Q::T, Q::B>),
    FlowLm(StreamingMultiheadAttention<Q>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mimi,
    /// Queries and the attention output are `attn_dim` wide, and keys and values have
    /// `num_kv_heads` heads, each shared by `num_heads / num_kv_heads` query heads (grouped-query
    /// attention). `d_model` and `num_heads` give the usual attention.
    FlowLm {
        attn_dim: usize,
        num_kv_heads: usize,
    },
}

/// FFN activation. `erf` is the exact GELU; Pocket TTS uses `gelu(approximate="tanh")`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gelu {
    Tanh,
    #[default]
    Erf,
}

/// The tanh GELU, `0.5 x (1 + tanh(u))` with `u = sqrt(2 / pi) (x + 0.044715 x^3)`, computed as
/// `x sigmoid(2u)`: the same function, and faster on CPU than xn's `tanh` or its `gelu_erf`.
fn gelu_tanh<T: WithDTypeF, B: Backend>(x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
    let u2 =
        x.sqr()?.scale_add(T::from_f32(0.071_354_82), T::from_f32(1.595_769))?.broadcast_mul(x)?;
    u2.sigmoid()?.broadcast_mul(x)
}

pub struct StreamingTransformerLayer<Q: BackendQ> {
    self_attn: AttentionKind<Q>,
    norm1: LayerNorm<Q::T, Q::B>,
    norm2: LayerNorm<Q::T, Q::B>,
    linear1: Q::LinearQ,
    linear2: Q::LinearQ,
    layer_scale_1: Option<LayerScale<Q::T, Q::B>>,
    layer_scale_2: Option<LayerScale<Q::T, Q::B>>,
    pub gelu: Gelu,
}

impl<Q: BackendQ> StreamingTransformerLayer<Q> {
    pub fn load(
        vb: &Path<Q::B>,
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        context: Option<usize>,
        layer_scale: Option<f64>,
        kind: Kind,
    ) -> Result<Self> {
        let self_attn = match kind {
            Kind::Mimi => AttentionKind::Mimi(MimiStreamingMultiheadAttention::load(
                &vb.pp("self_attn"),
                d_model,
                num_heads,
                context.unwrap_or(250),
            )?),
            Kind::FlowLm { attn_dim, num_kv_heads } => {
                AttentionKind::FlowLm(StreamingMultiheadAttention::load(
                    &vb.pp("self_attn"),
                    d_model,
                    num_heads,
                    attn_dim,
                    num_kv_heads,
                )?)
            }
        };

        let norm1 = LayerNorm::load(vb.pp("norm1"), d_model, 1e-5)?;
        let norm2 = LayerNorm::load(vb.pp("norm2"), d_model, 1e-5)?;
        let linear1 = Q::linear_load(vb.pp("linear1"), d_model, dim_feedforward)?;
        let linear2 = Q::linear_load(vb.pp("linear2"), dim_feedforward, d_model)?;

        let layer_scale_1 = if layer_scale.is_some() {
            Some(LayerScale::load(&vb.pp("layer_scale_1"), d_model)?)
        } else {
            None
        };
        let layer_scale_2 = if layer_scale.is_some() {
            Some(LayerScale::load(&vb.pp("layer_scale_2"), d_model)?)
        } else {
            None
        };

        Ok(Self {
            self_attn,
            norm1,
            norm2,
            linear1,
            linear2,
            layer_scale_1,
            layer_scale_2,
            gelu: Gelu::default(),
        })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<LayerAttentionState<Q::T, Q::B>> {
        let s = match &self.self_attn {
            AttentionKind::Mimi(attn) => LayerAttentionState::Mimi(attn.init_state()?),
            AttentionKind::FlowLm(attn) => {
                LayerAttentionState::FlowLm(attn.init_state(batch_size, sequence_length)?)
            }
        };
        Ok(s)
    }

    #[tracing::instrument(name = "transformer-layer", skip_all)]
    pub(crate) fn forward(
        &self,
        x: &Tensor<Q::T, Q::B>,
        rope: &RotaryEmbedding<Q::T, Q::B>,
        state: &mut LayerAttentionState<Q::T, Q::B>,
        mask: Option<&Tensor<Q::T, Q::B>>,
        step: Option<&DeviceStep<Q::T, Q::B>>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;

        // Self-attention block: x + layer_scale_1(attn(norm1(x)))
        let norm1 = self.norm1.forward(x)?;
        let mut attn_out = match (&self.self_attn, state) {
            (AttentionKind::Mimi(attn), LayerAttentionState::Mimi(cache)) if step.is_none() => {
                attn.forward(&norm1, rope, cache, mask)?
            }
            (AttentionKind::FlowLm(attn), LayerAttentionState::FlowLm(mha_state)) => {
                attn.forward(&norm1, rope, mha_state, mask, step)?
            }
            _ => xn::bail!("attention kind and state type mismatch"),
        };
        if let Some(ls) = &self.layer_scale_1 {
            attn_out = ls.forward(&attn_out)?;
        }
        let x = x.add(&attn_out)?;

        // FF block: x + layer_scale_2(ff(norm2(x)))
        let norm2 = self.norm2.forward(&x)?;
        let mut ff_out = self.linear1.forward(&norm2)?;
        ff_out = match self.gelu {
            Gelu::Tanh => gelu_tanh(&ff_out)?,
            Gelu::Erf => ff_out.gelu_erf()?,
        };
        ff_out = self.linear2.forward(&ff_out)?;
        if let Some(ls) = &self.layer_scale_2 {
            ff_out = ls.forward(&ff_out)?;
        }
        x.add(&ff_out)
    }
}

// ---- StreamingTransformer ----

pub struct StreamingTransformer<Q: BackendQ> {
    pub layers: Vec<StreamingTransformerLayer<Q>>,
    max_period: f32,
    head_dim: usize,
}

impl<Q: BackendQ> StreamingTransformer<Q> {
    pub fn set_gelu(&mut self, gelu: Gelu) {
        self.layers.iter_mut().for_each(|l| l.gelu = gelu);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<Q::B>,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        layer_scale: Option<f64>,
        dim_feedforward: usize,
        context: Option<usize>,
        max_period: f32,
        kind: Kind,
    ) -> Result<Self> {
        let (attn_dim, num_kv_heads) = match kind {
            Kind::Mimi => (d_model, num_heads),
            Kind::FlowLm { attn_dim, num_kv_heads } => (attn_dim, num_kv_heads),
        };
        if num_heads == 0 || !attn_dim.is_multiple_of(num_heads) {
            xn::bail!("the config's {num_heads} attention heads do not divide its width {attn_dim}")
        }
        if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
            xn::bail!(
                "the config's {num_kv_heads} key/value heads do not divide its {num_heads} \
                 attention heads"
            )
        }
        let head_dim = attn_dim / num_heads;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(StreamingTransformerLayer::load(
                &vb.pp("layers").pp(i),
                d_model,
                num_heads,
                dim_feedforward,
                context,
                layer_scale,
                kind,
            )?);
        }

        Ok(Self { layers, head_dim, max_period })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<StreamingTransformerState<Q::T, Q::B>> {
        let layer_states = self
            .layers
            .iter()
            .map(|l| l.init_state(batch_size, sequence_length))
            .collect::<Result<Vec<_>>>()?;
        Ok(StreamingTransformerState { layer_states })
    }

    pub fn forward(
        &self,
        x: &Tensor<Q::T, Q::B>,
        state: &mut StreamingTransformerState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let mut x = x.clone();
        let (_, seq_len, _) = x.dims3()?;
        // At single-token decode every cached key is visible, so the causal mask is identically
        // zero: the FlowLm branch masks nothing at all, and the Mimi branch trims its cache to
        // `context - 1` before this point, which makes its window condition vacuous too. Building
        // that mask means filling a Vec on the host, uploading it and adding it per layer, all
        // to add zero.
        let mask = if seq_len == 1 {
            None
        } else {
            match state.layer_states.first() {
                Some(LayerAttentionState::Mimi(kv_cache)) => {
                    let kv_seq_len = kv_cache.current_seq_len()?;
                    let context = kv_cache.context;
                    // Causal mask of shape (1, 1, seq_len, kv_seq_len + seq_len) with -inf in upper triangle
                    let mask_data = (0..seq_len)
                        .flat_map(|seq_idx| {
                            let seq_idx = seq_idx + kv_seq_len;
                            (0..kv_seq_len + seq_len).map(move |attn_idx| {
                                // delta = seq_idx - attn_idx in 0..context
                                if attn_idx <= seq_idx && seq_idx < attn_idx + context {
                                    Q::T::from_f32(0.0)
                                } else {
                                    Q::T::from_f32(f32::NEG_INFINITY)
                                }
                            })
                        })
                        .collect::<Vec<_>>();
                    let mask = Tensor::from_vec(
                        mask_data,
                        (1, 1, seq_len, kv_seq_len + seq_len),
                        x.device(),
                    )?;
                    Some(mask)
                }
                Some(LayerAttentionState::FlowLm(kv_cache)) => {
                    Some(kv_cache.materialize_causal_mask(seq_len)?)
                }
                _ => None,
            }
        };
        let offset = state
            .layer_states
            .first()
            .map(|s| match s {
                LayerAttentionState::Mimi(kv_cache) => kv_cache.absolute_offset,
                LayerAttentionState::FlowLm(mha_state) => mha_state.current_end,
            })
            .unwrap_or(0);
        let rope = self.rope_at(offset, seq_len, x.device())?;
        for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            x = layer.forward(&x, &rope, layer_state, mask.as_ref(), None)?;
        }
        Ok(x)
    }

    /// The RoPE table for positions `offset..offset + len`.
    pub(crate) fn rope_at(
        &self,
        offset: usize,
        len: usize,
        device: &Q::B,
    ) -> Result<RotaryEmbedding<Q::T, Q::B>> {
        RotaryEmbedding::new(self.head_dim, offset, len, self.max_period, device)
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    /// The RoPE table for positions `0..len`, for [`DeviceStep::new`].
    pub(crate) fn rope_table(
        &self,
        len: usize,
        device: &Q::B,
    ) -> Result<RotaryEmbedding<Q::T, Q::B>> {
        self.rope_at(0, len, device)
    }

    /// The attention's number of heads and its head dim.
    pub fn heads(&self) -> (usize, usize) {
        let h = self.layers.first().map_or(0, |l| match &l.self_attn {
            AttentionKind::FlowLm(a) => a.num_heads,
            AttentionKind::Mimi(a) => a.num_heads,
        });
        (h, self.head_dim)
    }

    /// One token at the position `step` holds, see [`DeviceStep`]. Leaves the host-side
    /// position alone: the caller advances it.
    pub fn forward_step(
        &self,
        x: &Tensor<Q::T, Q::B>,
        state: &mut StreamingTransformerState<Q::T, Q::B>,
        step: &DeviceStep<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let mut x = x.clone();
        for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            x = layer.forward(&x, &step.rope, layer_state, None, Some(step))?;
        }
        Ok(x)
    }
}

// ---- ProjectedTransformer ----

pub struct ProjectedTransformer<Q: BackendQ> {
    pub transformer: StreamingTransformer<Q>,
    input_proj: Option<Linear<Q::T, Q::B>>,
    output_projs: Vec<Option<Linear<Q::T, Q::B>>>,
}

impl<Q: BackendQ> ProjectedTransformer<Q> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<Q::B>,
        input_dimension: usize,
        output_dimensions: &[usize],
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        layer_scale: Option<f64>,
        context: usize,
        max_period: f32,
        dim_feedforward: usize,
    ) -> Result<Self> {
        let transformer = StreamingTransformer::load(
            &vb.pp("transformer"),
            d_model,
            num_heads,
            num_layers,
            layer_scale,
            dim_feedforward,
            Some(context),
            max_period,
            Kind::Mimi,
        )?;

        let input_proj = if d_model != input_dimension {
            Some(Linear::load(vb.pp("input_proj"), input_dimension, d_model)?)
        } else {
            None
        };

        let mut output_projs = Vec::new();
        for (i, &out_dim) in output_dimensions.iter().enumerate() {
            if d_model == out_dim {
                output_projs.push(None);
            } else {
                let proj = Linear::load(vb.pp("output_proj").pp(i), d_model, out_dim)?;
                output_projs.push(Some(proj));
            }
        }

        Ok(Self { transformer, input_proj, output_projs })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<StreamingTransformerState<Q::T, Q::B>> {
        self.transformer.init_state(batch_size, sequence_length)
    }

    /// Forward pass. Input x is [B, C, T] (conv layout).
    #[tracing::instrument(name = "transformer", skip_all)]
    pub fn forward(
        &self,
        x: &Tensor<Q::T, Q::B>,
        state: &mut StreamingTransformerState<Q::T, Q::B>,
    ) -> Result<Vec<Tensor<Q::T, Q::B>>> {
        // [B, C, T] -> [B, T, C]
        let x = x.transpose(1, 2)?.contiguous()?;

        let x = match &self.input_proj {
            Some(proj) => proj.forward(&x)?,
            None => x,
        };

        let z = self.transformer.forward(&x, state)?;

        let mut ys = Vec::with_capacity(self.output_projs.len());
        for proj in &self.output_projs {
            let y = match proj {
                Some(p) => p.forward(&z)?,
                None => z.clone(),
            };
            ys.push(y.transpose(1, 2)?.contiguous()?);
        }
        Ok(ys)
    }
}

#[cfg(test)]
mod with_seq_budget_tests {
    use super::*;
    use xn::CpuDevice;

    fn filled_state(seq: usize, used: usize) -> StreamingTransformerState<f32, CpuDevice> {
        // k[b, s, h, d] = s + 100*h so a copied prefix is recognisable, and the
        // never-written tail stays zero.
        let (b, h, d) = (1, 2, 3);
        let data: Vec<f32> = (0..b * seq * h * d)
            .map(|i| {
                let s_ = (i / (h * d)) % seq;
                let h_ = (i / d) % h;
                (s_ + 100 * h_) as f32
            })
            .collect();
        let k = Tensor::from_vec(data.clone(), (b, seq, h, d), &CpuDevice).unwrap();
        let v = Tensor::from_vec(data.iter().map(|x| -x).collect(), (b, seq, h, d), &CpuDevice)
            .unwrap();
        StreamingTransformerState {
            layer_states: vec![LayerAttentionState::FlowLm(StreamingMHAState {
                k_cache: k,
                v_cache: v,
                current_end: used,
            })],
        }
    }

    fn flat(t: &Tensor<f32, CpuDevice>) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    #[test]
    fn copies_the_used_prefix_and_zero_fills_the_rest() {
        let small = filled_state(4, 3);
        let big = small.with_seq_budget(10).unwrap();
        let LayerAttentionState::FlowLm(mha) = &big.layer_states[0] else { panic!() };
        assert_eq!(mha.current_end, 3);
        assert_eq!(mha.k_cache.dims(), &[1, 10, 2, 3]);
        let k = flat(&mha.k_cache);
        for pos in 0..3 {
            for h in 0..2 {
                for d in 0..3 {
                    assert_eq!(k[(pos * 2 + h) * 3 + d], (pos + 100 * h) as f32, "pos {pos} h {h}");
                }
            }
        }
        assert!(k[3 * 6..].iter().all(|x| *x == 0.0));
        assert_eq!(
            flat(&mha.v_cache)[0..18],
            flat(&{
                let LayerAttentionState::FlowLm(m) = &small.layer_states[0] else { panic!() };
                m.v_cache.narrow(1, 0..3).unwrap().contiguous().unwrap()
            })[..]
        );
    }

    #[test]
    fn the_copy_does_not_share_storage_with_the_source() {
        // The whole reason this exists rather than `Clone`.
        let small = filled_state(4, 2);
        let big = small.with_seq_budget(8).unwrap();
        let LayerAttentionState::FlowLm(dst) = &big.layer_states[0] else { panic!() };
        let LayerAttentionState::FlowLm(src) = &small.layer_states[0] else { panic!() };
        let poison = Tensor::from_vec(vec![9e9f32; 6], (1, 1, 2, 3), &CpuDevice).unwrap();
        dst.k_cache.slice_set(&poison, 1usize, 0).unwrap();
        // Index 3 is (pos 0, head 1) = 100.0 in the pattern, so this also fails if the source
        // were zeros all along.
        assert_eq!(flat(&src.k_cache)[3], 100.0, "writing the copy must not reach the source");
    }

    #[test]
    fn a_budget_smaller_than_the_prefix_is_refused() {
        let err = filled_state(4, 3).with_seq_budget(2).unwrap_err().to_string();
        assert!(err.contains("3") && err.contains("2"), "{err}");
        assert!(filled_state(4, 3).with_seq_budget(3).is_ok());
    }
}

#[cfg(test)]
mod grouped_query_tests {
    use super::*;
    use xn::CpuDevice;

    type Q = xn::Unquantized<f32, CpuDevice>;

    /// Grouped-query attention, here also wider than the model, is the usual attention with each
    /// k/v head's projection rows repeated for the query heads that share it: same outputs over
    /// a prompt and then single steps.
    #[test]
    fn grouped_query_attention_is_attention_with_repeated_kv_heads() -> Result<()> {
        let (embed, h, kvh, d) = (6, 4, 2, 2);
        let (attn, group) = (h * d, h / kvh);
        let mut seed = 1u64;
        let mut rand = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                })
                .collect()
        };
        let (q_w, k_w, v_w) = (rand(attn * embed), rand(kvh * d * embed), rand(kvh * d * embed));
        let (o_w, x) = (rand(embed * attn), rand(5 * embed));
        let repeat = |w: &[f32]| -> Vec<f32> {
            (0..h)
                .flat_map(|j| w[j / group * d * embed..(j / group + 1) * d * embed].to_vec())
                .collect()
        };
        let attention = |num_kv_heads, k: Vec<f32>, v: Vec<f32>| -> Result<_> {
            let in_w = [q_w.clone(), k, v].concat();
            let rows = in_w.len() / embed;
            Ok(StreamingMultiheadAttention::<Q> {
                in_proj: Linear::new(Tensor::from_vec(in_w, (rows, embed), &CpuDevice)?),
                out_proj: Linear::new(Tensor::from_vec(o_w.clone(), (embed, attn), &CpuDevice)?),
                embed_dim: embed,
                num_heads: h,
                attn_dim: attn,
                num_kv_heads,
                device: CpuDevice,
            })
        };
        let run = |m: StreamingMultiheadAttention<Q>| -> Result<Vec<f32>> {
            let mut state = m.init_state(1, 8)?;
            let mut out = Vec::new();
            for (start, len) in [(0, 3), (3, 1), (4, 1)] {
                let xs = x[start * embed..(start + len) * embed].to_vec();
                let xs = Tensor::from_vec(xs, (1, len, embed), &CpuDevice)?;
                let mask = (len > 1).then(|| state.materialize_causal_mask(len)).transpose()?;
                let rope = RotaryEmbedding::new(d, start, len, 10_000., &CpuDevice)?;
                out.extend(m.forward(&xs, &rope, &mut state, mask.as_ref(), None)?.to_vec()?);
            }
            Ok(out)
        };
        let grouped = run(attention(kvh, k_w.clone(), v_w.clone())?)?;
        let full = run(attention(h, repeat(&k_w), repeat(&v_w))?)?;
        assert_eq!(grouped.len(), 5 * embed);
        for (a, b) in grouped.iter().zip(&full) {
            assert!((a - b).abs() < 1e-6, "{grouped:?}\n{full:?}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod bad_config_tests {
    use super::*;
    use xn::CpuDevice;
    use xn::nn::VB;

    fn load(d_model: usize, num_heads: usize, num_kv_heads: usize) -> Result<()> {
        // An empty safetensors file: the check runs before any weight is read.
        let empty = [2u64.to_le_bytes().as_slice(), b"{}"].concat();
        let vb = VB::from_bytes(vec![empty], CpuDevice)?.root();
        StreamingTransformer::<xn::Unquantized<f32, CpuDevice>>::load(
            &vb,
            d_model,
            num_heads,
            1,
            None,
            4,
            None,
            10_000.,
            Kind::FlowLm { attn_dim: d_model, num_kv_heads },
        )
        .map(drop)
    }

    #[test]
    fn heads_that_do_not_divide_the_width_are_an_error() {
        for heads in [0, 3] {
            let err = load(8, heads, 1).err().unwrap_or_else(|| panic!("{heads} heads for 8 wide"));
            assert!(err.to_string().contains("heads"), "{err}");
        }
    }

    #[test]
    fn kv_heads_that_do_not_divide_the_heads_are_an_error() {
        for kv_heads in [0, 3] {
            let err = load(8, 4, kv_heads).err().unwrap_or_else(|| panic!("{kv_heads} kv heads"));
            assert!(err.to_string().contains("key/value heads"), "{err}");
        }
    }
}
