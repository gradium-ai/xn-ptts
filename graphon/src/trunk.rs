//! The trunk: a decoder over text tokens with Qwen3's attention (RMSNorm on queries and keys
//! before rope, rope rotating halves rather than pairs) and a SiLU-gated FFN. Each step reads the
//! previous text token plus one Gemimi latent frame and predicts the next text token.
//!
//! The audio-token input of the training model is masked: it is zero on every step but the
//! first, which carries the initial token's embeddings (`initial_audio_embedding`). The summed
//! conditioners are a per-stream constant (`condition_sum`), precomputed at export.

use crate::attention::{RingKv, rope_tables};
use crate::config::TrunkConfig;
use xn::nn::{Embedding, LayerNorm, RmsNorm, var_builder::Path};
use xn::{Backend, BackendQ, ModuleT, Result, Tensor, WithDTypeF};

fn rms_norm<T: WithDTypeF, B: Backend>(
    vb: &Path<B>,
    dim: usize,
    eps: f32,
) -> Result<RmsNorm<T, B>> {
    // Stored as moshi's `alpha`, shaped to broadcast against the activations.
    let alpha = vb.tensor::<T>("alpha", vb.shape("alpha").cloned().unwrap_or((dim,).into()))?;
    Ok(RmsNorm::new(alpha.reshape((dim,))?, eps))
}

struct Layer<Q: BackendQ> {
    norm1: RmsNorm<Q::T, Q::B>,
    in_proj: Q::LinearQ,
    q_norm: RmsNorm<Q::T, Q::B>,
    k_norm: RmsNorm<Q::T, Q::B>,
    out_proj: Q::LinearQ,
    norm2: RmsNorm<Q::T, Q::B>,
    gate_in: Q::LinearQ,
    gate_out: Q::LinearQ,
}

#[derive(Debug, Clone)]
pub struct TrunkState<T: WithDTypeF, B: Backend> {
    layers: Vec<RingKv<T, B>>,
    step: usize,
    /// The token the next step reads: the initial token, then each step's own output.
    prev_token: u32,
}

impl<T: WithDTypeF, B: Backend> TrunkState<T, B> {
    pub fn step(&self) -> usize {
        self.step
    }
}

pub struct Trunk<Q: BackendQ> {
    text_emb: Embedding<Q::T, Q::B>,
    text_emb_proj: Option<Q::LinearQ>,
    cond_proj: Q::LinearQ,
    cond_norm: LayerNorm<Q::T, Q::B>,
    initial_audio: Tensor<Q::T, Q::B>,
    condition_sum: Tensor<Q::T, Q::B>,
    layers: Vec<Layer<Q>>,
    out_norm: RmsNorm<Q::T, Q::B>,
    text_linear: Q::LinearQ,
    cfg: TrunkConfig,
    card: usize,
    device: Q::B,
}

impl<Q: BackendQ> Trunk<Q> {
    pub fn load(
        vb: &Path<Q::B>,
        cfg: &TrunkConfig,
        card: usize,
        latent_dim: usize,
    ) -> Result<Self> {
        if cfg.num_kv_heads != cfg.num_heads {
            xn::bail!("grouped-query attention is not supported ({} kv heads)", cfg.num_kv_heads)
        }
        let dim = cfg.dim;
        let (h, kv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let emb_dim = cfg.text_emb_low_rank.unwrap_or(dim);
        let text_emb = Embedding::load(vb.pp("text_emb"), card + 1, emb_dim)?;
        let text_emb_proj = match cfg.text_emb_low_rank {
            Some(rank) => Some(Q::linear_load(vb.pp("text_emb").pp("low_rank"), rank, dim)?),
            None => None,
        };
        let vb_cond = vb.pp("self_wav");
        let cond_proj = Q::linear_load(vb_cond.pp("output_proj"), latent_dim, dim)?;
        let cond_norm = LayerNorm::load(vb_cond.pp("output_norm"), dim, 1e-5)?;
        let initial_audio = vb.tensor("initial_audio_embedding", (dim,))?.reshape((1, 1, dim))?;
        let condition_sum = vb.tensor("condition_sum", (dim,))?.reshape((1, 1, dim))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let vb = vb.pp("transformer").pp("layers").pp(i);
            let attn = vb.pp("self_attn");
            layers.push(Layer {
                norm1: rms_norm(&vb.pp("norm1"), dim, cfg.norm_eps)?,
                in_proj: Q::linear_load(attn.pp("in_proj"), dim, (h + 2 * kv) * hd)?,
                q_norm: rms_norm(&attn.pp("q_norm"), hd, cfg.qk_norm_eps)?,
                k_norm: rms_norm(&attn.pp("k_norm"), hd, cfg.qk_norm_eps)?,
                out_proj: Q::linear_load(attn.pp("out_proj"), h * hd, dim)?,
                norm2: rms_norm(&vb.pp("norm2"), dim, cfg.norm_eps)?,
                gate_in: Q::linear_load(vb.pp("gating").pp("linear_in"), dim, 2 * cfg.ffn_hidden)?,
                gate_out: Q::linear_load(vb.pp("gating").pp("linear_out"), cfg.ffn_hidden, dim)?,
            });
        }
        let out_norm = rms_norm(&vb.pp("out_norm"), dim, cfg.norm_eps)?;
        let text_linear = Q::linear_load(vb.pp("text_linear"), dim, card)?;
        Ok(Self {
            text_emb,
            text_emb_proj,
            cond_proj,
            cond_norm,
            initial_audio,
            condition_sum,
            layers,
            out_norm,
            text_linear,
            cfg: cfg.clone(),
            card,
            device: vb.device().clone(),
        })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<TrunkState<Q::T, Q::B>> {
        if batch_size != 1 {
            xn::bail!("only one stream at a time is supported")
        }
        let (context, kv, hd) = (self.cfg.context, self.cfg.num_kv_heads, self.cfg.head_dim);
        let layers = (0..self.cfg.num_layers)
            .map(|_| RingKv::new(context, kv, hd, &self.device))
            .collect::<Result<Vec<_>>>()?;
        Ok(TrunkState { layers, step: 0, prev_token: self.card as u32 })
    }

    /// One step: a `[1, 1, latent_dim]` Gemimi frame in, the text token out.
    #[tracing::instrument(name = "trunk", skip_all)]
    pub fn step(
        &self,
        latent: &Tensor<Q::T, Q::B>,
        state: &mut TrunkState<Q::T, Q::B>,
    ) -> Result<u32> {
        let logits = self.step_logits(latent, state)?;
        let logits = logits.to::<f32>()?.to_vec()?;
        let token = argmax(&logits) as u32;
        state.prev_token = token;
        Ok(token)
    }

    /// The text logits `[1, 1, card]` of one step. Advances the state but leaves the token the
    /// next step reads to the caller ([`Trunk::step`] picks the argmax).
    pub fn step_logits(
        &self,
        latent: &Tensor<Q::T, Q::B>,
        state: &mut TrunkState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let (h, hd) = (self.cfg.num_heads, self.cfg.head_dim);
        let token = Tensor::from_vec(vec![state.prev_token as i64], (1, 1), &self.device)?;
        let mut x = self.text_emb.forward(&token)?;
        if let Some(proj) = &self.text_emb_proj {
            x = proj.forward(&x)?;
        }
        if state.step == 0 {
            x = x.add(&self.initial_audio)?;
        }
        x = x.add(&self.condition_sum)?;
        let cond = self.cond_norm.forward(&self.cond_proj.forward(latent)?)?;
        x = x.add(&cond)?;

        let (cos, sin) = rope_tables(hd, self.cfg.max_period, state.step, &self.device)?;
        for (layer, kv) in self.layers.iter().zip(state.layers.iter_mut()) {
            let y = layer.norm1.forward(&x)?;
            let qkv = layer.in_proj.forward(&y)?;
            let q = qkv.narrow(2, 0..h * hd)?.contiguous()?.reshape((1, h, 1, hd))?;
            let k = qkv.narrow(2, h * hd..2 * h * hd)?.contiguous()?.reshape((1, h, 1, hd))?;
            let v = qkv.narrow(2, 2 * h * hd..3 * h * hd)?.contiguous()?.reshape((1, 1, h, hd))?;
            let q = layer.q_norm.forward(&q)?.rope(&cos, &sin, 0)?.reshape((1, 1, h, hd))?;
            let k = layer.k_norm.forward(&k)?.rope(&cos, &sin, 0)?.reshape((1, 1, h, hd))?;
            let attn = kv.attend(&q, &k, &v)?;
            x = x.add(&layer.out_proj.forward(&attn)?)?;

            let y = layer.norm2.forward(&x)?;
            let g = layer.gate_in.forward(&y)?;
            let hidden = self.cfg.ffn_hidden;
            let gate = g.narrow(2, 0..hidden)?.contiguous()?.silu()?;
            let up = g.narrow(2, hidden..2 * hidden)?.contiguous()?;
            x = x.add(&layer.gate_out.forward(&gate.mul(&up)?)?)?;
        }
        state.step += 1;
        let x = self.out_norm.forward(&x)?;
        self.text_linear.forward(&x)
    }

    /// Feed the token the next step reads, for callers sampling from [`Trunk::step_logits`].
    pub fn set_token(&self, state: &mut TrunkState<Q::T, Q::B>, token: u32) {
        state.prev_token = token;
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}
