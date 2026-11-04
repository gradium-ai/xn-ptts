//! The Gemimi encoder: Mimi's SEANet encoder and encoder transformer, with a latent head (frames
//! stacked in pairs, LayerNorm, linear) in place of Mimi's strided downsampling. No quantizer and
//! no decoder: the trunk reads the continuous latent.
//!
//! The encoder transformer is Mimi's (LayerNorm, interleaved rope, layer scale, GELU), with its
//! attention over a ring buffer of the last `transformer_context` frames, one query frame at a
//! time.

use crate::attention::{RingKv, rope_tables};
use crate::config::GemimiConfig;
use ptts::conv::PadMode;
use ptts::seanet::{SEANetEncoder, SEANetEncoderState};
use xn::nn::{LayerNorm, var_builder::Path};
use xn::{Backend, BackendQ, ModuleT, Result, Tensor, WithDTypeF};

struct Layer<Q: BackendQ> {
    norm1: LayerNorm<Q::T, Q::B>,
    in_proj: Q::LinearQ,
    out_proj: Q::LinearQ,
    scale1: Tensor<Q::T, Q::B>,
    norm2: LayerNorm<Q::T, Q::B>,
    linear1: Q::LinearQ,
    linear2: Q::LinearQ,
    scale2: Tensor<Q::T, Q::B>,
}

pub struct Gemimi<Q: BackendQ> {
    seanet: SEANetEncoder<Q::T, Q::B>,
    input_proj: Q::LinearQ,
    layers: Vec<Layer<Q>>,
    latent_norm: LayerNorm<Q::T, Q::B>,
    latent_proj: Q::LinearQ,
    num_heads: usize,
    head_dim: usize,
    context: usize,
    max_period: f32,
    stack: usize,
    frame_size: usize,
    device: Q::B,
}

#[derive(Debug, Clone)]
pub struct GemimiState<T: WithDTypeF, B: Backend> {
    seanet: SEANetEncoderState<T, B>,
    layers: Vec<RingKv<T, B>>,
}

impl<Q: BackendQ> Gemimi<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &GemimiConfig) -> Result<Self> {
        let seanet = SEANetEncoder::load(
            &vb.pp("encoder"),
            cfg.channels,
            cfg.dimension,
            cfg.n_filters,
            cfg.n_residual_layers,
            &cfg.ratios,
            cfg.kernel_size,
            cfg.last_kernel_size,
            cfg.residual_kernel_size,
            cfg.dilation_base,
            PadMode::Constant,
            cfg.compress,
        )?;
        let d = cfg.transformer_d_model;
        if !d.is_multiple_of(cfg.transformer_num_heads) {
            xn::bail!("{} heads do not divide the encoder width {d}", cfg.transformer_num_heads)
        }
        // The transformer keeps its own width: its output projection is the identity.
        let vb_t = vb.pp("encoder_transformer");
        let input_proj = Q::linear_load(vb_t.pp("input_proj"), cfg.dimension, d)?;
        let ff = cfg.transformer_dim_feedforward;
        let mut layers = Vec::with_capacity(cfg.transformer_num_layers);
        for i in 0..cfg.transformer_num_layers {
            let vb = vb_t.pp("transformer").pp("layers").pp(i);
            layers.push(Layer {
                norm1: LayerNorm::load(vb.pp("norm1"), d, 1e-5)?,
                in_proj: Q::linear_load(vb.pp("self_attn").pp("in_proj"), d, 3 * d)?,
                out_proj: Q::linear_load(vb.pp("self_attn").pp("out_proj"), d, d)?,
                scale1: vb.pp("layer_scale_1").tensor("scale", (d,))?,
                norm2: LayerNorm::load(vb.pp("norm2"), d, 1e-5)?,
                linear1: Q::linear_load(vb.pp("linear1"), d, ff)?,
                linear2: Q::linear_load(vb.pp("linear2"), ff, d)?,
                scale2: vb.pp("layer_scale_2").tensor("scale", (d,))?,
            });
        }
        let stacked = d * cfg.latent_stack;
        let latent_norm = LayerNorm::load(vb.pp("downsample").pp("norm"), stacked, 1e-5)?;
        let latent_proj = Q::linear_load(vb.pp("downsample").pp("proj"), stacked, cfg.latent_dim)?;
        let hop: usize = cfg.ratios.iter().product();
        Ok(Self {
            seanet,
            input_proj,
            layers,
            latent_norm,
            latent_proj,
            num_heads: cfg.transformer_num_heads,
            head_dim: d / cfg.transformer_num_heads,
            context: cfg.transformer_context,
            max_period: cfg.transformer_max_period,
            stack: cfg.latent_stack,
            frame_size: hop * cfg.latent_stack,
            device: vb.device().clone(),
        })
    }

    /// Audio samples per latent frame.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn init_state(&self, batch_size: usize) -> Result<GemimiState<Q::T, Q::B>> {
        if batch_size != 1 {
            xn::bail!("only one stream at a time is supported")
        }
        let layers = (0..self.layers.len())
            .map(|_| RingKv::new(self.context, self.num_heads, self.head_dim, &self.device))
            .collect::<Result<Vec<_>>>()?;
        Ok(GemimiState { seanet: self.seanet.init_state(batch_size)?, layers })
    }

    /// `[1, 1, n * frame_size]` PCM to `[1, n, latent_dim]` latent frames.
    #[tracing::instrument(name = "gemimi", skip_all)]
    pub fn step(
        &self,
        pcm: &Tensor<Q::T, Q::B>,
        state: &mut GemimiState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let emb = self.seanet(pcm, state)?;
        let x = self.transformer(&emb, state)?;
        self.latent_head(&x)
    }

    /// The SEANet convolutions: `[1, 1, n * frame_size]` PCM to `[1, dimension, n * stack]`.
    pub fn seanet(
        &self,
        pcm: &Tensor<Q::T, Q::B>,
        state: &mut GemimiState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let len = pcm.dim(2usize)?;
        if !len.is_multiple_of(self.frame_size) {
            xn::bail!("{len} samples is not a whole number of {}-sample frames", self.frame_size)
        }
        self.seanet.forward(pcm, &mut state.seanet)
    }

    /// The encoder transformer, `[1, dimension, t]` to `[1, t, d_model]`.
    pub fn transformer(
        &self,
        emb: &Tensor<Q::T, Q::B>,
        state: &mut GemimiState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let (h, hd) = (self.num_heads, self.head_dim);
        let d = h * hd;
        let t = emb.dim(2usize)?;
        let mut x = self.input_proj.forward(&emb.transpose(1, 2)?.contiguous()?)?;
        // Every layer sees the same positions.
        let start = state.layers.first().map_or(0, |kv| kv.pos());
        let rope = (start..start + t)
            .map(|pos| rope_tables(hd, self.max_period, pos, &self.device))
            .collect::<Result<Vec<_>>>()?;
        for (layer, kv) in self.layers.iter().zip(state.layers.iter_mut()) {
            let qkv = layer.in_proj.forward(&layer.norm1.forward(&x)?)?;
            let mut rows = Vec::with_capacity(t);
            for (i, (cos, sin)) in rope.iter().enumerate() {
                let row = qkv.narrow(1, i..i + 1)?.contiguous()?;
                let part = |j: usize| row.narrow(2, j * d..(j + 1) * d)?.contiguous();
                let rotate = |x: Tensor<Q::T, Q::B>| {
                    x.reshape((1, h, 1, hd))?.rope_i(cos, sin, 0)?.reshape((1, 1, h, hd))
                };
                let q = rotate(part(0)?)?;
                let k = rotate(part(1)?)?;
                let v = part(2)?.reshape((1, 1, h, hd))?;
                rows.push(kv.attend(&q, &k, &v)?);
            }
            let attn = match rows.len() {
                1 => rows.swap_remove(0),
                _ => Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)?,
            };
            let attn = layer.out_proj.forward(&attn)?.broadcast_mul(&layer.scale1)?;
            x = x.add(&attn)?;
            let y = layer.linear1.forward(&layer.norm2.forward(&x)?)?.gelu_erf()?;
            x = x.add(&layer.linear2.forward(&y)?.broadcast_mul(&layer.scale2)?)?;
        }
        Ok(x)
    }

    /// The latent head, `[1, n * stack, d_model]` to `[1, n, latent_dim]`: each latent frame is
    /// consecutive transformer frames side by side.
    pub fn latent_head(&self, x: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        let (b, t, c) = x.dims3()?;
        let x = x.reshape((b, t / self.stack, c * self.stack))?;
        let x = self.latent_norm.forward(&x)?;
        self.latent_proj.forward(&x)
    }
}
