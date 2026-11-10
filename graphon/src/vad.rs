//! Voice activity heads: for each of a few windows, the probability that no speech follows within
//! that window, an end-of-turn signal. They read some trunk layers' outputs through a layer mixer
//! and leave the trunk's text output alone, as prod's serving does.
//!
//! The mixer layer-normalizes each tapped layer, scores each normed input with a small shared MLP,
//! adds a learnt per-layer bias and sums the inputs with the softmax of those scores. Each head is a
//! stack of pre-norm residual MLP blocks (`x + linear_out(gelu(linear_in(norm(x))))`) and a
//! two-way linear; class 0 is "no speech in the window".

use crate::config::VadConfig;
use xn::nn::{LayerNorm, Linear, var_builder::Path};
use xn::{BackendQ, ModuleT, Result, Tensor, WithDTypeF};

struct Mixer<T: WithDTypeF, B: xn::Backend> {
    norms: Vec<LayerNorm<T, B>>,
    /// The per-layer bias, `[1, 1, layers]`.
    bias: Tensor<T, B>,
    scorer_in: Linear<T, B>,
    scorer_out: Linear<T, B>,
}

impl<T: WithDTypeF, B: xn::Backend> Mixer<T, B> {
    fn load(vb: &Path<B>, layers: usize, dim: usize, affine: bool) -> Result<Self> {
        let norms = (0..layers)
            .map(|i| match affine {
                true => LayerNorm::load(vb.pp("norms").pp(i), dim, 1e-5),
                false => LayerNorm::new(
                    Tensor::full(T::from_f32(1.0), (dim,), vb.device())?,
                    Tensor::zeros((dim,), vb.device())?,
                    1e-5,
                ),
            })
            .collect::<Result<Vec<_>>>()?;
        let bias = vb.tensor("weights", (layers,))?.reshape((1, 1, layers))?;
        let hidden = usize::max(dim / 32, 32);
        let scorer_in = Linear::load_b(vb.pp("scorer").pp(0), dim, hidden)?;
        let scorer_out = Linear::load(vb.pp("scorer").pp(2), hidden, 1)?;
        Ok(Self { norms, bias, scorer_in, scorer_out })
    }

    /// `[1, 1, dim]` inputs, one per tapped layer, mixed into one.
    fn forward(&self, inputs: &[Tensor<T, B>]) -> Result<Tensor<T, B>> {
        let mut normed = Vec::with_capacity(inputs.len());
        let mut scores = Vec::with_capacity(inputs.len());
        for (norm, x) in self.norms.iter().zip(inputs) {
            let x = norm.forward(x)?;
            scores.push(self.scorer_out.forward(&self.scorer_in.forward(&x)?.tanh()?)?);
            normed.push(x);
        }
        let scores = Tensor::cat(&scores.iter().collect::<Vec<_>>(), 2)?;
        let weights = scores.add(&self.bias)?.softmax()?;
        // [1, 1, layers] @ [1, layers, dim]
        let normed = Tensor::cat(&normed.iter().collect::<Vec<_>>(), 1)?;
        weights.matmul(&normed)
    }
}

struct Block<Q: BackendQ> {
    norm: LayerNorm<Q::T, Q::B>,
    linear_in: Q::LinearQ,
    linear_out: Q::LinearQ,
}

struct Head<Q: BackendQ> {
    blocks: Vec<Block<Q>>,
    linear: Linear<Q::T, Q::B>,
}

/// The heads and their mixer. The blocks' linears take the trunk's weight format `Q`.
pub struct Vad<Q: BackendQ> {
    mixer: Mixer<Q::T, Q::B>,
    heads: Vec<Head<Q>>,
    cfg: VadConfig,
}

impl<Q: BackendQ> Vad<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &VadConfig, dim: usize) -> Result<Self> {
        let mixer =
            Mixer::load(&vb.pp("extra_heads_mixer"), cfg.layers.len(), dim, cfg.mixer_affine)?;
        let heads = (0..cfg.windows.len())
            .map(|i| {
                let vb = vb.pp("extra_heads").pp(i);
                let blocks = (0..cfg.residual_blocks)
                    .map(|j| {
                        let vb = vb.pp(j);
                        Ok(Block {
                            norm: LayerNorm::load(vb.pp("norm"), dim, 1e-5)?,
                            linear_in: Q::linear_load(vb.pp("mlp").pp(0), dim, cfg.hidden_dim)?,
                            linear_out: Q::linear_load(vb.pp("mlp").pp(2), cfg.hidden_dim, dim)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let linear = Linear::load(vb.pp(cfg.residual_blocks), dim, 2)?;
                Ok(Head { blocks, linear })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { mixer, heads, cfg: cfg.clone() })
    }

    /// The trunk layers whose outputs the mixer reads, in its order.
    pub fn layers(&self) -> &[usize] {
        &self.cfg.layers
    }

    /// The windows, in seconds, in the order of [`Vad::inactivity`]'s output.
    pub fn windows(&self) -> &[f32] {
        &self.cfg.windows
    }

    /// For each window, the probability that no speech follows within it, from the outputs of
    /// [`Vad::layers`] at one step.
    #[tracing::instrument(name = "vad", skip_all)]
    pub fn inactivity(&self, taps: &[Tensor<Q::T, Q::B>]) -> Result<Vec<f32>> {
        let x = self.mixer.forward(taps)?;
        let mut probs = Vec::with_capacity(self.heads.len());
        for head in &self.heads {
            let mut y = x.clone();
            for block in &head.blocks {
                let h = block.linear_in.forward(&block.norm.forward(&y)?)?.gelu_erf()?;
                y = y.add(&block.linear_out.forward(&h)?)?;
            }
            probs.push(head.linear.forward(&y)?.softmax()?);
        }
        // One read back for all the heads: [1, 1, 2 * heads], each head's pair in turn.
        let probs = Tensor::cat(&probs.iter().collect::<Vec<_>>(), 2)?.to::<f32>()?.to_vec()?;
        Ok(probs.chunks(2).map(|p| p[0]).collect())
    }
}
