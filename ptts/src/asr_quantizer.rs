//! Encode side of the Mimi residual vector quantizer.
//!
//! `ptts::mimi::MimiEncoder` stops at the continuous latent because the TTS
//! runtime feeds those latents straight into the flow LM. The ASR model wants
//! discrete codes instead, so this adds the quantizer back: latents in,
//! `n_q` codebook indices per frame out. Only the encode path is needed —
//! nothing here reconstructs audio.

use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

/// One codebook of a residual VQ layer.
///
/// The checkpoint stores running sums rather than centroids, so the centroids
/// are recovered at load time. `c2` caches `|c|^2 / 2` so the nearest-centroid
/// search becomes a single matmul plus an argmin.
struct Codebook<T: WithDTypeF, B: Backend> {
    embedding: Tensor<T, B>,
    c2: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> Codebook<T, B> {
    fn load(vb: &Path<B>, dim: usize, bins: usize) -> Result<Self> {
        let cluster_usage = vb.tensor::<T>("cluster_usage", (bins,))?;
        let embedding_sum = vb.tensor::<T>("embedding_sum", (bins, dim))?;
        let epsilon = Tensor::full(T::from_f32(1e-5), (bins,), cluster_usage.device())?;
        let cluster_usage = cluster_usage.maximum(&epsilon)?.unsqueeze(1)?;
        let embedding = embedding_sum.broadcast_div(&cluster_usage)?;
        let c2 =
            embedding.sqr()?.sum_keepdim(vec![1])?.scale(T::from_f32(0.5))?.reshape((bins,))?;
        Ok(Self { embedding, c2 })
    }

    /// `xs` is `(.., dim)`, the result drops the last dimension.
    fn encode(&self, xs: &Tensor<T, B>) -> Result<Tensor<i64, B>> {
        let mut out_dims = xs.dims().to_vec();
        out_dims.pop();
        let xs = xs.flatten(0, xs.rank().saturating_sub(2))?;
        // argmin over |x - c|^2 == argmin over |c|^2/2 - x.c, the |x|^2 term is
        // constant across centroids.
        let dists = self.c2.broadcast_sub(&xs.matmul_t(&self.embedding)?)?;
        self.reshape_codes(dists.argmin(1)?, out_dims)
    }

    fn reshape_codes(&self, codes: Tensor<i64, B>, dims: Vec<usize>) -> Result<Tensor<i64, B>> {
        if dims.is_empty() { Ok(codes) } else { codes.reshape(dims) }
    }

    /// `codes` is `(..)`, the result appends the codebook dimension.
    fn decode(&self, codes: &Tensor<i64, B>) -> Result<Tensor<T, B>> {
        let mut out_dims = codes.dims().to_vec();
        out_dims.push(self.embedding.dim(1usize)?);
        let codes = codes.flatten(0, codes.rank().saturating_sub(1))?;
        self.embedding.index_select(&codes, 0)?.reshape(out_dims)
    }
}

/// A residual VQ stack behind a 1x1 input projection.
struct Rvq<T: WithDTypeF, B: Backend> {
    input_proj: Tensor<T, B>,
    layers: Vec<Codebook<T, B>>,
}

impl<T: WithDTypeF, B: Backend> Rvq<T, B> {
    fn load(vb: &Path<B>, input_dim: usize, dim: usize, n_q: usize, bins: usize) -> Result<Self> {
        let input_proj = vb.pp("input_proj").tensor("weight", (dim, input_dim, 1))?;
        let vb_l = vb.pp("vq").pp("layers");
        let layers = (0..n_q)
            .map(|i| Codebook::load(&vb_l.pp(i).pp("_codebook"), dim, bins))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { input_proj, layers })
    }

    /// `xs` is `(batch, input_dim, steps)`; one index tensor per codebook.
    ///
    /// Deliberately not stacked into a single `(batch, n_q, steps)` tensor. WGSL
    /// has no 64-bit integer, so xn's WebGPU backend computes in float dtypes
    /// only and every i64 op -- `cat`/`stack` via `copy2d`, and `to_dtype` --
    /// falls back to a host round trip. Stacking cost one such trip per codebook
    /// per frame and dominated ASR on that backend.
    ///
    /// The loop itself is entirely on-device: `decode` consumes the index tensor
    /// directly, so no value is needed on the host until the frame is finished.
    /// Reading the codes then costs one flush, with the remaining reads landing
    /// on work that has already completed.
    fn encode(&self, xs: &Tensor<T, B>) -> Result<Vec<Tensor<i64, B>>> {
        let xs = xs.conv1d(&self.input_proj, None, 1, 0, 1, 1)?;
        // Work in (batch, steps, dim) so the codebook matmul needs no transposes.
        let mut residual = xs.transpose(1, 2)?.contiguous()?;
        let mut codes = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter() {
            let indices = layer.encode(&residual)?;
            residual = residual.sub(&layer.decode(&indices)?)?;
            codes.push(indices);
        }
        Ok(codes)
    }
}

/// Mimi's split quantizer: one semantic codebook plus `n_q - 1` acoustic ones,
/// each stack fed the same latent.
pub struct MimiQuantizer<T: WithDTypeF, B: Backend> {
    rvq_first: Rvq<T, B>,
    rvq_rest: Rvq<T, B>,
}

impl<T: WithDTypeF, B: Backend> MimiQuantizer<T, B> {
    pub fn load(
        vb: &Path<B>,
        input_dim: usize,
        dim: usize,
        n_q: usize,
        bins: usize,
    ) -> Result<Self> {
        if n_q < 2 {
            xn::bail!("expected at least 2 codebooks, got {n_q}")
        }
        let rvq_first = Rvq::load(&vb.pp("rvq_first"), input_dim, dim, 1, bins)?;
        let rvq_rest = Rvq::load(&vb.pp("rvq_rest"), input_dim, dim, n_q - 1, bins)?;
        Ok(Self { rvq_first, rvq_rest })
    }

    /// `xs` is `(batch, input_dim, steps)`, the result is `(batch, n_q, steps)`.
    /// One index tensor per codebook, semantic first; see [`Rvq::encode`] for
    /// why they are not stacked.
    pub fn encode(&self, xs: &Tensor<T, B>) -> Result<Vec<Tensor<i64, B>>> {
        let mut codes = self.rvq_first.encode(xs)?;
        codes.extend(self.rvq_rest.encode(xs)?);
        Ok(codes)
    }
}
