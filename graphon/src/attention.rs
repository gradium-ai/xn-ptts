//! Causal attention for one stream over a sliding window of past positions, the keys and values
//! kept in a ring buffer: no copying of the cache as it fills or slides, and every query, one
//! position at a time, goes through `sdpa_decode`.

use xn::{Backend, Result, Tensor, WithDTypeF};

/// Keys and values of the last `context` positions, position `p` at slot `p % context`. Keys are
/// stored rotated, so their order in the buffer does not matter to attention.
#[derive(Debug, Clone)]
pub struct RingKv<T: WithDTypeF, B: Backend> {
    k: Tensor<T, B>,
    v: Tensor<T, B>,
    pos: usize,
}

impl<T: WithDTypeF, B: Backend> RingKv<T, B> {
    pub fn new(context: usize, heads: usize, head_dim: usize, device: &B) -> Result<Self> {
        let shape = (1, context, heads, head_dim);
        Ok(Self { k: Tensor::zeros(shape, device)?, v: Tensor::zeros(shape, device)?, pos: 0 })
    }

    /// The position of the next key.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Store one position's key and value, `[1, 1, heads, head_dim]`, then attend `q` (same
    /// shape) over the window that ends at it: that position and the `context - 1` before it.
    /// Returns `[1, 1, heads * head_dim]`.
    pub fn attend(
        &mut self,
        q: &Tensor<T, B>,
        k: &Tensor<T, B>,
        v: &Tensor<T, B>,
    ) -> Result<Tensor<T, B>> {
        let (_, context, _, head_dim) = self.k.dims4()?;
        let slot = self.pos % context;
        self.k.slice_set(k, 1usize, slot)?;
        self.v.slice_set(v, 1usize, slot)?;
        self.pos += 1;
        let filled = self.pos.min(context);
        let keys = self.k.narrow(1, 0..filled)?;
        let values = self.v.narrow(1, 0..filled)?;
        q.sdpa_decode(&keys, &values, None, 1f32 / (head_dim as f32).sqrt())
    }
}

/// cos and sin of rope at one position, `[1, head_dim / 2]`.
pub fn rope_tables<T: WithDTypeF, B: Backend>(
    head_dim: usize,
    max_period: f32,
    pos: usize,
    device: &B,
) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
    let half = head_dim / 2;
    let (mut cos, mut sin) = (Vec::with_capacity(half), Vec::with_capacity(half));
    for i in 0..half {
        let freq = 1f32 / max_period.powf(i as f32 / half as f32);
        let angle = pos as f32 * freq;
        cos.push(T::from_f32(angle.cos()));
        sin.push(T::from_f32(angle.sin()));
    }
    Ok((Tensor::from_vec(cos, (1, half), device)?, Tensor::from_vec(sin, (1, half), device)?))
}
