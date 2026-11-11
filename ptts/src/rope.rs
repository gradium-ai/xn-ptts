use xn::{Backend, Result, Tensor, WithDTypeF};

/// Rotary position embedding (RoPE).
/// Precomputes cos/sin tables and wraps the tensor `rope_i` primitive.
pub struct RotaryEmbedding<T: WithDTypeF, B: Backend> {
    cos: Tensor<T, B>,
    sin: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> RotaryEmbedding<T, B> {
    pub fn new(
        head_dim: usize,
        offset: usize,
        max_seq_len: usize,
        max_period: f32,
        device: &B,
    ) -> Result<Self> {
        let positions = (0..max_seq_len).map(|pos| (pos + offset) as f32);
        Self::at_positions(head_dim, positions, max_seq_len, max_period, device)
    }

    /// Tables that move `len` keys, already embedded at their positions, `shift` positions
    /// back: every row is the rotation by `-shift`. For a cache that drops `shift` positions in
    /// front of them.
    pub fn shift_back(
        head_dim: usize,
        shift: usize,
        len: usize,
        max_period: f32,
        device: &B,
    ) -> Result<Self> {
        let positions = std::iter::repeat_n(-(shift as f32), len);
        Self::at_positions(head_dim, positions, len, max_period, device)
    }

    fn at_positions(
        head_dim: usize,
        positions: impl Iterator<Item = f32>,
        len: usize,
        max_period: f32,
        device: &B,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv_freq.push(1.0f32 / max_period.powf(i as f32 / half_dim as f32));
        }

        let mut cos_data = Vec::with_capacity(len * half_dim);
        let mut sin_data = Vec::with_capacity(len * half_dim);
        for pos in positions {
            for &freq in &inv_freq {
                let angle = pos * freq;
                cos_data.push(T::from_f32(angle.cos()));
                sin_data.push(T::from_f32(angle.sin()));
            }
        }

        let cos = Tensor::from_vec(cos_data, (len, half_dim), device)?;
        let sin = Tensor::from_vec(sin_data, (len, half_dim), device)?;
        Ok(Self { cos, sin })
    }

    /// Apply RoPE to one tensor in [B, T, H, D] layout.
    pub fn rotate(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let x = x.transpose(1, 2)?.contiguous()?;
        x.rope_i(&self.cos, &self.sin, 0)?.transpose(1, 2)?.contiguous()
    }

    /// Apply RoPE to q and k tensors in [B, T, H, D] layout.
    /// Internally transposes to [B, H, T, D] for the `rope_i` primitive.
    pub fn forward(
        &self,
        q: &Tensor<T, B>,
        k: &Tensor<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        // [B, T, H, D] -> [B, H, T, D]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let q = q.rope_i(&self.cos, &self.sin, 0)?;
        let k = k.rope_i(&self.cos, &self.sin, 0)?;
        // [B, H, T, D] -> [B, T, H, D]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        Ok((q, k))
    }
}
