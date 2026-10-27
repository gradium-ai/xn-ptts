use std::sync::{Mutex, PoisonError};
use xn::nn::{LayerNorm, Linear, var_builder::Path};
use xn::{Backend, D, Result, Tensor, WithDTypeF};

fn modulate<T: WithDTypeF, B: Backend>(
    x: &Tensor<T, B>,
    shift: &Tensor<T, B>,
    scale: &Tensor<T, B>,
) -> Result<Tensor<T, B>> {
    let one_plus_scale = scale.add_scalar(T::from_f32(1.0))?;
    x.broadcast_mul(&one_plus_scale)?.broadcast_add(shift)
}

// ---- TimestepEmbedder ----

/// The RMS norm closing each time embedder, which has to match the training code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeRmsNorm {
    /// `x * a / sqrt(mean(x^2) + eps)`, uncentred, as audiocraft (every phonon model) computes
    /// it. Phonon exports set it in their config.
    Rms,
    /// `x * a / sqrt(var(x) + eps)`, the variance centred and unbiased, as pocket-tts computes it
    /// (`torch.var`). The default, so pocket-tts checkpoints load unchanged. Before the choice
    /// existed this crate computed this with `eps` scaled by `n / (n - 1)`, for every checkpoint.
    #[default]
    RmsVar,
}

pub struct TimestepEmbedder<T: WithDTypeF, B: Backend> {
    linear1: Linear<T, B>,
    linear2: Linear<T, B>,
    alpha: Tensor<T, B>,
    norm: TimeRmsNorm,
    freqs: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> TimestepEmbedder<T, B> {
    pub fn load(
        vb: &Path<B>,
        hidden_size: usize,
        frequency_embedding_size: usize,
        norm: TimeRmsNorm,
    ) -> Result<Self> {
        let mlp = vb.pp("mlp");
        let linear1 = Linear::load_b(mlp.pp("0"), frequency_embedding_size, hidden_size)?;
        let linear2 = Linear::load_b(mlp.pp("2"), hidden_size, hidden_size)?;

        let alpha = mlp.tensor("3.alpha", (hidden_size,))?;
        let freqs = vb.tensor("freqs", (frequency_embedding_size / 2,))?;
        Ok(Self { linear1, linear2, alpha, norm, freqs })
    }

    #[tracing::instrument(name = "ts-embedder", skip_all)]
    pub fn forward(&self, t: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // t: [..., 1] -> frequency embedding
        let args = t.broadcast_mul(&self.freqs)?;
        let cos = args.cos()?;
        let sin = args.sin()?;
        let embedding = Tensor::cat(&[&cos, &sin], D::Minus1)?;
        let mut x = self.linear1.forward(&embedding)?;
        x = x.silu()?;
        x = self.linear2.forward(&x)?;
        self.normalize(&x)
    }

    fn normalize(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let eps = T::from_f32(1e-5);
        let last = x.rank() - 1;
        let n = x.dims()[last] as f32;
        match self.norm {
            TimeRmsNorm::Rms => {
                let ms = x.sqr()?.sum_keepdim(vec![last])?.scale(T::from_f32(1.0 / n))?;
                x.broadcast_mul(&ms.add_scalar(eps)?.rsqrt()?)?.broadcast_mul(&self.alpha)
            }
            TimeRmsNorm::RmsVar => {
                let mean = x.sum_keepdim(vec![last])?.scale(T::from_f32(1.0 / n))?;
                let var = x.broadcast_sub(&mean)?.sqr()?.sum_keepdim(vec![last])?;
                let inv = var.scale(T::from_f32(1.0 / (n - 1.0)))?.add_scalar(eps)?.rsqrt()?;
                x.broadcast_mul(&inv)?.broadcast_mul(&self.alpha)
            }
        }
    }
}

// ---- ResBlock ----

pub struct ResBlock<T: WithDTypeF, B: Backend> {
    in_ln: LayerNorm<T, B>,
    mlp_linear1: Linear<T, B>,
    mlp_linear2: Linear<T, B>,
    ada_ln_silu_linear: Linear<T, B>,
}

impl<T: WithDTypeF, B: Backend> ResBlock<T, B> {
    pub fn load(vb: &Path<B>, channels: usize) -> Result<Self> {
        let in_ln = LayerNorm::load(vb.pp("in_ln"), channels, 1e-6)?;
        let mlp = vb.pp("mlp");
        let mlp_linear1 = Linear::load_b(mlp.pp("0"), channels, channels)?;
        let mlp_linear2 = Linear::load_b(mlp.pp("2"), channels, channels)?;
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear = Linear::load_b(ada.pp("1"), channels, 3 * channels)?;
        Ok(Self { in_ln, mlp_linear1, mlp_linear2, ada_ln_silu_linear })
    }

    #[tracing::instrument(name = "resblock", skip_all)]
    pub fn forward(&self, x: &Tensor<T, B>, y: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let ada = self.ada_ln_silu_linear.forward(&y.silu()?)?;
        let channels = x.dim(xn::D::Minus1)?;
        let shift_mlp = ada.narrow(ada.rank() - 1, 0..channels)?.contiguous()?;
        let scale_mlp = ada.narrow(ada.rank() - 1, channels..2 * channels)?.contiguous()?;
        let gate_mlp = ada.narrow(ada.rank() - 1, 2 * channels..3 * channels)?.contiguous()?;

        // h = modulate(ln(x), shift, scale)
        let h = self.in_ln.forward(x)?;
        let h = modulate(&h, &shift_mlp, &scale_mlp)?;

        // MLP
        let h = self.mlp_linear1.forward(&h)?;
        let h = h.silu()?;
        let h = self.mlp_linear2.forward(&h)?;
        x.add(&gate_mlp.broadcast_mul(&h)?)
    }
}

// ---- FinalLayer ----

pub struct FinalLayer<T: WithDTypeF, B: Backend> {
    norm_final: LayerNorm<T, B>,
    linear: Linear<T, B>,
    ada_ln_silu_linear: Linear<T, B>,
}

impl<T: WithDTypeF, B: Backend> FinalLayer<T, B> {
    pub fn load(vb: &Path<B>, model_channels: usize, out_channels: usize) -> Result<Self> {
        let zeros = Tensor::zeros(model_channels, vb.device())?;
        let ones = zeros.add_scalar(T::from_f32(1.0))?;
        let norm_final = LayerNorm::new(ones, zeros, 1e-6)?;
        let linear = Linear::load_b(vb.pp("linear"), model_channels, out_channels)?;
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear = Linear::load_b(ada.pp("1"), model_channels, 2 * model_channels)?;
        Ok(Self { norm_final, linear, ada_ln_silu_linear })
    }

    pub fn forward(&self, x: &Tensor<T, B>, c: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let ada = self.ada_ln_silu_linear.forward(&c.silu()?)?;
        let model_channels = x.dim(D::Minus1)?;
        let shift = ada.narrow(D::Minus1, 0..model_channels)?.contiguous()?;
        let scale = ada.narrow(D::Minus1, model_channels..2 * model_channels)?.contiguous()?;
        let x = self.norm_final.forward(x)?;
        let x = modulate(&x, &shift, &scale)?;
        self.linear.forward(&x)
    }
}

// ---- SimpleMLPAdaLN ----

/// Time conditionings by the bits of their times.
type TimeConds<T, B> = Mutex<Vec<(Vec<u32>, Tensor<T, B>)>>;

pub struct SimpleMLPAdaLN<T: WithDTypeF, B: Backend> {
    time_embeds: Vec<TimestepEmbedder<T, B>>,
    /// Each time conditioning computed so far. It depends on the times alone, and the sampler
    /// asks for the same few every step.
    time_conds: TimeConds<T, B>,
    cond_embed: Linear<T, B>,
    input_proj: Linear<T, B>,
    res_blocks: Vec<ResBlock<T, B>>,
    final_layer: FinalLayer<T, B>,
    pub num_time_conds: usize,
}

impl<T: WithDTypeF, B: Backend> SimpleMLPAdaLN<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        model_channels: usize,
        out_channels: usize,
        cond_channels: usize,
        num_res_blocks: usize,
        num_time_conds: usize,
        time_norm: TimeRmsNorm,
    ) -> Result<Self> {
        let mut time_embeds = Vec::new();
        for i in 0..num_time_conds {
            time_embeds.push(TimestepEmbedder::load(
                &vb.pp("time_embed").pp(i),
                model_channels,
                256,
                time_norm,
            )?);
        }

        let cond_embed = Linear::load_b(vb.pp("cond_embed"), cond_channels, model_channels)?;
        let input_proj = Linear::load_b(vb.pp("input_proj"), in_channels, model_channels)?;
        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            res_blocks.push(ResBlock::load(&vb.pp("res_blocks").pp(i), model_channels)?);
        }
        let final_layer = FinalLayer::load(&vb.pp("final_layer"), model_channels, out_channels)?;
        let time_conds = Mutex::new(Vec::new());
        Ok(Self {
            time_embeds,
            time_conds,
            cond_embed,
            input_proj,
            res_blocks,
            final_layer,
            num_time_conds,
        })
    }

    /// The time conditioning at `ts`, one time per time embedding: the embeddings averaged, as
    /// one row. Computed once for each distinct `ts`, then kept.
    fn time_condition(&self, ts: &[f32], dev: &B) -> Result<Tensor<T, B>> {
        let key: Vec<u32> = ts.iter().map(|t| t.to_bits()).collect();
        let mut kept = self.time_conds.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, cond)) = kept.iter().find(|(k, _)| *k == key) {
            return Ok(cond.clone());
        }
        let time = |t: f32| Tensor::full(T::from_f32(t), (1, 1), dev);
        let mut cond = self.time_embeds[0].forward(&time(ts[0])?)?;
        for (embed, &t) in self.time_embeds[1..].iter().zip(&ts[1..]) {
            cond = cond.add(&embed.forward(&time(t)?)?)?;
        }
        let cond = cond.scale(T::from_f32(1.0 / self.num_time_conds as f32))?;
        kept.push((key, cond.clone()));
        Ok(cond)
    }

    /// Forward pass.
    /// c: conditioning from AR transformer
    /// ts: one time per time embedding, e.g. the start and target times
    /// x: input tensor [N, C]
    #[tracing::instrument(name = "mlp-adaln", skip_all)]
    pub fn forward(&self, c: &Tensor<T, B>, ts: &[f32], x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let mut x = self.input_proj.forward(x)?;
        let c = self.cond_embed.forward(c)?;
        let y = c.broadcast_add(&self.time_condition(ts, x.device())?)?;
        for block in &self.res_blocks {
            x = block.forward(&x, &y)?;
        }
        self.final_layer.forward(&x, &y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xn::CpuDevice;

    #[test]
    fn both_time_norms_follow_their_formulas() {
        // A row with a clear mean, where centring matters, and alpha = 2.
        let xs = [1.0f32, 2.0, 3.0, 6.0];
        let x = Tensor::from_vec(xs.to_vec(), (1, 4), &CpuDevice).unwrap();
        let alpha = Tensor::from_vec(vec![2.0f32; 4], 4, &CpuDevice).unwrap();
        let run = |norm| {
            let e = TimestepEmbedder::<f32, CpuDevice> {
                linear1: Linear::new(Tensor::zeros((1, 1), &CpuDevice).unwrap()),
                linear2: Linear::new(Tensor::zeros((1, 1), &CpuDevice).unwrap()),
                alpha: alpha.clone(),
                norm,
                freqs: Tensor::zeros(1, &CpuDevice).unwrap(),
            };
            e.normalize(&x).unwrap().flatten_all().unwrap().to_vec1().unwrap()
        };
        let (n, eps) = (4.0f32, 1e-5f32);
        let mean = xs.iter().sum::<f32>() / n;
        let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (n - 1.0);
        let ms = xs.iter().map(|v| v * v).sum::<f32>() / n;
        let close = |got: Vec<f32>, want: Vec<f32>| {
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-5, "{got:?} vs {want:?}");
            }
        };
        close(run(TimeRmsNorm::Rms), xs.iter().map(|v| 2.0 * v / (ms + eps).sqrt()).collect());
        close(run(TimeRmsNorm::RmsVar), xs.iter().map(|v| 2.0 * v / (var + eps).sqrt()).collect());
    }

    #[test]
    fn the_config_names_the_time_norms_and_defaults_to_rms_var() {
        assert_eq!(serde_json::from_str::<TimeRmsNorm>("\"rms\"").unwrap(), TimeRmsNorm::Rms);
        assert_eq!(
            serde_json::from_str::<TimeRmsNorm>("\"rms_var\"").unwrap(),
            TimeRmsNorm::RmsVar
        );
        assert_eq!(TimeRmsNorm::default(), TimeRmsNorm::RmsVar);
    }

    /// Each pair of times is computed once, and only that pair reuses it.
    #[test]
    fn a_time_condition_is_kept_by_its_times() {
        let dev = &CpuDevice;
        let tensor = |rows: usize, cols: usize| {
            let v = (0..rows * cols).map(|i| (i as f32 * 0.37).sin()).collect();
            Tensor::from_vec(v, (rows, cols), dev).unwrap()
        };
        let unused = || Linear::new(Tensor::zeros((1, 1), dev).unwrap());
        let time_embeds = (0..2)
            .map(|i| TimestepEmbedder {
                linear1: Linear::new(tensor(4, 256)),
                linear2: Linear::new(tensor(4, 4)),
                alpha: Tensor::full(1f32, 4, dev).unwrap(),
                norm: TimeRmsNorm::Rms,
                freqs: Tensor::full(1.5f32 + i as f32, 128, dev).unwrap(),
            })
            .collect();
        let norm_final = LayerNorm::new(
            Tensor::full(1f32, 1, dev).unwrap(),
            Tensor::zeros(1, dev).unwrap(),
            1e-6,
        );
        let net = SimpleMLPAdaLN::<f32, CpuDevice> {
            time_embeds,
            time_conds: Mutex::new(Vec::new()),
            cond_embed: unused(),
            input_proj: unused(),
            res_blocks: Vec::new(),
            final_layer: FinalLayer {
                norm_final: norm_final.unwrap(),
                linear: unused(),
                ada_ln_silu_linear: unused(),
            },
            num_time_conds: 2,
        };
        let cond = |ts: [f32; 2]| net.time_condition(&ts, dev).unwrap().to_vec().unwrap();
        let (a, b) = (cond([0.0, 1.0]), cond([0.25, 0.5]));
        assert_ne!(a, b);
        assert_eq!(cond([0.0, 1.0]), a);
        assert_eq!(cond([0.25, 0.5]), b);
        assert_eq!(net.time_conds.lock().unwrap().len(), 2);
    }
}
