use crate::conditioners::LUTConditioner;
use crate::mlp::SimpleMLPAdaLN;
use crate::transformer::{StreamingTransformer, StreamingTransformerState};
use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, BackendQ, Result, Tensor, WithDTypeF};

pub trait Rng {
    fn sample(&mut self) -> f32;
}

/// The [`Rng`] every frontend actually uses: seeded Gaussian noise for the flow-matching
/// sampler, with standard deviation `sqrt(temperature)`.
///
/// Named for the distribution rather than the generator, since it wraps `rand::rngs::StdRng`
/// and the two are easy to confuse.
pub struct NormalRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl NormalRng {
    pub fn new(temperature: f32, seed: u64) -> Result<Self> {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, temperature.sqrt()).map_err(xn::Error::wrap)?;
        Ok(Self { inner: rand::rngs::StdRng::seed_from_u64(seed), distr })
    }
}

impl Rng for NormalRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng as _;
        self.inner.sample(self.distr)
    }
}

/// Lets a boxed noise source be passed where an `impl Rng` is expected, so a
/// caller can choose one at runtime.
impl Rng for Box<dyn Rng + Send> {
    fn sample(&mut self) -> f32 {
        (**self).sample()
    }
}

/// Replays a fixed sequence of values, cycling when exhausted.
///
/// The sampler's only source of randomness is [`Rng`], so feeding it a recorded
/// sequence makes a generation reproducible across implementations — which is
/// how this crate is compared against the reference one step for step.
pub struct ReplayRng {
    values: Vec<f32>,
    index: usize,
}

impl ReplayRng {
    pub fn new(values: Vec<f32>) -> Result<Self> {
        if values.is_empty() {
            xn::bail!("ReplayRng needs at least one value")
        }
        Ok(Self { values, index: 0 })
    }
}

impl Rng for ReplayRng {
    fn sample(&mut self) -> f32 {
        let value = self.values[self.index % self.values.len()];
        self.index += 1;
        value
    }
}

/// Lagrangian Self Distillation decode.
/// Rebuilds the data sample from starting point x_0.
fn lsd_decode<T: WithDTypeF, B: Backend>(
    flow_net: &SimpleMLPAdaLN<T, B>,
    transformer_out: &Tensor<T, B>,
    x_0: &Tensor<T, B>,
    num_steps: usize,
) -> Result<Tensor<T, B>> {
    let mut current = x_0.clone();
    let dev = x_0.device();

    for i in 0..num_steps {
        let s_val = i as f32 / num_steps as f32;
        let t_val = (i + 1) as f32 / num_steps as f32;

        // Create s and t tensors matching x_0 shape but with last dim = 1
        let shape: Vec<usize> =
            x_0.dims().iter().copied().take(x_0.rank() - 1).chain([1]).collect();
        let s = Tensor::full(T::from_f32(s_val), shape.clone(), dev)?;
        let t = Tensor::full(T::from_f32(t_val), shape, dev)?;

        let flow_dir = flow_net.forward(transformer_out, &[&s, &t], &current)?;
        let step_scale = T::from_f32(1.0 / num_steps as f32);
        current = current.add(&flow_dir.scale(step_scale)?)?;
    }
    Ok(current)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlowLMConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dim_feedforward: usize,
    pub max_period: f32,
    pub n_bins: usize,
    pub lut_dim: usize,
    pub flow_dim: usize,
    pub flow_depth: usize,
    pub ldim: usize,
}

/// Transformer-based flow language model.
pub struct FlowLM<Q: BackendQ> {
    pub conditioner: LUTConditioner<Q::T, Q::B>,
    pub num_speakers: Option<Tensor<Q::T, Q::B>>,
    flow_net: SimpleMLPAdaLN<Q::T, Q::B>,
    pub transformer: StreamingTransformer<Q>,
    pub emb_std: Tensor<Q::T, Q::B>,
    pub emb_mean: Tensor<Q::T, Q::B>,
    /// On the device: reading it back at load would deadlock in a browser.
    bos_emb: Tensor<Q::T, Q::B>,
    pub input_linear: Linear<Q::T, Q::B>,
    out_norm_weight: Tensor<Q::T, Q::B>,
    out_norm_bias: Tensor<Q::T, Q::B>,
    out_eos: Linear<Q::T, Q::B>,
    pub dim: usize,
    pub ldim: usize,
}

/// What a sampling step is conditioned on.
///
/// Saying "first step" in the type, rather than with a sentinel the model would
/// have to read back to notice, is what keeps the step on the device. A NaN in
/// `Latent` carries no meaning and reaches the model as it is.
pub enum StepInput<'a, Q: BackendQ> {
    /// Start the sequence; every row of the batch starts from `bos_emb`.
    Bos {
        batch: usize,
    },
    Latent(&'a Tensor<Q::T, Q::B>),
}

#[derive(Clone, Debug)]
pub struct FlowLMState<Q: BackendQ> {
    pub transformer_state: StreamingTransformerState<Q::T, Q::B>,
}

impl<Q: BackendQ> FlowLM<Q> {
    pub fn load(
        vb: &Path<Q::B>,
        tokenizer: Box<dyn crate::Tokenizer + Send + Sync>,
        cfg: &FlowLMConfig,
    ) -> Result<Self> {
        let conditioner = LUTConditioner::load(
            &vb.pp("conditioner"),
            cfg.n_bins,
            Some(tokenizer),
            cfg.lut_dim,
            cfg.d_model,
        )?;
        let num_speakers = {
            let vb = vb.pp("condition_provider.conditioners.num_speakers");
            if vb.contains("embed.weight") {
                let conditioner = LUTConditioner::load(&vb, 31, None, 16, cfg.d_model)?;
                let condition_tensor = conditioner.embed_tokens(&[1])?;
                Some(condition_tensor)
            } else {
                None
            }
        };

        let flow_net = SimpleMLPAdaLN::load(
            &vb.pp("flow_net"),
            cfg.ldim,       // in_channels
            cfg.flow_dim,   // model_channels
            cfg.ldim,       // out_channels
            cfg.d_model,    // cond_channels
            cfg.flow_depth, // num_res_blocks
            2,              // num_time_conds
        )?;

        let transformer = StreamingTransformer::load(
            &vb.pp("transformer"),
            cfg.d_model,
            cfg.num_heads,
            cfg.num_layers,
            None,
            cfg.dim_feedforward,
            None,
            cfg.max_period,
            crate::transformer::Kind::FlowLm,
        )?;

        let emb_std = vb.tensor("emb_std", (cfg.ldim,))?;
        let emb_mean = vb.tensor("emb_mean", (cfg.ldim,))?;
        let bos_emb = vb.tensor("bos_emb", (cfg.ldim,))?;
        let input_linear = Linear::load(vb.pp("input_linear"), cfg.ldim, cfg.d_model)?;
        let out_norm_weight = vb.pp("out_norm").tensor("weight", (cfg.d_model,))?;
        let out_norm_bias = vb.pp("out_norm").tensor("bias", (cfg.d_model,))?;
        let out_eos = Linear::load_b(vb.pp("out_eos"), cfg.d_model, 1)?;

        Ok(Self {
            conditioner,
            num_speakers,
            flow_net,
            transformer,
            emb_std,
            emb_mean,
            bos_emb,
            input_linear,
            out_norm_weight,
            out_norm_bias,
            out_eos,
            dim: cfg.d_model,
            ldim: cfg.ldim,
        })
    }

    pub fn init_state(&self, batch_size: usize, sequence_length: usize) -> Result<FlowLMState<Q>> {
        let transformer_state = self.transformer.init_state(batch_size, sequence_length)?;
        Ok(FlowLMState { transformer_state })
    }

    /// Run the backbone: concat text_embeddings + input, run transformer, strip prefix.
    fn backbone(
        &self,
        input: &Tensor<Q::T, Q::B>,
        text_embeddings: &Tensor<Q::T, Q::B>,
        seq_len: usize,
        state: &mut FlowLMState<Q>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let input = match self.num_speakers.as_ref() {
            Some(ns) => input.broadcast_add(ns)?,
            None => input.clone(),
        };
        let input = Tensor::cat(&[text_embeddings, &input], 1)?;
        let out = self.transformer.forward(&input, &mut state.transformer_state)?;
        let out = out.layer_norm(&self.out_norm_weight, &self.out_norm_bias, 1e-5)?;
        // Remove prefix, keep only last seq_len positions
        let total = out.dim(1usize)?;
        let start = total - seq_len;
        out.narrow(1, start..total)?.contiguous()
    }

    fn step_embedding(&self, input: StepInput<'_, Q>) -> Result<Tensor<Q::T, Q::B>> {
        match input {
            StepInput::Latent(t) => Ok(t.clone()),
            StepInput::Bos { batch } => self
                .bos_emb
                .reshape((1, 1, self.ldim))?
                .broadcast_as((batch, 1, self.ldim))?
                .contiguous(),
        }
    }

    /// The tail both samplers share. `t_out` is the backbone's last position, `[b, dim]`.
    #[allow(clippy::type_complexity)]
    fn sample_from(
        &self,
        t_out: &Tensor<Q::T, Q::B>,
        b: usize,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
        let eos_logit = self.out_eos.forward(t_out)?;
        let noise_data: Vec<Q::T> =
            (0..b * self.ldim).map(|_| Q::T::from_f32(rng.sample())).collect();
        let noise = Tensor::from_vec(noise_data, (b, self.ldim), t_out.device())?;
        let latent = lsd_decode(&self.flow_net, t_out, &noise, lsd_decode_steps)?;
        Ok((latent.reshape((b, 1, self.ldim))?, eos_logit))
    }

    /// Records a sampling step, returning the latent and the *raw* eos logit.
    ///
    /// Thresholding the logit here would mean reading it back, which a browser
    /// cannot block on. Every other op only records into the backend's batch, so
    /// leaving it a tensor is what makes the whole step callable from one.
    #[allow(clippy::type_complexity)]
    pub fn sample_next_latent_parts(
        &self,
        input: StepInput<'_, Q>,
        text_embeddings: &Tensor<Q::T, Q::B>,
        state: &mut FlowLMState<Q>,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
        let sequence = self.step_embedding(input)?;
        let (b, s, _) = sequence.dims3()?;

        let input = self.input_linear.forward(&sequence)?;
        let t_out = self.backbone(&input, text_embeddings, s, state)?;
        let t_len = t_out.dim(1usize)?;
        let t_out = t_out.narrow(1, t_len - 1..t_len)?.contiguous()?.reshape((b, self.dim))?;

        self.sample_from(&t_out, b, lsd_decode_steps, rng)
    }

    /// Threshold an eos logit already on the host. Only row 0 decides; an empty
    /// slice is never eos.
    pub fn eos_from_logit(eos_val: &[Q::T], eos_threshold: f32) -> bool {
        eos_val.first().is_some_and(|v| v.to_f32() > eos_threshold)
    }

    /// Resolves eos itself, so unlike [`Self::sample_next_latent_parts`] it reads
    /// back once per step and a browser cannot drive it. Nothing needs that yet.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn sample_next_latent_cfg(
        &self,
        input: StepInput<'_, Q>,
        text_embeddings: &Tensor<Q::T, Q::B>,
        state: &mut FlowLMState<Q>,
        null_state: &mut FlowLMState<Q>,
        cfg_coef: f32,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
        eos_threshold: f32,
    ) -> Result<(Tensor<Q::T, Q::B>, bool)> {
        let sequence = self.step_embedding(input)?;
        let (b, s, _) = sequence.dims3()?;

        let x = self.input_linear.forward(&sequence)?;
        let t_out = self.backbone(&x, text_embeddings, s, state)?;
        let t_len = t_out.dim(1usize)?;
        let t_out = t_out.narrow(1, t_len - 1..t_len)?.contiguous()?.reshape((b, self.dim))?;
        let null_out = self.backbone(&x, text_embeddings, s, null_state)?;
        let null_out =
            null_out.narrow(1, t_len - 1..t_len)?.contiguous()?.reshape((b, self.dim))?;
        let t_out = t_out.sub(&null_out)?.scale(Q::T::from_f32(cfg_coef))?.add(&null_out)?;

        let (latent, eos_logit) = self.sample_from(&t_out, b, lsd_decode_steps, rng)?;
        Ok((latent, Self::eos_from_logit(&eos_logit.to_vec()?, eos_threshold)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_rng_is_reproducible_from_its_seed() {
        let mut a = NormalRng::new(0.7, 42).unwrap();
        let mut b = NormalRng::new(0.7, 42).unwrap();
        for _ in 0..16 {
            assert_eq!(a.sample(), b.sample());
        }
    }

    #[test]
    fn normal_rng_differs_between_seeds() {
        let mut a = NormalRng::new(0.7, 1).unwrap();
        let mut b = NormalRng::new(0.7, 2).unwrap();
        let xs: Vec<f32> = (0..8).map(|_| a.sample()).collect();
        let ys: Vec<f32> = (0..8).map(|_| b.sample()).collect();
        assert_ne!(xs, ys);
    }

    #[test]
    fn zero_temperature_removes_the_noise() {
        // std = sqrt(0) = 0, so the sampler becomes deterministic rather than erroring.
        let mut rng = NormalRng::new(0.0, 7).unwrap();
        for _ in 0..8 {
            assert_eq!(rng.sample(), 0.0);
        }
    }

    #[test]
    fn replay_rng_cycles_and_rejects_empty() {
        let mut rng = ReplayRng::new(vec![1.0, 2.0]).unwrap();
        assert_eq!([rng.sample(), rng.sample(), rng.sample()], [1.0, 2.0, 1.0]);
        assert!(ReplayRng::new(vec![]).is_err());
    }

    #[test]
    fn a_boxed_rng_forwards_to_its_inner_source() {
        let mut boxed: Box<dyn Rng + Send> = Box::new(ReplayRng::new(vec![3.0, 4.0]).unwrap());
        assert_eq!([boxed.sample(), boxed.sample()], [3.0, 4.0]);
    }
}
