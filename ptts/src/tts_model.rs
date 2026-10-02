use crate::conditioners::LUTConditioner;
use crate::flow_lm::{FlowLM, FlowLMConfig, FlowLMState};
use crate::mimi::{MimiConfig, MimiDecoder, MimiDecoderState, MimiEncoder};
use xn::nn::{Linear, var_builder::Path};
use xn::{BackendQ, Result, Tensor, Unquantized};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FuserConfig {
    pub sum: Vec<String>,
    pub streaming_sum: Vec<String>,
    pub prepend: Vec<String>,
    pub cross: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LutConditioner {
    pub n_bins: usize,
    pub dim: usize,
    pub possible_values: Vec<String>,
    pub tokenizer: String,
    /// What an unknown value maps to in training (`''` is the padding slot). Informational.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_value: Option<String>,
}

fn default_continuous_max_period() -> f32 {
    10000.0
}

/// A float attribute embedded with audiocraft's `create_sin_embedding` at `scale_factor *
/// value` (audiocraft `ContinuousAttributeConditioner`), e.g. `duration_delta`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ContinuousConditioner {
    pub scale_factor: f32,
    pub dim: usize,
    #[serde(default = "default_continuous_max_period")]
    pub max_period: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConditionerInnerConfig {
    Lut { lut: LutConditioner },
    Continuous { continuous: ContinuousConditioner },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConditionerConfig {
    pub name: String,
    #[serde(flatten)]
    pub inner: ConditionerInnerConfig,
}

fn default_audio_prompt_min_duration() -> f32 {
    10.0
}

fn default_audio_prompt_max_duration() -> f32 {
    10.0
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelId {
    pub sig: String,
    pub epoch: usize,
    pub mimi_sig: String,
    pub mimi_epoch: usize,
}

/// Optional separate Mimi codec used only for speaker (voice-prompt) encoding.
/// When set, `MimiEnc` loads its encoder weights from `prefix` and uses
/// `mimi` as the codec config, instead of the main `TTSConfig.mimi`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpeakerMimiConfig {
    /// Weight-name prefix for the speaker mimi (e.g. `"speaker_mimi"`).
    pub prefix: String,
    pub mimi: MimiConfig,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TTSConfig {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    pub temp: f32,
    pub lsd_decode_steps: usize,
    pub eos_threshold: f32,
    pub fuser: FuserConfig,
    pub conditioners: Vec<ConditionerConfig>,
    pub model_id: Option<ModelId>,
    /// Minimum allowed duration in seconds for an audio prompt passed to
    /// `get_state_for_audio`. If zero, an empty audio prompt is allowed, in
    /// which case the conditioned state skips the `prompt_audio` call entirely.
    #[serde(default = "default_audio_prompt_min_duration")]
    pub audio_prompt_min_duration: f32,
    /// Maximum allowed duration in seconds for an audio prompt. Frontends that
    /// trim long audio (e.g. the `ptts` example) should trim to this
    /// value rather than a hardcoded 10s.
    #[serde(default = "default_audio_prompt_max_duration")]
    pub audio_prompt_max_duration: f32,
    /// If true, the CFG null state is built without any audio prompting (the
    /// `prompt_audio` step is skipped on the null state). If false, the null
    /// state is prompted with the encoding of a zero waveform matching the
    /// real audio prompt's length, which preserves the historical behavior.
    #[serde(default)]
    pub cfg_null_audio_empty: bool,
    /// Optional, when set the speaker encoder loads from this prefix using
    /// this dedicated `MimiConfig` rather than the main `mimi` codec.
    #[serde(default)]
    pub speaker_mimi: Option<SpeakerMimiConfig>,
}

impl TTSConfig {
    pub fn v202601(temp: f32) -> Self {
        Self {
            flow_lm: FlowLMConfig {
                d_model: 1024,
                num_heads: 16,
                num_layers: 6,
                dim_feedforward: 4096,
                max_period: 10000.0,
                n_bins: 4000,
                lut_dim: 1024,
                flow_dim: 512,
                flow_depth: 6,
                ldim: 32,
            },
            mimi: MimiConfig {
                channels: 1,
                sample_rate: 24000,
                frame_rate: 12.5,
                dimension: 512,
                quantizer_dimension: 32,
                quantizer_output_dimension: 512,
                n_filters: 64,
                n_residual_layers: 1,
                ratios: vec![6, 5, 4],
                kernel_size: 7,
                last_kernel_size: 3,
                residual_kernel_size: 3,
                dilation_base: 2,
                compress: 2,
                transformer_d_model: 512,
                transformer_num_heads: 8,
                transformer_num_layers: 2,
                transformer_layer_scale: 0.01,
                transformer_context: 250,
                transformer_max_period: 10000.0,
                transformer_dim_feedforward: 2048,
                downsample_channel_wise: false,
            },
            temp,
            lsd_decode_steps: 1,
            eos_threshold: -4.0,
            conditioners: vec![],
            fuser: FuserConfig {
                sum: vec![],
                streaming_sum: vec![],
                prepend: vec![],
                cross: vec![],
            },
            model_id: None,
            audio_prompt_min_duration: 10.0,
            audio_prompt_max_duration: 10.0,
            cfg_null_audio_empty: false,
            speaker_mimi: None,
        }
    }

    pub fn model_ext(&self) -> Option<String> {
        self.model_id.as_ref().map(|id| format!("{}@{}", id.sig, id.epoch))
    }

    /// Returns the `MimiConfig` used to encode the voice prompt — the
    /// dedicated `speaker_mimi.mimi` if set, otherwise the main `mimi`.
    pub fn speaker_mimi_cfg(&self) -> &MimiConfig {
        match self.speaker_mimi.as_ref() {
            Some(s) => &s.mimi,
            None => &self.mimi,
        }
    }

    /// Returns the weight-name prefix used to load the speaker encoder.
    pub fn speaker_mimi_prefix(&self) -> &str {
        match self.speaker_mimi.as_ref() {
            Some(s) => s.prefix.as_str(),
            None => "mimi",
        }
    }
}

pub struct TTSModel<Q: BackendQ> {
    pub flow_lm: FlowLM<Q>,
    pub mimi: MimiDecoder<Unquantized<f32, Q::B>>,
    speaker_proj: Option<Linear<f32, Q::B>>,
    sum_luts: Vec<SumLut<Q>>,
    sum_continuous: Vec<SumContinuous<Q>>,
    lsd_decode_steps: usize,
    eos_threshold: f32,
}

/// A LUT conditioning summed into every audio frame whose value is chosen per state, e.g. a
/// fixed voice (audium's `config/conditioner/tts_voice_lut.yaml`). These are the `fuser.sum`
/// entries of `conditioners` other than `num_speakers`, which keeps its fixed-value path.
pub struct SumLut<Q: BackendQ> {
    pub name: String,
    pub values: Vec<String>,
    cond: LUTConditioner<Q::T, Q::B>,
}

/// A continuous conditioning summed into every audio frame whose value is chosen per state, e.g.
/// `duration_delta` (audium's `config/conditioner/tts_pocket_duration_delta.yaml`).
pub struct SumContinuous<Q: BackendQ> {
    pub name: String,
    cfg: ContinuousConditioner,
    output_proj: Linear<Q::T, Q::B>,
    learnt_padding: Option<Tensor<Q::T, Q::B>>,
}

impl<Q: BackendQ> SumContinuous<Q> {
    fn load(
        vb: &Path<Q::B>,
        name: &str,
        cfg: &ContinuousConditioner,
        d_model: usize,
    ) -> Result<Self> {
        if cfg.dim < 4 || !cfg.dim.is_multiple_of(2) {
            xn::bail!(
                "continuous conditioning '{name}': dim must be even and >= 4, got {}",
                cfg.dim
            )
        }
        let output_proj = Linear::load(vb.pp("output_proj"), cfg.dim, d_model)?;
        let learnt_padding = if vb.contains("learnt_padding") {
            Some(vb.tensor("learnt_padding", (1, 1, d_model))?)
        } else {
            None
        };
        Ok(Self { name: name.to_string(), cfg: cfg.clone(), output_proj, learnt_padding })
    }

    /// The `[1, 1, d_model]` term for `value`, a float as a string as training reads it, or
    /// `None` for a dropped attribute: the learnt padding, or nothing without one.
    fn embed(&self, value: Option<&str>) -> Result<Option<Tensor<Q::T, Q::B>>> {
        let Some(value) = value else { return Ok(self.learnt_padding.clone()) };
        let x: f32 = match value.trim().parse() {
            Ok(x) if f32::is_finite(x) => x,
            _ => xn::bail!("'{}' takes a finite number, got '{value}'", self.name),
        };
        let emb = sin_embedding(self.cfg.scale_factor * x, self.cfg.dim, self.cfg.max_period);
        let dev = self.output_proj.weight().device();
        let emb = Tensor::<f32, Q::B>::from_vec(emb, (1, 1, self.cfg.dim), dev)?.to::<Q::T>()?;
        Ok(Some(self.output_proj.forward(&emb)?))
    }
}

/// audiocraft's `create_sin_embedding` for one position: `[cos(phase), sin(phase)]` with
/// `phase_i = pos / max_period^(i / (dim/2 - 1))`.
fn sin_embedding(pos: f32, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let phases: Vec<f32> =
        (0..half).map(|i| pos / max_period.powf(i as f32 / (half - 1) as f32)).collect();
    phases.iter().map(|p| p.cos()).chain(phases.iter().map(|p| p.sin())).collect()
}

/// Refuse a summed LUT whose ids this crate would get wrong. Training's `noop` and `whitespace`
/// tokenizers give a known value its position in `possible_values` and put padding at `n_bins`
/// (audiocraft `conditioners/text.py`, `_WordToToken`); `whitespace` also splits a value on
/// spaces into several summed ids, which [`lut_id`] does not do.
fn check_sum_lut(name: &str, lut: &LutConditioner) -> Result<()> {
    match lut.tokenizer.as_str() {
        "noop" => {}
        "whitespace" => {
            if let Some(v) = lut.possible_values.iter().find(|v| v.split_whitespace().count() != 1)
            {
                xn::bail!("summed LUT '{name}': value '{v}' is not a single whitespace token")
            }
        }
        other => xn::bail!("summed LUT '{name}': unsupported tokenizer '{other}'"),
    }
    if lut.possible_values.len() > lut.n_bins {
        xn::bail!(
            "summed LUT '{name}' lists {} values but has only {} bins",
            lut.possible_values.len(),
            lut.n_bins
        )
    }
    Ok(())
}

/// The embedding row for `value` of a summed LUT, or `None` for padding with no learnt padding
/// row: training multiplies a dropped attribute's embedding by its zero mask and adds the learnt
/// padding if there is one, so without one the attribute contributes nothing.
fn lut_id(
    name: &str,
    values: &[String],
    learnt_padding_id: Option<u32>,
    value: Option<&str>,
) -> Result<Option<u32>> {
    match value {
        None => Ok(learnt_padding_id),
        Some(value) => match values.iter().position(|v| v == value) {
            Some(index) => Ok(Some(index as u32)),
            None => xn::bail!("unknown value '{value}' for '{name}', expected one of {values:?}"),
        },
    }
}

#[derive(Clone, Debug)]
pub struct TTSState<Q: BackendQ> {
    pub flow_lm_state: FlowLMState<Q>,
}

impl<Q: BackendQ> TTSModel<Q> {
    pub fn load(
        vb: &Path<Q::B>,
        tokenizer: Box<dyn crate::Tokenizer + Send + Sync>,
        cfg: &TTSConfig,
    ) -> Result<Self> {
        let flow_lm = FlowLM::load(&vb.pp("flow_lm"), tokenizer, &cfg.flow_lm)?;
        let mimi = MimiDecoder::load(&vb.pp("mimi"), &cfg.mimi)?;
        let speaker_proj = crate::loader::load_speaker_proj(vb, cfg)?;
        let mut sum_luts = vec![];
        let mut sum_continuous = vec![];
        for cond in cfg.conditioners.iter() {
            if cond.name == "num_speakers" || !cfg.fuser.sum.contains(&cond.name) {
                continue;
            }
            let lut = match &cond.inner {
                ConditionerInnerConfig::Lut { lut } => lut,
                ConditionerInnerConfig::Continuous { continuous } => {
                    let vb =
                        vb.pp(format!("flow_lm.condition_provider.conditioners.{}", cond.name));
                    let d_model = cfg.flow_lm.d_model;
                    sum_continuous.push(SumContinuous::load(&vb, &cond.name, continuous, d_model)?);
                    continue;
                }
            };
            check_sum_lut(&cond.name, lut)?;
            let vb = vb.pp(format!("flow_lm.condition_provider.conditioners.{}", cond.name));
            let lut_cond =
                LUTConditioner::load(&vb, lut.n_bins, None, lut.dim, cfg.flow_lm.d_model)?;
            sum_luts.push(SumLut {
                name: cond.name.clone(),
                values: lut.possible_values.clone(),
                cond: lut_cond,
            });
        }
        if sum_luts.len() > 1 {
            let names: Vec<_> = sum_luts.iter().map(|lut| lut.name.as_str()).collect();
            tracing::warn!(
                "several summed LUT conditionings {names:?}: `Synth` registers no voices for \
                 them, pick values with `TTSModel::set_sum_conditions`"
            );
        }

        Ok(Self {
            flow_lm,
            mimi,
            speaker_proj,
            sum_luts,
            sum_continuous,
            lsd_decode_steps: cfg.lsd_decode_steps,
            eos_threshold: cfg.eos_threshold,
        })
    }

    pub fn with_eos_threshold(mut self, eos_threshold: f32) -> Self {
        self.eos_threshold = eos_threshold;
        self
    }

    pub fn sample_rate(&self) -> usize {
        self.mimi.sample_rate
    }

    /// The checkpoint's speaker projection, when it has one. Voice files holding stored
    /// speaker latents go through it in [`crate::loader::load_voice_emb`].
    pub fn speaker_proj(&self) -> Option<&Linear<f32, Q::B>> {
        self.speaker_proj.as_ref()
    }

    /// The per-state summed LUT conditionings, see [`SumLut`].
    pub fn sum_luts(&self) -> &[SumLut<Q>] {
        &self.sum_luts
    }

    /// The per-state summed continuous conditionings, see [`SumContinuous`].
    pub fn sum_continuous(&self) -> &[SumContinuous<Q>] {
        &self.sum_continuous
    }

    /// Choose the value of every per-state summed conditioning (LUT or continuous) for `state`.
    /// A name mapped to `Some(value)` embeds that value, a float as a string for a continuous
    /// one; a name that is absent or mapped to `None` gets what training feeds for a dropped
    /// attribute (the learnt padding, or nothing when there is none), which is what a CFG null
    /// state or a voice with no LUT value needs. Naming a conditioning the model does not have,
    /// or a value it does not take, is an error rather than a silent fall back to padding.
    pub fn set_sum_conditions(
        &self,
        state: &mut TTSState<Q>,
        values: &std::collections::HashMap<String, Option<String>>,
    ) -> Result<()> {
        let known = || {
            let luts = self.sum_luts.iter().map(|lut| lut.name.as_str());
            luts.chain(self.sum_continuous.iter().map(|c| c.name.as_str()))
        };
        for name in values.keys() {
            if !known().any(|known| known == name) {
                let known: Vec<_> = known().collect();
                xn::bail!("the model has no summed conditioning '{name}', it has {known:?}")
            }
        }
        let mut total: Option<Tensor<Q::T, Q::B>> = None;
        for lut in self.sum_luts.iter() {
            let value = values.get(&lut.name).and_then(|v| v.as_deref());
            let Some(id) = lut_id(&lut.name, &lut.values, lut.cond.learnt_padding_id(), value)?
            else {
                continue;
            };
            let emb = lut.cond.embed_tokens(&[id])?;
            total = Some(match total {
                Some(total) => total.broadcast_add(&emb)?,
                None => emb,
            });
        }
        for cond in self.sum_continuous.iter() {
            let Some(emb) = cond.embed(values.get(&cond.name).and_then(|v| v.as_deref()))? else {
                continue;
            };
            total = Some(match total {
                Some(total) => total.broadcast_add(&emb)?,
                None => emb,
            });
        }
        state.flow_lm_state.extra_sum = total;
        Ok(())
    }

    /// Initialize flow LM state with the given sequence length budget. Every per-state summed
    /// LUT starts as a dropped attribute (see [`Self::set_sum_conditions`], which picks values),
    /// so a speaker-prompted voice on a LUT model still sees what training fed it.
    pub fn init_flow_lm_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<TTSState<Q>> {
        let mut state =
            TTSState { flow_lm_state: self.flow_lm.init_state(batch_size, sequence_length)? };
        self.set_sum_conditions(&mut state, &Default::default())?;
        Ok(state)
    }

    /// Run flow LM step with text tokens. Increments state.
    pub fn prompt_text(&self, state: &mut TTSState<Q>, text_tokens: &[u32]) -> Result<()> {
        let text_embeddings = self.flow_lm.conditioner.embed_tokens(text_tokens)?;
        let dev = text_embeddings.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    /// Run flow LM step with text tokens. Increments state.
    pub fn prompt_text_with_padding(
        &self,
        state: &mut TTSState<Q>,
        text_tokens: &[u32],
        pad_to: usize,
    ) -> Result<()> {
        let text_embeddings = self.flow_lm.conditioner.embed_tokens(text_tokens)?;
        let (batch_size, seq_len, dim) = text_embeddings.dims3()?;
        let padding_required = pad_to.saturating_sub(seq_len);
        let text_embeddings = if padding_required > 0
            && let Some(padding_embeds) = self.flow_lm.conditioner.learnt_padding()
        {
            let padding_embeds =
                padding_embeds.expand((batch_size, padding_required, dim))?.contiguous()?;
            Tensor::cat(&[&text_embeddings, &padding_embeds], 1)?
        } else {
            text_embeddings
        };
        let dev = text_embeddings.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    pub fn prompt_text_null(&self, state: &mut TTSState<Q>) -> Result<()> {
        let empty_text = match self.flow_lm.conditioner.learnt_padding() {
            None => xn::bail!("Model does not support null text prompt"),
            Some(p) => p,
        };
        let dev = empty_text.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, empty_text, &empty_latents)?;
        Ok(())
    }

    /// Run flow LM step with audio conditioning. Increments state.
    pub fn prompt_audio(
        &self,
        state: &mut TTSState<Q>,
        audio_conditioning: &Tensor<Q::T, Q::B>,
    ) -> Result<()> {
        // Nothing to prompt, e.g. a model conditioned on a summed voice with no voice prefix.
        // Running the backbone on zero frames fails on CUDA (CUDA_ERROR_INVALID_VALUE).
        if audio_conditioning.dims3()?.1 == 0 {
            return Ok(());
        }
        let dev = audio_conditioning.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        let text_embeddings = Tensor::cat(&[&self.empty_text()?, audio_conditioning], 1)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    /// One autoregressive step, returning the latent and the *raw* eos logit.
    /// Reads nothing back, so this is the entry point a browser can drive.
    #[allow(clippy::type_complexity)]
    pub fn generate_step_parts(
        &self,
        state: &mut TTSState<Q>,
        input: crate::flow_lm::StepInput<'_, Q>,
        rng: &mut impl crate::flow_lm::Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
        self.flow_lm.sample_next_latent_parts(
            input,
            &self.empty_text()?,
            &mut state.flow_lm_state,
            self.lsd_decode_steps,
            rng,
        )
    }

    /// Run one autoregressive generation step.
    /// Returns (next_latent [B, 1, ldim], is_eos).
    #[allow(clippy::type_complexity)]
    pub fn generate_step(
        &self,
        state: &mut TTSState<Q>,
        input: crate::flow_lm::StepInput<'_, Q>,
        rng: &mut impl crate::flow_lm::Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, bool)> {
        let (latent, eos_logit) = self.generate_step_parts(state, input, rng)?;
        Ok((latent, self.eos_from_logit(&eos_logit.to_vec()?)))
    }

    /// Threshold an eos logit the caller has brought back to the host.
    pub fn eos_from_logit(&self, eos_val: &[Q::T]) -> bool {
        crate::flow_lm::FlowLM::<Q>::eos_from_logit(eos_val, self.eos_threshold)
    }

    /// As [`Self::generate_step`], with classifier-free guidance. Reads the eos
    /// logit back every step, so unlike [`Self::generate_step_parts`] it is not
    /// browser-safe.
    #[allow(clippy::type_complexity)]
    pub fn generate_step_cfg(
        &self,
        state: &mut TTSState<Q>,
        null_state: &mut TTSState<Q>,
        cfg_coef: f32,
        input: crate::flow_lm::StepInput<'_, Q>,
        rng: &mut impl crate::flow_lm::Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, bool)> {
        let (latent, is_eos) = self.flow_lm.sample_next_latent_cfg(
            input,
            &self.empty_text()?,
            &mut state.flow_lm_state,
            &mut null_state.flow_lm_state,
            cfg_coef,
            self.lsd_decode_steps,
            rng,
            self.eos_threshold,
        )?;

        Ok((latent, is_eos))
    }

    /// Decode latent to audio using mimi (streaming).
    pub fn decode_latent(
        &self,
        latent: &Tensor<Q::T, Q::B>,
        mimi_state: &mut MimiDecoderState<f32, Q::B>,
    ) -> Result<Tensor<f32, Q::B>> {
        let denorm =
            latent.broadcast_mul(&self.flow_lm.emb_std)?.broadcast_add(&self.flow_lm.emb_mean)?;

        // [B, T, C] -> [B, C, T]
        let transposed = denorm.transpose(1, 2)?.contiguous()?;
        // Convert from Q::T to f32 for mimi
        let f32_transposed = transposed.to()?;
        let quantized = self.mimi.quantizer.forward(&f32_transposed)?;
        self.mimi.decode_from_latent_step(&quantized, mimi_state)
    }

    /// Initialize mimi streaming state.
    ///
    /// The decoder transformer's context window is fixed at load time from
    /// `MimiConfig::transformer_context`; nothing about this state is sized per call, so unlike
    /// [`Self::init_flow_lm_state`] there is no budget to pass.
    pub fn init_mimi_state(&self, batch_size: usize) -> Result<MimiDecoderState<f32, Q::B>> {
        // `sequence_length` reaches only the flow-LM attention kind, which a Mimi decoder has
        // none of, so any value here is discarded.
        self.mimi.init_state(batch_size, 0)
    }

    fn run_backbone_and_increment(
        &self,
        state: &mut TTSState<Q>,
        text_embeddings: &Tensor<Q::T, Q::B>,
        backbone_input_latents: &Tensor<Q::T, Q::B>,
    ) -> Result<()> {
        let input = self.flow_lm.input_linear.forward(backbone_input_latents)?;
        let input = Tensor::cat(&[text_embeddings, &input], 1)?;
        let _out =
            self.flow_lm.transformer.forward(&input, &mut state.flow_lm_state.transformer_state)?;
        Ok(())
    }

    pub fn device(&self) -> &Q::B {
        self.flow_lm.input_linear.device()
    }

    /// The empty text prefix a step with no new tokens is conditioned on.
    fn empty_text(&self) -> Result<Tensor<Q::T, Q::B>> {
        Tensor::zeros((1, 0, self.flow_lm.conditioner.dim), self.device())
    }
}

/// The speaker encoder: speaker-Mimi latents from audio, projected to the flow LM's width.
pub struct MimiEnc<Q: BackendQ> {
    speaker_proj: Linear<Q::T, Q::B>,
    mimi: MimiEncoder<Unquantized<f32, Q::B>>,
}

impl<Q: BackendQ> MimiEnc<Q> {
    /// Fails when the checkpoint has no speaker projection: the encoder's latents are not a
    /// voice embedding on their own, and conditioning on them makes the model babble or stop
    /// at once with no other symptom, so a missing weight is better caught here.
    pub fn load(vb: &Path<Q::B>, cfg: &TTSConfig) -> Result<Self> {
        let mimi_cfg = cfg.speaker_mimi_cfg();
        let mimi = MimiEncoder::load(&vb.pp(cfg.speaker_mimi_prefix()), mimi_cfg)?;
        if !vb.contains(crate::loader::SPEAKER_PROJ_WEIGHT) {
            xn::bail!(
                "checkpoint has a speaker encoder under `{}` but no speaker projection \
                 (`{}`), which voice cloning needs. A GGUF written by an older `quantize \
                 --no-mimi-encoder` dropped it: regenerate the GGUF from the safetensors \
                 checkpoint.",
                cfg.speaker_mimi_prefix(),
                crate::loader::SPEAKER_PROJ_WEIGHT
            )
        }
        let weights = vb.tensor(
            crate::loader::SPEAKER_PROJ_WEIGHT,
            (cfg.flow_lm.d_model, mimi_cfg.dimension),
        )?;
        Ok(Self { speaker_proj: Linear::new(weights), mimi })
    }

    /// Encode audio for voice conditioning. Returns [1, T', dim].
    pub fn encode_audio(&self, audio: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        let f32_audio = audio.to::<f32>()?;
        let encoded = self.mimi.encode_to_latent(&f32_audio)?;
        // [B, C, T] -> [B, T, C]
        let latents = encoded.transpose(1, 2)?.contiguous()?.to::<Q::T>()?;
        self.speaker_proj.forward(&latents)
    }
}

pub const MAX_TOKENS_PER_CHUNK: usize = 50;

/// Split text into sentence-aligned chunks that fit within a token budget.
///
/// This mirrors the Python `split_into_best_sentences` function: it prepares the text,
/// tokenizes it, finds sentence boundaries (after `.`, `!`, `...`, `?` tokens), then
/// greedily groups sentences into chunks of at most `max_tokens` tokens each.
pub fn split_into_best_sentences(
    tokenizer: &dyn crate::Tokenizer,
    text: &str,
    max_tokens: Option<usize>,
) -> Result<Vec<String>> {
    let max_tokens = max_tokens.unwrap_or(MAX_TOKENS_PER_CHUNK);
    let (prepared, _) = prepare_text_prompt(text);
    let prepared = prepared.trim().to_string();
    let tokens = tokenizer.encode(&prepared)?;

    // Get end-of-sentence token ids by tokenizing ".!...?" and skipping the first token
    // (the first token includes the leading space marker from sentencepiece).
    let eos_marker_tokens = tokenizer.encode(".!...?")?;
    let eos_tokens =
        if eos_marker_tokens.len() > 1 { &eos_marker_tokens[1..] } else { &eos_marker_tokens[..] };

    // Find sentence boundary indices: positions where a non-EOS token follows one or more EOS tokens.
    let mut sentence_boundaries = vec![0usize];
    let mut prev_was_eos = false;

    for (idx, &token) in tokens.iter().enumerate() {
        if eos_tokens.contains(&token) {
            prev_was_eos = true;
        } else {
            if prev_was_eos {
                sentence_boundaries.push(idx);
            }
            prev_was_eos = false;
        }
    }
    sentence_boundaries.push(tokens.len());

    // Build (token_count, sentence_text) pairs by decoding each token sub-range.
    let mut sentences = Vec::new();
    for window in sentence_boundaries.windows(2) {
        let (start, end) = (window[0], window[1]);
        let text = tokenizer.decode(&tokens[start..end])?;
        sentences.push((end - start, text));
    }

    // Greedily group sentences into chunks that stay under max_tokens.
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_token_count = 0;

    for (nb_tokens, sentence) in sentences {
        if current_chunk.is_empty() {
            current_chunk = sentence;
            current_token_count = nb_tokens;
            continue;
        }

        if current_token_count + nb_tokens > max_tokens {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = sentence;
            current_token_count = nb_tokens;
        } else {
            current_chunk.push(' ');
            current_chunk.push_str(&sentence);
            current_token_count += nb_tokens;
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    Ok(chunks)
}

/// Prepare text for generation: capitalize, add punctuation, pad short text.
pub fn prepare_text_prompt(text: &str) -> (String, usize) {
    let text = text.trim().to_string();
    if text.is_empty() {
        return (text, 3);
    }
    let text = text.replace(['\n', '\r'], " ");
    let mut text: String = text.split_whitespace().collect::<Vec<_>>().join(" ");

    let number_of_words = text.split_whitespace().count();
    let frames_after_eos = if number_of_words <= 4 { 3 } else { 1 };
    let mut chars = text.chars();
    if let Some(first) = chars.next() {
        text = first.to_uppercase().to_string() + chars.as_str();
    }
    if text.chars().last().is_some_and(|c| c.is_alphanumeric()) {
        text.push('.');
    }
    (text, frames_after_eos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_not_padded() {
        // pocket-tts prepended 8 spaces to texts of fewer than 5 words; audium-trained models
        // never see those spaces, and short texts go wrong with them.
        assert_eq!(prepare_text_prompt("not a thing"), ("Not a thing.".to_string(), 3));
        assert_eq!(
            prepare_text_prompt("one two three four five"),
            ("One two three four five.".to_string(), 1)
        );
    }

    fn lut(tokenizer: &str, n_bins: usize, values: &[&str]) -> LutConditioner {
        LutConditioner {
            n_bins,
            dim: 4,
            possible_values: values.iter().map(|v| v.to_string()).collect(),
            tokenizer: tokenizer.to_string(),
            default_value: Some(String::new()),
        }
    }

    #[test]
    fn a_value_is_its_position_and_padding_is_the_learnt_row_or_nothing() {
        let values = ["a".to_string(), "b".to_string()];
        assert_eq!(lut_id("v", &values, Some(3), Some("a")).unwrap(), Some(0));
        assert_eq!(lut_id("v", &values, Some(3), Some("b")).unwrap(), Some(1));
        // Padding is the learnt row appended after `n_bins`, never row `n_bins` itself.
        assert_eq!(lut_id("v", &values, Some(3), None).unwrap(), Some(3));
        assert_eq!(lut_id("v", &values, None, None).unwrap(), None);
    }

    #[test]
    fn an_unknown_value_is_an_error_not_padding() {
        let values = ["a".to_string()];
        let err = lut_id("v", &values, Some(2), Some("z")).unwrap_err().to_string();
        assert!(err.contains("unknown value 'z'"), "{err}");
    }

    #[test]
    fn sin_embedding_matches_audiocraft() {
        // torch: create_sin_embedding(torch.tensor([[[300.]]]), 8)
        let emb = sin_embedding(300.0, 8, 10000.0);
        let phases =
            [300.0f32, 300.0 / 10000f32.powf(1.0 / 3.0), 300.0 / 10000f32.powf(2.0 / 3.0), 0.03];
        let expected: Vec<f32> =
            phases.iter().map(|p| p.cos()).chain(phases.iter().map(|p| p.sin())).collect();
        for (a, b) in emb.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{emb:?} vs {expected:?}");
        }
        assert_eq!(sin_embedding(0.0, 4, 10000.0), [1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_continuous_config_reads_the_exported_block() {
        let cfg: ConditionerConfig = serde_json::from_str(
            r#"{"name":"duration_delta","type":"continuous",
                "continuous":{"scale_factor":1000.0,"dim":128,"zero_init":true}}"#,
        )
        .unwrap();
        let ConditionerInnerConfig::Continuous { continuous } = cfg.inner else { panic!() };
        assert_eq!((continuous.scale_factor, continuous.dim), (1000.0, 128));
        assert_eq!(continuous.max_period, 10000.0);
    }

    #[test]
    fn a_lut_is_checked_at_load() {
        check_sum_lut("v", &lut("noop", 2, &["a b", "c"])).unwrap();
        check_sum_lut("v", &lut("whitespace", 2, &["a", "c"])).unwrap();
        // More values than bins would reach the padding row or past the table.
        assert!(check_sum_lut("v", &lut("noop", 1, &["a", "b"])).is_err());
        assert!(check_sum_lut("v", &lut("whitespace", 2, &["a b"])).is_err());
        assert!(check_sum_lut("v", &lut("sentencepiece", 2, &["a"])).is_err());
    }
}
