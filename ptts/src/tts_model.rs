//! [`TTSModel`], the flow LM and the Mimi decoder together, and [`TTSConfig`], a checkpoint's
//! `config.json`.
//!
//! It exposes generation one step at a time: prompt the state with a voice and text, step out
//! latents, decode them. [`crate::synth`] drives these steps on two threads. A caller with its
//! own event loop, such as the browser build, drives them directly.

pub use crate::conditioners::{
    ConditionerConfig, ContinuousConditionerConfig, FuserConfig, LutConditionerConfig,
};
use crate::flow_lm::{FlowLM, FlowLMConfig, FlowLMState};
use crate::mimi::{MimiConfig, MimiDecoder, MimiDecoderState, MimiEncoder};
use std::collections::HashMap;
use xn::nn::{Linear, var_builder::Path};
use xn::{BackendQ, Result, Tensor, Unquantized};

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
pub struct VoiceConfig {
    pub name: String,
    #[serde(default)]
    pub conditions: HashMap<String, String>,
    pub prefix: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TTSConfig {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    pub lsd_decode_steps: usize,
    pub eos_threshold: f32,
    pub model_id: Option<ModelId>,
    #[serde(default = "default_audio_prompt_min_duration")]
    pub audio_prompt_min_duration: f32,
    #[serde(default = "default_audio_prompt_max_duration")]
    pub audio_prompt_max_duration: f32,
    #[serde(default)]
    pub cfg_null_audio_empty: bool,
    /// If true, the CFG null branch gets no text at all. That is what a dropped transcript gives
    /// in training with the fuser's packed prefix (`pad_prepend` false): its masked positions are
    /// removed, so nothing of it reaches the model. If false, the null text is the text
    /// conditioner's learnt padding row, which preserves the historical behavior.
    #[serde(default)]
    pub cfg_null_text_empty: bool,
    /// Learnt tokens in front of the `speaker_wavs` and `transcript_in_segment` conditionings.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub prefix_sinks: std::collections::BTreeMap<String, PrefixSinkConfig>,
    #[serde(default)]
    pub speaker_mimi: Option<SpeakerMimiConfig>,
    #[serde(default)]
    pub conditioners: Vec<ConditionerConfig>,
    #[serde(default)]
    pub fuser: Option<FuserConfig>,
    #[serde(default)]
    pub voices: Vec<VoiceConfig>,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub embeddings_ext: Option<Vec<String>>,
}

impl TTSConfig {
    /// The FFN activation when the config names none: tanh for Pocket TTS checkpoints (no
    /// `model_id`), erf otherwise.
    pub fn default_gelu(&self) -> crate::transformer::Gelu {
        use crate::transformer::Gelu;
        if self.model_id.is_some() { Gelu::Erf } else { Gelu::Tanh }
    }

    fn mimi_with_gelu(&self, mimi: &MimiConfig) -> MimiConfig {
        MimiConfig { gelu: Some(mimi.gelu.unwrap_or_else(|| self.default_gelu())), ..mimi.clone() }
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
    lsd_decode_steps: usize,
    eos_threshold: f32,
    cfg_null_text_empty: bool,
    speaker_sink: Option<PrefixSink<Q>>,
    text_sink: Option<PrefixSink<Q>>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PrefixSinkConfig {
    /// Whether a `sink_dropped` token replaces the sink when the conditioning is empty.
    #[serde(default)]
    pub dropped_token: bool,
}

struct PrefixSink<Q: BackendQ> {
    sink: Tensor<Q::T, Q::B>,
    dropped: Option<Tensor<Q::T, Q::B>>,
}

impl<Q: BackendQ> PrefixSink<Q> {
    fn load(vb: &Path<Q::B>, prefix: &str, cfg: &PrefixSinkConfig, d_model: usize) -> Result<Self> {
        let sink = vb.tensor(&format!("{prefix}.sink"), (1, 1, d_model))?;
        let dropped = cfg
            .dropped_token
            .then(|| vb.tensor(&format!("{prefix}.sink_dropped"), (1, 1, d_model)))
            .transpose()?;
        Ok(Self { sink, dropped })
    }

    fn token(&self, present: bool) -> &Tensor<Q::T, Q::B> {
        match (&self.dropped, present) {
            (Some(dropped), false) => dropped,
            _ => &self.sink,
        }
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
        conditions: &HashMap<String, String>,
    ) -> Result<Self> {
        let flow_lm = FlowLM::load(&vb.pp("flow_lm"), tokenizer, cfg, conditions)?;
        let mimi = MimiDecoder::load(&vb.pp("mimi"), &cfg.mimi_with_gelu(&cfg.mimi))?;
        let speaker_proj = crate::loader::load_speaker_proj(vb, cfg)?;
        let (mut speaker_sink, mut text_sink) = (None, None);
        for (name, sink) in cfg.prefix_sinks.iter() {
            let (prefix, slot) = match name.as_str() {
                "speaker_wavs" => {
                    ("flow_lm.condition_provider.conditioners.speaker_wavs", &mut speaker_sink)
                }
                "transcript_in_segment" => ("flow_lm.conditioner", &mut text_sink),
                other => xn::bail!("unsupported prefix sink {other:?}"),
            };
            *slot = Some(PrefixSink::load(vb, prefix, sink, cfg.flow_lm.d_model)?);
        }
        if !cfg.prefix_sinks.is_empty() && !cfg.cfg_null_audio_empty {
            xn::bail!("a checkpoint with prefix_sinks needs cfg_null_audio_empty: true")
        }

        Ok(Self {
            flow_lm,
            mimi,
            speaker_proj,
            lsd_decode_steps: cfg.lsd_decode_steps,
            eos_threshold: cfg.eos_threshold,
            cfg_null_text_empty: cfg.cfg_null_text_empty,
            speaker_sink,
            text_sink,
        })
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn lsd_decode_steps(&self) -> usize {
        self.lsd_decode_steps
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

    /// Initialize flow LM state with the given sequence length budget.
    pub fn init_flow_lm_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<TTSState<Q>> {
        Ok(TTSState { flow_lm_state: self.flow_lm.init_state(batch_size, sequence_length)? })
    }

    /// Run flow LM step with text tokens. Increments state.
    pub fn prompt_text(&self, state: &mut TTSState<Q>, text_tokens: &[u32]) -> Result<()> {
        let text_embeddings = self.flow_lm.conditioner.embed_tokens(text_tokens)?;
        let text_embeddings = match &self.text_sink {
            Some(sink) => Tensor::cat(&[sink.token(!text_tokens.is_empty()), &text_embeddings], 1)?,
            None => text_embeddings,
        };
        let dev = text_embeddings.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    /// The CFG null branch's text: nothing with `cfg_null_text_empty`, as training's packed
    /// prefix gives for a dropped transcript, else the text conditioner's learnt padding row.
    pub fn prompt_text_null(&self, state: &mut TTSState<Q>) -> Result<()> {
        if let Some(sink) = &self.text_sink {
            let token = sink.token(false);
            let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), token.device())?;
            return self.run_backbone_and_increment(state, token, &empty_latents);
        }
        if self.cfg_null_text_empty {
            return Ok(());
        }
        let empty_text = match self.flow_lm.conditioner.learnt_padding() {
            None => xn::bail!("Model does not support null text prompt"),
            Some(p) => p,
        };
        let dev = empty_text.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, empty_text, &empty_latents)?;
        Ok(())
    }

    /// Positions [`Self::prompt_audio`] fills for a `frames`-long voice prompt.
    pub fn voice_prompt_len(&self, frames: usize) -> usize {
        frames + usize::from(self.speaker_sink.is_some())
    }

    /// Prompt with no voice, which only adds the voice sink's dropped token, if any.
    pub fn prompt_no_voice(&self, state: &mut TTSState<Q>) -> Result<()> {
        match self.speaker_sink {
            Some(_) => self.prompt_audio(state, &self.empty_text()?),
            None => Ok(()),
        }
    }

    /// Run flow LM step with audio conditioning. Increments state.
    pub fn prompt_audio(
        &self,
        state: &mut TTSState<Q>,
        audio_conditioning: &Tensor<Q::T, Q::B>,
    ) -> Result<()> {
        let dev = audio_conditioning.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        let text_embeddings = match &self.speaker_sink {
            Some(sink) => {
                let present = audio_conditioning.dims3()?.1 > 0;
                Tensor::cat(&[&self.empty_text()?, sink.token(present), audio_conditioning], 1)?
            }
            None => Tensor::cat(&[&self.empty_text()?, audio_conditioning], 1)?,
        };
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
        Tensor::zeros((1, 0, self.flow_lm.conditioner.output_dim), self.device())
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
        let mimi_cfg = cfg.mimi_with_gelu(cfg.speaker_mimi_cfg());
        let mimi = MimiEncoder::load(&vb.pp(cfg.speaker_mimi_prefix()), &mimi_cfg)?;
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
            (cfg.flow_lm.d_model, mimi_cfg.encoder_latent_dim()),
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

/// Prepare text for generation: collapse whitespace, capitalize and end with punctuation. Also
/// returns how many frames to generate after end of speech, more for short text.
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
        // pocket-tts prepended 8 spaces to texts of fewer than 5 words; Phonon models
        // never see those spaces, and short texts go wrong with them.
        assert_eq!(prepare_text_prompt("not a thing"), ("Not a thing.".to_string(), 3));
        assert_eq!(
            prepare_text_prompt("one two three four five"),
            ("One two three four five.".to_string(), 1)
        );
    }

    #[test]
    fn prepare_text_prompt_edge_cases() {
        let cases: &[(&str, &str, usize)] = &[
            ("", "", 3),
            ("  \n ", "", 3),
            // Only a trailing letter or digit gets a full stop; other punctuation is kept.
            ("is it?", "Is it?", 3),
            ("hello world!", "Hello world!", 3),
            ("call me at 5", "Call me at 5.", 3),
            // Line breaks, CRLF included, and runs of spaces become one space.
            ("one\r\ntwo   three\nfour five", "One two three four five.", 1),
            ("éclair au chocolat", "Éclair au chocolat.", 3),
        ];
        for &(input, text, frames) in cases {
            assert_eq!(prepare_text_prompt(input), (text.to_string(), frames), "{input:?}");
        }
    }

    #[test]
    fn prefix_sinks_read_phonon_conditioner_names() {
        let mut json = serde_json::to_value(crate::test_config()).unwrap();
        assert!(json.get("prefix_sinks").is_none(), "no sinks, no key");
        json["prefix_sinks"] = serde_json::json!({
            "speaker_wavs": {"dropped_token": true},
            "transcript_in_segment": {},
        });
        let cfg: TTSConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.prefix_sinks["speaker_wavs"].dropped_token);
        assert!(!cfg.prefix_sinks["transcript_in_segment"].dropped_token);
    }

    #[test]
    fn config_round_trips_and_fills_defaults() {
        let cfg = crate::test_config();
        let json = serde_json::to_value(&cfg).unwrap();
        let back: TTSConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&back).unwrap(), json);

        // A config written before these fields existed still loads, with their defaults.
        let mut old = json;
        let fields = old.as_object_mut().unwrap();
        for key in [
            "audio_prompt_min_duration",
            "audio_prompt_max_duration",
            "cfg_null_audio_empty",
            "cfg_null_text_empty",
            "speaker_mimi",
            "conditioners",
            "fuser",
        ] {
            assert!(fields.remove(key).is_some(), "{key} is no longer in the config");
        }
        let old: TTSConfig = serde_json::from_value(old).unwrap();
        assert_eq!(old.audio_prompt_min_duration, default_audio_prompt_min_duration());
        assert_eq!(old.audio_prompt_max_duration, default_audio_prompt_max_duration());
        assert!(!old.cfg_null_audio_empty);
        assert!(!old.cfg_null_text_empty);
        assert!(old.speaker_mimi.is_none());
        assert!(old.conditioners.is_empty() && old.fuser.is_none());
    }

    #[test]
    fn gelu_defaults_to_erf_except_for_pocket_tts_checkpoints() {
        use crate::transformer::Gelu;
        // No `model_id`: a Pocket TTS checkpoint, trained with tanh.
        let pocket = crate::test_config();
        assert!(pocket.model_id.is_none() && pocket.flow_lm.gelu.is_none());
        assert_eq!(pocket.default_gelu(), Gelu::Tanh);
        assert_eq!(pocket.mimi_with_gelu(&pocket.mimi).gelu, Some(Gelu::Tanh));

        // A phonon export without the key (written before it existed) was trained with erf.
        let mut json = serde_json::to_value(crate::test_config()).unwrap();
        let id = serde_json::json!({"sig": "78217628", "epoch": 200});
        json.as_object_mut().unwrap().insert("model_id".into(), id);
        let phonon: TTSConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(phonon.default_gelu(), Gelu::Erf);
        assert_eq!(phonon.mimi_with_gelu(&phonon.mimi).gelu, Some(Gelu::Erf));

        // A key in the config wins either way, and round-trips.
        json["flow_lm"]["gelu"] = serde_json::json!("tanh");
        json["mimi"]["gelu"] = serde_json::json!("tanh");
        let named: TTSConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(named.flow_lm.gelu, Some(Gelu::Tanh));
        assert_eq!(named.mimi_with_gelu(&named.mimi).gelu, Some(Gelu::Tanh));
        assert_eq!(serde_json::to_value(&named).unwrap()["mimi"]["gelu"], "tanh");
        assert_eq!(Gelu::default(), Gelu::Erf);
    }

    #[test]
    fn architecture_keys_default_to_the_released_model() {
        use crate::flow_lm::FlowType;
        use serde_json::json;
        let cfg = crate::test_config();
        let f = &cfg.flow_lm;
        assert_eq!(
            (f.flow_type, f.z_dim, f.attn_dim, f.num_kv_heads),
            (FlowType::Lsd, None, None, None)
        );
        assert_eq!(cfg.mimi.encoder_latent_dim(), cfg.mimi.dimension);

        let mut json = serde_json::to_value(&cfg).unwrap();
        json["flow_lm"]["flow_type"] = json!("pointwise");
        json["flow_lm"]["z_dim"] = json!(64);
        json["flow_lm"]["attn_dim"] = json!(192);
        json["flow_lm"]["num_kv_heads"] = json!(1);
        json["mimi"]["inner_dim"] = json!(32);
        let small: TTSConfig = serde_json::from_value(json.clone()).unwrap();
        let f = &small.flow_lm;
        let keys = (f.flow_type, f.z_dim, f.attn_dim, f.num_kv_heads);
        assert_eq!(keys, (FlowType::Pointwise, Some(64), Some(192), Some(1)));
        assert_eq!(small.mimi.encoder_latent_dim(), 32);
        assert_eq!(serde_json::to_value(&small).unwrap(), json);
        assert_eq!(
            serde_json::from_value::<FlowType>(json!("drifting")).unwrap(),
            FlowType::Drifting
        );
    }

    #[test]
    fn config_ignores_keys_it_no_longer_reads() {
        // Published config.json files still carry these, and have to keep loading.
        let mut json = serde_json::to_value(crate::test_config()).unwrap();
        let fields = json.as_object_mut().unwrap();
        fields.insert("temp".into(), serde_json::json!(0.7));
        let fuser = serde_json::json!({"sum": [], "streaming_sum": [], "prepend": [], "cross": []});
        fields.insert("fuser".into(), fuser);
        let id = serde_json::json!({"sig": "abc", "epoch": 1, "mimi_sig": "def", "mimi_epoch": 2});
        fields.insert("model_id".into(), id);
        let cfg: TTSConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.model_ext().as_deref(), Some("abc@1"));
    }
}
