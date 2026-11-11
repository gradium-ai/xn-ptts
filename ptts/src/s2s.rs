//! Speech to speech: a Phonon checkpoint conditioned on the input audio's own Mimi latents, one
//! per frame, instead of text, emitting the output latents `audio_delay` frames behind.
//!
//! [`SpeechToSpeech::voice`] runs a voice prompt once, and [`SpeechToSpeech::session`] starts a
//! stream from it: [`Session::push`] takes input audio as it arrives, in pieces of any size, and
//! returns the output of every frame it completes; [`Session::flush`] ends the stream.

use crate::flow_lm::{NormalRng, StepInput};
use crate::mimi::{MimiDecoderState, MimiEncoder, MimiEncoderState};
use crate::tts_model::{TTSConfig, TTSModel, TTSState};
use std::time::{Duration, Instant};
use xn::nn::{Linear, var_builder::Path};
use xn::{BackendQ, Result, Tensor, Unquantized, WithDTypeF};

/// Input frames a session keeps in view by default: the 30 s windows the checkpoints were
/// trained on.
pub const DEFAULT_CONTEXT: usize = 375;

struct NoTokenizer;

impl crate::Tokenizer for NoTokenizer {
    fn encode(&self, _text: &str) -> xn::Result<Vec<u32>> {
        xn::bail!("a speech-to-speech checkpoint has no tokenizer")
    }

    fn decode(&self, _tokens: &[u32]) -> xn::Result<String> {
        xn::bail!("a speech-to-speech checkpoint has no tokenizer")
    }
}

pub struct SpeechToSpeech<Q: BackendQ> {
    pub model: TTSModel<Q>,
    encoder: MimiEncoder<Unquantized<f32, Q::B>>,
    source_proj: Linear<Q::T, Q::B>,
    neg_mean: Tensor<Q::T, Q::B>,
    inv_std: Tensor<Q::T, Q::B>,
    delay: usize,
    lsd_decode_steps: usize,
}

/// The flow LM's state after a voice prompt, which every session in that voice starts from.
pub struct Voice<Q: BackendQ> {
    state: TTSState<Q>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct SessionOptions {
    pub temperature: f32,
    pub seed: u64,
    /// Flow steps per frame; the checkpoint's `lsd_decode_steps` when unset.
    pub lsd_decode_steps: Option<usize>,
    /// Input frames the flow LM keeps in view. A longer stream drops the oldest third of them
    /// whenever this many are cached, so memory and the cost of a step stay bounded.
    pub context: usize,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            seed: 4242424242424242,
            lsd_decode_steps: None,
            context: DEFAULT_CONTEXT,
        }
    }
}

/// Time a session has spent in each stage, summed over its frames. On a GPU the stages only
/// queue work, which the decoder's read back waits for, so most of it lands in `decode` there.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimes {
    /// The Mimi encoder and the source projection.
    pub encode: Duration,
    /// The flow LM.
    pub flow: Duration,
    /// The Mimi decoder, reading the audio back included.
    pub decode: Duration,
}

impl<Q: BackendQ> SpeechToSpeech<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &TTSConfig) -> Result<Self> {
        let model = TTSModel::load(vb, Box::new(NoTokenizer), cfg, &Default::default())?;
        let encoder = MimiEncoder::load(&vb.pp("mimi"), &cfg.mimi_with_gelu(&cfg.mimi))?;
        let src = vb.pp("flow_lm.condition_provider.conditioners.source_latents");
        let ldim = cfg.flow_lm.ldim;
        let source_proj = Linear::load(src.pp("output_proj"), ldim, cfg.flow_lm.d_model)?;
        let mean = src.tensor::<f32>("latent_mean", (ldim,))?.to_vec()?;
        let std = src.tensor::<f32>("latent_std", (ldim,))?.to_vec()?;
        let dev = vb.device();
        let neg_mean = mean.iter().map(|m| Q::T::from_f32(-m)).collect();
        let inv_std = std.iter().map(|s| Q::T::from_f32(1.0 / s)).collect();
        Ok(Self {
            model,
            encoder,
            source_proj,
            neg_mean: Tensor::from_vec(neg_mean, (ldim,), dev)?,
            inv_std: Tensor::from_vec(inv_std, (ldim,), dev)?,
            delay: cfg.audio_delay,
            lsd_decode_steps: cfg.lsd_decode_steps,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.model.sample_rate()
    }

    /// Audio samples per frame, in and out.
    pub fn frame_size(&self) -> usize {
        self.encoder.frame_size()
    }

    /// Frames the output lags the input by, beyond the frame being filled.
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// Runs the voice prompt `voice`, a `[1, T, d_model]` embedding as
    /// [`crate::loader::load_voice_emb`] returns it.
    pub fn voice(&self, voice: &Tensor<f32, Q::B>) -> Result<Voice<Q>> {
        let len = voice.dim(1usize)?;
        let mut state = self.model.init_flow_lm_state(1, len)?;
        self.model.prompt_audio(&mut state, &voice.to::<Q::T>()?)?;
        Ok(Voice { state, len })
    }

    pub fn session(&self, voice: &Voice<Q>, opts: &SessionOptions) -> Result<Session<'_, Q>> {
        if opts.context < 3 {
            xn::bail!("a session needs a context of at least 3 frames, not {}", opts.context)
        }
        let transformer_state = voice
            .state
            .flow_lm_state
            .transformer_state
            .with_seq_budget(voice.len + opts.context)?;
        let flow_lm_state = crate::flow_lm::FlowLMState { transformer_state, conditions: None };
        let d_model = self.model.flow_lm.conditioner.output_dim;
        Ok(Session {
            s2s: self,
            encoder_state: self.encoder.init_state(1)?,
            flow: TTSState { flow_lm_state },
            decoder_state: self.model.init_mimi_state(1)?,
            rng: NormalRng::new(opts.temperature, opts.seed)?,
            lsd_decode_steps: opts.lsd_decode_steps.unwrap_or(self.lsd_decode_steps),
            empty_text: Tensor::zeros((1, 0, d_model), self.model.device())?,
            pending: Vec::new(),
            steps: 0,
            in_view: 0,
            voice_len: voice.len,
            context: opts.context,
            previous: None,
            last_condition: None,
            samples_in: 0,
            samples_out: 0,
            times: StageTimes::default(),
        })
    }

    /// Runs `pcm`, at the checkpoint's sample rate, through the model with `voice`, a
    /// `[1, T, d_model]` embedding as [`crate::loader::load_voice_emb`] returns it. The output has
    /// the input's duration.
    pub fn run(
        &self,
        pcm: &[f32],
        voice: &Tensor<f32, Q::B>,
        temperature: f32,
        seed: u64,
        lsd_decode_steps: usize,
    ) -> Result<Vec<f32>> {
        let opts = SessionOptions {
            temperature,
            seed,
            lsd_decode_steps: Some(lsd_decode_steps),
            ..Default::default()
        };
        let mut session = self.session(&self.voice(voice)?, &opts)?;
        let mut out = session.push(pcm)?;
        out.extend(session.flush()?);
        Ok(out)
    }

    /// The flow LM's per-frame conditions, `[1, T, d_model]`, from encoder latents `[1, C, T]`.
    fn conditions(&self, latents: &Tensor<f32, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        let latents = latents.transpose(1, 2)?.contiguous()?.to::<Q::T>()?;
        let normalized = latents.broadcast_add(&self.neg_mean)?.broadcast_mul(&self.inv_std)?;
        self.source_proj.forward(&normalized)
    }
}

/// One stream through a [`SpeechToSpeech`] model, see the module docs.
pub struct Session<'a, Q: BackendQ> {
    s2s: &'a SpeechToSpeech<Q>,
    encoder_state: MimiEncoderState<f32, Q::B>,
    flow: TTSState<Q>,
    decoder_state: MimiDecoderState<f32, Q::B>,
    rng: NormalRng,
    lsd_decode_steps: usize,
    empty_text: Tensor<Q::T, Q::B>,
    /// Input samples short of a whole frame.
    pending: Vec<f32>,
    /// Flow LM steps run, one per input frame.
    steps: usize,
    /// Frames in the flow LM's cache, after the voice prompt.
    in_view: usize,
    voice_len: usize,
    context: usize,
    /// The last step's latent, the next one's input.
    previous: Option<Tensor<Q::T, Q::B>>,
    /// The last input frame's condition, repeated for the `audio_delay` steps after the input
    /// ends.
    last_condition: Option<Tensor<Q::T, Q::B>>,
    samples_in: usize,
    samples_out: usize,
    times: StageTimes,
}

impl<Q: BackendQ> Session<'_, Q> {
    /// Feeds input audio at the checkpoint's sample rate, returning the output of every frame
    /// it completes.
    pub fn push(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.samples_in += pcm.len();
        self.pending.extend_from_slice(pcm);
        let frame_size = self.s2s.frame_size();
        let whole = self.pending.len() / frame_size * frame_size;
        if whole == 0 {
            return Ok(Vec::new());
        }
        let audio: Vec<f32> = self.pending.drain(..whole).collect();
        let mut out = Vec::with_capacity(whole);
        self.frames(audio, &mut out)?;
        self.samples_out += out.len();
        Ok(out)
    }

    /// Ends the stream: the last partial frame is padded with silence and the delayed frames
    /// are generated, so that the whole output has the input's duration.
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            let mut audio = std::mem::take(&mut self.pending);
            audio.resize(self.s2s.frame_size(), 0.0);
            self.frames(audio, &mut out)?;
        }
        if let Some(condition) = self.last_condition.take() {
            for _ in 0..self.s2s.delay {
                self.step(&condition, &mut out)?;
            }
        }
        out.truncate(self.samples_in.saturating_sub(self.samples_out));
        self.samples_out += out.len();
        Ok(out)
    }

    /// Time spent in each stage so far.
    pub fn times(&self) -> StageTimes {
        self.times
    }

    /// Encodes `audio`, whole frames of it, and runs a step per frame.
    fn frames(&mut self, audio: Vec<f32>, out: &mut Vec<f32>) -> Result<()> {
        let start = Instant::now();
        let len = audio.len();
        let audio = Tensor::from_vec(audio, (1, 1, len), self.s2s.model.device())?;
        let latents = self.s2s.encoder.encode_step(&audio, &mut self.encoder_state)?;
        let conditions = self.s2s.conditions(&latents)?;
        self.times.encode += start.elapsed();
        let frames = conditions.dim(1usize)?;
        for t in 0..frames {
            let condition = conditions.narrow(1, t..t + 1)?.contiguous()?;
            self.step(&condition, out)?;
            self.last_condition = Some(condition);
        }
        Ok(())
    }

    fn step(&mut self, condition: &Tensor<Q::T, Q::B>, out: &mut Vec<f32>) -> Result<()> {
        let start = Instant::now();
        let model = &self.s2s.model;
        if self.in_view == self.context {
            let drop = self.context / 3;
            model.flow_lm.transformer.drop_positions(
                &mut self.flow.flow_lm_state.transformer_state,
                self.voice_len,
                drop,
            )?;
            self.in_view -= drop;
        }
        self.flow.flow_lm_state.conditions = Some(condition.clone());
        let delay = self.s2s.delay;
        let input = match &self.previous {
            Some(latent) if self.steps > delay => StepInput::Latent(latent),
            _ => StepInput::Bos { batch: 1 },
        };
        let (latent, _eos) = model.flow_lm.sample_next_latent_parts(
            input,
            &self.empty_text,
            &mut self.flow.flow_lm_state,
            self.lsd_decode_steps,
            &mut self.rng,
        )?;
        self.steps += 1;
        self.in_view += 1;
        let decode = Instant::now();
        self.times.flow += decode - start;
        if self.steps > delay {
            out.extend(model.decode_latent(&latent, &mut self.decoder_state)?.to_vec()?);
            self.times.decode += decode.elapsed();
        }
        self.previous = Some(latent);
        Ok(())
    }
}
