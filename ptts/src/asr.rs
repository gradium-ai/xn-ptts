//! Streaming ASR with the phonon (moshi-STT) model.
//!
//! Audio goes through the Mimi encoder 80 ms at a time; each frame becomes `n_q`
//! codebook indices that the LM consumes alongside the text token it emitted on
//! the previous step. Words close when the model emits one of the end tokens.
//!
//! The step is deliberately split in two. A frame needs its codes on the host
//! before the LM can consume them, and the sampled token on the host before the
//! word decoder can, so there are two unavoidable readbacks per frame. Splitting
//! them lets a browser await each one instead of blocking, the same shape as
//! [`crate::flow_lm::FlowLM::sample_next_latent_parts`]; native callers can drive
//! the whole frame through [`AsrModel::step`].

use crate::asr_lm::{Config, LmModel};
use crate::asr_quantizer::MimiQuantizer;
use crate::mimi::{MimiConfig, MimiEncoder};
use xn::nn::var_builder::Path;
use xn::{BackendQ, Result, Tensor, Unquantized};

pub const SAMPLE_RATE: usize = 24_000;
pub const FRAME_RATE: f64 = 12.5;
/// Samples per model step: 80 ms at 24 kHz.
pub const FRAME_SIZE: usize = 1920;
/// Frames of silence prepended before the audio, as the serving worker does.
pub const INITIAL_SILENCE_FRAMES: usize = 2;

/// Text tokens that close the current word.
pub const TOKEN_EOP: u32 = 0;
pub const TOKEN_EOS: u32 = 2;
pub const TOKEN_PAD: u32 = 3;
pub const TOKEN_SILENCE_PAD: u32 = 4;

#[derive(Debug, Clone)]
pub enum Event {
    Word { tokens: Vec<u32>, start_time: f64 },
    EndWord { stop_time: f64 },
    EndOfStream,
}

/// Turns the token stream into words, closing each one when the model says so.
pub struct WordDecoder {
    delay: usize,
    step_idx: usize,
    word_tokens: Vec<u32>,
    unended_word: bool,
    last_stop_time: f64,
}

impl WordDecoder {
    pub fn new(delay: usize) -> Self {
        Self { delay, step_idx: 0, word_tokens: vec![], unended_word: false, last_stop_time: 0.0 }
    }

    pub fn push(&mut self, token: u32) -> Vec<Event> {
        self.step_idx += 1;
        let mut events = vec![];
        if self.step_idx < self.delay {
            return events;
        }
        let closes_word = matches!(token, TOKEN_EOP | TOKEN_EOS | TOKEN_PAD | TOKEN_SILENCE_PAD);
        if closes_word {
            if !self.word_tokens.is_empty() {
                let tokens = std::mem::take(&mut self.word_tokens);
                self.unended_word = true;
                events.push(Event::Word { tokens, start_time: self.last_stop_time });
            }
        } else {
            self.word_tokens.push(token);
        }
        if token == TOKEN_EOP || token == TOKEN_EOS {
            let stop_time = (self.step_idx - self.delay) as f64 / FRAME_RATE;
            if self.unended_word {
                self.unended_word = false;
                events.push(Event::EndWord { stop_time });
            }
            self.last_stop_time = stop_time;
        }
        if token == TOKEN_EOS {
            events.push(Event::EndOfStream);
        }
        events
    }
}

pub fn mimi_config() -> MimiConfig {
    MimiConfig {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        frame_rate: FRAME_RATE,
        dimension: 512,
        quantizer_dimension: 256,
        quantizer_output_dimension: 512,
        n_filters: 64,
        n_residual_layers: 1,
        ratios: vec![8, 6, 5, 4],
        kernel_size: 7,
        last_kernel_size: 3,
        residual_kernel_size: 3,
        dilation_base: 2,
        compress: 2,
        transformer_d_model: 512,
        transformer_num_heads: 8,
        transformer_num_layers: 8,
        transformer_layer_scale: 0.01,
        transformer_context: 250,
        transformer_max_period: 10000.0,
        transformer_dim_feedforward: 2048,
        downsample_channel_wise: false,
    }
}

/// Weight-name rewrites for the ASR checkpoints.
pub fn remap_key(name: &str) -> Option<String> {
    let name = name.replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight");
    let name = name.replace(".conv.conv.", ".conv.");
    Some(name)
}

type EncState<B> = <MimiEncoder<Unquantized<f32, B>> as EncStateOwner<B>>::State;

/// Helper so the state type can be named without spelling out the encoder's
/// associated types at every use site.
trait EncStateOwner<B: xn::Backend> {
    type State;
}

impl<B: xn::Backend> EncStateOwner<B> for MimiEncoder<Unquantized<f32, B>> {
    type State = crate::mimi::MimiEncoderState<f32, B>;
}

pub struct AsrModel<Q: BackendQ> {
    pub mimi: MimiEncoder<Unquantized<f32, Q::B>>,
    pub quantizer: MimiQuantizer<f32, Q::B>,
    pub lm: LmModel<Q>,
    pub cfg: Config,
    condition: Option<Tensor<Q::T, Q::B>>,
    audio_pad: Vec<u32>,
}

pub struct AsrState<Q: BackendQ> {
    enc: EncState<Q::B>,
    lm: crate::asr_lm::LmState<Q::T, Q::B>,
    pub decoder: WordDecoder,
    text_token: u32,
    frame_idx: usize,
}

impl<Q: BackendQ> AsrModel<Q> {
    /// `mimi_vb` and `lm_vb` are the roots of the two checkpoints; Mimi stays in
    /// f32 whatever `Q` quantizes the LM to, matching the TTS path.
    pub fn load(
        mimi_vb: &Path<Q::B>,
        lm_vb: &Path<Q::B>,
        cfg: Config,
        language: Option<&str>,
    ) -> Result<Self> {
        let mimi_cfg = mimi_config();
        let mimi: MimiEncoder<Unquantized<f32, Q::B>> = MimiEncoder::load(mimi_vb, &mimi_cfg)?;
        let quantizer: MimiQuantizer<f32, Q::B> = MimiQuantizer::load(
            &mimi_vb.pp("quantizer"),
            mimi_cfg.dimension,
            mimi_cfg.quantizer_dimension,
            cfg.n_q,
            2048,
        )?;
        let lm: LmModel<Q> = LmModel::load(lm_vb, &cfg)?;
        // The model is conditioned on the spoken language. With none given it
        // gets the learnt padding and picks for itself.
        let condition = match lm.conditioners.get("languages_in_segment") {
            None => None,
            Some(conditioner) => {
                if let Some(language) = language
                    && !conditioner.possible_values().iter().any(|v| v == language)
                {
                    xn::bail!(
                        "unknown language {language:?}, expected one of {:?}",
                        conditioner.possible_values()
                    )
                }
                Some(conditioner.condition(language)?)
            }
        };
        let audio_pad = vec![lm.audio_pad_token(); cfg.n_q];
        Ok(Self { mimi, quantizer, lm, cfg, condition, audio_pad })
    }

    pub fn init_state(&self) -> Result<AsrState<Q>> {
        Ok(AsrState {
            enc: self.mimi.init_state(1, 1)?,
            lm: self.lm.init_state(),
            decoder: WordDecoder::new(self.cfg.asr_delay_in_tokens),
            text_token: self.lm.text_start_token(),
            frame_idx: 0,
        })
    }

    pub fn delay_frames(&self) -> usize {
        self.cfg.asr_delay_in_tokens
    }

    /// Record the Mimi encode and quantize for one frame of `FRAME_SIZE` samples.
    /// Returns one index tensor per codebook; nothing has been read back yet.
    pub fn encode_frame(
        &self,
        state: &mut AsrState<Q>,
        pcm: &[f32],
    ) -> Result<Vec<Tensor<i64, Q::B>>> {
        if pcm.len() != FRAME_SIZE {
            xn::bail!("asr frame must be {FRAME_SIZE} samples, got {}", pcm.len())
        }
        let dev = self.lm.device();
        let audio = Tensor::from_vec(pcm.to_vec(), (1, 1, FRAME_SIZE), dev)?;
        let latent = self.mimi.encode_to_latent_step(&audio, &mut state.enc)?;
        self.quantizer.encode(&latent)
    }

    /// Record one LM step from a frame's codes, returning the sampled text token
    /// as a tensor. `codes` is what [`Self::encode_frame`] produced, on the host.
    pub fn lm_step(
        &self,
        state: &mut AsrState<Q>,
        codes: &[u32],
        temperature: f32,
    ) -> Result<Tensor<i64, Q::B>> {
        // The first slice is dropped: the encoder has no history yet, so the
        // model is handed pad tokens instead.
        let audio_tokens = if state.frame_idx == 0 { &self.audio_pad } else { codes };
        let (logits, _ys) =
            self.lm.step(&mut state.lm, state.text_token, audio_tokens, self.condition.as_ref())?;
        state.frame_idx += 1;
        let logits = logits.reshape((1, self.cfg.text_card))?;
        xn::nn::sampling::gumbel_max(&logits, temperature, 1)
    }

    /// Feed back the sampled token and collect any words it closed.
    pub fn push_token(&self, state: &mut AsrState<Q>, token: u32) -> Vec<Event> {
        state.text_token = token;
        state.decoder.push(token)
    }

    /// One whole frame, blocking on both readbacks. Native callers only: a
    /// browser has to await them, which is what the split above is for.
    pub fn step(
        &self,
        state: &mut AsrState<Q>,
        pcm: &[f32],
        temperature: f32,
    ) -> Result<Vec<Event>> {
        let codes_t = self.encode_frame(state, pcm)?;
        // The first read flushes the frame; the rest land on completed work.
        let mut codes = Vec::with_capacity(codes_t.len());
        for c in &codes_t {
            codes.push(c.to_vec()?[0] as u32);
        }
        let sampled = self.lm_step(state, &codes, temperature)?;
        let token = sampled.to_vec()?[0] as u32;
        Ok(self.push_token(state, token))
    }
}
