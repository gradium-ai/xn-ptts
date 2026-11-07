//! Speech to speech: a Phonon checkpoint conditioned on the input audio's own Mimi latents, one
//! per frame, instead of text, emitting the output latents `audio_delay` frames behind.

use crate::flow_lm::{NormalRng, StepInput};
use crate::mimi::MimiEncoder;
use crate::tts_model::{TTSConfig, TTSModel};
use xn::nn::{Linear, var_builder::Path};
use xn::{BackendQ, Result, Tensor, Unquantized, WithDTypeF};

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
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.model.sample_rate()
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
        let dev = self.model.device();
        let audio = Tensor::from_vec(pcm.to_vec(), (1, 1, pcm.len()), dev)?;
        let latents = self.encoder.encode_to_latent(&audio)?;
        let latents = latents.transpose(1, 2)?.contiguous()?.to::<Q::T>()?;
        let frames = latents.dim(1usize)?;
        let last = latents.narrow(1, frames - 1..frames)?.contiguous()?;
        let parts: Vec<&Tensor<Q::T, Q::B>> =
            std::iter::once(&latents).chain(std::iter::repeat_n(&last, self.delay)).collect();
        let padded = Tensor::cat(&parts, 1)?;
        let normalized = padded.broadcast_add(&self.neg_mean)?.broadcast_mul(&self.inv_std)?;
        let conditions = self.source_proj.forward(&normalized)?;
        let total = frames + self.delay;
        let voice = voice.to::<Q::T>()?;
        let mut state = self.model.init_flow_lm_state(1, voice.dim(1usize)? + total + 1)?;
        self.model.prompt_audio(&mut state, &voice)?;
        let mut mimi_state = self.model.init_mimi_state(1)?;
        let mut rng = NormalRng::new(temperature, seed)?;
        let empty_text = Tensor::zeros((1, 0, self.model.flow_lm.conditioner.output_dim), dev)?;
        let mut out = Vec::new();
        let mut prev = None;
        for t in 0..total {
            state.flow_lm_state.conditions = Some(conditions.narrow(1, t..t + 1)?.contiguous()?);
            let input = match &prev {
                Some(latent) if t > self.delay => StepInput::Latent(latent),
                _ => StepInput::Bos { batch: 1 },
            };
            let (latent, _eos) = self.model.flow_lm.sample_next_latent_parts(
                input,
                &empty_text,
                &mut state.flow_lm_state,
                lsd_decode_steps,
                &mut rng,
            )?;
            if t >= self.delay {
                out.extend(self.model.decode_latent(&latent, &mut mimi_state)?.to_vec()?);
            }
            prev = Some(latent);
        }
        out.truncate(pcm.len());
        Ok(out)
    }
}
