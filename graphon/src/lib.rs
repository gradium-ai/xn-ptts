//! Graphon: streaming speech-to-text.
//!
//! 24 kHz audio goes through the Gemimi encoder ([`gemimi`]), one latent frame per 80 ms step,
//! and each frame is summed into the input of a text decoder ([`trunk`]) that emits one text
//! token per step, a fixed number of steps ([`config::Config::delay_steps`]) behind the audio.
//! [`Graphon`] holds both and [`Stream`] drives them over a stream of samples. A checkpoint with
//! voice activity heads ([`vad`]) also gives, at every step, the probability that the speaker
//! stays silent for each of a few windows.

mod attention;
pub mod config;
pub mod gemimi;
pub mod tokenizer;
pub mod trunk;
pub mod vad;

use config::Config;
use gemimi::{Gemimi, GemimiState};
use trunk::{Trunk, TrunkState};
use vad::Vad;
use xn::nn::VB;
use xn::{BackendQ, Result, Tensor, WithDTypeF};

/// The trunk with weights `Q`, the encoder with weights `C`. Both compute in the same dtype.
pub struct Graphon<Q: BackendQ, C: BackendQ<T = Q::T, B = Q::B> = Q> {
    pub config: Config,
    pub gemimi: Gemimi<C>,
    pub trunk: Trunk<Q>,
    pub vad: Option<Vad<Q>>,
    device: Q::B,
}

impl<Q: BackendQ, C: BackendQ<T = Q::T, B = Q::B>> Graphon<Q, C> {
    /// Load from an export directory: `config.json`, `model.safetensors`, `gemimi.safetensors`.
    pub fn load(dir: &std::path::Path, device: Q::B) -> Result<Self> {
        let config = Config::load(&dir.join("config.json"))?;
        Self::load_files(
            config,
            &dir.join("model.safetensors"),
            &dir.join("gemimi.safetensors"),
            device,
        )
    }

    pub fn load_files(
        config: Config,
        model: &std::path::Path,
        gemimi: &std::path::Path,
        device: Q::B,
    ) -> Result<Self> {
        let vb = VB::load(&[gemimi], device.clone())?;
        let gemimi = Gemimi::load(&vb.root(), &config.gemimi)?;
        let vb = VB::load(&[model], device.clone())?.root();
        let mut trunk =
            Trunk::load(&vb, &config.trunk, config.text.card, config.gemimi.latent_dim)?;
        let vad = match &config.vad {
            Some(cfg) => {
                trunk.set_taps(&cfg.layers)?;
                Some(Vad::load(&vb, cfg, config.trunk.dim)?)
            }
            None => None,
        };
        if gemimi.frame_size() != config.frame_size() {
            xn::bail!(
                "the encoder makes one frame per {} samples, the config steps every {}",
                gemimi.frame_size(),
                config.frame_size()
            )
        }
        Ok(Self { config, gemimi, trunk, vad, device })
    }

    pub fn device(&self) -> &Q::B {
        &self.device
    }

    pub fn stream(&self) -> Result<Stream<'_, Q, C>> {
        Ok(Stream {
            model: self,
            gemimi: self.gemimi.init_state(1)?,
            trunk: self.trunk.init_state(1)?,
            pending: Vec::new(),
            vad: self.vad.is_some(),
            inactivity: Vec::new(),
        })
    }
}

/// One stream of audio. Push samples as they arrive; each whole step's worth runs at once.
pub struct Stream<'a, Q: BackendQ, C: BackendQ<T = Q::T, B = Q::B>> {
    model: &'a Graphon<Q, C>,
    pub gemimi: GemimiState<Q::T, Q::B>,
    pub trunk: TrunkState<Q::T, Q::B>,
    pending: Vec<f32>,
    /// Whether steps run the voice activity heads.
    vad: bool,
    inactivity: Vec<f32>,
}

impl<Q: BackendQ, C: BackendQ<T = Q::T, B = Q::B>> Stream<'_, Q, C> {
    /// Feed 24 kHz mono samples; returns the text tokens of the steps that ran, padding
    /// included.
    pub fn push(&mut self, pcm: &[f32]) -> Result<Vec<u32>> {
        let mut pending = std::mem::take(&mut self.pending);
        pending.extend_from_slice(pcm);
        let frame = self.model.config.frame_size();
        let whole = pending.len() / frame * frame;
        let tokens = pending[..whole].chunks(frame).map(|c| self.step(c)).collect();
        pending.drain(..whole);
        self.pending = pending;
        tokens
    }

    /// One step: exactly one frame of samples in, one text token out.
    pub fn step(&mut self, frame: &[f32]) -> Result<u32> {
        let model = self.model;
        let samples = frame.iter().map(|&s| Q::T::from_f32(s)).collect();
        let pcm = Tensor::from_vec(samples, (1, 1, frame.len()), &model.device)?;
        let latent = model.gemimi.step(&pcm, &mut self.gemimi)?;
        let token = model.trunk.step(&latent, &mut self.trunk)?;
        if let (true, Some(vad)) = (self.vad, &model.vad) {
            self.inactivity = vad.inactivity(self.trunk.taps())?;
        }
        Ok(token)
    }

    /// Run the voice activity heads at every step (the default when the model has them) or not.
    pub fn set_vad(&mut self, on: bool) {
        self.vad = on && self.model.vad.is_some();
        self.inactivity.clear();
    }

    /// The last step's probabilities that no speech follows within each of the model's windows
    /// ([`vad::Vad::windows`]); empty when the heads did not run.
    pub fn inactivity(&self) -> &[f32] {
        &self.inactivity
    }

    /// End the stream: pads the last partial frame, then feeds [`config::Config::flush_steps`]
    /// steps of silence, enough for the last words to come out.
    pub fn flush(&mut self) -> Result<Vec<u32>> {
        self.flush_steps(self.model.config.flush_steps())
    }

    /// [`Stream::flush`] with `steps` steps of silence after the last partial frame.
    pub fn flush_steps(&mut self, steps: usize) -> Result<Vec<u32>> {
        let frame = self.model.config.frame_size();
        let pad = (frame - self.pending.len() % frame) % frame;
        let mut tail = std::mem::take(&mut self.pending);
        tail.resize(tail.len() + pad + steps * frame, 0.0);
        self.push(&tail)
    }

    /// Whether a token is text rather than padding.
    pub fn is_text(&self, token: u32) -> bool {
        let text = &self.model.config.text;
        token != text.padding_id && token != text.end_of_padding_id
    }
}
