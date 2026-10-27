//! One-call speech synthesis.
//!
//! [`Synth`] wraps everything between "I have some text" and "I have PCM":
//! locating and loading a checkpoint, choosing a device and weight format,
//! registering voices, splitting text into sentence chunks, priming the
//! transformer state, running the flow-matching solver, and streaming the
//! latents through the Mimi decoder on a second thread.
//!
//! ```no_run
//! # fn main() -> ptts::Result<()> {
//! use ptts::synth::Synth;
//! use ptts::checkpoint::read_config;
//!
//! use ptts::preprocess::Lang;
//!
//! let tts = Synth::builder(
//!     read_config("model/config.json")?,
//!     "model/model.safetensors",
//!     Lang::En,
//! )
//! .tokenizer_file("model/tokenizer.json")
//!     .add_voice("default", "model/default-voice.safetensors")
//!     .build()?;
//! let pcm = tts.say("Hello world")?;
//! ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate())?;
//! # Ok(())
//! # }
//! ```
//!
//! Audio arrives incrementally from [`SynthApi::stream`], which is the primitive
//! [`SynthApi::say`] is built on:
//!
//! ```no_run
//! # fn main() -> ptts::Result<()> {
//! # let cfg = ptts::checkpoint::read_config("model/config.json")?;
//! # let norm = ptts::preprocess::Lang::En;
//! # let tts = ptts::synth::Synth::builder(cfg, "model/model.safetensors", norm)
//! #     .tokenizer_file("model/tokenizer.json")
//! #     .build()?;
//! for chunk in tts.stream("Hello world")? {
//!     let pcm: Vec<f32> = chunk?;
//!     // hand `pcm` to an audio sink
//! }
//! # Ok(())
//! # }
//! ```
//!
//! A voice is conditioned on once per [`Synth`], whichever entry point is used;
//! [`SynthApi::session`] additionally pins the KV budget for a stream of requests.
//!
//! Text is normalized before it is tokenized — see [`crate::preprocess`]. Which
//! language, or [`Normalize::OFF`], is a required argument to
//! [`SynthBuilder::new`]: the model reads normalized text noticeably better,
//! but the spoken forms are per-language, so guessing is worse than doing
//! nothing.
//!
//! Driving the loop by hand, from an event loop with no threads to spawn, as
//! `ptts-wasm` does, is what [`crate::tts_model::TTSModel`]'s primitives are for.

use crate::comp::{AudioModule, Chain, ModuleConfig};
use crate::conditioners::Conditioners;
use crate::flow_lm::{NormalRng, StepInput};
use crate::loader;
use crate::plan::{self, Chunk, EosPolicy};
use crate::preprocess::Normalize;
use crate::step_graph::{GraphPool, StepGraph};
use crate::tts_model::{MAX_TOKENS_PER_CHUNK, MimiEnc, MimiLatent, TTSConfig, TTSModel, TTSState};
use crate::{Error, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex};
use xn::{Backend, BackendQ, Tensor};

/// Which device to run on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceKind {
    /// The most capable compiled-in backend for the weight format: CUDA, then
    /// Vulkan, then Metal, then the CPU for f32 weights, and the CPU for
    /// quantized ones, which only it runs.
    #[default]
    Auto,
    Cpu,
    Cuda,
    Vulkan,
    Metal,
}

impl DeviceKind {
    /// Resolve [`Self::Auto`] against the backends this build was compiled with.
    pub fn resolve(self) -> Self {
        if self != Self::Auto {
            return self;
        }
        if cfg!(feature = "cuda") {
            Self::Cuda
        } else if cfg!(feature = "vulkan") {
            Self::Vulkan
        } else if cfg!(feature = "metal") {
            Self::Metal
        } else {
            Self::Cpu
        }
    }

    /// Resolve [`Self::Auto`] for weights in `quant`: quantized weights run on
    /// the CPU whatever GPU backend is compiled in, so `auto` never picks a
    /// device that would refuse them.
    pub fn resolve_for(self, quant: Quant) -> Self {
        if self == Self::Auto && quant != Quant::F32 { Self::Cpu } else { self.resolve() }
    }
}

impl std::str::FromStr for DeviceKind {
    type Err = crate::Error;

    fn from_str(name: &str) -> crate::Result<Self> {
        match name.trim().to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "vulkan" => Ok(Self::Vulkan),
            "metal" => Ok(Self::Metal),
            _ => Err(Error::invalid_argument(format!(
                "unknown device '{name}'; expected auto, cpu, cuda, vulkan or metal"
            ))),
        }
    }
}

/// Weight format for the flow-LM transformer linears. Quantization is CPU-only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quant {
    #[default]
    F32,
    Q80,
    Q81,
    Q8k,
    Q6k,
    Q50,
    Q51,
    Q5k,
    Q40,
    Q41,
    Q4k,
}

impl Quant {
    /// Error if the device backend was not compiled in or this weight format cannot run on it.
    /// This checks build features without initializing the device.
    ///
    /// [`SynthBuilder::build`] checks this too, but a caller that downloads a
    /// checkpoint before building should check first, so an impossible
    /// combination fails in milliseconds rather than after a few hundred
    /// megabytes.
    pub fn check_device(self, device: DeviceKind) -> Result<()> {
        let device = device.resolve_for(self);
        if device != DeviceKind::Cpu && self != Self::F32 {
            return Err(Error::unsupported(format!(
                "quantization ({}) is CPU-only, but the selected device is {device:?}",
                self.as_str()
            )));
        }
        let backends = [
            (DeviceKind::Cuda, "cuda", cfg!(feature = "cuda")),
            (DeviceKind::Vulkan, "vulkan", cfg!(feature = "vulkan")),
            (DeviceKind::Metal, "metal", cfg!(feature = "metal")),
        ];
        if let Some((_, name, _)) =
            backends.iter().find(|(kind, _, compiled)| *kind == device && !compiled)
        {
            let available: Vec<_> = backends
                .iter()
                .filter(|(_, _, compiled)| *compiled)
                .map(|(_, name, _)| *name)
                .chain(["cpu"])
                .collect();
            return Err(Error::unsupported(format!(
                "device '{name}' is not available in this build; available devices: {}",
                available.join(", ")
            )));
        }
        Ok(())
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q80 => "q8_0",
            Self::Q81 => "q8_1",
            Self::Q8k => "q8k",
            Self::Q6k => "q6k",
            Self::Q50 => "q5_0",
            Self::Q51 => "q5_1",
            Self::Q5k => "q5k",
            Self::Q40 => "q4_0",
            Self::Q41 => "q4_1",
            Self::Q4k => "q4k",
        }
    }
}

/// Parse the spellings the CLIs accept.
impl std::str::FromStr for Quant {
    type Err = crate::Error;

    fn from_str(name: &str) -> crate::Result<Self> {
        match name.trim().to_lowercase().as_str() {
            "f32" | "none" => Ok(Self::F32),
            "q8" | "q8_0" => Ok(Self::Q80),
            "q8_1" => Ok(Self::Q81),
            "q8k" => Ok(Self::Q8k),
            "q6k" => Ok(Self::Q6k),
            "q5" | "q5_0" => Ok(Self::Q50),
            "q5_1" => Ok(Self::Q51),
            "q5k" => Ok(Self::Q5k),
            "q4" | "q4_0" => Ok(Self::Q40),
            "q4_1" => Ok(Self::Q41),
            "q4k" => Ok(Self::Q4k),
            _ => Err(Error::invalid_argument(format!(
                "unsupported quantization '{name}'; expected one of \
                 f32, q8_0, q8_1, q8k, q6k, q5_0, q5_1, q5k, q4_0, q4_1, q4k"
            ))),
        }
    }
}

/// Per-request overrides. Anything left `None` falls back to what the
/// [`SynthBuilder`] was configured with.
#[derive(Clone, Debug, Default)]
pub struct SpeechOptions {
    pub voice: Option<String>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
    /// Classifier-free guidance coefficient. `1.0` and `None` both disable it.
    pub cfg_coef: Option<f32>,
    pub max_tokens_per_chunk: Option<usize>,
    /// Audio modules run over the PCM, in order, before it is handed out.
    pub post_process: Option<Vec<ModuleConfig>>,
}

impl SpeechOptions {
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn cfg_coef(mut self, cfg_coef: f32) -> Self {
        self.cfg_coef = Some(cfg_coef);
        self
    }

    pub fn max_tokens_per_chunk(mut self, max_tokens: usize) -> Self {
        self.max_tokens_per_chunk = Some(max_tokens);
        self
    }

    pub fn post_process(mut self, modules: impl IntoIterator<Item = ModuleConfig>) -> Self {
        self.post_process = Some(modules.into_iter().collect());
        self
    }
}

/// Defaults applied to every request unless overridden per call.
#[derive(Clone, Debug)]
struct Defaults {
    voice: Option<String>,
    temperature: f32,
    seed: u64,
    cfg_coef: Option<f32>,
    max_tokens_per_chunk: usize,
    post_process: Vec<ModuleConfig>,
}

/// Merge per-request overrides onto the settings a [`SynthBuilder`] was given.
///
/// A `cfg_coef` of 1.0 becomes `None`: guidance at 1.0 is the identity, and
/// computing it would cost a second forward pass.
///
/// The temperature is checked here rather than at the three `NormalRng::new` call sites: this
/// is the one place every request passes through, so the builder's default is covered as well
/// as a per-request override. Left unchecked it reaches `rand_distr::Normal` and comes back as
/// a `BadVariance` wrapped in a tensor error, which is a `RuntimeError` in Python carrying a
/// message about variance rather than about the argument the caller passed.
fn resolve(defaults: &Defaults, opts: &SpeechOptions) -> Result<Defaults> {
    let temperature = opts.temperature.unwrap_or(defaults.temperature);
    // NaN spelled out rather than `!(t >= 0)`, which clippy rejects on partially ordered
    // types. Zero is fine: it is greedy sampling, and `Normal::new` takes a zero std dev.
    if temperature.is_nan() || temperature < 0.0 {
        return Err(Error::invalid_argument(format!(
            "temperature must be >= 0, got {temperature}"
        )));
    }
    Ok(Defaults {
        voice: opts.voice.clone().or_else(|| defaults.voice.clone()),
        temperature,
        seed: opts.seed.unwrap_or(defaults.seed),
        cfg_coef: match opts.cfg_coef.or(defaults.cfg_coef) {
            Some(coef) if coef != 1.0 => Some(coef),
            _ => None,
        },
        max_tokens_per_chunk: opts.max_tokens_per_chunk.unwrap_or(defaults.max_tokens_per_chunk),
        post_process: opts.post_process.clone().unwrap_or_else(|| defaults.post_process.clone()),
    })
}

/// A registered voice: the conditioning embedding, plus the encoding of
/// equal-length silence when the model needs one for CFG.
struct Voice<Q: BackendQ> {
    emb: Tensor<Q::T, Q::B>,
    null_emb: Option<Tensor<Q::T, Q::B>>,
    conditions: Option<Tensor<Q::T, Q::B>>,
}

/// A loaded model, with the weight format fixed at compile time. Always reached through
/// [`Synth`], which erases `Q` so the format can be chosen at runtime.
struct SynthOf<Q: BackendQ> {
    /// The weight format it was loaded as.
    quant: Quant,
    model: Arc<TTSModel<Q>>,
    mimi_enc: Option<MimiEnc<Q>>,
    cfg: TTSConfig,
    voices: BTreeMap<String, Voice<Q>>,
    defaults: Defaults,
    /// How every request's text is normalized. See [`SynthBuilder::new`].
    normalize: Normalize,
    /// Each voice's primed KV prefix, keyed by name and by whether guidance is on: the null
    /// branch's prefix differs, the coefficient does not enter the state. Sized to the prompt;
    /// copied out to each generation's budget.
    primed: Mutex<HashMap<(String, bool), Primed<Q>>>,
    conditioners: Conditioners<Q::T, Q::B>,
    conditions: Option<Tensor<Q::T, Q::B>>,
    /// Whether a session should decode each frame on the flow-LM's own thread rather than
    /// handing it to a second thread. A device with one shared command queue (Metal) has no
    /// second execution unit for a second thread to overlap onto, so the split only adds
    /// cross-thread contention over that queue; the CPU's separate cores, and CUDA's second
    /// stream for Mimi, make it a real win there. Set once per backend in
    /// [`SynthBuilder::build_cuda`] and friends.
    inline_decode: bool,
    /// Recorded steps, kept from one generation to the next. Empty off CUDA.
    graphs: Arc<GraphPool<Q>>,
}

/// How many primed prefixes to keep. A process that clones voices under fresh names -- which
/// `ptts-pyo3` exposes -- would otherwise grow the map without bound. Past the cap it is cleared;
/// generation still works, it just re-primes.
const MAX_PRIMED: usize = 16;

/// A voice's conditioning, run through the transformer once.
struct Primed<Q: BackendQ> {
    state: TTSState<Q>,
    null_state: Option<TTSState<Q>>,
}

impl<Q: BackendQ> SynthOf<Q> {
    /// Build a session, sized to `seq_budget`.
    fn session_at(&self, settings: &Defaults, seq_budget: usize) -> Result<SessionOf<Q>> {
        let (base, cfg_base) =
            self.primed_state(settings.voice.as_deref(), seq_budget, settings.cfg_coef)?;
        Ok(SessionOf {
            prompt_len: primed_len(&base),
            in_flight: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            model: Arc::clone(&self.model),
            frame_rate: self.cfg.mimi.frame_rate,
            temperature: settings.temperature,
            seed: settings.seed,
            max_tokens_per_chunk: settings.max_tokens_per_chunk,
            base,
            cfg_base,
            seq_budget,
            normalize: self.normalize,
            post_process: settings.post_process.clone(),
            inline_decode: self.inline_decode,
            graphs: Arc::clone(&self.graphs),
        })
    }

    /// Called whenever a voice is (re)registered, so a replaced embedding is never generated
    /// from the old conditioning.
    fn forget_primed(&self, name: &str) {
        let mut primed = self.primed.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        primed.retain(|(voice, _), _| voice != name);
    }

    /// The voice a request speaks in: the one it names, else the default. `None` when there
    /// are no voices, and generation is then unconditioned.
    fn voice_for(&self, name: Option<&str>) -> Result<Option<(&str, &Voice<Q>)>> {
        let default = if name.is_none() { self.default_voice() } else { None };
        let Some(name) = name.or(default.as_deref()) else { return Ok(None) };
        match self.voices.get_key_value(name) {
            Some((name, voice)) => Ok(Some((name.as_str(), voice))),
            None => Err(Error::UnknownVoice { name: name.to_string(), known: self.voices() }),
        }
    }

    /// The state every chunk starts from, sized to `seq_budget` and conditioned on the voice.
    ///
    /// Conditioning is the expensive part and depends only on (voice, guidance), so it happens
    /// once per `SynthOf`; each call copies the kept prefix into fresh buffers rather than
    /// cloning it, since a clone would share the KV storage.
    #[allow(clippy::type_complexity)]
    fn primed_state(
        &self,
        voice: Option<&str>,
        seq_budget: usize,
        cfg_coef: Option<f32>,
    ) -> Result<(TTSState<Q>, Option<(f32, TTSState<Q>)>)> {
        let Some((name, voice)) = self.voice_for(voice)? else {
            let mut state = self.unprompted_state(seq_budget)?;
            state.flow_lm_state.conditions = self.conditions.clone();
            let cfg_state = match cfg_coef {
                None => None,
                Some(coef) => {
                    let mut null_state = self.unprompted_state(seq_budget)?;
                    null_state.flow_lm_state.conditions = self.conditions.clone();
                    Some((coef, null_state))
                }
            };
            return Ok((state, cfg_state));
        };

        let frames = self.model.voice_prompt_len(voice.emb.dim(1usize)?);
        if frames >= seq_budget {
            // Its own variant rather than `SeqBudgetExceeded`: nothing about the text is
            // wrong here, so "split the text" would be useless advice, and the prompt
            // length alone cannot say what budget would actually work.
            return Err(Error::VoicePromptTooLong { frames, budget: seq_budget });
        }

        let key = (name.to_string(), cfg_coef.is_some());
        // Prime outside the lock so first callers for different voices do not serialise; a
        // same-voice race primes twice and the second insert wins, harmlessly.
        let cached = {
            let primed = self.primed.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            primed.get(&key).map(|p| (p.state.clone(), p.null_state.clone()))
        };
        let (prefix, null_prefix) = match cached {
            Some(hit) => hit,
            None => {
                let fresh = self.prime(voice, cfg_coef.is_some(), frames)?;
                let out = (fresh.state.clone(), fresh.null_state.clone());
                let mut primed =
                    self.primed.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                if primed.len() >= MAX_PRIMED {
                    primed.clear();
                }
                primed.insert(key, fresh);
                out
            }
        };

        let state = grow(&prefix, seq_budget)?;
        let cfg_state = match (cfg_coef, null_prefix) {
            (Some(coef), Some(null)) => Some((coef, grow(&null, seq_budget)?)),
            (None, _) => None,
            // A guidance-keyed entry always carries a null branch. Spelled out rather than
            // folded into a catch-all, which would silently generate without guidance.
            (Some(_), None) => {
                return Err(Error::Tensor(xn::Error::msg(
                    "internal: a primed entry keyed with guidance has no null branch",
                )));
            }
        };
        Ok((state, cfg_state))
    }

    fn check_no_baked_voices(&self) -> Result<()> {
        if self.cfg.voices.is_empty() {
            Ok(())
        } else {
            Err(Error::unsupported(
                "this checkpoint has baked-in voices and supports no other voice",
            ))
        }
    }

    fn unprompted_state(&self, seq_budget: usize) -> Result<TTSState<Q>> {
        let mut state = self.model.init_flow_lm_state(1, seq_budget)?;
        self.model.prompt_no_voice(&mut state)?;
        Ok(state)
    }

    /// Run the voice prompt, and with guidance on the null branch, into states just large
    /// enough to hold them.
    fn prime(&self, voice: &Voice<Q>, cfg_on: bool, frames: usize) -> Result<Primed<Q>> {
        let conditions = voice.conditions.clone().or_else(|| self.conditions.clone());
        let mut state = self.model.init_flow_lm_state(1, frames)?;
        state.flow_lm_state.conditions = conditions.clone();
        self.model.prompt_audio(&mut state, &voice.emb)?;

        let null_state = if !cfg_on {
            None
        } else {
            // Sized to what the null branch will consume, so the `emb`/`null_emb` length
            // invariant stays local to registration rather than load-bearing here.
            let null_frames = match voice.null_emb.as_ref() {
                Some(null_emb) if !self.cfg.cfg_null_audio_empty => {
                    self.model.voice_prompt_len(null_emb.dim(1usize)?)
                }
                _ => frames,
            };
            let mut null_state = self.model.init_flow_lm_state(1, null_frames)?;
            null_state.flow_lm_state.conditions = conditions;
            if self.cfg.cfg_null_audio_empty {
                self.model.prompt_no_voice(&mut null_state)?;
            } else {
                match voice.null_emb.as_ref() {
                    Some(null_emb) => self.model.prompt_audio(&mut null_state, null_emb)?,
                    None if !self.cfg.voices.is_empty() => {
                        return Err(Error::unsupported(
                            "this checkpoint has baked-in voices, which do not support CFG: \
                             disable it (cfg_coef of 1.0 or unset)",
                        ));
                    }
                    None => {
                        return Err(Error::unsupported(
                            "this model conditions its CFG null branch on silence \
                             (cfg_null_audio_empty=false), which needs the voice's source \
                             audio. Register the voice with add_voice_from_pcm instead of a \
                             precomputed embedding, or disable CFG.",
                        ));
                    }
                }
            }
            Some(null_state)
        };
        Ok(Primed { state, null_state })
    }
}

impl<Q: BackendQ> SynthApi for SynthOf<Q> {
    fn sample_rate(&self) -> u32 {
        self.model.sample_rate() as u32
    }

    fn config(&self) -> &TTSConfig {
        &self.cfg
    }

    fn normalization(&self) -> Normalize {
        self.normalize
    }

    fn device_name(&self) -> String {
        self.model.device().name()
    }

    fn languages(&self) -> &[String] {
        self.cfg.languages.as_slice()
    }

    fn voices(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    fn default_voice(&self) -> Option<String> {
        let found = self.voices.get_key_value("default").or_else(|| self.voices.iter().next());
        self.defaults.voice.clone().or_else(|| found.map(|(name, _)| name.clone()))
    }

    fn supports_voice_cloning(&self) -> bool {
        self.mimi_enc.is_some() && self.cfg.voices.is_empty()
    }

    fn voice_prompt_sample_rate(&self) -> u32 {
        self.cfg.speaker_mimi_cfg().sample_rate as u32
    }

    fn add_voice_file(&mut self, name: &str, path: &FsPath) -> Result<()> {
        self.check_no_baked_voices()?;
        let dev = self.model.device().clone();
        let embeddings_ext = self.cfg.embeddings_ext.as_deref();
        let emb = loader::load_voice_emb(path, embeddings_ext, self.model.speaker_proj(), &dev)?
            .to::<Q::T>()?;
        self.forget_primed(name);
        self.voices.insert(name.to_string(), Voice { emb, null_emb: None, conditions: None });
        Ok(())
    }

    fn add_voice_from_pcm(&mut self, name: &str, pcm: &[f32]) -> Result<()> {
        self.check_no_baked_voices()?;
        let enc = match self.mimi_enc.as_ref() {
            Some(enc) => enc,
            None => {
                return Err(Error::unsupported(
                    "this checkpoint has no speaker encoder, so it cannot clone voices",
                ));
            }
        };
        let sr = self.voice_prompt_sample_rate();
        let min_len = (sr as f32 * self.cfg.audio_prompt_min_duration).round() as usize;
        let max_len = (sr as f32 * self.cfg.audio_prompt_max_duration).round() as usize;
        if pcm.len() < min_len {
            return Err(Error::invalid_argument(format!(
                "voice prompt is too short: got {} samples ({:.2}s at {sr}Hz), need at least \
                 {min_len} ({:.2}s)",
                pcm.len(),
                pcm.len() as f32 / sr as f32,
                self.cfg.audio_prompt_min_duration
            )));
        }
        let mut pcm = pcm[..pcm.len().min(max_len)].to_vec();
        crate::utils::normalize_loudness(&mut pcm, sr)?;

        let dev = self.model.device().clone();
        let pcm = Tensor::from_vec(pcm, (1, 1, ()), &dev)?.to::<Q::T>()?;
        let emb = enc.encode_audio(&pcm)?;
        // Only needed for CFG, and only when the model conditions its null
        // branch on silence rather than on nothing at all.
        let null_emb = if self.cfg.cfg_null_audio_empty {
            None
        } else {
            Some(enc.encode_audio(&pcm.zeros_like()?)?)
        };
        self.forget_primed(name);
        self.voices.insert(name.to_string(), Voice { emb, null_emb, conditions: None });
        Ok(())
    }

    fn add_voice_from_embedding(
        &mut self,
        name: &str,
        emb: &[f32],
        frames: usize,
        dim: usize,
        null_emb: Option<&[f32]>,
    ) -> Result<()> {
        self.check_no_baked_voices()?;
        if emb.len() != frames * dim {
            return Err(Error::invalid_argument(format!(
                "embedding has {} values, expected {frames} x {dim}",
                emb.len()
            )));
        }
        let dev = self.model.device().clone();
        let to_tensor = |data: &[f32]| -> xn::Result<Tensor<Q::T, Q::B>> {
            Tensor::from_vec(data.to_vec(), (1, frames, dim), &dev)?.to::<Q::T>()
        };
        let emb = to_tensor(emb)?;
        let null_emb = match null_emb {
            None => None,
            Some(null) if null.len() != frames * dim => {
                return Err(Error::invalid_argument(format!(
                    "null embedding has {} values, expected {frames} x {dim}",
                    null.len()
                )));
            }
            Some(null) => Some(to_tensor(null)?),
        };
        self.forget_primed(name);
        self.voices.insert(name.to_string(), Voice { emb, null_emb, conditions: None });
        Ok(())
    }

    fn add_voice_from_latents(
        &mut self,
        name: &str,
        latents: &[f32],
        channels: usize,
        frames: usize,
    ) -> Result<()> {
        self.check_no_baked_voices()?;
        if latents.len() != channels * frames {
            return Err(Error::invalid_argument(format!(
                "latents have {} values, expected {channels} x {frames}",
                latents.len()
            )));
        }
        let dev = self.model.device().clone();
        let latents = Tensor::from_vec(latents.to_vec(), (1, channels, frames), &dev)?;
        let emb =
            loader::project_latents(&latents, "the voice latents", self.model.speaker_proj())?
                .to::<Q::T>()?;
        self.forget_primed(name);
        self.voices.insert(name.to_string(), Voice { emb, null_emb: None, conditions: None });
        Ok(())
    }

    fn set_conditions(&mut self, conditions: HashMap<String, String>) -> Result<()> {
        let mut values = conditions.clone();
        if let Some(voice) = self.cfg.voices.first() {
            for (k, v) in voice.conditions.iter() {
                values.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        let default = self.conditioners.sum(&values)?;
        let mut baked = Vec::with_capacity(self.cfg.voices.len());
        for voice in self.cfg.voices.iter() {
            let mut values = conditions.clone();
            values.extend(voice.conditions.clone());
            baked.push((voice.name.clone(), self.conditioners.sum(&values)?));
        }
        for (name, c) in baked {
            if let Some(voice) = self.voices.get_mut(&name) {
                voice.conditions = c;
            }
        }
        self.conditions = default;
        self.primed.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
        Ok(())
    }

    fn say(&self, text: &str) -> Result<Vec<f32>> {
        self.say_with(text, &SpeechOptions::default())
    }

    fn say_with(&self, text: &str, opts: &SpeechOptions) -> Result<Vec<f32>> {
        let mut pcm = Vec::new();
        for chunk in self.stream_with(text, opts)? {
            pcm.extend_from_slice(&chunk?);
        }
        Ok(pcm)
    }

    fn stream(&self, text: &str) -> Result<SpeechStream> {
        self.stream_with(text, &SpeechOptions::default())
    }

    fn session(&self, opts: &SpeechOptions, max_seq_len: usize) -> Result<Session> {
        Ok(Session(Box::new(self.session_at(&resolve(&self.defaults, opts)?, max_seq_len)?)))
    }

    fn session_default(&self, opts: &SpeechOptions) -> Result<Session> {
        let settings = resolve(&self.defaults, opts)?;
        let tokens = settings.max_tokens_per_chunk.min(plan::MAX_FIT_TOKENS);
        let prompt = match self.voice_for(settings.voice.as_deref())? {
            Some((_, voice)) => voice.emb.dim(1usize)?,
            None => 0,
        };
        let seq_budget = prompt.max(plan::PROMPT_SEQ_HEADROOM)
            + tokens
            + plan::frame_budget(tokens, self.cfg.mimi.frame_rate);
        Ok(Session(Box::new(self.session_at(&settings, seq_budget)?)))
    }

    fn stream_with(&self, text: &str, opts: &SpeechOptions) -> Result<SpeechStream> {
        let settings = resolve(&self.defaults, opts)?;
        let rng = Box::new(NormalRng::new(settings.temperature, settings.seed)?);
        self.stream_with_rng(text, opts, rng)
    }

    fn stream_with_rng(
        &self,
        text: &str,
        opts: &SpeechOptions,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream> {
        let settings = resolve(&self.defaults, opts)?;
        let chunks = plan_chunks(
            &self.model,
            self.cfg.mimi.frame_rate,
            text,
            settings.max_tokens_per_chunk,
            self.normalize,
            Some(plan::MAX_FIT_TOKENS),
        )?;
        // A one-shot call is a session sized to this text and dropped afterwards,
        // so there is one generation path rather than two. The voice prompt gets its
        // real length, as the session's own check counts it, and never less than
        // PROMPT_SEQ_HEADROOM: a reserve alone would leave a long prompt no room.
        let prompt = match self.voice_for(settings.voice.as_deref())? {
            Some((_, voice)) => voice.emb.dim(1usize)?,
            None => 0,
        };
        let text = chunks.iter().map(|c| c.tokens.len() + c.frame_budget).max().unwrap_or(0);
        let seq_budget = text + prompt.max(plan::PROMPT_SEQ_HEADROOM);
        self.session_at(&settings, seq_budget)?.stream_chunks(chunks, rng)
    }

    fn quant(&self) -> Quant {
        self.quant
    }
}

/// A fresh, independently-owned state at `seq_budget`, seeded with `prefix`'s filled positions.
fn grow<Q: BackendQ>(prefix: &TTSState<Q>, seq_budget: usize) -> Result<TTSState<Q>> {
    let transformer_state = prefix.flow_lm_state.transformer_state.with_seq_budget(seq_budget)?;
    let conditions = prefix.flow_lm_state.conditions.clone();
    Ok(TTSState { flow_lm_state: crate::flow_lm::FlowLMState { transformer_state, conditions } })
}

/// Split `text` into chunks and work out the budgets for each, cutting any chunk of more than
/// `fit_to` tokens with [`plan::fit`].
///
/// Free rather than a method because both [`Synth`] and [`Session`] need
/// it, and it depends only on the tokenizer inside the model and the codec's
/// frame rate.
fn plan_chunks<Q: BackendQ>(
    model: &TTSModel<Q>,
    frame_rate: f64,
    text: &str,
    max_tokens_per_chunk: usize,
    normalize: Normalize,
    fit_to: Option<usize>,
) -> Result<Vec<Chunk>> {
    let tokenizer = match model.flow_lm.conditioner.tokenizer.as_ref() {
        Some(tokenizer) => tokenizer.as_ref(),
        None => {
            return Err(Error::unsupported(
                "this model was loaded without a tokenizer; pass one to \
                 SynthBuilder::tokenizer, or use the lower-level TTSModel API with \
                 pre-tokenized input",
            ));
        }
    };
    let chunks = plan::chunks(tokenizer, text, normalize, max_tokens_per_chunk, frame_rate)?;
    match fit_to {
        Some(max) => plan::fit_or_error(chunks, max, tokenizer, frame_rate),
        None => Ok(chunks),
    }
}

/// The generic session behind [`Session`], whose docs cover its concurrency rules.
struct SessionOf<Q: BackendQ> {
    model: Arc<TTSModel<Q>>,
    frame_rate: f64,
    temperature: f32,
    seed: u64,
    max_tokens_per_chunk: usize,
    base: TTSState<Q>,
    cfg_base: Option<(f32, TTSState<Q>)>,
    seq_budget: usize,
    /// How text is normalized, inherited from the [`Synth`] this session was
    /// built from.
    normalize: Normalize,
    /// Slots the voice prompt already occupies, so the budget check can use it
    /// instead of [`plan::PROMPT_SEQ_HEADROOM`]'s fixed reserve.
    prompt_len: usize,
    /// Audio modules run over every generation's PCM, fresh for each one.
    post_process: Vec<ModuleConfig>,
    /// Set while a generation is running. See the note on this type.
    in_flight: Arc<std::sync::atomic::AtomicBool>,
    /// See [`SynthOf::inline_decode`].
    inline_decode: bool,
    /// See [`SynthOf::graphs`].
    graphs: Arc<GraphPool<Q>>,
}

impl<Q: BackendQ> SessionOf<Q> {
    /// As [`Self::stream`], with an explicit noise source.
    fn stream_with_rng(
        &self,
        text: &str,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream> {
        // The budget is fixed, so a sentence too long for it is cut rather than refused, and
        // never left longer than the model speaks well.
        let room = self.seq_budget.saturating_sub(self.prompt_len);
        let fit_to = plan::max_tokens_for(room, self.frame_rate).min(plan::MAX_FIT_TOKENS);
        // With no room for even an empty chunk, cutting cannot help: skip it, so the budget
        // check reports what the text actually needs.
        let fit_to = (fit_to > 0).then_some(fit_to);
        let chunks = plan_chunks(
            &self.model,
            self.frame_rate,
            text,
            self.max_tokens_per_chunk,
            self.normalize,
            fit_to,
        )?;
        self.stream_chunks(chunks, rng)
    }

    /// Start the two worker threads for an already-planned set of chunks.
    ///
    /// The single place generation is driven from: [`SynthApi::stream_with_rng`]
    /// reaches it through an ephemeral session.
    fn stream_chunks(
        &self,
        chunks: Vec<Chunk>,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream> {
        // `Chunk::seq_budget` reserves PROMPT_SEQ_HEADROOM for a voice prompt whose
        // real length this session knows, so it over-states what is needed for
        // a short prompt and under-states it for a long one.
        let needed = chunks
            .iter()
            .map(|c| self.prompt_len + c.tokens.len() + c.frame_budget)
            .max()
            .unwrap_or(0);
        if needed > self.seq_budget {
            return Err(Error::SeqBudgetExceeded { needed, budget: self.seq_budget });
        }
        // Claimed before anything is cloned: the state clone shares its KV
        // storage, so a second generation would write into the same buffers.
        if self
            .in_flight
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            return Err(Error::busy(
                "a generation is already in flight on this session; finish or drop that \
                 SpeechStream first, or build a second session",
            ));
        }
        let in_flight = InFlight(Arc::clone(&self.in_flight));

        // Each request starts from the primed state rather than re-conditioning
        // on the voice.
        let base_state = self.base.clone();
        let cfg_base = self.cfg_base.clone();
        let mimi_init = self.model.init_mimi_state(1)?;
        let mut post = Chain::from_modules(self.sample_rate(), &self.post_process);

        let (pcm_tx, pcm_rx) = std::sync::mpsc::sync_channel::<Result<Vec<f32>>>(PCM_QUEUE);

        let workers = if self.inline_decode {
            // One shared command queue (Metal): a second thread has no second execution
            // unit to overlap onto, and only adds cross-thread contention for that queue. Decode
            // each frame inline, on the flow-LM's own thread, instead -- see `inline_decode`.
            let model = Arc::clone(&self.model);
            let graphs = Arc::clone(&self.graphs);
            let handle = std::thread::spawn(move || {
                let _in_flight = in_flight;
                let result = run_inline(
                    &model, &graphs, chunks, base_state, cfg_base, rng, mimi_init, &pcm_tx, post,
                );
                if let Err(e) = result {
                    let _ = pcm_tx.send(Err(e));
                }
            });
            vec![handle]
        } else {
            let (latent_tx, latent_rx) = std::sync::mpsc::sync_channel::<Frame<Q>>(LATENT_QUEUE);
            // The decoder says when the first frame's audio is out, and the flow-LM waits for it
            // before its second step. Run side by side, the two compete for the device and the
            // first decode, which is all the time to first audio waits on, comes out slower; held
            // back, the flow-LM loses one decode's time, far ahead of real time.
            let (first_tx, first_rx) = std::sync::mpsc::channel::<()>();

            // Decoder: latents in, PCM out. Reset between chunks so each chunk
            // starts from a clean codec state, matching the pre-refactor behavior.
            let decode_model = Arc::clone(&self.model);
            let decode_tx = pcm_tx.clone();
            let decode_handle = std::thread::spawn(move || {
                let mut first_tx = Some(first_tx);
                let mut state = mimi_init.clone();
                // The post-processing chain delays its output: its first `skip` samples are the
                // zeros it was primed with, dropped here, and the stream's last are flushed below.
                let mut skip = post.latency();
                while let Ok(frame) = latent_rx.recv() {
                    let first = match frame {
                        Frame::ChunkEnd => {
                            state = mimi_init.clone();
                            continue;
                        }
                        Frame::Latent(latent) => match decode_model.mimi_latent(latent) {
                            Ok(latent) => latent,
                            Err(e) => {
                                let _ = decode_tx.send(Err(e.into()));
                                return;
                            }
                        },
                    };
                    // Decode every latent the flow-LM has already queued in one
                    // call: the decoder is exact for any number of frames, and one
                    // call over several is much cheaper than one call per frame.
                    // Only what is already waiting is taken, so no frame is ever
                    // held back for a batch to fill — the first one included.
                    let mut reset_after = false;
                    let mut batch = vec![first];
                    while batch.len() < DECODE_BATCH {
                        let Ok(frame) = latent_rx.try_recv() else { break };
                        match frame {
                            // Past a chunk boundary the codec state resets, so the
                            // batch has to stop here and resume on the next frame.
                            Frame::ChunkEnd => {
                                reset_after = true;
                                break;
                            }
                            Frame::Latent(latent) => match decode_model.mimi_latent(latent) {
                                Ok(latent) => batch.push(latent),
                                Err(e) => {
                                    let _ = decode_tx.send(Err(e.into()));
                                    return;
                                }
                            },
                        }
                    }
                    let latent = match batch.len() {
                        1 => batch.pop().expect("just checked"),
                        _ => match Tensor::cat(&batch.iter().collect::<Vec<_>>(), 1) {
                            Ok(latent) => latent,
                            Err(e) => {
                                let _ = decode_tx.send(Err(e.into()));
                                return;
                            }
                        },
                    };
                    let pcm = decode_model
                        .decode_latent(&latent, &mut state)
                        .and_then(|audio| audio.narrow(0, ..1)?.contiguous()?.to_vec())
                        .map(|mut pcm| {
                            post.process(&mut pcm);
                            let n = skip.min(pcm.len());
                            skip -= n;
                            pcm.drain(..n);
                            pcm
                        })
                        .map_err(Error::from);
                    let failed = pcm.is_err();
                    if decode_tx.send(pcm).is_err() || failed {
                        return;
                    }
                    if let Some(first_tx) = first_tx.take() {
                        let _ = first_tx.send(());
                    }
                    if reset_after {
                        state = mimi_init.clone();
                    }
                }
                let latency = post.latency();
                if latency > skip {
                    let mut tail = vec![0.0; latency];
                    post.process(&mut tail);
                    tail.drain(..skip);
                    let _ = decode_tx.send(Ok(tail));
                }
            });

            // Flow-LM: text in, latents out.
            let model = Arc::clone(&self.model);
            let graphs = Arc::clone(&self.graphs);
            let backbone_handle = std::thread::spawn(move || {
                // Dropped when this thread ends, which is after its last write.
                let _in_flight = in_flight;
                let result = run_backbone(
                    &model, &graphs, chunks, base_state, cfg_base, rng, &latent_tx, first_rx,
                );
                if let Err(e) = result {
                    let _ = pcm_tx.send(Err(e));
                }
            });

            vec![backbone_handle, decode_handle]
        };

        Ok(SpeechStream {
            rx: Some(pcm_rx),
            sample_rate: self.sample_rate(),
            failed: false,
            workers: Some(workers),
        })
    }
}

impl<Q: BackendQ> SessionApi for SessionOf<Q> {
    fn seq_budget(&self) -> usize {
        self.seq_budget
    }

    fn sample_rate(&self) -> u32 {
        self.model.sample_rate() as u32
    }

    fn normalization(&self) -> Normalize {
        self.normalize
    }

    fn say(&self, text: &str) -> Result<Vec<f32>> {
        let mut pcm = Vec::new();
        for chunk in self.stream(text)? {
            pcm.extend_from_slice(&chunk?);
        }
        Ok(pcm)
    }

    fn stream(&self, text: &str) -> Result<SpeechStream> {
        let rng = Box::new(NormalRng::new(self.temperature, self.seed)?);
        self.stream_with_rng(text, rng)
    }

    fn stream_seeded(&self, text: &str, seed: u64) -> Result<SpeechStream> {
        let rng = Box::new(NormalRng::new(self.temperature, seed)?);
        self.stream_with_rng(text, rng)
    }

    fn say_seeded(&self, text: &str, seed: u64) -> Result<Vec<f32>> {
        let mut pcm = Vec::new();
        for chunk in self.stream_seeded(text, seed)? {
            pcm.extend_from_slice(&chunk?);
        }
        Ok(pcm)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.model.flow_lm.conditioner.tokenize(text)?)
    }

    fn stream_tokens(
        &self,
        tokens: Vec<u32>,
        frames_after_eos: usize,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream> {
        if tokens.is_empty() {
            return Err(Error::invalid_argument("nothing to synthesize: no tokens"));
        }
        let frame_budget = plan::frame_budget(tokens.len(), self.frame_rate);
        let chunk = Chunk { text: String::new(), tokens, frames_after_eos, frame_budget };
        self.stream_chunks(vec![chunk], rng)
    }
}

/// Slots the primed state already occupies — the voice prompt's frames.
///
/// Read off the first flow-LM layer: every layer advances together, and a Mimi
/// layer has no such position.
fn primed_len<Q: BackendQ>(state: &TTSState<Q>) -> usize {
    state
        .flow_lm_state
        .transformer_state
        .layer_states
        .iter()
        .find_map(|layer| match layer {
            crate::transformer::LayerAttentionState::FlowLm(mha) => Some(mha.current_end),
            _ => None,
        })
        .unwrap_or(0)
}

/// Clears a session's in-flight flag when the generation that set it ends.
///
/// Moved into the flow-LM thread's closure, so the flag clears exactly when the
/// last write to the shared KV buffers happens — including the early-drop case,
/// where the thread runs one more step before it notices the closed channel.
struct InFlight(Arc<std::sync::atomic::AtomicBool>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::Release);
    }
}

// Fixed queues propagate slow-consumer backpressure through both generation workers.
const PCM_QUEUE: usize = 4;
const LATENT_QUEUE: usize = 16;
const DECODE_BATCH: usize = LATENT_QUEUE;

/// Messages from the flow-LM thread to the decoder thread.
enum Frame<Q: BackendQ> {
    Latent(MimiLatent<Q>),
    ChunkEnd,
}

/// A chunk's flow-LM state: the stream's own copy of the primed state, with its null branch
/// under guidance, or the state of a recorded step. One per chunk, so its size is no concern.
#[allow(clippy::large_enum_variant)]
enum Flow<'a, Q: BackendQ> {
    Own(TTSState<Q>, Option<(f32, TTSState<Q>)>),
    Graph(&'a mut StepGraph<Q>),
}

impl<'a, Q: BackendQ> Flow<'a, Q> {
    /// Start a chunk from the primed state, prompted with its tokens.
    fn start(
        model: &TTSModel<Q>,
        base: &TTSState<Q>,
        cfg_base: &Option<(f32, TTSState<Q>)>,
        graph: Option<&'a mut StepGraph<Q>>,
        tokens: &[u32],
    ) -> Result<Self> {
        if let Some(graph) = graph {
            graph.begin(model, base, tokens)?;
            return Ok(Self::Graph(graph));
        }
        let mut state = base.clone();
        let mut cfg_state = cfg_base.clone();
        model.prompt_text(&mut state, tokens)?;
        if let Some((_, null_state)) = cfg_state.as_mut() {
            model.prompt_text_null(null_state)?;
        }
        Ok(Self::Own(state, cfg_state))
    }

    /// One frame of the autoregressive loop, from BOS when there is no previous latent.
    #[allow(clippy::type_complexity)]
    fn step(
        &mut self,
        model: &TTSModel<Q>,
        prev: Option<&Tensor<Q::T, Q::B>>,
        rng: &mut Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<(Tensor<Q::T, Q::B>, FrameEos<Q>)> {
        let input = match prev {
            None => StepInput::Bos { batch: 1 },
            Some(t) => StepInput::Latent(t),
        };
        Ok(match self {
            Self::Graph(graph) => {
                let (next, logit) = graph.step(model, prev, rng)?;
                (next, FrameEos::Logit(logit))
            }
            Self::Own(state, Some((coef, null_state))) => {
                let (next, is_eos) =
                    model.generate_step_cfg(state, null_state, *coef, input, rng)?;
                (next, FrameEos::Known(is_eos))
            }
            Self::Own(state, None) => {
                let (next, logit) = model.generate_step_parts(state, input, rng)?;
                (next, FrameEos::Logit(logit))
            }
        })
    }
}

/// Whether a frame ends its chunk: decided already, or a logit still on the device.
enum FrameEos<Q: BackendQ> {
    Known(bool),
    Logit(Tensor<Q::T, Q::B>),
}

impl<Q: BackendQ> FrameEos<Q> {
    fn resolve(self, model: &TTSModel<Q>) -> Result<bool> {
        match self {
            Self::Known(is_eos) => Ok(is_eos),
            Self::Logit(logit) => Ok(model.eos_from_logit(&logit.to_vec()?)),
        }
    }
}

/// The step graph a generation from `base_state` uses, if any: none under guidance.
fn step_graph<Q: BackendQ>(
    model: &TTSModel<Q>,
    graphs: &GraphPool<Q>,
    base_state: &TTSState<Q>,
    cfg_base: &Option<(f32, TTSState<Q>)>,
) -> Result<Option<StepGraph<Q>>> {
    match cfg_base {
        None => Ok(graphs.step(model, base_state)?),
        Some(_) => Ok(None),
    }
}

/// The autoregressive loop, shared by every frontend and every chunk.
#[allow(clippy::too_many_arguments)]
fn run_backbone<Q: BackendQ>(
    model: &TTSModel<Q>,
    graphs: &GraphPool<Q>,
    chunks: Vec<Chunk>,
    base_state: TTSState<Q>,
    cfg_base: Option<(f32, TTSState<Q>)>,
    rng: Box<dyn crate::flow_lm::Rng + Send>,
    latent_tx: &std::sync::mpsc::SyncSender<Frame<Q>>,
    first_decoded: std::sync::mpsc::Receiver<()>,
) -> Result<()> {
    let mut graph = step_graph(model, graphs, &base_state, &cfg_base)?;
    let result = backbone_chunks(
        model,
        graph.as_mut(),
        chunks,
        base_state,
        cfg_base,
        rng,
        latent_tx,
        first_decoded,
    );
    if let (Ok(()), Some(graph)) = (&result, graph) {
        graphs.put_step(graph);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn backbone_chunks<Q: BackendQ>(
    model: &TTSModel<Q>,
    mut graph: Option<&mut StepGraph<Q>>,
    chunks: Vec<Chunk>,
    base_state: TTSState<Q>,
    cfg_base: Option<(f32, TTSState<Q>)>,
    mut rng: Box<dyn crate::flow_lm::Rng + Send>,
    latent_tx: &std::sync::mpsc::SyncSender<Frame<Q>>,
    first_decoded: std::sync::mpsc::Receiver<()>,
) -> Result<()> {
    // Taken after the first frame: only the stream's first audio is worth the wait.
    let mut first_decoded = Some(first_decoded);
    for chunk in chunks.iter() {
        let mut flow =
            Flow::start(model, &base_state, &cfg_base, graph.as_deref_mut(), &chunk.tokens)?;
        let mut prev: Option<Tensor<Q::T, Q::B>> = None;
        let mut eos = EosPolicy::new(chunk.frames_after_eos);

        for _ in 0..chunk.frame_budget {
            let (next, frame_eos) = flow.step(model, prev.as_ref(), &mut rng)?;
            let is_eos = frame_eos.resolve(model)?;
            // A closed channel means the consumer went away; stop quietly and
            // let the decoder thread report any error of its own.
            if latent_tx.send(Frame::Latent(model.latent_for_mimi(&next)?)).is_err() {
                return Ok(());
            }
            // A closed channel here means the decoder stopped, for the same reasons.
            if let Some(first_decoded) = first_decoded.take()
                && first_decoded.recv().is_err()
            {
                return Ok(());
            }
            if eos.should_stop(is_eos) {
                break;
            }
            prev = Some(next);
        }
        if latent_tx.send(Frame::ChunkEnd).is_err() {
            return Ok(());
        }
    }
    Ok(())
}

/// Like [`run_backbone`], but decodes each frame immediately on the same thread instead of
/// handing it to a second one over a channel -- see [`SynthOf::inline_decode`]. With nothing
/// running concurrently there is no second thread to rendezvous with before the second step,
/// unlike `run_backbone`.
#[allow(clippy::too_many_arguments)]
fn run_inline<Q: BackendQ>(
    model: &TTSModel<Q>,
    graphs: &GraphPool<Q>,
    chunks: Vec<Chunk>,
    base_state: TTSState<Q>,
    cfg_base: Option<(f32, TTSState<Q>)>,
    rng: Box<dyn crate::flow_lm::Rng + Send>,
    mimi_init: crate::mimi::MimiDecoderState<f32, Q::B>,
    pcm_tx: &std::sync::mpsc::SyncSender<Result<Vec<f32>>>,
    post: Chain,
) -> Result<()> {
    let mut graph = step_graph(model, graphs, &base_state, &cfg_base)?;
    let result = inline_chunks(
        model,
        graph.as_mut(),
        chunks,
        base_state,
        cfg_base,
        rng,
        mimi_init,
        pcm_tx,
        post,
    );
    if let (Ok(()), Some(graph)) = (&result, graph) {
        graphs.put_step(graph);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn inline_chunks<Q: BackendQ>(
    model: &TTSModel<Q>,
    mut graph: Option<&mut StepGraph<Q>>,
    chunks: Vec<Chunk>,
    base_state: TTSState<Q>,
    cfg_base: Option<(f32, TTSState<Q>)>,
    mut rng: Box<dyn crate::flow_lm::Rng + Send>,
    mimi_init: crate::mimi::MimiDecoderState<f32, Q::B>,
    pcm_tx: &std::sync::mpsc::SyncSender<Result<Vec<f32>>>,
    mut post: Chain,
) -> Result<()> {
    let mut mimi_state = mimi_init.clone();
    // The post-processing chain delays its output: its first `skip` samples are the zeros it
    // was primed with, dropped here, and the stream's last are flushed below.
    let mut skip = post.latency();
    for chunk in chunks.iter() {
        let mut flow =
            Flow::start(model, &base_state, &cfg_base, graph.as_deref_mut(), &chunk.tokens)?;
        let mut prev: Option<Tensor<Q::T, Q::B>> = None;
        let mut eos = EosPolicy::new(chunk.frames_after_eos);

        for _ in 0..chunk.frame_budget {
            let (next, frame_eos) = flow.step(model, prev.as_ref(), &mut rng)?;
            let pcm = model
                .decode_latent(&next, &mut mimi_state)
                .and_then(|audio| audio.narrow(0, ..1)?.contiguous()?.to_vec())
                .map(|mut pcm| {
                    post.process(&mut pcm);
                    let n = skip.min(pcm.len());
                    skip -= n;
                    pcm.drain(..n);
                    pcm
                })
                .map_err(Error::from);
            let failed = pcm.is_err();
            // A closed channel means the consumer went away; stop quietly and let the caller
            // report any error of its own.
            if pcm_tx.send(pcm).is_err() || failed {
                return Ok(());
            }
            // Without guidance the eos logit is read back after the frame's audio, so the frame
            // waits on the device once instead of twice.
            if eos.should_stop(frame_eos.resolve(model)?) {
                break;
            }
            prev = Some(next);
        }
        mimi_state = mimi_init.clone();
    }
    let latency = post.latency();
    if latency > skip {
        let mut tail = vec![0.0; latency];
        post.process(&mut tail);
        tail.drain(..skip);
        let _ = pcm_tx.send(Ok(tail));
    }
    Ok(())
}

/// PCM chunks from a running generation.
///
/// Each item is mono `f32` samples at [`Self::sample_rate`] — one Mimi frame's
/// worth, or several when the decoder finds more than one frame already queued
/// and decodes them together. The iterator ends when generation finishes; an
/// `Err` item is terminal.
///
/// Generation runs ahead only while its fixed audio and latent buffers have room.
/// Leaving the stream unread pauses generation once they fill; consuming audio
/// lets it resume. Iteration and cleanup block, including waiting for the current
/// model operation on cancellation or error. Use a blocking thread in async code.
pub struct SpeechStream {
    /// `Option` so [`Drop`] can release it before joining: closing this receiver
    /// wakes a decoder blocked on output, which closes the latent receiver
    /// and wakes the backbone in turn.
    rx: Option<std::sync::mpsc::Receiver<Result<Vec<f32>>>>,
    sample_rate: u32,
    failed: bool,
    /// The generation thread(s), joined once the channel closes so that a panic in any of them
    /// surfaces as an error rather than as truncated audio. Two threads (flow LM and Mimi
    /// decoder) on the CPU, one thread doing both inline on devices with a single shared
    /// command queue; see [`SynthOf::inline_decode`].
    workers: Option<Vec<std::thread::JoinHandle<()>>>,
}

impl SpeechStream {
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn join_workers(&mut self) -> bool {
        let mut died = false;
        if let Some(workers) = self.workers.take() {
            for worker in workers {
                died |= worker.join().is_err();
            }
        }
        died
    }
}

/// Joining on drop is what makes a session's in-flight flag exact: dropping the
/// receiver closes the channels, the workers stop, and only once they have
/// stopped writing does the flag clear. Without it, "drop the stream, start the
/// next one" could still be refused.
impl Drop for SpeechStream {
    fn drop(&mut self) {
        // Releasing the receiver first is what stops the workers; joining
        // before that would wait for the whole generation.
        self.rx.take();
        self.join_workers();
    }
}

impl Iterator for SpeechStream {
    type Item = Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        match self.rx.as_ref()?.recv() {
            Ok(Err(e)) => {
                self.failed = true;
                self.rx.take();
                self.join_workers();
                Some(Err(e))
            }
            Ok(ok) => Some(ok),
            // Every sender dropped: generation finished, or a worker died.
            Err(_) => {
                self.failed = true;
                self.workers.as_ref()?;
                let died = self.join_workers();
                if died {
                    Some(Err(Error::Tensor(xn::Error::msg(
                        "a generation worker panicked; the audio is incomplete",
                    ))))
                } else {
                    None
                }
            }
        }
    }
}

/// Configures and loads a [`Synth`].
///
/// The config, the weights path and the normalization policy are required, and
/// none has a default. Which files a checkpoint ships, what they are called and
/// where its voices live all change from one release to the next, so locating
/// them belongs to the frontend; this reads the files it is handed. The
/// language is required for a different reason: see [`Self::new`].
pub struct SynthBuilder {
    config: TTSConfig,
    weights: PathBuf,
    tokenizer_file: Option<PathBuf>,
    device: DeviceKind,
    quant: Quant,
    tokenizer: Option<Box<dyn crate::Tokenizer + Send + Sync>>,
    temperature: f32,
    seed: u64,
    cfg_coef: Option<f32>,
    eos_threshold: Option<f32>,
    voice: Option<String>,
    max_tokens_per_chunk: usize,
    post_process: Vec<ModuleConfig>,
    voices: Vec<(String, PathBuf)>,
    normalize: Normalize,
    conditions: HashMap<String, String>,
}

impl SynthBuilder {
    /// A checkpoint's config, its weights file (GGUF or safetensors), and how
    /// to normalize text.
    ///
    /// `normalize` has no default on purpose. Normalization makes the model
    /// noticeably better, so it should not be something a caller forgets to
    /// turn on — but the spoken forms of `@`, `+` and `=` are per-language, so
    /// normalizing German as English makes it say "at" where it should say
    /// "ät". Guessing is worse than doing nothing, so the caller says which:
    /// [`Normalize::for_lang`] with a language, or [`Normalize::OFF`] to hand text
    /// to the tokenizer as written.
    ///
    /// ```no_run
    /// # fn main() -> ptts::Result<()> {
    /// use ptts::preprocess::Lang;
    /// use ptts::synth::SynthBuilder;
    /// use ptts::checkpoint::read_config;
    ///
    /// let tts = SynthBuilder::new(
    ///     read_config("model/config.json")?,
    ///     "model/model.safetensors",
    ///     Lang::De,
    /// )
    /// .tokenizer_file("model/tokenizer.json")
    /// .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        config: TTSConfig,
        weights: impl Into<PathBuf>,
        normalize: impl Into<Normalize>,
    ) -> Self {
        Self {
            config,
            weights: weights.into(),
            normalize: normalize.into(),
            tokenizer_file: None,
            device: DeviceKind::Auto,
            quant: Quant::F32,
            tokenizer: None,
            temperature: 0.3,
            seed: 4242424242424242,
            cfg_coef: None,
            eos_threshold: None,
            voice: None,
            max_tokens_per_chunk: MAX_TOKENS_PER_CHUNK,
            post_process: vec![],
            voices: vec![],
            conditions: HashMap::new(),
        }
    }

    /// The `tokenizer.json` the checkpoint ships, read by [`crate::tok::Tok`].
    /// Ignored when [`Self::tokenizer`] supplies one directly.
    pub fn tokenizer_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.tokenizer_file = Some(path.into());
        self
    }

    pub fn device(mut self, device: DeviceKind) -> Self {
        self.device = device;
        self
    }

    /// Weight format for the flow-LM transformer linears. CPU only.
    pub fn quant(mut self, quant: Quant) -> Self {
        self.quant = quant;
        self
    }

    /// Supply the tokenizer explicitly. Required when the checkpoint ships no
    /// tokenizer file, or when the `hf` feature is not enabled.
    pub fn tokenizer(mut self, tokenizer: Box<dyn crate::Tokenizer + Send + Sync>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Classifier-free guidance coefficient applied to every request. `1.0`
    /// disables it.
    pub fn cfg_coef(mut self, cfg_coef: f32) -> Self {
        self.cfg_coef = Some(cfg_coef);
        self
    }

    pub fn condition(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.conditions.insert(name.into(), value.into());
        self
    }

    /// Override the config's EOS log-probability threshold. Lower values let
    /// the model run longer before it decides an utterance is finished.
    pub fn eos_threshold(mut self, eos_threshold: f32) -> Self {
        self.eos_threshold = Some(eos_threshold);
        self
    }

    /// Default voice for requests that do not name one. Without it,
    /// [`SynthApi::default_voice`] picks one.
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Maximum text tokens per synthesized chunk. Longer text is split on
    /// sentence boundaries.
    pub fn max_tokens_per_chunk(mut self, max_tokens: usize) -> Self {
        self.max_tokens_per_chunk = max_tokens;
        self
    }

    /// Audio modules run over the PCM of every request, in order, unless the
    /// request's [`SpeechOptions::post_process`] replaces them.
    pub fn post_process(mut self, modules: impl IntoIterator<Item = ModuleConfig>) -> Self {
        self.post_process = modules.into_iter().collect();
        self
    }

    /// Register a precomputed voice embedding at load time. A voice that fails
    /// to load fails the whole load; a frontend registering a checkpoint's own
    /// voices, where one bad embedding should not make the model unusable, can
    /// instead loop over [`SynthApi::add_voice_file`] afterwards and warn.
    pub fn add_voice(mut self, name: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        self.voices.push((name.into(), path.into()));
        self
    }

    /// Load the checkpoint, choosing the device and weight format from
    /// [`Self::device`] and [`Self::quant`].
    pub fn build(self) -> Result<Synth> {
        if let Some(warning) = self.normalize.language_warning(&self.config.languages) {
            tracing::warn!("{warning}");
        }
        let device = self.device.resolve_for(self.quant);
        self.quant.check_device(device)?;
        match device {
            DeviceKind::Cpu => self.build_cpu(),
            DeviceKind::Cuda => self.build_cuda(),
            DeviceKind::Vulkan => self.build_vulkan(),
            DeviceKind::Metal => self.build_metal(),
            DeviceKind::Auto => unreachable!("resolved above"),
        }
    }

    fn build_cpu(self) -> Result<Synth> {
        macro_rules! cpu {
            ($q:ty) => {
                Ok(Synth(Box::new(self.load::<$q>(xn::CPU, false)?)))
            };
        }
        match self.quant {
            Quant::F32 => cpu!(xn::Unquantized<f32, xn::CpuDevice>),
            Quant::Q80 => cpu!(xn::quantized::Q80F32),
            Quant::Q81 => cpu!(xn::quantized::Q81F32),
            Quant::Q8k => cpu!(xn::quantized::Q8kF32),
            Quant::Q6k => cpu!(xn::quantized::Q6kF32),
            Quant::Q50 => cpu!(xn::quantized::Q50F32),
            Quant::Q51 => cpu!(xn::quantized::Q51F32),
            Quant::Q5k => cpu!(xn::quantized::Q5kF32),
            Quant::Q40 => cpu!(xn::quantized::Q40F32),
            Quant::Q41 => cpu!(xn::quantized::Q41F32),
            Quant::Q4k => cpu!(xn::quantized::Q4kF32),
        }
    }

    #[cfg(feature = "cuda")]
    fn build_cuda(self) -> Result<Synth> {
        use xn::cuda_backend::Device;
        // `quant()` reports `self.quant`, which `check_device` keeps at F32 here.
        debug_assert_eq!(self.quant, Quant::F32);
        // Mimi gets a device handle of its own, so a stream and a cuBLAS handle of its own, and
        // decodes on its own thread, overlapping the flow LM. Both handles share the context.
        let mimi_dev = Device::new(0)?;
        let dev = Device::new(0)?;
        // Event tracking costs a few percent and this workload never queries events.
        unsafe { dev.disable_event_tracking() };
        let synth = self.load_on::<xn::Unquantized<half::bf16, _>>(dev, Some(mimi_dev), false)?;
        Ok(Synth(Box::new(synth)))
    }

    #[cfg(feature = "vulkan")]
    fn build_vulkan(self) -> Result<Synth> {
        // `quant()` reports `self.quant`, which `check_device` keeps at F32 here.
        debug_assert_eq!(self.quant, Quant::F32);
        let dev = xn::vulkan_backend::Device::new(0)?;
        let synth = self.load::<xn::Unquantized<f32, _>>(dev, false)?;
        Ok(Synth(Box::new(synth)))
    }

    #[cfg(feature = "metal")]
    fn build_metal(self) -> Result<Synth> {
        // `quant()` reports `self.quant`, which `check_device` keeps at F32 here.
        debug_assert_eq!(self.quant, Quant::F32);
        let dev = xn::metal_backend::Device::new(0)?;
        // One shared command queue, so decode inline; see `inline_decode`.
        let synth = self.load::<xn::Unquantized<half::bf16, _>>(dev, true)?;
        Ok(Synth(Box::new(synth)))
    }

    // Feature-disabled stubs keep dispatch compilable. The shared check rejects these
    // backends before dispatch; the stubs retain errors as a backstop.
    #[cfg(not(feature = "cuda"))]
    fn build_cuda(self) -> Result<Synth> {
        Err(Error::unsupported("this build has no CUDA support; rebuild with the `cuda` feature"))
    }

    #[cfg(not(feature = "vulkan"))]
    fn build_vulkan(self) -> Result<Synth> {
        Err(Error::unsupported(
            "this build has no Vulkan support; rebuild with the `vulkan` feature",
        ))
    }

    #[cfg(not(feature = "metal"))]
    fn build_metal(self) -> Result<Synth> {
        Err(Error::unsupported("this build has no Metal support; rebuild with the `metal` feature"))
    }

    /// Load the weights and register the voices. `inline_decode` is forwarded to
    /// [`SynthOf`]; see its field doc.
    fn load<Q: BackendQ>(self, device: Q::B, inline_decode: bool) -> Result<SynthOf<Q>> {
        self.load_on::<Q>(device, None, inline_decode)
    }

    /// As [`Self::load`], with Mimi on `mimi_device` when given.
    fn load_on<Q: BackendQ>(
        mut self,
        device: Q::B,
        mimi_device: Option<Q::B>,
        inline_decode: bool,
    ) -> Result<SynthOf<Q>> {
        if !self.weights.is_file() {
            return Err(Error::not_found(format!(
                "weights file not found: {}",
                self.weights.display()
            )));
        }
        let config = self.config.clone();
        let tokenizer = self.take_tokenizer()?;

        let vb = loader::load_weights::<Q>(&self.weights, &device)?;
        let mut conditions = self.conditions.clone();
        if let Some(voice) = config.voices.first() {
            for (k, v) in voice.conditions.iter() {
                conditions.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        let model = TTSModel::<Q>::load(&vb, tokenizer, &config, &conditions)?;
        let mut model = match self.eos_threshold {
            Some(threshold) => model.with_eos_threshold(threshold),
            None => model,
        };
        if let Some(mimi_device) = mimi_device {
            let mimi_vb = loader::load_weights::<Q>(&self.weights, &mimi_device)?;
            model.move_mimi_to(&mimi_vb, &config)?;
        }
        // A dedicated speaker codec ships its encoder under its own prefix, so
        // probe there rather than assuming `mimi.encoder.*`.
        let probe = format!("{}.encoder.model.0.conv.weight", config.speaker_mimi_prefix());
        let mimi_enc =
            if vb.contains(&probe) { Some(MimiEnc::<Q>::load(&vb, &config)?) } else { None };
        let baked_voices = loader::load_config_voices(&vb, &config, model.speaker_proj())?;
        let conditioners =
            Conditioners::load(&vb.pp("flow_lm.condition_provider.conditioners"), &config)?;
        vb.check_all_used_with_ignore(loader::is_unused_by_tts_model)?;

        let mut synth = SynthOf {
            quant: self.quant,
            model: Arc::new(model),
            mimi_enc,
            cfg: config,
            voices: BTreeMap::new(),
            primed: Mutex::new(HashMap::new()),
            defaults: Defaults {
                voice: self.voice.clone(),
                temperature: self.temperature,
                seed: self.seed,
                cfg_coef: self.cfg_coef,
                max_tokens_per_chunk: self.max_tokens_per_chunk,
                post_process: self.post_process.clone(),
            },
            normalize: self.normalize,
            conditioners,
            conditions: None,
            inline_decode,
            graphs: Arc::default(),
        };

        for (name, emb) in baked_voices {
            synth
                .voices
                .insert(name, Voice { emb: emb.to::<Q::T>()?, null_emb: None, conditions: None });
        }
        synth.set_conditions(self.conditions.clone())?;
        for (name, path) in self.voices.iter() {
            synth.add_voice_file(name, path)?;
        }

        if let Some(name) = synth.defaults.voice.as_ref()
            && !synth.voices.contains_key(name)
        {
            let known = synth.voices();
            return Err(Error::UnknownVoice { name: name.clone(), known });
        }
        Ok(synth)
    }

    /// A caller-supplied tokenizer wins, for callers with no filesystem to read
    /// one from. Otherwise load the one the checkpoint shipped, if a tokenizer
    /// backend is compiled in.
    fn take_tokenizer(&mut self) -> Result<Box<dyn crate::Tokenizer + Send + Sync>> {
        if let Some(tokenizer) = self.tokenizer.take() {
            return Ok(tokenizer);
        }
        match self.tokenizer_file.as_deref() {
            #[cfg(feature = "hf")]
            Some(path) => Ok(Box::new(crate::tok::Tok::open(path)?)),
            #[cfg(not(feature = "hf"))]
            Some(path) => Err(Error::unsupported(format!(
                "cannot read the tokenizer at {}: `ptts` was built without the `hf` feature.",
                path.display()
            ))),
            None => Err(Error::unsupported(
                "no tokenizer available: no `tokenizer.json` was found beside the checkpoint, \
                 and none was passed to SynthBuilder::tokenizer or SynthBuilder::tokenizer_file.",
            )),
        }
    }
}

mod sealed {
    /// Lets only this module implement [`super::SynthApi`] and [`super::SessionApi`], so
    /// methods can be added to them without breaking anyone.
    pub trait Sealed {}
    impl<Q: xn::BackendQ> Sealed for super::SynthOf<Q> {}
    impl<Q: xn::BackendQ> Sealed for super::SessionOf<Q> {}
}

/// What a loaded model does, whatever its weight format and device.
///
/// [`Synth`] dereferences to this, so these are called on a `Synth` directly, with no import.
pub trait SynthApi: sealed::Sealed + Send + Sync {
    /// Sample rate of the audio this model produces, in Hz.
    fn sample_rate(&self) -> u32;

    /// The config the model was loaded with.
    fn config(&self) -> &TTSConfig;

    /// How requests are normalized.
    ///
    /// Every text-taking method here applies it already; it is public for
    /// callers that drive [`SessionApi::stream_tokens`] and so tokenize
    /// themselves, and want [`Normalize::apply`] first.
    fn normalization(&self) -> Normalize;

    /// Name of the device the model is running on, e.g. `"cpu"` or `"cuda:0"`.
    fn device_name(&self) -> String;

    /// Registered voice names, sorted.
    fn voices(&self) -> Vec<String>;

    /// The languages the model was trained for.
    fn languages(&self) -> &[String];

    /// The voice a request that names none speaks in: the builder's [`SynthBuilder::voice`],
    /// else `default` (a checkpoint's own `default-voice.safetensors`), else the first by name.
    /// Decided per request, so a voice registered after the build counts. `None` when there
    /// are no voices, and generation is then unconditioned.
    fn default_voice(&self) -> Option<String>;

    /// True if this checkpoint carries a speaker encoder, which voice cloning
    /// from raw audio requires.
    fn supports_voice_cloning(&self) -> bool;

    /// The sample rate [`Self::add_voice_from_pcm`] expects.
    fn voice_prompt_sample_rate(&self) -> u32;

    /// Register a precomputed voice embedding, replacing any voice of the same name.
    fn add_voice_file(&mut self, name: &str, path: &FsPath) -> Result<()>;

    /// Register every voice of a JSONL catalog, see [`loader::read_voice_catalog`].
    fn add_voice_catalog(&mut self, path: &FsPath) -> Result<()> {
        for (name, path) in loader::read_voice_catalog(path)? {
            self.add_voice_file(&name, &path)?;
        }
        Ok(())
    }

    /// Clone a voice from a mono audio prompt.
    ///
    /// `pcm` must be at [`Self::voice_prompt_sample_rate`] and last between
    /// `audio_prompt_min_duration` and `audio_prompt_max_duration` seconds:
    /// longer input is trimmed, shorter input is an error. The caller's slice is
    /// not modified — loudness normalization runs on an internal copy.
    fn add_voice_from_pcm(&mut self, name: &str, pcm: &[f32]) -> Result<()>;

    /// Register a voice from a conditioning embedding already in memory, laid
    /// out as `frames` rows of `dim`.
    ///
    /// This is the mimi encoder's output -- what [`Self::add_voice_from_pcm`]
    /// computes internally. Callers that compute or cache embeddings themselves
    /// (`ptts-pyo3` hands one over from numpy) use this.
    ///
    /// `null_emb`, when given, is the encoding of equal-length silence, which
    /// CFG needs on models where `cfg_null_audio_empty` is false.
    fn add_voice_from_embedding(
        &mut self,
        name: &str,
        emb: &[f32],
        frames: usize,
        dim: usize,
        null_emb: Option<&[f32]>,
    ) -> Result<()>;

    /// Register a voice from the speaker encoder's latents already in memory,
    /// laid out as `channels` rows of `frames`.
    ///
    /// This is the `speaker_wavs` tensor of a voice file, `[1, channels, frames]`
    /// before the checkpoint's speaker projection: what
    /// [`Self::add_voice_file`] reads from such a file, and the form in which
    /// voices are stored and exchanged. The projection is applied here, so it
    /// fails on a checkpoint that has none.
    fn add_voice_from_latents(
        &mut self,
        name: &str,
        latents: &[f32],
        channels: usize,
        frames: usize,
    ) -> Result<()>;

    fn set_conditions(&mut self, conditions: HashMap<String, String>) -> Result<()>;

    /// Synthesize `text` and return the whole waveform, as mono `f32` at
    /// [`Self::sample_rate`].
    fn say(&self, text: &str) -> Result<Vec<f32>>;

    /// Synthesize `text` with per-request overrides.
    fn say_with(&self, text: &str, opts: &SpeechOptions) -> Result<Vec<f32>>;

    /// Start generating `text`, yielding PCM as the decoder produces it.
    ///
    /// Generation pauses once the bounded buffers fill if the stream is unread.
    /// Consuming audio lets it resume; dropping the stream stops and joins its workers.
    fn stream(&self, text: &str) -> Result<SpeechStream>;

    /// Prime a voice once and keep it, for callers that generate repeatedly.
    ///
    /// The budget covers the configured token target, the selected voice prompt and the
    /// corresponding frame limit. At the default 50-token target and 12.5 Hz, it reserves at
    /// least 796 KV positions. Longer sentences are cut to fit the session's budget.
    /// Use [`Self::session`] to choose a fixed budget explicitly.
    ///
    /// ```no_run
    /// # fn main() -> ptts::Result<()> {
    /// # let tts: ptts::synth::Synth = todo!();
    /// let session = tts.session_default(&ptts::synth::SpeechOptions::default())?;
    /// let pcm = session.say("Hello world")?;
    /// # Ok(())
    /// # }
    /// ```
    fn session_default(&self, opts: &SpeechOptions) -> Result<Session>;

    /// Prime a voice with an explicit KV budget, for callers that generate repeatedly.
    ///
    /// ```no_run
    /// # fn main() -> ptts::Result<()> {
    /// # let tts: ptts::synth::Synth = todo!();
    /// let session = tts.session(&ptts::synth::SpeechOptions::default().voice("default"), 1024)?;
    /// for line in ["First.", "Second.", "Third."] {
    ///     let pcm = session.say(line)?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// `max_seq_len` is the KV budget, allocated up front and held until the
    /// session is dropped. At 12.5 Hz a full [`MAX_TOKENS_PER_CHUNK`]-token
    /// chunk needs 796, so 1024 covers the usual target. Further sentences use
    /// the same cache. A single sentence too long for the budget, or longer than
    /// [`plan::MAX_FIT_TOKENS`], is cut with [`plan::fit`], down to single characters
    /// if needed. A single character that still exceeds either limit is rejected.
    fn session(&self, opts: &SpeechOptions, max_seq_len: usize) -> Result<Session>;

    /// Start generating `text` with per-request overrides.
    ///
    /// Generation runs on two background threads: one for the flow-LM, one for
    /// the Mimi decoder, so decoding overlaps the next backbone step. Generation
    /// pauses once the bounded buffers fill if the stream is unread, and resumes
    /// as audio is consumed. Dropping the returned [`SpeechStream`] stops and joins both.
    fn stream_with(&self, text: &str, opts: &SpeechOptions) -> Result<SpeechStream>;

    /// As [`Self::stream_with`], but with an explicit noise source.
    ///
    /// The solver's only source of randomness is this trait, so replaying a
    /// fixed sequence (see [`crate::flow_lm::ReplayRng`]) makes a generation
    /// reproducible across implementations — which is how this crate is
    /// compared against the reference one. `temperature` and `seed` are ignored.
    fn stream_with_rng(
        &self,
        text: &str,
        opts: &SpeechOptions,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream>;

    /// The weight format actually loaded. GPU backends are always unquantized.
    fn quant(&self) -> Quant;
}

/// What a [`Session`] does: generate repeatedly from one primed voice.
///
/// [`Session`] dereferences to this, so these are called on a `Session` directly.
pub trait SessionApi: sealed::Sealed + Send + Sync {
    /// The KV budget this session was primed with.
    fn seq_budget(&self) -> usize;

    fn sample_rate(&self) -> u32;

    /// How text is normalized.
    ///
    /// [`Self::stream`] and [`Self::say`] apply it themselves. Callers that
    /// tokenize by hand for [`Self::stream_tokens`] should run their text
    /// through [`Normalize::apply`] first, before [`crate::tts_model::prepare_text_prompt`].
    fn normalization(&self) -> Normalize;

    /// Synthesize `text`, returning the whole waveform.
    fn say(&self, text: &str) -> Result<Vec<f32>>;

    /// Synthesize `text`, yielding PCM as the decoder produces it.
    ///
    /// Generation pauses once the bounded buffers fill if the stream is unread.
    /// Consuming audio lets it resume; dropping the stream stops and joins its workers.
    fn stream(&self, text: &str) -> Result<SpeechStream>;

    /// As [`Self::stream`], with an explicit seed for this request.
    fn stream_seeded(&self, text: &str, seed: u64) -> Result<SpeechStream>;

    /// As [`Self::say`], with an explicit seed for this request. Every call on a
    /// session otherwise draws the same noise, since the seed is the session's.
    fn say_seeded(&self, text: &str, seed: u64) -> Result<Vec<f32>>;

    /// Tokenize `text` as given, with none of the preparation [`Self::stream`]
    /// does first — no [`crate::tts_model::prepare_text_prompt`], no sentence splitting.
    ///
    /// Paired with [`Self::stream_tokens`] for callers that want one utterance
    /// per request and prepare the text themselves.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;

    /// Synthesize from tokens produced elsewhere, as one chunk.
    ///
    /// Generation uses the bounded buffers and cleanup described on [`SpeechStream`].
    ///
    /// `frames_after_eos` is the tail [`crate::tts_model::prepare_text_prompt`] would have
    /// chosen: 3 for a very short prompt, 1 otherwise.
    fn stream_tokens(
        &self,
        tokens: Vec<u32>,
        frames_after_eos: usize,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream>;
}

/// A voice primed once, ready to generate repeatedly. Its methods are on [`SessionApi`].
///
/// # One generation at a time
///
/// A session runs one generation at a time, and says so: starting a second
/// while the first is still running is an error, not silent corruption.
///
/// Starting one clones the primed state, and cloning an `xn` tensor shares its
/// storage rather than copying it, so two overlapping generations would write
/// into the same KV buffers and each would attend over the other's keys. The
/// hazard outlives the call — [`SpeechStream`] is `'static` and its workers
/// keep writing after `stream` returns — so a flag held for the life of those
/// workers is what actually enforces it; `!Sync` or `&mut self` cannot see it.
/// Dropping a stream joins its workers, so finishing or dropping one and
/// starting the next always works.
///
/// A server wanting genuine concurrency builds one session per connection,
/// which is what `ptts-ws-server` does.
///
/// Built by [`SynthApi::session`]. Every generation clones the primed state
/// rather than re-running `prompt_audio` over the voice prompt. A one-shot
/// caller should just use [`SynthApi::say`].
///
/// The KV budget is fixed at construction.
pub struct Session(Box<dyn SessionApi>);

impl std::ops::Deref for Session {
    type Target = dyn SessionApi;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("sample_rate", &self.sample_rate())
            .field("seq_budget", &self.seq_budget())
            .finish()
    }
}

/// A loaded Phonon model, ready to synthesize speech.
///
/// See the [module docs](self) for the short version. The weight format and device are chosen
/// at load time by [`SynthBuilder`]; the methods are on [`SynthApi`], which `Synth`
/// dereferences to.
pub struct Synth(Box<dyn SynthApi>);

impl Synth {
    /// A builder over a checkpoint's config and weights file.
    ///
    /// Finding those -- and the tokenizer and voices beside them -- is the
    /// caller's job: see [`SynthBuilder`].
    pub fn builder(
        config: TTSConfig,
        weights: impl Into<PathBuf>,
        normalize: impl Into<Normalize>,
    ) -> SynthBuilder {
        SynthBuilder::new(config, weights, normalize)
    }
}

impl std::ops::Deref for Synth {
    type Target = dyn SynthApi;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl std::ops::DerefMut for Synth {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

impl std::fmt::Debug for Synth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Synth")
            .field("device", &self.device_name())
            .field("weights", &self.quant().as_str())
            .field("sample_rate", &self.sample_rate())
            .field("voices", &self.voices())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::Lang;

    #[test]
    fn auto_runs_quantized_weights_on_the_cpu() {
        // In a GPU build too: a GPU backend refuses quantized weights, so `auto` must not pick one.
        assert_eq!(DeviceKind::Auto.resolve_for(Quant::Q80), DeviceKind::Cpu);
        assert!(Quant::Q80.check_device(DeviceKind::Auto).is_ok());
        assert_eq!(DeviceKind::Auto.resolve_for(Quant::F32), DeviceKind::Auto.resolve());
        assert_eq!(DeviceKind::Metal.resolve_for(Quant::Q80), DeviceKind::Metal);
    }

    fn defaults() -> Defaults {
        Defaults {
            voice: Some("alba".into()),
            temperature: 0.5,
            seed: 42,
            cfg_coef: None,
            max_tokens_per_chunk: 50,
            post_process: vec![],
        }
    }

    #[test]
    fn a_nonsensical_temperature_is_rejected_as_an_argument() {
        // Unchecked it reaches `rand_distr::Normal` and returns as a tensor error about
        // variance, which tells the caller nothing about what they passed.
        for bad in [-0.5f32, f32::NAN] {
            let opts = SpeechOptions::default().temperature(bad);
            let err = resolve(&defaults(), &opts).unwrap_err();
            assert!(matches!(err, Error::InvalidArgument(_)), "{bad}: {err:?}");
        }
        // Zero is greedy sampling, not an error.
        let opts = SpeechOptions::default().temperature(0.0);
        assert_eq!(resolve(&defaults(), &opts).unwrap().temperature, 0.0);
    }

    #[test]
    fn unset_options_leave_the_defaults_alone() {
        let d = defaults();
        let got = resolve(&d, &SpeechOptions::default()).unwrap();
        assert_eq!(got.voice, d.voice);
        assert_eq!(got.temperature, d.temperature);
        assert_eq!(got.seed, d.seed);
        assert_eq!(got.max_tokens_per_chunk, d.max_tokens_per_chunk);
    }

    /// The regression this function exists for: `session` used to drop the
    /// caller's temperature and seed in favour of the builder's. Each field is
    /// also set on its own, since clobbering the rest is the other way to fail.
    #[test]
    fn each_field_overrides_independently() {
        let all = SpeechOptions::default()
            .voice("marius")
            .temperature(0.9)
            .seed(7)
            .max_tokens_per_chunk(80);
        let got = resolve(&defaults(), &all).unwrap();
        assert_eq!(got.voice.as_deref(), Some("marius"));
        assert_eq!(got.temperature, 0.9);
        assert_eq!(got.seed, 7);
        assert_eq!(got.max_tokens_per_chunk, 80);

        let one = resolve(&defaults(), &SpeechOptions::default().seed(99)).unwrap();
        assert_eq!(one.seed, 99);
        assert_eq!(one.temperature, 0.5);
        assert_eq!(one.voice.as_deref(), Some("alba"));
    }

    /// The builder carries the policy it was constructed with, unchanged:
    /// there is no setter that could quietly replace it, and no default that
    /// could stand in for a language the caller never named.
    #[test]
    fn the_builder_keeps_the_policy_it_was_given() {
        for norm in [Normalize::for_lang(Lang::En), Normalize::for_lang(Lang::De), Normalize::OFF] {
            let b = SynthBuilder::new(crate::test_config(), "model.safetensors", norm);
            assert_eq!(b.normalize, norm);
        }
    }

    /// The in-flight flag is what makes "one generation at a time" an error
    /// rather than silent corruption, and the guard is what clears it.
    #[test]
    fn the_in_flight_guard_clears_on_drop() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let flag = Arc::new(AtomicBool::new(false));
        assert!(
            flag.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok(),
            "an idle session must be claimable"
        );
        assert!(
            flag.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err(),
            "a claimed session must refuse a second generation"
        );

        let guard = InFlight(Arc::clone(&flag));
        drop(guard);
        assert!(!flag.load(Ordering::Acquire), "the guard must release the session");
        assert!(
            flag.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok(),
            "and the next generation must be able to claim it"
        );
    }

    #[test]
    fn dropping_a_stream_unblocks_and_joins_both_workers() {
        use std::sync::mpsc;
        use std::time::Duration;

        let (pcm_tx, pcm_rx) = mpsc::sync_channel(0);
        let (latent_tx, latent_rx) = mpsc::sync_channel(1);
        let (full_tx, full_rx) = mpsc::sync_channel(1);
        let backbone = std::thread::spawn(move || {
            latent_tx.send(()).unwrap();
            latent_tx.send(()).unwrap();
            full_tx.send(()).unwrap();
            // The decoder is blocked on PCM output and the latent queue is full.
            assert!(latent_tx.send(()).is_err());
        });
        let decoder = std::thread::spawn(move || {
            while latent_rx.recv().is_ok() {
                if pcm_tx.send(Ok(vec![0.1])).is_err() {
                    return;
                }
            }
        });
        full_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let stream = SpeechStream {
            rx: Some(pcm_rx),
            sample_rate: 24000,
            failed: false,
            workers: Some(vec![backbone, decoder]),
        };
        let (done_tx, done_rx) = mpsc::sync_channel(1);
        let dropper = std::thread::spawn(move || {
            drop(stream);
            done_tx.send(()).unwrap();
        });
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        dropper.join().unwrap();
    }

    #[test]
    fn a_terminal_stream_error_closes_output_and_joins_workers() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let stopped = Arc::new(AtomicBool::new(false));
        let worker_stopped = stopped.clone();
        let worker = std::thread::spawn(move || {
            tx.send(Err(Error::invalid_argument("failed"))).unwrap();
            while tx.send(Ok(vec![0.1])).is_ok() {}
            worker_stopped.store(true, Ordering::Release);
        });
        let other = std::thread::spawn(|| {});
        let mut stream = SpeechStream {
            rx: Some(rx),
            sample_rate: 24000,
            failed: false,
            workers: Some(vec![worker, other]),
        };
        assert!(stream.next().unwrap().is_err());
        assert!(stopped.load(Ordering::Acquire));
        assert!(stream.next().is_none());
    }

    #[test]
    fn guidance_at_one_is_normalized_away() {
        let from_request = SpeechOptions::default().cfg_coef(1.0);
        assert_eq!(resolve(&defaults(), &from_request).unwrap().cfg_coef, None);
        assert_eq!(
            resolve(&defaults(), &SpeechOptions::default().cfg_coef(2.0)).unwrap().cfg_coef,
            Some(2.0)
        );
        // From the builder, and when a request turns the builder's off.
        let enabled = Defaults { cfg_coef: Some(3.0), ..defaults() };
        assert_eq!(
            resolve(&Defaults { cfg_coef: Some(1.0), ..defaults() }, &SpeechOptions::default())
                .unwrap()
                .cfg_coef,
            None
        );
        assert_eq!(resolve(&enabled, &from_request).unwrap().cfg_coef, None);
        assert_eq!(resolve(&enabled, &SpeechOptions::default()).unwrap().cfg_coef, Some(3.0));
    }
}
