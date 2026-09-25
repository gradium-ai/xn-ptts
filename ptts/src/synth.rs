//! One-call speech synthesis.
//!
//! [`Synth`] wraps everything between "I have some text" and "I have PCM":
//! locating and loading a checkpoint, choosing a device and weight format,
//! registering voices, splitting text into sentence chunks, priming the
//! transformer state, running the flow-matching solver, and streaming the
//! latents through the Mimi decoder on a second thread.
//!
//! ```no_run
//! # fn main() -> xn::Result<()> {
//! use ptts::synth::Synth;
//! use ptts::tts_model::TTSConfig;
//!
//! use ptts::preprocess::{Lang, Normalize};
//!
//! let tts = Synth::builder(
//!     TTSConfig::v202601(0.3),
//!     "model/model.safetensors",
//!     Normalize::For(Lang::En),
//! )
//! .tokenizer_file("model/tokenizer.json")
//!     .add_voice("alba", "model/voices/alba.safetensors")
//!     .build()?;
//! let pcm = tts.say("Hello world")?;
//! ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate() as u32)?;
//! # Ok(())
//! # }
//! ```
//!
//! Audio arrives incrementally from [`Synth::stream`], which is the primitive
//! [`Synth::say`] is built on:
//!
//! ```no_run
//! # fn main() -> xn::Result<()> {
//! # let cfg = ptts::tts_model::TTSConfig::v202601(0.3);
//! # let norm = ptts::preprocess::Normalize::For(ptts::preprocess::Lang::En);
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
//! [`Synth::session`] additionally pins the KV budget for a stream of requests.
//!
//! Text is normalized before it is tokenized — see [`crate::preprocess`]. Which
//! language, or [`Normalize::Off`], is a required argument to
//! [`SynthBuilder::new`]: the model reads normalized text noticeably better,
//! but the spoken forms are per-language, so guessing is worse than doing
//! nothing.
//!
//! Callers that want to name the weight format at compile time — `ptts-wasm`
//! supports exactly two — can use [`SynthOf<Q>`] directly via
//! [`SynthBuilder::load`], and skip the runtime dispatch in [`Synth`]. Driving
//! the loop by hand, from an event loop with no threads to spawn, is what
//! [`crate::tts_model::TTSModel`]'s primitives are for.

use crate::flow_lm::{NormalRng, StepInput};
use crate::loader;
use crate::plan::{self, EosPolicy};
use crate::preprocess::Normalize;
use crate::tts_model::{
    MAX_TOKENS_PER_CHUNK, MimiEnc, TTSConfig, TTSModel, TTSState, prepare_text_prompt,
    split_into_best_sentences,
};
use crate::{Error, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex};
use xn::{Backend, BackendQ, Tensor};

/// Which device to run on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceKind {
    /// The most capable compiled-in backend: CUDA, then Vulkan, then Metal,
    /// then the CPU.
    #[default]
    Auto,
    Cpu,
    Cuda,
    Vulkan,
    Metal,
}

impl DeviceKind {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "vulkan" => Ok(Self::Vulkan),
            "metal" => Ok(Self::Metal),
            other => Err(Error::invalid_argument(format!(
                "unknown device '{other}'; expected auto, cpu, cuda, vulkan or metal"
            ))),
        }
    }

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
    /// Parse the spellings the CLIs accept.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
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
            other => Err(Error::invalid_argument(format!(
                "unsupported quantization '{other}'; expected one of \
                 f32, q8_0, q8_1, q8k, q6k, q5_0, q5_1, q5k, q4_0, q4_1, q4k"
            ))),
        }
    }

    /// Error if this weight format cannot run on `device`.
    ///
    /// [`SynthBuilder::build`] checks this too, but a caller that downloads a
    /// checkpoint before building should check first, so an impossible
    /// combination fails in milliseconds rather than after a few hundred
    /// megabytes.
    pub fn check_device(self, device: DeviceKind) -> Result<()> {
        let device = device.resolve();
        if device != DeviceKind::Cpu && self != Self::F32 {
            return Err(Error::unsupported(format!(
                "quantization ({}) is CPU-only, but the selected device is {device:?}",
                self.as_str()
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
}

/// Defaults applied to every request unless overridden per call.
#[derive(Clone, Debug)]
struct Defaults {
    voice: Option<String>,
    temperature: f32,
    seed: u64,
    cfg_coef: Option<f32>,
    max_tokens_per_chunk: usize,
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
    })
}

/// A registered voice: the conditioning embedding, plus the encoding of
/// equal-length silence when the model needs one for CFG.
struct Voice<Q: BackendQ> {
    emb: Tensor<Q::T, Q::B>,
    null_emb: Option<Tensor<Q::T, Q::B>>,
}

/// A loaded model, with the weight format fixed at compile time.
///
/// Most callers want [`Synth`], which erases `Q` so the format can be chosen at
/// runtime.
pub struct SynthOf<Q: BackendQ> {
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
    pub fn sample_rate(&self) -> usize {
        self.model.sample_rate()
    }

    pub fn config(&self) -> &TTSConfig {
        &self.cfg
    }

    /// How requests are normalized.
    ///
    /// Every text-taking method here applies it already; it is public for
    /// callers that drive [`SessionOf::stream_tokens`] and so tokenize
    /// themselves, and want [`Normalize::apply`] first.
    pub fn normalization(&self) -> Normalize {
        self.normalize
    }

    pub fn device_name(&self) -> String {
        self.model.device().name()
    }

    /// Registered voice names, sorted.
    pub fn voices(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    /// True if this checkpoint carries a speaker encoder, which voice cloning
    /// from raw audio requires.
    pub fn supports_voice_cloning(&self) -> bool {
        self.mimi_enc.is_some()
    }

    /// The sample rate [`Self::add_voice_from_pcm`] expects.
    pub fn voice_prompt_sample_rate(&self) -> usize {
        self.cfg.speaker_mimi_cfg().sample_rate
    }

    /// Register a precomputed voice embedding, replacing any voice of the same name.
    pub fn add_voice_file(&mut self, name: &str, path: &FsPath) -> Result<()> {
        let dev = self.model.device().clone();
        let model_ext = self.cfg.model_ext();
        let emb =
            loader::load_voice_emb(path, model_ext.as_deref(), self.model.speaker_proj(), &dev)?
                .to::<Q::T>()?;
        self.forget_primed(name);
        self.voices.insert(name.to_string(), Voice { emb, null_emb: None });
        Ok(())
    }

    /// Clone a voice from a mono audio prompt.
    ///
    /// `pcm` must be at [`Self::voice_prompt_sample_rate`] and last between
    /// `audio_prompt_min_duration` and `audio_prompt_max_duration` seconds:
    /// longer input is trimmed, shorter input is an error. The caller's slice is
    /// not modified — loudness normalization runs on an internal copy.
    pub fn add_voice_from_pcm(&mut self, name: &str, pcm: &[f32]) -> Result<()> {
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
        crate::utils::normalize_loudness(&mut pcm, sr as u32)?;

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
        self.voices.insert(name.to_string(), Voice { emb, null_emb });
        Ok(())
    }

    /// Register a voice from a conditioning embedding already in memory, laid
    /// out as `frames` rows of `dim`.
    ///
    /// This is the mimi encoder's output -- what [`Self::add_voice_from_pcm`]
    /// computes internally. Callers that compute or cache embeddings themselves
    /// (`ptts-pyo3` hands one over from numpy) use this.
    ///
    /// `null_emb`, when given, is the encoding of equal-length silence, which
    /// CFG needs on models where `cfg_null_audio_empty` is false.
    pub fn add_voice_from_embedding(
        &mut self,
        name: &str,
        emb: &[f32],
        frames: usize,
        dim: usize,
        null_emb: Option<&[f32]>,
    ) -> Result<()> {
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
        self.voices.insert(name.to_string(), Voice { emb, null_emb });
        Ok(())
    }

    /// Synthesize `text` and return the whole waveform.
    pub fn say(&self, text: &str) -> Result<Vec<f32>> {
        self.say_with(text, &SpeechOptions::default())
    }

    /// Synthesize `text` with per-request overrides.
    pub fn say_with(&self, text: &str, opts: &SpeechOptions) -> Result<Vec<f32>> {
        let mut pcm = Vec::new();
        for chunk in self.stream_with(text, opts)? {
            pcm.extend_from_slice(&chunk?);
        }
        Ok(pcm)
    }

    /// Start generating `text`, yielding PCM as the decoder produces it.
    pub fn stream(&self, text: &str) -> Result<SpeechStream> {
        self.stream_with(text, &SpeechOptions::default())
    }

    /// Prime a voice once and keep it, for callers that generate repeatedly.
    ///
    /// ```no_run
    /// # fn main() -> xn::Result<()> {
    /// # let tts: ptts::synth::Synth = todo!();
    /// let session = tts.session(&ptts::synth::SpeechOptions::default().voice("alba"), 1024)?;
    /// for line in ["First.", "Second.", "Third."] {
    ///     let pcm = session.say(line)?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// `max_seq_len` is the KV budget, allocated up front and held until the
    /// session is dropped. At 12.5 Hz a full [`MAX_TOKENS_PER_CHUNK`]-token
    /// chunk needs 796, so 1024 covers any
    /// single chunk; longer text is split into chunks of that size rather than
    /// needing more.
    pub fn session(&self, opts: &SpeechOptions, max_seq_len: usize) -> Result<SessionOf<Q>> {
        self.session_at(&resolve(&self.defaults, opts)?, max_seq_len)
    }

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
        })
    }

    /// Start generating `text` with per-request overrides.
    ///
    /// Generation runs on two background threads — one for the flow-LM, one for
    /// the Mimi decoder — so decoding overlaps the next backbone step. Dropping
    /// the returned [`SpeechStream`] stops both.
    pub fn stream_with(&self, text: &str, opts: &SpeechOptions) -> Result<SpeechStream> {
        let settings = resolve(&self.defaults, opts)?;
        let rng = Box::new(NormalRng::new(settings.temperature, settings.seed)?);
        self.stream_with_rng(text, opts, rng)
    }

    /// As [`Self::stream_with`], but with an explicit noise source.
    ///
    /// The solver's only source of randomness is this trait, so replaying a
    /// fixed sequence (see [`crate::flow_lm::ReplayRng`]) makes a generation
    /// reproducible across implementations — which is how this crate is
    /// compared against the reference one. `temperature` and `seed` are ignored.
    pub fn stream_with_rng(
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
        )?;
        // A one-shot call is a session sized to this text and dropped afterwards,
        // so there is one generation path rather than two.
        let seq_budget = chunks.iter().map(|c| c.seq_budget).max().unwrap_or(0);
        self.session_at(&settings, seq_budget)?.stream_chunks(chunks, rng)
    }

    /// Called whenever a voice is (re)registered, so a replaced embedding is never generated
    /// from the old conditioning.
    fn forget_primed(&self, name: &str) {
        let mut primed = self.primed.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        primed.retain(|(voice, _), _| voice != name);
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
        let voice = match voice {
            None if self.voices.is_empty() => None,
            None => {
                return Err(Error::not_found(format!(
                    "no voice selected; this model has {}",
                    self.voices.keys().cloned().collect::<Vec<_>>().join(", ")
                )));
            }
            Some(name) => match self.voices.get(name) {
                Some(v) => Some((name, v)),
                None => {
                    return Err(Error::UnknownVoice {
                        name: name.to_string(),
                        known: self.voices(),
                    });
                }
            },
        };

        let Some((name, voice)) = voice else {
            let state = self.model.init_flow_lm_state(1, seq_budget)?;
            let cfg_state = match cfg_coef {
                None => None,
                Some(coef) => Some((coef, self.model.init_flow_lm_state(1, seq_budget)?)),
            };
            return Ok((state, cfg_state));
        };

        let frames = voice.emb.dim(1usize)?;
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

    /// Run the voice prompt, and with guidance on the null branch, into states just large
    /// enough to hold them.
    fn prime(&self, voice: &Voice<Q>, cfg_on: bool, frames: usize) -> Result<Primed<Q>> {
        let mut state = self.model.init_flow_lm_state(1, frames)?;
        self.model.prompt_audio(&mut state, &voice.emb)?;

        let null_state = if !cfg_on {
            None
        } else {
            // Sized to what the null branch will consume, so the `emb`/`null_emb` length
            // invariant stays local to registration rather than load-bearing here.
            let null_frames = match voice.null_emb.as_ref() {
                Some(null_emb) if !self.cfg.cfg_null_audio_empty => null_emb.dim(1usize)?,
                _ => frames,
            };
            let mut null_state = self.model.init_flow_lm_state(1, null_frames)?;
            if !self.cfg.cfg_null_audio_empty {
                match voice.null_emb.as_ref() {
                    Some(null_emb) => self.model.prompt_audio(&mut null_state, null_emb)?,
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

impl<Q: BackendQ> std::fmt::Debug for SynthOf<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SynthOf")
            .field("device", &self.device_name())
            .field("sample_rate", &self.sample_rate())
            .field("voices", &self.voices())
            .finish()
    }
}

/// A fresh, independently-owned state at `seq_budget`, seeded with `prefix`'s filled positions.
fn grow<Q: BackendQ>(prefix: &TTSState<Q>, seq_budget: usize) -> Result<TTSState<Q>> {
    let transformer_state = prefix.flow_lm_state.transformer_state.with_seq_budget(seq_budget)?;
    Ok(TTSState { flow_lm_state: crate::flow_lm::FlowLMState { transformer_state } })
}

/// Split `text` into chunks and work out the budgets for each.
///
/// Free rather than a method because both [`SynthOf`] and [`SessionOf`] need
/// it, and it depends only on the tokenizer inside the model and the codec's
/// frame rate.
fn plan_chunks<Q: BackendQ>(
    model: &TTSModel<Q>,
    frame_rate: f64,
    text: &str,
    max_tokens_per_chunk: usize,
    normalize: Normalize,
) -> Result<Vec<ChunkPlan>> {
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
    // Normalization runs first, on the whole input: it rewrites the characters
    // the sentence splitter looks for, and `prepare_text_prompt` pads short
    // text with leading spaces that normalization would collapse away.
    let text = normalize.apply(text);
    let texts = split_into_best_sentences(tokenizer, &text, Some(max_tokens_per_chunk))?;
    let mut chunks = Vec::with_capacity(texts.len());
    for text in texts {
        let (prepared, frames_after_eos) = prepare_text_prompt(&text);
        let tokens = model.flow_lm.conditioner.tokenize(&prepared)?;
        let frame_budget = plan::frame_budget(tokens.len(), frame_rate);
        let seq_budget = plan::seq_budget(tokens.len(), frame_budget);
        chunks.push(ChunkPlan { tokens, frame_budget, frames_after_eos, seq_budget });
    }
    if chunks.is_empty() {
        return Err(Error::invalid_argument("nothing to synthesize: the text is empty"));
    }
    Ok(chunks)
}

/// A voice primed once, ready to generate repeatedly.
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
/// Built by [`SynthOf::session`]. Every generation clones the primed state
/// rather than re-running `prompt_audio` over the voice prompt. A one-shot
/// caller should just use [`SynthOf::say`].
///
/// The KV budget is fixed at construction.
pub struct SessionOf<Q: BackendQ> {
    model: Arc<TTSModel<Q>>,
    frame_rate: f64,
    temperature: f32,
    seed: u64,
    max_tokens_per_chunk: usize,
    base: TTSState<Q>,
    cfg_base: Option<(f32, TTSState<Q>)>,
    seq_budget: usize,
    /// How text is normalized, inherited from the [`SynthOf`] this session was
    /// built from.
    normalize: Normalize,
    /// Slots the voice prompt already occupies, so the budget check can use it
    /// instead of [`plan::PROMPT_SEQ_HEADROOM`]'s fixed reserve.
    prompt_len: usize,
    /// Set while a generation is running. See the note on this type.
    in_flight: Arc<std::sync::atomic::AtomicBool>,
}

impl<Q: BackendQ> SessionOf<Q> {
    /// The KV budget this session was primed with.
    pub fn seq_budget(&self) -> usize {
        self.seq_budget
    }

    pub fn sample_rate(&self) -> usize {
        self.model.sample_rate()
    }

    /// How text is normalized.
    ///
    /// [`Self::stream`] and [`Self::say`] apply it themselves. Callers that
    /// tokenize by hand for [`Self::stream_tokens`] should run their text
    /// through [`Normalize::apply`] first, before [`prepare_text_prompt`]:
    /// normalization collapses runs of whitespace, including the padding
    /// `prepare_text_prompt` adds to short text.
    pub fn normalization(&self) -> Normalize {
        self.normalize
    }

    /// Synthesize `text`, returning the whole waveform.
    pub fn say(&self, text: &str) -> Result<Vec<f32>> {
        let mut pcm = Vec::new();
        for chunk in self.stream(text)? {
            pcm.extend_from_slice(&chunk?);
        }
        Ok(pcm)
    }

    /// Synthesize `text`, yielding PCM as the decoder produces it.
    pub fn stream(&self, text: &str) -> Result<SpeechStream> {
        let rng = Box::new(NormalRng::new(self.temperature, self.seed)?);
        self.stream_with_rng(text, rng)
    }

    /// As [`Self::stream`], with an explicit seed for this request.
    pub fn stream_seeded(&self, text: &str, seed: u64) -> Result<SpeechStream> {
        let rng = Box::new(NormalRng::new(self.temperature, seed)?);
        self.stream_with_rng(text, rng)
    }

    /// As [`Self::say`], with an explicit seed for this request.
    pub fn say_seeded(&self, text: &str, seed: u64) -> Result<Vec<f32>> {
        let mut pcm = Vec::new();
        for chunk in self.stream_seeded(text, seed)? {
            pcm.extend_from_slice(&chunk?);
        }
        Ok(pcm)
    }

    /// As [`Self::stream`], with an explicit noise source.
    pub fn stream_with_rng(
        &self,
        text: &str,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream> {
        let chunks = plan_chunks(
            &self.model,
            self.frame_rate,
            text,
            self.max_tokens_per_chunk,
            self.normalize,
        )?;
        self.stream_chunks(chunks, rng)
    }

    /// Tokenize `text` as given, with none of the preparation [`Self::stream`]
    /// does first — no [`prepare_text_prompt`], no sentence splitting.
    ///
    /// Paired with [`Self::stream_tokens`] for callers that want one utterance
    /// per request and prepare the text themselves.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.model.flow_lm.conditioner.tokenize(text)?)
    }

    /// Synthesize from tokens produced elsewhere, as one chunk.
    ///
    /// `frames_after_eos` is the tail [`prepare_text_prompt`] would have
    /// chosen: 3 for a very short prompt, 1 otherwise.
    pub fn stream_tokens(
        &self,
        tokens: Vec<u32>,
        frames_after_eos: usize,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream> {
        if tokens.is_empty() {
            return Err(Error::invalid_argument("nothing to synthesize: no tokens"));
        }
        let frame_budget = plan::frame_budget(tokens.len(), self.frame_rate);
        let seq_budget = plan::seq_budget(tokens.len(), frame_budget);
        let chunk = ChunkPlan { tokens, frame_budget, frames_after_eos, seq_budget };
        self.stream_chunks(vec![chunk], rng)
    }

    /// Start the two worker threads for an already-planned set of chunks.
    ///
    /// The single place generation is driven from: [`SynthOf::stream_with_rng`]
    /// reaches it through an ephemeral session.
    fn stream_chunks(
        &self,
        chunks: Vec<ChunkPlan>,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream> {
        // `c.seq_budget` reserves PROMPT_SEQ_HEADROOM for a voice prompt whose
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

        let (pcm_tx, pcm_rx) = std::sync::mpsc::channel::<Result<Vec<f32>>>();
        let (latent_tx, latent_rx) = std::sync::mpsc::channel::<Frame<Q>>();

        // Decoder: latents in, PCM out. Reset between chunks so each chunk
        // starts from a clean codec state, matching the pre-refactor behavior.
        let decode_model = Arc::clone(&self.model);
        let decode_tx = pcm_tx.clone();
        let decode_handle = std::thread::spawn(move || {
            let mut state = mimi_init.clone();
            while let Ok(frame) = latent_rx.recv() {
                let first = match frame {
                    Frame::ChunkEnd => {
                        state = mimi_init.clone();
                        continue;
                    }
                    Frame::Latent(latent) => latent,
                };
                // Decode every latent the flow-LM has already queued in one
                // call: the decoder is exact for any number of frames, and one
                // call over several is much cheaper than one call per frame.
                // Only what is already waiting is taken, so no frame is ever
                // held back for a batch to fill — the first one included.
                let mut reset_after = false;
                let mut batch = vec![first];
                while let Ok(frame) = latent_rx.try_recv() {
                    match frame {
                        // Past a chunk boundary the codec state resets, so the
                        // batch has to stop here and resume on the next frame.
                        Frame::ChunkEnd => {
                            reset_after = true;
                            break;
                        }
                        Frame::Latent(latent) => batch.push(latent),
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
                    .map_err(Error::from);
                let failed = pcm.is_err();
                if decode_tx.send(pcm).is_err() || failed {
                    return;
                }
                if reset_after {
                    state = mimi_init.clone();
                }
            }
        });

        // Flow-LM: text in, latents out.
        let model = Arc::clone(&self.model);
        let backbone_handle = std::thread::spawn(move || {
            // Dropped when this thread ends, which is after its last write.
            let _in_flight = in_flight;
            let result = run_backbone(&model, chunks, base_state, cfg_base, rng, &latent_tx);
            if let Err(e) = result {
                let _ = pcm_tx.send(Err(e));
            }
        });

        Ok(SpeechStream {
            rx: Some(pcm_rx),
            sample_rate: self.sample_rate(),
            failed: false,
            workers: Some([backbone_handle, decode_handle]),
        })
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

/// What one text chunk will need.
struct ChunkPlan {
    tokens: Vec<u32>,
    frame_budget: usize,
    frames_after_eos: usize,
    seq_budget: usize,
}

/// Messages from the flow-LM thread to the decoder thread.
enum Frame<Q: BackendQ> {
    Latent(Tensor<Q::T, Q::B>),
    ChunkEnd,
}

/// The autoregressive loop, shared by every frontend and every chunk.
fn run_backbone<Q: BackendQ>(
    model: &TTSModel<Q>,
    chunks: Vec<ChunkPlan>,
    base_state: TTSState<Q>,
    cfg_base: Option<(f32, TTSState<Q>)>,
    mut rng: Box<dyn crate::flow_lm::Rng + Send>,
    latent_tx: &std::sync::mpsc::Sender<Frame<Q>>,
) -> Result<()> {
    for chunk in chunks.iter() {
        let mut state = base_state.clone();
        let mut cfg_state = cfg_base.clone();
        model.prompt_text(&mut state, &chunk.tokens)?;
        if let Some((_, null_state)) = cfg_state.as_mut() {
            model.prompt_text_null(null_state)?;
        }

        let mut prev: Option<Tensor<Q::T, Q::B>> = None;
        let mut eos = EosPolicy::new(chunk.frames_after_eos);

        for _ in 0..chunk.frame_budget {
            let input = match &prev {
                None => StepInput::Bos { batch: 1 },
                Some(t) => StepInput::Latent(t),
            };
            let (next, is_eos) = match cfg_state.as_mut() {
                Some((coef, null_state)) => {
                    model.generate_step_cfg(&mut state, null_state, *coef, input, &mut rng)?
                }
                None => model.generate_step(&mut state, input, &mut rng)?,
            };
            // A closed channel means the consumer went away; stop quietly and
            // let the decoder thread report any error of its own.
            if latent_tx.send(Frame::Latent(next.clone())).is_err() {
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

/// PCM chunks from a running generation.
///
/// Each item is mono `f32` samples at [`Self::sample_rate`] — one Mimi frame's
/// worth, or several when the decoder finds more than one frame already queued
/// and decodes them together. The iterator ends when generation finishes; an
/// `Err` item is terminal.
pub struct SpeechStream {
    /// `Option` so [`Drop`] can release it *before* joining: the channel is
    /// unbounded, so a worker only notices it should stop once the receiver is
    /// gone, and fields drop after `Drop::drop` has run.
    rx: Option<std::sync::mpsc::Receiver<Result<Vec<f32>>>>,
    sample_rate: usize,
    failed: bool,
    /// The flow-LM and decoder threads, joined once the channel closes so that
    /// a panic in either surfaces as an error rather than as truncated audio.
    workers: Option<[std::thread::JoinHandle<()>; 2]>,
}

impl SpeechStream {
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
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
        if let Some(workers) = self.workers.take() {
            for worker in workers {
                let _ = worker.join();
            }
        }
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
                self.workers.take();
                Some(Err(e))
            }
            Ok(ok) => Some(ok),
            // Every sender dropped: generation finished, or a worker died.
            Err(_) => {
                self.failed = true;
                let workers = self.workers.take()?;
                let died = workers.into_iter().any(|h| h.join().is_err());
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

/// Configures and loads a [`SynthOf`].
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
    voices: Vec<(String, PathBuf)>,
    normalize: Normalize,
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
    /// [`Normalize::For`] with a language, or [`Normalize::Off`] to hand text
    /// to the tokenizer as written.
    ///
    /// ```no_run
    /// # fn main() -> xn::Result<()> {
    /// use ptts::preprocess::{Lang, Normalize};
    /// use ptts::synth::SynthBuilder;
    /// use ptts::tts_model::TTSConfig;
    ///
    /// let tts = SynthBuilder::new(
    ///     TTSConfig::v202601(0.3),
    ///     "model/model.safetensors",
    ///     Normalize::For(Lang::De),
    /// )
    /// .tokenizer_file("model/tokenizer.model")
    /// .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: TTSConfig, weights: impl Into<PathBuf>, normalize: Normalize) -> Self {
        Self {
            config,
            weights: weights.into(),
            normalize,
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
            voices: vec![],
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

    /// Override the config's EOS log-probability threshold. Lower values let
    /// the model run longer before it decides an utterance is finished.
    pub fn eos_threshold(mut self, eos_threshold: f32) -> Self {
        self.eos_threshold = Some(eos_threshold);
        self
    }

    /// Default voice for requests that do not name one. Defaults to the first
    /// registered voice by name.
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

    /// Register a precomputed voice embedding at load time. A voice that fails
    /// to load fails the whole load; a frontend registering a checkpoint's own
    /// voices, where one bad embedding should not make the model unusable, can
    /// instead loop over [`SynthOf::add_voice_file`] afterwards and warn.
    pub fn add_voice(mut self, name: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        self.voices.push((name.into(), path.into()));
        self
    }

    /// Load the checkpoint, choosing the device and weight format from
    /// [`Self::device`] and [`Self::quant`].
    pub fn build(self) -> Result<Synth> {
        let device = self.device.resolve();
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
            ($variant:ident, $q:ty) => {{
                let synth = self.load::<$q>(xn::CPU)?;
                Ok(Synth(SynthV::$variant(synth)))
            }};
        }
        match self.quant {
            Quant::F32 => cpu!(Cpu, xn::Unquantized<f32, xn::CpuDevice>),
            Quant::Q80 => cpu!(Q80, xn::quantized::Q80F32),
            Quant::Q81 => cpu!(Q81, xn::quantized::Q81F32),
            Quant::Q8k => cpu!(Q8k, xn::quantized::Q8kF32),
            Quant::Q6k => cpu!(Q6k, xn::quantized::Q6kF32),
            Quant::Q50 => cpu!(Q50, xn::quantized::Q50F32),
            Quant::Q51 => cpu!(Q51, xn::quantized::Q51F32),
            Quant::Q5k => cpu!(Q5k, xn::quantized::Q5kF32),
            Quant::Q40 => cpu!(Q40, xn::quantized::Q40F32),
            Quant::Q41 => cpu!(Q41, xn::quantized::Q41F32),
            Quant::Q4k => cpu!(Q4k, xn::quantized::Q4kF32),
        }
    }

    #[cfg(feature = "cuda")]
    fn build_cuda(self) -> Result<Synth> {
        let dev = xn::cuda_backend::Device::new(0)?;
        // Event tracking costs a few percent and this workload never queries events.
        unsafe { dev.disable_event_tracking() };
        let synth = self.load::<xn::Unquantized<half::bf16, _>>(dev)?;
        Ok(Synth(SynthV::Cuda(synth)))
    }

    #[cfg(feature = "vulkan")]
    fn build_vulkan(self) -> Result<Synth> {
        let dev = xn::vulkan_backend::Device::new(0)?;
        let synth = self.load::<xn::Unquantized<f32, _>>(dev)?;
        Ok(Synth(SynthV::Vulkan(synth)))
    }

    #[cfg(feature = "metal")]
    fn build_metal(self) -> Result<Synth> {
        let dev = xn::metal_backend::Device::new(0)?;
        let synth = self.load::<xn::Unquantized<half::bf16, _>>(dev)?;
        Ok(Synth(SynthV::Metal(synth)))
    }

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

    /// Load the weights and register the voices.
    pub fn load<Q: BackendQ>(mut self, device: Q::B) -> Result<SynthOf<Q>> {
        if !self.weights.is_file() {
            return Err(Error::not_found(format!(
                "weights file not found: {}",
                self.weights.display()
            )));
        }
        let config = self.config.clone();
        let tokenizer = self.take_tokenizer()?;

        let vb = loader::load_weights::<Q>(&self.weights, &device)?;
        let model = TTSModel::<Q>::load(&vb, tokenizer, &config)?;
        let model = match self.eos_threshold {
            Some(threshold) => model.with_eos_threshold(threshold),
            None => model,
        };
        // A dedicated speaker codec ships its encoder under its own prefix, so
        // probe there rather than assuming `mimi.encoder.*`.
        let probe = format!("{}.encoder.model.0.conv.weight", config.speaker_mimi_prefix());
        let mimi_enc =
            if vb.contains(&probe) { Some(MimiEnc::<Q>::load(&vb, &config)?) } else { None };
        vb.check_all_used_with_ignore(loader::is_unused_by_tts_model)?;

        let mut synth = SynthOf {
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
            },
            normalize: self.normalize,
        };

        for (name, path) in self.voices.iter() {
            synth.add_voice_file(name, path)?;
        }

        // Default to the first voice by name when the caller named none, so a
        // bare `say` works out of the box.
        if synth.defaults.voice.is_none() {
            synth.defaults.voice = synth.voices.keys().next().cloned();
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

/// Every weight format and device this build supports.
///
/// [`Synth`] exists so that a caller who picks a format from a command-line
/// flag or a config file does not have to be generic over `Q`. The runtime
/// dispatch happens once per method call and costs nothing next to a
/// transformer step.
enum SynthV {
    Cpu(SynthOf<xn::Unquantized<f32, xn::CpuDevice>>),
    Q80(SynthOf<xn::quantized::Q80F32>),
    Q81(SynthOf<xn::quantized::Q81F32>),
    Q8k(SynthOf<xn::quantized::Q8kF32>),
    Q6k(SynthOf<xn::quantized::Q6kF32>),
    Q50(SynthOf<xn::quantized::Q50F32>),
    Q51(SynthOf<xn::quantized::Q51F32>),
    Q5k(SynthOf<xn::quantized::Q5kF32>),
    Q40(SynthOf<xn::quantized::Q40F32>),
    Q41(SynthOf<xn::quantized::Q41F32>),
    Q4k(SynthOf<xn::quantized::Q4kF32>),
    #[cfg(feature = "cuda")]
    Cuda(SynthOf<xn::Unquantized<half::bf16, xn::cuda_backend::Device>>),
    #[cfg(feature = "vulkan")]
    Vulkan(SynthOf<xn::Unquantized<f32, xn::vulkan_backend::Device>>),
    #[cfg(feature = "metal")]
    Metal(SynthOf<xn::Unquantized<half::bf16, xn::metal_backend::Device>>),
}

/// Forward a method to whichever [`SynthOf`] is inside, binding it to `$s`.
macro_rules! dispatch {
    ($synth:expr, |$s:ident| $body:expr) => {
        match $synth {
            SynthV::Cpu($s) => $body,
            SynthV::Q80($s) => $body,
            SynthV::Q81($s) => $body,
            SynthV::Q8k($s) => $body,
            SynthV::Q6k($s) => $body,
            SynthV::Q50($s) => $body,
            SynthV::Q51($s) => $body,
            SynthV::Q5k($s) => $body,
            SynthV::Q40($s) => $body,
            SynthV::Q41($s) => $body,
            SynthV::Q4k($s) => $body,
            #[cfg(feature = "cuda")]
            SynthV::Cuda($s) => $body,
            #[cfg(feature = "vulkan")]
            SynthV::Vulkan($s) => $body,
            #[cfg(feature = "metal")]
            SynthV::Metal($s) => $body,
        }
    };
}

/// The erased counterpart of [`SessionOf`], for callers that chose their weight
/// format at runtime.
enum SessionV {
    Cpu(SessionOf<xn::Unquantized<f32, xn::CpuDevice>>),
    Q80(SessionOf<xn::quantized::Q80F32>),
    Q81(SessionOf<xn::quantized::Q81F32>),
    Q8k(SessionOf<xn::quantized::Q8kF32>),
    Q6k(SessionOf<xn::quantized::Q6kF32>),
    Q50(SessionOf<xn::quantized::Q50F32>),
    Q51(SessionOf<xn::quantized::Q51F32>),
    Q5k(SessionOf<xn::quantized::Q5kF32>),
    Q40(SessionOf<xn::quantized::Q40F32>),
    Q41(SessionOf<xn::quantized::Q41F32>),
    Q4k(SessionOf<xn::quantized::Q4kF32>),
    #[cfg(feature = "cuda")]
    Cuda(SessionOf<xn::Unquantized<half::bf16, xn::cuda_backend::Device>>),
    #[cfg(feature = "vulkan")]
    Vulkan(SessionOf<xn::Unquantized<f32, xn::vulkan_backend::Device>>),
    #[cfg(feature = "metal")]
    Metal(SessionOf<xn::Unquantized<half::bf16, xn::metal_backend::Device>>),
}

macro_rules! dispatch_session {
    ($session:expr, |$s:ident| $body:expr) => {
        match $session {
            SessionV::Cpu($s) => $body,
            SessionV::Q80($s) => $body,
            SessionV::Q81($s) => $body,
            SessionV::Q8k($s) => $body,
            SessionV::Q6k($s) => $body,
            SessionV::Q50($s) => $body,
            SessionV::Q51($s) => $body,
            SessionV::Q5k($s) => $body,
            SessionV::Q40($s) => $body,
            SessionV::Q41($s) => $body,
            SessionV::Q4k($s) => $body,
            #[cfg(feature = "cuda")]
            SessionV::Cuda($s) => $body,
            #[cfg(feature = "vulkan")]
            SessionV::Vulkan($s) => $body,
            #[cfg(feature = "metal")]
            SessionV::Metal($s) => $body,
        }
    };
}

/// A voice primed once, ready to generate repeatedly.
///
/// Runs one generation at a time: starting a second while the first is still
/// running is an error. See [`SessionOf`] for why, and for the concurrency
/// story.
pub struct Session(SessionV);

impl Session {
    /// The KV budget this session was primed with.
    pub fn seq_budget(&self) -> usize {
        dispatch_session!(&self.0, |s| s.seq_budget())
    }

    pub fn sample_rate(&self) -> usize {
        dispatch_session!(&self.0, |s| s.sample_rate())
    }

    /// How text is normalized — see [`SessionOf::normalization`].
    pub fn normalization(&self) -> Normalize {
        dispatch_session!(&self.0, |s| s.normalization())
    }

    /// Synthesize `text`, returning the whole waveform.
    pub fn say(&self, text: &str) -> Result<Vec<f32>> {
        dispatch_session!(&self.0, |s| s.say(text))
    }

    /// Synthesize `text`, yielding PCM as the decoder produces it.
    pub fn stream(&self, text: &str) -> Result<SpeechStream> {
        dispatch_session!(&self.0, |s| s.stream(text))
    }

    /// As [`Self::say`], with an explicit seed for this request. Every call on a
    /// session otherwise draws the same noise, since the seed is the session's.
    pub fn say_seeded(&self, text: &str, seed: u64) -> Result<Vec<f32>> {
        dispatch_session!(&self.0, |s| s.say_seeded(text, seed))
    }

    /// As [`Self::stream`], with an explicit seed for this request.
    pub fn stream_seeded(&self, text: &str, seed: u64) -> Result<SpeechStream> {
        dispatch_session!(&self.0, |s| s.stream_seeded(text, seed))
    }

    /// Tokenize `text` as given — see [`SessionOf::tokenize`].
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        dispatch_session!(&self.0, |s| s.tokenize(text))
    }

    /// Synthesize from tokens produced elsewhere, as one chunk.
    pub fn stream_tokens(
        &self,
        tokens: Vec<u32>,
        frames_after_eos: usize,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream> {
        dispatch_session!(&self.0, |s| s.stream_tokens(tokens, frames_after_eos, rng))
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

/// A loaded Pocket TTS model, ready to synthesize speech.
///
/// See the [module docs](self) for the short version. The weight format and
/// device are chosen at load time by [`SynthBuilder`] and erased here.
pub struct Synth(SynthV);

impl Synth {
    /// A builder over a checkpoint's config and weights file.
    ///
    /// Finding those -- and the tokenizer and voices beside them -- is the
    /// caller's job: see [`SynthBuilder`].
    pub fn builder(
        config: TTSConfig,
        weights: impl Into<PathBuf>,
        normalize: Normalize,
    ) -> SynthBuilder {
        SynthBuilder::new(config, weights, normalize)
    }

    /// Prime a voice once and keep it, for callers that generate repeatedly.
    ///
    /// See [`SynthOf::session`]. `max_seq_len` is the KV budget allocated up
    /// front; text needing more is rejected rather than silently re-primed.
    pub fn session(&self, opts: &SpeechOptions, max_seq_len: usize) -> Result<Session> {
        Ok(Session(match &self.0 {
            SynthV::Cpu(s) => SessionV::Cpu(s.session(opts, max_seq_len)?),
            SynthV::Q80(s) => SessionV::Q80(s.session(opts, max_seq_len)?),
            SynthV::Q81(s) => SessionV::Q81(s.session(opts, max_seq_len)?),
            SynthV::Q8k(s) => SessionV::Q8k(s.session(opts, max_seq_len)?),
            SynthV::Q6k(s) => SessionV::Q6k(s.session(opts, max_seq_len)?),
            SynthV::Q50(s) => SessionV::Q50(s.session(opts, max_seq_len)?),
            SynthV::Q51(s) => SessionV::Q51(s.session(opts, max_seq_len)?),
            SynthV::Q5k(s) => SessionV::Q5k(s.session(opts, max_seq_len)?),
            SynthV::Q40(s) => SessionV::Q40(s.session(opts, max_seq_len)?),
            SynthV::Q41(s) => SessionV::Q41(s.session(opts, max_seq_len)?),
            SynthV::Q4k(s) => SessionV::Q4k(s.session(opts, max_seq_len)?),
            #[cfg(feature = "cuda")]
            SynthV::Cuda(s) => SessionV::Cuda(s.session(opts, max_seq_len)?),
            #[cfg(feature = "vulkan")]
            SynthV::Vulkan(s) => SessionV::Vulkan(s.session(opts, max_seq_len)?),
            #[cfg(feature = "metal")]
            SynthV::Metal(s) => SessionV::Metal(s.session(opts, max_seq_len)?),
        }))
    }

    /// Synthesize `text` and return the whole waveform as mono `f32` at
    /// [`Self::sample_rate`].
    pub fn say(&self, text: &str) -> Result<Vec<f32>> {
        dispatch!(&self.0, |s| s.say(text))
    }

    /// Synthesize `text` with per-request overrides.
    pub fn say_with(&self, text: &str, opts: &SpeechOptions) -> Result<Vec<f32>> {
        dispatch!(&self.0, |s| s.say_with(text, opts))
    }

    /// Start generating `text`, yielding PCM as the decoder produces it.
    pub fn stream(&self, text: &str) -> Result<SpeechStream> {
        dispatch!(&self.0, |s| s.stream(text))
    }

    /// Start generating `text` with per-request overrides.
    pub fn stream_with(&self, text: &str, opts: &SpeechOptions) -> Result<SpeechStream> {
        dispatch!(&self.0, |s| s.stream_with(text, opts))
    }

    /// As [`Self::stream_with`], but with an explicit noise source — see
    /// [`SynthOf::stream_with_rng`].
    pub fn stream_with_rng(
        &self,
        text: &str,
        opts: &SpeechOptions,
        rng: Box<dyn crate::flow_lm::Rng + Send>,
    ) -> Result<SpeechStream> {
        dispatch!(&self.0, |s| s.stream_with_rng(text, opts, rng))
    }

    pub fn sample_rate(&self) -> usize {
        dispatch!(&self.0, |s| s.sample_rate())
    }

    pub fn config(&self) -> &TTSConfig {
        dispatch!(&self.0, |s| s.config())
    }

    /// How requests are normalized — see [`SynthOf::normalization`].
    pub fn normalization(&self) -> Normalize {
        dispatch!(&self.0, |s| s.normalization())
    }

    /// Name of the device the model is running on, e.g. `"cpu"` or `"cuda:0"`.
    pub fn device_name(&self) -> String {
        dispatch!(&self.0, |s| s.device_name())
    }

    /// Registered voice names, sorted.
    pub fn voices(&self) -> Vec<String> {
        dispatch!(&self.0, |s| s.voices())
    }

    /// True if this checkpoint carries a speaker encoder, which
    /// [`Self::add_voice_from_pcm`] requires.
    pub fn supports_voice_cloning(&self) -> bool {
        dispatch!(&self.0, |s| s.supports_voice_cloning())
    }

    /// The sample rate [`Self::add_voice_from_pcm`] expects.
    pub fn voice_prompt_sample_rate(&self) -> usize {
        dispatch!(&self.0, |s| s.voice_prompt_sample_rate())
    }

    /// Register a precomputed voice embedding, replacing any voice of the same name.
    pub fn add_voice_file(&mut self, name: &str, path: &FsPath) -> Result<()> {
        dispatch!(&mut self.0, |s| s.add_voice_file(name, path))
    }

    /// Clone a voice from a mono audio prompt at
    /// [`Self::voice_prompt_sample_rate`].
    pub fn add_voice_from_pcm(&mut self, name: &str, pcm: &[f32]) -> Result<()> {
        dispatch!(&mut self.0, |s| s.add_voice_from_pcm(name, pcm))
    }

    /// Register a voice from a conditioning embedding already in memory -- see
    /// [`SynthOf::add_voice_from_embedding`].
    pub fn add_voice_from_embedding(
        &mut self,
        name: &str,
        emb: &[f32],
        frames: usize,
        dim: usize,
        null_emb: Option<&[f32]>,
    ) -> Result<()> {
        dispatch!(&mut self.0, |s| s.add_voice_from_embedding(name, emb, frames, dim, null_emb))
    }

    /// The weight format actually loaded. GPU backends are always unquantized.
    pub fn quant(&self) -> Quant {
        match &self.0 {
            SynthV::Cpu(_) => Quant::F32,
            SynthV::Q80(_) => Quant::Q80,
            SynthV::Q81(_) => Quant::Q81,
            SynthV::Q8k(_) => Quant::Q8k,
            SynthV::Q6k(_) => Quant::Q6k,
            SynthV::Q50(_) => Quant::Q50,
            SynthV::Q51(_) => Quant::Q51,
            SynthV::Q5k(_) => Quant::Q5k,
            SynthV::Q40(_) => Quant::Q40,
            SynthV::Q41(_) => Quant::Q41,
            SynthV::Q4k(_) => Quant::Q4k,
            #[cfg(feature = "cuda")]
            SynthV::Cuda(_) => Quant::F32,
            #[cfg(feature = "vulkan")]
            SynthV::Vulkan(_) => Quant::F32,
            #[cfg(feature = "metal")]
            SynthV::Metal(_) => Quant::F32,
        }
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

    fn defaults() -> Defaults {
        Defaults {
            voice: Some("alba".into()),
            temperature: 0.5,
            seed: 42,
            cfg_coef: None,
            max_tokens_per_chunk: 50,
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
        for norm in [Normalize::For(Lang::En), Normalize::For(Lang::De), Normalize::Off] {
            let b = SynthBuilder::new(TTSConfig::v202601(0.5), "model.safetensors", norm);
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
