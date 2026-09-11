//! Speech synthesis behind one call.
//!
//! [`SynthOf`] wraps everything between "I have some text" and "I have PCM":
//! loading a checkpoint, registering voices, splitting text into sentence
//! chunks, priming the transformer state, running the flow-matching solver, and
//! streaming the latents through the Mimi decoder on a second thread.
//!
//! ```no_run
//! # fn main() -> xn::Result<()> {
//! use ptts::synth::SynthBuilder;
//! use ptts::tts_model::TTSConfig;
//!
//! let tts = SynthBuilder::new(TTSConfig::v202601(0.7), "model/model.safetensors")
//!     .tokenizer_file("model/tokenizer.model")
//!     .add_voice("alba", "model/voices/alba.safetensors")
//!     .load::<xn::Unquantized<f32, xn::CpuDevice>>(xn::CPU)?;
//! let pcm = tts.say("Hello world")?;
//! ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate() as u32)?;
//! # Ok(())
//! # }
//! ```
//!
//! Audio arrives incrementally from [`SynthOf::stream`], which is the primitive
//! [`SynthOf::say`] is built on:
//!
//! ```no_run
//! # fn main() -> xn::Result<()> {
//! # let cfg = ptts::tts_model::TTSConfig::v202601(0.7);
//! # let tts = ptts::synth::SynthBuilder::new(cfg, "model/model.safetensors")
//! #     .tokenizer_file("model/tokenizer.model")
//! #     .load::<xn::Unquantized<f32, xn::CpuDevice>>(xn::CPU)?;
//! for chunk in tts.stream("Hello world")? {
//!     let pcm: Vec<f32> = chunk?;
//!     // hand `pcm` to an audio sink
//! }
//! # Ok(())
//! # }
//! ```
//!
//! The weight format is a type parameter, so a caller that picks one from a
//! command-line flag has to be generic over `Q`. A following change adds a
//! `Synth` that erases it.
//!
//! `SynthOf` is a composition of the [`crate::tts_model::TTSModel`] primitives,
//! not a replacement for them: callers that need to drive generation themselves
//! -- a browser build stepping from an event loop, a server interleaving
//! requests -- should keep using those directly.
use crate::flow_lm::NormalRng;
use crate::loader;
use crate::plan::{self, EosPolicy};
use crate::tts_model::{
    MAX_TOKENS_PER_CHUNK, MimiEnc, TTSConfig, TTSModel, TTSState, prepare_text_prompt,
    split_into_best_sentences,
};
use std::collections::BTreeMap;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use xn::{Backend, BackendQ, Result, Tensor};

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
            other => {
                xn::bail!("unknown device '{other}'; expected auto, cpu, cuda, vulkan or metal")
            }
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
            other => xn::bail!(
                "unsupported quantization '{other}'; expected one of \
                 f32, q8_0, q8_1, q8k, q6k, q5_0, q5_1, q5k, q4_0, q4_1, q4k"
            ),
        }
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
}

impl<Q: BackendQ> SynthOf<Q> {
    pub fn sample_rate(&self) -> usize {
        self.model.sample_rate()
    }

    pub fn config(&self) -> &TTSConfig {
        &self.cfg
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
        let emb = loader::load_voice_emb(path, model_ext.as_deref(), &dev)?.to::<Q::T>()?;
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
            None => xn::bail!("this checkpoint has no speaker encoder, so it cannot clone voices"),
        };
        let sr = self.voice_prompt_sample_rate();
        let min_len = (sr as f32 * self.cfg.audio_prompt_min_duration).round() as usize;
        let max_len = (sr as f32 * self.cfg.audio_prompt_max_duration).round() as usize;
        if pcm.len() < min_len {
            xn::bail!(
                "voice prompt is too short: got {} samples ({:.2}s at {sr}Hz), need at least \
                 {min_len} ({:.2}s)",
                pcm.len(),
                pcm.len() as f32 / sr as f32,
                self.cfg.audio_prompt_min_duration
            );
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

    /// Start generating `text` with per-request overrides.
    ///
    /// Generation runs on two background threads — one for the flow-LM, one for
    /// the Mimi decoder — so decoding overlaps the next backbone step. Dropping
    /// the returned [`SpeechStream`] stops both.
    pub fn stream_with(&self, text: &str, opts: &SpeechOptions) -> Result<SpeechStream> {
        let temperature = opts.temperature.unwrap_or(self.defaults.temperature);
        let seed = opts.seed.unwrap_or(self.defaults.seed);
        let rng = Box::new(NormalRng::new(temperature, seed)?);
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
        let max_tokens_per_chunk =
            opts.max_tokens_per_chunk.unwrap_or(self.defaults.max_tokens_per_chunk);
        let cfg_coef = match opts.cfg_coef.or(self.defaults.cfg_coef) {
            Some(coef) if coef != 1.0 => Some(coef),
            _ => None,
        };

        let chunks = self.plan_chunks(text, max_tokens_per_chunk)?;
        let seq_budget = chunks.iter().map(|c| c.seq_budget).max().unwrap_or(0);

        let voice_name = opts.voice.as_ref().or(self.defaults.voice.as_ref());
        let (base_state, cfg_base) = self.primed_state(voice_name, seq_budget, cfg_coef)?;

        let mimi_init = self.model.init_mimi_state(1)?;
        let ldim = self.model.flow_lm.ldim;

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
                            let _ = decode_tx.send(Err(e));
                            return;
                        }
                    },
                };
                let pcm = decode_model
                    .decode_latent(&latent, &mut state)
                    .and_then(|audio| audio.narrow(0, ..1)?.contiguous()?.to_vec());
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
            let result = run_backbone(&model, chunks, base_state, cfg_base, rng, ldim, &latent_tx);
            if let Err(e) = result {
                let _ = pcm_tx.send(Err(e));
            }
        });

        Ok(SpeechStream {
            rx: pcm_rx,
            sample_rate: self.sample_rate(),
            failed: false,
            workers: Some([backbone_handle, decode_handle]),
        })
    }

    /// Split `text` into chunks and work out the budgets for each.
    fn plan_chunks(&self, text: &str, max_tokens_per_chunk: usize) -> Result<Vec<ChunkPlan>> {
        let tokenizer = match self.model.flow_lm.conditioner.tokenizer.as_ref() {
            Some(tokenizer) => tokenizer.as_ref(),
            None => xn::bail!(
                "this model was loaded without a tokenizer; pass one to \
                 SynthBuilder::tokenizer, or use the lower-level TTSModel API with \
                 pre-tokenized input"
            ),
        };
        let texts = split_into_best_sentences(tokenizer, text, Some(max_tokens_per_chunk))?;
        let frame_rate = self.cfg.mimi.frame_rate;
        let mut chunks = Vec::with_capacity(texts.len());
        for text in texts {
            let (prepared, frames_after_eos) = prepare_text_prompt(&text);
            let tokens = self.model.flow_lm.conditioner.tokenize(&prepared)?;
            let frame_budget = plan::frame_budget(tokens.len(), frame_rate);
            let seq_budget = plan::seq_budget(tokens.len(), frame_budget);
            chunks.push(ChunkPlan { tokens, frame_budget, frames_after_eos, seq_budget });
        }
        if chunks.is_empty() {
            xn::bail!("nothing to synthesize: the text is empty");
        }
        Ok(chunks)
    }

    /// Build the state every chunk starts from: allocated, then conditioned on
    /// the voice. Cloning it per chunk is much cheaper than re-priming.
    #[allow(clippy::type_complexity)]
    fn primed_state(
        &self,
        voice: Option<&String>,
        seq_budget: usize,
        cfg_coef: Option<f32>,
    ) -> Result<(TTSState<Q>, Option<(f32, TTSState<Q>)>)> {
        let voice = match voice {
            None if self.voices.is_empty() => None,
            None => xn::bail!(
                "no voice selected; this model has {}",
                self.voices.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
            Some(name) => match self.voices.get(name) {
                Some(voice) => Some(voice),
                None => xn::bail!(
                    "unknown voice '{name}'; available voices are {}",
                    self.voices.keys().cloned().collect::<Vec<_>>().join(", ")
                ),
            },
        };

        let mut state = self.model.init_flow_lm_state(1, seq_budget)?;
        if let Some(voice) = voice {
            self.model.prompt_audio(&mut state, &voice.emb)?;
        }

        let cfg_state = match cfg_coef {
            None => None,
            Some(coef) => {
                let mut null_state = self.model.init_flow_lm_state(1, seq_budget)?;
                if !self.cfg.cfg_null_audio_empty
                    && let Some(voice) = voice
                {
                    match voice.null_emb.as_ref() {
                        Some(null_emb) => self.model.prompt_audio(&mut null_state, null_emb)?,
                        None => xn::bail!(
                            "this model conditions its CFG null branch on silence \
                             (cfg_null_audio_empty=false), which needs the voice's source audio. \
                             Register the voice with add_voice_from_pcm instead of a precomputed \
                             embedding, or disable CFG."
                        ),
                    }
                }
                Some((coef, null_state))
            }
        };
        Ok((state, cfg_state))
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
    ldim: usize,
    latent_tx: &std::sync::mpsc::Sender<Frame<Q>>,
) -> Result<()> {
    let device = model.device().clone();
    for chunk in chunks.iter() {
        let mut state = base_state.clone();
        let mut cfg_state = cfg_base.clone();
        model.prompt_text(&mut state, &chunk.tokens)?;
        if let Some((_, null_state)) = cfg_state.as_mut() {
            model.prompt_text_null(null_state)?;
        }

        // A NaN latent marks the start of the sequence.
        let nan = vec![f32::NAN; ldim];
        let mut prev: Tensor<Q::T, Q::B> =
            Tensor::from_vec(nan, (1, 1, ldim), &device)?.to::<Q::T>()?;
        let mut eos = EosPolicy::new(chunk.frames_after_eos);

        for _ in 0..chunk.frame_budget {
            let (next, is_eos) = match cfg_state.as_mut() {
                Some((coef, null_state)) => {
                    model.generate_step_cfg(&mut state, null_state, *coef, &prev, &mut rng)?
                }
                None => model.generate_step(&mut state, &prev, &mut rng)?,
            };
            // A closed channel means the consumer went away; stop quietly and
            // let the decoder thread report any error of its own.
            if latent_tx.send(Frame::Latent(next.clone())).is_err() {
                return Ok(());
            }
            if eos.should_stop(is_eos) {
                break;
            }
            prev = next;
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
    rx: std::sync::mpsc::Receiver<Result<Vec<f32>>>,
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

impl Iterator for SpeechStream {
    type Item = Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        match self.rx.recv() {
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
                    Some(Err(xn::Error::msg(
                        "a generation worker panicked; the audio is incomplete",
                    )))
                } else {
                    None
                }
            }
        }
    }
}

/// Configures and loads a [`SynthOf`].
///
/// The config and the weights path are required, and neither has a default:
/// which files a checkpoint ships, what they are called and where its voices
/// live all change from one release to the next, so locating them belongs to
/// the frontend. This reads the files it is handed.
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
}

impl SynthBuilder {
    /// A checkpoint's config and its weights file, GGUF or safetensors.
    pub fn new(config: TTSConfig, weights: impl Into<PathBuf>) -> Self {
        Self {
            config,
            weights: weights.into(),
            tokenizer_file: None,
            device: DeviceKind::Auto,
            quant: Quant::F32,
            tokenizer: None,
            temperature: 0.7,
            seed: 4242424242424242,
            cfg_coef: None,
            eos_threshold: None,
            voice: None,
            max_tokens_per_chunk: MAX_TOKENS_PER_CHUNK,
            voices: vec![],
        }
    }

    /// The tokenizer file the checkpoint ships, read by [`crate::tok::Tok`],
    /// which picks the family from the extension. Ignored when
    /// [`Self::tokenizer`] supplies one directly.
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
    /// tokenizer file, or when neither the `sp` nor the `hf` feature is enabled.
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

    /// Load the weights and register the voices.
    pub fn load<Q: BackendQ>(mut self, device: Q::B) -> Result<SynthOf<Q>> {
        if !self.weights.is_file() {
            xn::bail!("weights file not found: {}", self.weights.display())
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
            defaults: Defaults {
                voice: self.voice.clone(),
                temperature: self.temperature,
                seed: self.seed,
                cfg_coef: self.cfg_coef,
                max_tokens_per_chunk: self.max_tokens_per_chunk,
            },
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
            xn::bail!(
                "default voice '{name}' was not found; available voices are {}",
                synth.voices.keys().cloned().collect::<Vec<_>>().join(", ")
            );
        }
        Ok(synth)
    }

    /// A caller-supplied tokenizer wins — `ptts-wasm` tokenizes in JavaScript
    /// and has no tokenizer file at all. Otherwise load the one the checkpoint
    /// shipped, if a tokenizer backend is compiled in.
    fn take_tokenizer(&mut self) -> Result<Box<dyn crate::Tokenizer + Send + Sync>> {
        if let Some(tokenizer) = self.tokenizer.take() {
            return Ok(tokenizer);
        }
        #[cfg(any(feature = "sp", feature = "hf"))]
        if let Some(path) = self.tokenizer_file.as_deref() {
            return Ok(Box::new(crate::tok::Tok::open(path)?));
        }
        xn::bail!(
            "no tokenizer available: none was passed to SynthBuilder::tokenizer, the checkpoint \
             shipped none, and neither the `sp` nor the `hf` feature of `ptts` is enabled."
        )
    }
}
