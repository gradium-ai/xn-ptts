use anyhow::{Context as _, Result};
use ptts::tts_model::{TTSConfig, TTSModel, TTSState};
use std::collections::HashMap;
use std::sync::Arc;
use xn::nn::VB;
use xn::{BackendQ, Tensor};

pub const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

pub const DEFAULT_REPO_ID: &str = "kyutai/pocket-tts";
pub const DEFAULT_MODEL_FILE: &str = "tts_b6369a24.safetensors";

pub struct StdRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl StdRng {
    pub fn new(temperature: f32, seed: u64) -> Self {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, temperature.sqrt()).unwrap();
        let inner = rand::rngs::StdRng::seed_from_u64(seed);
        Self { inner, distr }
    }
}

impl ptts::flow_lm::Rng for StdRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

pub enum Tok {
    Sp(std::sync::Arc<sentencepiece::SentencePieceProcessor>),
    Hf(Box<tokenizers::Tokenizer>),
}

impl From<sentencepiece::SentencePieceProcessor> for Tok {
    fn from(sp: sentencepiece::SentencePieceProcessor) -> Self {
        Tok::Sp(std::sync::Arc::new(sp))
    }
}

impl From<tokenizers::Tokenizer> for Tok {
    fn from(tok: tokenizers::Tokenizer) -> Self {
        Tok::Hf(Box::new(tok))
    }
}

impl ptts::Tokenizer for Tok {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
        let tokens = match self {
            Tok::Sp(sp) => {
                sp.encode(text).map_err(xn::Error::wrap)?.into_iter().map(|v| v.id).collect()
            }
            Tok::Hf(tok) => tok.encode(text, false).map_err(xn::Error::wrap)?.get_ids().to_vec(),
        };
        Ok(tokens)
    }

    fn decode(&self, ids: &[u32]) -> xn::Result<String> {
        let decoded = match self {
            Tok::Sp(sp) => sp.decode_piece_ids(ids).map_err(xn::Error::wrap)?,
            Tok::Hf(tok) => tok.decode(ids, true).map_err(xn::Error::wrap)?,
        };
        Ok(decoded)
    }
}

fn remap_key(name: &str) -> Option<String> {
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
    {
        return None;
    }
    let mut name = name.to_string();
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");
    Some(name)
}

fn load_voice_embedding<B: xn::Backend>(
    voice_path: &std::path::Path,
    device: &B,
) -> Result<Tensor<f32, B>> {
    let voice_vb = VB::load(&[voice_path], device.clone())?;
    let voice_names = voice_vb.tensor_names();
    let voice_key = voice_names.first().context("no tensors found in voice embedding file")?;
    let voice_shape = voice_vb.shape(voice_key).context("voice tensor not found")?;
    let voice_dims = voice_shape.dims();
    let voice_emb: Tensor<f32, B> = voice_vb.tensor(voice_key, voice_shape.clone())?;
    if voice_dims.len() == 2 {
        Ok(voice_emb.reshape((1, voice_dims[0], voice_dims[1]))?)
    } else {
        Ok(voice_emb)
    }
}

pub struct AppStateB<Q: BackendQ> {
    pub model: Arc<TTSModel<Q>>,
    pub voices: HashMap<String, Tensor<Q::T, Q::B>>,
    pub default_voice: String,
    pub max_seq_len: usize,
    pub temperature: f32,
    pub seed_base: u64,
    pub sample_rate: u32,
    pub frame_size: u32,
    /// How this state was built (`webgpu f16`, `cpu q8_0`, ...). Reported to the
    /// browser so a measurement is always labelled with what produced it.
    pub backend: String,
}

#[derive(Clone)]
pub enum AppState {
    Cpu(Arc<AppStateB<xn::Unquantized<f32, xn::CpuDevice>>>),
    Q80(Arc<AppStateB<xn::quantized::Q80F32>>),
    Q81(Arc<AppStateB<xn::quantized::Q81F32>>),
    Q8k(Arc<AppStateB<xn::quantized::Q8kF32>>),
    Q6k(Arc<AppStateB<xn::quantized::Q6kF32>>),
    Q50(Arc<AppStateB<xn::quantized::Q50F32>>),
    Q51(Arc<AppStateB<xn::quantized::Q51F32>>),
    Q5k(Arc<AppStateB<xn::quantized::Q5kF32>>),
    Q40(Arc<AppStateB<xn::quantized::Q40F32>>),
    Q41(Arc<AppStateB<xn::quantized::Q41F32>>),
    Q4k(Arc<AppStateB<xn::quantized::Q4kF32>>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<AppStateB<xn::Unquantized<half::bf16, xn::CudaDevice>>>),
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<AppStateB<xn::Unquantized<f32, xn::VulkanDevice>>>),
    #[cfg(feature = "metal")]
    Metal(Arc<AppStateB<xn::Unquantized<f32, xn::MetalDevice>>>),
    #[cfg(feature = "webgpu")]
    WebGpu(Arc<AppStateB<xn::Unquantized<f32, xn::WebGpuDevice>>>),
    #[cfg(feature = "webgpu")]
    WebGpuF16(Arc<AppStateB<xn::Unquantized<half::f16, xn::WebGpuDevice>>>),
    #[cfg(feature = "webgpu")]
    WebGpuQ80(Arc<AppStateB<xn::webgpu_backend::quantization::Q80F32>>),
    #[cfg(feature = "webgpu")]
    WebGpuQ80F16(Arc<AppStateB<xn::webgpu_backend::quantization::Q80F16>>),
}

/// Runs `$body` with `$s` bound to the concrete `Arc<AppStateB<Q>>` inside an
/// `AppState`. Every arm has a different `Q`, so this has to be a macro rather
/// than a method: the arms only agree once `$body` is generic over `Q`.
macro_rules! dispatch {
    ($state:expr, |$s:ident| $body:expr) => {
        match $state {
            AppState::Cpu($s) => $body,
            AppState::Q80($s) => $body,
            AppState::Q81($s) => $body,
            AppState::Q8k($s) => $body,
            AppState::Q6k($s) => $body,
            AppState::Q50($s) => $body,
            AppState::Q51($s) => $body,
            AppState::Q5k($s) => $body,
            AppState::Q40($s) => $body,
            AppState::Q41($s) => $body,
            AppState::Q4k($s) => $body,
            #[cfg(feature = "cuda")]
            AppState::Cuda($s) => $body,
            #[cfg(feature = "vulkan")]
            AppState::Vulkan($s) => $body,
            #[cfg(feature = "metal")]
            AppState::Metal($s) => $body,
            #[cfg(feature = "webgpu")]
            AppState::WebGpu($s) => $body,
            #[cfg(feature = "webgpu")]
            AppState::WebGpuF16($s) => $body,
            #[cfg(feature = "webgpu")]
            AppState::WebGpuQ80($s) => $body,
            #[cfg(feature = "webgpu")]
            AppState::WebGpuQ80F16($s) => $body,
        }
    };
}
pub(crate) use dispatch;

/// What `/api/info` reports: enough for the UI to label a run and populate its
/// voice picker without a websocket round-trip.
#[derive(serde::Serialize)]
pub struct AppInfo {
    pub backend: String,
    pub device: String,
    pub sample_rate: u32,
    pub frame_size: u32,
    pub temperature: f32,
    pub max_seq_len: usize,
    pub default_voice: String,
    pub voices: Vec<String>,
    pub threads: usize,
}

impl AppState {
    pub fn info(&self) -> AppInfo {
        dispatch!(self, |s| {
            let mut voices: Vec<String> = s.voices.keys().cloned().collect();
            voices.sort();
            AppInfo {
                backend: s.backend.clone(),
                device: xn::Backend::name(s.model.device()),
                sample_rate: s.sample_rate,
                frame_size: s.frame_size,
                temperature: s.temperature,
                max_seq_len: s.max_seq_len,
                default_voice: s.default_voice.clone(),
                voices,
                threads: xn::get_num_threads(),
            }
        })
    }
}

struct LoadedModel<Q: BackendQ> {
    cfg: TTSConfig,
    voices: HashMap<String, Tensor<Q::T, Q::B>>,
    tokenizer_path: std::path::PathBuf,
    model_path: std::path::PathBuf,
}

impl<Q: BackendQ> LoadedModel<Q> {
    fn load_from_hf(repo_id: &str, temperature: f32, dev: &Q::B) -> Result<Self> {
        tracing::info!("downloading model artifacts");
        let repo = crate::utils::HfRepo::model(repo_id)?;
        let config_path = repo.get("config.json")?;
        let mut cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)
            .with_context(|| "failed to read config from file {config:?}")?;
        cfg.temp = temperature;

        let model_path = repo.get("model.q8.gguf")?;
        tracing::info!(?model_path, "model weights ready");
        let tokenizer_path = repo.get("tokenizer.model")?;

        let mut voices: HashMap<String, Tensor<Q::T, Q::B>> = HashMap::new();
        let default_voice = load_voice_embedding(&repo.get("default-voice.safetensors")?, dev)
            .with_context(|| "failed to load default voice embedding")?
            .to::<Q::T>()
            .with_context(|| "failed to convert default voice embedding")?;
        voices.insert("default".to_string(), default_voice);
        tracing::info!(num_voices = voices.len(), "voice embeddings loaded");

        Ok(Self { cfg, voices, tokenizer_path, model_path })
    }

    fn load_pocket_from_hf(temperature: f32, dev: &Q::B) -> Result<Self> {
        tracing::info!("downloading model artifacts");
        let repo = crate::utils::HfRepo::model(DEFAULT_REPO_ID)?;
        let model_path = repo.get(DEFAULT_MODEL_FILE)?;
        tracing::info!(?model_path, "model weights ready");
        let tokenizer_path = repo.get("tokenizer.model")?;

        let mut voices: HashMap<String, Tensor<Q::T, Q::B>> = HashMap::new();
        for &voice in VOICES {
            let voice_file = format!("embeddings/{voice}.safetensors");
            match repo.get(&voice_file) {
                Ok(voice_path) => match load_voice_embedding(&voice_path, dev) {
                    Ok(emb) => match emb.to::<Q::T>() {
                        Ok(emb) => {
                            voices.insert(voice.to_string(), emb);
                        }
                        Err(e) => {
                            tracing::warn!(?voice, error = %e, "failed to convert voice embedding")
                        }
                    },
                    Err(e) => tracing::warn!(?voice, error = %e, "failed to load voice embedding"),
                },
                Err(e) => tracing::warn!(?voice, error = %e, "failed to download voice embedding"),
            }
        }
        tracing::info!(num_voices = voices.len(), "voice embeddings loaded");

        let cfg = TTSConfig::v202601(temperature);
        Ok(Self { cfg, voices, tokenizer_path, model_path })
    }

    fn load_from_path(config: &std::path::PathBuf, temperature: f32, dev: &Q::B) -> Result<Self> {
        let parent_dir = config
            .parent()
            .with_context(|| format!("failed to get parent directory of config path {config:?}"))?;
        let mut cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(config)?)
            .with_context(|| "failed to read config from file {config:?}")?;
        cfg.temp = temperature;
        let model_path = if parent_dir.join("model.safetensors").is_file() {
            parent_dir.join("model.safetensors")
        } else if parent_dir.join("model.q8.gguf").is_file() {
            parent_dir.join("model.q8.gguf")
        } else {
            anyhow::bail!(
                "model file not found in directory {parent_dir:?}; expected model.safetensors or model.gguf"
            );
        };
        let tokenizer_path = parent_dir.join("tokenizer.model");
        // Voices come from a `voices/` subdirectory when there is one, and from a
        // sibling `default-voice.safetensors` otherwise. Neither is required here:
        // `--voice-dir` can supply them, and having none at all is reported later
        // by `load_ptts` with a clearer message than a `read_dir` error.
        let mut voices: HashMap<String, Tensor<Q::T, Q::B>> = HashMap::new();
        let voices_dir = parent_dir.join("voices");
        if voices_dir.is_dir() {
            load_voices_from_dir::<Q>(&voices_dir, dev, &mut voices);
        }
        let default_voice_file = parent_dir.join("default-voice.safetensors");
        if default_voice_file.is_file() {
            match load_voice_embedding(&default_voice_file, dev).and_then(|e| Ok(e.to::<Q::T>()?)) {
                Ok(emb) => {
                    voices.insert("default".to_string(), emb);
                }
                Err(e) => tracing::warn!(error = %e, "failed to load default voice embedding"),
            }
        }
        tracing::info!(num_voices = voices.len(), "voice embeddings loaded");
        Ok(Self { cfg, voices, tokenizer_path, model_path })
    }
}

/// Load every `*.safetensors` file in `dir` as a voice embedding, keyed by file
/// stem. Errors (unreadable directory, bad file, conversion failure) are logged
/// and skipped rather than propagated.
fn load_voices_from_dir<Q: BackendQ>(
    dir: &std::path::Path,
    dev: &Q::B,
    voices: &mut HashMap<String, Tensor<Q::T, Q::B>>,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(?dir, error = %e, "failed to read voice directory");
            return;
        }
    };
    for entry in entries {
        let path = match entry {
            Ok(entry) => entry.path(),
            Err(e) => {
                tracing::warn!(?dir, error = %e, "failed to read voice directory entry");
                continue;
            }
        };
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        let voice_name = match path.file_stem().and_then(|s| s.to_str()) {
            Some(name) => name.to_string(),
            None => {
                tracing::warn!(?path, "invalid voice file name");
                continue;
            }
        };
        match load_voice_embedding(&path, dev) {
            Ok(emb) => match emb.to::<Q::T>() {
                Ok(emb) => {
                    voices.insert(voice_name, emb);
                }
                Err(e) => {
                    tracing::warn!(?voice_name, error = %e, "failed to convert voice embedding")
                }
            },
            Err(e) => tracing::warn!(?voice_name, error = %e, "failed to load voice embedding"),
        }
    }
}

/// Everything `load_ptts` needs that does not depend on the backend type, so the
/// many `AppState` arms in `main` can share one value instead of threading seven
/// positional arguments through each.
pub struct LoadOpts<'a> {
    pub config: Option<&'a std::path::PathBuf>,
    /// Explicit weights file, overriding the `model.safetensors` / `model.q8.gguf`
    /// lookup next to the config. A gpu backend that quantizes at load time wants
    /// the f32 safetensors even when a gguf sits beside it.
    pub model: Option<&'a std::path::PathBuf>,
    pub voice_dir: Option<&'a std::path::PathBuf>,
    pub temperature: f32,
    pub seed_base: u64,
    pub max_seq_len: usize,
    pub backend: String,
}

pub fn load_ptts<Q: BackendQ>(opts: &LoadOpts<'_>, dev: Q::B) -> Result<AppStateB<Q>> {
    let LoadOpts {
        config,
        model: model_override,
        voice_dir,
        temperature,
        seed_base,
        max_seq_len,
        backend,
    } = opts;
    let (temperature, seed_base, max_seq_len) = (*temperature, *seed_base, *max_seq_len);
    let mut m = match config {
        Some(config) if config.is_file() || config.extension().is_some_and(|v| v == "json") => {
            LoadedModel::<Q>::load_from_path(config, temperature, &dev)?
        }
        Some(repo_id) => {
            let repo_id = repo_id.to_str().context("invalid repo ID path")?;
            LoadedModel::<Q>::load_from_hf(repo_id, temperature, &dev)?
        }
        None => LoadedModel::<Q>::load_pocket_from_hf(temperature, &dev)?,
    };
    if let Some(model_override) = model_override {
        anyhow::ensure!(
            model_override.is_file(),
            "--model {} does not exist",
            model_override.display()
        );
        tracing::info!(model = %model_override.display(), "using explicit model weights");
        m.model_path = (*model_override).clone();
    }
    if let Some(voice_dir) = voice_dir {
        load_voices_from_dir::<Q>(voice_dir, &dev, &mut m.voices);
        tracing::info!(num_voices = m.voices.len(), "voice embeddings loaded (incl. voice-dir)");
    }
    let tokenizer_path = m.tokenizer_path.to_str().context("invalid tokenizer path")?;
    let tokenizer = if tokenizer_path.ends_with(".model") {
        tracing::info!("loading SentencePiece tokenizer");
        let sp = sentencepiece::SentencePieceProcessor::open(tokenizer_path)
            .with_context(|| format!("failed to open tokenizer at {tokenizer_path}"))?;
        Tok::Sp(sp.into())
    } else {
        tracing::info!("loading Hugging Face tokenizer");
        let tok = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::format_err!("failed to load tokenizer: {e}"))?;
        Tok::Hf(Box::new(tok))
    };

    let vb = if m.model_path.extension().and_then(|v| v.to_str()) == Some("gguf") {
        let reader = std::fs::File::open(&m.model_path)?;
        let reader = std::io::BufReader::new(reader);
        VB::load_gguf_with_key_map(reader, dev, remap_key)?
    } else {
        VB::load_with_key_map(&[&m.model_path], dev, remap_key)?
    };
    let vb = vb.root();
    let model: TTSModel<Q> = TTSModel::load(&vb, Box::new(tokenizer), &m.cfg)?;
    vb.check_all_used_with_ignore(|v| {
        v == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
            || v.starts_with("mimi.encoder")
            || v.starts_with("mimi.downsample.")
            || v == "flow_lm.speaker_proj_weight"
            || v.starts_with("mimi.quantizer")
    })?;

    let sample_rate = model.sample_rate() as u32;
    let frame_size = (sample_rate as f64 / m.cfg.mimi.frame_rate).round() as u32;
    let default_voice = match m.voices.keys().min() {
        Some(name) => name.clone(),
        None => anyhow::bail!("no voice embeddings found in model"),
    };
    tracing::info!(backend = %backend, device = %xn::Backend::name(model.device()), "model ready");
    Ok(AppStateB {
        model: Arc::new(model),
        voices: m.voices,
        default_voice,
        max_seq_len,
        temperature,
        seed_base,
        sample_rate,
        frame_size,
        backend: backend.clone(),
    })
}

/// Run a single text-to-audio generation, sending each decoded PCM chunk as it
/// becomes available. Designed to be called inside `tokio::task::spawn_blocking`.
pub fn generate_chunks<Q: BackendQ>(
    model: Arc<TTSModel<Q>>,
    mut state: TTSState<Q>,
    tokens: Vec<u32>,
    temperature: f32,
    seed: u64,
    frames_after_eos: usize,
    audio_tx: tokio::sync::mpsc::UnboundedSender<Vec<f32>>,
) -> Result<(), xn::Error> {
    let device = model.device();
    let num_tokens = tokens.len();
    let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
    let mut rng = StdRng::new(temperature, seed);
    let mut mimi_state = model.init_mimi_state(1, 250)?;

    model.prompt_text(&mut state, &tokens)?;

    let ldim = model.flow_lm.ldim;
    let nan_data = vec![f32::NAN; ldim];
    let mut prev_latent: Tensor<Q::T, Q::B> =
        Tensor::from_vec(nan_data, (1, 1, ldim), device)?.to::<Q::T>()?;

    let (latent_tx, latent_rx) = std::sync::mpsc::channel::<Tensor<Q::T, Q::B>>();

    let decode_model = Arc::clone(&model);
    let decode_audio_tx = audio_tx.clone();
    let decode_handle = std::thread::spawn(move || -> Result<(), xn::Error> {
        while let Ok(latent) = latent_rx.recv() {
            let audio_chunk = decode_model.decode_latent(&latent, &mut mimi_state)?;
            let pcm = audio_chunk.narrow(0, ..1)?.contiguous()?.to_vec()?;
            if decode_audio_tx.send(pcm).is_err() {
                // Client gone — stop draining.
                break;
            }
        }
        Ok(())
    });

    let mut eos_countdown: Option<usize> = None;
    for _ in 0..max_frames {
        let (next_latent, is_eos) = model.generate_step(&mut state, &prev_latent, &mut rng)?;
        if latent_tx.send(next_latent.clone()).is_err() {
            break;
        }
        if is_eos && eos_countdown.is_none() {
            eos_countdown = Some(frames_after_eos);
        }
        if let Some(ref mut countdown) = eos_countdown {
            if *countdown == 0 {
                break;
            }
            *countdown -= 1;
        }
        prev_latent = next_latent;
    }
    drop(latent_tx);
    decode_handle.join().map_err(|_| xn::Error::msg("decode thread panicked"))??;
    Ok(())
}
