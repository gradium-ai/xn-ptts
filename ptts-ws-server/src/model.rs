use anyhow::{Context as _, Result};
use ptts::flow_lm::NormalRng;
use ptts::loader::{is_unused_by_tts_model, load_voice_emb, load_weights};
use ptts::tok::Tok;
use ptts::tts_model::{TTSConfig, TTSModel, TTSState};
use std::collections::HashMap;
use std::sync::Arc;
use xn::{BackendQ, Tensor};

pub const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

pub const DEFAULT_REPO_ID: &str = "kyutai/pocket-tts";
pub const DEFAULT_MODEL_FILE: &str = "tts_b6369a24.safetensors";

pub struct AppStateB<Q: BackendQ> {
    pub model: Arc<TTSModel<Q>>,
    pub voices: HashMap<String, Tensor<Q::T, Q::B>>,
    pub default_voice: String,
    pub max_seq_len: usize,
    pub temperature: f32,
    pub seed_base: u64,
    pub sample_rate: u32,
    pub frame_size: u32,
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
        let default_voice = load_voice_emb(&repo.get("default-voice.safetensors")?, None, dev)
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
                Ok(voice_path) => match load_voice_emb(&voice_path, None, dev) {
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
        let mut voices: HashMap<String, Tensor<Q::T, Q::B>> = HashMap::new();
        for voice in parent_dir.join("voices").read_dir()? {
            let voice = match voice {
                Ok(v) => v,
                Err(_) => continue,
            };
            let voice = voice.path();
            if voice.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }
            let voice_name =
                voice.file_stem().and_then(|s| s.to_str()).context("invalid voice file name")?;
            match load_voice_emb(&voice, None, dev) {
                Ok(emb) => match emb.to::<Q::T>() {
                    Ok(emb) => {
                        voices.insert(voice_name.to_string(), emb);
                    }
                    Err(e) => {
                        tracing::warn!(?voice_name, error = %e, "failed to convert voice embedding")
                    }
                },
                Err(e) => tracing::warn!(?voice_name, error = %e, "failed to load voice embedding"),
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
        match load_voice_emb(&path, None, dev) {
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

pub fn load_ptts<Q: BackendQ>(
    config: Option<&std::path::PathBuf>,
    voice_dir: Option<&std::path::PathBuf>,
    temperature: f32,
    seed_base: u64,
    max_seq_len: usize,
    dev: Q::B,
) -> Result<AppStateB<Q>> {
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
    if let Some(voice_dir) = voice_dir {
        load_voices_from_dir::<Q>(voice_dir, &dev, &mut m.voices);
        tracing::info!(num_voices = m.voices.len(), "voice embeddings loaded (incl. voice-dir)");
    }
    let tokenizer = Tok::open(&m.tokenizer_path)
        .with_context(|| format!("failed to open tokenizer at {}", m.tokenizer_path.display()))?;

    let vb = load_weights::<Q>(&m.model_path, &dev)?;
    let model: TTSModel<Q> = TTSModel::load(&vb, Box::new(tokenizer), &m.cfg)?;
    vb.check_all_used_with_ignore(is_unused_by_tts_model)?;

    let sample_rate = model.sample_rate() as u32;
    let frame_size = (sample_rate as f64 / m.cfg.mimi.frame_rate).round() as u32;
    let default_voice = match m.voices.keys().min() {
        Some(name) => name.clone(),
        None => anyhow::bail!("no voice embeddings found in model"),
    };
    Ok(AppStateB {
        model: Arc::new(model),
        voices: m.voices,
        default_voice,
        max_seq_len,
        temperature,
        seed_base,
        sample_rate,
        frame_size,
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
    let mut rng = NormalRng::new(temperature, seed)?;
    let mut mimi_state = model.init_mimi_state(1)?;

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
        let (next_latent, is_eos, _eos_logit) =
            model.generate_step(&mut state, &prev_latent, &mut rng)?;
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
