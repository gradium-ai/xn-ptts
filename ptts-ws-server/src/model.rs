//! Locating a checkpoint on disk or on the Hub, and loading it into a [`Synth`].
//!
//! Everything about *how* speech is generated lives in `ptts::synth`. What is
//! left here is deciding which files to load, which voices to register, and
//! holding the result for the request handlers.

use anyhow::{Context as _, Result};
use ptts::preprocess::Normalize;
use ptts::synth::{DeviceKind, Quant, Synth, SynthBuilder};
use ptts::tts_model::TTSConfig;
use std::sync::Arc;

pub const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

pub const DEFAULT_REPO_ID: &str = "kyutai/pocket-tts";
pub const DEFAULT_MODEL_FILE: &str = "tts_b6369a24.safetensors";

/// The loaded model and the request defaults, shared by every connection.
///
/// `Synth` erases the weight format, so this is one struct rather than the
/// fourteen-variant enum the handlers used to match on.
#[derive(Clone)]
pub struct AppState(Arc<Inner>);

pub struct Inner {
    pub synth: Synth,
    pub voices: Vec<String>,
    pub default_voice: String,
    pub max_seq_len: usize,
    pub temperature: f32,
    pub seed_base: u64,
    pub sample_rate: u32,
    pub frame_size: u32,
}

impl std::ops::Deref for AppState {
    type Target = Inner;

    fn deref(&self) -> &Inner {
        &self.0
    }
}

/// A checkpoint whose files are located. The voices are loaded once the model is, since a
/// file of stored speaker latents goes through the checkpoint's speaker projection.
struct LoadedModel {
    cfg: TTSConfig,
    /// Voice name to embedding file.
    voice_files: Vec<(String, std::path::PathBuf)>,
    tokenizer_path: std::path::PathBuf,
    model_path: std::path::PathBuf,
}

impl LoadedModel {
    async fn load_from_hf(repo_id: &str, temperature: f32) -> Result<Self> {
        tracing::info!("downloading model artifacts");
        let repo = crate::utils::HfRepo::model(repo_id)?;
        let config_path = repo.get("config.json").await?;
        let mut cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)
            .with_context(|| "failed to read config from file {config:?}")?;
        cfg.temp = temperature;

        let model_path = repo.get("model.q8.gguf").await?;
        tracing::info!(?model_path, "model weights ready");
        let tokenizer_path = repo.get("tokenizer.json").await?;

        let default_voice_path = repo.get("default-voice.safetensors").await?;
        let voice_files = vec![("default".to_string(), default_voice_path)];

        Ok(Self { cfg, voice_files, tokenizer_path, model_path })
    }

    async fn load_pocket_from_hf(temperature: f32) -> Result<Self> {
        tracing::info!("downloading model artifacts");
        let repo = crate::utils::HfRepo::model(DEFAULT_REPO_ID)?;
        let model_path = repo.get(DEFAULT_MODEL_FILE).await?;
        tracing::info!(?model_path, "model weights ready");
        let tokenizer_path = repo.get("tokenizer.json").await?;

        let mut voice_files = Vec::new();
        for &voice in VOICES {
            let voice_file = format!("embeddings/{voice}.safetensors");
            match repo.get(&voice_file).await {
                Ok(voice_path) => voice_files.push((voice.to_string(), voice_path)),
                Err(e) => tracing::warn!(?voice, error = %e, "failed to download voice embedding"),
            }
        }

        let cfg = TTSConfig::v202601(temperature);
        Ok(Self { cfg, voice_files, tokenizer_path, model_path })
    }

    fn load_from_path(config: &std::path::PathBuf, temperature: f32) -> Result<Self> {
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
        let tokenizer_path = parent_dir.join("tokenizer.json");
        let mut voice_files = Vec::new();
        collect_voice_files(&parent_dir.join("voices"), &mut voice_files);
        Ok(Self { cfg, voice_files, tokenizer_path, model_path })
    }
}

/// Every `*.safetensors` file in `dir`, keyed by file stem, appended to `files`. An unreadable
/// directory or entry is logged and skipped rather than propagated.
fn collect_voice_files(dir: &std::path::Path, files: &mut Vec<(String, std::path::PathBuf)>) {
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
        match path.file_stem().and_then(|s| s.to_str()) {
            Some(name) => files.push((name.to_string(), path)),
            None => tracing::warn!(?path, "invalid voice file name"),
        }
    }
}

/// Load the model named by `config` -- a local `config.json`, a Hub repo id, or
/// nothing for the published checkpoint.
#[allow(clippy::too_many_arguments)]
pub async fn load_ptts(
    config: Option<&std::path::PathBuf>,
    voice_dir: Option<&std::path::PathBuf>,
    device: DeviceKind,
    quant: Quant,
    temperature: f32,
    seed_base: u64,
    max_seq_len: usize,
    normalize: Normalize,
) -> Result<AppState> {
    let mut m = match config {
        Some(config) if config.is_file() || config.extension().is_some_and(|v| v == "json") => {
            LoadedModel::load_from_path(config, temperature)?
        }
        Some(repo_id) => {
            let repo_id = repo_id.to_str().context("invalid repo ID path")?;
            LoadedModel::load_from_hf(repo_id, temperature).await?
        }
        None => LoadedModel::load_pocket_from_hf(temperature).await?,
    };
    if let Some(voice_dir) = voice_dir {
        collect_voice_files(voice_dir, &mut m.voice_files);
    }
    let frame_rate = m.cfg.mimi.frame_rate;
    let mut synth = SynthBuilder::new(m.cfg, &m.model_path, normalize)
        .tokenizer_file(&m.tokenizer_path)
        .device(device)
        .quant(quant)
        .temperature(temperature)
        .build()?;
    // Registered after the build, not through it: the builder propagates a bad
    // voice file and one should not take the server down. Order is preserved,
    // so a --voice-dir entry still overrides a bundled voice of the same name.
    for (name, path) in m.voice_files.iter() {
        if let Err(e) = synth.add_voice_file(name, path) {
            tracing::warn!(voice = %name, error = %e, "failed to load voice embedding");
        }
    }

    let voices = synth.voices();
    let default_voice = voices.first().context("no voice embeddings found in model")?.clone();
    let sample_rate = synth.sample_rate() as u32;
    let frame_size = (sample_rate as f64 / frame_rate).round() as u32;
    tracing::info!(
        device = %synth.device_name(),
        weights = %synth.quant().as_str(),
        num_voices = voices.len(),
        %default_voice,
        lang = normalize.as_str(),
        "model loaded"
    );

    Ok(AppState(Arc::new(Inner {
        synth,
        voices,
        default_voice,
        max_seq_len,
        temperature,
        seed_base,
        sample_rate,
        frame_size,
    })))
}
