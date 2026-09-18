//! Locating a checkpoint on disk or on the Hub, and loading it into a [`Synth`].
//!
//! Everything about *how* speech is generated lives in `ptts::synth`. What is
//! left here is deciding which files to load, which voices to register, and
//! holding the result for the request handlers.

use anyhow::{Context as _, Result};
use ptts::synth::{DeviceKind, Quant, Synth, SynthBuilder};
use ptts::tts_model::TTSConfig;
use std::sync::Arc;

/// Voices the published checkpoint ships, used to name the files to fetch.
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

/// A checkpoint's files, located but not yet loaded.
///
/// No longer generic over the weight format: these are paths, and the voices
/// are registered by path too, so nothing here needs to know the backend.
struct Artifacts {
    cfg: TTSConfig,
    /// Voice name to embedding file.
    voices: Vec<(String, std::path::PathBuf)>,
    tokenizer_path: std::path::PathBuf,
    model_path: std::path::PathBuf,
}

impl Artifacts {
    /// A repo laid out with `config.json`, `model.q8.gguf` and one default voice.
    fn from_hf(repo_id: &str, temperature: f32) -> Result<Self> {
        tracing::info!("downloading model artifacts");
        let repo = crate::utils::HfRepo::model(repo_id)?;
        let config_path = repo.get("config.json")?;
        let mut cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
            .with_context(|| format!("failed to read config from {}", config_path.display()))?;
        cfg.temp = temperature;

        let model_path = repo.get("model.q8.gguf")?;
        tracing::info!(?model_path, "model weights ready");
        let tokenizer_path = repo.get("tokenizer.model")?;
        let voices = vec![("default".to_string(), repo.get("default-voice.safetensors")?)];

        Ok(Self { cfg, voices, tokenizer_path, model_path })
    }

    /// The published pocket-tts repo, whose voices sit under `embeddings/`.
    fn pocket_from_hf(temperature: f32) -> Result<Self> {
        tracing::info!("downloading model artifacts");
        let repo = crate::utils::HfRepo::model(DEFAULT_REPO_ID)?;
        let model_path = repo.get(DEFAULT_MODEL_FILE)?;
        tracing::info!(?model_path, "model weights ready");
        let tokenizer_path = repo.get("tokenizer.model")?;

        let mut voices = vec![];
        for &voice in VOICES {
            match repo.get(&format!("embeddings/{voice}.safetensors")) {
                Ok(path) => voices.push((voice.to_string(), path)),
                // One missing voice should not stop the server from starting.
                Err(e) => tracing::warn!(?voice, error = %e, "failed to download voice embedding"),
            }
        }
        Ok(Self { cfg: TTSConfig::v202601(temperature), voices, tokenizer_path, model_path })
    }

    /// A local directory named by its `config.json`.
    fn from_path(config: &std::path::Path, temperature: f32) -> Result<Self> {
        let parent = config
            .parent()
            .with_context(|| format!("config path {config:?} has no parent directory"))?;
        let mut cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(config)?)
            .with_context(|| format!("failed to read config from {}", config.display()))?;
        cfg.temp = temperature;

        let model_path = if parent.join("model.safetensors").is_file() {
            parent.join("model.safetensors")
        } else if parent.join("model.q8.gguf").is_file() {
            parent.join("model.q8.gguf")
        } else {
            anyhow::bail!(
                "no model file in {parent:?}; expected model.safetensors or model.q8.gguf"
            )
        };
        let mut voices = vec![];
        collect_voices(&parent.join("voices"), &mut voices);
        Ok(Self { cfg, voices, tokenizer_path: parent.join("tokenizer.model"), model_path })
    }
}

/// Add every `*.safetensors` file in `dir` to `voices`, keyed by file stem.
/// A missing or unreadable directory is logged, not fatal: voices are optional
/// until the server finds it has none at all.
fn collect_voices(dir: &std::path::Path, voices: &mut Vec<(String, std::path::PathBuf)>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(?dir, error = %e, "failed to read voice directory");
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        match path.file_stem().and_then(|s| s.to_str()) {
            Some(name) => voices.push((name.to_string(), path)),
            None => tracing::warn!(?path, "invalid voice file name"),
        }
    }
}

/// Load the model named by `config` — a local `config.json`, a Hub repo id, or
/// nothing for the published checkpoint.
pub fn load_ptts(
    config: Option<&std::path::PathBuf>,
    voice_dir: Option<&std::path::PathBuf>,
    device: DeviceKind,
    quant: Quant,
    temperature: f32,
    seed_base: u64,
    max_seq_len: usize,
) -> Result<AppState> {
    let mut artifacts = match config {
        Some(config) if config.is_file() || config.extension().is_some_and(|v| v == "json") => {
            Artifacts::from_path(config, temperature)?
        }
        Some(repo_id) => {
            Artifacts::from_hf(repo_id.to_str().context("invalid repo ID path")?, temperature)?
        }
        None => Artifacts::pocket_from_hf(temperature)?,
    };
    if let Some(voice_dir) = voice_dir {
        collect_voices(voice_dir, &mut artifacts.voices);
    }
    artifacts.voices.sort();
    artifacts.voices.dedup_by(|a, b| a.0 == b.0);

    let frame_rate = artifacts.cfg.mimi.frame_rate;
    let mut builder = SynthBuilder::new(artifacts.cfg, &artifacts.model_path)
        .tokenizer_file(&artifacts.tokenizer_path)
        .device(device)
        .quant(quant)
        .temperature(temperature);
    for (name, path) in artifacts.voices.iter() {
        builder = builder.add_voice(name, path);
    }
    let synth = builder.build()?;

    let voices = synth.voices();
    let default_voice = voices.first().context("no voice embeddings found in model")?.clone();
    let sample_rate = synth.sample_rate() as u32;
    let frame_size = (sample_rate as f64 / frame_rate).round() as u32;
    tracing::info!(
        device = %synth.device_name(),
        weights = %synth.quant().as_str(),
        num_voices = voices.len(),
        %default_voice,
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
