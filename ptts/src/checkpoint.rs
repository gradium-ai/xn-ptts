//! Shared native checkpoint file resolution.
//!
//! [`Checkpoint::locate`] accepts a model directory or config file, or a Hugging Face repo as
//! `hf://owner/name[@revision]` (see [`crate::hub`]). Frontends load [`DEFAULT_MODEL`] when none
//! is given.
//! Each checkpoint supplies its own config, tokenizer, weights, and optional voices.

use crate::hub::HubSource;
use crate::preprocess::Normalize;
use crate::synth::{Quant, Synth, SynthBuilder};
use crate::tts_model::TTSConfig;
use crate::{Error, Result};
use std::path::{Path, PathBuf};

/// The model every frontend loads when none is given, pinned to a commit so a default install is
/// reproducible. Loading it needs the `hub` feature. The browser package keeps a copy in
/// `ptts-wasm/js/hub.js`, which its tests hold to this one.
pub const DEFAULT_MODEL: &str =
    "hf://gradium/phonon-31m-en@8c64b55ec301728e2d1695bfea87a08a267d0c2d";

/// Standard weight filenames, ordered by the requested quantization.
pub fn weight_candidates(quant: Quant) -> [&'static str; 2] {
    if quant == Quant::Q4k {
        ["model.q4k.gguf", "model.safetensors"]
    } else if quant == Quant::Q80 {
        ["model.q8.gguf", "model.safetensors"]
    } else {
        ["model.safetensors", "model.q8.gguf"]
    }
}

/// Explicit native resolution choices.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResolveOptions<'a> {
    pub quant: Quant,
    /// A weights path relative to the supplied model directory.
    pub weights: Option<&'a str>,
}

/// Checkpoint files ready to hand to a native frontend or [`SynthBuilder`].
#[derive(Debug)]
pub struct Checkpoint {
    pub config: TTSConfig,
    pub weights: PathBuf,
    /// The checkpoint's tokenizer, unless the caller supplies one separately.
    pub tokenizer: Option<PathBuf>,
    /// External voices, sorted by name. Baked voices come from the config and weights.
    pub voices: Vec<(String, PathBuf)>,
    pub quant: Quant,
}

impl Checkpoint {
    /// Resolve a model source: `hf://owner/name[@revision]` on the Hugging Face Hub, which needs
    /// the `hub` feature, or else a local model directory or config file.
    pub fn locate(source: &str, options: ResolveOptions<'_>) -> Result<Self> {
        match HubSource::parse(source)? {
            Some(hub) => crate::hub::resolve(&hub, options),
            None => Self::resolve(source, options),
        }
    }

    /// Resolve a supplied model directory or config file with default options.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::resolve(path, ResolveOptions::default())
    }

    pub fn resolve(path: impl AsRef<Path>, options: ResolveOptions<'_>) -> Result<Self> {
        let input = path.as_ref();
        let path = std::fs::canonicalize(input).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => Error::NotFound(match input.to_str() {
                // `owner/name` was how a Hub repo used to be given.
                Some(s)
                    if !s.starts_with(['.', '/'])
                        && !s.contains([':', '\\'])
                        && !s.ends_with(".json")
                        && s.split('/').count() == 2 =>
                {
                    format!(
                        "no checkpoint at {s}; for a Hugging Face repo, use {}{s}",
                        crate::hub::SCHEME
                    )
                }
                _ => format!("no checkpoint at {input:?}"),
            }),
            _ => Error::Io(e),
        })?;
        let (dir, config_path) = if path.is_dir() {
            (path.clone(), path.join("config.json"))
        } else {
            // HF cache files can link to blobs outside the snapshot. Their neighbors
            // belong to the supplied path's parent rather than the blob directory.
            let parent =
                input.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
            (std::fs::canonicalize(parent)?, path)
        };
        let config = read_config(config_path)?;
        let weights = match options.weights {
            Some(name) => {
                validate_relative_path(name)?;
                required_file(&dir.join(name))?
            }
            None => weight_candidates(options.quant)
                .iter()
                .map(|name| dir.join(name))
                .find(|p| p.is_file())
                .ok_or_else(|| {
                    Error::NotFound(format!(
                        "no weights in {dir:?}; expected one of {} or an explicit weights path",
                        weight_candidates(options.quant).join(", ")
                    ))
                })?,
        };
        let tokenizer = dir.join("tokenizer.json");
        let tokenizer = tokenizer.is_file().then_some(tokenizer);
        let mut voices = crate::loader::checkpoint_voices(&dir);
        voices.sort();
        Ok(Self { config, weights, tokenizer, voices, quant: options.quant })
    }

    pub fn builder(&self, normalize: impl Into<Normalize>) -> SynthBuilder {
        let mut builder =
            SynthBuilder::new(self.config.clone(), &self.weights, normalize).quant(self.quant);
        if let Some(path) = &self.tokenizer {
            builder = builder.tokenizer_file(path);
        }
        builder
    }

    /// Add the voices of a JSONL catalog, see [`crate::loader::read_voice_catalog`],
    /// replacing any of the same name.
    pub fn add_voice_catalog(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let mut voices: std::collections::BTreeMap<_, _> = self.voices.drain(..).collect();
        for (name, path) in crate::loader::read_voice_catalog(path.as_ref())? {
            voices.insert(name, path);
        }
        self.voices = voices.into_iter().collect();
        Ok(())
    }

    /// Register checkpoint voices, warning if an optional voice is unreadable.
    pub fn register_voices(&self, synth: &mut Synth) {
        for (name, path) in &self.voices {
            if let Err(e) = synth.add_voice_file(name, path) {
                tracing::warn!(voice = %name, error = %e, "skipping voice embedding");
            }
        }
    }
}

/// The source [`Checkpoint::locate`] takes for the older way of naming a model, still accepted
/// by the servers' `--config` and Python's `config=`: a local path, or a bare `owner/name` repo
/// ID with a separate revision.
pub fn legacy_source(config: &str, revision: Option<&str>) -> Result<String> {
    let path = Path::new(config);
    let local = path.exists()
        || path.has_root()
        || path.starts_with(".")
        || path.components().next().is_some_and(|c| c.as_os_str() == "..")
        || path.extension().is_some_and(|ext| ext == "json");
    if config.starts_with(crate::hub::SCHEME) && revision.is_some() {
        return Err(Error::InvalidArgument(
            "the revision goes in the model: hf://owner/name@revision".into(),
        ));
    }
    if local || config.starts_with(crate::hub::SCHEME) {
        if revision.is_some() {
            return Err(Error::InvalidArgument(
                "a revision requires a Hugging Face repo ID".into(),
            ));
        }
        return Ok(config.to_string());
    }
    let hub = HubSource { repo: config.to_string(), revision: revision.map(str::to_string) };
    Ok(hub.to_string())
}

/// Read the checkpoint's config with consistent error classes and context.
pub fn read_config(path: impl AsRef<Path>) -> Result<TTSConfig> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            Error::NotFound(format!("cannot read config {path:?}: {e}"))
        }
        _ => Error::Io(e),
    })?;
    serde_json::from_str(&text)
        .map_err(|e| Error::InvalidData(format!("cannot parse config {path:?}: {e}")))
}

pub(crate) fn validate_relative_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.contains(['\\', ':'])
        || path.starts_with('/')
        || path.split('/').any(|p| p.is_empty() || p == "." || p == "..")
    {
        return Err(Error::InvalidData(format!(
            "weights path '{path}' must be relative to the checkpoint root"
        )));
    }
    Ok(())
}

fn required_file(path: &Path) -> Result<PathBuf> {
    if !path.is_file() {
        return Err(Error::NotFound(format!("checkpoint file is missing: {path:?}")));
    }
    Ok(path.to_path_buf())
}
