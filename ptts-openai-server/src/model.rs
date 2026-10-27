//! Locating a checkpoint on disk or on the Hub, and loading it into a [`Synth`].
//!
//! Everything about *how* speech is generated lives in `ptts::synth`. What is
//! left here is deciding which files to load, which voices to register, and
//! holding the result for the request handlers.

use crate::compat::{Aliases, Compat};
use anyhow::{Context as _, Result};
use ptts::checkpoint::{Checkpoint, ResolveOptions};
use ptts::hub::HubSource;
use ptts::preprocess::Normalize;
use ptts::synth::{DeviceKind, Quant, Synth};
use ptts::voice_catalog::{DEFAULT_VOICE_CATALOG, VoiceCatalog};
use std::sync::Arc;

/// The loaded model and the request defaults, shared by every request.
#[derive(Clone)]
pub struct AppState(Arc<Inner>);

pub struct Inner {
    pub synth: Synth,
    pub requests: Arc<tokio::sync::Semaphore>,
    /// The checkpoint that loaded: its repo id, or for a local config the name of its folder.
    pub model_name: String,
    /// Where the checkpoint is from: its Hub page, or this repository for a local folder.
    pub model_url: String,
    /// How text is normalized. Wyoming lists the voices in its language.
    pub normalize: Normalize,
    pub voices: Vec<String>,
    /// Other names requests may give for those voices.
    pub aliases: Aliases,
    /// Accept a `speed` other than 1 and speak at the normal rate.
    pub ignore_speed: bool,
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

/// Resolve `model`, `hf://owner/name[@revision]` or a local checkpoint folder or `config.json`
/// in one, with the name it is reported by and where it is from.
async fn locate(model: &str, quant: Quant) -> Result<(Checkpoint, String, String)> {
    let hub = HubSource::parse(model)?;
    let source = model.to_string();
    // The blocking Hub client runs its own runtime, which cannot be started inside this one.
    let checkpoint = tokio::task::spawn_blocking(move || {
        Checkpoint::locate(&source, ResolveOptions { quant, weights: None })
    })
    .await??;
    if let Some(hub) = hub {
        return Ok((checkpoint, hub.repo.clone(), hub.url()));
    }
    let absolute = std::fs::canonicalize(model)?;
    let dir = if absolute.is_dir() {
        absolute.as_path()
    } else {
        absolute.parent().context("config has a parent directory")?
    };
    let name = dir.file_name().unwrap_or_default().to_string_lossy().into_owned();
    Ok((checkpoint, name, env!("CARGO_PKG_REPOSITORY").to_string()))
}

/// The voices of the catalog `source` names, each downloaded on first use: those in `lang` and
/// those with no language, or all of them when `lang` is `none`. A voice that cannot be fetched
/// is logged and skipped.
async fn catalog_voices(source: &str, lang: &str) -> Result<Vec<(String, std::path::PathBuf)>> {
    let (source, lang) = (source.to_string(), lang.to_string());
    // The blocking Hub client runs its own runtime, which cannot be started inside this one.
    tokio::task::spawn_blocking(move || {
        let catalog = VoiceCatalog::new(&source)?;
        let mut voices = vec![];
        for voice in catalog.voices()? {
            if lang != "none" && voice.language.as_deref().is_some_and(|l| l != lang) {
                continue;
            }
            match catalog.fetch(voice) {
                Ok(path) => voices.push((voice.name.clone(), path)),
                Err(e) => {
                    tracing::warn!(voice = %voice.name, error = %e, "failed to fetch catalog voice")
                }
            }
        }
        Ok(voices)
    })
    .await?
}

/// Load the model named by `model`. The source is required.
#[allow(clippy::too_many_arguments)]
pub async fn load_ptts(
    model: &str,
    voice_dir: Option<&std::path::PathBuf>,
    voice_catalog: Option<&str>,
    device: DeviceKind,
    quant: Quant,
    temperature: f32,
    seed_base: u64,
    normalize: Normalize,
    conditions: &[(String, String)],
    max_concurrent_requests: std::num::NonZeroUsize,
    compat: &Compat,
) -> Result<AppState> {
    anyhow::ensure!(
        max_concurrent_requests.get() <= tokio::sync::Semaphore::MAX_PERMITS,
        "--max-concurrent-requests must be at most {}",
        tokio::sync::Semaphore::MAX_PERMITS
    );
    quant.check_device(device)?;
    let (m, model_name, model_url) = locate(model, quant).await?;
    let mut extra_voices = if let Some(voice_dir) = voice_dir {
        let found = ptts::loader::voices_in(voice_dir);
        if found.is_empty() {
            tracing::warn!(?voice_dir, "no voice files found in --voice-dir");
        }
        found
    } else {
        Vec::new()
    };
    if let Some(catalog) = voice_catalog {
        match catalog_voices(catalog, normalize.as_str()).await {
            Ok(voices) => extra_voices.extend(voices),
            // Offline, a server that needs no catalog voice still starts.
            Err(e) if catalog == DEFAULT_VOICE_CATALOG => {
                tracing::warn!(error = %e, "the default voice catalog could not be read")
            }
            Err(e) => return Err(e.context(format!("voice catalog {catalog}"))),
        }
    }
    let frame_rate = m.config.mimi.frame_rate;
    let mut builder = m.builder(normalize).device(device).quant(quant).temperature(temperature);
    for (name, value) in conditions {
        builder = builder.condition(name, value);
    }
    let mut synth = builder.build()?;
    m.register_voices(&mut synth);
    // Additional user voices remain optional and override names from the checkpoint.
    for (name, path) in extra_voices {
        if let Err(e) = synth.add_voice_file(&name, &path) {
            tracing::warn!(voice = %name, error = %e, "failed to load additional voice embedding");
        }
    }

    let voices = synth.voices();
    let default_voice = synth.default_voice().context("no voice embeddings found in model")?;
    let aliases = Aliases::new(compat, &voices)?;
    let sample_rate = synth.sample_rate();
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
        requests: Arc::new(tokio::sync::Semaphore::new(max_concurrent_requests.get())),
        model_name,
        model_url,
        normalize,
        voices,
        aliases,
        ignore_speed: compat.ignore_speed,
        seed_base,
        sample_rate,
        frame_size,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_catalog_voices_served_are_those_in_the_language() {
        let dir = std::env::temp_dir().join(format!("ptts-server-catalog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("voices.jsonl"),
            r#"{"name": "Zoey", "filename": "voices/a.safetensors", "language": "en"}
{"name": "Annika", "filename": "voices/b.safetensors", "language": "de"}
{"name": "Mine", "filename": "voices/c.safetensors"}
"#,
        )
        .unwrap();
        let source = dir.to_str().unwrap();
        let names = |voices: Vec<(String, std::path::PathBuf)>| -> Vec<String> {
            voices.into_iter().map(|(name, _)| name).collect()
        };
        assert_eq!(names(catalog_voices(source, "en").await.unwrap()), ["Zoey", "Mine"]);
        assert_eq!(
            names(catalog_voices(source, "none").await.unwrap()),
            ["Zoey", "Annika", "Mine"]
        );
        let (_, path) = catalog_voices(source, "de").await.unwrap().remove(0);
        assert_eq!(path, dir.join("voices/b.safetensors"));
        assert!(catalog_voices(dir.join("missing").to_str().unwrap(), "en").await.is_err());
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL pointing to a private q8 checkpoint"]
    async fn private_checkpoint_loads_with_the_shared_default_voice() {
        let path = std::path::PathBuf::from(
            std::env::var("PTTS_TEST_MODEL").expect("set PTTS_TEST_MODEL"),
        );
        let checkpoint =
            Checkpoint::resolve(&path, ResolveOptions { quant: Quant::Q80, weights: None })
                .unwrap();
        let suffix =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let extra_dir =
            std::env::temp_dir().join(format!("ptts-extra-voices-{}-{suffix}", std::process::id()));
        std::fs::create_dir(&extra_dir).unwrap();
        std::fs::write(extra_dir.join("invalid.safetensors"), b"unreadable optional voice")
            .unwrap();
        let loaded = load_ptts(
            path.to_str().unwrap(),
            Some(&extra_dir),
            None,
            DeviceKind::Cpu,
            Quant::Q80,
            0.3,
            7,
            Normalize::OFF,
            &[],
            std::num::NonZeroUsize::new(1).unwrap(),
            &Default::default(),
        )
        .await;
        std::fs::remove_dir_all(extra_dir).unwrap();
        let state = loaded.unwrap();
        assert!(!state.voices.iter().any(|v| v == "invalid"));
        assert_eq!(state.sample_rate, checkpoint.config.mimi.sample_rate as u32);
        let audio = state.synth.say("Hello from Phonon.").unwrap();
        assert!(audio.len() > state.sample_rate as usize / 4);
        assert!(audio.iter().all(|x| x.is_finite()));
        assert!(audio.iter().any(|x| x.abs() > 0.01));
    }
}
