//! Locating a checkpoint on disk or on the Hub, and loading it into a [`Synth`].
//!
//! Everything about *how* speech is generated lives in `ptts::synth`. What is
//! left here is deciding which files to load, which voices to register, and
//! holding the result for the request handlers.

use anyhow::{Context as _, Result};
use ptts::checkpoint::{Checkpoint, ResolveOptions};
use ptts::hub::HubSource;
use ptts::preprocess::{Normalize, Rules};
use ptts::synth::{DeviceKind, Quant, Synth};
use ptts::voice_catalog::{DEFAULT_VOICE_CATALOG, VoiceCatalog};
use std::sync::Arc;

/// The loaded model and the request defaults, shared by every connection.
#[derive(Clone)]
pub struct AppState(Arc<Inner>);

pub struct Inner {
    pub synth: Synth,
    pub requests: Arc<tokio::sync::Semaphore>,
    /// The checkpoint that loaded: its repo id, or for a local config the name of its folder.
    pub model_name: String,
    pub voices: Vec<String>,
    pub default_voice: String,
    /// What a session that sends no `lang` is normalized as.
    pub default_lang: Normalize,
    /// The rewrite rules every session's language is combined with.
    pub rules: Rules,
    pub max_seq_len: usize,
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
/// in one, with the name it is reported by.
async fn locate(model: &str, quant: Quant) -> Result<(Checkpoint, String)> {
    let hub = HubSource::parse(model)?;
    let source = model.to_string();
    // The blocking Hub client runs its own runtime, which cannot be started inside this one.
    let checkpoint = tokio::task::spawn_blocking(move || {
        Checkpoint::locate(&source, ResolveOptions { quant, weights: None })
    })
    .await??;
    if let Some(hub) = hub {
        return Ok((checkpoint, hub.repo));
    }
    let absolute = std::fs::canonicalize(model)?;
    let dir = if absolute.is_dir() {
        absolute.as_path()
    } else {
        absolute.parent().context("config has a parent directory")?
    };
    let name = dir.file_name().unwrap_or_default().to_string_lossy().into_owned();
    Ok((checkpoint, name))
}

/// The languages whose catalog voices a server offers. Sessions pick their language, so these are
/// the languages the checkpoint lists plus the server's own `--lang`, which keeps a single-language
/// checkpoint offering the voices of its `--lang`. `--lang none` offers them all (`None`).
fn catalog_languages(lang: &str, listed: &[String]) -> Option<Vec<String>> {
    if lang == "none" {
        return None;
    }
    let mut langs = listed.to_vec();
    if !langs.iter().any(|l| l == lang) {
        langs.push(lang.to_string());
    }
    Some(langs)
}

/// Adds the catalog's voices after those of `--voice-dir`, which win a name both have: a user's own
/// voice must not be replaced by a catalog voice that happens to share its name.
fn add_catalog_voices(
    extra_voices: &mut Vec<(String, std::path::PathBuf)>,
    catalog: Vec<(String, std::path::PathBuf)>,
) {
    for (name, path) in catalog {
        if extra_voices.iter().any(|(own, _)| *own == name) {
            tracing::warn!(voice = %name, "catalog voice skipped: --voice-dir has one of that name");
        } else {
            extra_voices.push((name, path));
        }
    }
}

/// The voices of the catalog `source` names, each downloaded on first use: those in one of
/// `langs` and those with no language, or all of them when `langs` is `None`. A voice that
/// cannot be fetched is logged and skipped.
async fn catalog_voices(
    source: &str,
    langs: Option<Vec<String>>,
) -> Result<Vec<(String, std::path::PathBuf)>> {
    let source = source.to_string();
    // The blocking Hub client runs its own runtime, which cannot be started inside this one.
    tokio::task::spawn_blocking(move || {
        let catalog = VoiceCatalog::new(&source)?;
        let mut voices = vec![];
        for voice in catalog.voices()? {
            let unwanted =
                |l: &str| langs.as_ref().is_some_and(|langs| !langs.iter().any(|x| x == l));
            if voice.language.as_deref().is_some_and(unwanted) {
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
    max_seq_len: usize,
    normalize: Normalize,
    rules: Rules,
    conditions: &[(String, String)],
    max_concurrent_requests: std::num::NonZeroUsize,
) -> Result<AppState> {
    anyhow::ensure!(
        max_concurrent_requests.get() <= tokio::sync::Semaphore::MAX_PERMITS,
        "--max-concurrent-requests must be at most {}",
        tokio::sync::Semaphore::MAX_PERMITS
    );
    quant.check_device(device)?;
    let (m, model_name) = locate(model, quant).await?;
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
        let langs = catalog_languages(normalize.as_str(), &m.config.languages);
        match catalog_voices(catalog, langs).await {
            Ok(voices) => add_catalog_voices(&mut extra_voices, voices),
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
        voices,
        default_voice,
        default_lang: normalize,
        rules,
        max_seq_len,
        seed_base,
        sample_rate,
        frame_size,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_voice_dir_voice_wins_a_name_the_catalog_also_has() {
        let voice = |name: &str, file: &str| (name.to_string(), std::path::PathBuf::from(file));
        let mut extra = vec![voice("Alba", "mine.safetensors")];
        add_catalog_voices(
            &mut extra,
            vec![voice("Alba", "catalog.safetensors"), voice("Zoey", "z")],
        );
        assert_eq!(extra, [voice("Alba", "mine.safetensors"), voice("Zoey", "z")]);
    }

    #[test]
    fn the_catalog_languages_are_the_checkpoints_plus_the_servers() {
        let listed = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let multi = listed(&["en", "fr", "de", "pt", "es"]);
        assert_eq!(catalog_languages("pt", &multi), Some(multi.clone()));
        assert_eq!(
            catalog_languages("ja", &multi),
            Some(listed(&["en", "fr", "de", "pt", "es", "ja"]))
        );
        assert_eq!(catalog_languages("fr", &listed(&["en"])), Some(listed(&["en", "fr"])));
        assert_eq!(catalog_languages("fr", &[]), Some(listed(&["fr"])));
        assert_eq!(catalog_languages("none", &multi), None);
    }

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
        let langs = |langs: &[&str]| Some(langs.iter().map(|l| l.to_string()).collect());
        assert_eq!(names(catalog_voices(source, langs(&["en"])).await.unwrap()), ["Zoey", "Mine"]);
        assert_eq!(
            names(catalog_voices(source, langs(&["en", "de"])).await.unwrap()),
            ["Zoey", "Annika", "Mine"]
        );
        assert_eq!(names(catalog_voices(source, None).await.unwrap()), ["Zoey", "Annika", "Mine"]);
        let (_, path) = catalog_voices(source, langs(&["de"])).await.unwrap().remove(0);
        assert_eq!(path, dir.join("voices/b.safetensors"));
        assert!(catalog_voices(dir.join("missing").to_str().unwrap(), None).await.is_err());
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL pointing to a private q8 checkpoint"]
    async fn private_checkpoint_loads_with_the_shared_default_voice() {
        let source = std::env::var("PTTS_TEST_MODEL").expect("set PTTS_TEST_MODEL");
        check_private_checkpoint(&source).await;
    }

    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL_DIR pointing to a local folder of the private q8 checkpoint"]
    async fn private_checkpoint_folder_loads_with_the_shared_default_voice() {
        let source = std::env::var("PTTS_TEST_MODEL_DIR").expect("set PTTS_TEST_MODEL_DIR");
        check_private_checkpoint(&source).await;
    }

    async fn check_private_checkpoint(source: &str) {
        let checkpoint =
            Checkpoint::locate(source, ResolveOptions { quant: Quant::Q80, weights: None })
                .unwrap();
        let suffix =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let extra_dir =
            std::env::temp_dir().join(format!("ptts-extra-voices-{}-{suffix}", std::process::id()));
        std::fs::create_dir(&extra_dir).unwrap();
        std::fs::write(extra_dir.join("invalid.safetensors"), b"unreadable optional voice")
            .unwrap();
        let loaded = load_ptts(
            source,
            Some(&extra_dir),
            None,
            DeviceKind::Cpu,
            Quant::Q80,
            0.3,
            7,
            1024,
            Normalize::OFF,
            Rules::DEFAULT,
            &[],
            std::num::NonZeroUsize::new(1).unwrap(),
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
