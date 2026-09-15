//! Bits the examples share that are not worth a place in the library.
//!
//! Two kinds of thing live here. The first is re-exports: weight loading, key remapping and
//! voice-embedding loading live in `ptts::loader` and the tokenizer in `ptts::tok`, and this
//! gives each example one place to look. `ptts::plan` is used directly, since it is not tied to
//! a backend type.
//!
//! The second is everything specific to the *published* pocket-tts checkpoint: the repo it
//! lives in, the names its files go by, the voices it bundles, and the config to assume when a
//! directory ships none. That set changes with every release, so `ptts` does not carry it --
//! the library reads the config, weights, tokenizer and voice files it is handed, and finding
//! them belongs to whatever tracks a particular checkpoint. For the examples, that is here.
#![allow(dead_code, unused_imports)]

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub use ptts::loader::{is_unused_by_tts_model, load_voice_emb, load_weights, remap_key};
#[cfg(feature = "sp")]
pub use ptts::tok::Tok;

use ptts::synth::{Synth, SynthBuilder};
use ptts::tts_model::TTSConfig;

/// Hugging Face repo holding the published checkpoint.
pub const REPO_ID: &str = "kyutai/pocket-tts";

/// Voices the published checkpoint bundles, under `embeddings/`.
pub const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

/// Weight file names tried, in order. `tts_b6369a24.safetensors` is what the published repo
/// calls its f32 weights today; a local directory more often holds one of the first two.
pub const WEIGHT_CANDIDATES: &[&str] =
    &["model.safetensors", "model.q8.gguf", "tts_b6369a24.safetensors"];

/// Tokenizer file names tried, in order. `tokenizer.json` is the Hugging Face `tokenizers`
/// format, `tokenizer.model` is SentencePiece, and `ptts::tok::Tok` picks the reader by
/// extension -- so the order follows which of the two this build can read. A checkpoint that
/// ships both is common, and picking the one whose feature is off would fail a load that had
/// a usable tokenizer sitting beside it.
pub const TOKENIZER_CANDIDATES: &[&str] = if cfg!(feature = "hf") {
    &["tokenizer.json", "tokenizer.model"]
} else {
    &["tokenizer.model", "tokenizer.json"]
};

/// A checkpoint whose files have been located and whose config is parsed.
pub struct Checkpoint {
    pub config: TTSConfig,
    pub weights: PathBuf,
    /// `None` when no tokenizer file was found; the caller must then pass one to
    /// [`SynthBuilder::tokenizer`].
    pub tokenizer: Option<PathBuf>,
    /// Voice name to embedding file, sorted by name.
    pub voices: Vec<(String, PathBuf)>,
}

impl Checkpoint {
    /// A local model directory, or the published repo on the Hub when `dir` is `None`.
    pub fn locate(dir: Option<&Path>) -> Result<Self> {
        match dir {
            Some(dir) => Self::from_dir(dir),
            None => Self::from_hub(REPO_ID),
        }
    }

    /// Download from a Hugging Face model repo laid out like [`REPO_ID`].
    pub fn from_hub(repo_id: &str) -> Result<Self> {
        let repo = HubRepo::open(repo_id)?;
        tracing::info!(?repo_id, "resolving checkpoint on the Hugging Face Hub");

        let config = match repo.get_optional("config.json") {
            Some(path) => read_config(&path)?,
            None => shipped_config(),
        };
        let weights = match WEIGHT_CANDIDATES.iter().find_map(|name| repo.get_optional(name)) {
            Some(path) => path,
            None => anyhow::bail!(
                "no weights file in `{repo_id}`; expected one of {}",
                WEIGHT_CANDIDATES.join(", ")
            ),
        };
        let tokenizer = TOKENIZER_CANDIDATES.iter().find_map(|name| repo.get_optional(name));

        let mut voices = vec![];
        for voice in VOICES {
            if let Some(path) = repo.get_optional(&format!("embeddings/{voice}.safetensors")) {
                voices.push((voice.to_string(), path));
            }
        }
        if let Some(path) = repo.get_optional("default-voice.safetensors") {
            voices.push(("default".to_string(), path));
        }
        voices.sort();

        Ok(Self { config, weights, tokenizer, voices })
    }

    /// A local directory holding `config.json`, a weights file, a tokenizer and an optional
    /// `voices/` or `embeddings/` subdirectory -- both layouts are in circulation.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        if !dir.is_dir() {
            anyhow::bail!("not a directory: {}", dir.display())
        }
        let config_path = dir.join("config.json");
        let config =
            if config_path.is_file() { read_config(&config_path)? } else { shipped_config() };

        let weights = WEIGHT_CANDIDATES
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.is_file())
            .with_context(|| {
                format!(
                    "no weights file in {}; expected one of {}",
                    dir.display(),
                    WEIGHT_CANDIDATES.join(", ")
                )
            })?;
        let tokenizer =
            TOKENIZER_CANDIDATES.iter().map(|name| dir.join(name)).find(|path| path.is_file());

        let mut voices = vec![];
        for sub in ["voices", "embeddings"] {
            collect_voice_dir(&dir.join(sub), &mut voices);
        }
        let default_voice = dir.join("default-voice.safetensors");
        if default_voice.is_file() {
            voices.push(("default".to_string(), default_voice));
        }
        voices.sort();
        voices.dedup_by(|a, b| a.0 == b.0);

        Ok(Self { config, weights, tokenizer, voices })
    }

    /// A builder over this checkpoint, with its tokenizer file set.
    ///
    /// The bundled voices are deliberately not registered here: see
    /// [`Self::register_voices`].
    pub fn builder(&self) -> SynthBuilder {
        let mut builder = SynthBuilder::new(self.config.clone(), &self.weights);
        if let Some(tokenizer) = self.tokenizer.as_ref() {
            builder = builder.tokenizer_file(tokenizer);
        }
        builder
    }

    /// Register the bundled voices, warning about any that fail to load rather than failing
    /// the run: one bad embedding -- an interrupted download, a voice from another
    /// checkpoint -- should not make the model unusable. A voice the user named explicitly
    /// goes through `SynthBuilder::add_voice`, where a failure is fatal.
    pub fn register_voices(&self, tts: &mut Synth) {
        for (name, path) in self.voices.iter() {
            if let Err(e) = tts.add_voice_file(name, path) {
                tracing::warn!(voice = %name, error = %e, "skipping voice embedding");
            }
        }
    }
}

/// The config the published checkpoint ships, for repos and directories that carry no
/// `config.json`. `temp` is not read by the runtime -- sampling temperature reaches the model
/// through `SynthBuilder::temperature` -- so any value does.
fn shipped_config() -> TTSConfig {
    TTSConfig::v202601(0.7)
}

fn read_config(path: &Path) -> Result<TTSConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read config {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse config {}", path.display()))
}

/// Adds every `*.safetensors` file in `dir` to `voices`, keyed by file stem. A missing or
/// unreadable directory is not an error: voices are optional.
fn collect_voice_dir(dir: &Path, voices: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
            voices.push((name.to_string(), path));
        }
    }
}

/// A Hugging Face model repo, wrapped so a download failure names the repo, the file and the
/// URL -- `hf_hub`'s own errors mention none of the three, which makes a gated repo or a
/// renamed file hard to diagnose.
struct HubRepo {
    repo: hf_hub::api::sync::ApiRepo,
    repo_id: String,
}

impl HubRepo {
    fn open(repo_id: &str) -> Result<Self> {
        use hf_hub::{Repo, RepoType, api::sync::Api};
        let api = Api::new().context("cannot reach the Hugging Face Hub")?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
        Ok(Self { repo, repo_id: repo_id.to_string() })
    }

    fn get(&self, filename: &str) -> Result<PathBuf> {
        self.repo.get(filename).map_err(|e| {
            let url = self.repo.url(filename);
            anyhow::anyhow!(
                "failed to fetch `{filename}` from `{}` ({url}): {e}\n\
                 If the repo is gated, accept its terms on huggingface.co and run \
                 `huggingface-cli login` (or set HF_TOKEN).",
                self.repo_id
            )
        })
    }

    /// Like [`Self::get`] but maps any failure to `None`, for files that may legitimately be
    /// absent from a given repo layout.
    fn get_optional(&self, filename: &str) -> Option<PathBuf> {
        self.repo.get(filename).ok()
    }
}
