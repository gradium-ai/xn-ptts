//! Voice catalogs: voices listed in a JSONL file, each downloaded only when it is used.
//!
//! A catalog is a `voices.jsonl` with one object per line: `name` and `filename`, the voice
//! file's path relative to the catalog, and optionally `voice_id`, `language` and
//! `description`. It lives in a local folder or a Hugging Face repo. Frontends fall back to
//! [`DEFAULT_VOICES`] for a voice their checkpoint does not have, through
//! [`VoiceCatalog::register`]: only the catalog itself and the one voice file are fetched.

use crate::hub::HubSource;
use crate::synth::Synth;
use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The catalog frontends look a voice up in when their checkpoint has none of that name,
/// pinned to a commit so a default install is reproducible. Loading it needs the `hub` feature.
pub const DEFAULT_VOICES: &str = "hf://gr4d/phonon-voices@1c006b05c58b0206f6c8ef7c99a2a60213f22888";

/// The catalog file in a catalog folder or repo.
pub const CATALOG_FILE: &str = "voices.jsonl";

/// One voice of a catalog.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct CatalogVoice {
    pub name: String,
    /// The voice file, relative to the catalog.
    pub filename: String,
    pub voice_id: Option<String>,
    pub language: Option<String>,
    pub description: Option<String>,
}

/// A catalog of voices, read on first use.
#[derive(Debug)]
pub struct VoiceCatalog {
    source: Source,
    voices: OnceLock<Vec<CatalogVoice>>,
}

#[derive(Debug)]
enum Source {
    /// The catalog file.
    Local(PathBuf),
    Hub(HubSource),
}

impl VoiceCatalog {
    /// A catalog in a Hugging Face repo, as `hf://owner/name[@revision]` with a `voices.jsonl`
    /// at its root, or a local `voices.jsonl` or folder holding one. Nothing is read until the
    /// voices are needed, so this succeeds offline.
    pub fn new(source: &str) -> Result<Self> {
        let source = match HubSource::parse(source)? {
            Some(hub) => Source::Hub(hub),
            None => {
                let path = PathBuf::from(source);
                Source::Local(if path.is_dir() { path.join(CATALOG_FILE) } else { path })
            }
        };
        Ok(Self { source, voices: OnceLock::new() })
    }

    /// The catalog's voices, in file order. The first call reads or downloads the catalog.
    pub fn voices(&self) -> Result<&[CatalogVoice]> {
        if let Some(voices) = self.voices.get() {
            return Ok(voices);
        }
        let path = match &self.source {
            Source::Local(path) => path.clone(),
            Source::Hub(hub) => crate::hub::download(hub, CATALOG_FILE)?,
        };
        let voices = parse(&path)?;
        Ok(self.voices.get_or_init(|| voices))
    }

    /// The voice `query` names: its name in any case and with or without accents, so `kaua`
    /// finds `Kauã`, or its exact `voice_id`.
    pub fn find(&self, query: &str) -> Result<Option<&CatalogVoice>> {
        let folded = fold(query);
        Ok(self
            .voices()?
            .iter()
            .find(|v| fold(&v.name) == folded || v.voice_id.as_deref() == Some(query)))
    }

    /// The local path of `voice`'s file, downloaded on first use.
    pub fn fetch(&self, voice: &CatalogVoice) -> Result<PathBuf> {
        let file = &voice.filename;
        if file.is_empty()
            || file.contains(['\\', ':'])
            || file.starts_with('/')
            || file.split('/').any(|p| p.is_empty() || p == "." || p == "..")
        {
            return Err(Error::invalid_data(format!(
                "voice '{}' names the file '{file}', which is not a path inside the catalog",
                voice.name
            )));
        }
        match &self.source {
            Source::Local(path) => Ok(path.parent().unwrap_or(Path::new("")).join(file)),
            Source::Hub(hub) => crate::hub::download(hub, file),
        }
    }

    /// The voice name to pass to `synth` for `query`. A voice `synth` already has wins,
    /// matched as [`Self::find`] matches; otherwise the catalog voice `query` names is
    /// downloaded and registered under its name. A name neither knows comes back unchanged, for
    /// `synth` to report with the voices it has.
    pub fn register(&self, synth: &mut Synth, query: &str) -> Result<String> {
        let known = synth.voices();
        let folded = fold(query);
        if let Some(name) =
            known.iter().find(|v| *v == query).or_else(|| known.iter().find(|v| fold(v) == folded))
        {
            return Ok(name.clone());
        }
        let Some(voice) = self.find(query)? else { return Ok(query.to_string()) };
        if !known.contains(&voice.name) {
            let path = self.fetch(voice)?;
            synth.add_voice_file(&voice.name, &path)?;
        }
        Ok(voice.name.clone())
    }
}

fn parse(path: &Path) -> Result<Vec<CatalogVoice>> {
    let text = std::fs::read_to_string(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            Error::not_found(format!("no voice catalog at {}", path.display()))
        }
        _ => Error::Io(e),
    })?;
    let mut voices = vec![];
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let voice = serde_json::from_str(line)
            .map_err(|e| Error::invalid_data(format!("{}:{}: {e}", path.display(), i + 1)))?;
        voices.push(voice);
    }
    Ok(voices)
}

/// `name` as voice lookups compare it: lower case, with accents dropped.
fn fold(name: &str) -> String {
    name.chars()
        .flat_map(char::to_lowercase)
        .map(|c| match c {
            'à'..='å' => 'a',
            'ç' => 'c',
            'è'..='ë' => 'e',
            'ì'..='ï' => 'i',
            'ñ' => 'n',
            'ò'..='ö' | 'ø' => 'o',
            'ù'..='ü' => 'u',
            'ý' | 'ÿ' => 'y',
            c => crate::preprocess::unaccented(c).unwrap_or(c),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CATALOG: &str = r#"{"filename": "voices/a1.safetensors", "name": "Zoey", "voice_id": "a1", "language": "en", "locale": "en-us"}

{"filename": "voices/b2.safetensors", "name": "Kauã", "voice_id": "b2", "language": "pt", "description": "Calm"}
{"filename": "voices/c3.safetensors", "name": "Lucía-Sol"}
"#;

    fn catalog(text: &str) -> (VoiceCatalog, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "ptts-voice-catalog-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(CATALOG_FILE), text).unwrap();
        (VoiceCatalog::new(dir.to_str().unwrap()).unwrap(), dir)
    }

    #[test]
    fn a_catalog_reads_its_voices_and_ignores_unknown_fields() {
        let (catalog, _) = catalog(CATALOG);
        let voices = catalog.voices().unwrap();
        assert_eq!(voices.len(), 3);
        assert_eq!(voices[1].name, "Kauã");
        assert_eq!(voices[1].description.as_deref(), Some("Calm"));
        assert_eq!(voices[2].voice_id, None);
    }

    #[test]
    fn names_match_in_any_case_and_without_accents() {
        let (catalog, _) = catalog(CATALOG);
        let name = |q: &str| catalog.find(q).unwrap().map(|v| v.name.clone());
        assert_eq!(name("Kauã").as_deref(), Some("Kauã"));
        assert_eq!(name("kaua").as_deref(), Some("Kauã"));
        assert_eq!(name("KAUA").as_deref(), Some("Kauã"));
        assert_eq!(name("lucia-sol").as_deref(), Some("Lucía-Sol"));
        assert_eq!(name("zoey").as_deref(), Some("Zoey"));
        assert_eq!(name("nobody"), None);
    }

    #[test]
    fn voice_ids_match_exactly() {
        let (catalog, _) = catalog(CATALOG);
        assert_eq!(catalog.find("b2").unwrap().unwrap().name, "Kauã");
        assert_eq!(catalog.find("B2").unwrap(), None);
    }

    #[test]
    fn a_local_voice_file_is_found_beside_the_catalog() {
        let (catalog, dir) = catalog(CATALOG);
        let zoey = catalog.find("zoey").unwrap().unwrap().clone();
        assert_eq!(catalog.fetch(&zoey).unwrap(), dir.join("voices/a1.safetensors"));
    }

    #[test]
    fn a_file_outside_the_catalog_is_refused() {
        let (catalog, _) = catalog(r#"{"filename": "../secret.safetensors", "name": "X"}"#);
        let x = catalog.voices().unwrap()[0].clone();
        assert!(matches!(catalog.fetch(&x), Err(Error::InvalidData(_))));
    }

    #[test]
    fn a_bad_line_is_reported_with_its_number() {
        let (catalog, _) =
            catalog("{\"name\": \"A\", \"filename\": \"a.safetensors\"}\nnot json\n");
        let err = catalog.voices().unwrap_err().to_string();
        assert!(err.contains(":2:"), "{err}");
    }

    #[test]
    fn a_missing_catalog_is_not_found() {
        let catalog = VoiceCatalog::new("/nonexistent/voices.jsonl").unwrap();
        assert!(matches!(catalog.voices(), Err(Error::NotFound(_))));
    }

    #[test]
    fn the_default_catalog_is_a_pinned_hub_repo() {
        let hub = HubSource::parse(DEFAULT_VOICES).unwrap().unwrap();
        assert_eq!(hub.revision.map(|r| r.len()), Some(40));
    }
}
