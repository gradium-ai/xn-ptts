//! Checkpoints on the Hugging Face Hub, named `hf://owner/name` or `hf://owner/name@revision`.
//!
//! Parsing a name is always available. Downloading needs the `hub` feature, which is what
//! [`Checkpoint::locate`] uses for an `hf://` source. Files are downloaded to the Hugging Face
//! cache, or found there, and resolved exactly as a local checkpoint folder is: its own
//! `config.json`, weights, `tokenizer.json`, and voices under `voices/` or `embeddings/` plus an
//! optional `default-voice.safetensors`. Only the files that are used are downloaded.

use crate::checkpoint::{Checkpoint, ResolveOptions};
use crate::{Error, Result};

/// The prefix that marks a model source as a Hub repo rather than a local path.
pub const SCHEME: &str = "hf://";

/// A Hub model repo and an optional revision: a branch, tag or commit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HubSource {
    pub repo: String,
    pub revision: Option<String>,
}

impl HubSource {
    /// `Ok(None)` for a source without the `hf://` prefix, which names a local path.
    pub fn parse(source: &str) -> Result<Option<Self>> {
        let Some(rest) = source.strip_prefix(SCHEME) else { return Ok(None) };
        let (repo, revision) = match rest.split_once('@') {
            Some((repo, revision)) => (repo, Some(revision)),
            None => (rest, None),
        };
        let valid_repo = matches!(repo.split('/').collect::<Vec<_>>()[..], [owner, name]
            if !owner.is_empty() && !name.is_empty());
        if !valid_repo || revision.is_some_and(str::is_empty) {
            return Err(Error::InvalidArgument(format!(
                "`{source}` is not a Hugging Face model; expected {SCHEME}owner/name or \
                 {SCHEME}owner/name@revision"
            )));
        }
        Ok(Some(Self { repo: repo.to_string(), revision: revision.map(str::to_string) }))
    }

    /// The repo's page on the Hub.
    pub fn url(&self) -> String {
        format!("https://huggingface.co/{}", self.repo)
    }
}

impl std::fmt::Display for HubSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{SCHEME}{}", self.repo)?;
        match &self.revision {
            Some(revision) => write!(f, "@{revision}"),
            None => Ok(()),
        }
    }
}

#[cfg(not(feature = "hub"))]
pub(crate) fn resolve(source: &HubSource, _: ResolveOptions<'_>) -> Result<Checkpoint> {
    Err(Error::Unsupported(format!(
        "cannot download {source}: ptts was built without the `hub` feature"
    )))
}

#[cfg(not(feature = "hub"))]
pub(crate) fn download(source: &HubSource, _: &str) -> Result<std::path::PathBuf> {
    Err(Error::Unsupported(format!(
        "cannot download {source}: ptts was built without the `hub` feature"
    )))
}

/// Download one file of `source`, or find it in the cache.
#[cfg(feature = "hub")]
pub(crate) fn download(source: &HubSource, filename: &str) -> Result<std::path::PathBuf> {
    Repo::open(source)?.get(filename)
}

/// The client reads `HF_TOKEN`, `HF_ENDPOINT` and the cache location from the environment,
/// falling back to the token `hf auth login` stores.
#[cfg(feature = "hub")]
pub(crate) fn resolve(source: &HubSource, options: ResolveOptions<'_>) -> Result<Checkpoint> {
    use crate::checkpoint::{read_config, validate_relative_path, weight_candidates};
    use std::path::PathBuf;

    tracing::info!(
        repo = source.repo,
        revision = source.revision.as_deref().unwrap_or("main"),
        "resolving checkpoint on the Hugging Face Hub"
    );
    let repo = Repo::open(source)?;
    let config = read_config(repo.get("config.json")?)?;
    // One listing rather than a request per guessed name, and the only way to learn a repo's
    // voices. If listing is unavailable, only standard weight and default-voice paths are tried.
    let listing = repo.list().ok();
    // Without a listing every name is a guess, and one that was never cached fails offline with
    // a network error rather than a not-found. So it counts as absent, letting a later name the
    // cache does hold be found, and the first such error is reported only if no weights resolve.
    let first_err = std::cell::RefCell::new(None);
    let get_optional = |name: &str| -> Result<Option<PathBuf>> {
        if listing.as_ref().is_some_and(|files: &Vec<String>| !files.iter().any(|f| f == name)) {
            return Ok(None);
        }
        match repo.get(name) {
            Ok(path) => Ok(Some(path)),
            Err(Error::NotFound(_)) => Ok(None),
            Err(e) if listing.is_none() => {
                first_err.borrow_mut().get_or_insert(e);
                Ok(None)
            }
            Err(e) => Err(e),
        }
    };

    let weights = match options.weights {
        Some(name) => {
            validate_relative_path(name)?;
            repo.get(name)?
        }
        None => {
            let candidates = weight_candidates(options.quant);
            let mut found = None;
            for name in candidates {
                if let Some(path) = get_optional(name)? {
                    found = Some(path);
                    break;
                }
            }
            found.ok_or_else(|| {
                first_err.borrow_mut().take().unwrap_or_else(|| {
                    Error::NotFound(format!(
                        "no weights in {source}; expected one of {} or an explicit weights path",
                        candidates.join(", ")
                    ))
                })
            })?
        }
    };
    let tokenizer = repo.get("tokenizer.json")?;

    // `voices/` beats `embeddings/`, and both beat `default-voice.safetensors`, as in a local
    // folder. A voice that will not download is skipped, as one that will not load is.
    let mut voices: Vec<(String, PathBuf)> = vec![];
    let files = crate::loader::voice_paths(listing.clone().unwrap_or_default());
    let default = ("default".to_string(), crate::loader::DEFAULT_VOICE_FILE.to_string());
    for (name, file) in files.into_iter().chain([default]) {
        if !voices.iter().any(|(known, _)| *known == name)
            && let Ok(Some(path)) = get_optional(&file)
        {
            voices.push((name, path));
        }
    }
    voices.sort();
    Ok(Checkpoint { config, weights, tokenizer: Some(tokenizer), voices, quant: options.quant })
}

#[cfg(feature = "hub")]
struct Repo<'a> {
    repo: hf_hub::HFRepositorySync<hf_hub::repository::RepoTypeModel>,
    source: &'a HubSource,
}

#[cfg(feature = "hub")]
impl<'a> Repo<'a> {
    fn open(source: &'a HubSource) -> Result<Self> {
        // Not `NotFound`: a client that will not start is an environment failure, not a name
        // that failed to resolve.
        let client = hf_hub::HFClientSync::new().map_err(|e| {
            Error::Io(std::io::Error::other(format!("cannot reach the Hugging Face Hub: {e}")))
        })?;
        let (owner, name) = hf_hub::split_id(&source.repo);
        Ok(Self { repo: client.model(owner, name), source })
    }

    /// Download `filename`, or find it in the local cache.
    fn get(&self, filename: &str) -> Result<std::path::PathBuf> {
        // Only a genuine not-found is a name that failed to resolve. Offline, a timeout, a 429
        // or a 5xx are environment failures, which callers may retry. `hf_hub` names the file
        // and repo for a missing file but not for an HTTP or authentication failure, which
        // makes a gated repo hard to diagnose.
        self.repo
            .download_file()
            .filename(filename)
            .maybe_revision(self.source.revision.clone())
            .send()
            .map_err(|e| match e {
                hf_hub::HFError::EntryNotFound { .. }
                | hf_hub::HFError::LocalEntryNotFound { .. } => {
                    Error::NotFound(format!("`{filename}` is not in {}: {e}", self.source))
                }
                other => Error::Io(std::io::Error::other(format!(
                    "cannot fetch `{filename}` from {}: {other}. If the repo is gated, accept \
                     its terms on huggingface.co and run `hf auth login` or set HF_TOKEN.",
                    self.source
                ))),
            })
    }

    fn list(&self) -> Result<Vec<String>> {
        let entries = self
            .repo
            .list_tree()
            .maybe_revision(self.source.revision.clone())
            .recursive(true)
            .send()
            .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
        Ok(entries
            .into_iter()
            .filter_map(|entry| match entry {
                hf_hub::repository::files::RepoTreeEntry::File { path, .. } => Some(path),
                _ => None,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_sources_parse() {
        assert_eq!(HubSource::parse("/models/phonon").unwrap(), None);
        assert_eq!(HubSource::parse("owner/name").unwrap(), None);
        let source = HubSource::parse("hf://owner/name").unwrap().unwrap();
        assert_eq!(source, HubSource { repo: "owner/name".into(), revision: None });
        assert_eq!(source.to_string(), "hf://owner/name");
        let source = HubSource::parse("hf://owner/name@refs/pr/1").unwrap().unwrap();
        assert_eq!(source.revision.as_deref(), Some("refs/pr/1"));
        assert_eq!(source.to_string(), "hf://owner/name@refs/pr/1");
        assert_eq!(source.url(), "https://huggingface.co/owner/name");
        for bad in ["hf://", "hf://name", "hf://owner/", "hf://a/b/c", "hf://owner/name@"] {
            assert!(
                matches!(HubSource::parse(bad), Err(Error::InvalidArgument(_))),
                "{bad} should not parse"
            );
        }
    }
}
