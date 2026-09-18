use anyhow::{Context as _, Result};
use hf_hub::HFClientSync;
use hf_hub::repository::RepoTypeModel;
use std::path::PathBuf;

/// Thin wrapper around an `hf_hub` model repository.
///
/// The point of the wrapper is error context: `hf_hub` names the file and the
/// repo when a file is missing, but an HTTP or authentication failure says
/// neither, which makes server logs hard to act on. Every fallible call here is
/// annotated with the repo id and the filename that was being fetched.
///
/// We only ever talk to model repos, so the repo type is hard-coded.
pub struct HfRepo {
    repo: hf_hub::HFRepositorySync<RepoTypeModel>,
    repo_id: String,
}

impl HfRepo {
    /// Open the model repo `repo_id` (e.g. `"kyutai/pocket-tts"`) on the Hub.
    /// The client reads `HF_TOKEN`, `HF_ENDPOINT` and the cache location from
    /// the environment.
    pub fn model(repo_id: &str) -> Result<Self> {
        let client =
            HFClientSync::new().context("failed to initialize the Hugging Face Hub client")?;
        let (owner, name) = hf_hub::split_id(repo_id);
        Ok(Self { repo: client.model(owner, name), repo_id: repo_id.to_string() })
    }

    /// Download `filename` (or fetch it from the local cache), returning its
    /// path on disk. On failure the error names the repo and the file.
    pub fn get(&self, filename: &str) -> Result<PathBuf> {
        self.repo.download_file().filename(filename).send().with_context(|| {
            format!("failed to fetch `{filename}` from model repo `{}`", self.repo_id)
        })
    }
}
