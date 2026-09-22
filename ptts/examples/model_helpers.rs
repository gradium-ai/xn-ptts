//! Bits the examples share that are not worth a place in the library.
//!
//! Locating a checkpoint used to live here: the repo it sits in, the names its files go by, the
//! voices it bundles and the config to assume when it ships none. That moved to
//! [`ptts::loader::ModelSource`], because a library user needs it as much as an example does --
//! reproducing `say.rs` meant re-deriving a layout the crate already knew.
//!
//! What is left is the two things that really are the frontend's: which repo to default to, and
//! how the examples want their logs.
#![allow(dead_code, unused_imports)]

pub use ptts::loader::{
    Checkpoint, ModelSource, is_unused_by_tts_model, load_voice_emb, load_weights, remap_key,
};
#[cfg(feature = "sp")]
pub use ptts::tok::Tok;

/// Hugging Face repo the examples download from unless `--repo` names another.
///
/// A deployment choice rather than a fact about the format, which is why it is here and not in
/// `ptts`: the library loads whatever repo it is pointed at.
pub const REPO_ID: &str = "kyutai/pocket-tts";

/// Default `tracing` directives for the examples: `info` for everything but the Hub download
/// stack. `hf_hub` transfers through the Xet backend, which reports every retry policy and
/// range probe at `info` -- a dozen lines per file that say nothing to a user waiting for a
/// download. `RUST_LOG` overrides this.
pub const LOG_DIRECTIVES: &str =
    "info,xet=warn,xet_client=warn,xet_data=warn,xet_runtime=warn,xet_core_structures=warn";
