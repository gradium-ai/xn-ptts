//! Shared by the CLI and native examples.
//!
//! Checkpoint resolution, local or from the Hub, lives in `ptts::checkpoint`.
#![allow(dead_code, unused_imports)]

pub use ptts::checkpoint::{Checkpoint, DEFAULT_MODEL, ResolveOptions};
pub use ptts::loader::{is_unused_by_tts_model, load_voice_emb, load_weights};

/// Default tracing directives. `RUST_LOG` overrides these.
pub const LOG_DIRECTIVES: &str =
    "info,xet=warn,xet_client=warn,xet_data=warn,xet_runtime=warn,xet_core_structures=warn";

/// Help for the `--model` flag the CLI and examples share.
pub const MODEL_HELP: &str = "Model to load: a local directory holding config.json, weights, \
    tokenizer and voices, or a Hugging Face repo as hf://owner/name or \
    hf://owner/name@revision. Use a commit as the revision to reproduce a release. Defaults to \
    the default model, downloaded from the Hub and cached on first use.";
