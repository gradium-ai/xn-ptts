//! The error type the API layer returns.
//!
//! [`crate::synth`] and [`crate::loader`] return [`Error`]; the model modules below them
//! ([`crate::tts_model`], [`crate::flow_lm`], [`crate::mimi`] and friends) keep `xn::Result`,
//! because a shape mismatch inside the codec is not something a caller acts on. `?` crosses the
//! boundary in both directions: [`Error`] is `From<xn::Error>`, and `xn::Error` is
//! `From<Error>`, so a frontend whose own functions return `xn::Result` keeps compiling.
//!
//! # For binding authors
//!
//! Three of this repo's frontends are bindings, and each has to turn a failure into whatever its
//! host language calls an exception. Matching twenty-odd variants to do that is the wrong shape,
//! so [`Error::kind`] classifies each one into an [`ErrorKind`] -- the distinctions a program
//! acts on, rather than the distinctions a message draws:
//!
//! ```
//! use ptts::{Error, ErrorKind};
//!
//! fn describe(e: &Error) -> &'static str {
//!     match e.kind() {
//!         ErrorKind::InvalidArgument => "ValueError",
//!         ErrorKind::NotFound => "LookupError",
//!         ErrorKind::Unsupported => "NotImplementedError",
//!         ErrorKind::PermissionDenied => "PermissionError",
//!         ErrorKind::Network => "ConnectionError",
//!         ErrorKind::InvalidData | ErrorKind::Busy | ErrorKind::Internal => "RuntimeError",
//!         // `ErrorKind` is `#[non_exhaustive]` too, so a kind added later falls here rather
//!         // than being silently folded into one of the above.
//!         _ => "RuntimeError",
//!     }
//! }
//! ```
//!
//! Reach past `kind` for a variant when there is something specific to do with its fields --
//! [`Error::UnknownVoice`] carries the names that would have worked, and
//! [`Error::SeqBudgetExceeded`] carries the budget to ask for next time.

use crate::synth::{DeviceKind, Quant};
use std::path::PathBuf;

/// A [`Result`](std::result::Result) over [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// What went wrong, as a class rather than as a message.
///
/// The point of this type is that it is small and stable: a binding maps it once onto its host
/// language's exceptions and does not change when a variant is added to [`Error`]. See the
/// module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The caller passed something this call cannot accept: an unparseable name, a mis-shaped
    /// array, an empty text, a duration out of range.
    InvalidArgument,
    /// Something named does not exist: a voice that is not registered, a file that is not on
    /// disk, a checkpoint whose weights could not be located.
    NotFound,
    /// The request is well formed but this build or this checkpoint cannot serve it: a backend
    /// that was not compiled in, quantization on a GPU, cloning without a speaker encoder.
    Unsupported,
    /// Access was refused. A gated Hugging Face repo, in practice.
    PermissionDenied,
    /// A network call failed.
    Network,
    /// A file was found and is not what it claims to be: a checkpoint whose config does not
    /// match its weights, an unreadable voice embedding.
    InvalidData,
    /// The resource is in use and the call would corrupt it. Retrying later works.
    Busy,
    /// A failure inside the library or the tensor layer, including a panicked worker. Not
    /// something the caller can fix by calling differently.
    Internal,
}

/// Everything the `ptts` API layer can fail with.
///
/// `#[non_exhaustive]`: variants get added as the API grows, so match with a `_` arm, or match on
/// [`Self::kind`] instead.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    // --- the caller named something that is not there ---
    /// A voice was named that is not registered.
    #[error("no voice '{name}' is registered; this model has {}", known.join(", "))]
    UnknownVoice { name: String, known: Vec<String> },

    /// A generation was started with no voice, on a model that has some.
    #[error("no voice selected; this model has {}", known.join(", "))]
    NoVoiceSelected { known: Vec<String> },

    /// [`DeviceKind::parse`] was given a name it does not know.
    #[error("unknown device '{name}'; expected auto, cpu, cuda, vulkan or metal")]
    UnknownDevice { name: String },

    /// [`Quant::parse`] was given a name it does not know.
    #[error(
        "unsupported quantization '{name}'; expected one of \
         f32, q8_0, q8_1, q8k, q6k, q5_0, q5_1, q5k, q4_0, q4_1, q4k"
    )]
    UnknownQuant { name: String },

    // --- the caller's input is out of range or the wrong shape ---
    /// A conditioning embedding did not hold `frames * dim` values.
    #[error("embedding has {got} values, expected {frames} x {dim}")]
    EmbeddingShape { got: usize, frames: usize, dim: usize },

    /// The text, or the tokens it produced, held nothing to say.
    #[error("nothing to synthesize: the input produced no tokens")]
    NothingToSynthesize,

    /// A voice prompt was shorter than the checkpoint's minimum. Longer prompts are trimmed
    /// rather than rejected, so there is no matching too-long case.
    #[error("voice prompt is {got_s:.2}s, need at least {min_s:.2}s")]
    VoicePromptTooShort { got_s: f32, min_s: f32 },

    /// More KV slots were needed than the session was primed with. Build the session with
    /// `max_seq_len` at least `needed`, or split the text.
    #[error(
        "needs a KV budget of {needed} but the session was primed with {budget}; build the \
         session with a larger max_seq_len, or split the text"
    )]
    SeqBudgetExceeded { needed: usize, budget: usize },

    // --- this build, or this checkpoint, cannot do it ---
    /// A device was asked for that this build was not compiled with.
    #[error("this build has no {device} support; rebuild with the `{feature}` feature")]
    BackendUnavailable { device: DeviceKind, feature: &'static str },

    /// Quantized weights were asked for on a non-CPU device. Quantization is CPU-only.
    #[error("quantization ({quant}) is CPU-only, but the selected device is {device}")]
    QuantUnsupportedOnDevice { quant: Quant, device: DeviceKind },

    /// Cloning was asked for on a checkpoint that ships no speaker encoder. See
    /// `SynthOf::supports_voice_cloning`.
    #[error("this checkpoint has no speaker encoder, so it cannot clone voices")]
    VoiceCloningUnsupported,

    /// No tokenizer was supplied and none could be built: the checkpoint shipped no tokenizer
    /// file, or neither the `sp` nor the `hf` feature is enabled.
    #[error(
        "no tokenizer available: none was passed to SynthBuilder::tokenizer, the checkpoint \
         shipped none, and neither the `sp` nor the `hf` feature of `ptts` is enabled"
    )]
    NoTokenizer,

    /// Loading from the Hugging Face Hub was asked for in a build without the `hub` feature.
    #[error(
        "cannot load `{repo_id}`: reading from the Hugging Face Hub needs the `hub` feature of \
         the `ptts` crate. Either enable it, or download the checkpoint yourself and use \
         `ModelSource::dir`."
    )]
    HubFeatureDisabled { repo_id: String },

    /// Classifier-free guidance on this checkpoint conditions its null branch on silence, which
    /// needs the voice's source audio -- so a voice registered from a precomputed embedding
    /// cannot be used with CFG.
    #[error(
        "this model conditions its CFG null branch on silence (cfg_null_audio_empty=false), \
         which needs the voice's source audio. Register the voice with add_voice_from_pcm \
         instead of a precomputed embedding, or disable CFG."
    )]
    CfgNeedsSourceAudio,

    // --- state ---
    /// A second generation was started on a session while the first was still running. A session
    /// runs one at a time; finish or drop the [`crate::synth::SpeechStream`], or build a second
    /// session.
    #[error(
        "a generation is already in flight on this session; finish or drop that SpeechStream \
         first, or build a second session"
    )]
    GenerationInProgress,

    /// A generation worker thread panicked. Whatever audio arrived before it is incomplete.
    #[error("a generation worker panicked; the audio is incomplete")]
    WorkerPanicked,

    // --- locating and reading a checkpoint ---
    /// A checkpoint, or a file inside one, could not be located. The message names what was
    /// looked for and where, and on the Hub what the repo does hold.
    #[error("{detail}")]
    CheckpointNotFound { detail: String },

    /// A checkpoint was found but cannot be read as one: an unparseable config, a voice
    /// embedding of the wrong shape or from another model.
    #[error("{detail}")]
    Checkpoint { detail: String },

    /// The weights hold a different number of transformer layers than the config describes.
    /// Almost always a checkpoint that ships no `config.json`, loaded against the assumed one.
    #[error(
        "{} holds {found} transformer layers but the config describes {expected}. This \
         checkpoint needs its own `config.json`; the one assumed for a checkpoint that ships \
         none describes a {expected}-layer model.",
        path.display()
    )]
    LayerCountMismatch { path: PathBuf, found: usize, expected: usize },

    /// A Hugging Face repo refused the download because it is gated.
    #[error(
        "`{repo_id}` is gated, so `{filename}` cannot be fetched. Accept its terms at \
         https://huggingface.co/{repo_id}, then authenticate with `huggingface-cli login` or by \
         setting HF_TOKEN."
    )]
    Gated { repo_id: String, filename: String },

    /// A Hugging Face request failed for any other reason: no such repo or revision, or the
    /// network.
    #[error("failed to fetch {what} from `{repo_id}`: {source}")]
    Hub {
        repo_id: String,
        what: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A file could not be read or written.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// A failure from the tensor layer, or from anything below the API layer that reports
    /// through it.
    #[error(transparent)]
    Tensor(#[from] xn::Error),
}

impl Error {
    /// The class of this failure. See [`ErrorKind`] and the module documentation.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::UnknownDevice { .. }
            | Self::UnknownQuant { .. }
            | Self::EmbeddingShape { .. }
            | Self::NothingToSynthesize
            | Self::VoicePromptTooShort { .. }
            | Self::SeqBudgetExceeded { .. } => ErrorKind::InvalidArgument,

            Self::UnknownVoice { .. }
            | Self::NoVoiceSelected { .. }
            | Self::CheckpointNotFound { .. } => ErrorKind::NotFound,

            Self::BackendUnavailable { .. }
            | Self::QuantUnsupportedOnDevice { .. }
            | Self::VoiceCloningUnsupported
            | Self::NoTokenizer
            | Self::HubFeatureDisabled { .. }
            | Self::CfgNeedsSourceAudio => ErrorKind::Unsupported,

            Self::Gated { .. } => ErrorKind::PermissionDenied,
            Self::Hub { .. } => ErrorKind::Network,

            Self::Checkpoint { .. } | Self::LayerCountMismatch { .. } => ErrorKind::InvalidData,

            Self::GenerationInProgress => ErrorKind::Busy,

            // A missing file is a missing file however it was reached, so the io kind decides
            // rather than the fact that it arrived as io::Error.
            Self::Io(e) => match e.kind() {
                std::io::ErrorKind::NotFound => ErrorKind::NotFound,
                std::io::ErrorKind::PermissionDenied => ErrorKind::PermissionDenied,
                _ => ErrorKind::Internal,
            },

            Self::WorkerPanicked | Self::Tensor(_) => ErrorKind::Internal,
        }
    }

    /// A checkpoint or one of its files could not be located.
    pub(crate) fn not_found(detail: impl std::fmt::Display) -> Self {
        Self::CheckpointNotFound { detail: detail.to_string() }
    }

    /// A checkpoint was found and is not readable as one.
    pub(crate) fn checkpoint(detail: impl std::fmt::Display) -> Self {
        Self::Checkpoint { detail: detail.to_string() }
    }
}

/// Lets a frontend whose own functions return `xn::Result` keep using `?` on this crate.
///
/// The variant is flattened to its message, so this direction loses the classification. Going
/// the other way -- `Error::Tensor` -- keeps the tensor error whole.
impl From<Error> for xn::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Tensor(e) => e,
            Error::Io(e) => xn::Error::Io(e),
            other => xn::Error::msg(other.to_string()),
        }
    }
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Auto => "auto",
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::Vulkan => "Vulkan",
            Self::Metal => "Metal",
        };
        f.write_str(name)
    }
}

impl std::fmt::Display for Quant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tensor_error_survives_the_round_trip_to_xn_and_back() {
        // The bridge exists so a frontend on `xn::Result` keeps compiling. A tensor error must
        // not pick up a layer of message wrapping on the way through.
        let inner = xn::Error::msg("device out of memory");
        let there: xn::Error = Error::Tensor(inner).into();
        assert_eq!(there.to_string(), "device out of memory");
        assert_eq!(Error::from(there).kind(), ErrorKind::Internal);
    }

    #[test]
    fn a_typed_error_keeps_its_message_when_flattened() {
        let e = Error::UnknownVoice {
            name: "nobody".into(),
            known: vec!["alba".into(), "marius".into()],
        };
        let flattened: xn::Error = e.into();
        let msg = flattened.to_string();
        assert!(msg.contains("nobody"), "{msg}");
        assert!(msg.contains("alba, marius"), "{msg}");
    }

    #[test]
    fn every_kind_is_reachable_and_messages_name_their_subject() {
        let cases: Vec<(Error, ErrorKind, &str)> = vec![
            (Error::UnknownDevice { name: "tpu".into() }, ErrorKind::InvalidArgument, "tpu"),
            (
                Error::UnknownVoice { name: "nobody".into(), known: vec!["alba".into()] },
                ErrorKind::NotFound,
                "nobody",
            ),
            (
                Error::QuantUnsupportedOnDevice { quant: Quant::Q40, device: DeviceKind::Cuda },
                ErrorKind::Unsupported,
                "q4_0",
            ),
            (
                Error::Gated { repo_id: "kyutai/pocket-tts".into(), filename: "m.st".into() },
                ErrorKind::PermissionDenied,
                "HF_TOKEN",
            ),
            (
                Error::Hub {
                    repo_id: "a/b".into(),
                    what: "config.json".into(),
                    source: Box::new(std::io::Error::other("offline")),
                },
                ErrorKind::Network,
                "config.json",
            ),
            (
                Error::LayerCountMismatch { path: "m.safetensors".into(), found: 24, expected: 6 },
                ErrorKind::InvalidData,
                "24 transformer layers",
            ),
            (Error::GenerationInProgress, ErrorKind::Busy, "already in flight"),
            (Error::WorkerPanicked, ErrorKind::Internal, "panicked"),
        ];
        for (error, kind, needle) in cases {
            let msg = error.to_string();
            assert_eq!(error.kind(), kind, "{msg}");
            assert!(msg.contains(needle), "expected {needle:?} in: {msg}");
        }
    }

    #[test]
    fn the_gated_message_says_what_to_do_about_it() {
        // The first thing a new user meets, so it is worth pinning: the message has to name the
        // repo, where to accept its terms, and how to authenticate.
        let e = Error::Gated {
            repo_id: "kyutai/pocket-tts".into(),
            filename: "tts_b6369a24.safetensors".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("huggingface.co/kyutai/pocket-tts"), "{msg}");
        assert!(msg.contains("huggingface-cli login"), "{msg}");
        assert!(msg.contains("HF_TOKEN"), "{msg}");
        assert_eq!(e.kind(), ErrorKind::PermissionDenied);
    }

    #[test]
    fn devices_and_quants_read_as_names_not_as_debug() {
        assert_eq!(DeviceKind::Cuda.to_string(), "CUDA");
        assert_eq!(DeviceKind::Cpu.to_string(), "CPU");
        assert_eq!(Quant::Q4k.to_string(), "q4k");
        assert_eq!(Quant::F32.to_string(), "f32");
    }
}
