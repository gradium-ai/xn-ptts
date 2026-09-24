//! The error type the API layer returns.
//!
//! [`crate::synth`] and [`crate::loader`] return [`Error`]; the model modules below them
//! ([`crate::tts_model`], [`crate::flow_lm`], [`crate::mimi`] and friends) keep `xn::Result`,
//! because a shape mismatch inside the codec is not something a caller acts on. `?` crosses the
//! boundary in both directions: [`Error`] is `From<xn::Error>`, and `xn::Error` is `From<Error>`,
//! so a frontend whose own functions return `xn::Result` keeps compiling.
//!
//! The variants are failure *classes*, not one per message: three of this repo's frontends are
//! bindings, and what each needs is to turn a failure into the right kind of exception. So a
//! binding matches the enum once, as a table:
//!
//! ```
//! # use ptts::Error;
//! fn exception_for(e: &Error) -> &'static str {
//!     match e {
//!         Error::InvalidArgument(_) => "ValueError",
//!         Error::UnknownVoice { .. } | Error::NotFound(_) => "LookupError",
//!         Error::Unsupported(_) => "NotImplementedError",
//!         // `Error` is `#[non_exhaustive]`, so a variant added later lands here rather than
//!         // being silently folded into one of the above.
//!         _ => "RuntimeError",
//!     }
//! }
//! ```
//!
//! Two variants carry fields instead of only a message, because a caller does something with
//! them: [`Error::UnknownVoice`] hands back the names that would have worked, and
//! [`Error::SeqBudgetExceeded`] hands back the budget this text needs.

/// A [`Result`](std::result::Result) over [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Everything the `ptts` API layer can fail with.
///
/// `#[non_exhaustive]`: variants get added as the API grows, so always leave a `_` arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A voice was named that is not registered. `known` is what would have worked.
    #[error("no voice '{name}' is registered; this model has {}", known.join(", "))]
    UnknownVoice { name: String, known: Vec<String> },

    /// More KV slots were needed than the session was primed with. Build the session with
    /// `max_seq_len` at least `needed`, or split the text.
    ///
    /// `needed` is what this text needs, so a session rebuilt with it will fit this text. A
    /// *longer* one will not: size the session for the longest text it will see, not the first
    /// one that failed.
    #[error(
        "needs a KV budget of {needed} but the session was primed with {budget}; build the \
         session with a larger max_seq_len, or split the text"
    )]
    SeqBudgetExceeded { needed: usize, budget: usize },

    /// The voice prompt alone does not fit the session's KV budget, before any text is
    /// considered. Splitting the text cannot help, which is why this is not
    /// [`Self::SeqBudgetExceeded`]; `frames + 1` is the floor a budget has to clear, not a
    /// value that will work.
    #[error(
        "the voice prompt alone is {frames} frames, which does not fit a KV budget of \
         {budget}; build the session with a larger max_seq_len, or use a shorter voice prompt"
    )]
    VoicePromptTooLong { frames: usize, budget: usize },

    /// The caller passed something this call cannot accept: an unparseable name, a mis-shaped
    /// array, an empty text, a prompt that is too short.
    #[error("{0}")]
    InvalidArgument(String),

    /// Something named does not exist: a checkpoint, or a file inside one.
    #[error("{0}")]
    NotFound(String),

    /// Well formed, but this build or this checkpoint cannot serve it: a backend that was not
    /// compiled in, quantization on a GPU, cloning without a speaker encoder.
    #[error("{0}")]
    Unsupported(String),

    /// A file was found and is not what it claims to be: a checkpoint whose config does not
    /// match its weights, an unreadable voice embedding.
    #[error("{0}")]
    InvalidData(String),

    /// The resource is in use and the call would corrupt it. Retrying later works.
    #[error("{0}")]
    Busy(String),

    /// A file could not be read or written.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// A failure from the tensor layer, or from anything below the API layer that reports
    /// through it -- including a panicked generation worker.
    #[error(transparent)]
    Tensor(xn::Error),
}

/// Hand-written rather than `#[from]`, so an I/O failure gets one class wherever it was caught.
///
/// `xn::Error` carries its own `Io` variant, and plenty of I/O reaches us through it: opening a
/// voice file goes `VB::load` -> `File::open` -> `xn::Error::Io`. Deriving `From` would file all
/// of that under `Tensor`, so a missing voice path would be a `RuntimeError` in Python while a
/// missing weights path was a `LookupError` and a missing config was an `OSError`.
impl From<xn::Error> for Error {
    fn from(e: xn::Error) -> Self {
        match io_kind(&e) {
            // Rebuilt rather than unwrapped. The useful half of the message is usually in the
            // wrapper -- `WithPath` is what carries the filename -- so keep the whole rendering
            // and take only the kind from the `io::Error` at the centre.
            Some(kind) => Self::Io(std::io::Error::new(kind, e.to_string())),
            None => Self::Tensor(e),
        }
    }
}

/// The kind of the I/O failure at the centre of `e`, if there is one.
///
/// `xn::Error` nests: by the time a failed `File::open` reaches this crate it is normally inside
/// a `WithPath`, and may also be inside a `Context` or a `WithBacktrace`. Matching only the bare
/// `Io` variant would miss nearly every real one.
fn io_kind(e: &xn::Error) -> Option<std::io::ErrorKind> {
    match e {
        xn::Error::Io(io) => Some(io.kind()),
        xn::Error::WithPath { inner, .. }
        | xn::Error::Context { inner, .. }
        | xn::Error::WithBacktrace { inner, .. } => io_kind(inner),
        _ => None,
    }
}

// The `allow`s below are for wasm only, where `synth` is not compiled and these three lose
// their only callers. Per function rather than on the `impl`, so anything else going dead
// still warns.
impl Error {
    pub(crate) fn invalid_argument(detail: impl std::fmt::Display) -> Self {
        Self::InvalidArgument(detail.to_string())
    }
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn not_found(detail: impl std::fmt::Display) -> Self {
        Self::NotFound(detail.to_string())
    }
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn unsupported(detail: impl std::fmt::Display) -> Self {
        Self::Unsupported(detail.to_string())
    }
    pub(crate) fn invalid_data(detail: impl std::fmt::Display) -> Self {
        Self::InvalidData(detail.to_string())
    }
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn busy(detail: impl std::fmt::Display) -> Self {
        Self::Busy(detail.to_string())
    }
}

/// Lets a frontend whose own functions return `xn::Result` keep using `?` on this crate.
///
/// A classified variant flattens to its message, so this direction loses the class. The
/// passthrough variants cross whole, which is what keeps a tensor failure from picking up a
/// layer of wrapping on the way through.
impl From<Error> for xn::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Tensor(e) => e,
            Error::Io(e) => xn::Error::Io(e),
            other => xn::Error::msg(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_passthrough_variants_survive_the_round_trip_to_xn() {
        // The bridge exists so a frontend on `xn::Result` keeps compiling. A tensor error must
        // not pick up a layer of message wrapping on the way through.
        let there: xn::Error = Error::Tensor(xn::Error::msg("device out of memory")).into();
        // `xn::Error::msg` captures a backtrace when `RUST_BACKTRACE` is set, and prints it
        // below the message -- so compare the first line rather than the whole string. What
        // this asserts is that nothing wrapped the message, not that xn kept it bare.
        assert_eq!(there.to_string().lines().next(), Some("device out of memory"));
        assert!(matches!(Error::from(there), Error::Tensor(_)));

        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        assert!(matches!(xn::Error::from(Error::Io(io)), xn::Error::Io(_)));
    }

    #[test]
    fn an_io_failure_gets_one_class_whichever_layer_caught_it() {
        // The reason `From<xn::Error>` is hand-written. Opening a voice file goes through the
        // tensor layer and comes back as `xn::Error::Io`; opening a config does not. Both are
        // the same failure and must not land in different variants.
        let bare = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        assert!(matches!(Error::from(xn::Error::Io(bare)), Error::Io(_)));

        // And through the wrappers `xn` actually uses -- a failed `File::open` arrives inside a
        // `WithPath`, so matching only the bare variant would miss nearly every real one.
        let nested = xn::Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "nope"))
            .with_path("/some/voice.safetensors");
        let converted = Error::from(nested);
        assert!(matches!(converted, Error::Io(_)), "{converted:?}");
        // The path lives in the wrapper, so it has to survive the conversion.
        assert!(converted.to_string().contains("voice.safetensors"), "{converted}");

        // Anything else from that layer is still a tensor failure.
        assert!(matches!(Error::from(xn::Error::msg("shape mismatch")), Error::Tensor(_)));
    }

    #[test]
    fn a_classified_error_keeps_its_message_when_flattened() {
        let e = Error::UnknownVoice {
            name: "nobody".into(),
            known: vec!["alba".into(), "marius".into()],
        };
        let msg = xn::Error::from(e).to_string();
        assert!(msg.contains("nobody"), "{msg}");
        assert!(msg.contains("alba, marius"), "{msg}");
    }

    #[test]
    fn unknown_voice_hands_back_the_names_that_would_have_worked() {
        // The reason this variant carries fields at all: a caller can correct itself, or print
        // the list, without parsing it back out of the message.
        let e = Error::UnknownVoice { name: "nobody".into(), known: vec!["alba".into()] };
        let Error::UnknownVoice { known, .. } = &e else { panic!("{e:?}") };
        assert_eq!(known, &["alba"]);
    }
}
