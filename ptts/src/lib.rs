//! Phonon: text to 24 kHz speech, on device.
//!
//! Text is tokenized, a flow-matching language model turns the tokens into Mimi
//! codec latents, and the Mimi decoder turns those into PCM. Generation is
//! streaming throughout: latents are decoded as they are produced.
//!
//! # Getting started
//!
//! [`synth::Synth`] is the whole pipeline behind one call:
//!
//! ```no_run
//! # fn main() -> ptts::Result<()> {
//! use ptts::preprocess::Lang;
//! use ptts::synth::Synth;
//! use ptts::checkpoint::read_config;
//!
//! let tts = Synth::builder(
//!     read_config("model/config.json")?,
//!     "model/model.safetensors",
//!     Lang::En,
//! )
//! .tokenizer_file("model/tokenizer.json")
//! .add_voice("default", "model/default-voice.safetensors")
//! .build()?;
//! let pcm = tts.say("Hello world")?;
//! ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate())?;
//! # Ok(())
//! # }
//! ```
//!
//! # Layers
//!
//! | Module | Role |
//! |---|---|
//! | [`synth`] | The one-call API: load, prime, generate, decode. Start here. |
//! | [`checkpoint`] | Resolving a model: a local directory, an `hf://` repo, or the default. |
//! | [`voice_catalog`] | Voices listed in a catalog, downloaded when first used. |
//! | [`loader`] | Reading weights and voice files, and the checkpoint key mapping. |
//! | [`error`] | [`Error`], what those two return. |
//! | [`plan`] | Frame and KV budgets, the end-of-speech policy. |
//! | [`preprocess`] | Per-language text normalization, applied before tokenizing. |
//! | [`tok`] | The Hugging Face tokenizer, behind the `hf` feature. |
//! | [`audio`] | Decoding and resampling audio files for voice cloning, behind `audio`. |
//! | [`tts_model`] | [`tts_model::TTSModel`], the streaming primitives `synth` drives. |
//! | [`flow_lm`], [`transformer`] | The token-conditioned flow-matching LM. |
//! | [`mimi`] | The neural audio codec. |
//!
//! # Errors
//!
//! [`synth`] and [`loader`] return [`Error`]; the model modules below them keep `xn::Result`,
//! because a shape mismatch inside the codec is not something a caller acts on. `?` crosses the
//! boundary in both directions, so a frontend whose own functions return `xn::Result` keeps
//! compiling. The variants are failure classes, so a binding maps them onto its host language's
//! exceptions in one match.
//!
//! [`checkpoint`] resolves a model for native callers: a local directory, or with the `hub`
//! feature an `hf://owner/name` repo on the Hugging Face Hub, which is the only time the library
//! downloads anything. Frontends load [`checkpoint::DEFAULT_MODEL`] when none is given. The
//! lower-level builder still accepts explicit files.
//!
//! Text is normalized before it is tokenized — see [`preprocess`]. Which
//! language, or [`preprocess::Normalize::OFF`], is a required argument to
//! [`synth::SynthBuilder::new`]: the model reads normalized text noticeably
//! better, but the spoken forms are per-language, so guessing is worse than
//! doing nothing.
//!
//! A voice is conditioned on once per [`synth::Synth`], whichever entry point
//! is used; [`synth::SynthApi::session`] additionally pins the KV budget for a
//! stream of requests.
//!
//! Callers that need to drive the loop themselves — a browser build stepping
//! from an event loop, with no threads to spawn — should use
//! [`tts_model::TTSModel`] directly. `Synth` is a composition of those
//! primitives, not a replacement for them.

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "audio")]
pub mod audio;
#[cfg(not(target_arch = "wasm32"))]
pub mod checkpoint;
pub mod comp;
mod conditioners;
mod conv;
mod dummy_quantizer;
pub mod error;
pub mod flow_lm;
#[cfg(not(target_arch = "wasm32"))]
pub mod hub;
mod layer_scale;
pub mod loader;
pub mod mimi;
mod mlp;
pub mod plan;
pub mod preprocess;
mod resample;
mod rope;
mod seanet;
// Not on wasm: `Synth` runs the flow LM and the Mimi decoder on two `std::thread`s, and
// `wasm32-unknown-unknown` has none -- `spawn` there compiles and then panics. Browser
// frontends drive `tts_model::TTSModel` directly instead.
//
// Deliberately `//` and not `///`: an outer doc comment here is concatenated ahead of
// `synth.rs`'s own `//!` header, whose intra-doc links then resolve in this scope and all
// come out unresolved.
#[cfg(not(target_arch = "wasm32"))]
pub mod synth;
#[cfg(feature = "hf")]
pub mod tok;
pub mod transformer;
pub mod tts_model;
pub mod utils;
#[cfg(not(target_arch = "wasm32"))]
pub mod voice_catalog;
pub mod wav;

pub use error::{Error, Result};

pub trait Tokenizer {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> xn::Result<String>;
}

#[cfg(test)]
fn test_config() -> tts_model::TTSConfig {
    serde_json::from_str(include_str!("../tests/fixtures/config.json")).unwrap()
}
