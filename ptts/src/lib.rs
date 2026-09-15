//! Pocket TTS: text to 24 kHz speech, on device.
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
//! # fn main() -> xn::Result<()> {
//! use ptts::synth::Synth;
//! use ptts::tts_model::TTSConfig;
//!
//! let tts = Synth::builder(TTSConfig::v202601(0.7), "model/model.safetensors")
//!     .tokenizer_file("model/tokenizer.model")
//!     .add_voice("alba", "model/voices/alba.safetensors")
//!     .build()?;
//! let pcm = tts.say("Hello world")?;
//! ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate() as u32)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Layers
//!
//! | Module | Role |
//! |---|---|
//! | [`synth`] | The one-call API: load, prime, generate, decode. Start here. |
//! | [`loader`] | Reading weights and voice files, and the checkpoint key mapping. |
//! | [`plan`] | Frame and KV budgets, the end-of-speech policy. |
//! | [`preprocess`] | Per-language text normalization, applied before tokenizing. |
//! | [`tok`] | Tokenizers, behind the `sp` / `hf` features. |
//! | [`tts_model`] | [`tts_model::TTSModel`], the streaming primitives `synth` drives. |
//! | [`flow_lm`], [`transformer`] | The token-conditioned flow-matching LM. |
//! | [`mimi`], [`seanet`] | The neural audio codec. |
//!
//! Which files a checkpoint ships, and what they are called, is the caller's to
//! know: `ptts` reads the config, weights, tokenizer and voice files it is
//! handed, and never guesses at names or downloads anything itself.
//!
//! Callers that need to drive generation themselves — a browser build stepping
//! from an event loop, a server interleaving requests — should use
//! [`tts_model::TTSModel`] directly. `Synth` is a composition of those
//! primitives, not a replacement for them.

pub mod conditioners;
pub mod conv;
pub mod dummy_quantizer;
pub mod flow_lm;
pub mod layer_scale;
pub mod loader;
pub mod mimi;
pub mod mlp;
pub mod plan;
pub mod preprocess;
pub mod resample;
pub mod rope;
pub mod seanet;
pub mod synth;
#[cfg(any(feature = "sp", feature = "hf"))]
pub mod tok;
pub mod transformer;
pub mod tts_model;
pub mod utils;
pub mod wav;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> xn::Result<String>;
}
