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
//! let tts = Synth::builder(TTSConfig::v202601(0.5), "model/model.safetensors")
//!     .tokenizer_file("model/tokenizer.json")
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
//! | [`tok`] | The Hugging Face tokenizer, behind the `hf` feature. |
//! | [`audio`] | Decoding and resampling audio files for voice cloning, behind `audio`. |
//! | [`tts_model`] | [`tts_model::TTSModel`], the streaming primitives `synth` drives. |
//! | [`flow_lm`], [`transformer`] | The token-conditioned flow-matching LM. |
//! | [`mimi`], [`seanet`] | The neural audio codec. |
//!
//! Which files a checkpoint ships, and what they are called, is the caller's to
//! know: `ptts` reads the config, weights, tokenizer and voice files it is
//! handed, and never guesses at names or downloads anything itself.
//!
//! A server answering many requests for one voice wants
//! [`synth::Synth::session`], which conditions on the voice prompt once
//! instead of per request.
//!
//! Callers that need to drive the loop themselves — a browser build stepping
//! from an event loop, with no threads to spawn — should use
//! [`tts_model::TTSModel`] directly. `Synth` is a composition of those
//! primitives, not a replacement for them.

#[cfg(feature = "audio")]
pub mod audio;
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
#[cfg(feature = "hf")]
pub mod tok;
pub mod transformer;
pub mod tts_model;
pub mod utils;
pub mod wav;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> xn::Result<String>;
}
