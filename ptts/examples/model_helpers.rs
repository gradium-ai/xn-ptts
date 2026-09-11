//! Bits the examples share that are not worth a place in the library.
//!
//! Weight loading, key remapping and voice-embedding loading now live in `ptts::loader`, the
//! tokenizer in `ptts::tok` and the frame budget in `ptts::plan`; this re-exports the first two
//! so each example has one place to look. `ptts::plan` is used directly, since it is not tied
//! to a backend type.
#![allow(dead_code, unused_imports)]

pub use ptts::loader::{is_unused_by_tts_model, load_voice_emb, load_weights, remap_key};
#[cfg(feature = "sp")]
pub use ptts::tok::Tok;
