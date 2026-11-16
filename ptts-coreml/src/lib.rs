//! Run Phonon through Core ML, with the flow LM on the Neural Engine.
//!
//! This is the implementation behind the PhononTTS Swift package (`ios/PhononTTS`), which apps
//! integrate; its API is not meant to be used directly and may change with the package.
//!
//! CoreML is an ahead-of-time graph format: you hand it a compiled program and call predict.
//! That is why this is not an `xn` backend -- `xn::Backend` is an eager per-op trait and CoreML
//! has no eager API. Instead the graphs are emitted from Rust as ML Programs (`mil`, `package`,
//! `blob`), exported once from a checkpoint (`ptts`'s `export_coreml` example), and driven by
//! `phonon::driver` through the CoreML runtime (`run`).
//!
//! Conventions here were taken from a model produced by coremltools and read back field by
//! field: specification version 9, program version 1, a `main` function (the flow LM has one
//! function per graph instead) with opset `CoreML8`, and `const` ops that carry their value in
//! `attributes["val"]` rather than as an input binding.
pub mod blob;
pub mod mil;
pub mod package;
pub mod phonon;
pub mod proto;
#[cfg(target_vendor = "apple")]
pub mod run;
pub mod weights;

pub use mil::{Builder, DType, Var};
pub use weights::Weights;
