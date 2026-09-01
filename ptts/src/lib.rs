pub mod asr;
pub mod asr_lm;
pub mod asr_quantizer;
pub mod conditioners;
pub mod conv;
pub mod dummy_quantizer;
pub mod flow_lm;
pub mod layer_scale;
pub mod mimi;
pub mod mlp;
pub mod resample;
pub mod rope;
pub mod seanet;
pub mod transformer;
pub mod tts_model;
pub mod utils;
pub mod wav;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> xn::Result<String>;
}
