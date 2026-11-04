//! The `config.json` that `scripts/export_audium.py` writes next to the weights.

use std::collections::HashMap;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub sample_rate: usize,
    pub frame_rate: f64,
    /// How many steps the text runs behind the audio.
    pub delay_steps: usize,
    pub text: TextConfig,
    pub trunk: TrunkConfig,
    pub gemimi: GemimiConfig,
    /// The values `condition_sum` was computed for, as a record.
    #[serde(default)]
    pub conditions: HashMap<String, String>,
}

impl Config {
    pub fn load(path: &std::path::Path) -> xn::Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| xn::Error::msg(format!("reading {}: {e}", path.display())))?;
        serde_json::from_str(&json)
            .map_err(|e| xn::Error::msg(format!("parsing {}: {e}", path.display())))
    }

    /// Steps of silence a stream needs at its end for its last words to come out: the text delay
    /// plus a second, since the model writes a word's last pieces a few steps after the delay.
    pub fn flush_steps(&self) -> usize {
        self.delay_steps + self.frame_rate.ceil() as usize
    }

    /// Audio samples per step.
    pub fn frame_size(&self) -> usize {
        (self.sample_rate as f64 / self.frame_rate).round() as usize
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TextConfig {
    /// Output vocabulary; the input table has one more row, the initial token.
    pub card: usize,
    pub padding_id: u32,
    pub end_of_padding_id: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrunkConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub ffn_hidden: usize,
    /// Attention window, in steps.
    pub context: usize,
    pub max_period: f32,
    pub norm_eps: f32,
    pub qk_norm_eps: f32,
    #[serde(default)]
    pub text_emb_low_rank: Option<usize>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GemimiConfig {
    pub channels: usize,
    pub dimension: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_base: usize,
    pub compress: usize,
    pub transformer_d_model: usize,
    pub transformer_num_heads: usize,
    pub transformer_num_layers: usize,
    pub transformer_context: usize,
    pub transformer_max_period: f32,
    pub transformer_dim_feedforward: usize,
    /// Encoder-transformer frames concatenated into one latent frame.
    pub latent_stack: usize,
    pub latent_dim: usize,
}
