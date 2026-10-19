use crate::Tokenizer;
use crate::tts_model::TTSConfig;
use std::collections::HashMap;
use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

fn default_max_period() -> f32 {
    10000.0
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ConditionerConfig {
    Lut { name: String, lut: LutConditionerConfig },
    Continuous { name: String, continuous: ContinuousConditionerConfig },
}

impl ConditionerConfig {
    pub fn name(&self) -> &str {
        match self {
            Self::Lut { name, .. } | Self::Continuous { name, .. } => name,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LutConditionerConfig {
    pub n_bins: usize,
    pub dim: usize,
    pub tokenizer: String,
    #[serde(default)]
    pub possible_values: Option<Vec<String>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ContinuousConditionerConfig {
    pub scale_factor: f32,
    pub dim: usize,
    #[serde(default = "default_max_period")]
    pub max_period: f32,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct FuserConfig {
    #[serde(default)]
    pub sum: Vec<String>,
}

fn default_condition(name: &str) -> Option<&'static str> {
    match name {
        "num_speakers" => Some("1"),
        "duration_delta" | "padding_bonus" => Some("0.0"),
        _ => None,
    }
}

pub fn load_summed_conditions<T: WithDTypeF, B: Backend>(
    vb: &Path<B>,
    config: &TTSConfig,
    values: &HashMap<String, String>,
) -> Result<Option<Tensor<T, B>>> {
    Conditioners::load(vb, config)?.sum(values)
}

enum Condition<T: WithDTypeF, B: Backend> {
    Lut { possible: Vec<String>, lut: LUTConditioner<T, B> },
    Continuous { cfg: ContinuousConditionerConfig, proj: Option<Linear<T, B>> },
}

pub struct Conditioners<T: WithDTypeF, B: Backend> {
    conditions: Vec<(String, Condition<T, B>)>,
    device: B,
}

impl<T: WithDTypeF, B: Backend> Conditioners<T, B> {
    pub fn load(vb: &Path<B>, config: &TTSConfig) -> Result<Self> {
        let output_dim = config.flow_lm.d_model;
        let mut conditions = Vec::with_capacity(config.conditioners.len());
        for cfg in config.conditioners.iter() {
            let name = cfg.name();
            if let Some(fuser) = &config.fuser
                && !fuser.sum.iter().any(|s| s == name)
            {
                xn::bail!(
                    "conditioner {name:?} is not in the fuser's sum, the only fusion supported"
                )
            }
            let vb = vb.pp(name);
            let condition = match cfg {
                ConditionerConfig::Lut { lut, .. } => load_lut_condition(&vb, lut, output_dim)?,
                ConditionerConfig::Continuous { continuous, .. } => {
                    load_continuous_condition(&vb, continuous, output_dim)?
                }
            };
            conditions.push((name.to_string(), condition));
        }
        Ok(Self { conditions, device: vb.device().clone() })
    }

    pub fn sum(&self, values: &HashMap<String, String>) -> Result<Option<Tensor<T, B>>> {
        if let Some(name) = values.keys().find(|k| !self.conditions.iter().any(|(n, _)| n == *k)) {
            xn::bail!("no conditioner named {name:?} in this checkpoint")
        }
        let mut sum: Option<Tensor<T, B>> = None;
        for (name, condition) in self.conditions.iter() {
            let Some(value) =
                values.get(name).map(String::as_str).or_else(|| default_condition(name))
            else {
                xn::bail!("conditioner {name:?} has no default, a value has to be given for it")
            };
            let cond = match condition {
                Condition::Lut { possible, lut } => {
                    let Some(id) = possible.iter().position(|v| v == value) else {
                        xn::bail!("{value:?} is not one of the conditioner's values {possible:?}")
                    };
                    lut.embed_tokens(&[id as u32])?
                }
                Condition::Continuous { cfg, proj } => {
                    let value: f32 = value
                        .parse()
                        .map_err(|_| xn::Error::msg(format!("{value:?} is not a number")))?;
                    let emb = sin_embedding(cfg.scale_factor * value, cfg.dim, cfg.max_period)?;
                    let emb = Tensor::from_vec(
                        emb.into_iter().map(T::from_f32).collect(),
                        (1, 1, cfg.dim),
                        &self.device,
                    )?;
                    match proj {
                        Some(proj) => proj.forward(&emb)?,
                        None => emb,
                    }
                }
            };
            sum = Some(match sum {
                Some(s) => s.add(&cond)?,
                None => cond,
            });
        }
        Ok(sum)
    }
}

fn load_lut_condition<T: WithDTypeF, B: Backend>(
    vb: &Path<B>,
    cfg: &LutConditionerConfig,
    output_dim: usize,
) -> Result<Condition<T, B>> {
    if cfg.tokenizer != "noop" {
        xn::bail!("unsupported tokenizer {:?} for a lut conditioner", cfg.tokenizer)
    }
    let Some(possible) = cfg.possible_values.clone() else {
        xn::bail!("a lut conditioner without possible_values is not supported")
    };
    let lut = LUTConditioner::load(vb, cfg.n_bins, None, cfg.dim, output_dim)?;
    Ok(Condition::Lut { possible, lut })
}

fn load_continuous_condition<T: WithDTypeF, B: Backend>(
    vb: &Path<B>,
    cfg: &ContinuousConditionerConfig,
    output_dim: usize,
) -> Result<Condition<T, B>> {
    if vb.contains("learnt_padding") {
        vb.tensor::<T>("learnt_padding", (1, 1, output_dim))?;
    }
    let proj = if vb.contains("output_proj.weight") {
        Some(Linear::load(vb.pp("output_proj"), cfg.dim, output_dim)?)
    } else if cfg.dim == output_dim {
        None
    } else {
        xn::bail!("conditioner of dim {} has no output_proj to {output_dim}", cfg.dim)
    };
    Ok(Condition::Continuous { cfg: cfg.clone(), proj })
}

fn sin_embedding(position: f32, dim: usize, max_period: f32) -> Result<Vec<f32>> {
    if dim < 4 || !dim.is_multiple_of(2) {
        xn::bail!("a sinusoidal embedding needs an even dim of at least 4, got {dim}")
    }
    let half = dim / 2;
    let phases: Vec<f32> =
        (0..half).map(|i| position / max_period.powf(i as f32 / (half - 1) as f32)).collect();
    Ok(phases.iter().map(|p| p.cos()).chain(phases.iter().map(|p| p.sin())).collect())
}

pub struct LUTConditioner<T: WithDTypeF, B: Backend> {
    pub tokenizer: Option<Box<dyn Tokenizer + Send + Sync>>,
    embed: Tensor<T, B>,
    learnt_padding: Option<Tensor<T, B>>,
    learnt_padding_id: Option<u32>,
    pub dim: usize,
    pub output_dim: usize,
}

impl<T: WithDTypeF, B: Backend> LUTConditioner<T, B> {
    pub fn load(
        vb: &Path<B>,
        n_bins: usize,
        tokenizer: Option<Box<dyn Tokenizer + Send + Sync>>,
        dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let embed = vb.tensor("embed.weight", (n_bins + 1, dim))?;
        let learnt_padding = if vb.contains("learnt_padding") {
            Some(vb.tensor("learnt_padding", (1, 1, output_dim))?)
        } else {
            None
        };
        let embed = if vb.contains("output_proj.weight") {
            let proj = Linear::load(vb.pp("output_proj"), dim, output_dim)?;
            proj.forward(&embed)?
        } else {
            embed
        };
        let (embed, learnt_padding_id) = match learnt_padding.as_ref() {
            Some(learnt_padding) => {
                let learnt_padding = learnt_padding.squeeze(0)?;
                let embed = Tensor::cat(&[&embed, &learnt_padding], 0)?;
                (embed, Some(n_bins as u32 + 1))
            }
            None => (embed, None),
        };
        Ok(Self { tokenizer, embed, dim, output_dim, learnt_padding, learnt_padding_id })
    }

    pub fn learnt_padding_id(&self) -> Option<u32> {
        self.learnt_padding_id
    }

    /// Tokenize text and return token ids.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        match self.tokenizer.as_ref() {
            Some(tokenizer) => Ok(tokenizer.encode(text)?),
            None => xn::bail!("No tokenizer available for LUTConditioner"),
        }
    }

    /// Get embeddings for token ids. Returns [1, num_tokens, dim].
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor<T, B>> {
        if token_ids.is_empty() {
            let dev = self.embed.device();
            return Tensor::zeros((1, 0, self.output_dim), dev);
        }
        // `index_select` does not bounds-check, so an id past the table panics instead of
        // erroring. The usual cause is a tokenizer from another checkpoint. The learnt padding,
        // when there is one, is the table's last row and is never looked up by id.
        let rows = self.learnt_padding_id.map_or(self.embed.dims()[0], |id| id as usize);
        if let Some(&id) = token_ids.iter().find(|&&id| id as usize >= rows) {
            xn::bail!(
                "token id {id} is past this checkpoint's {rows}-entry embedding table: the \
                 tokenizer.json does not belong to this checkpoint"
            )
        }
        let ids_t = Tensor::from_vec(
            token_ids.iter().map(|&x| x as i64).collect(),
            token_ids.len(),
            self.embed.device(),
        )?;
        let emb = self.embed.index_select(&ids_t, 0)?;
        let emb = emb.reshape((1, token_ids.len(), self.output_dim))?;
        Ok(emb)
    }

    pub fn learnt_padding(&self) -> Option<&Tensor<T, B>> {
        self.learnt_padding.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xn::CpuDevice;

    fn conditioner(rows: usize, dim: usize) -> LUTConditioner<f32, CpuDevice> {
        let data = (0..rows * dim).map(|i| i as f32).collect();
        let embed = Tensor::from_vec(data, (rows, dim), &CpuDevice).unwrap();
        LUTConditioner {
            tokenizer: None,
            embed,
            learnt_padding: None,
            learnt_padding_id: None,
            dim,
            output_dim: dim,
        }
    }

    #[test]
    fn sin_embedding_matches_the_reference_formula() {
        let emb = sin_embedding(2.0, 4, 10000.0).unwrap();
        let expected = [2f32.cos(), 0.0002f32.cos(), 2f32.sin(), 0.0002f32.sin()];
        for (a, b) in emb.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{emb:?}");
        }
    }

    #[test]
    fn conditioner_configs_parse() {
        let json = serde_json::json!([
            {"type": "lut", "name": "num_speakers",
             "lut": {"n_bins": 31, "dim": 16, "tokenizer": "noop", "possible_values": ["0", "1"]}},
            {"type": "continuous", "name": "padding_bonus",
             "continuous": {"scale_factor": 100.0, "dim": 128, "max_period": 10000.0}},
        ]);
        let cfgs: Vec<ConditionerConfig> = serde_json::from_value(json).unwrap();
        assert_eq!(
            cfgs.iter().map(|c| c.name()).collect::<Vec<_>>(),
            ["num_speakers", "padding_bonus"]
        );
        assert!(
            matches!(&cfgs[1], ConditionerConfig::Continuous { continuous, .. } if continuous.dim == 128)
        );
    }

    #[test]
    fn a_token_id_past_the_table_is_an_error() {
        let c = conditioner(4, 2);
        assert!(c.embed_tokens(&[0, 3]).is_ok());
        let err = c.embed_tokens(&[1, 4]).expect_err("id 4 is past a 4-row table");
        assert!(err.to_string().contains("tokenizer"), "{err}");
    }

    #[test]
    fn no_tokens_embed_to_the_output_width() {
        // A low-rank table's `dim`-wide rows are projected to `output_dim` at load.
        let mut c = conditioner(4, 3);
        c.dim = 2;
        assert_eq!(c.embed_tokens(&[]).unwrap().dims(), &[1, 0, 3]);
    }

    #[test]
    fn the_learnt_padding_row_is_not_a_token_id() {
        // Three vocabulary rows, then the padding row appended at id 3.
        let mut c = conditioner(4, 2);
        c.learnt_padding_id = Some(3);
        assert!(c.embed_tokens(&[0, 2]).is_ok());
        assert!(c.embed_tokens(&[3]).is_err(), "a tokenizer one id too long reached the padding");
    }
}
