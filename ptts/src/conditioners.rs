use crate::Tokenizer;
use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, Result, Tensor, WithDTypeF};

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
            return Tensor::zeros((1, 0, self.dim), dev);
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

    /// Embed several token sequences of the same length as one batch. Returns `[B, len, dim]`.
    ///
    /// The rows must have the same number of tokens: every row of a state sits at the same
    /// position, and the model has no inert token to pad a shorter row with. Its learnt
    /// padding embedding is a real input that changes what it says, so rows of different
    /// lengths are an error rather than padded.
    pub fn embed_tokens_batch(&self, rows: &[&[u32]]) -> Result<Tensor<T, B>> {
        if rows.is_empty() {
            xn::bail!("embed_tokens_batch: the batch is empty")
        }
        let (ids, len) = stacked_ids(rows)?;
        let dev = self.embed.device();
        if len == 0 {
            return Tensor::zeros((rows.len(), 0, self.output_dim), dev);
        }
        let ids_t = Tensor::from_vec(ids, rows.len() * len, dev)?;
        let emb = self.embed.index_select(&ids_t, 0)?;
        emb.reshape((rows.len(), len, self.output_dim))
    }

    pub fn learnt_padding(&self) -> Option<&Tensor<T, B>> {
        self.learnt_padding.as_ref()
    }
}

/// Lays `rows` out as one `[B, len]` block of ids, or fails if they differ in length.
fn stacked_ids(rows: &[&[u32]]) -> Result<(Vec<i64>, usize)> {
    let len = rows.first().map_or(0, |r| r.len());
    if rows.iter().any(|r| r.len() != len) {
        let lens: Vec<usize> = rows.iter().map(|r| r.len()).collect();
        xn::bail!(
            "cannot batch text rows of different lengths {lens:?}: the rows of a batch must \
             have the same number of tokens"
        )
    }
    Ok((rows.iter().flat_map(|r| r.iter().map(|&x| x as i64)).collect(), len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_rows_are_stacked_in_order() {
        let (ids, len) = stacked_ids(&[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        assert_eq!((ids, len), (vec![1, 2, 3, 4, 5, 6], 3));
    }

    #[test]
    fn rows_of_different_lengths_are_an_error() {
        let err = stacked_ids(&[&[1, 2], &[3]]).unwrap_err().to_string();
        assert!(err.contains("[2, 1]"), "{err}");
    }

    #[test]
    fn an_empty_batch_has_no_length() {
        assert_eq!(stacked_ids(&[]).unwrap(), (vec![], 0));
    }
}
