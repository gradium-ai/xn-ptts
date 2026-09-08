//! Weight loading shared by the `pocket_tts`, `bench` and `create_voice` examples.
//!
//! Each example includes this via `#[path = "model_helpers.rs"] mod model_helpers;`, the same
//! way they share `audio_helpers.rs`, so not every example uses every item here.
#![allow(dead_code)]

use anyhow::{Context, Result};
use xn::nn::{Path, VB};
use xn::{BackendQ, Tensor};

/// Maps upstream checkpoint names onto the names this crate's modules expect, dropping the
/// tensors the runtime has no use for.
pub fn remap_key(name: &str) -> Option<String> {
    // Skip keys we don't need
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
    {
        return None;
    }

    let mut name = name.to_string();

    // Order matters: more specific replacements first
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");

    Some(name)
}

/// Loads GGUF or safetensors weights, picking the format from the extension.
pub fn load_weights<Q: BackendQ>(path: &std::path::Path, dev: &Q::B) -> Result<Path<Q::B>> {
    let vb = if path.extension().and_then(|v| v.to_str()) == Some("gguf") {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        VB::load_gguf_with_key_map(reader, dev.clone(), remap_key)?
    } else {
        VB::load_with_key_map(&[path], dev.clone(), remap_key)?
    };
    Ok(vb.root())
}

/// Tensors that `TTSModel::load` legitimately leaves untouched: the encoder side is only pulled
/// in later by `MimiEnc::load`, and the quantizer is replaced by `dummy_quantizer`.
pub fn is_unused_by_tts_model(name: &str) -> bool {
    name == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
        || name.starts_with("mimi.quantizer")
        || name.starts_with("mimi.encoder")
        || name.starts_with("speaker_mimi")
        || name == "flow_lm.speaker_proj_weight"
        || name == "mimi.downsample.conv.conv.weight"
}

/// Loads a precomputed voice embedding as `[1, T, dim]`, checking it was made for this model.
pub fn load_voice_emb<Q: BackendQ>(
    path: &std::path::Path,
    model_ext: Option<&str>,
    dev: &Q::B,
) -> Result<Tensor<Q::T, Q::B>> {
    let vb = VB::load(&[path], dev.clone())?;
    let names = vb.tensor_names();
    let key = names.first().context("no tensors found in voice embedding file")?;
    let shape = vb.shape(key).context("voice tensor not found")?;
    let dims = shape.dims().to_vec();
    let emb: Tensor<f32, Q::B> = vb.tensor(key, shape)?;
    // Voice files hold either [T, dim] or an already batched [1, T, dim].
    let emb = if dims.len() == 2 { emb.reshape((1, dims[0], dims[1]))? } else { emb };
    if let Some(model_ext) = model_ext {
        let file_content = std::fs::read(path)?;
        let (_, metadata) = safetensors::SafeTensors::read_metadata(&file_content)?;
        if let Some(metadata) = metadata.metadata()
            && let Some(voice_model_ext) = metadata.get("model_ext")
        {
            tracing::info!(?voice_model_ext, "voice embedding model_ext from metadata");
            if voice_model_ext.as_str() != model_ext {
                anyhow::bail!(
                    "voice embedding model_ext '{voice_model_ext}' does not match config model_ext '{model_ext}'"
                )
            }
        }
    }
    Ok(emb.to::<Q::T>()?)
}

/// Frames an utterance of `num_tokens` tokens is allowed to generate before it is cut off.
pub fn max_frames_for(num_tokens: usize) -> usize {
    ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize
}
