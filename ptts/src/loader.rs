//! Reading checkpoints and voice embeddings off disk.
//!
//! Every frontend -- the examples, the ws-server, the Python bindings, the wasm build -- has to
//! rename the same checkpoint keys, skip the same unused tensors and unpack voice files the same
//! way. Keeping that here means a checkpoint layout change is one edit rather than four.

use xn::nn::{Path, VB};
use xn::{Backend, BackendQ, Result, Tensor};

/// Maps upstream checkpoint names onto the names this crate's modules expect, dropping the
/// tensors the runtime has no use for.
pub fn remap_key(name: &str) -> Option<String> {
    // Skip keys we don't need.
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
    {
        return None;
    }

    let mut name = name.to_string();

    // Order matters: more specific replacements first.
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

/// Tensors that [`crate::tts_model::TTSModel::load`] legitimately leaves untouched: the encoder
/// side is only pulled in later by `MimiEnc::load`, and the quantizer is replaced by
/// [`crate::dummy_quantizer`].
///
/// Pass to `check_all_used_with_ignore` so a genuinely unused tensor is still an error.
pub fn is_unused_by_tts_model(name: &str) -> bool {
    name == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
        || name.starts_with("mimi.quantizer")
        || name.starts_with("mimi.encoder")
        || name.starts_with("speaker_mimi")
        || name == "flow_lm.speaker_proj_weight"
        // A prefix, not the single `conv.conv.weight` the examples used to name: the ws-server
        // already matched it this way, and taking the union keeps every caller as permissive as
        // it was.
        || name.starts_with("mimi.downsample.")
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

/// Loads a precomputed voice embedding as `[1, T, dim]`.
///
/// Voice files hold either `[T, dim]` or an already batched `[1, T, dim]`. When `model_ext` is
/// given and the file records one of its own, the two must agree -- a voice conditioned on a
/// different checkpoint produces confident nonsense rather than an error, so it is worth
/// catching here. Pass `None` to skip the check.
///
/// The result is f32 regardless of the backend's quantization; convert with `to::<Q::T>()`.
pub fn load_voice_emb<B: Backend>(
    path: &std::path::Path,
    model_ext: Option<&str>,
    dev: &B,
) -> Result<Tensor<f32, B>> {
    use xn::error::Context;

    let vb = VB::load(&[path], dev.clone())?;
    let names = vb.tensor_names();
    let key = names.first().context("no tensors found in voice embedding file")?;
    let shape = vb.shape(key).context("voice tensor not found")?;
    let dims = shape.dims().to_vec();
    let emb: Tensor<f32, B> = vb.tensor(key, shape)?;
    let emb = if dims.len() == 2 { emb.reshape((1, dims[0], dims[1]))? } else { emb };
    if let Some(model_ext) = model_ext {
        check_model_ext(path, model_ext)?;
    }
    Ok(emb)
}

/// Fails if the voice file records a `model_ext` other than `model_ext`. A file that records
/// none is accepted: older voices predate the metadata.
fn check_model_ext(path: &std::path::Path, model_ext: &str) -> Result<()> {
    let header = read_safetensors_header(path)?;
    // Not `SafeTensors::read_metadata`: it validates that the buffer holds the
    // tensor data as well as the header, which is exactly what is not read here.
    let header: serde_json::Value = serde_json::from_slice(&header).map_err(|e| {
        xn::Error::msg(format!("cannot parse safetensors header of {}: {e}", path.display()))
    })?;
    if let Some(voice_model_ext) = header.get("__metadata__").and_then(|m| m.get("model_ext"))
        && let Some(voice_model_ext) = voice_model_ext.as_str()
    {
        tracing::info!(?voice_model_ext, "voice embedding model_ext from metadata");
        if voice_model_ext != model_ext {
            xn::bail!(
                "voice embedding model_ext '{voice_model_ext}' does not match config model_ext '{model_ext}'"
            )
        }
    }
    Ok(())
}

/// Reads a safetensors file's JSON header: the 8-byte little-endian header
/// length, then that many bytes.
///
/// `load_voice_emb` is called once per voice while a model loads, and the
/// published checkpoint ships eight of them, so reading whole files here would
/// double the bytes touched at startup for no reason.
fn read_safetensors_header(path: &std::path::Path) -> Result<Vec<u8>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    // A file that is not safetensors can claim an absurd header length; refuse
    // to allocate on its word.
    if header_len > 100 * 1024 * 1024 {
        xn::bail!("{} does not look like a safetensors file", path.display())
    }
    let mut buf = vec![0u8; header_len as usize];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_key_renames_and_drops() {
        // Dropped outright.
        for dropped in [
            "flow_lm.flow.w_s_t",
            "mimi.quantizer.vq.something",
            "mimi.quantizer.logvar_proj.weight",
        ] {
            assert_eq!(remap_key(dropped), None, "expected {dropped} to be dropped");
        }

        let cases = [
            (
                "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
                "flow_lm.speaker_proj_weight",
            ),
            (
                "flow_lm.condition_provider.conditioners.transcript_in_segment.emb.weight",
                "flow_lm.conditioner.emb.weight",
            ),
            (
                "flow_lm.backbone.layers.0.linear1.weight",
                "flow_lm.transformer.layers.0.linear1.weight",
            ),
            ("mimi.model.decoder.layers.0.weight", "mimi.decoder.layers.0.weight"),
        ];
        for (from, to) in cases {
            assert_eq!(remap_key(from).as_deref(), Some(to), "input: {from}");
        }

        // Anything unrecognized passes through untouched.
        assert_eq!(
            remap_key("mimi.decoder.layers.0.weight").as_deref(),
            Some("mimi.decoder.layers.0.weight")
        );
    }

    #[test]
    fn flow_lm_flow_is_renamed_not_dropped() {
        // `flow.w_s_t` is dropped but `flow.` alone is a rename; the two must not be confused.
        assert_eq!(
            remap_key("flow_lm.flow.layers.0.weight").as_deref(),
            Some("flow_lm.flow_net.layers.0.weight")
        );
    }

    #[test]
    fn model_ext_is_read_from_a_header_only_read() {
        // One f32 of tensor data, so the header alone is not the whole file:
        // `check_model_ext` must not need the data section to parse the header.
        let write = |name: &str, header: &str| {
            let path = std::env::temp_dir().join(name);
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&0f32.to_le_bytes());
            std::fs::write(&path, &bytes).unwrap();
            path
        };
        let tensor = r#""emb":{"dtype":"F32","shape":[1,1,1],"data_offsets":[0,4]}"#;

        let matching = write(
            "ptts-loader-voice-match.safetensors",
            &format!(r#"{{"__metadata__":{{"model_ext":"abc@1"}},{tensor}}}"#),
        );
        check_model_ext(&matching, "abc@1").unwrap();
        assert!(check_model_ext(&matching, "def@2").is_err());

        // A voice from before the metadata existed is accepted as-is.
        let bare = write("ptts-loader-voice-bare.safetensors", &format!(r#"{{{tensor}}}"#));
        check_model_ext(&bare, "abc@1").unwrap();
    }

    #[test]
    fn unused_tensors_are_recognized() {
        for unused in [
            "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding",
            "mimi.quantizer.output_proj.weight",
            "mimi.encoder.layers.0.weight",
            "speaker_mimi.encoder.weight",
            "flow_lm.speaker_proj_weight",
            "mimi.downsample.conv.conv.weight",
            "mimi.downsample.something_else",
        ] {
            assert!(is_unused_by_tts_model(unused), "expected {unused} to be ignorable");
        }
        assert!(!is_unused_by_tts_model("flow_lm.transformer.layers.0.linear1.weight"));
        assert!(!is_unused_by_tts_model("mimi.decoder.layers.0.weight"));
    }
}
