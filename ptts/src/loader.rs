//! Reading checkpoints and voice embeddings off disk.
//!
//! Every frontend -- the examples, the ws-server, the Python bindings, the wasm build -- has to
//! rename the same checkpoint keys, skip the same unused tensors and unpack voice files the same
//! way. Keeping that here means a checkpoint layout change is one edit rather than four.

use crate::tts_model::TTSConfig;
use crate::{Error, Result};
use xn::nn::{Linear, Path, VB};
use xn::{Backend, BackendQ, Tensor};

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

/// Name of the speaker projection weight, as [`remap_key`] spells it.
pub const SPEAKER_PROJ_WEIGHT: &str = "flow_lm.speaker_proj_weight";

/// Tensor name of a precomputed voice embedding, as `create_voice` writes it.
pub const EMB_TENSOR: &str = "emb";
/// Tensor name of stored speaker-Mimi latents, as the training pipeline writes them.
pub const SPEAKER_WAVS_TENSOR: &str = "speaker_wavs";

/// The checkpoint's speaker projection, when it has one: the linear map from speaker-Mimi
/// latents (`speaker_mimi_cfg().dimension` wide) to the flow LM's `d_model`. It turns stored
/// `speaker_wavs` latents into the voice embedding the flow LM is conditioned on, see
/// [`load_voice_emb`]. Kept in f32 like the embeddings it produces.
pub fn load_speaker_proj<B: Backend>(
    vb: &Path<B>,
    cfg: &TTSConfig,
) -> Result<Option<Linear<f32, B>>> {
    if !vb.contains(SPEAKER_PROJ_WEIGHT) {
        return Ok(None);
    }
    let shape = (cfg.flow_lm.d_model, cfg.speaker_mimi_cfg().dimension);
    let weight = vb.tensor(SPEAKER_PROJ_WEIGHT, shape)?;
    Ok(Some(Linear::new(weight)))
}

/// What a voice file holds, told apart by tensor name.
enum VoiceTensor {
    /// A voice embedding, `[T, dim]` or `[1, T, dim]`.
    Emb,
    /// Speaker-Mimi latents, `[C, T]` or `[1, C, T]`.
    Latents,
}

/// Loads a voice embedding as `[1, T, dim]`.
///
/// Two kinds of file are understood:
///
/// - An `emb` tensor, `[T, dim]` or `[1, T, dim]`: a precomputed embedding, as `create_voice`
///   writes it. A file whose single tensor goes by another name -- the published voices call
///   theirs `audio_prompt` -- is read the same way.
/// - A `speaker_wavs` tensor, `[C, T]` or `[1, C, T]`: speaker-Mimi latents as the training
///   pipeline stores them. They are transposed to `[1, T, C]` and projected to the flow LM's
///   width by `speaker_proj`, see [`load_speaker_proj`]. Such a file cannot be used without the
///   projection -- unprojected latents make the model babble or stop at once, with no other
///   symptom -- so `None` is an error for this kind of file and is ignored for the other.
///
/// When `model_ext` is given and the file records one of its own, the two must agree -- a voice
/// conditioned on a different checkpoint produces confident nonsense rather than an error, so it
/// is worth catching here. Pass `None` to skip the check.
///
/// The result is f32 regardless of the backend's quantization; convert with `to::<Q::T>()`.
pub fn load_voice_emb<B: Backend>(
    path: &std::path::Path,
    model_ext: Option<&str>,
    speaker_proj: Option<&Linear<f32, B>>,
    dev: &B,
) -> Result<Tensor<f32, B>> {
    let vb = VB::load(&[path], dev.clone())?;
    let label = path.display().to_string();
    let emb = voice_emb_from_vb(&vb, &label, speaker_proj)?;
    if let Some(model_ext) = model_ext {
        check_model_ext(&read_safetensors_header(path)?, &label, model_ext)?;
    }
    Ok(emb)
}

/// As [`load_voice_emb`], for a voice file already in memory -- the wasm build, which has no
/// filesystem, fetches its voices.
pub fn load_voice_emb_from_bytes<B: Backend>(
    bytes: &[u8],
    model_ext: Option<&str>,
    speaker_proj: Option<&Linear<f32, B>>,
    dev: &B,
) -> Result<Tensor<f32, B>> {
    let vb = VB::from_bytes(vec![bytes.to_vec()], dev.clone())?;
    let label = "the voice file";
    let emb = voice_emb_from_vb(&vb, label, speaker_proj)?;
    if let Some(model_ext) = model_ext {
        check_model_ext(safetensors_header_from_bytes(bytes, label)?, label, model_ext)?;
    }
    Ok(emb)
}

/// The part of [`load_voice_emb`] that does not care where the file came from. `label` names
/// the file in error messages.
fn voice_emb_from_vb<B: Backend>(
    vb: &VB<B>,
    label: &str,
    speaker_proj: Option<&Linear<f32, B>>,
) -> Result<Tensor<f32, B>> {
    use xn::error::Context;

    let names = vb.tensor_names();
    let (name, kind) = if names.contains(&EMB_TENSOR) {
        (EMB_TENSOR, VoiceTensor::Emb)
    } else if names.contains(&SPEAKER_WAVS_TENSOR) {
        (SPEAKER_WAVS_TENSOR, VoiceTensor::Latents)
    } else {
        let first = names.first().ok_or_else(|| {
            Error::invalid_data(format!("no tensors found in voice file {label}"))
        })?;
        (*first, VoiceTensor::Emb)
    };
    let shape = vb.shape(name).context("voice tensor not found")?;
    let dims = shape.dims().to_vec();
    let tensor: Tensor<f32, B> = vb.tensor(name, shape)?;
    let tensor = match dims.as_slice() {
        [a, b] => tensor.reshape((1, *a, *b))?,
        [_, _, _] => tensor,
        _ => {
            return Err(Error::invalid_data(format!(
                "voice tensor `{name}` in {label} has shape {dims:?}, expected two or three \
                 dimensions"
            )));
        }
    };
    let emb = match kind {
        VoiceTensor::Emb => tensor,
        VoiceTensor::Latents => {
            // [1, C, T] -> [1, T, C]
            let latents = tensor.transpose(1, 2)?.contiguous()?;
            let Some(proj) = speaker_proj else {
                return Err(Error::invalid_data(format!(
                    "{label} holds `{SPEAKER_WAVS_TENSOR}` latents, but this checkpoint has no \
                     speaker projection (`{SPEAKER_PROJ_WEIGHT}`) to turn them into a voice \
                     embedding. A GGUF written by an older `quantize --no-mimi-encoder` dropped \
                     it: regenerate the GGUF from the safetensors checkpoint, or use a \
                     precomputed `{EMB_TENSOR}` voice."
                )));
            };
            let channels = latents.dim(2usize)?;
            let in_dim = proj.weight().dims()[1];
            if channels != in_dim {
                return Err(Error::invalid_data(format!(
                    "`{SPEAKER_WAVS_TENSOR}` in {label} has {channels} channels but the speaker \
                     projection takes {in_dim}"
                )));
            }
            proj.forward(&latents)?
        }
    };
    Ok(emb)
}

/// Fails if the voice file's safetensors `header` records a `model_ext` other than
/// `model_ext`. A file that records none is accepted: older voices predate the metadata.
fn check_model_ext(header: &[u8], label: &str, model_ext: &str) -> Result<()> {
    // Not `SafeTensors::read_metadata`: it validates that the buffer holds the
    // tensor data as well as the header, which is exactly what is not read here.
    let header: serde_json::Value = serde_json::from_slice(header).map_err(|e| {
        Error::invalid_data(format!("cannot parse safetensors header of {label}: {e}"))
    })?;
    if let Some(voice_model_ext) = header.get("__metadata__").and_then(|m| m.get("model_ext"))
        && let Some(voice_model_ext) = voice_model_ext.as_str()
    {
        tracing::info!(?voice_model_ext, "voice embedding model_ext from metadata");
        if voice_model_ext != model_ext {
            return Err(Error::invalid_data(format!(
                "voice embedding model_ext '{voice_model_ext}' does not match config model_ext \
                 '{model_ext}'"
            )));
        }
    }
    Ok(())
}

/// The JSON header of a safetensors file held in memory: the 8-byte little-endian header
/// length, then that many bytes.
fn safetensors_header_from_bytes<'a>(bytes: &'a [u8], label: &str) -> Result<&'a [u8]> {
    let not_safetensors =
        || Error::invalid_data(format!("{label} does not look like a safetensors file"));
    let len_bytes: [u8; 8] =
        bytes.get(..8).and_then(|b| b.try_into().ok()).ok_or_else(not_safetensors)?;
    let header_len =
        usize::try_from(u64::from_le_bytes(len_bytes)).map_err(|_| not_safetensors())?;
    bytes.get(8..8usize.saturating_add(header_len)).ok_or_else(not_safetensors)
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
        return Err(Error::invalid_data(format!(
            "{} does not look like a safetensors file",
            path.display()
        )));
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
        let header = read_safetensors_header(&matching).unwrap();
        check_model_ext(&header, "voice", "abc@1").unwrap();
        assert!(check_model_ext(&header, "voice", "def@2").is_err());

        // A voice from before the metadata existed is accepted as-is.
        let bare = write("ptts-loader-voice-bare.safetensors", &format!(r#"{{{tensor}}}"#));
        check_model_ext(&read_safetensors_header(&bare).unwrap(), "voice", "abc@1").unwrap();
    }

    #[test]
    fn voice_emb_loads_from_bytes() {
        let file = |header: &str| {
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0f32, 1., 2., 3.].map(f32::to_le_bytes).concat());
            bytes
        };
        let tensor = r#""emb":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}"#;
        let bytes = file(&format!(r#"{{"__metadata__":{{"model_ext":"abc@1"}},{tensor}}}"#));

        let emb = load_voice_emb_from_bytes(&bytes, Some("abc@1"), None, &xn::CPU).unwrap();
        assert_eq!(emb.dims(), [1, 2, 2]);
        assert_eq!(emb.to_vec().unwrap(), [0., 1., 2., 3.]);
        assert!(load_voice_emb_from_bytes(&bytes, Some("def@2"), None, &xn::CPU).is_err());

        // Too short to hold a header length, and a length past the end of the buffer.
        assert!(safetensors_header_from_bytes(&[1, 2, 3], "voice").is_err());
        assert!(safetensors_header_from_bytes(&u64::MAX.to_le_bytes(), "voice").is_err());
    }

    #[test]
    fn unused_tensors_are_recognized() {
        for unused in [
            "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding",
            "mimi.quantizer.output_proj.weight",
            "mimi.encoder.layers.0.weight",
            "speaker_mimi.encoder.weight",
            "mimi.downsample.conv.conv.weight",
            "mimi.downsample.something_else",
        ] {
            assert!(is_unused_by_tts_model(unused), "expected {unused} to be ignorable");
        }
        assert!(!is_unused_by_tts_model("flow_lm.transformer.layers.0.linear1.weight"));
        assert!(!is_unused_by_tts_model("mimi.decoder.layers.0.weight"));
        // Read by `TTSModel::load` for `load_voice_emb`, so no longer ignorable.
        assert!(!is_unused_by_tts_model(SPEAKER_PROJ_WEIGHT));
    }

    fn save_voice(name: &str, file: &str, t: Tensor<f32, xn::CpuDevice>) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(file);
        let tensors =
            std::collections::HashMap::from([(name.to_string(), xn::TypedTensor::F32(t))]);
        xn::safetensors::save_with_data_info(&tensors, None, &path).unwrap();
        path
    }

    fn values(t: &Tensor<f32, xn::CpuDevice>) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    #[test]
    fn voice_emb_is_batched_whatever_its_name() {
        let dev = xn::CpuDevice;
        let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
        for (name, file) in [
            ("emb", "ptts-loader-voice-emb.safetensors"),
            ("audio_prompt", "ptts-loader-voice-legacy.safetensors"),
        ] {
            let t = Tensor::from_vec(data.clone(), (3, 2), &dev).unwrap();
            let path = save_voice(name, file, t);
            let emb = load_voice_emb(&path, None, None, &dev).unwrap();
            assert_eq!(emb.dims(), &[1, 3, 2], "{name}");
            assert_eq!(values(&emb), data, "{name}");
        }
    }

    #[test]
    fn speaker_wavs_are_transposed_and_projected() {
        let dev = xn::CpuDevice;
        // C = 2 channels, T = 3 frames, latents[c][t] = 10 * c + t.
        let latents = Tensor::from_vec(vec![0., 1., 2., 10., 11., 12.], (2, 3), &dev).unwrap();
        let path = save_voice("speaker_wavs", "ptts-loader-voice-latents.safetensors", latents);

        // Latents are unusable without the projection.
        let err = load_voice_emb(&path, None, None, &dev).unwrap_err().to_string();
        assert!(err.contains("no speaker projection"), "{err}");

        // W = [[1, 0], [0, 2]] maps [x0, x1] to [x0, 2 * x1].
        let w = Tensor::from_vec(vec![1., 0., 0., 2.], (2, 2), &dev).unwrap();
        let emb = load_voice_emb(&path, None, Some(&Linear::new(w)), &dev).unwrap();
        assert_eq!(emb.dims(), &[1, 3, 2]);
        assert_eq!(values(&emb), vec![0., 20., 1., 22., 2., 24.]);

        // A projection expecting three channels rejects two-channel latents.
        let w = Tensor::from_vec(vec![0.; 6], (2, 3), &dev).unwrap();
        let err = load_voice_emb(&path, None, Some(&Linear::new(w)), &dev).unwrap_err().to_string();
        assert!(err.contains("2 channels") && err.contains("takes 3"), "{err}");
    }
}
