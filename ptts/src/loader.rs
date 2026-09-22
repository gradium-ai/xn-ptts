//! Reading checkpoints and voice embeddings off disk.
//!
//! Every frontend -- the examples, the ws-server, the Python bindings, the wasm build -- has to
//! rename the same checkpoint keys, skip the same unused tensors and unpack voice files the same
//! way. Keeping that here means a checkpoint layout change is one edit rather than four.

use crate::tts_model::TTSConfig;
use xn::nn::{Linear, Path, VB};
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
    use xn::error::Context;

    let vb = VB::load(&[path], dev.clone())?;
    let names = vb.tensor_names();
    let (name, kind) = if names.contains(&EMB_TENSOR) {
        (EMB_TENSOR, VoiceTensor::Emb)
    } else if names.contains(&SPEAKER_WAVS_TENSOR) {
        (SPEAKER_WAVS_TENSOR, VoiceTensor::Latents)
    } else {
        let first = names.first().context("no tensors found in voice embedding file")?;
        (*first, VoiceTensor::Emb)
    };
    let shape = vb.shape(name).context("voice tensor not found")?;
    let dims = shape.dims().to_vec();
    let tensor: Tensor<f32, B> = vb.tensor(name, shape)?;
    let tensor = match dims.as_slice() {
        [a, b] => tensor.reshape((1, *a, *b))?,
        [_, _, _] => tensor,
        _ => xn::bail!(
            "voice tensor `{name}` in {} has shape {dims:?}, expected two or three dimensions",
            path.display()
        ),
    };
    let emb = match kind {
        VoiceTensor::Emb => tensor,
        VoiceTensor::Latents => {
            // [1, C, T] -> [1, T, C]
            let latents = tensor.transpose(1, 2)?.contiguous()?;
            let Some(proj) = speaker_proj else {
                xn::bail!(
                    "{} holds `{SPEAKER_WAVS_TENSOR}` latents, but this checkpoint has no speaker \
                     projection (`{SPEAKER_PROJ_WEIGHT}`) to turn them into a voice embedding. A \
                     GGUF written by an older `quantize --no-mimi-encoder` dropped it: regenerate \
                     the GGUF from the safetensors checkpoint, or use a precomputed `{EMB_TENSOR}` \
                     voice.",
                    path.display()
                )
            };
            let channels = latents.dim(2usize)?;
            let in_dim = proj.weight().dims()[1];
            if channels != in_dim {
                xn::bail!(
                    "`{SPEAKER_WAVS_TENSOR}` in {} has {channels} channels but the speaker \
                     projection takes {in_dim}",
                    path.display()
                )
            }
            proj.forward(&latents)?
        }
    };
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

// ---------------------------------------------------------------------------
// Locating a checkpoint
// ---------------------------------------------------------------------------

/// Weight file names tried, in order, when the caller names none.
///
/// A search order across the layouts in circulation, not a promise about any one release: a
/// checkpoint is free to call its weights anything, and [`ModelSource::weights`] names the file
/// when it does.
pub const WEIGHT_CANDIDATES: &[&str] =
    &["model.safetensors", "model.q8.gguf", "tts_b6369a24.safetensors"];

/// Tokenizer file names tried, in order. `tokenizer.json` is the Hugging Face `tokenizers`
/// format and `tokenizer.model` is SentencePiece; [`crate::tok::Tok`] picks the reader by
/// extension, so the order follows which of the two this build can read. A checkpoint that ships
/// both is common, and picking the one whose feature is off would fail a load that had a usable
/// tokenizer sitting beside it.
pub const TOKENIZER_CANDIDATES: &[&str] = if cfg!(feature = "hf") {
    &["tokenizer.json", "tokenizer.model"]
} else {
    &["tokenizer.model", "tokenizer.json"]
};

/// Subdirectories searched for voice embeddings, relative to the checkpoint.
pub const VOICE_DIRS: &[&str] = &["voices", "embeddings"];

/// A voice file that sits beside the weights rather than in a voice directory, registered under
/// the name `default`.
pub const DEFAULT_VOICE_FILE: &str = "default-voice.safetensors";

/// Where a checkpoint's files come from.
///
/// [`load_weights`] and [`load_voice_emb`] read files someone has already located. This decides
/// *which* files, across the two layouts in circulation -- a Hugging Face model repo and a local
/// directory -- so that every frontend does not carry its own copy of the search.
///
/// ```no_run
/// # fn main() -> xn::Result<()> {
/// use ptts::loader::ModelSource;
///
/// // The published checkpoint, from the Hub.
/// let tts = ModelSource::hub("kyutai/pocket-tts").resolve()?.builder().build()?;
///
/// // A specific checkpoint inside a repo that ships several, with its quantized weights.
/// let checkpoint = ModelSource::hub("kyutai/pocket-tts")
///     .subdir("languages/italian")
///     .weights("model.q8.gguf")
///     .resolve()?;
/// # Ok(())
/// # }
/// ```
///
/// Callers that already have explicit paths do not need this: pass them to
/// [`crate::synth::SynthBuilder::new`] directly.
#[derive(Clone, Debug)]
pub struct ModelSource {
    location: Location,
    subdir: Option<String>,
    weights: Option<String>,
    revision: Option<String>,
}

/// The two checkpoint layouts [`ModelSource`] knows how to search.
#[derive(Clone, Debug)]
enum Location {
    /// A Hugging Face model repo id, e.g. `kyutai/pocket-tts`.
    Hub(String),
    /// A local directory.
    Dir(std::path::PathBuf),
}

impl ModelSource {
    /// A Hugging Face model repo, by id. Requires the `hub` feature.
    ///
    /// Only the files that get used are downloaded, so naming the weights file with
    /// [`Self::weights`] avoids fetching the f32 weights of a repo that also ships a quantized
    /// GGUF.
    pub fn hub(repo_id: impl Into<String>) -> Self {
        Self {
            location: Location::Hub(repo_id.into()),
            subdir: None,
            weights: None,
            revision: None,
        }
    }

    /// A local directory holding a weights file, optionally `config.json`, a tokenizer and a
    /// `voices/` or `embeddings/` subdirectory.
    pub fn dir(path: impl Into<std::path::PathBuf>) -> Self {
        Self { location: Location::Dir(path.into()), subdir: None, weights: None, revision: None }
    }

    /// Look inside this subdirectory of the repo or directory, for checkpoints that ship several
    /// side by side -- `languages/italian`, say. Everything a checkpoint owns is searched for
    /// under it: the config, the weights, the tokenizer and the voice directories.
    pub fn subdir(mut self, subdir: impl Into<String>) -> Self {
        self.subdir = Some(subdir.into());
        self
    }

    /// Name the weights file, e.g. `model.q8.gguf`, for a checkpoint that ships several. When
    /// unset, the first of [`WEIGHT_CANDIDATES`] that exists is used.
    pub fn weights(mut self, filename: impl Into<String>) -> Self {
        self.weights = Some(filename.into());
        self
    }

    /// Pin a git revision -- a branch, tag or commit sha. Hub sources only; defaults to the
    /// repo's main branch.
    ///
    /// Worth setting for anything reproducible: the published checkpoints are updated in place,
    /// so `main` is not a fixed set of weights.
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// The subdirectory, trimmed of a trailing slash the caller may have written and of the
    /// empty string. Both searches go through this so a subdir means the same thing on disk as
    /// it does on the Hub: if the two disagreed, a checkpoint would resolve locally and miss
    /// remotely.
    fn prefix(&self) -> Option<&str> {
        self.subdir.as_deref().map(|s| s.trim_end_matches('/')).filter(|s| !s.is_empty())
    }

    /// Repo-relative path of `name` within the source, honouring [`Self::subdir`]. The local
    /// search joins [`Self::prefix`] onto a `Path` instead, so this is only reached on the Hub
    /// path -- but the contract it encodes is tested in every build.
    #[cfg_attr(not(feature = "hub"), allow(dead_code))]
    fn at(&self, name: &str) -> String {
        match self.prefix() {
            Some(prefix) => format!("{prefix}/{name}"),
            None => name.to_string(),
        }
    }

    /// Locate every file this source provides and parse the config, downloading what is needed.
    pub fn resolve(&self) -> Result<Checkpoint> {
        match &self.location {
            Location::Dir(dir) => self.resolve_dir(dir),
            Location::Hub(repo_id) => self.resolve_hub(repo_id),
        }
    }

    fn resolve_dir(&self, dir: &std::path::Path) -> Result<Checkpoint> {
        if !dir.is_dir() {
            xn::bail!("not a directory: {}", dir.display())
        }
        let root = match self.prefix() {
            Some(prefix) => dir.join(prefix),
            None => dir.to_path_buf(),
        };
        if !root.is_dir() {
            xn::bail!(
                "no subdirectory `{}` in {}",
                self.subdir.as_deref().unwrap_or(""),
                dir.display()
            )
        }

        let config_path = root.join("config.json");
        let (config, config_read) = if config_path.is_file() {
            (read_config(&config_path)?, true)
        } else {
            (assumed_config(), false)
        };

        let weights = match self.weights.as_deref() {
            Some(name) => {
                let path = root.join(name);
                if !path.is_file() {
                    xn::bail!("no weights file `{name}` in {}", root.display())
                }
                path
            }
            None => WEIGHT_CANDIDATES
                .iter()
                .map(|name| root.join(name))
                .find(|path| path.is_file())
                .ok_or_else(|| {
                    xn::Error::msg(format!(
                        "no weights file in {}; expected one of {}",
                        root.display(),
                        WEIGHT_CANDIDATES.join(", ")
                    ))
                })?,
        };

        let tokenizer =
            TOKENIZER_CANDIDATES.iter().map(|name| root.join(name)).find(|path| path.is_file());

        let mut voices = vec![];
        for sub in VOICE_DIRS {
            let Ok(entries) = std::fs::read_dir(root.join(sub)) else { continue };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                    continue;
                }
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    voices.push((name.to_string(), path));
                }
            }
        }
        let default_voice = root.join(DEFAULT_VOICE_FILE);
        if default_voice.is_file() {
            voices.push(("default".to_string(), default_voice));
        }
        voices.sort();
        voices.dedup_by(|a, b| a.0 == b.0);

        Ok(Checkpoint { config, config_read, weights, tokenizer, voices })
    }
}

/// The config assumed for a checkpoint that ships no `config.json`.
///
/// `temp` is not read by the runtime -- sampling temperature reaches the model through
/// [`crate::synth::SynthBuilder::temperature`] -- so any value does.
fn assumed_config() -> TTSConfig {
    TTSConfig::v202601(0.5)
}

fn read_config(path: &std::path::Path) -> Result<TTSConfig> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| xn::Error::msg(format!("cannot read config {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| xn::Error::msg(format!("cannot parse config {}: {e}", path.display())))
}

/// A checkpoint whose files have been located and whose config is parsed.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    /// The checkpoint's config, read from `config.json` or assumed -- see [`Self::config_read`].
    pub config: TTSConfig,
    /// True when `config` was read from the source, false when it was assumed because the source
    /// ships none. An assumed config describes one architecture; a checkpoint built to another
    /// will fail to load rather than load wrongly, but the error is clearer when a caller can
    /// say which case it is in.
    pub config_read: bool,
    pub weights: std::path::PathBuf,
    /// `None` when the source carries no tokenizer file; the caller must then pass one to
    /// [`crate::synth::SynthBuilder::tokenizer`].
    pub tokenizer: Option<std::path::PathBuf>,
    /// Voice name to embedding file, sorted by name.
    pub voices: Vec<(String, std::path::PathBuf)>,
}

impl Checkpoint {
    /// A builder over this checkpoint, with its config, weights and tokenizer file set.
    ///
    /// The bundled voices are deliberately not registered here: see [`Self::register_voices`].
    pub fn builder(&self) -> crate::synth::SynthBuilder {
        let mut builder = crate::synth::SynthBuilder::new(self.config.clone(), &self.weights);
        if let Some(tokenizer) = self.tokenizer.as_ref() {
            builder = builder.tokenizer_file(tokenizer);
        }
        builder
    }

    /// Register the bundled voices on a loaded model, and pick a default if it has none.
    ///
    /// A voice that fails to load is warned about rather than fatal: one bad embedding -- an
    /// interrupted download, a voice from another checkpoint -- should not make the model
    /// unusable. A voice the user named explicitly goes through
    /// [`crate::synth::SynthBuilder::add_voice`], where a failure *is* fatal.
    ///
    /// The default goes to a voice named `default` -- what a checkpoint's
    /// [`DEFAULT_VOICE_FILE`] is registered as -- and otherwise to the first by name. Setting it
    /// here rather than on the builder is what makes a bare [`crate::synth::Synth::say`] work:
    /// `build` picks its own default before these voices exist, so it finds none.
    ///
    /// Returns the number of voices registered.
    pub fn register_voices(&self, tts: &mut crate::synth::Synth) -> usize {
        let mut registered = vec![];
        for (name, path) in self.voices.iter() {
            match tts.add_voice_file(name, path) {
                Ok(()) => registered.push(name.as_str()),
                Err(e) => tracing::warn!(voice = %name, error = %e, "skipping voice embedding"),
            }
        }
        if tts.default_voice().is_none()
            && let Some(pick) = registered.iter().find(|n| **n == "default").or(registered.first())
            // Registered a moment ago, so this cannot fail; a warning beats an unwrap either way.
            && let Err(e) = tts.set_default_voice(pick)
        {
            tracing::warn!(error = %e, "could not set the default voice");
        }
        registered.len()
    }
}

#[cfg(not(feature = "hub"))]
impl ModelSource {
    fn resolve_hub(&self, repo_id: &str) -> Result<Checkpoint> {
        xn::bail!(
            "cannot load `{repo_id}`: reading from the Hugging Face Hub needs the `hub` feature \
             of the `ptts` crate. Either enable it, or download the checkpoint yourself and use \
             `ModelSource::dir`."
        )
    }
}

#[cfg(feature = "hub")]
impl ModelSource {
    fn resolve_hub(&self, repo_id: &str) -> Result<Checkpoint> {
        let repo = HubRepo::open(repo_id, self.revision.clone())?;
        tracing::info!(repo_id, subdir = self.prefix(), "resolving checkpoint on the Hub");

        // Listed before anything is fetched. The Hub serves a repo's file tree without
        // authentication even when the weights themselves are gated, so this both tells us what
        // the checkpoint actually ships -- rather than guessing names and reading a failed
        // download as "absent" -- and leaves the first real 401 to `get`, which explains it.
        let listing = repo.list(self.prefix().unwrap_or(""))?;
        let has = |name: &str| listing.iter().any(|f| f == name);

        let weights = match self.weights.as_deref() {
            Some(name) => {
                if !has(name) {
                    xn::bail!("{}", self.no_such_file(repo_id, name, &listing))
                }
                name.to_string()
            }
            None => match WEIGHT_CANDIDATES.iter().find(|name| has(name)) {
                Some(name) => name.to_string(),
                None => xn::bail!("{}", self.no_weights(repo_id, &listing)),
            },
        };
        // The first fetch, and the one that decides whether this user can read the repo at all.
        let weights = repo.get(&self.at(&weights))?;

        let (config, config_read) = match has("config.json") {
            true => (read_config(&repo.get(&self.at("config.json"))?)?, true),
            false => (assumed_config(), false),
        };

        let tokenizer = match TOKENIZER_CANDIDATES.iter().find(|name| has(name)) {
            Some(name) => Some(repo.get(&self.at(name))?),
            None => None,
        };

        // Listed rather than guessed by name: the published repo ships eight voices at its root
        // and twenty-six in each `languages/*` checkpoint, so a hardcoded list would reach a
        // fraction of them and would go stale with every release.
        let mut voices = vec![];
        for sub in VOICE_DIRS {
            let dir = self.at(sub);
            let Ok(files) = repo.list(&dir) else { continue };
            for file in files {
                let Some(name) = file.strip_suffix(".safetensors") else { continue };
                let (name, path) = (name.to_string(), format!("{dir}/{file}"));
                match repo.get(&path) {
                    Ok(path) => voices.push((name, path)),
                    Err(e) => tracing::warn!(voice = %name, error = %e, "skipping voice embedding"),
                }
            }
        }
        if has(DEFAULT_VOICE_FILE) {
            voices.push(("default".to_string(), repo.get(&self.at(DEFAULT_VOICE_FILE))?));
        }
        voices.sort();
        voices.dedup_by(|a, b| a.0 == b.0);

        Ok(Checkpoint { config, config_read, weights, tokenizer, voices })
    }

    /// Message for a weights file the caller named that the checkpoint does not have.
    fn no_such_file(&self, repo_id: &str, name: &str, listing: &[String]) -> String {
        format!("no file `{name}` in `{repo_id}`{}. It holds: {}", self.under(), summarize(listing))
    }

    /// Message for a checkpoint with no weights file under any name we know.
    ///
    /// Names what the checkpoint does hold: the usual cause is a repo that keeps its checkpoints
    /// in subdirectories, and seeing `languages` in the listing is what tells a reader to reach
    /// for [`Self::subdir`].
    fn no_weights(&self, repo_id: &str, listing: &[String]) -> String {
        format!(
            "no weights file in `{repo_id}`{}; expected one of {}. It holds: {}. Name the file \
             with `ModelSource::weights`, or point at a subdirectory with `ModelSource::subdir`.",
            self.under(),
            WEIGHT_CANDIDATES.join(", "),
            summarize(listing)
        )
    }

    fn under(&self) -> String {
        self.prefix().map(|p| format!(" under `{p}`")).unwrap_or_default()
    }
}

/// A listing, shortened: enough to recognize the layout, not a directory dump.
#[cfg(feature = "hub")]
fn summarize(listing: &[String]) -> String {
    const MAX: usize = 12;
    if listing.is_empty() {
        return "nothing".to_string();
    }
    if listing.len() <= MAX {
        return listing.join(", ");
    }
    format!("{}, and {} more", listing[..MAX].join(", "), listing.len() - MAX)
}

/// A Hugging Face model repo, wrapped so a download failure names the repo and the file.
///
/// `hf_hub` says so for a missing file but not for an HTTP or authentication failure, which is
/// exactly the case a first-time user hits: the published checkpoint is gated, and an
/// unauthenticated fetch comes back as a bare 401.
#[cfg(feature = "hub")]
struct HubRepo {
    repo: hf_hub::HFRepositorySync<hf_hub::repository::RepoTypeModel>,
    repo_id: String,
    revision: Option<String>,
}

#[cfg(feature = "hub")]
impl HubRepo {
    /// The client reads `HF_TOKEN`, `HF_ENDPOINT` and the cache location from the environment,
    /// falling back to the token `huggingface-cli login` stores.
    fn open(repo_id: &str, revision: Option<String>) -> Result<Self> {
        let client = hf_hub::HFClientSync::new()
            .map_err(|e| xn::Error::msg(format!("cannot reach the Hugging Face Hub: {e}")))?;
        let (owner, name) = hf_hub::split_id(repo_id);
        Ok(Self { repo: client.model(owner, name), repo_id: repo_id.to_string(), revision })
    }

    /// Download `filename`, or find it in the local cache.
    fn get(&self, filename: &str) -> Result<std::path::PathBuf> {
        self.repo
            .download_file()
            .filename(filename)
            .maybe_revision(self.revision.clone())
            .send()
            .map_err(|e| xn::Error::msg(self.explain(filename, &e)))
    }

    /// Names of the entries directly inside `dir`, sorted, relative to `dir` itself. Pass `""`
    /// for the repo root. Directories are included, so a caller can tell an empty checkpoint
    /// from one whose files are a level down.
    fn list(&self, dir: &str) -> Result<Vec<String>> {
        let entries = self
            .repo
            .list_tree()
            .maybe_revision(self.revision.clone())
            .maybe_path_in_repo((!dir.is_empty()).then(|| dir.to_string()))
            .recursive(false)
            .send()
            .map_err(|e| {
                let what = if dir.is_empty() { "the file listing" } else { dir };
                xn::Error::msg(self.explain(what, &e))
            })?;
        // The Hub returns repo-relative paths; callers want names within `dir`.
        let strip = |path: String| match dir.is_empty() {
            true => Some(path),
            false => path.strip_prefix(dir)?.trim_start_matches('/').to_string().into(),
        };
        let mut names: Vec<String> = entries
            .into_iter()
            .filter_map(|entry| match entry {
                hf_hub::repository::RepoTreeEntry::File { path, .. } => strip(path),
                hf_hub::repository::RepoTreeEntry::Directory { path, .. } => strip(path),
            })
            .filter(|name| !name.is_empty())
            .collect();
        names.sort();
        Ok(names)
    }

    /// Turn a Hub failure into something a reader can act on.
    ///
    /// Authentication is the one worth spelling out: the published checkpoint is gated, so a
    /// rejected fetch is the first thing a new user meets, and neither "Authentication required"
    /// nor a bare 401 says what to do about it. Matched on the typed variants rather than on the
    /// message, which is not ours and can be reworded.
    fn explain(&self, what: &str, error: &hf_hub::HFError) -> String {
        use hf_hub::HFError;

        let repo_id = &self.repo_id;
        let base = format!("failed to fetch {what} from `{repo_id}`: {error}");
        match error {
            HFError::AuthRequired { .. } | HFError::Forbidden { .. } => format!(
                "{base}\n\
                 This repo is gated. Accept its terms at https://huggingface.co/{repo_id}, then \
                 authenticate with `huggingface-cli login` or by setting HF_TOKEN."
            ),
            HFError::RepoNotFound { .. } => format!(
                "{base}\n\
                 No such repo, or it is private and this token cannot see it."
            ),
            HFError::RevisionNotFound { .. } => format!(
                "{base}\n\
                 No such revision `{}` in `{repo_id}`.",
                self.revision.as_deref().unwrap_or("main")
            ),
            _ => base,
        }
    }
}

#[cfg(test)]
mod source_tests {
    use super::*;

    /// A directory laid out like a checkpoint, under a unique name so tests do not collide.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ptts-source-{name}"));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_missing_directory_names_itself() {
        let err =
            ModelSource::dir("/definitely/not/a/model/dir").resolve().unwrap_err().to_string();
        assert!(err.contains("/definitely/not/a/model/dir"), "{err}");
    }

    #[test]
    fn an_empty_directory_lists_the_weight_files_it_looked_for() {
        let dir = scratch("empty");
        let err = ModelSource::dir(&dir).resolve().unwrap_err().to_string();
        for candidate in WEIGHT_CANDIDATES {
            assert!(err.contains(candidate), "should list `{candidate}`: {err}");
        }
    }

    #[test]
    fn a_directory_without_a_config_says_the_config_was_assumed() {
        let dir = scratch("no-config");
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        let checkpoint = ModelSource::dir(&dir).resolve().unwrap();
        assert!(!checkpoint.config_read);
        assert_eq!(checkpoint.weights, dir.join("model.safetensors"));
        assert!(checkpoint.tokenizer.is_none());
    }

    #[test]
    fn a_config_beside_the_weights_is_read() {
        let dir = scratch("with-config");
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        let mut config = assumed_config();
        config.lsd_decode_steps = 17;
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();

        let checkpoint = ModelSource::dir(&dir).resolve().unwrap();
        assert!(checkpoint.config_read);
        assert_eq!(checkpoint.config.lsd_decode_steps, 17);
    }

    #[test]
    fn voices_are_collected_from_either_directory_and_sorted() {
        let dir = scratch("voices");
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        std::fs::create_dir_all(dir.join("embeddings")).unwrap();
        std::fs::create_dir_all(dir.join("voices")).unwrap();
        std::fs::write(dir.join("embeddings/marius.safetensors"), b"").unwrap();
        std::fs::write(dir.join("voices/alba.safetensors"), b"").unwrap();
        // Not a voice: the extension is what decides.
        std::fs::write(dir.join("voices/README.md"), b"").unwrap();
        std::fs::write(dir.join(DEFAULT_VOICE_FILE), b"").unwrap();

        let checkpoint = ModelSource::dir(&dir).resolve().unwrap();
        let names: Vec<&str> = checkpoint.voices.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["alba", "default", "marius"]);
    }

    #[test]
    fn a_named_weights_file_is_required_to_exist() {
        let dir = scratch("named-weights");
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        let err =
            ModelSource::dir(&dir).weights("model.q4k.gguf").resolve().unwrap_err().to_string();
        assert!(err.contains("model.q4k.gguf"), "{err}");
    }

    #[test]
    fn a_subdir_scopes_every_file_the_checkpoint_owns() {
        let dir = scratch("subdir");
        // A decoy at the root: picking it up would mean the subdir was ignored.
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        std::fs::create_dir_all(dir.join("languages/italian/embeddings")).unwrap();
        std::fs::write(dir.join("languages/italian/model.safetensors"), b"").unwrap();
        std::fs::write(dir.join("languages/italian/tokenizer.model"), b"").unwrap();
        std::fs::write(dir.join("languages/italian/embeddings/lola.safetensors"), b"").unwrap();

        let checkpoint = ModelSource::dir(&dir).subdir("languages/italian").resolve().unwrap();
        assert_eq!(checkpoint.weights, dir.join("languages/italian/model.safetensors"));
        assert_eq!(checkpoint.voices.len(), 1, "{:?}", checkpoint.voices);
        assert_eq!(checkpoint.voices[0].0, "lola");
        assert!(checkpoint.tokenizer.is_some());
    }

    #[test]
    fn a_missing_subdir_names_it() {
        let dir = scratch("missing-subdir");
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        let err =
            ModelSource::dir(&dir).subdir("languages/klingon").resolve().unwrap_err().to_string();
        assert!(err.contains("languages/klingon"), "{err}");
    }

    #[test]
    fn at_joins_with_and_without_a_subdir() {
        let plain = ModelSource::dir(".");
        assert_eq!(plain.at("config.json"), "config.json");
        let nested = ModelSource::dir(".").subdir("languages/italian");
        assert_eq!(nested.at("config.json"), "languages/italian/config.json");
        // A trailing slash is the caller's to get wrong, not a reason to produce `a//b`.
        let slashed = ModelSource::dir(".").subdir("languages/italian/");
        assert_eq!(slashed.at("config.json"), "languages/italian/config.json");
    }

    #[cfg(not(feature = "hub"))]
    #[test]
    fn without_the_hub_feature_the_error_says_which_feature() {
        let err = ModelSource::hub("kyutai/pocket-tts").resolve().unwrap_err().to_string();
        assert!(err.contains("hub"), "{err}");
        assert!(err.contains("kyutai/pocket-tts"), "{err}");
    }
}
