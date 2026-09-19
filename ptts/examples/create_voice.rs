#[path = "audio_helpers.rs"]
mod audio_helpers;
#[path = "model_helpers.rs"]
mod model_helpers;

use anyhow::{Context, Result};
use clap::Parser;
use ptts::tts_model::MimiEnc;
use xn::Tensor;

#[derive(Parser, Debug)]
#[command(name = "create-voice")]
#[command(about = "Generate some embedding files for Pocket TTS")]
struct Args {
    #[arg(long)]
    config: String,

    #[arg(long)]
    weights: Option<String>,

    #[arg(long)]
    output: std::path::PathBuf,

    /// Voice to use: an audio file of ~10s, or a `.safetensors` file holding a
    /// `speaker_wavs` tensor of speaker-Mimi latents, `[1, C, T]`, as stored by
    /// the training pipeline. The latter skips the audio encoding and only
    /// applies the model's speaker projection.
    #[arg(long)]
    input: String,
}

fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(model_helpers::LOG_DIRECTIVES));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    let args = Args::parse();
    run(args)?;
    Ok(())
}

fn run(args: Args) -> Result<()> {
    use std::str::FromStr;

    let dev = xn::CpuDevice;
    tracing::info!("loading config from {}", args.config);
    let (cfg, model_path) = if args.config.ends_with("json") {
        let cfg: ptts::tts_model::TTSConfig =
            serde_json::from_str(&std::fs::read_to_string(&args.config)?)?;
        let config = std::fs::canonicalize(args.config)?;
        let parent = config.parent().context("config path has no parent")?;
        let model_path = match args.weights.as_ref() {
            None => parent.join("model.safetensors"),
            Some(p) => std::path::PathBuf::from_str(p)?,
        };
        (cfg, model_path)
    } else {
        let api = hf_hub::HFClientSync::new()?;
        let (owner, name) = hf_hub::split_id(&args.config);
        let repo = api.model(owner, name);
        let cfg = repo.download_file().filename("config.json").send()?;
        let cfg: ptts::tts_model::TTSConfig = serde_json::from_str(&std::fs::read_to_string(cfg)?)?;
        let model_path = match args.weights.as_ref() {
            None => repo.download_file().filename("model.safetensors").send()?,
            Some(p) => std::path::PathBuf::from_str(p)?,
        };
        (cfg, model_path)
    };
    let model_ext = cfg.model_ext();
    tracing::info!(?model_ext, "model extension");

    tracing::info!(?model_path, "loading model");
    let vb = model_helpers::load_weights::<xn::Unquantized<f32, xn::CpuDevice>>(&model_path, &dev)?;
    let mimi_enc: MimiEnc<xn::Unquantized<f32, xn::CpuDevice>> = MimiEnc::load(&vb, &cfg)?;

    let emb = if args.input.ends_with(".safetensors") {
        tracing::info!("loading speaker latents from {}", args.input);
        let latents = load_speaker_latents(&args.input, &cfg, &dev)?;
        tracing::info!(?latents, "loaded speaker latents");
        mimi_enc.embed_latents(&latents)?
    } else {
        tracing::info!("loading voice from audio file {}", args.input);
        let pcm_tensor = load_voice_audio(&args.input, &cfg, &dev)?;
        tracing::info!("encoding audio to latent");
        mimi_enc.encode_audio(&pcm_tensor)?
    };
    tracing::info!(?emb, "voice embedding");
    let tensors = std::collections::HashMap::from([("emb".to_string(), xn::TypedTensor::F32(emb))]);
    let data_info = std::collections::HashMap::from([(
        "model_ext".to_string(),
        model_ext.unwrap_or("unknown".to_string()),
    )]);
    xn::safetensors::save_with_data_info(&tensors, Some(data_info), &args.output)?;
    Ok(())
}

/// Decode, loudness-normalize and resample an audio file to the speaker encoder's rate,
/// trimmed to 10s, as a `[1, 1, samples]` tensor.
fn load_voice_audio(
    path: &str,
    cfg: &ptts::tts_model::TTSConfig,
    dev: &xn::CpuDevice,
) -> Result<Tensor<f32, xn::CpuDevice>> {
    let speaker_sr = cfg.speaker_mimi_cfg().sample_rate;
    let (mut pcm, sample_rate) = audio_helpers::pcm_decode(path)?;
    ptts::utils::normalize_loudness(&mut pcm, sample_rate)?;
    let sample_rate = sample_rate as usize;
    let pcm = if sample_rate != speaker_sr {
        audio_helpers::resample(&pcm, sample_rate, speaker_sr)?
    } else {
        pcm
    };
    tracing::info!("loaded audio with {} samples", pcm.len());
    // Trim it to 10s max.
    let pcm = if pcm.len() > speaker_sr * 10 {
        tracing::info!("trimming audio to 10 seconds");
        pcm[..speaker_sr * 10].to_vec()
    } else {
        pcm
    };
    Ok(Tensor::from_vec(pcm, (1, 1, ()), dev)?)
}

/// The `speaker_wavs` tensor of a safetensors file: speaker-Mimi latents as `[1, C, T]`, with
/// `C` the speaker encoder's dimension. A `[C, T]` tensor gets its batch dimension added.
fn load_speaker_latents(
    path: &str,
    cfg: &ptts::tts_model::TTSConfig,
    dev: &xn::CpuDevice,
) -> Result<Tensor<f32, xn::CpuDevice>> {
    const NAME: &str = "speaker_wavs";
    let vb = xn::nn::VB::load(&[path], *dev)?;
    let shape = vb
        .shape(NAME)
        .with_context(|| format!("no `{NAME}` tensor in {path}; found {:?}", vb.tensor_names()))?;
    let dims = shape.dims().to_vec();
    let latents: Tensor<f32, _> = vb.tensor(NAME, shape)?;
    let latents = match dims.as_slice() {
        [c, t] => latents.reshape((1, *c, *t))?,
        [_, _, _] => latents,
        _ => anyhow::bail!("`{NAME}` in {path} has shape {dims:?}, expected [1, C, T]"),
    };
    let dim = cfg.speaker_mimi_cfg().dimension;
    if latents.dims()[1] != dim {
        anyhow::bail!(
            "`{NAME}` in {path} has {} channels but the speaker encoder has dimension {dim}",
            latents.dims()[1]
        )
    }
    Ok(latents)
}
