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

    /// Voice to use
    #[arg(long)]
    input: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
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
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(args.config);
        let cfg = repo.get("config.json")?;
        let cfg: ptts::tts_model::TTSConfig = serde_json::from_str(&std::fs::read_to_string(cfg)?)?;
        let model_path = match args.weights.as_ref() {
            None => repo.get("model.safetensors")?,
            Some(p) => std::path::PathBuf::from_str(p)?,
        };
        (cfg, model_path)
    };
    let model_ext = cfg.model_ext();
    tracing::info!(?model_ext, "model extension");
    tracing::info!("loading voice from audio file {}", args.input);

    let speaker_sr = cfg.speaker_mimi_cfg().sample_rate;
    let (mut pcm, sample_rate) = audio_helpers::pcm_decode(&args.input)?;
    ptts::utils::normalize_loudness(&mut pcm, sample_rate)?;
    let sample_rate = sample_rate as usize;
    let pcm = if sample_rate != speaker_sr {
        audio_helpers::resample(&pcm, sample_rate, speaker_sr)?
    } else {
        pcm
    };
    tracing::info!("loaded audio with {} samples", pcm.len());
    let max_len = (speaker_sr as f32 * cfg.audio_prompt_max_duration).round() as usize;
    let pcm = if pcm.len() > max_len {
        let cut = audio_helpers::quiet_cut_point(&pcm, max_len, speaker_sr);
        tracing::info!(
            max_duration = cfg.audio_prompt_max_duration,
            cut_s = cut as f32 / speaker_sr as f32,
            "trimming audio"
        );
        pcm[..cut].to_vec()
    } else {
        pcm
    };
    let pcm_tensor = Tensor::from_vec(pcm, (1, 1, ()), &dev)?.to::<f32>()?;

    tracing::info!(?model_path, "loading model");
    let vb = model_helpers::load_weights::<xn::Unquantized<f32, xn::CpuDevice>>(&model_path, &dev)?;
    let mimi_enc: MimiEnc<xn::Unquantized<f32, xn::CpuDevice>> = MimiEnc::load(&vb, &cfg)?;
    tracing::info!("encoding audio to latent");
    let emb = mimi_enc.encode_audio(&pcm_tensor)?;
    tracing::info!(?emb, "encoded audio to latent");
    let tensors = std::collections::HashMap::from([("emb".to_string(), xn::TypedTensor::F32(emb))]);
    let data_info = std::collections::HashMap::from([(
        "model_ext".to_string(),
        model_ext.unwrap_or("unknown".to_string()),
    )]);
    xn::safetensors::save_with_data_info(&tensors, Some(data_info), &args.output)?;
    Ok(())
}
