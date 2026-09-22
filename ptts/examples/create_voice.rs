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

    /// Voice to use: an audio file of ~10s, or a `.safetensors` file holding
    /// either a `speaker_wavs` tensor of speaker-Mimi latents, `[1, C, T]`, as
    /// stored by the training pipeline, or an `emb` tensor. The former skips
    /// the audio encoding and only applies the model's speaker projection.
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
    let dev = xn::CpuDevice;
    tracing::info!("loading config from {}", args.config);

    // `--config` is either a local `config.json` -- in which case the checkpoint is the
    // directory holding it -- or a Hub repo id. `ModelSource` searches both the same way.
    let source = if args.config.ends_with("json") {
        let config = std::fs::canonicalize(&args.config)?;
        let parent = config.parent().context("config path has no parent")?;
        model_helpers::ModelSource::dir(parent)
    } else {
        model_helpers::ModelSource::hub(&args.config)
    };
    let checkpoint = source.resolve()?;
    let cfg = checkpoint.config;
    // `--weights` names a local file, not one inside the source: this example is usually
    // pointed at a checkpoint whose weights are already unpacked somewhere else.
    let model_path = match args.weights.as_ref() {
        None => checkpoint.weights,
        Some(p) => std::path::PathBuf::from(p),
    };
    let model_ext = cfg.model_ext();
    tracing::info!(?model_ext, "model extension");

    tracing::info!(?model_path, "loading model");
    let vb = model_helpers::load_weights::<xn::Unquantized<f32, xn::CpuDevice>>(&model_path, &dev)?;

    let emb = if args.input.ends_with(".safetensors") {
        // Stored `speaker_wavs` latents or an already computed `emb`: the loader tells them
        // apart and puts the former through the checkpoint's speaker projection. Only that
        // projection is read, so a GGUF written with `quantize --no-mimi-encoder` works here.
        tracing::info!("loading voice from safetensors file {}", args.input);
        let speaker_proj = ptts::loader::load_speaker_proj(&vb, &cfg)?;
        let path = std::path::Path::new(&args.input);
        model_helpers::load_voice_emb(path, None, speaker_proj.as_ref(), &dev)?
    } else {
        // Audio needs the speaker encoder, which only the full checkpoint carries.
        let mimi_enc: MimiEnc<xn::Unquantized<f32, xn::CpuDevice>> = MimiEnc::load(&vb, &cfg)?;
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
    let (mut pcm, sample_rate) = ptts::audio::decode_file(std::path::Path::new(path))?;
    ptts::utils::normalize_loudness(&mut pcm, sample_rate)?;
    // Unconditional: `resample` hands the buffer back untouched when the rates
    // already match.
    let pcm = ptts::audio::resample(pcm, sample_rate as usize, speaker_sr)?;
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
