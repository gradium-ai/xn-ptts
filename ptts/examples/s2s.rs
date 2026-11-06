//! Run an audio file through a Phonon speech-to-speech checkpoint.
//!
//! ```text
//! cargo run --release -p ptts --example s2s --features audio -- \
//!     --model /path/to/model --voice voice.safetensors input.wav -o out.wav
//! ```

#[path = "../src/bin/ptts/model_helpers.rs"]
mod model_helpers;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "s2s")]
#[command(about = "Run an audio file through a Phonon speech-to-speech checkpoint")]
struct Args {
    /// Input audio file.
    input: std::path::PathBuf,

    /// Output WAV file path.
    #[arg(short, long, default_value = "out.wav")]
    output: std::path::PathBuf,

    /// Voice: a `.safetensors` file holding `speaker_wavs` latents or an `emb` embedding.
    #[arg(short, long)]
    voice: std::path::PathBuf,

    #[arg(short, long, help = model_helpers::MODEL_HELP)]
    model: String,

    /// Weights file inside the model, when it has several.
    #[arg(long)]
    weights: Option<String>,

    /// Sampling temperature.
    #[arg(short, long, default_value_t = 0.8)]
    temperature: f32,

    /// Sampling seed.
    #[arg(short, long, default_value_t = 4242424242424242)]
    seed: u64,

    /// Flow steps per frame. Defaults to the checkpoint's `lsd_decode_steps`.
    #[arg(long)]
    lsd_steps: Option<usize>,
}

fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(model_helpers::LOG_DIRECTIVES));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    let args = Args::parse();
    let dev = xn::CpuDevice;
    let options =
        model_helpers::ResolveOptions { weights: args.weights.as_deref(), ..Default::default() };
    let checkpoint = model_helpers::Checkpoint::locate(&args.model, options)?;
    let cfg = checkpoint.config;
    tracing::info!(?checkpoint.weights, "loading model");
    let vb = model_helpers::load_weights::<xn::Unquantized<f32, xn::CpuDevice>>(
        &checkpoint.weights,
        &dev,
    )?;
    let model = ptts::s2s::SpeechToSpeech::<xn::Unquantized<f32, xn::CpuDevice>>::load(&vb, &cfg)?;
    let voice = model_helpers::load_voice_emb(
        &args.voice,
        cfg.embeddings_ext.as_deref(),
        model.model.speaker_proj(),
        &dev,
    )?;
    let sample_rate = model.sample_rate();
    let pcm = ptts::audio::load_mono_at(&args.input, sample_rate as u32)?;
    let start = std::time::Instant::now();
    let steps = args.lsd_steps.unwrap_or(cfg.lsd_decode_steps);
    let out = model.run(&pcm, &voice, args.temperature, args.seed, steps)?;
    ptts::wav::write_wav_file(&args.output, &out, sample_rate as u32)?;
    println!(
        "wrote {} ({:.2}s of audio in {:.1}s)",
        args.output.display(),
        out.len() as f32 / sample_rate as f32,
        start.elapsed().as_secs_f32()
    );
    Ok(())
}
