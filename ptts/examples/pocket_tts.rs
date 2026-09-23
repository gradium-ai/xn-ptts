//! Generate speech from text on the command line.
//!
//! ```text
//! cargo run --release --example pocket_tts --features hf,audio -- "hello world" -o out.wav
//! ```
//!
//! Everything between the text and the WAV file is [`ptts::synth::Synth`]; what
//! is left here is argument parsing, locating the published checkpoint (see
//! `model_helpers`), audio file decoding for `--voice <file>`, and the timing
//! report.

#[path = "model_helpers.rs"]
mod model_helpers;

use anyhow::Result;
use clap::Parser;
use ptts::preprocess::Normalize;
use ptts::synth::{DeviceKind, Quant, SpeechOptions};

#[derive(Parser, Debug)]
#[command(name = "pocket-tts", about = "Generate speech from text using Pocket TTS")]
struct Args {
    /// Text to synthesize.
    text: String,

    /// Output WAV file path.
    #[arg(short, long, default_value = "output.wav")]
    output: std::path::PathBuf,

    /// Voice: a bundled voice id, a path to a voice `.safetensors`, or a path to
    /// a ~10s audio file to clone. Defaults to the checkpoint's `default` voice,
    /// or its first bundled voice by name.
    #[arg(short, long)]
    voice: Option<String>,

    /// Sampling temperature.
    #[arg(short, long, default_value_t = 0.3)]
    temperature: f32,

    /// Sampling seed.
    #[arg(short, long, default_value_t = 4242424242424242)]
    seed: u64,

    /// Hugging Face model repo to download the checkpoint from: config.json,
    /// weights, tokenizer and voices.
    #[arg(long, default_value = model_helpers::REPO_ID, conflicts_with = "dir")]
    repo: String,

    /// Load from a local directory holding config.json, weights, tokenizer and
    /// voices/ instead of downloading from the Hugging Face Hub.
    #[arg(long)]
    dir: Option<std::path::PathBuf>,

    /// Tokenizer to load, as a path to a `tokenizer.json`. Defaults to the one
    /// the checkpoint ships. A checkpoint that carries only a SentencePiece
    /// `tokenizer.model` needs converting once with `scripts/convert-tokenizer.py`.
    #[arg(long)]
    tokenizer: Option<std::path::PathBuf>,

    /// Weights file to load from the repo or directory, e.g. `model.q8.gguf` for
    /// a checkpoint that ships both f32 and quantized weights. Defaults to the
    /// first of model.safetensors, model.q8.gguf or tts_b6369a24.safetensors
    /// that exists.
    #[arg(long)]
    weights: Option<String>,

    /// Device to run on: auto, cpu, cuda, vulkan or metal.
    #[arg(long, default_value = "auto")]
    device: String,

    /// Weight format for the flow-LM linears, e.g. q8_0 or q4k. CPU only.
    #[arg(long)]
    quant: Option<String>,

    /// Classifier-free guidance coefficient. 1.0 disables it.
    #[arg(long)]
    cfg_coef: Option<f32>,

    /// Replay noise from a JSON array of floats instead of sampling it, so a
    /// run can be compared against the reference implementation step for step.
    #[arg(long)]
    rng_values: Option<std::path::PathBuf>,

    /// Write a Chrome trace of the run to ./trace-<timestamp>.json.
    #[arg(long)]
    chrome_tracing: bool,

    /// Number of CPU threads for tensor ops. Defaults to one per logical core.
    #[arg(long)]
    threads: Option<usize>,

    /// Language the text is normalized as before tokenizing: `en`, `fr`, `de`, `es` or `pt`.
    /// Required: the spoken forms differ per language, so there is nothing safe to guess.
    /// `none` hands the text to the tokenizer as written, which the model reads less well.
    #[arg(long)]
    lang: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Parsed before anything is downloaded, so a bad --lang fails in milliseconds.
    let normalize = Normalize::parse(&args.lang)?;
    if let Some(threads) = args.threads {
        // Must happen before the first tensor op, since it sets the size of rayon's global pool.
        xn::set_num_threads(threads);
    }
    let _guard = init_tracing(args.chrome_tracing);
    tracing::info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        xn::with_avx(),
        xn::with_neon(),
        xn::with_simd128(),
        xn::with_f16c()
    );

    // Which files the checkpoint ships, and what they are called, is this
    // example's business rather than the library's.
    let source = match args.dir.as_deref() {
        Some(dir) => model_helpers::Source::Dir(dir),
        None => model_helpers::Source::Hub(&args.repo),
    };
    let checkpoint = model_helpers::Checkpoint::locate(source, args.weights.as_deref())?;
    let mut builder = checkpoint
        .builder(normalize)
        .device(DeviceKind::parse(&args.device)?)
        .temperature(args.temperature)
        .seed(args.seed);
    if let Some(tokenizer) = args.tokenizer.as_deref() {
        builder = builder.tokenizer_file(tokenizer);
    }
    if let Some(quant) = args.quant.as_deref() {
        builder = builder.quant(Quant::parse(quant)?);
    }
    if let Some(cfg_coef) = args.cfg_coef {
        builder = builder.cfg_coef(cfg_coef);
    }
    // An embedding file can be registered before the model loads; an audio file
    // has to wait until the speaker codec's sample rate is known.
    let voice = VoiceArg::parse(args.voice.as_deref());
    if let VoiceArg::Embedding(path) = &voice {
        builder = builder.add_voice(VoiceArg::REGISTERED, path.clone());
    }

    tracing::info!("loading model");
    let mut tts = builder.build()?;
    checkpoint.register_voices(&mut tts);
    tracing::info!(device = %tts.device_name(), voices = ?tts.voices(), "model loaded");

    let mut opts = SpeechOptions::default();
    match &voice {
        // The bundled voices are registered after the load, so the builder's
        // own "first voice by name" default never saw them; pick it here. A
        // checkpoint that ships a `default-voice.safetensors` gets that one.
        VoiceArg::Default => {
            let voices = tts.voices();
            let pick = voices.iter().find(|v| v.as_str() == "default").or(voices.first());
            if let Some(name) = pick {
                opts = opts.voice(name.clone());
            }
        }
        VoiceArg::Bundled(name) => opts = opts.voice(name.clone()),
        VoiceArg::Embedding(_) => opts = opts.voice(VoiceArg::REGISTERED),
        VoiceArg::Audio(path) => {
            let pcm = ptts::audio::load_mono_at(path, tts.voice_prompt_sample_rate())?;
            tts.add_voice_from_pcm(VoiceArg::REGISTERED, &pcm)?;
            opts = opts.voice(VoiceArg::REGISTERED);
        }
    }

    // `Synth` normalizes the text itself; log what it will see.
    let text = args.text.as_str();
    if normalize != Normalize::Off {
        tracing::info!(
            lang = normalize.as_str(),
            normalized = %normalize.apply(text),
            "normalizing text"
        );
    }

    tracing::info!("generating");
    let start = std::time::Instant::now();
    let stream = match args.rng_values.as_ref() {
        None => tts.stream_with(text, &opts)?,
        Some(path) => {
            let values: Vec<f32> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            let rng = ptts::flow_lm::ReplayRng::new(values)?;
            tts.stream_with_rng(text, &opts, Box::new(rng))?
        }
    };

    let sample_rate = stream.sample_rate();
    let mut pcm = Vec::new();
    let mut first_chunk_ms = None;
    for chunk in stream {
        pcm.extend_from_slice(&chunk?);
        first_chunk_ms.get_or_insert_with(|| start.elapsed().as_secs_f64() * 1000.0);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let duration = pcm.len() as f64 / sample_rate as f64;
    tracing::info!(
        "generated {duration:.2}s in {elapsed:.2}s (RTF={:.3}, first chunk {:.0}ms)",
        duration / elapsed,
        first_chunk_ms.unwrap_or(0.0),
    );
    // `getrusage` is unix-only, so this is absent on Windows.
    if let Some(rss_mb) = peak_rss_mb() {
        tracing::info!("peak RSS: {rss_mb:.2} MB");
    }

    ptts::wav::write_wav_file(&args.output, &pcm, sample_rate as u32)?;
    tracing::info!("wrote {}", args.output.display());
    Ok(())
}

/// What `--voice` was pointing at.
enum VoiceArg {
    /// Not given: use whichever voice the model defaults to.
    Default,
    /// A voice id shipped with the checkpoint, e.g. `alba`.
    Bundled(String),
    /// A precomputed voice embedding, e.g. from the `create_voice` example.
    Embedding(std::path::PathBuf),
    /// An audio file to clone, ~10s of speech.
    Audio(std::path::PathBuf),
}

impl VoiceArg {
    /// Name the two file forms get registered under.
    const REGISTERED: &'static str = "custom";

    fn parse(arg: Option<&str>) -> Self {
        match arg {
            None => Self::Default,
            Some(arg) if arg.ends_with(".safetensors") => Self::Embedding(arg.into()),
            Some(arg) if std::path::Path::new(arg).is_file() => Self::Audio(arg.into()),
            Some(arg) => Self::Bundled(arg.to_string()),
        }
    }
}

fn init_tracing(chrome_tracing: bool) -> Option<tracing_chrome::FlushGuard> {
    use tracing_subscriber::{EnvFilter, prelude::*};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(model_helpers::LOG_DIRECTIVES));
    let fmt = tracing_subscriber::fmt::Layer::new().with_target(false);
    if chrome_tracing {
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(fmt).with(chrome_layer).with(filter).init();
        Some(guard)
    } else {
        tracing_subscriber::registry().with(fmt).with(filter).init();
        None
    }
}

/// Peak resident set size, or `None` on platforms with no `getrusage`.
#[cfg(unix)]
fn peak_rss_mb() -> Option<f64> {
    let mut usage = std::mem::MaybeUninit::uninit();
    let maxrss = unsafe {
        libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr());
        usage.assume_init().ru_maxrss as f64
    };
    // ru_maxrss is in bytes on macOS but kilobytes on Linux.
    Some(if cfg!(target_os = "macos") { maxrss / (1024.0 * 1024.0) } else { maxrss / 1024.0 })
}

#[cfg(not(unix))]
fn peak_rss_mb() -> Option<f64> {
    None
}
