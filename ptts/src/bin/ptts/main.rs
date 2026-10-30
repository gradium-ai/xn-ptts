//! Generate speech from text on the command line.
//!
//! ```text
//! ptts --model /path/to/model --lang en "hello world" -o out.wav
//! ```
//!
//! Everything between the text and the WAV file is [`ptts::synth::Synth`]; what
//! is left here is argument parsing, locating the published checkpoint (see
//! `ptts::checkpoint`), audio file decoding for `--voice <file>`, and the timing
//! report.

mod model_helpers;

use anyhow::{Context, Result};
use clap::Parser;
use ptts::preprocess::{Normalize, Rules};
use ptts::synth::{DeviceKind, Quant, SpeechOptions};
use ptts::voice_catalog::VoiceCatalog;

#[derive(Parser, Debug)]
#[command(name = "ptts", version, about = "Generate speech with Phonon")]
struct Args {
    /// Text to synthesize.
    #[arg(required_unless_present = "list_voices")]
    text: Option<String>,

    /// Output WAV file path.
    #[arg(short, long, default_value = "out.wav")]
    output: std::path::PathBuf,

    /// Voice: a voice of the checkpoint or of the voice catalog, by name or catalog voice ID, a
    /// path to a voice `.safetensors`, or a path to a ~10s audio file to clone. Defaults to the
    /// checkpoint's `default` voice, or its first bundled voice by name.
    #[arg(short, long)]
    voice: Option<String>,

    /// Sampling temperature.
    #[arg(short, long, default_value_t = 0.3)]
    temperature: f32,

    /// Sampling seed.
    #[arg(short, long, default_value_t = 4242424242424242)]
    seed: u64,

    #[arg(
        short,
        long,
        alias = "dir",
        default_value = model_helpers::DEFAULT_MODEL,
        help = model_helpers::MODEL_HELP
    )]
    model: String,

    /// Voice catalog to look --voice up in when the checkpoint has no voice of that name: a
    /// Hugging Face repo as hf://owner/name[@revision], or a local JSONL file or folder holding
    /// voices.jsonl. Only the voice that is used is downloaded.
    #[arg(long, default_value = ptts::voice_catalog::DEFAULT_VOICE_CATALOG)]
    voice_catalog: String,

    /// List the voices of --voice-catalog, then exit.
    #[arg(long)]
    list_voices: bool,

    /// Tokenizer to load, as a path to a `tokenizer.json`. Defaults to the one
    /// the checkpoint ships.
    #[arg(long)]
    tokenizer: Option<std::path::PathBuf>,

    /// Weights file to load from the repo or directory, e.g. `model.q8.gguf` for
    /// a checkpoint that ships both f32 and quantized weights. Defaults to the
    /// standard weights matching --quant, then f32 weights.
    #[arg(long)]
    weights: Option<String>,

    /// Device to run on: auto, cpu, cuda, vulkan or metal. `auto` picks the GPU backend this
    /// build was compiled with, if any, and the CPU otherwise or when that GPU fails to start.
    #[arg(long, default_value = "auto")]
    device: String,

    /// Weight format for the flow-LM linears, e.g. q8_0 or q4k. CPU only.
    #[arg(long)]
    quant: Option<String>,

    /// Classifier-free guidance coefficient. 1.0 disables it.
    #[arg(long)]
    cfg_coef: Option<f32>,

    #[arg(
        long = "condition",
        value_name = "NAME=VALUE",
        help = "Set a conditioner, e.g. duration_delta=0.5; repeatable"
    )]
    conditions: Vec<String>,

    /// Replay noise from a JSON array of floats instead of sampling it, so a
    /// run can be compared against the reference implementation step for step.
    #[arg(long, help_heading = "Developer")]
    rng_values: Option<std::path::PathBuf>,

    /// Write a Chrome trace of the run to ./trace-<timestamp>.json.
    #[arg(long, help_heading = "Developer")]
    chrome_tracing: bool,

    /// Number of CPU threads for tensor ops. Defaults to one per logical core.
    #[arg(long)]
    threads: Option<usize>,

    /// Language the text is normalized as before tokenizing: `en`, `fr`, `de`, `es`, `pt`,
    /// `ar`, `hi`, `ko`, `pl`, `sv`, `fi`, `no`, `it`, `nl`, `ja`, `zh`, `da` or `tr`. Required:
    /// the spoken forms differ per language, so there is nothing safe to guess. `none` hands the
    /// text to the tokenizer as written, which the model reads less well.
    #[arg(long, required_unless_present = "list_voices")]
    lang: Option<String>,

    /// Which word rewrites run on the normalized text: `default` (numbers, currency,
    /// dashed-digits, emails, urls, abbreviations, elongations), `all` (those and phones, times,
    /// dates), `none`, or a comma-separated list of rule names. Has no effect with `--lang none`,
    /// and the rules only read `en`, `fr`, `de`, `es` and `pt`.
    #[arg(long, default_value = "default")]
    rewrites: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let catalog = VoiceCatalog::new(&args.voice_catalog)?;
    if args.list_voices {
        for voice in catalog.voices()? {
            let field = |f: &Option<String>| f.clone().unwrap_or_else(|| "-".to_string());
            println!(
                "{}\t{}\t{}\t{}",
                voice.name,
                field(&voice.language),
                field(&voice.voice_id),
                field(&voice.description)
            );
        }
        return Ok(());
    }
    // clap requires both unless --list-voices is given.
    let text = args.text.as_deref().context("nothing to say")?;
    let lang = args.lang.as_deref().context("--lang is required")?;
    // Parsed before anything is downloaded, so a bad --lang fails in milliseconds.
    let normalize = lang.parse::<Normalize>()?.with_rules(args.rewrites.parse::<Rules>()?);
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

    let quant = args.quant.as_deref().unwrap_or("f32").parse::<Quant>()?;
    let device = args.device.parse::<DeviceKind>()?;
    quant.check_device(device)?;
    let options = model_helpers::ResolveOptions { quant, weights: args.weights.as_deref() };
    let checkpoint = model_helpers::Checkpoint::locate(&args.model, options)?;
    let mut builder =
        checkpoint.builder(normalize).device(device).temperature(args.temperature).seed(args.seed);
    if let Some(tokenizer) = args.tokenizer.as_deref() {
        builder = builder.tokenizer_file(tokenizer);
    }
    if let Some(cfg_coef) = args.cfg_coef {
        builder = builder.cfg_coef(cfg_coef);
    }
    for condition in &args.conditions {
        let (name, value) = condition.split_once('=').context("--condition takes NAME=VALUE")?;
        builder = builder.condition(name, value);
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
    tracing::info!(device = %tts.device_name(), voices = ?tts.voices(), languages = ?tts.languages(), "model loaded");

    let mut opts = SpeechOptions::default();
    match &voice {
        VoiceArg::Default => {}
        VoiceArg::Bundled(name) => {
            let name = catalog.register(&mut tts, name)?;
            anyhow::ensure!(
                tts.voices().contains(&name),
                "no voice '{name}': the model has {}, and --list-voices lists the catalog's",
                tts.voices().join(", ")
            );
            opts = opts.voice(name);
        }
        VoiceArg::Embedding(_) => opts = opts.voice(VoiceArg::REGISTERED),
        VoiceArg::Audio(path) => {
            let pcm = ptts::audio::load_mono_at(path, tts.voice_prompt_sample_rate())?;
            tts.add_voice_from_pcm(VoiceArg::REGISTERED, &pcm)?;
            opts = opts.voice(VoiceArg::REGISTERED);
        }
    }

    // `Synth` normalizes the text itself; log what it will see.
    if normalize != Normalize::OFF {
        tracing::info!(
            lang = normalize.as_str(),
            rewrites = args.rewrites.as_str(),
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
    let rtf = if duration > 0.0 { format!("{:.4}", elapsed / duration) } else { "?".to_string() };
    tracing::info!(
        "generated {duration:.2}s in {elapsed:.2}s (RTF={rtf}, first chunk {:.0}ms)",
        first_chunk_ms.unwrap_or(0.0),
    );
    // `getrusage` is unix-only, so this is absent on Windows.
    if let Some(rss_mb) = peak_rss_mb() {
        tracing::info!("peak RSS: {rss_mb:.2} MB");
    }

    ptts::wav::write_wav_file(&args.output, &pcm, sample_rate)?;
    tracing::info!("wrote {}", args.output.display());
    Ok(())
}

/// What `--voice` was pointing at.
enum VoiceArg {
    /// Not given: use whichever voice the model defaults to.
    Default,
    /// A voice of the checkpoint or the catalog, e.g. `Zoey`.
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
    // Logs go to stderr, so `-o /dev/stdout` writes nothing but the WAV.
    let fmt = tracing_subscriber::fmt::Layer::new().with_target(false).with_writer(std::io::stderr);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_model_the_default_loads() {
        let args = Args::try_parse_from(["ptts", "--lang", "en", "hello"]).unwrap();
        assert_eq!(args.model, model_helpers::DEFAULT_MODEL);
    }

    #[test]
    fn list_voices_needs_no_text_or_lang() {
        let args = Args::try_parse_from(["ptts", "--list-voices", "--voice-catalog", "c"]).unwrap();
        assert!(args.list_voices);
        assert_eq!(args.voice_catalog, "c");
        let args = Args::try_parse_from(["ptts", "--list-voices"]).unwrap();
        assert_eq!(args.voice_catalog, ptts::voice_catalog::DEFAULT_VOICE_CATALOG);
        assert!(Args::try_parse_from(["ptts", "hello"]).is_err());
        assert!(Args::try_parse_from(["ptts", "--lang", "en"]).is_err());
    }

    #[test]
    fn dir_still_names_a_local_model() {
        let args = Args::try_parse_from(["ptts", "--dir", "m", "--lang", "en", "hi"]).unwrap();
        assert_eq!(args.model, "m");
    }
}
