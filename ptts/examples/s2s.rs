//! Run an audio file through a Phonon speech-to-speech checkpoint, streaming it in as a live
//! input would arrive.
//!
//! ```text
//! cargo run --release -p ptts --example s2s --features audio -- \
//!     --model /path/to/model --voice voice.safetensors input.wav -o out.wav
//! ```
//!
//! The input goes in `--chunk-ms` at a time (one frame by default), as fast as it is processed
//! or, with `--realtime`, at the pace a live input would deliver it; `--chunk-ms 0` passes it
//! whole. Prints how long each stage took per frame and how long pushes took, which a live
//! stream must turn around within the chunk's own duration.

#[path = "../src/bin/ptts/model_helpers.rs"]
mod model_helpers;

use anyhow::Result;
use clap::Parser;
use std::time::{Duration, Instant};
use xn::BackendQ;

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

    /// Weight quantization, e.g. `q8`. Required for GGUF weights; safetensors load as f32.
    #[arg(long)]
    quant: Option<String>,

    /// Run on the CPU even when the build has a GPU backend.
    #[arg(long)]
    cpu: bool,

    /// Worker threads; xn's default (one per logical core) when unset.
    #[arg(long)]
    threads: Option<usize>,

    /// Sampling temperature.
    #[arg(short, long, default_value_t = 0.8)]
    temperature: f32,

    /// Sampling seed.
    #[arg(short, long, default_value_t = 4242424242424242)]
    seed: u64,

    /// Flow steps per frame. Defaults to the checkpoint's `lsd_decode_steps`.
    #[arg(long)]
    lsd_steps: Option<usize>,

    /// Input frames the model keeps in view.
    #[arg(long, default_value_t = ptts::s2s::DEFAULT_CONTEXT)]
    context: usize,

    /// Milliseconds of input per push; 0 passes the whole file at once.
    #[arg(long, default_value_t = 80)]
    chunk_ms: usize,

    /// Push each chunk only once a live input would have recorded it.
    #[arg(long)]
    realtime: bool,
}

struct Run<'a>(&'a Args);

impl xn::WithQ for Run<'_> {
    type Output = ();

    fn run<Q: BackendQ>(self, dev: Q::B) -> xn::Result<()> {
        run::<Q>(self.0, dev).map_err(|e| xn::Error::msg(format!("{e:?}")))
    }
}

fn run<Q: BackendQ>(args: &Args, dev: Q::B) -> Result<()> {
    let options =
        model_helpers::ResolveOptions { weights: args.weights.as_deref(), ..Default::default() };
    let checkpoint = model_helpers::Checkpoint::locate(&args.model, options)?;
    let cfg = checkpoint.config;
    tracing::info!(?checkpoint.weights, "loading model");
    let vb = model_helpers::load_weights::<Q>(&checkpoint.weights, &dev)?;
    let model = ptts::s2s::SpeechToSpeech::<Q>::load(&vb, &cfg)?;
    let voice = model_helpers::load_voice_emb(
        &args.voice,
        cfg.embeddings_ext.as_deref(),
        model.model.speaker_proj(),
        &dev,
    )?;
    let sample_rate = model.sample_rate();
    let pcm = ptts::audio::load_mono_at(&args.input, sample_rate as u32)?;

    let start = Instant::now();
    let voice = model.voice(&voice)?;
    let voice_time = start.elapsed();
    let opts = ptts::s2s::SessionOptions {
        temperature: args.temperature,
        seed: args.seed,
        lsd_decode_steps: args.lsd_steps,
        context: args.context,
    };
    let mut session = model.session(&voice, &opts)?;
    let chunk = match args.chunk_ms {
        0 => pcm.len().max(1),
        ms => sample_rate * ms / 1000,
    };
    let mut out = Vec::with_capacity(pcm.len());
    let mut pushes = Vec::new();
    let start = Instant::now();
    for (i, piece) in pcm.chunks(chunk).enumerate() {
        if args.realtime {
            // A live input hands over each chunk once all of it has been recorded.
            let due = Duration::from_secs_f64(((i + 1) * chunk) as f64 / sample_rate as f64);
            std::thread::sleep(due.saturating_sub(start.elapsed()));
        }
        let push = Instant::now();
        out.extend(session.push(piece)?);
        pushes.push(push.elapsed());
    }
    out.extend(session.flush()?);
    let elapsed = start.elapsed();
    ptts::wav::write_wav_file(&args.output, &out, sample_rate as u32)?;

    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let seconds = out.len() as f64 / sample_rate as f64;
    let frames = out.len().div_ceil(model.frame_size()) as f64;
    let times = session.times();
    let frame_ms = model.frame_size() as f64 * 1000.0 / sample_rate as f64;
    if !args.realtime {
        println!(
            "wrote {} ({seconds:.2}s of audio in {:.2}s, RTF {:.3})",
            args.output.display(),
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / seconds
        );
    }
    println!(
        "per frame: encode {:.2} ms, flow {:.2} ms, decode {:.2} ms; voice prompt {:.1} ms",
        ms(times.encode) / frames,
        ms(times.flow) / frames,
        ms(times.decode) / frames,
        ms(voice_time)
    );
    pushes.sort();
    let at = |q: f64| ms(pushes[((pushes.len() - 1) as f64 * q) as usize]);
    println!(
        "push of {:.0} ms of input: p50 {:.1} ms, p95 {:.1} ms, max {:.1} ms; \
         the output lags the input by {:.0} ms plus that",
        chunk as f64 * 1000.0 / sample_rate as f64,
        at(0.5),
        at(0.95),
        at(1.0),
        (model.delay() + 1) as f64 * frame_ms
    );
    Ok(())
}

fn main() -> Result<()> {
    use std::str::FromStr;

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(model_helpers::LOG_DIRECTIVES));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    let args = Args::parse();
    if let Some(threads) = args.threads {
        // Must happen before the first tensor op, since it sets the size of rayon's global pool.
        xn::set_num_threads(threads);
    }
    let dtype = match args.quant.as_deref() {
        Some(quant) => xn::DTypeQ::from_str(quant)?,
        None => xn::DTypeQ::F32,
    };
    let cpu = args.cpu || dtype != xn::DTypeQ::F32;
    xn::Runner::new().cpu_only(cpu).dtype(dtype).run(Run(&args), 0)?;
    Ok(())
}
