//! Transcribe audio files with a Graphon export.
//!
//! ```bash
//! cargo run -r -p graphon --example transcribe -- --model graphon-export a.wav b.flac
//! ```
//!
//! Prints one line per file: its name and the transcript.

use anyhow::Result;
use clap::Parser;
use graphon::Graphon;
use graphon::tokenizer::Pieces;
use ptts::Tokenizer;

#[derive(Parser, Debug)]
struct Args {
    /// Export directory: config.json, model.safetensors, gemimi.safetensors, tokenizer.json.
    #[arg(long)]
    model: std::path::PathBuf,

    /// Weights of both the trunk and the encoder transformer: f32 or q8_0, quantized at load.
    /// The GPU takes f32 only.
    #[arg(long, default_value = "f32")]
    quant: String,

    /// cpu, or cuda (with the `cuda` feature).
    #[arg(long, default_value = "cpu")]
    device: String,

    #[arg(long)]
    threads: Option<usize>,

    /// Print every step's token id.
    #[arg(long)]
    tokens: bool,

    /// Steps of silence fed after the audio for the last words, instead of the stream's default.
    #[arg(long)]
    flush_steps: Option<usize>,

    /// Also write each file's audio, as the model reads it, to `<dir>/<stem>.f32`: raw
    /// little-endian f32 at the model's sample rate.
    #[arg(long)]
    save_pcm: Option<std::path::PathBuf>,

    /// Write each file's voice activity to `<dir>/<stem>.tsv`, one row per step: the step, the
    /// seconds of audio read so far, the token, then for each of the model's windows the
    /// probability that no speech follows within it. Needs a model with voice activity heads.
    #[arg(long)]
    vad: Option<std::path::PathBuf>,

    files: Vec<std::path::PathBuf>,
}

fn transcribe<Q: xn::BackendQ>(args: &Args, device: Q::B) -> Result<()> {
    let model = Graphon::<Q>::load(&args.model, device)?;
    let pieces = Pieces::load(&args.model.join("tokenizer.json"))?;
    let sample_rate = model.config.sample_rate as u32;
    let frame = model.config.frame_size();
    let flush = args.flush_steps.unwrap_or(model.config.flush_steps());
    let windows = match (&args.vad, &model.vad) {
        (Some(_), Some(vad)) => vad.windows().iter().map(|w| format!("\tsilent_{w}s")).collect(),
        (Some(_), None) => {
            anyhow::bail!("--vad: {} has no voice activity heads", args.model.display())
        }
        (None, _) => String::new(),
    };
    for file in args.files.iter() {
        let pcm = ptts::audio::load_mono_at(file, sample_rate)?;
        if let Some(dir) = &args.save_pcm {
            let stem = file.file_stem().unwrap_or_default().to_string_lossy();
            let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("{stem}.f32")), bytes)?;
        }
        let mut stream = model.stream()?;
        stream.set_vad(args.vad.is_some());
        // The audio, its last frame padded, then the flush's silence, one step at a time.
        let mut padded = pcm;
        padded.resize((padded.len().div_ceil(frame) + flush) * frame, 0.0);
        let mut tokens = Vec::with_capacity(padded.len() / frame);
        let mut rows = format!("step\tseconds\ttoken{windows}\n");
        for (step, samples) in padded.chunks(frame).enumerate() {
            let token = stream.step(samples)?;
            tokens.push(token);
            if args.vad.is_some() {
                let seconds = ((step + 1) * frame) as f64 / sample_rate as f64;
                rows += &format!("{step}\t{seconds:.2}\t{token}");
                for p in stream.inactivity() {
                    rows += &format!("\t{p:.4}");
                }
                rows += "\n";
            }
        }
        if let Some(dir) = &args.vad {
            let stem = file.file_stem().unwrap_or_default().to_string_lossy();
            std::fs::write(dir.join(format!("{stem}.tsv")), rows)?;
        }
        if args.tokens {
            println!("{tokens:?}");
        }
        let text: Vec<u32> = tokens.into_iter().filter(|&t| stream.is_text(t)).collect();
        println!("{}\t{}", file.display(), pieces.decode(&text)?.trim());
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(threads) = args.threads {
        xn::set_num_threads(threads);
    }
    match (args.device.as_str(), args.quant.as_str()) {
        ("cpu", "f32") => transcribe::<xn::Unquantized<f32, xn::CpuDevice>>(&args, xn::CPU),
        ("cpu", "q8_0" | "q8") => transcribe::<xn::quantized::Q80F32>(&args, xn::CPU),
        #[cfg(feature = "cuda")]
        ("cuda", "f32") => {
            let device = xn::cuda_backend::Device::new(0)?;
            transcribe::<xn::Unquantized<f32, xn::cuda_backend::Device>>(&args, device)
        }
        (device, quant) => anyhow::bail!("unsupported --device {device} with --quant {quant}"),
    }
}
