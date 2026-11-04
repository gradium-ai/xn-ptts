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
    #[arg(long, default_value = "f32")]
    quant: String,

    #[arg(long)]
    threads: Option<usize>,

    /// Print every step's token id.
    #[arg(long)]
    tokens: bool,

    /// Also write each file's audio, as the model reads it, to `<dir>/<stem>.f32`: raw
    /// little-endian f32 at the model's sample rate.
    #[arg(long)]
    save_pcm: Option<std::path::PathBuf>,

    files: Vec<std::path::PathBuf>,
}

fn transcribe<Q: xn::BackendQ<B = xn::CpuDevice>>(args: &Args) -> Result<()> {
    let model = Graphon::<Q>::load(&args.model, xn::CPU)?;
    let pieces = Pieces::load(&args.model.join("tokenizer.json"))?;
    let sample_rate = model.config.sample_rate as u32;
    for file in args.files.iter() {
        let pcm = ptts::audio::load_mono_at(file, sample_rate)?;
        if let Some(dir) = &args.save_pcm {
            let stem = file.file_stem().unwrap_or_default().to_string_lossy();
            let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("{stem}.f32")), bytes)?;
        }
        let mut stream = model.stream()?;
        let mut tokens = stream.push(&pcm)?;
        tokens.extend(stream.flush()?);
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
    match args.quant.as_str() {
        "f32" => transcribe::<xn::Unquantized<f32, xn::CpuDevice>>(&args),
        "q8_0" | "q8" => transcribe::<xn::quantized::Q80F32>(&args),
        other => anyhow::bail!("unsupported --quant {other}"),
    }
}
