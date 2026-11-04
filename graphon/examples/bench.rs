//! Benchmark Graphon on the CPU: time per 80 ms step, split into the Gemimi encoder and the
//! trunk, against the step's real-time budget.
//!
//! ```bash
//! cargo run -r -p graphon --features accelerate --example bench -- \
//!     --model graphon-export --audio speech.wav --threads 4
//! ```
//!
//! Each iteration runs a fresh stream over the whole file on this thread, one step at a time:
//! encoder, then trunk. `--encoder-chunk N` encodes N steps of audio per encoder call instead,
//! as a caller with buffered audio could; the trunk still runs one step at a time. Weights load
//! as f32 safetensors and `--quant` / `--encoder-quant` quantize the linears at load: all of the
//! trunk's and the encoder transformer's.
//!
//! Model load and the first `--warmup` iterations are left out of the statistics.

use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser;
use graphon::Graphon;
use graphon::config::Config;
use xn::{BackendQ, Tensor, WithDTypeF};

#[derive(Parser, Debug, Clone)]
struct Args {
    /// Export directory: config.json, model.safetensors, gemimi.safetensors.
    #[arg(long)]
    model: std::path::PathBuf,

    /// Audio to run on. Without it, `--seconds` of synthetic audio: the timing does not depend
    /// on what is said.
    #[arg(long)]
    audio: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = 20.0)]
    seconds: f64,

    /// Trunk weights: f32, q8_0, q4_0. The k-quants need widths that are multiples of 256.
    #[arg(long, default_value = "f32")]
    quant: String,

    /// Encoder transformer weights: f32, q8_0, q4_0.
    #[arg(long, default_value = "f32")]
    encoder_quant: String,

    /// Steps of audio per encoder call.
    #[arg(long, default_value_t = 1)]
    encoder_chunk: usize,

    #[arg(long)]
    threads: Option<usize>,

    #[arg(long, default_value_t = 3)]
    iters: usize,

    #[arg(long, default_value_t = 1)]
    warmup: usize,
}

/// What a step's time is split into, in order.
const STAGES: [&str; 4] = ["seanet", "enc-xfmr", "latent", "trunk"];

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn run<Q, C>(args: &Args, config: Config, pcm: &[f32]) -> Result<()>
where
    Q: BackendQ<B = xn::CpuDevice>,
    C: BackendQ<T = Q::T, B = Q::B>,
{
    let start = Instant::now();
    let model = Graphon::<Q, C>::load_files(
        config,
        &args.model.join("model.safetensors"),
        &args.model.join("gemimi.safetensors"),
        xn::CPU,
    )?;
    let load = start.elapsed();
    let frame = model.config.frame_size();
    let chunk = frame * args.encoder_chunk;
    let steps = pcm.len() / chunk * args.encoder_chunk;
    let pcm = &pcm[..steps * frame];
    let budget = Duration::from_secs_f64(1.0 / model.config.frame_rate);

    // Per stage, every measured step's time.
    let mut times: Vec<Vec<Duration>> = vec![Vec::new(); STAGES.len()];
    // Per measured iteration: the stream's first step, and its flush at the end.
    let (mut first, mut flush) = (Vec::new(), Vec::new());
    for iter in 0..args.warmup + args.iters {
        let mut stream = model.stream()?;
        let measured = iter >= args.warmup;
        for (c, samples) in pcm.chunks_exact(chunk).enumerate() {
            let first_step = Instant::now();
            let data = samples.iter().map(|&s| Q::T::from_f32(s)).collect();
            let t0 = Instant::now();
            let x = Tensor::from_vec(data, (1, 1, chunk), &xn::CPU)?;
            let emb = model.gemimi.seanet(&x, &mut stream.gemimi)?;
            let t1 = Instant::now();
            let x = model.gemimi.transformer(&emb, &mut stream.gemimi)?;
            let t2 = Instant::now();
            let latent = model.gemimi.latent_head(&x)?;
            let t3 = Instant::now();
            // An encoder call covers `encoder_chunk` steps: each gets an equal share.
            let n = args.encoder_chunk as u32;
            let shares = [(t1 - t0) / n, (t2 - t1) / n, (t3 - t2) / n];
            for i in 0..args.encoder_chunk {
                let row = latent.narrow(1, i..i + 1)?.contiguous()?;
                let t4 = Instant::now();
                model.trunk.step(&row, &mut stream.trunk)?;
                let trunk = t4.elapsed();
                if measured {
                    for (stage, d) in shares.iter().chain([&trunk]).enumerate() {
                        times[stage].push(*d);
                    }
                }
            }
            if measured && c == 0 {
                first.push(first_step.elapsed());
            }
        }
        // What an app does when the speech ends: silence until the text has caught up, as
        // fast as the device runs it.
        let t = Instant::now();
        stream.flush()?;
        if measured {
            flush.push(t.elapsed());
        }
    }

    let audio = steps as f64 * budget.as_secs_f64();
    let total: Vec<Duration> =
        (0..times[0].len()).map(|i| times.iter().map(|t| t[i]).sum()).collect();
    let encoder: Vec<Duration> =
        (0..times[0].len()).map(|i| times[..3].iter().map(|t| t[i]).sum()).collect();
    println!(
        "graphon bench: {} steps ({audio:.1} s of audio) x {} iters, {} threads, trunk {}, encoder {}, encoder chunk {}",
        steps,
        args.iters,
        xn::get_num_threads(),
        args.quant,
        args.encoder_quant,
        args.encoder_chunk
    );
    println!("  load        {:8.1} ms", ms(load));
    let mean = |xs: &[Duration]| ms(xs.iter().sum::<Duration>() / xs.len() as u32);
    println!(
        "  first step  {:8.1} ms   compute on top of the model's fixed text delay",
        mean(&first)
    );
    println!(
        "  flush       {:8.1} ms   end of speech to the end of the flush: {} silent steps back to back",
        mean(&flush),
        model.config.flush_steps()
    );
    println!("  stage          mean      p50      p95      p99      max   (ms per step)");
    let rows = STAGES
        .iter()
        .map(|s| s.to_string())
        .zip(times)
        .chain([("encoder".to_string(), encoder), ("total".to_string(), total.clone())]);
    for (name, mut xs) in rows {
        let mean = xs.iter().sum::<Duration>() / xs.len() as u32;
        xs.sort();
        println!(
            "  {name:<9} {:8.2} {:8.2} {:8.2} {:8.2} {:8.2}",
            ms(mean),
            ms(percentile(&xs, 0.5)),
            ms(percentile(&xs, 0.95)),
            ms(percentile(&xs, 0.99)),
            ms(*xs.last().unwrap())
        );
    }
    let total_mean = total.iter().sum::<Duration>() / total.len() as u32;
    let over = total.iter().filter(|d| **d > budget).count();
    println!(
        "  rtf {:.4}  ({:.1}x real time), steps over the {:.0} ms budget: {over}/{}",
        total_mean.as_secs_f64() / budget.as_secs_f64(),
        budget.as_secs_f64() / total_mean.as_secs_f64(),
        ms(budget),
        total.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(threads) = args.threads {
        xn::set_num_threads(threads);
    }
    let config = Config::load(&args.model.join("config.json"))?;
    let sr = config.sample_rate;
    let pcm = match &args.audio {
        Some(path) => ptts::audio::load_mono_at(path, sr as u32)?,
        None => (0..(args.seconds * sr as f64) as usize)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.1 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.01 * (((i as u64).wrapping_mul(2654435761) % 1000) as f32 / 500.0 - 1.0)
            })
            .collect(),
    };

    type F32 = xn::Unquantized<f32, xn::CpuDevice>;
    use xn::quantized::{Q40F32, Q80F32};
    macro_rules! encoder {
        ($q:ty) => {
            match args.encoder_quant.as_str() {
                "f32" => run::<$q, F32>(&args, config, &pcm),
                "q8_0" | "q8" => run::<$q, Q80F32>(&args, config, &pcm),
                "q4_0" | "q4" => run::<$q, Q40F32>(&args, config, &pcm),
                other => bail!("unsupported --encoder-quant {other}"),
            }
        };
    }
    match args.quant.as_str() {
        "f32" => encoder!(F32),
        "q8_0" | "q8" => encoder!(Q80F32),
        "q4_0" | "q4" => encoder!(Q40F32),
        other => bail!("unsupported --quant {other}"),
    }
}
