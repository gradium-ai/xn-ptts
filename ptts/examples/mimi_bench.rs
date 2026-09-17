//! Mimi decoder microbenchmark: loads a local model once, then decodes a
//! stream of latents frame by frame and reports the time per frame, in
//! isolation from the flow LM. Run under `XN_VULKAN_PROFILE=1` (or `nsys`)
//! for a per-kernel breakdown of the decoder alone.

#[path = "model_helpers.rs"]
mod model_helpers;

use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ptts::tts_model::{TTSConfig, TTSModel};
use xn::{BackendQ, Tensor};

#[derive(Parser, Debug)]
#[command(name = "mimi_bench")]
#[command(about = "Benchmark the Mimi decoder: time per decoded frame")]
struct Args {
    /// Model weights, safetensors or GGUF.
    #[arg(long)]
    model: std::path::PathBuf,

    /// Model config JSON.
    #[arg(long)]
    config: std::path::PathBuf,

    /// Weight quantization for GGUF weights, e.g. `q8`.
    #[arg(long)]
    quant: Option<String>,

    /// Use the cpu device even if a gpu backend is available.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Frames decoded per measured iteration.
    #[arg(long, default_value_t = 38)]
    frames: usize,

    /// Measured iterations.
    #[arg(long, default_value_t = 10)]
    iters: usize,

    /// Unmeasured iterations run first.
    #[arg(long, default_value_t = 2)]
    warmup: usize,
}

struct NoTokenizer;

impl ptts::Tokenizer for NoTokenizer {
    fn encode(&self, _text: &str) -> xn::Result<Vec<u32>> {
        xn::bail!("mimi_bench has no tokenizer")
    }
    fn decode(&self, _tokens: &[u32]) -> xn::Result<String> {
        xn::bail!("mimi_bench has no tokenizer")
    }
}

struct Bench<'a>(&'a Args);

impl xn::WithQ for Bench<'_> {
    type Output = ();

    fn run<Q: BackendQ>(self, dev: Q::B) -> xn::Result<()> {
        self.bench::<Q>(dev).map_err(|e| xn::Error::msg(format!("{e:?}")))
    }
}

impl Bench<'_> {
    fn bench<Q: BackendQ>(&self, dev: Q::B) -> Result<()> {
        let args = self.0;
        let cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(&args.config)?)
            .with_context(|| format!("failed to read config {}", args.config.display()))?;
        let vb = model_helpers::load_weights::<Q>(&args.model, &dev)?;
        // The decoder never tokenizes; the model still wants one at load.
        let model: TTSModel<Q> = TTSModel::load(&vb, Box::new(NoTokenizer), &cfg)?;
        let ldim = model.flow_lm.ldim;
        // A fixed pseudo-random latent stream; the decoder's cost does not
        // depend on the values.
        let latents: Vec<Tensor<Q::T, Q::B>> = (0..args.frames)
            .map(|f| {
                let v: Vec<f32> = (0..ldim)
                    .map(|i| (((i * 31 + f * 17) % 97) as f32 / 48.5 - 1.0) * 0.5)
                    .collect();
                Tensor::<f32, Q::B>::from_vec(v, (1, 1, ldim), &dev)?.to::<Q::T>()
            })
            .collect::<xn::Result<_>>()?;

        let one = || -> Result<(f64, usize)> {
            let mut state = model.init_mimi_state(1)?;
            let start = Instant::now();
            let mut samples = 0;
            for latent in &latents {
                let pcm = model.decode_latent(latent, &mut state)?.to_vec()?;
                samples += pcm.len();
            }
            Ok((start.elapsed().as_secs_f64() * 1e3, samples))
        };
        for _ in 0..args.warmup {
            one()?;
        }
        let mut totals = Vec::with_capacity(args.iters);
        let mut samples = 0;
        for _ in 0..args.iters {
            let (ms, s) = one()?;
            totals.push(ms);
            samples = s;
        }
        totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = totals.iter().sum::<f64>() / totals.len() as f64;
        println!(
            "mimi decode: {} frames -> {} samples; per iteration min {:.2} ms, mean {:.2} ms, max {:.2} ms; per frame {:.3} ms",
            args.frames,
            samples,
            totals[0],
            mean,
            totals[totals.len() - 1],
            mean / args.frames as f64
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    use std::str::FromStr;

    let args = Args::parse();
    let dtype = match args.quant.as_deref() {
        Some(quant) => xn::DTypeQ::from_str(quant)?,
        None if args.model.extension().and_then(|v| v.to_str()) == Some("gguf") => {
            anyhow::bail!("GGUF weights need an explicit --quant, e.g. --quant q8")
        }
        None => xn::DTypeQ::F32,
    };
    xn::Runner::new().cpu_only(args.cpu).dtype(dtype).run(Bench(&args), 0)?;
    Ok(())
}
