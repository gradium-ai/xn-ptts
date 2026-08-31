//! Benchmark harness for TTS generation.
//!
//! Loads a local model once, then generates the same utterance `--iters` times and reports
//! time-to-first-audio, per-frame time, total generate time and RTF. Model load and voice
//! conditioning are timed separately and excluded from the per-iteration statistics, since a
//! server pays them once and then serves many requests.
//!
//! ```bash
//! cargo run --release --features sp,accelerate --example bench -- \
//!   --model model/model.q8.gguf --config model/config.json --quant q8 \
//!   --voice voices/freya.safetensors --threads 8 --iters 20
//! ```
//!
//! Unlike `pocket_tts` this never downloads anything and only accepts precomputed voice
//! embeddings: it measures one specific model. Mimi decoding runs on the generating thread
//! rather than overlapped, so a frame's time is its sampling plus its decoding; `pocket_tts`
//! overlaps the two and will report a better RTF for the same weights.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use ptts::tts_model::{TTSConfig, TTSModel, TTSState};
use xn::nn::VB;
use xn::{BackendQ, Tensor};

/// Frames of Mimi decoder context, matching `pocket_tts`.
const MIMI_CONTEXT_SIZE: usize = 250;
/// Spare KV positions on top of what an utterance is calculated to need.
const SEQ_BUDGET_SLACK: usize = 16;

#[derive(Parser, Debug)]
#[command(name = "bench")]
#[command(about = "Benchmark TTS generation: TTFA, per-frame time, total runtime")]
struct Args {
    /// Model weights, either a safetensors file or a GGUF file (see the `quantize` example).
    #[arg(long)]
    model: std::path::PathBuf,

    /// Model config JSON.
    #[arg(long)]
    config: std::path::PathBuf,

    /// SentencePiece tokenizer. Defaults to `tokenizer.model` next to the config.
    #[arg(long)]
    tokenizer: Option<std::path::PathBuf>,

    /// Precomputed voice embedding safetensors.
    #[arg(long)]
    voice: std::path::PathBuf,

    /// Weight quantization, e.g. `q8`. Required for GGUF weights; safetensors load as f32.
    #[arg(long)]
    quant: Option<String>,

    /// Use the cpu device even if a gpu backend is available.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Number of CPU threads for tensor ops. Defaults to xn's own default of one per logical
    /// core, which is usually too many here: generation is a single autoregressive stream of
    /// small ops, so past a few threads the coordination cost outweighs the parallelism.
    #[arg(long)]
    threads: Option<usize>,

    #[arg(long, short, default_value = "Hello, this is a test of the pocket TTS system.")]
    input: String,

    #[arg(long, default_value_t = 0.4)]
    temperature: f32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Measured iterations.
    #[arg(long, default_value_t = 10)]
    iters: usize,

    /// Unmeasured iterations run first, to warm caches and the thread pool.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Print a line per iteration as well as the summary.
    #[arg(long, default_value_t = false)]
    per_iter: bool,
}

struct SpTokenizer(sentencepiece::SentencePieceProcessor);

impl ptts::Tokenizer for SpTokenizer {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
        Ok(self.0.encode(text).map_err(xn::Error::wrap)?.into_iter().map(|v| v.id).collect())
    }

    fn decode(&self, tokens: &[u32]) -> xn::Result<String> {
        self.0.decode_piece_ids(tokens).map_err(xn::Error::wrap)
    }
}

struct StdRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl StdRng {
    fn new(temperature: f32, seed: u64) -> Result<Self> {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, temperature.sqrt())?;
        Ok(Self { inner: rand::rngs::StdRng::seed_from_u64(seed), distr })
    }
}

impl ptts::flow_lm::Rng for StdRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

fn remap_key(name: &str) -> Option<String> {
    // Skip keys we don't need
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
    {
        return None;
    }

    let mut name = name.to_string();

    // Order matters: more specific replacements first
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");

    Some(name)
}

fn load_voice_emb<Q: BackendQ>(
    path: &std::path::Path,
    cfg: &TTSConfig,
    dev: &Q::B,
) -> Result<Tensor<Q::T, Q::B>> {
    let vb = VB::load(&[path], dev.clone())?;
    let names = vb.tensor_names();
    let key = names.first().context("no tensors found in voice embedding file")?;
    let shape = vb.shape(key).context("voice tensor not found")?;
    let dims = shape.dims().to_vec();
    let emb: Tensor<f32, Q::B> = vb.tensor(key, shape)?;
    // Voice files hold either [T, dim] or an already batched [1, T, dim].
    let emb = if dims.len() == 2 { emb.reshape((1, dims[0], dims[1]))? } else { emb };
    if let Some(model_ext) = cfg.model_ext() {
        let file_content = std::fs::read(path)?;
        let (_, metadata) = safetensors::SafeTensors::read_metadata(&file_content)?;
        if let Some(metadata) = metadata.metadata()
            && let Some(voice_model_ext) = metadata.get("model_ext")
            && voice_model_ext.as_str() != model_ext
        {
            anyhow::bail!(
                "voice embedding model_ext '{voice_model_ext}' does not match config model_ext '{model_ext}'"
            )
        }
    }
    Ok(emb.to::<Q::T>()?)
}

/// Frames an utterance of `num_tokens` tokens is allowed to generate before it is cut off.
fn max_frames_for(num_tokens: usize) -> usize {
    ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize
}

/// One iteration's timings.
struct Run {
    /// Start of the iteration to the first audio samples, so text conditioning is included but
    /// the voice conditioning shared by every iteration is not.
    ttfa: Duration,
    /// Per frame, sampling plus Mimi decoding.
    frames: Vec<Duration>,
    total: Duration,
    samples: usize,
}

/// Generates the utterance once, reusing the voice-conditioned state.
fn one<Q: BackendQ>(
    model: &TTSModel<Q>,
    base_state: &TTSState<Q>,
    chunks: &[(Vec<u32>, usize)],
    args: &Args,
) -> Result<Run> {
    let dev = model.device();
    let ldim = model.flow_lm.ldim;
    let mut rng = StdRng::new(args.temperature, args.seed)?;
    let mut frames = Vec::new();
    let mut ttfa = None;
    let mut samples = 0usize;
    let start = Instant::now();

    for (tokens, frames_after_eos) in chunks.iter() {
        let mut state = base_state.clone();
        model.prompt_text(&mut state, tokens)?;
        let mut mimi_state = model.init_mimi_state(1, MIMI_CONTEXT_SIZE)?;

        // BOS marker: an all-NaN latent.
        let nan: Tensor<f32, Q::B> = Tensor::from_vec(vec![f32::NAN; ldim], (1, 1, ldim), dev)?;
        let mut prev_latent = nan.to::<Q::T>()?;
        let mut eos_countdown: Option<usize> = None;

        for _ in 0..max_frames_for(tokens.len()) {
            let frame_start = Instant::now();
            let (next_latent, is_eos) = model.generate_step(&mut state, &prev_latent, &mut rng)?;
            // Decoding on this thread rather than overlapped, so the measurement attributes
            // sampling and decoding to the frame that caused them.
            let pcm = model.decode_latent(&next_latent, &mut mimi_state)?.to_vec()?;
            frames.push(frame_start.elapsed());
            if !pcm.is_empty() {
                ttfa.get_or_insert_with(|| start.elapsed());
                samples += pcm.len();
            }

            if is_eos && eos_countdown.is_none() {
                eos_countdown = Some(*frames_after_eos);
            }
            if let Some(countdown) = eos_countdown.as_mut() {
                if *countdown == 0 {
                    break;
                }
                *countdown -= 1;
            }
            prev_latent = next_latent;
        }
    }

    let total = start.elapsed();
    let ttfa = ttfa.context("no audio produced")?;
    Ok(Run { ttfa, frames, total, samples })
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

struct Stats {
    n: usize,
    min: f64,
    mean: f64,
    max: f64,
    p50: f64,
    p95: f64,
}

impl Stats {
    fn of(xs: &[f64]) -> Option<Self> {
        if xs.is_empty() {
            return None;
        }
        let mut s = xs.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pick = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
        Some(Stats {
            n: s.len(),
            min: s[0],
            mean: s.iter().sum::<f64>() / s.len() as f64,
            max: s[s.len() - 1],
            p50: pick(0.50),
            p95: pick(0.95),
        })
    }
}

fn row(label: &str, unit: &str, prec: usize, st: &Stats) {
    println!(
        "{label:<22} {:>5}  {:>9.*} {:>9.*} {:>9.*} {:>9.*} {:>9.*}  {unit}",
        st.n, prec, st.min, prec, st.mean, prec, st.p50, prec, st.p95, prec, st.max
    );
}

struct Bench<'a>(&'a Args);

impl xn::WithQ for Bench<'_> {
    fn run<Q: BackendQ>(self, dev: Q::B) -> xn::Result<()> {
        self.bench::<Q>(dev).map_err(|e| xn::Error::msg(format!("{e:?}")))
    }
}

impl Bench<'_> {
    fn bench<Q: BackendQ>(&self, dev: Q::B) -> Result<()> {
        let args = self.0;
        let cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(&args.config)?)
            .with_context(|| format!("failed to read config {}", args.config.display()))?;
        let tokenizer_path = match args.tokenizer.clone() {
            Some(path) => path,
            None => {
                args.config.parent().context("config path has no parent")?.join("tokenizer.model")
            }
        };

        let t_load = Instant::now();
        let tokenizer_path = tokenizer_path.to_str().context("invalid tokenizer path")?;
        let tokenizer = SpTokenizer(sentencepiece::SentencePieceProcessor::open(tokenizer_path)?);
        let vb = if args.model.extension().and_then(|v| v.to_str()) == Some("gguf") {
            let reader = std::io::BufReader::new(std::fs::File::open(&args.model)?);
            VB::load_gguf_with_key_map(reader, dev.clone(), remap_key)?
        } else {
            VB::load_with_key_map(&[&args.model], dev.clone(), remap_key)?
        };
        let vb = vb.root();
        let model: TTSModel<Q> = TTSModel::load(&vb, Box::new(tokenizer), &cfg)?;
        vb.check_all_used_with_ignore(|v| {
            v == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
                || v.starts_with("mimi.quantizer")
                || v.starts_with("mimi.encoder")
                || v.starts_with("speaker_mimi")
                || v == "flow_lm.speaker_proj_weight"
                || v == "mimi.downsample.conv.conv.weight"
        })?;
        let voice_emb = load_voice_emb::<Q>(&args.voice, &cfg, &dev)?;
        let load_ms = ms(t_load.elapsed());

        // Tokenize up front: the loop needs the tokens anyway, and the KV cache is sized from
        // them. Long inputs are split into sentences, as `pocket_tts` does.
        let chunks = ptts::tts_model::split_into_best_sentences(
            model.flow_lm.conditioner.tokenizer.as_deref().context("no tokenizer")?,
            &args.input,
            None,
        )?;
        let chunks = chunks
            .iter()
            .map(|chunk| {
                let (text, frames_after_eos) = ptts::tts_model::prepare_text_prompt(chunk);
                Ok((model.flow_lm.conditioner.tokenize(&text)?, frames_after_eos))
            })
            .collect::<Result<Vec<_>>>()?;

        // Condition on the voice once. Every iteration clones the resulting state, which is
        // what a server does per request, so the measurement is of generation rather than of
        // repeated voice conditioning.
        let voice_len = voice_emb.dim(1usize)?;
        let seq_budget = chunks
            .iter()
            .map(|(tokens, _)| voice_len + tokens.len() + max_frames_for(tokens.len()))
            .max()
            .unwrap_or(voice_len)
            + SEQ_BUDGET_SLACK;
        let t_voice = Instant::now();
        let mut base_state = model.init_flow_lm_state(1, seq_budget)?;
        model.prompt_audio(&mut base_state, &voice_emb)?;
        let voice_ms = ms(t_voice.elapsed());

        for _ in 0..args.warmup {
            one(&model, &base_state, &chunks, args)?;
        }
        let mut runs = Vec::with_capacity(args.iters);
        for i in 0..args.iters {
            let r = one(&model, &base_state, &chunks, args)?;
            if args.per_iter {
                println!(
                    "iter {i:>3}: total {:>8.2}ms  ttfa {:>7.2}ms  frames {:>4}",
                    ms(r.total),
                    ms(r.ttfa),
                    r.frames.len()
                );
            }
            runs.push(r);
        }
        if runs.is_empty() {
            anyhow::bail!("--iters must be at least 1")
        }

        let audio_ms = runs[0].samples as f64 / model.sample_rate() as f64 * 1e3;
        let totals: Vec<f64> = runs.iter().map(|r| ms(r.total)).collect();
        let ttfas: Vec<f64> = runs.iter().map(|r| ms(r.ttfa)).collect();
        // Pooled across iterations: per-frame variation matters more than which run it came
        // from, and one run has too few frames for a stable tail.
        let frames: Vec<f64> = runs.iter().flat_map(|r| r.frames.iter().copied().map(ms)).collect();
        // Audio produced per unit of wall time, so higher is faster than realtime.
        let rtfs: Vec<f64> = totals.iter().map(|t| audio_ms / t).collect();

        println!();
        println!(
            "model {}  threads {}  input {} chars  audio {audio_ms:.0}ms  frames/iter {}",
            args.model.display(),
            xn::get_num_threads(),
            args.input.len(),
            runs[0].frames.len(),
        );
        println!("load {load_ms:.1}ms, voice conditioning {voice_ms:.1}ms (both excluded below)");
        println!();
        println!(
            "{:<22} {:>5}  {:>9} {:>9} {:>9} {:>9} {:>9}",
            "metric", "n", "min", "mean", "p50", "p95", "max"
        );
        if let Some(s) = Stats::of(&totals) {
            row("total generate", "ms", 2, &s);
        }
        if let Some(s) = Stats::of(&ttfas) {
            row("time to first audio", "ms", 2, &s);
        }
        if let Some(s) = Stats::of(&frames) {
            row("per-frame", "ms", 3, &s);
        }
        if let Some(s) = Stats::of(&rtfs) {
            row("rtf (higher is better)", "x realtime", 2, &s);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    use std::str::FromStr;

    let args = Args::parse();
    if let Some(threads) = args.threads {
        // Must happen before the first tensor op, since it sets the size of rayon's global pool.
        xn::set_num_threads(threads);
    }
    let dtype = match args.quant.as_deref() {
        Some(quant) => xn::DTypeQ::from_str(quant)?,
        None if args.model.extension().and_then(|v| v.to_str()) == Some("gguf") => {
            anyhow::bail!("GGUF weights need an explicit --quant, e.g. --quant q8")
        }
        None => xn::DTypeQ::F32,
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        xn::with_avx(),
        xn::with_neon(),
        xn::with_simd128(),
        xn::with_f16c()
    );
    xn::Runner::new().cpu_only(args.cpu).dtype(dtype).run(Bench(&args), 0)?;
    Ok(())
}
