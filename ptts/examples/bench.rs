//! Benchmark harness for TTS generation.
//!
//! Loads a local model once, then generates the same utterance `--iters` times and reports
//! time-to-first-audio, per-frame time, total generate time and RTF. Model load and voice
//! conditioning are timed separately and excluded from the per-iteration statistics.
//!
//! By default each iteration is a `Synth::stream` call, so the numbers are the ones a caller
//! gets: the flow LM and Mimi on their own threads, Mimi decoding whatever frames have queued
//! up in one call. `--breakdown` instead runs both on this thread, one frame at a time, which
//! is slower overall but times each stage on its own.

#[path = "../src/bin/ptts/model_helpers.rs"]
mod model_helpers;

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use ptts::flow_lm::{NormalRng, StepInput};
use ptts::plan::{Chunk, EosPolicy};
use ptts::preprocess::{Normalize, Rules};
use ptts::synth::{DeviceKind, Quant, SpeechOptions, Synth, SynthBuilder};
use ptts::tok::Tok;
use ptts::tts_model::{MAX_TOKENS_PER_CHUNK, TTSConfig, TTSModel, TTSState};
use xn::{BackendQ, Tensor};

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

    /// Tokenizer json. Defaults to `tokenizer.json` next to the config.
    #[arg(long)]
    tokenizer: Option<std::path::PathBuf>,

    /// Precomputed voice embedding safetensors, or the name of a baked-in voice.
    #[arg(long)]
    voice: std::path::PathBuf,

    /// Weight quantization, e.g. `q8`. Required for GGUF weights; safetensors load as f32.
    #[arg(long)]
    quant: Option<String>,

    /// Use the cpu device even if a gpu backend is available.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Number of CPU threads for tensor ops.
    #[arg(long)]
    threads: Option<usize>,

    #[arg(
        long = "condition",
        value_name = "NAME=VALUE",
        help = "Set a conditioner, e.g. duration_delta=0.5; repeatable"
    )]
    conditions: Vec<String>,

    #[arg(long, short, default_value = "Hello, this is a test of the Phonon TTS system.")]
    input: String,

    #[arg(long, default_value_t = 0.3)]
    temperature: f32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Measured iterations.
    #[arg(long, default_value_t = 10, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    iters: usize,

    /// Unmeasured iterations run first, to warm caches and the thread pool.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Print a line per iteration as well as the summary.
    #[arg(long, default_value_t = false)]
    per_iter: bool,

    /// Run the flow LM and Mimi on this thread, one frame at a time, and time each stage,
    /// instead of measuring `Synth::stream`. Slower overall than the real path, and its time
    /// to first audio is lower, since nothing runs beside the first decode.
    #[arg(long, default_value_t = false)]
    breakdown: bool,

    /// Run the Mimi decoder's transformer and its two widest convolutions in q8_0. CPU only.
    #[arg(long)]
    quantize_mimi: bool,

    /// Language the input is normalized as before tokenizing: `en`, `fr`, `de`, `es` or `pt`.
    /// Required, as everywhere else: there is nothing safe to guess.
    /// `none` measures the unnormalized text, as runs that predate this flag did.
    #[arg(long)]
    lang: String,

    /// Which word rewrites run on the normalized text: `default` (numbers, currency,
    /// dashed-digits, emails, urls, abbreviations, elongations), `all` (those and phones, times,
    /// dates), `none`, or a comma-separated list of rule names. Has no effect with `--lang none`.
    #[arg(long, default_value = "default")]
    rewrites: String,
}

/// One iteration's timings.
struct Run {
    /// Start of the iteration to the first audio samples, so text conditioning is included but
    /// the voice conditioning shared by every iteration is not.
    ttfa: Duration,
    /// Per frame, sampling plus Mimi decoding. Only with `--breakdown`.
    frames: Vec<Duration>,
    /// The sampling half of each frame. Only with `--breakdown`.
    sample_t: Vec<Duration>,
    /// The Mimi decode half of each frame. Only with `--breakdown`.
    decode_t: Vec<Duration>,
    /// Frames generated.
    nframes: usize,
    total: Duration,
    samples: usize,
}

/// Generates the utterance once through `Synth`, as any caller would.
fn one_synth(tts: &Synth, opts: &SpeechOptions, args: &Args) -> Result<Run> {
    let start = Instant::now();
    let mut ttfa = None;
    let mut samples = 0usize;
    for pcm in tts.stream_with(&args.input, opts)? {
        let pcm = pcm?;
        if !pcm.is_empty() {
            ttfa.get_or_insert_with(|| start.elapsed());
            samples += pcm.len();
        }
    }
    let total = start.elapsed();
    let ttfa = ttfa.context("no audio produced")?;
    let frame_samples = tts.sample_rate() as f64 / tts.config().mimi.frame_rate;
    let nframes = (samples as f64 / frame_samples).round() as usize;
    Ok(Run { ttfa, frames: vec![], sample_t: vec![], decode_t: vec![], nframes, total, samples })
}

/// Generates the utterance once, reusing the voice-conditioned state.
fn one<Q: BackendQ>(
    model: &TTSModel<Q>,
    base_state: &TTSState<Q>,
    chunks: &[Chunk],
    args: &Args,
) -> Result<Run> {
    let mut rng = NormalRng::new(args.temperature, args.seed)?;
    let mut frames = Vec::new();
    let mut sample_t = Vec::new();
    let mut decode_t = Vec::new();
    let mut ttfa = None;
    let mut samples = 0usize;
    let start = Instant::now();

    for chunk in chunks {
        let mut state = base_state.clone();
        model.prompt_text(&mut state, &chunk.tokens)?;
        let mut mimi_state = model.init_mimi_state(1)?;

        let mut prev_latent: Option<Tensor<Q::T, Q::B>> = None;
        let mut eos = EosPolicy::new(chunk.frames_after_eos);

        for _ in 0..chunk.frame_budget {
            let frame_start = Instant::now();
            let input = match &prev_latent {
                None => StepInput::Bos { batch: 1 },
                Some(t) => StepInput::Latent(t),
            };
            let (next_latent, is_eos) = model.generate_step(&mut state, input, &mut rng)?;
            let sampled = Instant::now();
            // Decoding on this thread rather than overlapped, so the measurement attributes
            // sampling and decoding to the frame that caused them.
            let pcm = model.decode_latent(&next_latent, &mut mimi_state)?.to_vec()?;
            let done = Instant::now();
            sample_t.push(sampled - frame_start);
            decode_t.push(done - sampled);
            frames.push(done - frame_start);
            if !pcm.is_empty() {
                ttfa.get_or_insert_with(|| start.elapsed());
                samples += pcm.len();
            }

            if eos.should_stop(is_eos) {
                break;
            }
            prev_latent = Some(next_latent);
        }
    }

    let total = start.elapsed();
    let ttfa = ttfa.context("no audio produced")?;
    let nframes = frames.len();
    Ok(Run { ttfa, frames, sample_t, decode_t, nframes, total, samples })
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
    /// `xs` must be non-empty; `--iters` is validated to be at least 1.
    fn of(xs: &[f64]) -> Self {
        let mut s = xs.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pick = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
        Stats {
            n: s.len(),
            min: s[0],
            mean: s.iter().sum::<f64>() / s.len() as f64,
            max: s[s.len() - 1],
            p50: pick(0.50),
            p95: pick(0.95),
        }
    }
}

fn row(label: &str, unit: &str, prec: usize, st: &Stats) {
    println!(
        "{label:<22} {:>5}  {:>9.*} {:>9.*} {:>9.*} {:>9.*} {:>9.*}  {unit}",
        st.n, prec, st.min, prec, st.mean, prec, st.p50, prec, st.p95, prec, st.max
    );
}

struct Bench<'a>(&'a Args, Normalize);

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
        let tokenizer_path = match args.tokenizer.clone() {
            Some(path) => path,
            None => {
                args.config.parent().context("config path has no parent")?.join("tokenizer.json")
            }
        };

        let t_load = Instant::now();
        let tokenizer = Tok::open(&tokenizer_path)?;
        let vb = model_helpers::load_weights::<Q>(&args.model, &dev)?;
        let baked = cfg.voices.iter().find(|v| args.voice.as_os_str() == v.name.as_str());
        if baked.is_none() && !cfg.voices.is_empty() {
            let known: Vec<&str> = cfg.voices.iter().map(|v| v.name.as_str()).collect();
            anyhow::bail!(
                "unknown voice {:?}: this checkpoint has baked-in voices and supports no other \
                 voice, known voices: {known:?}",
                args.voice
            );
        }
        // A baked-in voice's own values win over those given, as in `SynthBuilder::build`.
        let mut conditions = std::collections::HashMap::new();
        for condition in &args.conditions {
            let (name, value) =
                condition.split_once('=').context("--condition takes NAME=VALUE")?;
            conditions.insert(name.to_string(), value.to_string());
        }
        conditions.extend(baked.map(|v| v.conditions.clone()).unwrap_or_default());
        let mut model: TTSModel<Q> = TTSModel::load(&vb, Box::new(tokenizer), &cfg, &conditions)?;
        if args.quantize_mimi {
            model.quantize_mimi()?;
        }
        let baked_voices = ptts::loader::load_config_voices(&vb, &cfg, model.speaker_proj())?;
        let speaker_prefix = format!("{}.", cfg.speaker_mimi_prefix());
        vb.check_all_used_with_ignore(|name| {
            model_helpers::is_unused_by_tts_model(name) || name.starts_with(&speaker_prefix)
        })?;
        let voice_emb = match baked {
            Some(voice) => baked_voices
                .into_iter()
                .find(|(name, _)| *name == voice.name)
                .map(|(_, emb)| emb)
                .context("baked-in voice not loaded")?,
            None => model_helpers::load_voice_emb(
                &args.voice,
                cfg.embeddings_ext.as_deref(),
                model.speaker_proj(),
                &dev,
            )?,
        }
        .to::<Q::T>()?;
        let load_ms = ms(t_load.elapsed());

        // Tokenize up front: the loop needs the tokens anyway, and the KV cache is sized from
        // them. The chunks are the ones `ptts` makes.
        let tokenizer = model.flow_lm.conditioner.tokenizer.as_deref().context("no tokenizer")?;
        let chunks = ptts::plan::chunks(
            tokenizer,
            &args.input,
            self.1,
            MAX_TOKENS_PER_CHUNK,
            cfg.mimi.frame_rate,
        )?;
        let chunks = ptts::plan::fit_or_error(
            chunks,
            ptts::plan::MAX_FIT_TOKENS,
            tokenizer,
            cfg.mimi.frame_rate,
        )?;

        // Condition on the voice once. Every iteration clones the resulting state, which is
        // what a server does per request, so the measurement is of generation rather than of
        // repeated voice conditioning.
        let voice_len = voice_emb.dim(1usize)?;
        let seq_budget = chunks
            .iter()
            .map(|chunk| voice_len + chunk.tokens.len() + chunk.frame_budget)
            .max()
            .unwrap_or(voice_len);
        let t_voice = Instant::now();
        let mut base_state = model.init_flow_lm_state(1, seq_budget)?;
        model.prompt_audio(&mut base_state, &voice_emb)?;
        let voice_ms = ms(t_voice.elapsed());

        let runs = measure(args, || one(&model, &base_state, &chunks, args))?;
        report(args, self.1, &runs, model.sample_rate() as f64, load_ms, voice_ms);
        Ok(())
    }
}

/// Runs `--warmup` unmeasured iterations, then `--iters` measured ones.
fn measure(args: &Args, mut one: impl FnMut() -> Result<Run>) -> Result<Vec<Run>> {
    for _ in 0..args.warmup {
        one()?;
    }
    let mut runs = Vec::with_capacity(args.iters);
    for i in 0..args.iters {
        let r = one()?;
        if args.per_iter {
            println!(
                "iter {i:>3}: total {:>8.2}ms  ttfa {:>7.2}ms  frames {:>4}",
                ms(r.total),
                ms(r.ttfa),
                r.nframes
            );
        }
        runs.push(r);
    }
    anyhow::ensure!(runs.iter().all(|r| r.samples > 0), "no audio generated, nothing to measure");
    Ok(runs)
}

fn report(
    args: &Args,
    normalize: Normalize,
    runs: &[Run],
    sample_rate: f64,
    load_ms: f64,
    voice_ms: f64,
) {
    let first = &runs[0];
    let audio_ms = |r: &Run| r.samples as f64 / sample_rate * 1e3;
    let totals: Vec<f64> = runs.iter().map(|r| ms(r.total)).collect();
    let ttfas: Vec<f64> = runs.iter().map(|r| ms(r.ttfa)).collect();
    // Wall time per unit of audio produced, so below 1.0 is faster than realtime.
    let rtfs: Vec<f64> = runs.iter().map(|r| ms(r.total) / audio_ms(r)).collect();

    println!();
    println!(
        "model {}  threads {}  input {} chars  audio {:.0}ms  frames/iter {}",
        args.model.display(),
        xn::get_num_threads(),
        normalize.apply(&args.input).len(),
        audio_ms(first),
        first.nframes,
    );
    if args.breakdown {
        println!("--breakdown: flow LM and Mimi on one thread, one frame at a time");
    } else {
        println!("Synth::stream: flow LM and Mimi on their own threads, as callers run it");
    }
    println!("load {load_ms:.1}ms, voice conditioning {voice_ms:.1}ms (both excluded below)");
    println!();
    println!(
        "{:<22} {:>5}  {:>9} {:>9} {:>9} {:>9} {:>9}",
        "metric", "n", "min", "mean", "p50", "p95", "max"
    );
    let mut rows =
        vec![("total generate", "ms", 2, totals), ("time to first audio", "ms", 2, ttfas)];
    if args.breakdown {
        // Pooled across iterations: per-frame variation matters more than which run it came
        // from, and one run has too few frames for a stable tail.
        let pooled = |f: fn(&Run) -> &Vec<Duration>| -> Vec<f64> {
            runs.iter().flat_map(|r| f(r).iter().copied().map(ms)).collect()
        };
        rows.push(("per-frame", "ms", 3, pooled(|r| &r.frames)));
        rows.push(("  flow_lm sample", "ms", 3, pooled(|r| &r.sample_t)));
        rows.push(("  mimi decode", "ms", 3, pooled(|r| &r.decode_t)));
    } else {
        // Frames overlap and Mimi decodes several at once, so only the average is meaningful.
        let per_frame = runs.iter().map(|r| ms(r.total) / r.nframes as f64).collect();
        rows.push(("per-frame (average)", "ms", 3, per_frame));
    }
    rows.push(("rtf (lower is better)", "ratio", 4, rtfs));
    for (label, unit, prec, xs) in rows {
        row(label, unit, prec, &Stats::of(&xs));
    }
}

/// Measures `Synth::stream`, loading the model the way the other frontends do.
fn bench_synth(args: &Args, normalize: Normalize, quant: Quant) -> Result<()> {
    let cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(&args.config)?)
        .with_context(|| format!("failed to read config {}", args.config.display()))?;
    let tokenizer_path = match args.tokenizer.clone() {
        Some(path) => path,
        None => args.config.parent().context("config path has no parent")?.join("tokenizer.json"),
    };
    let baked = cfg.voices.iter().any(|v| args.voice.as_os_str() == v.name.as_str());
    let voice = if baked { args.voice.to_string_lossy().into_owned() } else { "bench".into() };

    let t_load = Instant::now();
    // Quantized weights run only on the CPU, so a GPU build picks it for them, as `--breakdown`
    // does, rather than failing to build.
    let device = if args.cpu || quant != Quant::F32 || args.quantize_mimi {
        DeviceKind::Cpu
    } else {
        DeviceKind::Auto
    };
    let mut builder = SynthBuilder::new(cfg, &args.model, normalize)
        .tokenizer_file(tokenizer_path)
        .device(device)
        .quant(quant)
        .quantize_mimi(args.quantize_mimi)
        .temperature(args.temperature)
        .seed(args.seed)
        .voice(&voice);
    for condition in &args.conditions {
        let (name, value) = condition.split_once('=').context("--condition takes NAME=VALUE")?;
        builder = builder.condition(name, value);
    }
    if !baked {
        builder = builder.add_voice(&voice, &args.voice);
    }
    let tts = builder.build()?;
    let load_ms = ms(t_load.elapsed());

    // `Synth` conditions on a voice the first time it speaks in it and keeps the result, as a
    // server does, so this primes it and later iterations start from the kept state. The
    // session is thrown away, so its budget only has to exceed any voice prompt.
    let opts = SpeechOptions::default();
    let t_voice = Instant::now();
    drop(tts.session(&opts, 2048)?);
    let voice_ms = ms(t_voice.elapsed());

    let runs = measure(args, || one_synth(&tts, &opts, args))?;
    report(args, normalize, &runs, tts.sample_rate() as f64, load_ms, voice_ms);
    Ok(())
}

fn main() -> Result<()> {
    use std::str::FromStr;

    let args = Args::parse();
    // Parsed before the weights are read, so a bad --lang does not cost a model load.
    let normalize = args.lang.parse::<Normalize>()?.with_rules(args.rewrites.parse::<Rules>()?);
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
    if args.breakdown {
        // Mimi quantization is CPU-only, so it picks the CPU, as `bench_synth` does.
        let cpu = args.cpu || args.quantize_mimi;
        xn::Runner::new().cpu_only(cpu).dtype(dtype).run(Bench(&args, normalize), 0)?;
    } else {
        bench_synth(&args, normalize, Quant::from_str(args.quant.as_deref().unwrap_or("f32"))?)?;
    }
    Ok(())
}
