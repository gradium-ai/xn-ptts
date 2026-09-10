#[path = "audio_helpers.rs"]
mod audio_helpers;
#[path = "model_helpers.rs"]
mod model_helpers;

use anyhow::{Context, Result};
use clap::Parser;
use ptts::flow_lm::{NormalRng, ReplayRng};
use ptts::plan::{EosPolicy, frame_budget, seq_budget};
use ptts::tok::Tok;
use ptts::tts_model::{
    MimiEnc, TTSConfig, TTSModel, prepare_text_prompt, split_into_best_sentences,
};
use xn::Tensor;

#[derive(Parser, Debug)]
#[command(name = "pocket-tts")]
#[command(about = "Generate speech from text using Pocket TTS")]
struct Args {
    /// Text to synthesize
    text: String,

    #[arg(long)]
    config: Option<String>,

    #[arg(long)]
    weights: Option<String>,

    #[arg(long)]
    cfg_coef: Option<f32>,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: std::path::PathBuf,

    /// Voice to use
    #[arg(short, long)]
    voice: Option<String>,

    /// Sampling temperature
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Sampling seed
    #[arg(short, long, default_value_t = 4242424242424242)]
    seed: u64,

    /// Use the cpu device even if a gpu backend is available
    #[arg(long, default_value_t = false)]
    cpu: bool,

    #[arg(long)]
    quant: Option<String>,

    #[arg(long)]
    chrome_tracing: bool,

    #[arg(long)]
    rng_values: Option<String>,

    #[arg(long)]
    wait_to_decode: bool,

    /// Decode every queued frame in one call instead of one frame per call.
    #[arg(long, default_value_t = false)]
    decode_batching: bool,

    /// Number of CPU threads for tensor ops.
    #[arg(long)]
    threads: Option<usize>,

    #[arg(long)]
    pad_to: Option<usize>,

    /// Normalize the text for this language before tokenizing: `en`, `fr`, `de`, `es` or `pt`.
    /// Drops characters the model never saw and spells out `@`, `+` and `=`. Omitted, the text
    /// is tokenized exactly as given.
    #[arg(long)]
    lang: Option<String>,
}

const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

enum Voice {
    Safetensors(std::path::PathBuf),
    Audio(String),
}

fn download_files(voice: &str) -> Result<(std::path::PathBuf, std::path::PathBuf, Voice)> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let repo_id = "kyutai/pocket-tts";
    tracing::info!(?repo_id, "downloading weights...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let model_path = repo.get("tts_b6369a24.safetensors").context("model weights not found")?;
    tracing::info!(?model_path, "model weights downloaded");

    let tokenizer_path = repo.get("tokenizer.model").context("tokenizer not found")?;
    tracing::info!(?tokenizer_path, "tokenizer downloaded");

    let voice = if VOICES.contains(&voice) {
        let voice_file = format!("embeddings/{voice}.safetensors");
        let voice_path = repo
            .get(&voice_file)
            .with_context(|| format!("voice embedding '{voice}' not found"))?;
        tracing::info!(?voice_path, "voice embedding downloaded");
        Voice::Safetensors(voice_path)
    } else if voice.ends_with(".safetensors") {
        let voice_path = std::path::PathBuf::from(voice);
        if !voice_path.exists() {
            anyhow::bail!("voice embedding file '{}' not found", voice_path.display());
        }
        Voice::Safetensors(voice_path)
    } else {
        Voice::Audio(voice.to_string())
    };
    Ok((model_path, tokenizer_path, voice))
}

fn init_tracing(chrome_tracing: bool) -> Option<tracing_chrome::FlushGuard> {
    use tracing_subscriber::{EnvFilter, prelude::*};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    if chrome_tracing {
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new().build();
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::Layer::new().with_target(false))
            .with(chrome_layer)
            .with(filter)
            .init();
        Some(guard)
    } else {
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::Layer::new().with_target(false))
            .with(filter)
            .init();
        None
    }
}

fn run_cpu(args: Args) -> Result<()> {
    match args.quant.as_deref() {
        None => {
            tracing::info!("using cpu backend");
            run_for_device::<xn::Unquantized<f32, _>>(args, xn::CPU)
        }
        Some("q8" | "q8_0") => {
            tracing::info!("using cpu q8 backend");
            run_for_device::<xn::quantized::Q80F32>(args, xn::CPU)
        }
        Some("q8_1") => {
            tracing::info!("using cpu q8_1 backend");
            run_for_device::<xn::quantized::Q81F32>(args, xn::CPU)
        }
        Some("q8k") => {
            tracing::info!("using cpu q8k backend");
            run_for_device::<xn::quantized::Q8kF32>(args, xn::CPU)
        }
        Some("q6k") => {
            tracing::info!("using cpu q6k backend");
            run_for_device::<xn::quantized::Q6kF32>(args, xn::CPU)
        }
        Some("q5" | "q5_0") => {
            tracing::info!("using cpu q5 backend");
            run_for_device::<xn::quantized::Q50F32>(args, xn::CPU)
        }
        Some("q5_1") => {
            tracing::info!("using cpu q5_1 backend");
            run_for_device::<xn::quantized::Q51F32>(args, xn::CPU)
        }
        Some("q5k") => {
            tracing::info!("using cpu q5k backend");
            run_for_device::<xn::quantized::Q5kF32>(args, xn::CPU)
        }
        Some("q4" | "q4_0") => {
            tracing::info!("using cpu q4 backend");
            run_for_device::<xn::quantized::Q40F32>(args, xn::CPU)
        }
        Some("q4_1") => {
            tracing::info!("using cpu q4_1 backend");
            run_for_device::<xn::quantized::Q41F32>(args, xn::CPU)
        }
        Some("q4k") => {
            tracing::info!("using cpu q4k backend");
            run_for_device::<xn::quantized::Q4kF32>(args, xn::CPU)
        }
        Some(other) => anyhow::bail!("unsupported quantization option '{other}'"),
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(threads) = args.threads {
        // Must happen before the first tensor op, since it sets the size of rayon's global pool.
        xn::set_num_threads(threads);
    }
    let _guard = init_tracing(args.chrome_tracing);

    #[cfg(feature = "cuda")]
    {
        if args.cpu {
            run_cpu(args)?;
        } else {
            tracing::info!("using cuda backend");
            let dev = xn::cuda_backend::Device::new(0)?;
            unsafe {
                dev.disable_event_tracking();
            }
            run_for_device::<xn::Unquantized<half::bf16, _>>(args, dev)?;
        }
    }
    #[cfg(all(feature = "vulkan", not(feature = "cuda")))]
    {
        if args.cpu {
            run_cpu(args)?;
        } else {
            tracing::info!("using vulkan backend");
            let dev = xn::vulkan_backend::Device::new(0)?;
            run_for_device::<xn::Unquantized<f32, _>>(args, dev)?;
        }
    }
    #[cfg(all(feature = "metal", not(any(feature = "cuda", feature = "vulkan"))))]
    {
        if args.cpu {
            run_cpu(args)?;
        } else {
            tracing::info!("using metal backend");
            let dev = xn::metal_backend::Device::new(0)?;
            run_for_device::<xn::Unquantized<half::bf16, _>>(args, dev)?;
        }
    }
    #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "metal")))]
    {
        run_cpu(args)?;
    }

    if let Some(rss_mb) = peak_rss_mb() {
        tracing::info!("peak RSS: {rss_mb:.2} MB");
    }

    Ok(())
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

fn spawn<F, R>(f: F) -> std::thread::JoinHandle<R>
where
    F: FnOnce() -> Result<R>,
    F: Send + 'static,
    R: Send + 'static,
{
    std::thread::spawn(move || match f() {
        Err(e) => {
            tracing::error!(?e, "thread error");
            std::process::exit(1);
        }
        Ok(res) => res,
    })
}

fn run_for_device<Q: xn::BackendQ + 'static>(args: Args, dev: Q::B) -> Result<()> {
    use std::str::FromStr;
    let (model_path, tokenizer_path, voice, cfg) = match args.config.as_ref() {
        Some(config) => {
            let config = std::fs::canonicalize(config)?;
            let parent = config.parent().context("config path has no parent")?;
            let model_path = match args.weights.as_ref() {
                None => parent.join("model.safetensors"),
                Some(p) => std::path::PathBuf::from_str(p)?,
            };
            let tokenizer_path = parent.join("tokenizer.model");
            tracing::info!(?config, "using local config");
            let config: ptts::tts_model::TTSConfig =
                serde_json::from_str(&std::fs::read_to_string(config)?)?;
            let voice = match args.voice {
                None => None,
                Some(voice) if voice.ends_with(".safetensors") => {
                    let voice_path = std::path::PathBuf::from(voice);
                    if !voice_path.exists() {
                        anyhow::bail!("voice embedding file '{}' not found", voice_path.display());
                    }
                    Some(Voice::Safetensors(voice_path))
                }
                Some(voice) => Some(Voice::Audio(voice)),
            };
            (model_path, tokenizer_path, voice, config)
        }
        None => {
            if args.weights.is_some() {
                anyhow::bail!("--weights option is not supported without --config");
            }
            let voice = args.voice.unwrap_or("alba".to_string());
            if !VOICES.contains(&voice.as_str()) && !std::path::PathBuf::from(&voice).exists() {
                anyhow::bail!("unknown voice '{voice}'. Available voices: {}", VOICES.join(", "));
            }

            let (model_path, tokenizer_path, voice) = download_files(&voice)?;
            (model_path, tokenizer_path, Some(voice), TTSConfig::v202601(args.temperature))
        }
    };

    let tokenizer = Tok::open(&tokenizer_path)?;
    let text = match args.lang.as_deref() {
        None => std::borrow::Cow::Borrowed(args.text.as_str()),
        Some(lang) => {
            let lang = ptts::preprocess::Lang::from_str(lang)?;
            let normalized = ptts::preprocess::normalize_text(&args.text, lang);
            tracing::info!(?normalized, "normalized input text");
            std::borrow::Cow::Owned(normalized)
        }
    };
    let chunks = split_into_best_sentences(&tokenizer, &text, None)?;

    let mut rng: Box<dyn ptts::flow_lm::Rng + Send> = match args.rng_values {
        Some(path) => {
            let values: Vec<f32> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            Box::new(ReplayRng::new(values)?)
        }
        None => Box::new(NormalRng::new(args.temperature, args.seed)?),
    };

    tracing::info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        xn::with_avx(),
        xn::with_neon(),
        xn::with_simd128(),
        xn::with_f16c()
    );

    let model_ext = cfg.model_ext();
    tracing::info!(?model_path, ?model_ext, "loading model");
    let vb = model_helpers::load_weights::<Q>(&model_path, &dev)?;
    let model: TTSModel<Q> = TTSModel::load(&vb, Box::new(tokenizer), &cfg)?;
    vb.check_all_used_with_ignore(model_helpers::is_unused_by_tts_model)?;
    tracing::info!("model loaded successfully!");

    let mut max_seq_budget = 0;
    let mut all_tokens = vec![];
    for chunk in chunks.iter() {
        let (text, frames_after_eos) = prepare_text_prompt(chunk);
        let tokens = model.flow_lm.conditioner.tokenize(&text)?;
        let num_tokens = tokens.len();
        tracing::info!(?text, ?num_tokens, "processing text");
        let max_frames = frame_budget(num_tokens, cfg.mimi.frame_rate);
        max_seq_budget = max_seq_budget.max(seq_budget(num_tokens, max_frames));
        all_tokens.push((tokens, max_frames, frames_after_eos));
    }
    // Init states
    let mut tts_state = model.init_flow_lm_state(1, max_seq_budget)?;
    let mut cfg_state = match args.cfg_coef {
        Some(1.0) | None => None,
        Some(coef) => {
            tracing::info!(?coef, "using custom cfg coefficient");
            let null_state = model.init_flow_lm_state(1, max_seq_budget)?;
            Some((coef, null_state))
        }
    };
    let mimi_state = model.init_mimi_state(1)?;

    // Load voice embedding
    if let Some(voice) = voice {
        let (voice_emb, null_emb) = match voice {
            Voice::Safetensors(voice_path) => {
                if cfg_state.is_some() {
                    anyhow::bail!("cfg is not supported with pre-computed voice embeddings");
                }
                let voice_emb =
                    model_helpers::load_voice_emb(&voice_path, cfg.model_ext().as_deref(), &dev)?
                        .to::<Q::T>()?;
                (voice_emb, None)
            }
            Voice::Audio(path) => {
                tracing::info!("loading voice from audio file {}", path);
                let speaker_sr = cfg.speaker_mimi_cfg().sample_rate;
                let (mut pcm, sample_rate) = audio_helpers::pcm_decode(&path)?;
                ptts::utils::normalize_loudness(&mut pcm, sample_rate)?;
                let sample_rate = sample_rate as usize;
                let pcm = if sample_rate != speaker_sr {
                    audio_helpers::resample(&pcm, sample_rate, speaker_sr)?
                } else {
                    pcm
                };
                tracing::info!("loaded audio with {} samples", pcm.len());
                let sr = speaker_sr as f32;
                let min_len = (sr * cfg.audio_prompt_min_duration).round() as usize;
                let max_len = (sr * cfg.audio_prompt_max_duration).round() as usize;
                let pcm = if pcm.len() > max_len {
                    tracing::info!(
                        max_duration = cfg.audio_prompt_max_duration,
                        "trimming audio to max duration"
                    );
                    pcm[..max_len].to_vec()
                } else {
                    pcm
                };
                if pcm.len() < min_len {
                    anyhow::bail!(
                        "audio prompt is shorter than min duration ({}s, {} samples; got {})",
                        cfg.audio_prompt_min_duration,
                        min_len,
                        pcm.len()
                    );
                }
                let pcm_tensor = Tensor::from_vec(pcm, (1, 1, ()), &dev)?.to::<Q::T>()?;
                let mimi_enc: MimiEnc<Q> = MimiEnc::load(&vb, &cfg)?;
                let emb = mimi_enc.encode_audio(&pcm_tensor)?;
                tracing::info!(?emb, "encoded audio to latent");
                let null_emb = if cfg_state.is_some() && !cfg.cfg_null_audio_empty {
                    let null_pcm_tensor = pcm_tensor.zeros_like()?;
                    Some(mimi_enc.encode_audio(&null_pcm_tensor)?)
                } else {
                    None
                };
                (emb, null_emb)
            }
        };
        // Prompt with audio conditioning
        tracing::info!("prompting with voice conditioning ({} frames)...", voice_emb.dim(1usize)?);
        let start = std::time::Instant::now();
        model.prompt_audio(&mut tts_state, &voice_emb)?;
        if let Some((_, ref mut null_state)) = cfg_state
            && let Some(null_emb) = null_emb.as_ref()
        {
            model.prompt_audio(null_state, null_emb)?;
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::info!("done prompting with voice conditioning in {elapsed_ms:.2}ms");
    }

    let start = std::time::Instant::now();
    tracing::info!("starting generation...");
    let mut all_audios = vec![];
    let mut backbone_step_timings_ms = vec![];
    let model = std::sync::Arc::new(model);
    for (tokens, max_frames, frames_after_eos) in all_tokens.into_iter() {
        tracing::info!("prompting with text conditioning ({} tokens)...", tokens.len());
        let start = std::time::Instant::now();
        let mut tts_state = tts_state.clone();
        let mut mimi_state = mimi_state.clone();
        if let Some(pad_to) = args.pad_to {
            model.prompt_text_with_padding(&mut tts_state, &tokens, pad_to)?;
        } else {
            model.prompt_text(&mut tts_state, &tokens)?;
        }
        if let Some((_, ref mut null_state)) = cfg_state {
            model.prompt_text_null(null_state)?;
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::info!("done prompting with text conditioning in {elapsed_ms:.2}ms");

        // BOS marker: NaN tensor [1, 1, ldim]
        let ldim = cfg.flow_lm.ldim;
        let nan_data: Vec<f32> = vec![f32::NAN; ldim];
        let mut prev_latent: Tensor<Q::T, Q::B> =
            Tensor::from_vec(nan_data, (1, 1, ldim), &dev)?.to::<Q::T>()?;

        let mut eos = EosPolicy::new(frames_after_eos);

        let (latent_tx, latent_rx) = std::sync::mpsc::channel::<Tensor<Q::T, _>>();
        let is_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let jh = spawn({
            let wait_to_decode = args.wait_to_decode;
            let decode_batching = args.decode_batching;
            let model = model.clone();
            let is_done = is_done.clone();
            move || {
                let mut audio_chunks: Vec<Tensor<f32, Q::B>> = Vec::new();
                if wait_to_decode {
                    tracing::info!("waiting for generation to finish before decoding...");
                    while !is_done.load(std::sync::atomic::Ordering::SeqCst) {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                }
                while let Ok(next_latent) = latent_rx.recv() {
                    // Decode every latent the generator has queued in one streaming call: the
                    // decoder is exact for any number of frames and one call over several frames
                    // is much cheaper than one call per frame. Only what is already queued is
                    // taken, so the first frame never waits.
                    let mut latents = vec![next_latent];
                    if decode_batching {
                        while let Ok(more) = latent_rx.try_recv() {
                            latents.push(more);
                        }
                    }
                    let refs: Vec<&Tensor<Q::T, _>> = latents.iter().collect();
                    let batch = Tensor::cat(&refs, 1)?.to()?;
                    let audio_chunk = model.decode_latent(&batch, &mut mimi_state)?;
                    audio_chunks.push(audio_chunk);
                }
                // Concatenate audio
                let audio_refs: Vec<&Tensor<f32, Q::B>> = audio_chunks.iter().collect();
                let audio = Tensor::cat(&audio_refs, 2)?;
                let audio = audio.narrow(0, ..1)?.contiguous()?;
                Ok::<_, anyhow::Error>(audio)
            }
        });

        for step in 0..max_frames {
            let step_start = std::time::Instant::now();
            let (next_latent, is_eos) = match cfg_state.as_mut() {
                Some((coef, null_state)) => model.generate_step_cfg(
                    &mut tts_state,
                    null_state,
                    *coef,
                    &prev_latent,
                    &mut rng,
                )?,
                None => model.generate_step(&mut tts_state, &prev_latent, &mut rng)?,
            };
            backbone_step_timings_ms.push(step_start.elapsed().as_secs_f64() * 1000.0);
            latent_tx.send(next_latent.clone())?;

            if eos.should_stop(is_eos) {
                tracing::info!(?step, "reached eos");
                break;
            }

            prev_latent = next_latent;

            if (step + 1) % 25 == 0 {
                tracing::info!(?step, ?max_frames, "generation progress");
            }
        }
        std::mem::drop(latent_tx); // Close channel to signal generation thread to finish
        is_done.store(true, std::sync::atomic::Ordering::SeqCst);
        let audio = jh.join().map_err(|_| anyhow::anyhow!("cannot join thread"))?;
        all_audios.push(audio);
    }
    let all_audios = all_audios.iter().collect::<Vec<&Tensor<f32, Q::B>>>();
    let audio = Tensor::cat(&all_audios, 2)?;
    let pcm = audio.to_vec()?;
    let duration = pcm.len() as f64 / cfg.mimi.sample_rate as f64;

    let elapsed = start.elapsed().as_secs_f64();
    let rtf = duration / elapsed;
    tracing::info!("generated {duration:.2}s in {elapsed:.2}s (RTF={rtf:.3})");
    let nsteps = backbone_step_timings_ms.len();
    tracing::info!(
        ?nsteps,
        "average backbone step time: {:.2}ms",
        backbone_step_timings_ms.iter().sum::<f64>() / nsteps as f64
    );

    // Write WAV
    let output_file = std::fs::File::create(&args.output)?;
    let mut writer = std::io::BufWriter::new(output_file);
    ptts::wav::write_pcm_as_wav(&mut writer, &pcm, cfg.mimi.sample_rate as u32, 1)?;
    tracing::info!("saving output to {}", args.output.display());
    Ok(())
}
