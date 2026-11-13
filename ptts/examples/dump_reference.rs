//! Dumps reference tensors for one generation, so another implementation of the model can be
//! checked against this one step by step.
//!
//! Runs a checkpoint unquantized in f32 on the CPU through the library's streaming API
//! (`TTSModel::prompt_audio`, `prompt_text`, `generate_step_parts`, `decode_latent`), one frame
//! at a time, and writes into `--out`:
//!
//! - `manifest.json`: text, token ids, prefill layout, sampling and EOS settings, file layouts.
//! - `prefix_embeddings.bin`, `frame_bias.bin`, `step_inputs.bin`, `noise.bin`, `latents.bin`,
//!   `latents_denorm.bin`, `eos_logits.bin`, `audio.bin`: raw little-endian f32, row-major.
//! - `audio.wav`.
//!
//! The noise comes from `--seed` (as every frontend draws it) or, with `--noise`, from a file
//! of f32 `[frames, ldim]` values used as they are, so a second implementation can feed both
//! sides the same noise.
//!
//! ```text
//! cargo run --release --example dump_reference --features hf -- \
//!   --dir path/to/checkpoint --voice path/to/voice.safetensors --lang en \
//!   --seed 0 --out ref/hello "Hello world."
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use ptts::flow_lm::{NormalRng, ReplayRng, Rng, StepInput};
use ptts::plan::{EosPolicy, frame_budget};
use ptts::preprocess::{Normalize, Rules};
use ptts::tok::Tok;
use ptts::tts_model::{MAX_TOKENS_PER_CHUNK, TTSConfig, TTSModel, prepare_text_prompt};
use xn::{CpuDevice, Tensor, Unquantized};

type Q = Unquantized<f32, CpuDevice>;

#[derive(Parser, Debug)]
#[command(name = "dump_reference")]
#[command(about = "Dump reference tensors of one f32 CPU generation")]
struct Args {
    /// The text to speak. It has to fit in one chunk (at most 50 tokens, whole sentences).
    text: String,

    /// Checkpoint directory holding `config.json`, the weights and `tokenizer.json`.
    #[arg(long)]
    dir: PathBuf,

    /// Weights file inside `--dir`. Must be safetensors: the point is an unquantized run.
    #[arg(long, default_value = "model.safetensors")]
    weights: String,

    /// Voice safetensors (`speaker_wavs` latents or a precomputed `emb`). Defaults to the
    /// checkpoint's `default-voice.safetensors`.
    #[arg(long)]
    voice: Option<PathBuf>,

    /// Name recorded for the voice in the manifest. Defaults to the file stem, or `default`.
    #[arg(long)]
    voice_name: Option<String>,

    /// Text normalization language: `en`, `fr`, `de`, `es`, `pt` or `none`.
    #[arg(long)]
    lang: String,

    /// Which word rewrites run on the normalized text, as on the other examples.
    #[arg(long, default_value = "default")]
    rewrites: String,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Sampling temperature: the noise is N(0, temperature). 0.3 is what every frontend
    /// defaults to; the config's `temp` is not read by ptts.
    #[arg(long, default_value_t = 0.3)]
    temperature: f32,

    /// f32 little-endian `[frames, ldim]` file of noise to use instead of drawing it from the
    /// seed. Used as is (already at the intended scale): the temperature does not apply.
    /// Generation stops when it runs out.
    #[arg(long)]
    noise: Option<PathBuf>,

    /// Generate at most this many frames, on top of the usual budget and EOS rule.
    #[arg(long)]
    max_frames: Option<usize>,

    #[arg(
        long = "condition",
        value_name = "NAME=VALUE",
        help = "Set a conditioner, e.g. num_speakers=1; repeatable"
    )]
    conditions: Vec<String>,

    /// Number of CPU threads for tensor ops.
    #[arg(long)]
    threads: Option<usize>,

    /// Also run the same generation through `Synth` and check the audio matches.
    #[arg(long, default_value_t = false)]
    check_synth: bool,

    /// Output directory, created if missing.
    #[arg(long)]
    out: PathBuf,
}

/// Records every value the sampler draws, so the dump holds the noise that was actually used.
struct Recording {
    inner: Box<dyn Rng + Send>,
    values: Vec<f32>,
}

impl Rng for Recording {
    fn sample(&mut self) -> f32 {
        let v = self.inner.sample();
        self.values.push(v);
        v
    }
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).with_context(|| format!("cannot write {}", path.display()))
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    anyhow::ensure!(bytes.len() % 4 == 0, "{} is not a whole number of f32", path.display());
    let (words, _) = bytes.as_chunks::<4>();
    Ok(words.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

fn flat(t: &Tensor<f32, CpuDevice>) -> Result<Vec<f32>> {
    Ok(t.to_vec()?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let normalize = args.lang.parse::<Normalize>()?.with_rules(args.rewrites.parse::<Rules>()?);
    if let Some(threads) = args.threads {
        xn::set_num_threads(threads);
    }
    let dev = CpuDevice;

    let config_path = args.dir.join("config.json");
    let cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
        .with_context(|| format!("failed to read config {}", config_path.display()))?;
    let raw_config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    let weights = args.dir.join(&args.weights);
    anyhow::ensure!(
        weights.extension().and_then(|e| e.to_str()) == Some("safetensors"),
        "--weights must be safetensors for an unquantized reference"
    );
    let tokenizer_path = args.dir.join("tokenizer.json");
    let voice_path =
        args.voice.clone().unwrap_or_else(|| args.dir.join(ptts::loader::DEFAULT_VOICE_FILE));
    let voice_name = args.voice_name.clone().unwrap_or_else(|| {
        if args.voice.is_none() {
            "default".to_string()
        } else {
            voice_path.file_stem().and_then(|s| s.to_str()).unwrap_or("voice").to_string()
        }
    });

    let mut conditions = HashMap::new();
    for condition in &args.conditions {
        let (name, value) = condition.split_once('=').context("--condition takes NAME=VALUE")?;
        conditions.insert(name.to_string(), value.to_string());
    }
    anyhow::ensure!(cfg.voices.is_empty(), "checkpoints with baked-in voices are not supported");

    let vb = ptts::loader::load_weights::<Q>(&weights, &dev)?;
    let model: TTSModel<Q> =
        TTSModel::load(&vb, Box::new(Tok::open(&tokenizer_path)?), &cfg, &conditions)?;
    // Not public on `FlowLM`, and read from the same tensor `FlowLM::load` reads.
    let bos_emb: Tensor<f32, CpuDevice> =
        vb.pp("flow_lm").tensor("bos_emb", (cfg.flow_lm.ldim,))?;
    let ldim = cfg.flow_lm.ldim;
    let d_model = cfg.flow_lm.d_model;

    // Text: the same chunking every frontend does, and one chunk only.
    let tokenizer = model.flow_lm.conditioner.tokenizer.as_deref().context("no tokenizer")?;
    let normalized = normalize.apply(&args.text).into_owned();
    let chunks = ptts::plan::chunks(
        tokenizer,
        &args.text,
        normalize,
        MAX_TOKENS_PER_CHUNK,
        cfg.mimi.frame_rate,
    )?;
    anyhow::ensure!(
        chunks.len() == 1,
        "the text splits into {} chunks; use a text of at most {MAX_TOKENS_PER_CHUNK} tokens",
        chunks.len()
    );
    let chunk = &chunks[0];
    let (prepared, frames_after_eos) = prepare_text_prompt(&chunk.text);
    let tokens = chunk.tokens.clone();
    anyhow::ensure!(tokenizer.encode(&prepared)? == tokens, "prepared text does not re-encode");
    anyhow::ensure!(frames_after_eos == chunk.frames_after_eos);

    // Conditioning, exactly as the library computes it.
    let voice_emb = ptts::loader::load_voice_emb(
        &voice_path,
        cfg.embeddings_ext.as_deref(),
        model.speaker_proj(),
        &dev,
    )?;
    let voice_len = voice_emb.dim(1usize)?;
    let text_emb = model.flow_lm.conditioner.embed_tokens(&tokens)?;
    let prefix = Tensor::cat(&[&voice_emb, &text_emb], 1)?;
    let frame_bias = model.flow_lm.condition_providers.clone();

    // Noise source.
    let (inner, noise_source): (Box<dyn Rng + Send>, serde_json::Value) = match &args.noise {
        Some(path) => {
            let values = read_f32(path)?;
            anyhow::ensure!(
                !values.is_empty() && values.len() % ldim == 0,
                "{} holds {} values, not a whole number of {ldim}-wide frames",
                path.display(),
                values.len()
            );
            let src = serde_json::json!({
                "kind": "file",
                "path": path.display().to_string(),
                "frames": values.len() / ldim,
            });
            (Box::new(ReplayRng::new(values)?), src)
        }
        None => {
            let src = serde_json::json!({
                "kind": "seed",
                "seed": args.seed,
                "generator": "rand::rngs::StdRng::seed_from_u64(seed) sampled through \
                              rand_distr::Normal(0, sqrt(temperature)), 32 draws per frame \
                              in latent-channel order (ptts::flow_lm::NormalRng)",
            });
            (Box::new(NormalRng::new(args.temperature, args.seed)?), src)
        }
    };
    let noise_frames = args.noise.as_ref().map(|_| noise_source["frames"].as_u64().unwrap());
    let mut rng = Recording { inner, values: vec![] };

    let budget = frame_budget(tokens.len(), cfg.mimi.frame_rate);
    let mut max_frames = budget;
    if let Some(m) = args.max_frames {
        max_frames = max_frames.min(m);
    }
    if let Some(n) = noise_frames {
        max_frames = max_frames.min(n as usize);
    }

    // Prefill: the voice, then the text, as `Synth` does.
    let seq_budget = voice_len + tokens.len() + max_frames;
    let mut state = model.init_flow_lm_state(1, seq_budget)?;
    model.prompt_audio(&mut state, &voice_emb.to()?)?;
    model.prompt_text(&mut state, &tokens)?;

    let mut mimi_state = model.init_mimi_state(1)?;
    let mut eos = EosPolicy::new(frames_after_eos);
    let mut prev: Option<Tensor<f32, CpuDevice>> = None;
    let mut step_inputs = vec![];
    let mut latents = vec![];
    let mut latents_denorm = vec![];
    let mut eos_logits = vec![];
    let mut audio = vec![];
    let mut first_eos = None;
    let mut frames = 0usize;
    let mut stopped_by_eos = false;

    for step in 0..max_frames {
        // The transformer input of this step, recomputed from the same public pieces the
        // backbone uses: input_linear of the step's latent (bos_emb on the first) plus the
        // summed conditioners.
        let latent_in = match &prev {
            None => bos_emb.reshape((1, 1, ldim))?,
            Some(t) => t.clone(),
        };
        let x = model.flow_lm.input_linear.forward(&latent_in)?;
        let x = match &frame_bias {
            Some(b) => x.broadcast_add(b)?,
            None => x,
        };
        step_inputs.extend(flat(&x)?);

        let input = match &prev {
            None => StepInput::Bos { batch: 1 },
            Some(t) => StepInput::Latent(t),
        };
        let (latent, eos_logit) = model.generate_step_parts(&mut state, input, &mut rng)?;
        let eos_logit = eos_logit.to_vec()?;
        let is_eos = model.eos_from_logit(&eos_logit);
        if is_eos && first_eos.is_none() {
            first_eos = Some(step);
        }
        eos_logits.push(eos_logit[0]);
        latents.extend(flat(&latent)?);
        let denorm =
            latent.broadcast_mul(&model.flow_lm.emb_std)?.broadcast_add(&model.flow_lm.emb_mean)?;
        latents_denorm.extend(flat(&denorm)?);
        audio.extend(flat(&model.decode_latent(&latent, &mut mimi_state)?)?);
        frames += 1;

        if eos.should_stop(is_eos) {
            stopped_by_eos = true;
            break;
        }
        prev = Some(latent);
    }
    anyhow::ensure!(rng.values.len() == frames * ldim, "unexpected number of noise draws");

    std::fs::create_dir_all(&args.out)?;
    let out = |name: &str| args.out.join(name);
    write_f32(&out("prefix_embeddings.bin"), &flat(&prefix)?)?;
    let bias_values = match &frame_bias {
        Some(b) => flat(b)?,
        None => vec![0.0; d_model],
    };
    write_f32(&out("frame_bias.bin"), &bias_values)?;
    write_f32(&out("step_inputs.bin"), &step_inputs)?;
    write_f32(&out("noise.bin"), &rng.values)?;
    write_f32(&out("latents.bin"), &latents)?;
    write_f32(&out("latents_denorm.bin"), &latents_denorm)?;
    write_f32(&out("eos_logits.bin"), &eos_logits)?;
    write_f32(&out("audio.bin"), &audio)?;
    let sample_rate = model.sample_rate() as u32;
    ptts::wav::write_wav_file(out("audio.wav"), &audio, sample_rate)?;

    // Optional cross-check: the same generation through the threaded `Synth` path.
    let synth_check = if args.check_synth && args.noise.is_none() {
        use ptts::synth::{SpeechOptions, SynthBuilder};
        let mut builder = SynthBuilder::new(cfg.clone(), &weights, normalize)
            .tokenizer_file(&tokenizer_path)
            .device(ptts::synth::DeviceKind::Cpu)
            .temperature(args.temperature)
            .seed(args.seed)
            .add_voice(voice_name.clone(), voice_path.clone())
            .voice(voice_name.clone());
        for (k, v) in conditions.iter() {
            builder = builder.condition(k.clone(), v.clone());
        }
        let tts = builder.build()?;
        let pcm = tts.say_with(&args.text, &SpeechOptions::default())?;
        let max_diff = if pcm.len() == audio.len() {
            pcm.iter().zip(audio.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
        } else {
            f32::NAN
        };
        println!("synth check: {} vs {} samples, max abs diff {max_diff}", pcm.len(), audio.len());
        serde_json::json!({ "synth_samples": pcm.len(), "max_abs_diff": max_diff })
    } else {
        serde_json::Value::Null
    };

    let n_prefix = voice_len + tokens.len();
    let condition_values: serde_json::Map<String, serde_json::Value> = cfg
        .conditioners
        .iter()
        .map(|c| {
            let name = c.name().to_string();
            let given = conditions.get(&name).cloned();
            // Mirrors `conditioners::default_condition`.
            let default = match name.as_str() {
                "num_speakers" => Some("1"),
                "duration_delta" | "padding_bonus" => Some("0.0"),
                _ => None,
            };
            let value = given.clone().or(default.map(str::to_string));
            let index = match c {
                ptts::tts_model::ConditionerConfig::Lut { lut, .. } => lut
                    .possible_values
                    .as_ref()
                    .and_then(|p| p.iter().position(|v| Some(v) == value.as_ref())),
                _ => None,
            };
            (
                name,
                serde_json::json!({
                    "value": value,
                    "given_on_command_line": given.is_some(),
                    "lut_row": index,
                }),
            )
        })
        .collect();

    let manifest = serde_json::json!({
        "generator": "xn-ptts ptts/examples/dump_reference.rs",
        "checkpoint": {
            "dir": args.dir.display().to_string(),
            "weights": weights.display().to_string(),
            "tokenizer": tokenizer_path.display().to_string(),
            "model_id": raw_config.get("model_id"),
            "compute": "f32 on CPU (bf16 weights widened to f32 at load), unquantized",
            "threads": xn::get_num_threads(),
        },
        "text": {
            "input": args.text,
            "lang": args.lang,
            "rewrites": args.rewrites,
            "normalized": normalized,
            "chunk_text": chunk.text,
            "fed_to_tokenizer": prepared,
            "tokenizer_call": "tokenizers::Tokenizer::encode(text, add_special_tokens=false)",
            "token_ids": tokens,
            "n_tokens": tokens.len(),
        },
        "voice": {
            "name": voice_name,
            "path": voice_path.display().to_string(),
            "frames": voice_len,
            "processing": "speaker_wavs [1, 512, T] -> transpose -> [1, T, 512] -> \
                           speaker_proj (flow_lm.condition_provider.conditioners.\
                           speaker_wavs.output_proj.weight, [768, 512], no bias) -> [1, T, 768]",
        },
        "conditions": condition_values,
        "prefill": {
            "n_prefix": n_prefix,
            "segments": [
                {
                    "order": 0,
                    "name": "speaker_wavs",
                    "rows": [0, voice_len],
                    "length": voice_len,
                    "nature": "projected speaker latents; no learnt padding, no bias, no \
                               frame_bias added",
                    "transformer_call": "TTSModel::prompt_audio: one forward over these rows, \
                                         RoPE positions 0..T",
                },
                {
                    "order": 1,
                    "name": "transcript_in_segment",
                    "rows": [voice_len, n_prefix],
                    "length": tokens.len(),
                    "nature": "rows of transcript_in_segment.embed.weight [4001, 768] at the \
                               token ids; no output_proj, no learnt padding, no frame_bias added",
                    "transformer_call": "TTSModel::prompt_text: a second forward over these \
                                         rows, continuing the KV cache, RoPE positions T..T+n",
                },
            ],
            "not_in_prefill": [
                "BOS: it is the first generated step's input (input_linear(bos_emb)), not a \
                 prefix row",
                "learnt_padding of every conditioner (only CFG's null text branch uses \
                 transcript_in_segment.learnt_padding; CFG is off)",
                "num_speakers: added to generated steps only, see frame_bias",
            ],
            "equivalence": "the two forwards equal one causal forward over all n_prefix rows",
        },
        "frame_bias": {
            "present": frame_bias.is_some(),
            "what": "sum over cfg.conditioners: num_speakers -> lut row (index of the value in \
                     possible_values) of embed.weight [32, 16] -> output_proj [768, 16], no bias",
            "where": "added to input_linear(latent) of every generated step, before the \
                      transformer (FlowLM::backbone); never to prefix rows",
        },
        "sampling": {
            "temperature": if args.noise.is_some() { serde_json::Value::Null } else {
                serde_json::json!(args.temperature) },
            "noise_std": if args.noise.is_some() { serde_json::Value::Null } else {
                serde_json::json!(args.temperature.sqrt()) },
            "noise_source": noise_source,
            "noise_is": "x_0 of the flow, already scaled (N(0, temperature), std = \
                         sqrt(temperature)); not multiplied by anything afterwards",
            "config_temp_ignored": raw_config.get("temp"),
            "lsd_decode_steps": cfg.lsd_decode_steps,
            "lsd_formula": "latent = noise + flow_net(cond=t_out, s=0, t=1, x=noise) \
                            (one Euler step of the self-distilled flow map)",
            "cfg": "off",
        },
        "eos": {
            "threshold": cfg.eos_threshold,
            "rule": "is_eos = eos_logit > threshold (strict), eos_logit = out_eos(out_norm(h)) \
                     raw, no sigmoid",
            "min_frames_before_eos": 0,
            "frames_after_eos": frames_after_eos,
            "frames_after_eos_rule": "3 if the prepared text has <= 4 words, else 1",
            "tail": "the EOS frame is emitted, then frames_after_eos more frames, then stop; \
                     later eos flags are ignored",
            "max_frames_formula": "ceil((n_tokens / 3 + 2) * frame_rate)",
            "max_frames_budget": budget,
            "max_frames_used": max_frames,
            "first_eos_frame": first_eos,
            "stopped_by_eos": stopped_by_eos,
        },
        "frames": frames,
        "sample_rate": sample_rate,
        "samples_per_frame": (sample_rate as f64 / cfg.mimi.frame_rate).round() as usize,
        "samples": audio.len(),
        "dims": {"d_model": d_model, "ldim": ldim},
        "synth_check": synth_check,
        "files": {
            "prefix_embeddings.bin": {"dtype": "float32 little-endian", "shape": [n_prefix, d_model],
                "what": "the sequence fed to the flow transformer in the prefill: voice rows then text rows"},
            "frame_bias.bin": {"dtype": "float32 little-endian", "shape": [d_model],
                "what": "summed conditioners added to each generated step's input (zeros if none)"},
            "step_inputs.bin": {"dtype": "float32 little-endian", "shape": [frames, d_model],
                "what": "transformer input row at each generated step: input_linear(prev latent, \
                         normalized; bos_emb at step 0) + frame_bias"},
            "noise.bin": {"dtype": "float32 little-endian", "shape": [frames, ldim],
                "what": "x_0 used at each step, after temperature scaling"},
            "latents.bin": {"dtype": "float32 little-endian", "shape": [frames, ldim],
                "what": "flow output, normalized (what is fed back as the next step's latent)"},
            "latents_denorm.bin": {"dtype": "float32 little-endian", "shape": [frames, ldim],
                "what": "latents * emb_std + emb_mean, what Mimi's quantizer.output_proj receives"},
            "eos_logits.bin": {"dtype": "float32 little-endian", "shape": [frames],
                "what": "raw eos logit per step"},
            "audio.bin": {"dtype": "float32 little-endian", "shape": [audio.len()],
                "what": "Mimi output, one latent decoded per call, concatenated"},
            "audio.wav": {"dtype": "16-bit PCM WAV", "sample_rate": sample_rate},
        },
    });
    std::fs::write(out("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    println!(
        "{}: {} tokens, {} prefix rows, {} frames (first eos {:?}), {} samples",
        args.out.display(),
        tokens.len(),
        n_prefix,
        frames,
        first_eos,
        audio.len()
    );
    Ok(())
}
