//! Export a checkpoint as Core ML models for the iOS and macOS package (`ios/PhononTTS`).
//!
//! Writes one directory: the flow LM as two ML Programs (one step, and the batched text
//! prefill), the Mimi decoder, the few tensors the driver applies on the host, the tokenizer,
//! the voices, and a `bundle.json` describing it all with every file's size and SHA-256.
//!
//! ```bash
//! cargo run --release -p ptts --example export_coreml -- out/phonon-coreml
//! cargo run --release -p ptts --example export_coreml -- --dir path/to/checkpoint out/models
//! ```
//!
//! The graphs are built for the Neural Engine: fp16, fully static shapes, and a KV cache the
//! host keeps. `--max-tokens` is the longest sentence the prefill graph takes; longer text is
//! split into sentences at run time. Mimi stays f32 and runs on the CPU.

#[path = "model_helpers.rs"]
mod model_helpers;

use anyhow::{Context, Result};
use clap::Parser;
use model_helpers::{Checkpoint, Source};
use ptts_coreml::Weights;
use ptts_coreml::package::write_mlpackage_with_weights;
use ptts_coreml::phonon::{flow_lm as fl, mimi};
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Export a checkpoint as Core ML models for the PhononTTS Swift package")]
struct Args {
    /// Output directory.
    out: PathBuf,
    /// Hugging Face repo to download the checkpoint from.
    #[arg(long, default_value = model_helpers::REPO_ID)]
    repo: String,
    /// A local checkpoint directory instead of the Hub.
    #[arg(long)]
    dir: Option<PathBuf>,
    /// Weights file inside the repo or directory, when it has several.
    #[arg(long)]
    weights: Option<String>,
    /// A directory of voice `.safetensors` files, instead of the checkpoint's own.
    #[arg(long)]
    voices: Option<PathBuf>,
    /// A `tokenizer.json`, for a checkpoint that ships none.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Longest sentence, in tokens, the prefill graph takes. It also sizes the KV cache, and
    /// with it the cost of every step, so it is worth keeping near what ptts chunks text into.
    #[arg(long, default_value_t = 48)]
    max_tokens: usize,
}

fn write(out: &Path, name: &str, built: fl::Built) -> Result<()> {
    let (model, blob) = built;
    write_mlpackage_with_weights(&out.join(format!("{name}.mlpackage")), &model, blob)
        .with_context(|| format!("writing {name}"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt().with_env_filter(model_helpers::LOG_DIRECTIVES).init();
    let source = match args.dir.as_deref() {
        Some(dir) => Source::Dir(dir),
        None => Source::Hub(&args.repo),
    };
    let ck = Checkpoint::locate(source, args.weights.as_deref())?;
    let cfg = &ck.config;
    let f = &cfg.flow_lm;
    let dims = fl::Dims {
        d: f.d_model,
        heads: f.num_heads,
        layers: f.num_layers,
        ff: f.dim_feedforward,
        ldim: f.ldim,
        flow_d: f.flow_dim,
        flow_blocks: f.flow_depth,
    };
    anyhow::ensure!(
        cfg.lsd_decode_steps == 1,
        "the Core ML graph takes one flow step; this checkpoint wants {}",
        cfg.lsd_decode_steps
    );
    let m = &cfg.mimi;
    anyhow::ensure!(
        (m.transformer_d_model, m.transformer_num_heads) == (mimi::DIM, mimi::HEADS),
        "the Core ML Mimi graph is built for a {}-wide, {}-head decoder transformer",
        mimi::DIM,
        mimi::HEADS
    );
    let window = m.transformer_context;

    // Tensor names as ptts reads them, whichever naming the checkpoint uses.
    let wt = if ck.weights.extension().is_some_and(|e| e == "gguf") {
        Weights::open_gguf(&ck.weights)
    } else {
        Weights::open(&ck.weights)
    }
    .map_err(anyhow::Error::msg)?
    .renamed(ptts::loader::remap_key);

    // The supplied voices may hold speaker-Mimi latents rather than ready-to-use embeddings.
    // Project them with the same checkpoint weight the Rust runtime uses when adding a voice.
    let speaker_proj = match wt.get(ptts::loader::SPEAKER_PROJ_WEIGHT) {
        Ok((shape, data)) => {
            let expected = [dims.d, cfg.speaker_mimi_cfg().dimension];
            anyhow::ensure!(
                shape == expected,
                "speaker projection has shape {shape:?}, expected {expected:?}"
            );
            let weight =
                xn::Tensor::from_vec(data.to_vec(), (expected[0], expected[1]), &xn::CpuDevice)?;
            Some(xn::nn::Linear::new(weight))
        }
        Err(_) => None,
    };
    let model_ext = cfg.model_ext();

    std::fs::create_dir_all(args.out.join("voices"))?;
    let voices: Vec<(String, PathBuf)> = match args.voices.as_deref() {
        Some(dir) => {
            let mut v: Vec<(String, PathBuf)> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .map(|p| (p.file_stem().unwrap().to_string_lossy().into(), p))
                .collect();
            v.sort();
            v
        }
        None => ck.voices.clone(),
    };
    anyhow::ensure!(!voices.is_empty(), "no voices: pass --voices <dir>");
    // Voices go in as the embedding the flow LM is prompted with, `emb` [1, T, D].
    let mut vlen = 0;
    for (name, path) in &voices {
        let emb = ptts::loader::load_voice_emb(
            path,
            model_ext.as_deref(),
            speaker_proj.as_ref(),
            &xn::CpuDevice,
        )
        .with_context(|| format!("voice {name}"))?;
        let shape = emb.dims().to_vec();
        anyhow::ensure!(
            shape[2] == dims.d,
            "voice {name} is {shape:?}, the model is {}-wide",
            dims.d
        );
        vlen = vlen.max(shape[1]);
        let bytes: Vec<u8> = emb.to_vec()?.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, shape, &bytes)?;
        safetensors::serialize_to_file(
            [("emb", view)],
            None,
            &args.out.join(format!("voices/{name}.safetensors")),
        )?;
    }

    // `ctx` is baked into the graphs: voice, one chunk of text and its frames must fit.
    let max_frames = ptts::plan::frame_budget(args.max_tokens, m.frame_rate);
    let ctx = vlen + args.max_tokens + max_frames + 16;
    tracing::info!(?dims, vlen, max_frames, ctx, "building graphs");
    let bad = |e: String| anyhow::anyhow!(e);
    write(&args.out, &fl::package_name(ctx, 1), fl::build(&wt, &dims, ctx, 1).map_err(bad)?)?;
    let prefill = fl::build(&wt, &dims, ctx, args.max_tokens).map_err(bad)?;
    write(&args.out, &fl::package_name(ctx, args.max_tokens), prefill)?;
    write(
        &args.out,
        &mimi::package_name(window),
        mimi::build(&wt, mimi::cache_len(window)).map_err(bad)?,
    )?;

    // The host-side tensors, f32. `num_speakers` is the conditioning some checkpoints add to
    // every frame (ptts `FlowLM::num_speakers`, one speaker), and zeros for those without it.
    let get = |n: &str| wt.get(n).map_err(anyhow::Error::msg);
    let num_speakers: Vec<f32> = match (
        wt.get("flow_lm.condition_provider.conditioners.num_speakers.embed.weight"),
        wt.get("flow_lm.condition_provider.conditioners.num_speakers.output_proj.weight"),
    ) {
        (Ok((es, e)), Ok((_, p))) => {
            let lut = es[1];
            (0..dims.d).map(|o| (0..lut).map(|i| p[o * lut + i] * e[lut + i]).sum()).collect()
        }
        _ => vec![0f32; dims.d],
    };
    let mut host: Vec<(&str, Vec<usize>, Vec<u8>)> = Vec::new();
    for n in ["flow_lm.conditioner.embed.weight", "flow_lm.input_linear.weight", "flow_lm.bos_emb"]
    {
        let (shape, data) = get(n)?;
        host.push((n, shape.to_vec(), data.iter().flat_map(|v| v.to_le_bytes()).collect()));
    }
    host.push((
        "flow_lm.num_speakers",
        vec![dims.d],
        num_speakers.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ));
    let views: HashMap<&str, TensorView> = host
        .iter()
        .map(|(n, s, b)| Ok((*n, TensorView::new(Dtype::F32, s.clone(), b)?)))
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(&views, None, &args.out.join("host.safetensors"))?;

    let tokenizer = args
        .tokenizer
        .clone()
        .or(ck.tokenizer.clone())
        .context("no tokenizer.json: pass --tokenizer")?;
    anyhow::ensure!(
        tokenizer.extension().is_some_and(|e| e == "json"),
        "{} is not a tokenizer.json; convert it with scripts/convert-tokenizer.py",
        tokenizer.display()
    );
    std::fs::copy(&tokenizer, args.out.join("tokenizer.json"))?;

    // Every file with its size and SHA-256, so an app can fetch the bundle from any static host
    // and verify it. `built` changes on every export, so an installed copy can tell it is old.
    let mut files = Vec::new();
    list_files(&args.out, &args.out, &mut files)?;
    files.sort_by(|a: &serde_json::Value, b| a["path"].as_str().cmp(&b["path"].as_str()));
    let built = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
    let meta = serde_json::json!({
        "built": built,
        "ctx": ctx,
        "prefill_len": args.max_tokens,
        "max_frames": max_frames,
        "mimi_window": window,
        "eos_threshold": cfg.eos_threshold,
        "temperature": cfg.temp,
        "dims": {
            "d": dims.d, "heads": dims.heads, "layers": dims.layers, "ff": dims.ff,
            "ldim": dims.ldim, "flow_d": dims.flow_d, "flow_blocks": dims.flow_blocks,
        },
        "voices": voices.iter().map(|(n, _)| n).collect::<Vec<_>>(),
        "files": files,
    });
    std::fs::write(args.out.join("bundle.json"), serde_json::to_vec_pretty(&meta)?)?;
    let size: u64 = files.iter().filter_map(|f| f["size"].as_u64()).sum();
    println!("wrote {} ({:.0} MB, {} voices)", args.out.display(), size as f64 / 1e6, voices.len());
    Ok(())
}

fn list_files(root: &Path, dir: &Path, out: &mut Vec<serde_json::Value>) -> Result<()> {
    use sha2::Digest;
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            list_files(root, &p, out)?;
        } else if p.file_name().is_some_and(|n| n != "bundle.json") {
            let bytes = std::fs::read(&p)?;
            let rel: Vec<String> = p
                .strip_prefix(root)?
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into())
                .collect();
            out.push(serde_json::json!({
                "path": rel.join("/"),
                "size": bytes.len(),
                "sha256": format!("{:x}", sha2::Sha256::digest(&bytes)),
            }));
        }
    }
    Ok(())
}
