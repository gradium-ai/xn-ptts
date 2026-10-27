//! Export a checkpoint as Core ML models for the iOS and macOS package (`ios/PhononTTS`).
//!
//! Writes one directory: the flow LM as two ML Programs (one step, and the batched text
//! prefill), the Mimi decoder, the few tensors the driver applies on the host, the tokenizer,
//! the voices, and a `bundle.json` describing it all with every file's size and SHA-256.
//!
//! ```bash
//! cargo run --release -p ptts --example export_coreml --features hub -- out/models
//! cargo run --release -p ptts --example export_coreml -- --model path/to/checkpoint out/models
//! ```
//!
//! The graphs are built for the Neural Engine: fp16, fully static shapes, and a KV cache the
//! host keeps. `--max-tokens` is the longest sentence the prefill graph takes; longer text is
//! split into sentences at run time. Mimi stays f32 and runs on the CPU.

#[path = "../src/bin/ptts/model_helpers.rs"]
mod model_helpers;

use anyhow::{Context, Result};
use clap::Parser;
use ptts_coreml::Weights;
use ptts_coreml::mil::Gelu;
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
    #[arg(long, default_value = model_helpers::DEFAULT_MODEL, help = model_helpers::MODEL_HELP)]
    model: String,
    /// Weights file inside the repo or directory, when it has several.
    #[arg(long)]
    weights: Option<String>,
    /// Source weight format to select, e.g. f32 or q8. The exported graph sets its own precision.
    #[arg(long, default_value = "f32")]
    quant: String,
    /// A directory of voice `.safetensors` files, instead of the checkpoint's own.
    #[arg(long)]
    voices: Option<PathBuf>,
    /// Set a conditioner, e.g. duration_delta=0.5; repeatable. Fixed in the bundle: the app
    /// cannot change it.
    #[arg(long = "condition", value_name = "NAME=VALUE")]
    conditions: Vec<String>,
    /// A `tokenizer.json`, for a checkpoint that ships none.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Longest sentence, in tokens, the prefill graph takes. It also sizes the KV cache, and
    /// with it the cost of every step, so it is worth keeping near what ptts chunks text into.
    #[arg(long, default_value_t = 48)]
    max_tokens: usize,
    /// Sampling temperature the app speaks at. The default matches the other frontends.
    #[arg(long, default_value_t = 0.3)]
    temperature: f32,
}

fn write(out: &Path, name: &str, built: fl::Built) -> Result<()> {
    let (model, blob) = built;
    write_mlpackage_with_weights(&out.join(format!("{name}.mlpackage")), &model, blob)
        .with_context(|| format!("writing {name}"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt().with_env_filter(model_helpers::LOG_DIRECTIVES).init();
    let options = model_helpers::ResolveOptions {
        quant: args.quant.parse()?,
        weights: args.weights.as_deref(),
    };
    let ck = model_helpers::Checkpoint::locate(&args.model, options)?;
    let cfg = &ck.config;
    let f = &cfg.flow_lm;
    let dims = fl::Dims {
        d: f.d_model,
        heads: f.num_heads,
        attn: f.attn_dim.unwrap_or(f.d_model),
        kv_heads: f.num_kv_heads.unwrap_or(f.num_heads),
        layers: f.num_layers,
        ff: f.dim_feedforward,
        ldim: f.ldim,
        z: f.z_dim,
        flow_d: f.flow_dim,
        flow_blocks: f.flow_depth,
    };
    anyhow::ensure!(
        dims.heads > 0
            && dims.attn.is_multiple_of(dims.heads)
            && dims.kv_heads > 0
            && dims.heads.is_multiple_of(dims.kv_heads),
        "the config's attention heads do not divide its width or its key/value heads"
    );
    let flow = match f.flow_type {
        ptts::flow_lm::FlowType::Lsd => fl::FlowType::Lsd,
        ptts::flow_lm::FlowType::Drifting => fl::FlowType::Drifting,
        ptts::flow_lm::FlowType::Pointwise => fl::FlowType::Pointwise,
    };
    anyhow::ensure!(
        flow != fl::FlowType::Lsd || cfg.lsd_decode_steps == 1,
        "the Core ML graph takes one flow step; this checkpoint wants {}",
        cfg.lsd_decode_steps
    );
    let time_norm = match f.time_rms_norm {
        ptts::flow_lm::TimeRmsNorm::Rms => fl::TimeNorm::Rms,
        ptts::flow_lm::TimeRmsNorm::RmsVar => fl::TimeNorm::RmsVar,
    };
    // The FFN activation each transformer was trained with, as ptts picks it.
    let gelu = |g: Option<ptts::transformer::Gelu>| match g.unwrap_or_else(|| cfg.default_gelu()) {
        ptts::transformer::Gelu::Erf => Gelu::Erf,
        ptts::transformer::Gelu::Tanh => Gelu::Tanh,
    };
    let m = &cfg.mimi;
    let hop: usize = m.ratios.iter().product();
    let mimi_dims = mimi::Dims {
        ldim: m.quantizer_dimension,
        dim: m.dimension,
        d_model: m.transformer_d_model,
        heads: m.transformer_num_heads,
        layers: m.transformer_num_layers,
        ff: m.transformer_dim_feedforward,
        out_dim: m.decoder_dimension.unwrap_or(m.dimension),
        n_filters: m.n_filters,
        ratios: m.ratios.clone(),
        kernel_size: m.kernel_size,
        residual_kernel_size: m.residual_kernel_size,
        last_kernel_size: m.last_kernel_size,
        compress: m.compress,
        steps: (m.sample_rate as f64 / hop as f64 / m.frame_rate).round() as usize,
    };
    anyhow::ensure!(
        m.channels == 1
            && m.n_residual_layers == 1
            && mimi_dims.steps > 1
            && (m.quantizer_dimension, m.quantizer_output_dimension) == (f.ldim, m.dimension),
        "the Core ML Mimi graph takes a mono decoder with one residual block per stage that \
         upsamples the flow LM's latents"
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

    // The same weights through ptts's loader, for what the runtime computes from them rather
    // than what the graphs hold: voice embeddings and the conditioning.
    let vb = ptts::loader::load_weights::<xn::Unquantized<f32, xn::CpuDevice>>(
        &ck.weights,
        &xn::CpuDevice,
    )?;
    // Voices may hold speaker-Mimi latents rather than ready-to-use embeddings, projected with
    // the same checkpoint weight the Rust runtime uses when adding a voice.
    let speaker_proj = ptts::loader::load_speaker_proj(&vb, cfg)?;
    let embeddings_ext = cfg.embeddings_ext.as_deref();

    // What the flow LM adds to every frame's input, `D` wide: the checkpoint's conditioners
    // summed, at the values given and their defaults for the rest. Zeros without any.
    let mut given = HashMap::new();
    for condition in &args.conditions {
        let (name, value) = condition.split_once('=').context("--condition takes NAME=VALUE")?;
        given.insert(name.to_string(), value.to_string());
    }
    let conditions = |values: &HashMap<String, String>| -> Result<Vec<f32>> {
        Ok(match ptts::loader::load_conditions::<f32, _>(&vb, cfg, values)? {
            Some(sum) => sum.to_vec()?,
            None => vec![0f32; dims.d],
        })
    };

    // Each voice as the embedding the flow LM is prompted with, `emb` [1, T, D], and, for one
    // baked into the checkpoint, the `conditions` [D] it is spoken with in place of the default.
    let mut voices = Vec::new();
    if cfg.voices.is_empty() {
        let files = match args.voices.as_deref() {
            Some(dir) => ptts::loader::voices_in(dir),
            None => ck.voices.clone(),
        };
        for (name, path) in files {
            let emb = ptts::loader::load_voice_emb(
                &path,
                embeddings_ext,
                speaker_proj.as_ref(),
                &xn::CpuDevice,
            )
            .with_context(|| format!("voice {name}"))?;
            voices.push((name, emb, None));
        }
    } else {
        anyhow::ensure!(
            args.voices.is_none(),
            "this checkpoint has baked-in voices and supports no other voice"
        );
        let baked = ptts::loader::load_config_voices(&vb, cfg, speaker_proj.as_ref())?;
        for (voice, (name, emb)) in cfg.voices.iter().zip(baked) {
            // The voice's own values win, as in `SynthBuilder::build`.
            let mut values = given.clone();
            values.extend(voice.conditions.clone());
            voices.push((name, emb, Some(conditions(&values)?)));
        }
    }
    anyhow::ensure!(!voices.is_empty(), "no voices: pass --voices <dir>");
    std::fs::create_dir_all(args.out.join("voices"))?;
    let mut vlen = 0;
    for (name, emb, voice_conditions) in &voices {
        let shape = emb.dims().to_vec();
        anyhow::ensure!(
            shape[2] == dims.d,
            "voice {name} is {shape:?}, the model is {}-wide",
            dims.d
        );
        vlen = vlen.max(shape[1]);
        let bytes: Vec<u8> = emb.to_vec()?.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut views = vec![("emb", TensorView::new(Dtype::F32, shape, &bytes)?)];
        let cond_bytes: Vec<u8> =
            voice_conditions.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        if voice_conditions.is_some() {
            views.push(("conditions", TensorView::new(Dtype::F32, vec![dims.d], &cond_bytes)?));
        }
        safetensors::serialize_to_file(
            views,
            None,
            &args.out.join(format!("voices/{name}.safetensors")),
        )?;
    }

    // `ctx` is baked into the graphs: voice, one chunk of text and its frames must fit.
    let max_frames = ptts::plan::frame_budget(args.max_tokens, m.frame_rate);
    let ctx = vlen + args.max_tokens + max_frames + 16;
    tracing::info!(?dims, vlen, max_frames, ctx, "building graphs");
    let bad = |e: String| anyhow::anyhow!(e);
    write(
        &args.out,
        &fl::package_name(ctx, 1),
        fl::build(&wt, &dims, ctx, 1, time_norm, gelu(f.gelu), flow).map_err(bad)?,
    )?;
    let prefill =
        fl::build(&wt, &dims, ctx, args.max_tokens, time_norm, gelu(f.gelu), flow).map_err(bad)?;
    write(&args.out, &fl::package_name(ctx, args.max_tokens), prefill)?;
    write(
        &args.out,
        &mimi_dims.package_name(window),
        mimi::build(&wt, &mimi_dims, mimi_dims.cache_len(window), gelu(m.gelu)).map_err(bad)?,
    )?;

    // The host-side tensors, f32, and the conditioning a voice without its own is spoken with.
    // A baked-in voice's values fill in for any not given, as in `SynthBuilder::build`.
    let get = |n: &str| wt.get(n).map_err(anyhow::Error::msg);
    let mut defaults = given.clone();
    for (name, value) in cfg.voices.first().map(|v| &v.conditions).into_iter().flatten() {
        defaults.entry(name.clone()).or_insert_with(|| value.clone());
    }
    let mut host: Vec<(&str, Vec<usize>, Vec<u8>)> = Vec::new();
    // The text table at the model's width: a low-rank one through its `output_proj`, as ptts
    // projects it at load.
    let (shape, table) = get("flow_lm.conditioner.embed.weight")?;
    let (shape, table) = match wt.get("flow_lm.conditioner.output_proj.weight") {
        Ok((proj_shape, proj)) => {
            let (rows, rank, d) = (shape[0], shape[1], proj_shape[0]);
            let mut out = vec![0f32; rows * d];
            for (row, o) in table.chunks(rank).zip(out.chunks_mut(d)) {
                for (v, p) in o.iter_mut().zip(proj.chunks(rank)) {
                    *v = row.iter().zip(p).map(|(a, b)| a * b).sum();
                }
            }
            (vec![rows, d], out)
        }
        Err(_) => (shape.to_vec(), table.to_vec()),
    };
    host.push((
        "flow_lm.conditioner.embed.weight",
        shape,
        table.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ));
    for n in ["flow_lm.input_linear.weight", "flow_lm.bos_emb"] {
        let (shape, data) = get(n)?;
        host.push((n, shape.to_vec(), data.iter().flat_map(|v| v.to_le_bytes()).collect()));
    }
    host.push((
        "flow_lm.conditions",
        vec![dims.d],
        conditions(&defaults)?.iter().flat_map(|v| v.to_le_bytes()).collect(),
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
        "mimi": mimi_dims.to_json(),
        "mimi_window": window,
        "eos_threshold": cfg.eos_threshold,
        "temperature": args.temperature,
        "dims": {
            "d": dims.d, "heads": dims.heads, "attn": dims.attn, "kv_heads": dims.kv_heads,
            "layers": dims.layers, "ff": dims.ff, "ldim": dims.ldim, "z": dims.z,
            "flow_d": dims.flow_d, "flow_blocks": dims.flow_blocks,
        },
        "voices": voices.iter().map(|(n, _, _)| n).collect::<Vec<_>>(),
        "languages": cfg.languages,
        "conditions": given,
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
