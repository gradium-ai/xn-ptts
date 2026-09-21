mod encoder;
mod handler;
mod model;
mod protocol;
mod utils;
mod wav;

use anyhow::Result;
use axum::Router;
use axum::routing::any;
use clap::Parser;
use ptts::synth::{DeviceKind, Quant};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "ptts-ws-server")]
#[command(about = "WebSocket server for Pocket TTS")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8080")]
    addr: String,

    #[arg(long)]
    config: Option<std::path::PathBuf>,

    /// Optional directory of additional voice safetensors to load. Each
    /// `*.safetensors` file is loaded as a voice keyed by its file stem; load
    /// errors are logged and skipped rather than fatal.
    #[arg(long)]
    voice_dir: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = 0.4)]
    temperature: f32,

    #[arg(long, default_value_t = 4242424242424242)]
    seed: u64,

    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// Use the CUDA backend (requires building with --features cuda).
    #[arg(long, default_value_t = false)]
    cuda: bool,

    /// Use the Vulkan backend (requires building with --features vulkan).
    #[arg(long, default_value_t = false)]
    vulkan: bool,

    /// Use the Metal backend (requires building with --features metal).
    #[arg(long, default_value_t = false)]
    metal: bool,

    /// Quantization for the flow_lm transformer linear weights.
    /// One of: q8|q8_0, q8_1, q8k, q6k, q5|q5_0, q5_1, q5k, q4|q4_0, q4_1, q4k.
    /// CPU only.
    #[arg(long)]
    quant: Option<String>,
}

fn init_tracing() {
    // `info` for everything but the Hub download stack: `hf_hub` transfers through the Xet
    // backend, which reports every retry policy and range probe at `info`. Keep in sync with
    // `LOG_DIRECTIVES` in `ptts/examples/model_helpers.rs`.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "info,xet=warn,xet_client=warn,xet_data=warn,xet_runtime=warn,xet_core_structures=warn",
        )
    });
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::Layer::new().with_target(false))
        .with(filter)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();

    // The checkpoint is downloaded with hf-hub's async client on this runtime.
    // The weight loading that follows is CPU-bound and blocks the runtime
    // thread, which is fine here: nothing is served until it is done.
    let app_state = build_app_state(&args).await?;

    let app = Router::new()
        .route("/speech/tts", any(handler::ws_handler))
        .with_state(app_state)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    tracing::info!(addr = %args.addr, "listening on /speech/tts");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutdown requested");
        })
        .await?;
    Ok(())
}

async fn build_app_state(args: &Args) -> Result<model::AppState> {
    if args.cuda as u8 + args.vulkan as u8 + args.metal as u8 > 1 {
        anyhow::bail!("at most one of --cuda, --vulkan, and --metal can be used");
    }
    let device = if args.cuda {
        DeviceKind::Cuda
    } else if args.vulkan {
        DeviceKind::Vulkan
    } else if args.metal {
        DeviceKind::Metal
    } else {
        DeviceKind::Cpu
    };
    let quant = match args.quant.as_deref() {
        None => Quant::F32,
        Some(name) => Quant::parse(name)?,
    };
    // Both checks happen before `load_ptts` downloads anything: `SynthBuilder`
    // would catch them, but only after the checkpoint is on disk.
    quant.check_device(device)?;
    let unavailable = match device {
        DeviceKind::Cuda if !cfg!(feature = "cuda") => Some("cuda"),
        DeviceKind::Vulkan if !cfg!(feature = "vulkan") => Some("vulkan"),
        DeviceKind::Metal if !cfg!(feature = "metal") => Some("metal"),
        _ => None,
    };
    if let Some(flag) = unavailable {
        anyhow::bail!("--{flag} requested but binary was not built with --features {flag}");
    }
    model::load_ptts(
        args.config.as_ref(),
        args.voice_dir.as_ref(),
        device,
        quant,
        args.temperature,
        args.seed,
        args.max_seq_len,
    )
    .await
}
