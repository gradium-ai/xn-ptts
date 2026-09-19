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
use std::sync::Arc;
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

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();

    // Loading the model happens before the runtime exists: it downloads the
    // checkpoint through hf-hub's blocking client and then spends seconds in
    // CPU-bound weight loading, neither of which belongs on a runtime thread.
    // hf-hub 1.0 drives a tokio runtime of its own for the download; it keeps
    // that runtime on a dedicated thread so a nested `block_on` does not panic,
    // but this ordering does not have to rely on that.
    let app_state = build_app_state(&args)?;
    tokio::runtime::Runtime::new()?.block_on(serve(args, app_state))
}

async fn serve(args: Args, app_state: model::AppState) -> Result<()> {
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

fn build_app_state(args: &Args) -> Result<model::AppState> {
    if args.cuda as u8 + args.vulkan as u8 + args.metal as u8 > 1 {
        anyhow::bail!("at most one of --cuda, --vulkan, and --metal can be used");
    }
    if args.quant.is_some() && (args.cuda || args.vulkan || args.metal) {
        anyhow::bail!("--quant cannot be combined with a gpu backend; quantization is CPU-only");
    }
    if args.cuda {
        #[cfg(feature = "cuda")]
        {
            let dev = xn::CudaDevice::new(0)?;
            unsafe {
                dev.disable_event_tracking();
            }
            let s = model::load_ptts::<xn::Unquantized<half::bf16, _>>(
                args.config.as_ref(),
                args.voice_dir.as_ref(),
                args.temperature,
                args.seed,
                args.max_seq_len,
                dev,
            )?;
            return Ok(model::AppState::Cuda(Arc::new(s)));
        }
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("--cuda requested but binary was not built with --features cuda");
    }
    if args.vulkan {
        #[cfg(feature = "vulkan")]
        {
            let dev = xn::VulkanDevice::new(0)?;
            let s = model::load_ptts::<xn::Unquantized<f32, _>>(
                args.config.as_ref(),
                args.voice_dir.as_ref(),
                args.temperature,
                args.seed,
                args.max_seq_len,
                dev,
            )?;
            return Ok(model::AppState::Vulkan(Arc::new(s)));
        }
        #[cfg(not(feature = "vulkan"))]
        anyhow::bail!("--vulkan requested but binary was not built with --features vulkan");
    }
    if args.metal {
        #[cfg(feature = "metal")]
        {
            let dev = xn::MetalDevice::new(0)?;
            let s = model::load_ptts::<xn::Unquantized<f32, _>>(
                args.config.as_ref(),
                args.voice_dir.as_ref(),
                args.temperature,
                args.seed,
                args.max_seq_len,
                dev,
            )?;
            return Ok(model::AppState::Metal(Arc::new(s)));
        }
        #[cfg(not(feature = "metal"))]
        anyhow::bail!("--metal requested but binary was not built with --features metal");
    }
    build_cpu_state(args)
}

fn build_cpu_state(args: &Args) -> Result<model::AppState> {
    use model::AppState;
    let temp = args.temperature;
    let seed = args.seed;
    let mlen = args.max_seq_len;
    let config = args.config.as_ref();
    let voice_dir = args.voice_dir.as_ref();
    let state = match args.quant.as_deref() {
        None => {
            tracing::info!("using cpu backend (unquantized f32)");
            AppState::Cpu(Arc::new(model::load_ptts::<xn::Unquantized<f32, _>>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some("q8" | "q8_0") => {
            tracing::info!("using cpu q8_0 backend");
            AppState::Q80(Arc::new(model::load_ptts::<xn::quantized::Q80F32>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some("q8_1") => {
            tracing::info!("using cpu q8_1 backend");
            AppState::Q81(Arc::new(model::load_ptts::<xn::quantized::Q81F32>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some("q8k") => {
            tracing::info!("using cpu q8k backend");
            AppState::Q8k(Arc::new(model::load_ptts::<xn::quantized::Q8kF32>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some("q6k") => {
            tracing::info!("using cpu q6k backend");
            AppState::Q6k(Arc::new(model::load_ptts::<xn::quantized::Q6kF32>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some("q5" | "q5_0") => {
            tracing::info!("using cpu q5_0 backend");
            AppState::Q50(Arc::new(model::load_ptts::<xn::quantized::Q50F32>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some("q5_1") => {
            tracing::info!("using cpu q5_1 backend");
            AppState::Q51(Arc::new(model::load_ptts::<xn::quantized::Q51F32>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some("q5k") => {
            tracing::info!("using cpu q5k backend");
            AppState::Q5k(Arc::new(model::load_ptts::<xn::quantized::Q5kF32>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some("q4" | "q4_0") => {
            tracing::info!("using cpu q4_0 backend");
            AppState::Q40(Arc::new(model::load_ptts::<xn::quantized::Q40F32>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some("q4_1") => {
            tracing::info!("using cpu q4_1 backend");
            AppState::Q41(Arc::new(model::load_ptts::<xn::quantized::Q41F32>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some("q4k") => {
            tracing::info!("using cpu q4k backend");
            AppState::Q4k(Arc::new(model::load_ptts::<xn::quantized::Q4kF32>(
                config,
                voice_dir,
                temp,
                seed,
                mlen,
                xn::CPU,
            )?))
        }
        Some(other) => anyhow::bail!("unsupported --quant value '{other}'"),
    };
    Ok(state)
}
