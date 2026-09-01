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

    /// Explicit weights file, overriding the `model.safetensors` / `model.q8.gguf`
    /// lookup next to `--config`.
    #[arg(long)]
    model: Option<std::path::PathBuf>,

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

    /// Use the WebGPU (wgpu) backend (requires building with --features webgpu).
    #[arg(long, default_value_t = false)]
    webgpu: bool,

    /// Compute dtype for the WebGPU backend: f32, or f16 when the adapter
    /// advertises WGSL `shader-f16`. Ignored by the other backends.
    #[arg(long, default_value = "f32")]
    webgpu_dtype: String,

    /// Quantization for the flow_lm transformer linear weights.
    /// One of: q8|q8_0, q8_1, q8k, q6k, q5|q5_0, q5_1, q5k, q4|q4_0, q4_1, q4k.
    /// CPU only.
    #[arg(long)]
    quant: Option<String>,
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::Layer::new().with_target(false))
        .with(filter)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();

    let app_state = build_app_state(&args)?;

    let app = Router::new()
        .route("/speech/tts", any(handler::ws_handler))
        .route("/", axum::routing::get(index))
        .route("/api/info", axum::routing::get(api_info))
        .with_state(app_state)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    tracing::info!(addr = %args.addr, "serving the web app on / and websockets on /speech/tts");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutdown requested");
        })
        .await?;
    Ok(())
}

/// The single-page web app. Embedded rather than served from disk so a release
/// binary is self-contained.
async fn index() -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("../www/index.html"),
    )
        .into_response()
}

/// Hand-rolled rather than `axum::Json`: the workspace pins axum with
/// `default-features = false` and no `json` feature.
async fn api_info(
    axum::extract::State(app): axum::extract::State<model::AppState>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match serde_json::to_string(&app.info()) {
        Ok(body) => {
            ([(axum::http::header::CONTENT_TYPE, "application/json")], body).into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to serialize server info: {e}"),
        )
            .into_response(),
    }
}

fn build_app_state(args: &Args) -> Result<model::AppState> {
    if args.cuda as u8 + args.vulkan as u8 + args.metal as u8 + args.webgpu as u8 > 1 {
        anyhow::bail!("at most one of --cuda, --vulkan, --metal, and --webgpu can be used");
    }
    // WebGPU is the exception: xn's webgpu backend has its own q8_0 path, so
    // `--quant q8` there is a gpu quantization rather than a fallback to cpu.
    if args.quant.is_some() && (args.cuda || args.vulkan || args.metal) {
        anyhow::bail!(
            "--quant cannot be combined with --cuda/--vulkan/--metal; quantization there is CPU-only"
        );
    }
    let opts = |backend: &str| model::LoadOpts {
        config: args.config.as_ref(),
        model: args.model.as_ref(),
        voice_dir: args.voice_dir.as_ref(),
        temperature: args.temperature,
        seed_base: args.seed,
        max_seq_len: args.max_seq_len,
        backend: backend.to_string(),
    };
    if args.cuda {
        #[cfg(feature = "cuda")]
        {
            let dev = xn::CudaDevice::new(0)?;
            unsafe {
                dev.disable_event_tracking();
            }
            let s = model::load_ptts::<xn::Unquantized<half::bf16, _>>(&opts("cuda bf16"), dev)?;
            return Ok(model::AppState::Cuda(Arc::new(s)));
        }
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("--cuda requested but binary was not built with --features cuda");
    }
    if args.vulkan {
        #[cfg(feature = "vulkan")]
        {
            let dev = xn::VulkanDevice::new(0)?;
            let s = model::load_ptts::<xn::Unquantized<f32, _>>(&opts("vulkan f32"), dev)?;
            return Ok(model::AppState::Vulkan(Arc::new(s)));
        }
        #[cfg(not(feature = "vulkan"))]
        anyhow::bail!("--vulkan requested but binary was not built with --features vulkan");
    }
    if args.metal {
        #[cfg(feature = "metal")]
        {
            let dev = xn::MetalDevice::new(0)?;
            let s = model::load_ptts::<xn::Unquantized<f32, _>>(&opts("metal f32"), dev)?;
            return Ok(model::AppState::Metal(Arc::new(s)));
        }
        #[cfg(not(feature = "metal"))]
        anyhow::bail!("--metal requested but binary was not built with --features metal");
    }
    if args.webgpu {
        #[cfg(feature = "webgpu")]
        {
            return build_webgpu_state(args, &opts);
        }
        #[cfg(not(feature = "webgpu"))]
        anyhow::bail!("--webgpu requested but binary was not built with --features webgpu");
    }
    build_cpu_state(args, &opts)
}

/// The four WebGPU states: {f32, f16} activations x {unquantized, q8_0 weights}.
/// f16 needs the adapter to advertise WGSL `shader-f16`, so it is checked rather
/// than silently downgraded -- a silent downgrade would make a benchmark lie.
#[cfg(feature = "webgpu")]
fn build_webgpu_state<'a>(
    args: &'a Args,
    opts: &dyn Fn(&str) -> model::LoadOpts<'a>,
) -> Result<model::AppState> {
    let dev = xn::WebGpuDevice::new(0)?;
    let f16 = match args.webgpu_dtype.as_str() {
        "f32" => false,
        "f16" => true,
        other => anyhow::bail!("unsupported --webgpu-dtype '{other}', expected f32 or f16"),
    };
    if f16 && !dev.supports_f16() {
        anyhow::bail!(
            "--webgpu-dtype f16 requested but this adapter does not support WGSL shader-f16"
        );
    }
    let quant = args.quant.as_deref();
    let state = match (quant, f16) {
        (None, false) => model::AppState::WebGpu(Arc::new(model::load_ptts::<
            xn::Unquantized<f32, _>,
        >(&opts("webgpu f32"), dev)?)),
        (None, true) => model::AppState::WebGpuF16(Arc::new(model::load_ptts::<
            xn::Unquantized<half::f16, _>,
        >(&opts("webgpu f16"), dev)?)),
        (Some("q8" | "q8_0"), false) => {
            model::AppState::WebGpuQ80(Arc::new(model::load_ptts::<
                xn::webgpu_backend::quantization::Q80F32,
            >(&opts("webgpu q8_0/f32"), dev)?))
        }
        (Some("q8" | "q8_0"), true) => {
            model::AppState::WebGpuQ80F16(Arc::new(model::load_ptts::<
                xn::webgpu_backend::quantization::Q80F16,
            >(&opts("webgpu q8_0/f16"), dev)?))
        }
        (Some(other), _) => anyhow::bail!(
            "--quant '{other}' is not supported on the webgpu backend; only q8/q8_0 is"
        ),
    };
    Ok(state)
}

fn build_cpu_state<'a>(
    args: &'a Args,
    opts: &dyn Fn(&str) -> model::LoadOpts<'a>,
) -> Result<model::AppState> {
    use model::AppState;
    macro_rules! cpu {
        ($variant:ident, $q:ty, $label:expr) => {{
            tracing::info!(concat!("using cpu backend (", $label, ")"));
            AppState::$variant(Arc::new(model::load_ptts::<$q>(
                &opts(concat!("cpu ", $label)),
                xn::CPU,
            )?))
        }};
    }
    let state = match args.quant.as_deref() {
        None => cpu!(Cpu, xn::Unquantized<f32, _>, "f32"),
        Some("q8" | "q8_0") => cpu!(Q80, xn::quantized::Q80F32, "q8_0"),
        Some("q8_1") => cpu!(Q81, xn::quantized::Q81F32, "q8_1"),
        Some("q8k") => cpu!(Q8k, xn::quantized::Q8kF32, "q8k"),
        Some("q6k") => cpu!(Q6k, xn::quantized::Q6kF32, "q6k"),
        Some("q5" | "q5_0") => cpu!(Q50, xn::quantized::Q50F32, "q5_0"),
        Some("q5_1") => cpu!(Q51, xn::quantized::Q51F32, "q5_1"),
        Some("q5k") => cpu!(Q5k, xn::quantized::Q5kF32, "q5k"),
        Some("q4" | "q4_0") => cpu!(Q40, xn::quantized::Q40F32, "q4_0"),
        Some("q4_1") => cpu!(Q41, xn::quantized::Q41F32, "q4_1"),
        Some("q4k") => cpu!(Q4k, xn::quantized::Q4kF32, "q4k"),
        Some(other) => anyhow::bail!("unsupported --quant value '{other}'"),
    };
    Ok(state)
}
