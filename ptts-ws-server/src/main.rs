mod encoder;
mod handler;
mod model;
mod protocol;

use anyhow::{Context, Result};
use axum::Router;
use axum::routing::any;
use clap::Parser;
use ptts::preprocess::{Normalize, Rules};
use ptts::synth::{DeviceKind, Quant};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "ptts-ws-server")]
#[command(about = "WebSocket server for Phonon")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8080")]
    addr: String,

    /// Maximum open WebSocket sessions. Excess connections receive HTTP 429.
    #[arg(long, default_value = "1", value_parser = parse_request_limit)]
    max_concurrent_requests: std::num::NonZeroUsize,

    /// Model to load: a local checkpoint folder or a `config.json` in one, or a Hugging Face repo as
    /// `hf://owner/name` or `hf://owner/name@revision`. Use a commit as the revision to reproduce a
    /// release. Defaults to the default model, downloaded from the Hub and cached on first use.
    #[arg(long, alias = "config", default_value = ptts::checkpoint::DEFAULT_MODEL)]
    model: String,

    /// Optional directory of additional voice safetensors to load. Each
    /// `*.safetensors` file is loaded as a voice keyed by its file stem; load
    /// errors are logged and skipped rather than fatal.
    #[arg(long)]
    voice_dir: Option<std::path::PathBuf>,

    /// Voice catalog whose voices in --lang are served besides the checkpoint's: a Hugging Face
    /// repo as hf://owner/name[@revision], or a local JSONL file or folder holding voices.jsonl,
    /// one `{"name", "filename"}` object per line with `filename` relative to the catalog.
    /// Defaults to the default voice catalog; empty for none. Voices that fail to load are logged
    /// and skipped.
    #[arg(long, default_value = ptts::voice_catalog::DEFAULT_VOICE_CATALOG)]
    voice_catalog: String,

    #[arg(long, default_value_t = 0.3)]
    temperature: f32,

    #[arg(long, default_value_t = 4242424242424242)]
    seed: u64,

    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// Device to run on: auto, cpu, cuda, vulkan or metal. `auto` picks the GPU backend this
    /// build was compiled with, if any, and the CPU otherwise.
    #[arg(long, default_value = "auto")]
    device: String,

    /// Quantization for the flow_lm transformer linear weights.
    /// One of: q8|q8_0, q8_1, q8k, q6k, q5|q5_0, q5_1, q5k, q4|q4_0, q4_1, q4k.
    /// CPU only.
    #[arg(long)]
    quant: Option<String>,

    /// Language incoming text is normalized as before tokenizing: `en`, `fr`, `de`, `es` or
    /// `pt`. Required: the spoken forms differ per language, so there is nothing safe to guess.
    /// `none` serves the text as written, which the model reads less well.
    #[arg(long)]
    lang: String,

    /// Which word rewrites run on the normalized text: `default` (numbers, currency,
    /// dashed-digits, emails, urls, abbreviations, elongations), `all` (those and phones, times,
    /// dates), `none`, or a comma-separated list of rule names. Has no effect with `--lang none`.
    #[arg(long, default_value = "default")]
    rewrites: String,

    /// Set one of the checkpoint's conditioners, e.g. `duration_delta=0.5`; repeatable. Those
    /// not set take their defaults.
    #[arg(long = "condition", value_name = "NAME=VALUE")]
    conditions: Vec<String>,
}

fn parse_request_limit(value: &str) -> Result<std::num::NonZeroUsize, String> {
    let limit: std::num::NonZeroUsize = value.parse().map_err(|e| format!("{e}"))?;
    if limit.get() > tokio::sync::Semaphore::MAX_PERMITS {
        return Err(format!("limit must be at most {}", tokio::sync::Semaphore::MAX_PERMITS));
    }
    Ok(limit)
}

fn init_tracing() {
    // `info` for everything but the Hub download stack: `hf_hub` transfers through the Xet
    // backend, which reports every retry policy and range probe at `info`. Keep in sync with
    // `LOG_DIRECTIVES` in `ptts/src/bin/ptts/model_helpers.rs`.
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

    // The weight loading after the download is CPU-bound and blocks the runtime thread, which is
    // fine here: nothing is served until it is done.
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
    let device = args.device.parse::<DeviceKind>()?;
    let quant = match args.quant.as_deref() {
        None => Quant::F32,
        Some(name) => name.parse::<Quant>()?,
    };
    // Device and weight-format checks run before downloading: `SynthBuilder`
    // would catch them, but only after the checkpoint is on disk.
    quant.check_device(device)?;
    let normalize = args.lang.parse::<Normalize>()?.with_rules(args.rewrites.parse::<Rules>()?);
    let mut conditions = Vec::new();
    for condition in &args.conditions {
        let (name, value) = condition.split_once('=').context("--condition takes NAME=VALUE")?;
        conditions.push((name.to_string(), value.to_string()));
    }
    model::load_ptts(
        &args.model,
        args.voice_dir.as_ref(),
        Some(args.voice_catalog.as_str()).filter(|c| !c.is_empty()),
        device,
        quant,
        args.temperature,
        args.seed,
        args.max_seq_len,
        normalize,
        &conditions,
        args.max_concurrent_requests,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_limit_is_positive_and_defaults_to_one() {
        assert!(parse_request_limit(&usize::MAX.to_string()).is_err());
        let args = Args::try_parse_from(["server", "--model", "model", "--lang", "none"]).unwrap();
        assert_eq!(args.max_concurrent_requests.get(), 1);
        let args = Args::try_parse_from([
            "server",
            "--model",
            "model",
            "--lang",
            "none",
            "--max-concurrent-requests",
            "3",
        ])
        .unwrap();
        assert_eq!(args.max_concurrent_requests.get(), 3);
        assert!(
            Args::try_parse_from([
                "server",
                "--model",
                "model",
                "--lang",
                "none",
                "--max-concurrent-requests",
                "0",
            ])
            .is_err()
        );
    }
}
