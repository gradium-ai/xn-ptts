use crate::encoder::{Encoder, Format};
use crate::model::{AppState, AppStateB, generate_chunks};
use crate::protocol::{TtsReply, TtsRequest, error_codes};
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use ptts::tts_model::TTSState;
use std::sync::Arc;

pub async fn ws_handler(
    State(app): State<AppState>,
    ws: WebSocketUpgrade,
) -> axum::response::Response {
    async fn handle_socket(socket: WebSocket, app: AppState) {
        let result = crate::model::dispatch!(app, |s| serve_q(socket, s).await);
        if let Err(e) = result {
            tracing::error!(error = %e, "ws session terminated");
        }
    }
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

async fn serve_q<Q: xn::BackendQ>(socket: WebSocket, app: Arc<AppStateB<Q>>) -> Result<()> {
    use futures_util::{SinkExt, StreamExt};
    let (mut tx, mut rx) = socket.split();
    let (reply_tx, mut reply_rx) = tokio::sync::mpsc::unbounded_channel();

    let forwarder = tokio::spawn(async move {
        while let Some(reply) = reply_rx.recv().await {
            let json = serde_json::to_string(&reply)?;
            if tx.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
        let _ = tx.close().await;
        Ok::<_, anyhow::Error>(())
    });

    let outcome = run_session(app, &mut rx, &reply_tx).await;
    drop(reply_tx);
    let _ = forwarder.await;
    tracing::info!("websocket session ended");
    outcome
}

enum SessionState<Q: xn::BackendQ> {
    Awaiting,
    Ready { base_state: TTSState<Q>, text_buffer: String, stream_id: u32, encoder: Box<Encoder> },
}

async fn run_session<Q: xn::BackendQ>(
    app: Arc<AppStateB<Q>>,
    stream: &mut futures_util::stream::SplitStream<WebSocket>,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<()> {
    use futures_util::StreamExt;
    let mut sess: SessionState<Q> = SessionState::Awaiting;

    while let Some(msg) = stream.next().await {
        let msg = msg?;
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => return Ok(()),
            Message::Binary(_) | Message::Ping(_) | Message::Pong(_) => continue,
        };
        let req: TtsRequest = match serde_json::from_str(text.as_str()) {
            Ok(r) => r,
            Err(e) => {
                send_error(reply_tx, error_codes::BAD_REQUEST, format!("invalid request: {e}"))?;
                continue;
            }
        };
        match (&mut sess, req) {
            (
                SessionState::Awaiting,
                TtsRequest::Setup { model_name, output_format, voice, voice_id, voice_emb, .. },
            ) => match handle_setup(
                &app,
                model_name,
                output_format,
                voice,
                voice_id,
                voice_emb,
                reply_tx,
            )
            .await?
            {
                Some(new_state) => sess = new_state,
                None => continue,
            },
            (SessionState::Awaiting, _) => {
                send_error(
                    reply_tx,
                    error_codes::BAD_REQUEST,
                    "expected setup as first message".into(),
                )?;
            }
            (SessionState::Ready { .. }, TtsRequest::Setup { .. }) => {
                send_error(
                    reply_tx,
                    error_codes::BAD_REQUEST,
                    "session already initialized".into(),
                )?;
            }
            (SessionState::Ready { text_buffer, .. }, TtsRequest::Text { text }) => {
                text_buffer.push_str(&text);
            }
            (
                SessionState::Ready { base_state, text_buffer, stream_id, encoder },
                TtsRequest::Flush { flush_id },
            ) => {
                flush_buffer(&app, base_state, text_buffer, stream_id, encoder, reply_tx).await?;
                let _ = reply_tx.send(TtsReply::Flushed { flush_id });
            }
            (
                SessionState::Ready { base_state, text_buffer, stream_id, encoder },
                TtsRequest::EndOfStream,
            ) => {
                flush_buffer(&app, base_state, text_buffer, stream_id, encoder, reply_tx).await?;
                let _ = reply_tx.send(TtsReply::EndOfStream);
                tracing::info!("websocket stream closed by client (end of stream)");
                return Ok(());
            }
        }
    }
    tracing::info!("websocket stream closed by client");
    Ok(())
}

async fn flush_buffer<Q: xn::BackendQ>(
    app: &Arc<AppStateB<Q>>,
    base_state: &TTSState<Q>,
    text_buffer: &mut String,
    stream_id: &mut u32,
    encoder: &mut Encoder,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<()> {
    if text_buffer.is_empty() {
        return Ok(());
    }
    let stream_id_now = *stream_id;
    *stream_id = stream_id.saturating_add(1);
    let text = std::mem::take(text_buffer);
    if let Err(e) = generate_one(app, base_state, &text, stream_id_now, encoder, reply_tx).await {
        tracing::warn!(error = %e, stream_id = stream_id_now, "generation failed");
        send_error(reply_tx, error_codes::INTERNAL, format!("generation failed: {e}"))?;
    }
    Ok(())
}

async fn handle_setup<Q: xn::BackendQ>(
    app: &Arc<AppStateB<Q>>,
    model_name: String,
    output_format: String,
    voice: Option<String>,
    voice_id: Option<String>,
    voice_emb: Option<String>,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<Option<SessionState<Q>>> {
    if voice_emb.as_deref().is_some_and(|s| !s.is_empty()) {
        send_error(
            reply_tx,
            error_codes::NOT_IMPLEMENTED,
            "voice_emb prompts are not yet supported".into(),
        )?;
        return Ok(None);
    }
    let format = match output_format.parse::<Format>() {
        Ok(f) => f,
        Err(e) => {
            send_error(reply_tx, error_codes::BAD_REQUEST, format!("{e}"))?;
            return Ok(None);
        }
    };
    let encoder = match Encoder::new(format, app.frame_size as usize, app.sample_rate as usize) {
        Ok(e) => e,
        Err(e) => {
            send_error(
                reply_tx,
                error_codes::INTERNAL,
                format!("failed to create audio encoder: {e}"),
            )?;
            return Ok(None);
        }
    };
    let voice_name = voice_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .or(voice.as_deref().filter(|s| !s.is_empty()))
        .unwrap_or(&app.default_voice);
    let voice_name =
        if voice_name == "default" { &app.default_voice } else { voice_name }.to_string();
    let voice_emb_t = match app.voices.get(&voice_name) {
        Some(v) => v,
        None => {
            send_error(reply_tx, error_codes::NOT_FOUND, format!("unknown voice '{voice_name}'"))?;
            return Ok(None);
        }
    };
    let mut base_state = match app.model.init_flow_lm_state(1, app.max_seq_len) {
        Ok(s) => s,
        Err(e) => {
            send_error(reply_tx, error_codes::INTERNAL, format!("init_flow_lm_state failed: {e}"))?;
            return Ok(None);
        }
    };
    tracing::info!(?voice_name, "starting new TTS session");
    if let Err(e) = app.model.prompt_audio(&mut base_state, voice_emb_t) {
        send_error(reply_tx, error_codes::INTERNAL, format!("prompt_audio failed: {e}"))?;
        return Ok(None);
    }
    tracing::info!(?voice_name, "prompted voice embedding");
    let request_id = uuid::Uuid::new_v4().to_string();
    let model_name =
        if model_name.is_empty() { "kyutai/pocket-tts".to_string() } else { model_name };
    let ready = TtsReply::Ready {
        model_name,
        sample_rate: app.sample_rate,
        frame_size: app.frame_size,
        audio_stream_names: vec![],
        text_stream_names: vec![],
        request_id,
    };
    if reply_tx.send(ready).is_err() {
        anyhow::bail!("reply channel closed before ready");
    }
    if let Some(header) = encoder.header() {
        use base64::Engine;
        let audio = base64::engine::general_purpose::STANDARD.encode(header);
        let header_reply = TtsReply::Audio { audio, start_s: 0.0, stop_s: 0.0, stream_id: 0 };
        if reply_tx.send(header_reply).is_err() {
            anyhow::bail!("reply channel closed before header");
        }
    }
    Ok(Some(SessionState::Ready {
        base_state,
        text_buffer: String::new(),
        stream_id: 0,
        encoder: Box::new(encoder),
    }))
}

async fn generate_one<Q: xn::BackendQ>(
    app: &Arc<AppStateB<Q>>,
    base_state: &TTSState<Q>,
    text: &str,
    stream_id: u32,
    encoder: &mut Encoder,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<()> {
    use base64::Engine;

    let (prepared, frames_after_eos) = ptts::tts_model::prepare_text_prompt(text);
    let tokens = app.model.flow_lm.conditioner.tokenize(&prepared)?;
    let state = base_state.clone();
    let model = Arc::clone(&app.model);
    let temperature = app.temperature;
    let seed = app.seed_base ^ (stream_id as u64).wrapping_mul(0x9E3779B97F4A7C15);

    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    let num_tokens = tokens.len();
    // Timed from here, so the measurement covers what a client actually waits
    // for: text prompting, sampling, Mimi decoding and encoding. Model load and
    // voice conditioning happened once at setup and are excluded.
    let t0 = std::time::Instant::now();
    let join = tokio::task::spawn_blocking(move || {
        generate_chunks(model, state, tokens, temperature, seed, frames_after_eos, audio_tx)
    });

    let mut ttfa: Option<std::time::Duration> = None;
    let mut frame_ms: Vec<f64> = Vec::new();
    let mut samples: usize = 0;
    let mut last = t0;
    while let Some(pcm) = audio_rx.recv().await {
        let now = std::time::Instant::now();
        samples += pcm.len();
        // Interval between PCM chunks -- the cadence a streaming consumer sees --
        // rather than the cost of one sampling step in isolation.
        frame_ms.push((now - last).as_secs_f64() * 1e3);
        last = now;
        let encoded = encoder.encode(&pcm)?;
        let audio = base64::engine::general_purpose::STANDARD.encode(&encoded.data);
        if reply_tx
            .send(TtsReply::Audio {
                audio,
                start_s: encoded.start_s,
                stop_s: encoded.stop_s,
                stream_id,
            })
            .is_err()
        {
            break;
        }
        if ttfa.is_none() {
            ttfa = Some(now - t0);
        }
    }
    drop(audio_rx);
    let total = t0.elapsed();
    join.await??;

    let total_ms = total.as_secs_f64() * 1e3;
    let audio_ms = samples as f64 / app.sample_rate as f64 * 1e3;
    let stats = crate::protocol::GenStats {
        backend: app.backend.clone(),
        device: xn::Backend::name(app.model.device()),
        stream_id,
        chars: text.chars().count(),
        tokens: num_tokens,
        frames: frame_ms.len(),
        audio_ms,
        total_ms,
        ttfa_ms: ttfa.map(|d| d.as_secs_f64() * 1e3),
        // Audio produced per unit of wall time, so >1 is faster than realtime.
        rtf: if total_ms > 0.0 { audio_ms / total_ms } else { 0.0 },
        frame_ms_mean: fin(mean(&frame_ms)),
        frame_ms_p50: fin(percentile(&frame_ms, 50.0)),
        frame_ms_p95: fin(percentile(&frame_ms, 95.0)),
        frame_ms_max: fin(frame_ms.iter().copied().fold(f64::NAN, f64::max)),
        threads: xn::get_num_threads(),
    };
    tracing::info!(
        backend = %stats.backend,
        frames = stats.frames,
        ttfa_ms = ?stats.ttfa_ms.map(|v| (v * 10.0).round() / 10.0),
        total_ms = (stats.total_ms * 10.0).round() / 10.0,
        rtf = (stats.rtf * 100.0).round() / 100.0,
        "generation complete"
    );
    let _ = reply_tx.send(TtsReply::Stats { json_stats: serde_json::to_string(&stats)? });
    Ok(())
}

/// `None` for the empty-stream case, so the UI shows "n/a" rather than relying
/// on how the JSON encoder happens to render NaN.
fn fin(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
}

/// Nearest-rank percentile over a copy of `v`; `v` itself stays in arrival order.
fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn send_error(
    tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
    code: u32,
    message: String,
) -> Result<()> {
    tx.send(TtsReply::Error { message, code })
        .map_err(|_| anyhow::anyhow!("reply channel closed"))?;
    Ok(())
}
