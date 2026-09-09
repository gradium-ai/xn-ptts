use crate::encoder::{Encoder, Format};
use crate::model::{AppState, AppStateB, generate_chunks};
use crate::protocol::{TtsReply, TtsRequest, error_codes};
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use ptts::tts_model::{VoicePrefix, seq_budget_for};
use std::sync::Arc;

pub async fn ws_handler(
    State(app): State<AppState>,
    ws: WebSocketUpgrade,
) -> axum::response::Response {
    async fn handle_socket(socket: WebSocket, app: AppState) {
        let result = match app {
            AppState::Cpu(s) => serve_q(socket, s).await,
            AppState::Q80(s) => serve_q(socket, s).await,
            AppState::Q81(s) => serve_q(socket, s).await,
            AppState::Q8k(s) => serve_q(socket, s).await,
            AppState::Q6k(s) => serve_q(socket, s).await,
            AppState::Q50(s) => serve_q(socket, s).await,
            AppState::Q51(s) => serve_q(socket, s).await,
            AppState::Q5k(s) => serve_q(socket, s).await,
            AppState::Q40(s) => serve_q(socket, s).await,
            AppState::Q41(s) => serve_q(socket, s).await,
            AppState::Q4k(s) => serve_q(socket, s).await,
            #[cfg(feature = "cuda")]
            AppState::Cuda(s) => serve_q(socket, s).await,
            #[cfg(feature = "vulkan")]
            AppState::Vulkan(s) => serve_q(socket, s).await,
            #[cfg(feature = "metal")]
            AppState::Metal(s) => serve_q(socket, s).await,
        };
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
    Ready { voice: Arc<VoicePrefix<Q>>, text_buffer: String, stream_id: u32, encoder: Box<Encoder> },
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
                SessionState::Ready { voice, text_buffer, stream_id, encoder },
                TtsRequest::Flush { flush_id },
            ) => {
                flush_buffer(&app, voice, text_buffer, stream_id, encoder, reply_tx).await?;
                let _ = reply_tx.send(TtsReply::Flushed { flush_id });
            }
            (
                SessionState::Ready { voice, text_buffer, stream_id, encoder },
                TtsRequest::EndOfStream,
            ) => {
                flush_buffer(&app, voice, text_buffer, stream_id, encoder, reply_tx).await?;
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
    voice: &Arc<VoicePrefix<Q>>,
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
    if let Err(e) = generate_one(app, voice, &text, stream_id_now, encoder, reply_tx).await {
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
    if !app.voices.contains_key(&voice_name) {
        send_error(reply_tx, error_codes::NOT_FOUND, format!("unknown voice '{voice_name}'"))?;
        return Ok(None);
    }
    tracing::info!(?voice_name, "starting new TTS session");
    // Shared across sessions and built on the voice's first use, so only the first session in
    // a given voice pays for conditioning.
    let voice = match app.voice_prefix(&voice_name) {
        Ok(prefix) => prefix,
        Err(e) => {
            send_error(reply_tx, error_codes::INTERNAL, format!("voice conditioning failed: {e}"))?;
            return Ok(None);
        }
    };
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
        voice,
        text_buffer: String::new(),
        stream_id: 0,
        encoder: Box::new(encoder),
    }))
}

async fn generate_one<Q: xn::BackendQ>(
    app: &Arc<AppStateB<Q>>,
    voice: &Arc<VoicePrefix<Q>>,
    text: &str,
    stream_id: u32,
    encoder: &mut Encoder,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<()> {
    use base64::Engine;

    let (prepared, frames_after_eos) = ptts::tts_model::prepare_text_prompt(text);
    let tokens = app.model.flow_lm.conditioner.tokenize(&prepared)?;
    // Size the cache to this request rather than to `--max-seq-len`, which stays as the
    // operator's ceiling. Seeding from the prefix costs a pair of `slice_set`s per layer, where
    // conditioning would be a whole backbone pass.
    let seq_budget = seq_budget_for(voice.len(), [tokens.len()]).min(app.max_seq_len);
    let state = app.model.init_flow_lm_state_with_prefix(1, seq_budget, voice)?;
    let model = Arc::clone(&app.model);
    let temperature = app.temperature;
    let seed = app.seed_base ^ (stream_id as u64).wrapping_mul(0x9E3779B97F4A7C15);

    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    let join = tokio::task::spawn_blocking(move || {
        generate_chunks(model, state, tokens, temperature, seed, frames_after_eos, audio_tx)
    });

    while let Some(pcm) = audio_rx.recv().await {
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
    }
    drop(audio_rx);
    join.await??;
    Ok(())
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
