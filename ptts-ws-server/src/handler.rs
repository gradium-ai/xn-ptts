use crate::encoder::{Encoder, Format};
use crate::model::AppState;
use crate::protocol::{TtsReply, TtsRequest, error_codes};
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use ptts::synth::{Session, SpeechOptions};

pub async fn ws_handler(
    State(app): State<AppState>,
    ws: WebSocketUpgrade,
) -> axum::response::Response {
    async fn handle_socket(socket: WebSocket, app: AppState) {
        if let Err(e) = serve(socket, app).await {
            tracing::error!(error = %e, "ws session terminated");
        }
    }
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

async fn serve(socket: WebSocket, app: AppState) -> Result<()> {
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

enum SessionState {
    Awaiting,
    Ready { session: Session, text_buffer: String, stream_id: u32, encoder: Box<Encoder> },
}

async fn run_session(
    app: AppState,
    stream: &mut futures_util::stream::SplitStream<WebSocket>,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<()> {
    use futures_util::StreamExt;
    let mut sess: SessionState = SessionState::Awaiting;

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
                SessionState::Ready { session, text_buffer, stream_id, encoder },
                TtsRequest::Flush { flush_id },
            ) => {
                flush_buffer(&app, session, text_buffer, stream_id, encoder, reply_tx).await?;
                let _ = reply_tx.send(TtsReply::Flushed { flush_id });
            }
            (
                SessionState::Ready { session, text_buffer, stream_id, encoder },
                TtsRequest::EndOfStream,
            ) => {
                flush_buffer(&app, session, text_buffer, stream_id, encoder, reply_tx).await?;
                let _ = reply_tx.send(TtsReply::EndOfStream);
                tracing::info!("websocket stream closed by client (end of stream)");
                return Ok(());
            }
        }
    }
    tracing::info!("websocket stream closed by client");
    Ok(())
}

async fn flush_buffer(
    app: &AppState,
    session: &Session,
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
    if let Err(e) = generate_one(app, session, &text, stream_id_now, encoder, reply_tx).await {
        tracing::warn!(error = %e, stream_id = stream_id_now, "generation failed");
        send_error(reply_tx, error_codes::INTERNAL, format!("generation failed: {e}"))?;
    }
    Ok(())
}

async fn handle_setup(
    app: &AppState,
    model_name: String,
    output_format: String,
    voice: Option<String>,
    voice_id: Option<String>,
    voice_emb: Option<String>,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<Option<SessionState>> {
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
    if !app.voices.contains(&voice_name) {
        send_error(reply_tx, error_codes::NOT_FOUND, format!("unknown voice '{voice_name}'"))?;
        return Ok(None);
    }
    tracing::info!(?voice_name, "starting new TTS session");
    // Conditioning on the voice happens once here, not per request: every
    // generation below clones this primed state.
    let opts = SpeechOptions::default().voice(voice_name.clone());
    let session = match app.synth.session(&opts, app.max_seq_len) {
        Ok(session) => session,
        Err(e) => {
            send_error(reply_tx, error_codes::INTERNAL, format!("failed to prime voice: {e}"))?;
            return Ok(None);
        }
    };
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
        session,
        text_buffer: String::new(),
        stream_id: 0,
        encoder: Box::new(encoder),
    }))
}

async fn generate_one(
    app: &AppState,
    session: &Session,
    text: &str,
    stream_id: u32,
    encoder: &mut Encoder,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<()> {
    use base64::Engine;

    // One request is one utterance: prepare and tokenize it here rather than
    // letting `Session::stream` split it on sentence boundaries, which is what
    // this server did before and what its stream ids assume.
    let (prepared, frames_after_eos) = ptts::tts_model::prepare_text_prompt(text);
    let tokens = session.tokenize(&prepared)?;
    let seed = app.seed_base ^ (stream_id as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let rng = Box::new(ptts::flow_lm::NormalRng::new(app.temperature, seed)?);

    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    let stream = session.stream_tokens(tokens, frames_after_eos, rng)?;
    // The generation threads are `Synth`'s; this one just moves chunks onto the
    // tokio channel so the socket writer stays async.
    let join = tokio::task::spawn_blocking(move || -> Result<()> {
        for chunk in stream {
            // A send failure means the client went away; dropping `stream`
            // stops the workers.
            if audio_tx.send(chunk?).is_err() {
                break;
            }
        }
        Ok(())
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
