use crate::encoder::{Encoder, Format};
use crate::model::AppState;
use crate::protocol::{TtsReply, TtsRequest, error_codes};
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use ptts::synth::{Session, SpeechOptions};
use std::time::Duration;
use tokio::sync::{mpsc, watch};

#[cfg(test)]
#[path = "handler_tests.rs"]
mod buffering_tests;

const REPLY_QUEUE: usize = 16;
const AUDIO_QUEUE: usize = 4;
const REQUEST_QUEUE: usize = 16;
const MAX_INPUT_CHARS: usize = 4096;
const MAX_MESSAGE_BYTES: usize = 64 * 1024;
const SEND_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a session may wait for its next message before it is closed and its slot freed.
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);

pub async fn ws_handler(
    State(app): State<AppState>,
    ws: WebSocketUpgrade,
) -> axum::response::Response {
    let ws = ws.max_message_size(MAX_MESSAGE_BYTES);
    use axum::response::IntoResponse;
    let permit = match app.requests.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return (
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "the server is busy; retry after an active WebSocket session closes",
            )
                .into_response();
        }
    };
    async fn handle_socket(
        socket: WebSocket,
        app: AppState,
        _permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        if let Err(e) = serve(socket, app).await {
            tracing::error!(error = %e, "ws session terminated");
        }
    }
    // A session retains its primed voice and KV cache between messages, so its
    // slot is held for the whole connection, including while idle.
    ws.on_upgrade(move |socket| handle_socket(socket, app, permit))
}

async fn serve(socket: WebSocket, app: AppState) -> Result<()> {
    use futures_util::StreamExt;
    let (tx, rx) = socket.split();
    let (reply_tx, reply_rx) = mpsc::channel(REPLY_QUEUE);
    let (request_tx, mut request_rx) = mpsc::channel(REQUEST_QUEUE);
    let (cancel_tx, mut cancel_rx) = watch::channel(false);
    // Read independently so a peer Close cancels generation while the request queue has room.
    let reader = tokio::spawn(read_requests(rx, request_tx, cancel_tx.clone()));
    let forwarder = tokio::spawn(forward_replies(tx, reply_rx, cancel_rx.clone(), SEND_TIMEOUT));

    let outcome = run_session(app, &mut request_rx, &reply_tx, &mut cancel_rx).await;
    // run_session has joined any generation before it returns. Normal EOS still drains replies.
    reader.abort();
    let _ = reader.await;
    drop(reply_tx);
    let forwarded = forwarder.await?;
    drop(cancel_tx);
    tracing::info!("websocket session ended");
    outcome.and(forwarded)
}

async fn read_requests<S, E>(
    mut socket: S,
    requests: mpsc::Sender<Message>,
    cancel: watch::Sender<bool>,
) where
    S: futures_util::Stream<Item = std::result::Result<Message, E>> + Unpin,
{
    use futures_util::StreamExt;
    while let Some(msg) = socket.next().await {
        match msg {
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(Message::Ping(_) | Message::Pong(_) | Message::Binary(_)) => continue,
            Ok(msg) => {
                // A full queue pauses reading until generation catches up. A stalled socket
                // writer still times out and closes the reply channel, stopping the session.
                if requests.send(msg).await.is_err() {
                    break;
                }
            }
        }
    }
    let _ = cancel.send(true);
}

async fn cancelled(cancel: &mut watch::Receiver<bool>) {
    while !*cancel.borrow_and_update() {
        if cancel.changed().await.is_err() {
            return;
        }
    }
}

async fn forward_replies<S>(
    mut socket: S,
    mut replies: mpsc::Receiver<TtsReply>,
    mut cancel: watch::Receiver<bool>,
    timeout: Duration,
) -> Result<()>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::fmt::Display,
{
    use futures_util::SinkExt;
    loop {
        let reply = tokio::select! {
            biased;
            _ = cancelled(&mut cancel) => return Ok(()),
            reply = replies.recv() => match reply {
                Some(reply) => reply,
                None => break,
            },
        };
        let json = serde_json::to_string(&reply)?;
        tokio::select! {
            biased;
            _ = cancelled(&mut cancel) => return Ok(()),
            sent = tokio::time::timeout(timeout, socket.send(Message::Text(json.into()))) => {
                match sent {
                    Ok(Ok(())) => {},
                    Ok(Err(e)) => anyhow::bail!("websocket send failed: {e}"),
                    Err(_) => anyhow::bail!("websocket client stopped reading for {} seconds", timeout.as_secs()),
                }
            }
        }
    }
    let _ = tokio::time::timeout(timeout, socket.close()).await;
    Ok(())
}

enum SessionState {
    Awaiting,
    Ready { session: Session, text_buffer: String, stream_id: u32, encoder: Box<Encoder> },
}

async fn run_session(
    app: AppState,
    stream: &mut mpsc::Receiver<Message>,
    reply_tx: &mpsc::Sender<TtsReply>,
    cancel: &mut watch::Receiver<bool>,
) -> Result<()> {
    let mut sess: SessionState = SessionState::Awaiting;

    loop {
        let msg = tokio::select! {
            biased;
            _ = cancelled(cancel) => return Ok(()),
            _ = reply_tx.closed() => return Ok(()),
            msg = stream.recv() => match msg {
                Some(msg) => msg,
                None => break,
            },
            // Only waits between requests: generation and its audio run inside the arms below.
            _ = tokio::time::sleep(IDLE_TIMEOUT) => {
                let message = format!("no message for {} seconds; closing the session", IDLE_TIMEOUT.as_secs());
                tracing::info!("{message}");
                send_error(reply_tx, error_codes::TIMEOUT, message).await?;
                return Ok(());
            }
        };
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => return Ok(()),
            Message::Binary(_) | Message::Ping(_) | Message::Pong(_) => continue,
        };
        let req: TtsRequest = match serde_json::from_str(text.as_str()) {
            Ok(r) => r,
            Err(e) => {
                send_error(reply_tx, error_codes::BAD_REQUEST, format!("invalid request: {e}"))
                    .await?;
                continue;
            }
        };
        match (&mut sess, req) {
            (
                SessionState::Awaiting,
                TtsRequest::Setup { output_format, voice, voice_id, voice_emb, .. },
            ) => match handle_setup(&app, output_format, voice, voice_id, voice_emb, reply_tx)
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
                )
                .await?;
            }
            (SessionState::Ready { .. }, TtsRequest::Setup { .. }) => {
                send_error(
                    reply_tx,
                    error_codes::BAD_REQUEST,
                    "session already initialized".into(),
                )
                .await?;
            }
            (SessionState::Ready { text_buffer, .. }, TtsRequest::Text { text }) => {
                if !append_text(text_buffer, &text) {
                    send_error(
                        reply_tx,
                        error_codes::BAD_REQUEST,
                        format!("pending text exceeds the {MAX_INPUT_CHARS} character limit; flush first"),
                    ).await?;
                }
            }
            (
                SessionState::Ready { session, text_buffer, stream_id, encoder },
                TtsRequest::Flush { flush_id },
            ) => {
                flush_buffer(&app, session, text_buffer, stream_id, encoder, reply_tx, cancel)
                    .await?;
                let _ = reply_tx.send(TtsReply::Flushed { flush_id }).await;
            }
            (
                SessionState::Ready { session, text_buffer, stream_id, encoder },
                TtsRequest::EndOfStream,
            ) => {
                flush_buffer(&app, session, text_buffer, stream_id, encoder, reply_tx, cancel)
                    .await?;
                let _ = reply_tx.send(TtsReply::EndOfStream).await;
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
    reply_tx: &mpsc::Sender<TtsReply>,
    cancel: &mut watch::Receiver<bool>,
) -> Result<()> {
    if text_buffer.is_empty() {
        return Ok(());
    }
    let stream_id_now = *stream_id;
    *stream_id = stream_id.saturating_add(1);
    let text = std::mem::take(text_buffer);
    if let Err(e) =
        generate_one(app, session, &text, stream_id_now, encoder, reply_tx, cancel).await
    {
        tracing::warn!(error = %e, stream_id = stream_id_now, "generation failed");
        // Text the session cannot speak, such as one enormous word its KV budget
        // cannot hold even once cut, is the request's fault, not the server's.
        let code = match e.downcast_ref::<ptts::Error>() {
            Some(ptts::Error::SeqBudgetExceeded { .. } | ptts::Error::InvalidArgument(_)) => {
                error_codes::BAD_REQUEST
            }
            _ => error_codes::INTERNAL,
        };
        send_error(reply_tx, code, format!("generation failed: {e}")).await?;
    }
    Ok(())
}

async fn handle_setup(
    app: &AppState,
    output_format: String,
    voice: Option<String>,
    voice_id: Option<String>,
    voice_emb: Option<String>,
    reply_tx: &mpsc::Sender<TtsReply>,
) -> Result<Option<SessionState>> {
    if voice_emb.as_deref().is_some_and(|s| !s.is_empty()) {
        send_error(
            reply_tx,
            error_codes::NOT_IMPLEMENTED,
            "voice_emb prompts are not yet supported".into(),
        )
        .await?;
        return Ok(None);
    }
    let format = match output_format.parse::<Format>() {
        Ok(f) => f,
        Err(e) => {
            send_error(reply_tx, error_codes::BAD_REQUEST, format!("{e}")).await?;
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
            )
            .await?;
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
        send_error(reply_tx, error_codes::NOT_FOUND, format!("unknown voice '{voice_name}'"))
            .await?;
        return Ok(None);
    }
    tracing::info!(?voice_name, "starting new TTS session");
    // Conditioning on the voice happens once here, not per request: every
    // generation below clones this primed state.
    // Sentences are grouped up to the most a session speaks in one chunk, not
    // the usual few dozen tokens: every chunk boundary adds a pause, and a
    // request that fits in one chunk should be spoken as one.
    let opts = SpeechOptions::default()
        .voice(voice_name.clone())
        .max_tokens_per_chunk(ptts::plan::MAX_FIT_TOKENS);
    let session = match app.synth.session(&opts, app.max_seq_len) {
        Ok(session) => session,
        Err(e) => {
            send_error(reply_tx, error_codes::INTERNAL, format!("failed to prime voice: {e}"))
                .await?;
            return Ok(None);
        }
    };
    tracing::info!(?voice_name, "prompted voice embedding");
    let request_id = uuid::Uuid::new_v4().to_string();
    let ready = TtsReply::Ready {
        model_name: app.model_name.clone(),
        sample_rate: app.sample_rate,
        frame_size: app.frame_size,
        audio_stream_names: vec![],
        text_stream_names: vec![],
        request_id,
    };
    if reply_tx.send(ready).await.is_err() {
        anyhow::bail!("reply channel closed before ready");
    }
    if let Some(header) = encoder.header() {
        use base64::Engine;
        let audio = base64::engine::general_purpose::STANDARD.encode(header);
        let header_reply = TtsReply::Audio { audio, start_s: 0.0, stop_s: 0.0, stream_id: 0 };
        if reply_tx.send(header_reply).await.is_err() {
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
    reply_tx: &mpsc::Sender<TtsReply>,
    cancel: &mut watch::Receiver<bool>,
) -> Result<()> {
    // Normalization drops whole classes of characters, so a buffer that was
    // non-empty when it was flushed can be empty here: emoji or quotes on their
    // own. There is nothing to say, and empty text would come back to the
    // client as an INTERNAL error rather than as silence.
    if session.normalization().apply(text).trim().is_empty() {
        return Ok(());
    }
    let seed = app.seed_base ^ (stream_id as u64).wrapping_mul(0x9E3779B97F4A7C15);

    let (audio_tx, mut audio_rx) = mpsc::channel(AUDIO_QUEUE);
    // One stream id may contain several sentence chunks, each fitting this session's KV budget.
    let stream = session.stream_seeded(text, seed)?;
    let join = tokio::task::spawn_blocking(move || drain_audio(stream, audio_tx));
    let outcome =
        forward_audio(&mut audio_rx, app.frame_size as usize, stream_id, encoder, reply_tx, cancel)
            .await;
    // Close before joining: a slow socket can leave the producer blocked on its bounded queue.
    drop(audio_rx);
    let generated = join.await?;
    outcome.and(generated)
}

fn drain_audio(
    mut stream: impl Iterator<Item = ptts::Result<Vec<f32>>>,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<()> {
    while !tx.is_closed() {
        let Some(chunk) = stream.next() else { break };
        if tx.blocking_send(chunk?).is_err() {
            break;
        }
    }
    // Dropping SpeechStream joins both model workers before this task completes.
    Ok(())
}

async fn forward_audio(
    audio_rx: &mut mpsc::Receiver<Vec<f32>>,
    frame: usize,
    stream_id: u32,
    encoder: &mut Encoder,
    reply_tx: &mpsc::Sender<TtsReply>,
    cancel: &mut watch::Receiver<bool>,
) -> Result<()> {
    use base64::Engine;
    loop {
        let pcm = tokio::select! {
            biased;
            _ = cancelled(cancel) => return Ok(()),
            _ = reply_tx.closed() => return Ok(()),
            pcm = audio_rx.recv() => match pcm {
                Some(pcm) => pcm,
                None => return Ok(()),
            },
        };
        // A decoded chunk can contain several frames; resampled encoders need one at a time.
        for pcm in pcm.chunks(frame) {
            let encoded = encoder.encode(pcm)?;
            let audio = base64::engine::general_purpose::STANDARD.encode(&encoded.data);
            if reply_tx
                .send(TtsReply::Audio {
                    audio,
                    start_s: encoded.start_s,
                    stop_s: encoded.stop_s,
                    stream_id,
                })
                .await
                .is_err()
            {
                return Ok(());
            }
        }
    }
}

fn append_text(buffer: &mut String, text: &str) -> bool {
    if buffer.chars().count() + text.chars().count() > MAX_INPUT_CHARS {
        return false;
    }
    buffer.push_str(text);
    true
}

async fn send_error(tx: &mpsc::Sender<TtsReply>, code: u32, message: String) -> Result<()> {
    tx.send(TtsReply::Error { message, code })
        .await
        .map_err(|_| anyhow::anyhow!("reply channel closed"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite;

    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL pointing to a q8 checkpoint"]
    async fn session_limit_covers_idle_connections_and_releases_after_close() {
        use ptts::synth::{DeviceKind, Quant};
        let path = std::env::var("PTTS_TEST_MODEL").expect("set PTTS_TEST_MODEL");
        let app = crate::model::load_ptts(
            &path,
            None,
            None,
            DeviceKind::Cpu,
            Quant::Q80,
            0.3,
            7,
            1024,
            "en".parse().unwrap(),
            &[],
            std::num::NonZeroUsize::new(1).unwrap(),
        )
        .await
        .unwrap();
        let router = axum::Router::new()
            .route("/speech/tts", axum::routing::any(ws_handler))
            .with_state(app.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/speech/tts", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        let (mut first, response) = tokio_tungstenite::connect_async(&url).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::SWITCHING_PROTOCOLS);
        assert_eq!(app.requests.available_permits(), 0);
        let error = tokio_tungstenite::connect_async(&url).await.unwrap_err();
        let tungstenite::Error::Http(response) = error else {
            panic!("expected a busy HTTP response, got {error}");
        };
        assert_eq!(response.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
        let body = String::from_utf8(response.into_body().unwrap()).unwrap();
        assert!(body.contains("busy"));

        first
            .send(tungstenite::Message::Text(r#"{"type":"setup","output_format":"pcm"}"#.into()))
            .await
            .unwrap();
        let ready = first.next().await.unwrap().unwrap().into_text().unwrap();
        let ready: serde_json::Value = serde_json::from_str(&ready).unwrap();
        assert_eq!(ready["type"], "ready");
        assert_eq!(app.requests.available_permits(), 0);
        first.send(tungstenite::Message::Text(r#"{"type":"end_of_stream"}"#.into())).await.unwrap();
        let ended = first.next().await.unwrap().unwrap().into_text().unwrap();
        let ended: serde_json::Value = serde_json::from_str(&ended).unwrap();
        assert_eq!(ended["type"], "end_of_stream");
        let permit = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            app.requests.clone().acquire_owned(),
        )
        .await
        .expect("closed session leaked its slot")
        .unwrap();
        drop(permit);
        let (mut second, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
        second.close(None).await.unwrap();
        drop(second);
        let permit = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            app.requests.clone().acquire_owned(),
        )
        .await
        .expect("disconnect before setup leaked its slot")
        .unwrap();
        drop(permit);
        server.abort();
    }
}
