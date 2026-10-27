//! The OpenAI-compatible speech API: `POST /v1/audio/speech`, and the `GET /v1/models`,
//! `GET /v1/audio/voices` and `GET /health` that clients probe. Anything with an "OpenAI TTS
//! with a custom base URL" setting can use this server unchanged.

use crate::compat::{Aliases, OPENAI_VOICES};
use crate::encoder::{Encoder, Format};
use crate::model::AppState;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use ptts::synth::SpeechOptions;

/// OpenAI's limit on `input`.
const MAX_INPUT_CHARS: usize = 4096;

/// How long a client may stop reading before its response is ended and its slot freed.
const SEND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// The request body OpenAI specifies. `model`, `instructions` and `stream_format` are accepted
/// and ignored: there is one model, it takes no style prompt, and the reply is always raw audio.
#[derive(serde::Deserialize)]
struct SpeechRequest {
    input: String,
    #[serde(default)]
    voice: Option<Voice>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    speed: Option<f64>,
}

/// A voice name, or OpenAI's `{"id": ...}` form for a custom voice.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum Voice {
    Name(String),
    Custom { id: String },
}

/// The `speed` range OpenAI accepts.
const MIN_SPEED: f64 = 0.25;
const MAX_SPEED: f64 = 4.0;

/// OpenAI's `pcm`: headerless 16-bit mono at this rate, which clients assume.
const PCM_RATE: u32 = 24000;

/// A request that passed every check, ready to synthesize.
#[derive(Debug, PartialEq)]
struct Checked {
    input: String,
    /// `None` for the checkpoint's default voice.
    voice: Option<String>,
    format: Format,
    content_type: &'static str,
}

/// Why a request is a 400: the field at fault, if one is, and what to say.
type Rejection = (Option<&'static str>, String);

/// Check a request body against what this server can do: `voices` are the registered ones,
/// `aliases` other names for them, and `sample_rate` is the checkpoint's. With `ignore_speed`, a
/// `speed` in OpenAI's range is accepted and the text spoken at the normal rate.
fn check(
    body: &[u8],
    voices: &[String],
    aliases: &Aliases,
    ignore_speed: bool,
    sample_rate: u32,
) -> Result<Checked, Rejection> {
    let req: SpeechRequest =
        serde_json::from_slice(body).map_err(|e| (None, format!("invalid request: {e}")))?;
    if req.input.trim().is_empty() {
        return Err((Some("input"), "input is empty".into()));
    }
    let chars = req.input.chars().count();
    if chars > MAX_INPUT_CHARS {
        let message = format!("input is {chars} characters; the limit is {MAX_INPUT_CHARS}");
        return Err((Some("input"), message));
    }
    if let Some(speed) = req.speed {
        if !(MIN_SPEED..=MAX_SPEED).contains(&speed) {
            let message =
                format!("speed {speed} is outside OpenAI's range, {MIN_SPEED} to {MAX_SPEED}");
            return Err((Some("speed"), message));
        }
        if speed != 1.0 && !ignore_speed {
            let message = format!(
                "speed {speed} is not supported: this model has no rate control yet; omit speed, \
                 send 1.0, or start the server with --ignore-speed to speak at the normal rate"
            );
            return Err((Some("speed"), message));
        }
    }
    let format = req.response_format.as_deref().unwrap_or("mp3").to_ascii_lowercase();
    let (format, content_type) = match format.as_str() {
        "mp3" => (Format::Mp3, "audio/mpeg"),
        "opus" => (Format::Opus, "audio/opus"),
        "wav" => (Format::Wav, "audio/wav"),
        "pcm" if sample_rate == PCM_RATE => (Format::Pcm, "audio/pcm"),
        "pcm" => {
            let message = format!(
                "pcm is {PCM_RATE} Hz and this checkpoint speaks at {sample_rate} Hz; use wav"
            );
            return Err((Some("response_format"), message));
        }
        other => {
            let message =
                format!("response_format '{other}' is not supported; use mp3, opus, wav or pcm");
            return Err((Some("response_format"), message));
        }
    };
    let voice = match req.voice {
        None => None,
        Some(Voice::Name(name) | Voice::Custom { id: name }) => {
            if let Some(voice) = voices.iter().find(|v| v.eq_ignore_ascii_case(&name)) {
                Some(voice.clone())
            } else if let Some(target) = aliases.get(&name) {
                target.clone()
            } else if name.eq_ignore_ascii_case("default") {
                // `default` means the default voice, as on `ptts-ws-server`.
                None
            } else {
                let mut message =
                    format!("unknown voice '{name}'; this server has {}", voices.join(", "));
                if OPENAI_VOICES.iter().any(|v| v.eq_ignore_ascii_case(&name)) {
                    message += ". OpenAI's voice names work once the server is started with \
                                --openai-voices or --voice-alias";
                }
                return Err((Some("voice"), message));
            }
        }
    };
    Ok(Checked { input: req.input, voice, format, content_type })
}

pub async fn speech(State(app): State<AppState>, body: Bytes) -> Response {
    let Checked { input, voice, format, content_type } =
        match check(&body, &app.voices, &app.aliases, app.ignore_speed, app.sample_rate) {
            Ok(checked) => checked,
            Err((param, message)) => return error(StatusCode::BAD_REQUEST, param, message),
        };
    let permit = match app.requests.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => return busy(),
    };
    let mut opts = SpeechOptions::default().seed(next_seed(app.seed_base));
    if let Some(voice) = voice {
        opts = opts.voice(voice);
    }

    // Synthesis and encoding share one blocking thread. The response starts only once the
    // stream has, so a request the model refuses gets an error, not a 200 with no audio.
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<std::io::Result<Vec<u8>>>(16);
    let frame = app.frame_size as usize;
    let sample_rate = app.sample_rate;
    tokio::task::spawn_blocking(move || {
        // Keep the slot until encoding ends and SpeechStream joins its workers,
        // including when the client disconnects or model startup fails.
        let _permit = permit;
        if started_tx.is_closed() || tx.is_closed() {
            return;
        }
        let stream = match app.synth.stream_with(&input, &opts) {
            Ok(stream) => stream,
            Err(e) => {
                let _ = started_tx.send(Err(e));
                return;
            }
        };
        if started_tx.send(Ok(())).is_err() {
            // The handler was cancelled during setup. Dropping the stream joins its workers.
            return;
        }
        if let Err(e) = encode(stream, format, frame, sample_rate, &tx, SEND_TIMEOUT) {
            tracing::warn!(error = %e, "speech request failed after it started");
            send_within(&tx, Err(std::io::Error::other(e.to_string())), SEND_TIMEOUT);
        }
    });
    match started_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            let status = match e {
                ptts::Error::InvalidArgument(_) | ptts::Error::UnknownVoice { .. } => {
                    StatusCode::BAD_REQUEST
                }
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            return error(status, None, e.to_string());
        }
        Err(_) => {
            let message = "the generation task ended before it started";
            return error(StatusCode::INTERNAL_SERVER_ERROR, None, message);
        }
    }
    let body = futures_util::stream::poll_fn(move |cx| rx.poll_recv(cx));
    ([(header::CONTENT_TYPE, content_type)], Body::from_stream(body)).into_response()
}

/// Encode `stream` frame by frame onto `tx`. Returns early, without error, once the client has
/// gone or stopped reading for `timeout`: dropping `stream` stops the generation workers.
fn encode(
    stream: impl Iterator<Item = ptts::Result<Vec<f32>>>,
    format: Format,
    frame: usize,
    sample_rate: u32,
    tx: &tokio::sync::mpsc::Sender<std::io::Result<Vec<u8>>>,
    timeout: std::time::Duration,
) -> anyhow::Result<()> {
    let send = |bytes: Vec<u8>| bytes.is_empty() || send_within(tx, Ok(bytes), timeout);
    let mut encoder = Encoder::new(format, sample_rate)?;
    if !send(encoder.header()?) {
        return Ok(());
    }
    // A chunk can carry several frames, and the Opus encoder wants one at a time.
    let mut stream = stream;
    while !tx.is_closed() {
        let Some(chunk) = stream.next() else { break };
        for pcm in chunk?.chunks(frame) {
            if !send(encoder.encode(pcm)?) {
                return Ok(());
            }
        }
    }
    send(encoder.finish()?);
    Ok(())
}

/// Send from a blocking thread, giving up if the client has not read for `timeout`. Without the
/// limit, a client that stops reading holds its generation slot for as long as it stays connected.
fn send_within(
    tx: &tokio::sync::mpsc::Sender<std::io::Result<Vec<u8>>>,
    item: std::io::Result<Vec<u8>>,
    timeout: std::time::Duration,
) -> bool {
    let handle = tokio::runtime::Handle::current();
    match handle.block_on(tokio::time::timeout(timeout, tx.send(item))) {
        Ok(sent) => sent.is_ok(),
        Err(_) => {
            tracing::warn!(
                "client stopped reading for {}s; ending the response",
                timeout.as_secs()
            );
            false
        }
    }
}

/// A different seed per request, as `ptts-ws-server` gives each stream.
pub fn next_seed(base: u64) -> u64 {
    static REQUESTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = REQUESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    base ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

pub async fn models(State(app): State<AppState>) -> Response {
    let model = serde_json::json!({
        "id": app.model_name, "object": "model", "created": 0, "owned_by": "gradium",
        "languages": app.synth.languages(),
    });
    json(StatusCode::OK, serde_json::json!({ "object": "list", "data": [model] }))
}

/// The registered voices, as `{"voices": [...]}`, and the aliases set for them, as
/// `"aliases": {"alias": "voice"}`. Aliases are not in `voices`, so a client that offers that
/// list offers only real voices.
pub async fn voices(State(app): State<AppState>) -> Response {
    let body = serde_json::json!({ "voices": app.voices, "aliases": app.aliases.to_json() });
    json(StatusCode::OK, body)
}

pub async fn health() -> Response {
    json(StatusCode::OK, serde_json::json!({ "status": "ok" }))
}

/// An error in OpenAI's shape, which client SDKs parse into their own error types.
fn error(status: StatusCode, param: Option<&str>, message: impl Into<String>) -> Response {
    let kind = if status.is_client_error() { "invalid_request_error" } else { "server_error" };
    let error = serde_json::json!({
        "message": message.into(), "type": kind, "param": param, "code": null,
    });
    json(status, serde_json::json!({ "error": error }))
}

fn busy() -> Response {
    json(
        StatusCode::TOO_MANY_REQUESTS,
        serde_json::json!({ "error": {
            "message": "the server is busy; retry after an active speech request finishes",
            "type": "server_error", "param": null, "code": "server_busy",
        }}),
    )
}

fn json(status: StatusCode, value: serde_json::Value) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], value.to_string()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::Compat;

    fn voices(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    fn checked(body: &str) -> Checked {
        check(body.as_bytes(), &voices(&["Freya", "Toby"]), &Aliases::default(), false, PCM_RATE)
            .unwrap()
    }

    fn rejected(body: &str) -> Option<&'static str> {
        check(body.as_bytes(), &voices(&["Freya", "Toby"]), &Aliases::default(), false, PCM_RATE)
            .unwrap_err()
            .0
    }

    /// A server started with `--openai-voices default`, `--voice-alias narrator=Toby` and
    /// `--ignore-speed`.
    fn lenient(body: &str) -> Result<Checked, Rejection> {
        let voices = voices(&["Freya", "Toby"]);
        let compat = Compat {
            voice_aliases: vec!["narrator=Toby".into()],
            openai_voices: Some("default".into()),
            ignore_speed: true,
        };
        let aliases = Aliases::new(&compat, &voices).unwrap();
        check(body.as_bytes(), &voices, &aliases, true, PCM_RATE)
    }

    #[tokio::test]
    async fn busy_errors_have_the_openai_shape() {
        let response = busy();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "server_busy");
        assert_eq!(body["error"]["type"], "server_error");
        assert!(body["error"]["message"].as_str().unwrap().contains("retry"));
        assert!(body["error"]["param"].is_null());
    }

    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL pointing to a q8 checkpoint"]
    async fn admission_rejects_overload_and_recovers_after_disconnect_and_failure() {
        use ptts::synth::{DeviceKind, Quant};
        use std::future::Future;
        let path = std::env::var("PTTS_TEST_MODEL").expect("set PTTS_TEST_MODEL");
        let app = crate::model::load_ptts(
            &path,
            None,
            None,
            DeviceKind::Cpu,
            Quant::Q80,
            0.3,
            7,
            "en".parse().unwrap(),
            &[],
            std::num::NonZeroUsize::new(1).unwrap(),
            &Default::default(),
        )
        .await
        .unwrap();
        let request = |input: &str| {
            Bytes::from(serde_json::json!({ "input": input, "response_format": "pcm" }).to_string())
        };
        let permit = app.requests.clone().try_acquire_owned().unwrap();
        assert_eq!(
            speech(State(app.clone()), request("Hello.")).await.status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            speech(State(app.clone()), Bytes::from_static(b"invalid")).await.status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(health().await.status(), StatusCode::OK);
        assert_eq!(models(State(app.clone())).await.status(), StatusCode::OK);
        assert_eq!(super::voices(State(app.clone())).await.status(), StatusCode::OK);
        drop(permit);

        // Poll only through task startup, then drop the request before response headers.
        let mut pending = Box::pin(speech(State(app.clone()), request("Hello before headers.")));
        std::future::poll_fn(|cx| {
            assert!(pending.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        assert_eq!(app.requests.available_permits(), 0);
        drop(pending);
        let permit = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            app.requests.clone().acquire_owned(),
        )
        .await
        .expect("request cancelled before headers leaked its slot")
        .unwrap();
        drop(permit);

        // Hold an unread response so the producer fills its bounded output channel.
        let response =
            speech(State(app.clone()), request(&"Long speech keeps this slot busy. ".repeat(100)))
                .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            speech(State(app.clone()), request("Hello.")).await.status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(response);
        let permit = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            app.requests.clone().acquire_owned(),
        )
        .await
        .expect("generation did not stop after dropping the response")
        .unwrap();
        drop(permit);

        // Normalization removes an emoji-only input after admission, during startup.
        assert_eq!(
            speech(State(app.clone()), request("😀")).await.status(),
            StatusCode::BAD_REQUEST
        );
        let permit = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            app.requests.clone().acquire_owned(),
        )
        .await
        .expect("failed startup leaked its slot")
        .unwrap();
        drop(permit);
        let response = speech(State(app.clone()), request("Hello from Phonon.")).await;
        assert_eq!(response.status(), StatusCode::OK);
        let audio = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024).await.unwrap();
        assert!(!audio.is_empty());
    }

    #[tokio::test]
    async fn a_slow_http_reader_pauses_encoding_and_drop_releases_the_stream() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::time::Duration;

        struct EndlessAudio(Arc<AtomicUsize>, Arc<AtomicBool>, Arc<tokio::sync::Notify>);
        impl Iterator for EndlessAudio {
            type Item = ptts::Result<Vec<f32>>;
            fn next(&mut self) -> Option<Self::Item> {
                if self.0.fetch_add(1, Ordering::SeqCst) == 16 {
                    self.2.notify_one();
                }
                Some(Ok(vec![0.1; 480]))
            }
        }
        impl Drop for EndlessAudio {
            fn drop(&mut self) {
                self.1.store(true, Ordering::SeqCst);
            }
        }
        let produced = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        let full = Arc::new(tokio::sync::Notify::new());
        let stream = EndlessAudio(produced.clone(), dropped.clone(), full.clone());
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let producer = tokio::task::spawn_blocking(move || {
            encode(stream, Format::Pcm, 480, PCM_RATE, &tx, SEND_TIMEOUT)
        });
        tokio::time::timeout(Duration::from_secs(2), full.notified()).await.unwrap();
        // Sixteen queued frames plus the one blocked on sending.
        assert_eq!(produced.load(Ordering::SeqCst), 17);
        drop(rx);
        tokio::time::timeout(Duration::from_secs(2), producer).await.unwrap().unwrap().unwrap();
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_client_that_stops_reading_is_given_up_on() {
        use std::time::Duration;

        let stream = std::iter::repeat_with(|| Ok(vec![0.1; 480]));
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let producer = tokio::task::spawn_blocking(move || {
            encode(stream, Format::Pcm, 480, PCM_RATE, &tx, Duration::from_millis(100))
        });
        // `rx` stays open and unread, as with a connected client that stopped reading.
        tokio::time::timeout(Duration::from_secs(5), producer).await.unwrap().unwrap().unwrap();
        drop(rx);
    }

    #[test]
    fn a_bare_request_is_mp3_in_the_default_voice() {
        let got = checked(r#"{"model": "tts-1", "input": "Hello."}"#);
        assert_eq!(got.format, Format::Mp3);
        assert_eq!(got.content_type, "audio/mpeg");
        assert_eq!(got.voice, None);
    }

    #[test]
    fn each_format_has_its_encoder_and_type() {
        for (name, format, content_type) in [
            ("mp3", Format::Mp3, "audio/mpeg"),
            ("opus", Format::Opus, "audio/opus"),
            ("wav", Format::Wav, "audio/wav"),
            ("pcm", Format::Pcm, "audio/pcm"),
        ] {
            let got = checked(&format!(r#"{{"input": "Hi.", "response_format": "{name}"}}"#));
            assert_eq!((got.format, got.content_type), (format, content_type), "{name}");
        }
        assert_eq!(checked(r#"{"input": "Hi.", "response_format": "WAV"}"#).format, Format::Wav);
        assert_eq!(
            rejected(r#"{"input": "Hi.", "response_format": "aac"}"#),
            Some("response_format")
        );
        // OpenAI's pcm is 24 kHz, and a checkpoint at another rate is not resampled.
        let pcm = br#"{"input": "Hi.", "response_format": "pcm"}"#;
        let at_16k = check(pcm, &[], &Aliases::default(), false, 16000);
        assert_eq!(at_16k.unwrap_err().0, Some("response_format"));
    }

    #[test]
    fn voices_match_in_any_case_and_unknown_ones_are_rejected() {
        assert_eq!(
            checked(r#"{"input": "Hi.", "voice": "freya"}"#).voice.as_deref(),
            Some("Freya")
        );
        assert_eq!(
            checked(r#"{"input": "Hi.", "voice": {"id": "Toby"}}"#).voice.as_deref(),
            Some("Toby")
        );
        assert_eq!(checked(r#"{"input": "Hi.", "voice": "default"}"#).voice, None);
        assert_eq!(rejected(r#"{"input": "Hi.", "voice": "alloy"}"#), Some("voice"));
        // A voice registered as `default` is that voice, not an alias.
        let with_default = check(
            br#"{"input": "Hi.", "voice": "DEFAULT"}"#,
            &voices(&["default"]),
            &Aliases::default(),
            false,
            PCM_RATE,
        );
        assert_eq!(with_default.unwrap().voice.as_deref(), Some("default"));
    }

    #[test]
    fn requests_the_model_cannot_serve_are_rejected() {
        assert_eq!(checked(r#"{"input": "Hi.", "speed": 1.0}"#).input, "Hi.");
        assert_eq!(rejected(r#"{"input": "Hi.", "speed": 1.5}"#), Some("speed"));
        assert_eq!(rejected(r#"{"input": "Hi.", "speed": 4.5}"#), Some("speed"));
        assert_eq!(rejected(r#"{"input": "  "}"#), Some("input"));
        let long = format!(r#"{{"input": "{}"}}"#, "a".repeat(MAX_INPUT_CHARS + 1));
        assert_eq!(rejected(&long), Some("input"));
        assert_eq!(rejected(r#"{"voice": "Freya"}"#), None, "no input at all");
    }

    #[test]
    fn aliases_stand_for_their_voices_when_set() {
        assert_eq!(lenient(r#"{"input": "Hi.", "voice": "alloy"}"#).unwrap().voice, None);
        assert_eq!(lenient(r#"{"input": "Hi.", "voice": {"id": "Marin"}}"#).unwrap().voice, None);
        let narrator = lenient(r#"{"input": "Hi.", "voice": "NARRATOR"}"#).unwrap();
        assert_eq!(narrator.voice.as_deref(), Some("Toby"));
        let freya = lenient(r#"{"input": "Hi.", "voice": "freya"}"#).unwrap();
        assert_eq!(freya.voice.as_deref(), Some("Freya"));
        // Aliases are explicit: a name that is not one is still unknown.
        let unknown = lenient(r#"{"input": "Hi.", "voice": "someone"}"#).unwrap_err();
        assert_eq!(unknown.0, Some("voice"));
    }

    #[test]
    fn ignore_speed_accepts_openai_speeds_and_nothing_else() {
        for speed in ["0.25", "1.25", "4"] {
            let body = format!(r#"{{"input": "Hi.", "speed": {speed}}}"#);
            assert_eq!(lenient(&body).unwrap().input, "Hi.", "{speed}");
        }
        for speed in ["0.2", "4.5", "-1"] {
            let body = format!(r#"{{"input": "Hi.", "speed": {speed}}}"#);
            assert_eq!(lenient(&body).unwrap_err().0, Some("speed"), "{speed}");
        }
    }
}
