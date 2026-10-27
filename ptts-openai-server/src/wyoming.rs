//! The Wyoming protocol, which Home Assistant uses to talk to local voice services. A connection
//! carries events, each a JSON header line followed by optional JSON data and a binary payload.
//! This serves the text-to-speech part: `describe` lists the voices, `synthesize` speaks a text,
//! and `synthesize-start`, `-chunk` and `-stop` speak a reply a sentence at a time while it is
//! still being written.
//!
//! A request names a voice but never a language, so every voice is listed in the `--lang`
//! language. Requests share the HTTP API's generation slots, but wait for one instead of being
//! turned away.

use crate::api::next_seed;
use crate::model::AppState;
use ptts::synth::SpeechOptions;
use serde_json::{Value, json};
use std::io;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::Semaphore;

/// Sent in every header, as the Python library sends its own version. Clients do not check it.
const VERSION: &str = "1.7.0";

/// The most a header line, an event's JSON data and its payload may take. A request is some
/// text and a voice name, so these are far above what a real client sends.
const MAX_HEADER: usize = 64 * 1024;
const MAX_DATA: usize = 1 << 20;
const MAX_PAYLOAD: usize = 16 << 20;

/// How long a client may stop reading before its connection is closed and its slot freed.
const SEND_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a connection may send nothing before it is closed. Home Assistant opens one per
/// request, so this only ends abandoned ones.
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// Streamed text is spoken once a sentence ends at least this many characters in, so that an
/// abbreviation such as "Dr." is not spoken on its own; or once this much has arrived without one.
const MIN_PIECE_CHARS: usize = 20;
const MAX_PENDING_BYTES: usize = 4096;

/// Generates speech. A trait so the protocol can be tested without a checkpoint.
pub trait Speaker: Send + Sync + 'static {
    fn speak(&self, text: &str, opts: &SpeechOptions) -> ptts::Result<Pcm>;
}

/// The audio of one request, a chunk at a time.
pub type Pcm = Box<dyn Iterator<Item = ptts::Result<Vec<f32>>>>;

impl Speaker for AppState {
    fn speak(&self, text: &str, opts: &SpeechOptions) -> ptts::Result<Pcm> {
        Ok(Box::new(self.synth.stream_with(text, opts)?))
    }
}

pub struct Server<S> {
    speaker: S,
    voices: Vec<String>,
    /// The encoded `info` event.
    info: Vec<u8>,
    requests: Arc<Semaphore>,
    seed_base: u64,
    sample_rate: u32,
}

impl Server<AppState> {
    pub fn new(app: &AppState) -> anyhow::Result<Self> {
        let lang = app.normalize.as_str();
        anyhow::ensure!(
            lang != "none",
            "Wyoming lists each voice with its language: set --lang to the one the checkpoint speaks"
        );
        let attribution = json!({ "name": app.model_name, "url": app.model_url });
        let voices: Vec<_> = (app.voices.iter())
            .map(|v| {
                json!({
                    "name": v, "description": v, "attribution": attribution, "installed": true,
                    "languages": [lang],
                })
            })
            .collect();
        let program = json!({
            "name": "Phonon",
            "description": "Phonon text to speech",
            "attribution": { "name": "Gradium", "url": env!("CARGO_PKG_REPOSITORY") },
            "installed": true,
            "version": env!("CARGO_PKG_VERSION"),
            "voices": voices,
            "supports_synthesize_streaming": true,
        });
        Ok(Self {
            speaker: app.clone(),
            voices: app.voices.clone(),
            info: encode("info", &json!({ "tts": [program] }), &[]),
            requests: app.requests.clone(),
            seed_base: app.seed_base,
            sample_rate: app.sample_rate,
        })
    }
}

impl<S> Server<S> {
    /// The voice a request's `voice` names, in any case, or `None` for the default.
    fn voice(&self, voice: Option<&Value>) -> Result<Option<String>, String> {
        let Some(name) = voice.and_then(|v| v.get("name")).and_then(Value::as_str) else {
            return Ok(None);
        };
        match self.voices.iter().find(|v| v.eq_ignore_ascii_case(name)) {
            Some(voice) => Ok(Some(voice.clone())),
            // `default` means the default voice, as on the HTTP API.
            None if name.eq_ignore_ascii_case("default") => Ok(None),
            None => {
                Err(format!("unknown voice '{name}'; this server has {}", self.voices.join(", ")))
            }
        }
    }
}

/// Accept connections on `listener` until the process ends.
pub async fn serve<S: Speaker>(listener: tokio::net::TcpListener, server: Arc<Server<S>>) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let _ = stream.set_nodelay(true);
                let (read, write) = stream.into_split();
                let server = server.clone();
                tokio::spawn(async move {
                    let read = tokio::io::BufReader::new(read);
                    if let Err(e) = connection(read, write, server).await {
                        tracing::debug!(error = %e, "wyoming connection ended");
                    }
                });
            }
            Err(e) => {
                // Out of file descriptors, most likely. Wait rather than spin.
                tracing::warn!(error = %e, "cannot accept a wyoming connection");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

/// Text in progress between `synthesize-start` and `synthesize-stop`.
struct Stream {
    /// An unknown voice is `Err`: the error has been sent, and the rest of the stream is ignored.
    voice: Result<Option<String>, String>,
    pending: String,
    /// Whether `audio-start` has been sent.
    started: bool,
}

/// Serve one connection. Events are handled in order, so while a sentence is spoken, the text
/// after it waits in the socket.
async fn connection<R, W, S>(
    mut read: R,
    mut write: W,
    server: Arc<Server<S>>,
) -> anyhow::Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
    S: Speaker,
{
    let mut stream: Option<Stream> = None;
    while let Some(event) = tokio::time::timeout(IDLE_TIMEOUT, read_event(&mut read)).await?? {
        let text = event.data.get("text").and_then(Value::as_str).unwrap_or_default().to_string();
        let voice = server.voice(event.data.get("voice"));
        match event.kind.as_str() {
            "describe" => send(&mut write, &server.info).await?,
            "ping" => send(&mut write, &encode("pong", &json!({ "text": text }), &[])).await?,
            // Inside a stream, Home Assistant also sends the whole text this way, for servers
            // that predate streaming. It is spoken already.
            "synthesize" if stream.is_some() => {}
            "synthesize" => {
                let mut started = false;
                match voice {
                    Ok(voice) => {
                        speak(&server, &mut write, text, voice, false, &mut started).await?
                    }
                    Err(message) => send(&mut write, &error(&message)).await?,
                }
                // A client may wait for `audio-stop`, so it ends every request, failed or not.
                if !started {
                    send(&mut write, &audio_start(server.sample_rate)).await?;
                }
                send(&mut write, &encode("audio-stop", &json!({}), &[])).await?;
            }
            "synthesize-start" => {
                if let Err(message) = &voice {
                    send(&mut write, &error(message)).await?;
                }
                stream = Some(Stream { voice, pending: String::new(), started: false });
            }
            "synthesize-chunk" => {
                if let Some(Stream { voice: Ok(voice), pending, started }) = &mut stream {
                    pending.push_str(&text);
                    if let Some(piece) = take_sentences(pending) {
                        speak(&server, &mut write, piece, voice.clone(), true, started).await?;
                    }
                }
            }
            "synthesize-stop" => {
                if let Some(Stream { voice: Ok(voice), pending, mut started }) = stream.take() {
                    speak(&server, &mut write, pending, voice, true, &mut started).await?;
                    if started {
                        send(&mut write, &encode("audio-stop", &json!({}), &[])).await?;
                    }
                }
                send(&mut write, &encode("synthesize-stopped", &json!({}), &[])).await?;
            }
            other => tracing::debug!(event = other, "ignoring a wyoming event"),
        }
    }
    Ok(())
}

/// Speak `text`, sending `audio-start` before the first audio unless `started` says it was sent.
/// A streamed piece with nothing left to say once normalized, such as an emoji on its own, is
/// skipped rather than failing the whole reply.
async fn speak<W: AsyncWrite + Unpin, S: Speaker>(
    server: &Arc<Server<S>>,
    write: &mut W,
    text: String,
    voice: Option<String>,
    streamed: bool,
    started: &mut bool,
) -> io::Result<()> {
    if streamed && text.trim().is_empty() {
        return Ok(());
    }
    let permit = server.requests.clone().acquire_owned().await.expect("the semaphore stays open");
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let generator = server.clone();
    tokio::task::spawn_blocking(move || {
        // Released once the stream below has stopped.
        let _permit = permit;
        let mut opts = SpeechOptions::default().seed(next_seed(generator.seed_base));
        if let Some(voice) = voice {
            opts = opts.voice(voice);
        }
        let pcm = match generator.speaker.speak(&text, &opts) {
            Ok(pcm) => pcm,
            Err(e) => {
                let _ = tx.blocking_send(Err(e));
                return;
            }
        };
        for chunk in pcm {
            let failed = chunk.is_err();
            // A closed channel means the client is gone: dropping `pcm` stops the generation.
            if tx.blocking_send(chunk).is_err() || failed {
                return;
            }
        }
    });
    while let Some(chunk) = rx.recv().await {
        match chunk {
            Ok(samples) => {
                if !*started {
                    send(write, &audio_start(server.sample_rate)).await?;
                    *started = true;
                }
                let mut pcm = Vec::with_capacity(samples.len() * 2);
                ptts::wav::write_samples(&mut pcm, &samples)?;
                send(write, &encode("audio-chunk", &audio_format(server.sample_rate), &pcm))
                    .await?;
            }
            Err(ptts::Error::InvalidArgument(e)) if streamed => {
                tracing::warn!(error = %e, "skipping a piece of streamed text");
            }
            Err(e) => send(write, &error(&e.to_string())).await?,
        }
    }
    Ok(())
}

/// Take the streamed text up to its last sentence end off `pending`, if that is at least
/// [`MIN_PIECE_CHARS`] in; or all of it, past [`MAX_PENDING_BYTES`]. A sentence ends at a line
/// break, or at `.`, `!`, `?` or `…` followed by a space. A final mark with nothing after it
/// waits for the next chunk: "3." may be the start of "3.5".
fn take_sentences(pending: &mut String) -> Option<String> {
    const ENDS: &[char] = &['.', '!', '?', '…', '。', '！', '？'];
    let ends = pending.char_indices().zip(pending.chars().skip(1)).filter_map(|((i, c), next)| {
        (c == '\n' || ENDS.contains(&c) && next.is_whitespace()).then_some(i + c.len_utf8())
    });
    let end = ends
        .filter(|&end| pending[..end].trim().chars().count() >= MIN_PIECE_CHARS)
        .last()
        .or((pending.len() > MAX_PENDING_BYTES).then_some(pending.len()))?;
    let piece = pending[..end].trim().to_string();
    pending.replace_range(..end, "");
    Some(piece)
}

/// Write one event. A client that has not read for [`SEND_TIMEOUT`] is given up on, which ends
/// its connection and frees its slot.
async fn send<W: AsyncWrite + Unpin>(write: &mut W, event: &[u8]) -> io::Result<()> {
    let written = tokio::time::timeout(SEND_TIMEOUT, write.write_all(event)).await;
    written.map_err(|_| io::Error::from(io::ErrorKind::TimedOut))?
}

/// One event received. Its `data` is always an object.
#[derive(Debug, PartialEq)]
struct Event {
    kind: String,
    data: Value,
}

/// Read the next event, or `None` at the end of the connection. Payloads are read and dropped:
/// no event this server handles carries one.
async fn read_event<R: AsyncBufRead + Unpin>(read: &mut R) -> io::Result<Option<Event>> {
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
    let mut line = Vec::new();
    while line.iter().all(u8::is_ascii_whitespace) {
        line.clear();
        if (&mut *read).take(MAX_HEADER as u64 + 1).read_until(b'\n', &mut line).await? == 0 {
            return Ok(None);
        }
        if line.len() > MAX_HEADER {
            return Err(invalid("an event header is too long"));
        }
    }
    let header: Value = serde_json::from_slice(&line).map_err(|_| invalid("bad event header"))?;
    let kind = header["type"].as_str().ok_or_else(|| invalid("an event has no type"))?;
    // Data can be in the header itself, and after it.
    let mut data = header["data"].as_object().cloned().unwrap_or_default();
    let length = |key: &str| header[key].as_u64().unwrap_or(0) as usize;
    let (data_length, payload_length) = (length("data_length"), length("payload_length"));
    if data_length > MAX_DATA || payload_length > MAX_PAYLOAD {
        return Err(invalid("an event is too long"));
    }
    if data_length > 0 {
        let mut bytes = vec![0; data_length];
        read.read_exact(&mut bytes).await?;
        let more =
            serde_json::from_slice::<Value>(&bytes).map_err(|_| invalid("bad event data"))?;
        data.extend(more.as_object().cloned().ok_or_else(|| invalid("bad event data"))?);
    }
    read.read_exact(&mut vec![0; payload_length]).await?;
    Ok(Some(Event { kind: kind.to_string(), data: Value::Object(data) }))
}

/// An event, with its data left out when empty, as the Python library does.
fn encode(kind: &str, data: &Value, payload: &[u8]) -> Vec<u8> {
    let mut header = json!({ "type": kind, "version": VERSION });
    let data = match data.as_object() {
        Some(map) if !map.is_empty() => data.to_string().into_bytes(),
        _ => Vec::new(),
    };
    if !data.is_empty() {
        header["data_length"] = data.len().into();
    }
    if !payload.is_empty() {
        header["payload_length"] = payload.len().into();
    }
    let mut event = header.to_string().into_bytes();
    event.push(b'\n');
    event.extend(data);
    event.extend(payload);
    event
}

/// 16-bit mono, which is what Home Assistant writes into its WAV files.
fn audio_format(sample_rate: u32) -> Value {
    json!({ "rate": sample_rate, "width": 2, "channels": 1 })
}

fn audio_start(sample_rate: u32) -> Vec<u8> {
    encode("audio-start", &audio_format(sample_rate), &[])
}

fn error(text: &str) -> Vec<u8> {
    encode("error", &json!({ "text": text }), &[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[tokio::test]
    async fn events_round_trip_and_payloads_are_skipped() {
        let mut bytes = encode("audio-chunk", &json!({ "rate": 24000 }), &[1, 2, 3]);
        bytes.extend(encode("synthesize", &json!({ "text": "Hi ✓" }), &[]));
        // Data in the header itself, merged with the data after it, and a blank line before.
        bytes.extend(
            b"\n{\"type\": \"ping\", \"data\": {\"a\": 1}, \"data_length\": 8}\n{\"b\": 2}",
        );
        let mut read = &bytes[..];
        let chunk = read_event(&mut read).await.unwrap().unwrap();
        assert_eq!(chunk, Event { kind: "audio-chunk".into(), data: json!({ "rate": 24000 }) });
        assert_eq!(read_event(&mut read).await.unwrap().unwrap().data["text"], "Hi ✓");
        assert_eq!(read_event(&mut read).await.unwrap().unwrap().data, json!({ "a": 1, "b": 2 }));
        assert_eq!(read_event(&mut read).await.unwrap(), None);

        assert_eq!(
            encode("audio-stop", &json!({}), &[]),
            b"{\"type\":\"audio-stop\",\"version\":\"1.7.0\"}\n"
        );
        let long = [b"{\"type\": \"", &vec![b'a'; MAX_HEADER][..], b"\"}\n"].concat();
        assert!(read_event(&mut &long[..]).await.is_err());
    }

    #[test]
    fn streamed_text_is_cut_at_sentence_ends() {
        let mut pending = String::new();
        let mut pieces = vec![];
        let chunks =
            ["Sure. Dr. Smith is ", "in at 3.", "5 today! Turning on the", " lights now.", " Done"];
        for chunk in chunks {
            pending.push_str(chunk);
            pieces.extend(take_sentences(&mut pending));
        }
        assert_eq!(pieces, ["Sure. Dr. Smith is in at 3.5 today!", "Turning on the lights now."]);
        assert_eq!(pending, " Done");

        let mut pending = "These lights are now on:\nKitchen".to_string();
        assert_eq!(take_sentences(&mut pending).unwrap(), "These lights are now on:");
        assert_eq!(pending, "Kitchen");
        let mut pending = "word ".repeat(MAX_PENDING_BYTES / 5 + 1);
        assert!(take_sentences(&mut pending).is_some() && pending.is_empty());
    }

    /// Speaks every text as one chunk per word, refuses text without letters as normalization
    /// would, and records what it was asked.
    #[derive(Default)]
    struct Words(Mutex<Vec<(String, Option<String>)>>);

    impl Speaker for Arc<Words> {
        fn speak(&self, text: &str, opts: &SpeechOptions) -> ptts::Result<Pcm> {
            if !text.chars().any(char::is_alphabetic) {
                return Err(ptts::Error::InvalidArgument("nothing to synthesize".into()));
            }
            self.0.lock().unwrap().push((text.to_string(), opts.voice.clone()));
            Ok(Box::new((0..text.split_whitespace().count()).map(|_| Ok(vec![0.5; 4]))))
        }
    }

    /// Send `events` over a fresh connection. Returns the types of the events that came back,
    /// and what was spoken in which voice.
    async fn exchange(events: &[Vec<u8>]) -> (Vec<String>, Vec<(String, Option<String>)>) {
        let words = Arc::new(Words::default());
        let server = Arc::new(Server {
            speaker: words.clone(),
            voices: vec!["alba".into(), "marius".into()],
            info: encode("info", &json!({ "tts": [] }), &[]),
            requests: Arc::new(Semaphore::new(1)),
            seed_base: 0,
            sample_rate: 24000,
        });
        let (client, served) = tokio::io::duplex(1 << 16);
        let (read, write) = tokio::io::split(served);
        let task = tokio::spawn(connection(tokio::io::BufReader::new(read), write, server));
        let (from_server, mut to_server) = tokio::io::split(client);
        to_server.write_all(&events.concat()).await.unwrap();
        to_server.shutdown().await.unwrap();
        let mut received = vec![];
        let mut from_server = tokio::io::BufReader::new(from_server);
        while let Some(event) = read_event(&mut from_server).await.unwrap() {
            received.push(event.kind);
        }
        task.await.unwrap().unwrap();
        let spoken = words.0.lock().unwrap().clone();
        (received, spoken)
    }

    fn event(kind: &str, data: Value) -> Vec<u8> {
        encode(kind, &data, &[])
    }

    #[tokio::test]
    async fn a_synthesize_is_answered_with_audio() {
        let (received, spoken) = exchange(&[
            event("describe", json!({})),
            event(
                "synthesize",
                json!({ "text": "Hello there everyone.", "voice": { "name": "Alba" } }),
            ),
            event("synthesize", json!({ "text": "Hi.", "voice": { "name": "nobody" } })),
            event("synthesize", json!({ "text": " " })),
            // `default` is the default voice, though no voice has that name.
            event("synthesize", json!({ "text": "Good morning.", "voice": { "name": "default" } })),
        ])
        .await;
        let expected = [
            "info",
            "audio-start",
            "audio-chunk",
            "audio-chunk",
            "audio-chunk",
            "audio-stop",
            "error",
            "audio-start",
            "audio-stop",
            "error",
            "audio-start",
            "audio-stop",
            "audio-start",
            "audio-chunk",
            "audio-chunk",
            "audio-stop",
        ];
        assert_eq!(received, expected);
        let expected = [
            ("Hello there everyone.".to_string(), Some("alba".to_string())),
            ("Good morning.".to_string(), None),
        ];
        assert_eq!(spoken, expected);
    }

    #[tokio::test]
    async fn streamed_text_is_spoken_a_sentence_at_a_time() {
        // What Home Assistant sends: the chunks, then the whole text again, then the stop.
        let whole = "The living room lights are on. And now that is all done. 😀";
        let (received, spoken) = exchange(&[
            event("synthesize-start", json!({})),
            event("synthesize-chunk", json!({ "text": "The living room lights " })),
            event("synthesize-chunk", json!({ "text": "are on. And" })),
            event("synthesize-chunk", json!({ "text": " now that is all done. 😀" })),
            event("synthesize", json!({ "text": whole })),
            event("synthesize-stop", json!({})),
        ])
        .await;
        // The emoji left at the end has nothing to say, and is skipped without an error.
        let texts: Vec<_> = spoken.iter().map(|(text, _)| text.as_str()).collect();
        assert_eq!(texts, ["The living room lights are on.", "And now that is all done."]);
        let mut expected = vec!["audio-start"];
        expected.extend(["audio-chunk"; 12]);
        expected.extend(["audio-stop", "synthesize-stopped"]);
        assert_eq!(received, expected);

        // An unknown voice is one error, and the rest of the stream is ignored.
        let (received, spoken) = exchange(&[
            event("synthesize-start", json!({ "voice": { "name": "nobody" } })),
            event("synthesize-chunk", json!({ "text": "Turning the kitchen lights on. " })),
            event("synthesize", json!({ "text": "Turning the kitchen lights on. " })),
            event("synthesize-stop", json!({})),
        ])
        .await;
        assert_eq!((received, spoken), (vec!["error".into(), "synthesize-stopped".into()], vec![]));
    }
}
