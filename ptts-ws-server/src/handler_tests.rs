use super::*;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll};

struct StalledSocket(Arc<tokio::sync::Notify>);

impl futures_util::Sink<Message> for StalledSocket {
    type Error = std::io::Error;

    fn poll_ready(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.0.notify_one();
        Poll::Pending
    }

    fn start_send(self: Pin<&mut Self>, _: Message) -> Result<(), Self::Error> {
        unreachable!("the socket never becomes ready")
    }

    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Pending
    }

    fn poll_close(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
}

struct EndlessAudio {
    produced: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
    full: Arc<tokio::sync::Notify>,
}

impl Iterator for EndlessAudio {
    type Item = ptts::Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.produced.fetch_add(1, Ordering::SeqCst) + 1;
        if n == REPLY_QUEUE + AUDIO_QUEUE + 3 {
            self.full.notify_one();
        }
        Some(Ok(vec![0.1; 480]))
    }
}

impl Drop for EndlessAudio {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn a_slow_socket_bounds_audio_and_disconnect_joins_the_producer() {
    let (reply_tx, reply_rx) = mpsc::channel(REPLY_QUEUE);
    let (audio_tx, mut audio_rx) = mpsc::channel(AUDIO_QUEUE);
    let (cancel_tx, mut cancel_rx) = watch::channel(false);
    let stalled = Arc::new(tokio::sync::Notify::new());
    let forwarder = tokio::spawn(forward_replies(
        StalledSocket(stalled.clone()),
        reply_rx,
        cancel_rx.clone(),
        SEND_TIMEOUT,
    ));
    let produced = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    let full = Arc::new(tokio::sync::Notify::new());
    let stream =
        EndlessAudio { produced: produced.clone(), dropped: dropped.clone(), full: full.clone() };
    let generation = tokio::spawn(async move {
        let producer = tokio::task::spawn_blocking(move || drain_audio(stream, audio_tx));
        let mut encoder = Encoder::new(Format::default(), 480, 24000).unwrap();
        let outcome =
            forward_audio(&mut audio_rx, 480, 0, &mut encoder, &reply_tx, &mut cancel_rx).await;
        drop(audio_rx);
        producer.await.unwrap().unwrap();
        outcome.unwrap();
    });
    tokio::time::timeout(Duration::from_secs(2), stalled.notified()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), full.notified()).await.unwrap();
    // One chunk is in each forwarding stage and one is blocked in the producer.
    assert_eq!(produced.load(Ordering::SeqCst), REPLY_QUEUE + AUDIO_QUEUE + 3);
    cancel_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(2), generation).await.unwrap().unwrap();
    forwarder.await.unwrap().unwrap();
    assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn a_stalled_socket_times_out_and_closes_the_reply_channel() {
    let (tx, rx) = mpsc::channel(REPLY_QUEUE);
    let (_cancel_tx, cancel_rx) = watch::channel(false);
    tx.send(TtsReply::EndOfStream).await.unwrap();
    let error = forward_replies(
        StalledSocket(Arc::new(tokio::sync::Notify::new())),
        rx,
        cancel_rx,
        Duration::from_millis(10),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("stopped reading"));
    assert!(tx.is_closed());
}

#[tokio::test]
async fn normal_end_of_stream_drains_queued_replies_in_order() {
    let output = Arc::new(std::sync::Mutex::new(Vec::new()));
    let socket = futures_util::sink::unfold(output.clone(), |output, msg| async move {
        let Message::Text(text) = msg else { panic!("expected text reply") };
        output.lock().unwrap().push(text.to_string());
        Ok::<_, std::convert::Infallible>(output)
    });
    let (tx, rx) = mpsc::channel(REPLY_QUEUE);
    let (_cancel_tx, cancel_rx) = watch::channel(false);
    tx.send(TtsReply::Flushed { flush_id: 7 }).await.unwrap();
    tx.send(TtsReply::EndOfStream).await.unwrap();
    drop(tx);
    forward_replies(Box::pin(socket), rx, cancel_rx, SEND_TIMEOUT).await.unwrap();
    let output = output.lock().unwrap();
    assert_eq!(output.len(), 2);
    assert!(matches!(serde_json::from_str(&output[0]).unwrap(), TtsReply::Flushed { flush_id: 7 }));
    assert!(matches!(serde_json::from_str(&output[1]).unwrap(), TtsReply::EndOfStream));
}

#[test]
fn pending_text_is_limited_in_characters_and_a_rejected_append_preserves_it() {
    let mut buffer = "é".repeat(MAX_INPUT_CHARS - 1);
    assert!(append_text(&mut buffer, "🙂"));
    let before = buffer.clone();
    assert!(!append_text(&mut buffer, "x"));
    assert_eq!(buffer, before);
    buffer.clear();
    assert!(append_text(&mut buffer, "Hello."));
}

#[tokio::test(start_paused = true)]
async fn a_full_request_queue_waits_without_cancelling_a_healthy_client() {
    use futures_util::StreamExt;

    let queued = Arc::new(tokio::sync::Notify::new());
    let observed = Arc::new(AtomicUsize::new(0));
    let observed_reader = observed.clone();
    let queued_reader = queued.clone();
    let socket = futures_util::stream::iter(
        (0..REQUEST_QUEUE + 2)
            .map(|i| Ok::<_, std::io::Error>(Message::Text(i.to_string().into()))),
    )
    .inspect(move |_| {
        if observed_reader.fetch_add(1, Ordering::SeqCst) == REQUEST_QUEUE {
            queued_reader.notify_one();
        }
    });
    let (tx, mut rx) = mpsc::channel(REQUEST_QUEUE);
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let reader = tokio::spawn(read_requests(socket, tx, cancel_tx));
    tokio::time::timeout(Duration::from_secs(1), queued.notified()).await.unwrap();
    assert_eq!(rx.len(), REQUEST_QUEUE);
    assert_eq!(observed.load(Ordering::SeqCst), REQUEST_QUEUE + 1);

    // Healthy clients can queue text while an earlier flush takes longer than a write timeout.
    tokio::time::advance(SEND_TIMEOUT + Duration::from_secs(1)).await;
    tokio::task::yield_now().await;
    assert!(!reader.is_finished());
    assert!(!*cancel_rx.borrow());
    assert_eq!(observed.load(Ordering::SeqCst), REQUEST_QUEUE + 1);

    for i in 0..REQUEST_QUEUE + 2 {
        let Message::Text(text) = rx.recv().await.unwrap() else { panic!("expected text") };
        assert_eq!(text.as_str(), i.to_string());
    }
    reader.await.unwrap();
    assert!(*cancel_rx.borrow(), "EOF still cancels the session");
}

#[tokio::test]
async fn closing_a_full_request_queue_releases_the_reader() {
    use futures_util::StreamExt;

    let full = Arc::new(tokio::sync::Notify::new());
    let signal = full.clone();
    let observed = AtomicUsize::new(0);
    let socket = futures_util::stream::iter(
        (0..REQUEST_QUEUE + 1).map(|_| Ok::<_, std::io::Error>(Message::Text("text".into()))),
    )
    .inspect(move |_| {
        if observed.fetch_add(1, Ordering::SeqCst) == REQUEST_QUEUE {
            signal.notify_one();
        }
    });
    let (tx, rx) = mpsc::channel(REQUEST_QUEUE);
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let reader = tokio::spawn(read_requests(socket, tx, cancel_tx));
    tokio::time::timeout(Duration::from_secs(2), full.notified()).await.unwrap();
    drop(rx);
    tokio::time::timeout(Duration::from_secs(2), reader).await.unwrap().unwrap();
    assert!(*cancel_rx.borrow());
}

mod language_per_session {
    use super::*;
    use crate::model::load_ptts;
    use ptts::preprocess::Lang;

    const PT: &str = "Temos R$ 10,50 e a Dra. Ana, em 2 de março, às 15h.";
    const EN: &str = "We have $10.50 and Dr. Smith, on March 2nd at 3pm.";

    #[test]
    fn a_session_without_lang_gets_the_servers() {
        let en = Normalize::for_lang(Lang::En);
        assert_eq!(session_normalize(None, en, Rules::DEFAULT).unwrap(), en);
        for blank in ["", "  "] {
            assert_eq!(session_normalize(Some(blank), en, Rules::DEFAULT).unwrap(), en);
        }
    }

    #[test]
    fn lang_in_setup_is_parsed_like_the_flag() {
        let en = Normalize::for_lang(Lang::En);
        for (lang, want) in [("pt", "pt"), (" FR ", "fr"), ("nb", "no"), ("none", "none")] {
            let got = session_normalize(Some(lang), en, Rules::DEFAULT).unwrap();
            assert_eq!(got.as_str(), want, "{lang}");
        }
        assert_eq!(session_normalize(Some("off"), en, Rules::DEFAULT).unwrap(), Normalize::OFF);
    }

    #[test]
    fn setup_fixes_the_sessions_normalization_and_chunk_size() {
        let en = Normalize::for_lang(Lang::En);
        let own = setup_options(Some("pt"), en, Rules::DEFAULT).unwrap();
        assert_eq!(own.normalize, Some(Normalize::for_lang(Lang::Pt)));
        let servers = setup_options(None, en, Rules::DEFAULT).unwrap();
        assert_eq!(servers.normalize, Some(en));
        assert_eq!(own.max_tokens_per_chunk, Some(ptts::plan::MAX_FIT_TOKENS));
        assert!(setup_options(Some("xx"), en, Rules::DEFAULT).is_err());
    }

    #[test]
    fn the_language_warning_is_for_a_lang_the_session_sent() {
        let listed: Vec<String> = ["en", "fr"].map(String::from).into();
        let ja = Normalize::for_lang(Lang::Ja);
        assert!(setup_language_warning(None, ja, &listed).is_none());
        assert!(setup_language_warning(Some(" "), ja, &listed).is_none());
        assert!(setup_language_warning(Some("ja"), ja, &listed).is_some());
        assert!(
            setup_language_warning(Some("fr"), Normalize::for_lang(Lang::Fr), &listed).is_none()
        );
    }

    #[test]
    fn every_supported_language_is_accepted_in_setup() {
        let en = Normalize::for_lang(Lang::En);
        for &lang in Lang::ALL {
            let want = Normalize::for_lang(lang);
            let code = lang.as_str();
            assert_eq!(session_normalize(Some(code), en, Rules::DEFAULT).unwrap(), want, "{code}");
            let upper = format!(" {} ", code.to_uppercase());
            assert_eq!(
                session_normalize(Some(&upper), en, Rules::DEFAULT).unwrap(),
                want,
                "{code}"
            );
        }
    }

    #[test]
    fn a_session_language_takes_the_servers_rewrite_rules() {
        let en = Normalize::for_lang(Lang::En);
        let all = session_normalize(Some("pt"), en, Rules::ALL).unwrap();
        assert_eq!(all.rules(), Rules::ALL);
    }

    #[test]
    fn an_unparseable_lang_names_it_and_the_accepted_ones() {
        let en = Normalize::for_lang(Lang::En);
        let message = session_normalize(Some("xx"), en, Rules::DEFAULT).unwrap_err();
        assert!(message.contains("xx") && message.contains("pt"), "{message}");
        assert!(message.contains("invalid lang"), "{message}");
    }

    #[test]
    fn a_language_the_checkpoint_does_not_list_warns() {
        let listed: Vec<String> = ["en", "fr", "de", "pt", "es"].map(String::from).into();
        let norm = |lang: &str| session_normalize(Some(lang), Normalize::OFF, Rules::DEFAULT);
        assert!(norm("pt").unwrap().language_warning(&listed).is_none());
        assert!(norm("none").unwrap().language_warning(&listed).is_none());
        let warning = norm("ja").unwrap().language_warning(&listed).unwrap();
        assert!(warning.contains("ja") && warning.contains("fr"), "{warning}");
    }

    async fn load(catalog: bool) -> AppState {
        let model = std::env::var("PTTS_TEST_MODEL").expect("set PTTS_TEST_MODEL");
        load_ptts(
            &model,
            None,
            catalog.then_some(ptts::voice_catalog::DEFAULT_VOICE_CATALOG),
            ptts::synth::DeviceKind::Cpu,
            ptts::synth::Quant::Q80,
            0.3,
            7,
            1024,
            Normalize::for_lang(Lang::En),
            Rules::DEFAULT,
            &[],
            std::num::NonZeroUsize::new(2).unwrap(),
        )
        .await
        .unwrap()
    }

    /// Runs one session through `setup`, one text and `end_of_stream`: the PCM, or the error.
    async fn speak(
        app: &AppState,
        voice: Option<&str>,
        lang: Option<&str>,
        text: &str,
    ) -> Result<Vec<u8>, (u32, String)> {
        let (req_tx, mut req_rx) = mpsc::channel(REQUEST_QUEUE);
        let (reply_tx, mut reply_rx) = mpsc::channel(256);
        let (_cancel_tx, mut cancel_rx) = watch::channel(false);
        let mut setup = serde_json::json!({"type": "setup", "output_format": "pcm"});
        if let Some(voice) = voice {
            setup["voice"] = voice.into();
        }
        if let Some(lang) = lang {
            setup["lang"] = lang.into();
        }
        let msgs = [
            setup.to_string(),
            serde_json::json!({"type": "text", "text": text}).to_string(),
            r#"{"type":"end_of_stream"}"#.to_string(),
        ];
        for m in msgs {
            req_tx.send(Message::Text(m.into())).await.unwrap();
        }
        drop(req_tx);
        let app = app.clone();
        let run =
            tokio::spawn(
                async move { run_session(app, &mut req_rx, &reply_tx, &mut cancel_rx).await },
            );
        let mut pcm = Vec::new();
        while let Some(reply) = reply_rx.recv().await {
            match reply {
                TtsReply::Audio { audio, .. } => {
                    use base64::Engine;
                    pcm.extend(base64::engine::general_purpose::STANDARD.decode(audio).unwrap());
                }
                TtsReply::Error { message, code } => return Err((code, message)),
                _ => {}
            }
        }
        run.await.unwrap().unwrap();
        Ok(pcm)
    }

    fn is_speech(pcm: &[u8]) -> bool {
        pcm.as_chunks::<2>().0.iter().any(|b| i16::from_le_bytes(*b).abs() > 300)
    }

    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL: a multilingual checkpoint folder or hf:// repo"]
    async fn setup_accepts_any_lang_on_a_multilingual_checkpoint() {
        let app = load(false).await;
        for lang in [None, Some("pt"), Some("fr"), Some("none")] {
            let pcm = speak(&app, None, lang, PT).await.unwrap();
            assert!(is_speech(&pcm), "{lang:?}");
        }
    }

    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL: a multilingual checkpoint folder or hf:// repo"]
    async fn a_lang_the_checkpoint_does_not_list_still_speaks() {
        let app = load(false).await;
        assert!(app.synth.languages().iter().all(|l| l != "ja"));
        let pcm = speak(&app, None, Some("ja"), EN).await.unwrap();
        assert!(is_speech(&pcm));
    }

    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL: a multilingual checkpoint folder or hf:// repo"]
    async fn a_bad_lang_is_a_400_and_an_unknown_voice_a_404() {
        let app = load(false).await;
        let (code, message) = speak(&app, None, Some("xx"), EN).await.unwrap_err();
        assert_eq!(code, error_codes::BAD_REQUEST, "{message}");
        let (code, _) = speak(&app, Some("NoSuchVoice"), Some("pt"), PT).await.unwrap_err();
        assert_eq!(code, error_codes::NOT_FOUND);
    }

    /// Catalog voices speak with their own accent; `lang` only picks the reading rules.
    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL and the Hub voice catalog"]
    async fn any_catalog_voice_reads_any_language() {
        let app = load(true).await;
        let catalog =
            ptts::voice_catalog::VoiceCatalog::new(ptts::voice_catalog::DEFAULT_VOICE_CATALOG)
                .unwrap();
        let listed = app.synth.languages();
        let mut spoken = Vec::new();
        for voice in catalog.voices().unwrap() {
            let in_checkpoint = voice.language.as_ref().is_none_or(|l| listed.contains(l));
            assert_eq!(app.voices.contains(&voice.name), in_checkpoint, "{}", voice.name);
            if let Some(language) = &voice.language
                && !spoken.contains(language)
                && in_checkpoint
            {
                spoken.push(language.clone());
            }
        }
        assert!(spoken.len() > 1, "the catalog should cover several languages: {spoken:?}");
        let by_language = |lang: &str| {
            catalog
                .voices()
                .unwrap()
                .iter()
                .find(|v| v.language.as_deref() == Some(lang))
                .map(|v| v.name.clone())
        };
        let (pt_voice, en_voice) = (by_language("pt").unwrap(), by_language("en").unwrap());
        for (voice, lang, text) in
            [(&pt_voice, "en", EN), (&en_voice, "pt", PT), (&pt_voice, "pt", PT)]
        {
            let pcm = speak(&app, Some(voice), Some(lang), text).await.unwrap();
            assert!(is_speech(&pcm), "{voice} as {lang}");
        }
    }

    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL (the multilingual checkpoint: en, fr, de, pt, es) and the Hub voice catalog"]
    async fn one_instance_serves_several_languages_per_session() {
        let app = load(true).await;
        let catalog =
            ptts::voice_catalog::VoiceCatalog::new(ptts::voice_catalog::DEFAULT_VOICE_CATALOG)
                .unwrap();
        let voices = catalog.voices().unwrap();
        let voice_for = |lang: &str| {
            voices.iter().find(|v| v.language.as_deref() == Some(lang)).unwrap().name.clone()
        };
        let en = Normalize::for_lang(Lang::En);
        let opts = setup_options(Some("pt"), en, Rules::DEFAULT).unwrap().voice(voice_for("pt"));
        let session = app.synth.session(&opts, 1024).unwrap();
        assert_eq!(session.normalization(), Normalize::for_lang(Lang::Pt));
        let cases = [
            ("pt", "Temos R$ 10,50 e a Dra. Ana, em 2 de março, às 15h."),
            ("en", "We have $10.50 and Dr. Smith, on March 2nd at 3pm."),
            ("fr", "Nous avons 10,50 € et le Dr Martin, le 2 mars à 15 h."),
            ("de", "Wir haben 10,50 € und Dr. Müller, am 2. März um 15 Uhr."),
            ("es", "Tenemos 10,50 € y la Dra. Ana, el 2 de marzo a las 15 h."),
            ("pt", "We have $10.50 and Dr. Smith, on March 2nd at 3pm."),
        ];
        for pair in cases.chunks(2) {
            let app = &app;
            let runs = pair.iter().map(|&(lang, text)| {
                let voice = voice_for(lang);
                async move { (lang, text, speak(app, Some(&voice), Some(lang), text).await) }
            });
            for (lang, text, pcm) in futures_util::future::join_all(runs).await {
                let pcm = pcm.unwrap();
                assert!(is_speech(&pcm), "{lang}: {text}");
            }
        }
    }
}
