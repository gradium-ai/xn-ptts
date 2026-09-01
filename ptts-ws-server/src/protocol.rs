#[allow(dead_code)]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ErrorMsg {
    Error { message: String },
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TtsRequest {
    Setup {
        #[serde(default)]
        json_config: String,
        #[serde(default)]
        model_name: String,
        output_format: String,
        voice: Option<String>,
        voice_id: Option<String>,
        voice_emb: Option<String>,
    },
    Text {
        text: String,
    },
    Flush {
        flush_id: u64,
    },
    EndOfStream,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TtsReply {
    Text {
        text: String,
        start_s: f64,
        stop_s: f64,
        stream_id: u32,
    },
    Ready {
        model_name: String,
        sample_rate: u32,
        frame_size: u32,
        audio_stream_names: Vec<String>,
        text_stream_names: Vec<String>,
        request_id: String,
    },
    Audio {
        audio: String,
        start_s: f64,
        stop_s: f64,
        stream_id: u32,
    },
    Error {
        message: String,
        code: u32,
    },
    Stats {
        json_stats: String,
    },
    EndOfStream,
    Flushed {
        flush_id: u64,
    },
}

/// Per-generation timings, sent as the `json_stats` payload of `TtsReply::Stats`
/// once a stream finishes. This is what the web app renders, so it is also the
/// definition of "how well it works" for a given backend.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct GenStats {
    /// How the server was built and launched, e.g. `webgpu f16`.
    pub backend: String,
    /// The adapter/device xn actually chose.
    pub device: String,
    pub stream_id: u32,
    pub chars: usize,
    pub tokens: usize,
    pub frames: usize,
    pub audio_ms: f64,
    pub total_ms: f64,
    pub ttfa_ms: Option<f64>,
    /// Audio produced per unit of wall time; >1 is faster than realtime.
    pub rtf: f64,
    pub frame_ms_mean: Option<f64>,
    pub frame_ms_p50: Option<f64>,
    pub frame_ms_p95: Option<f64>,
    pub frame_ms_max: Option<f64>,
    pub threads: usize,
}

pub mod error_codes {
    pub const BAD_REQUEST: u32 = 400;
    pub const NOT_FOUND: u32 = 404;
    pub const INTERNAL: u32 = 500;
    pub const NOT_IMPLEMENTED: u32 = 501;
}
