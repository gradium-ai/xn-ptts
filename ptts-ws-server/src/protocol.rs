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
    EndOfStream,
    Flushed {
        flush_id: u64,
    },
}

pub mod error_codes {
    pub const BAD_REQUEST: u32 = 400;
    pub const NOT_FOUND: u32 = 404;
    pub const TIMEOUT: u32 = 408;
    pub const INTERNAL: u32 = 500;
    pub const NOT_IMPLEMENTED: u32 = 501;
}
