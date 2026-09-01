//! Pocket TTS running entirely in the browser on xn's WebGPU backend.
//!
//! The whole model -- flow-matching LM and Mimi decoder -- runs as WGSL compute
//! shaders through `wgpu`'s WebGPU backend. Nothing is computed on a server; the
//! page fetches weights and everything else happens on the GPU.
//!
//! Two things shape the design:
//!
//! * **Nothing may block.** A browser delivers GPU completion through the event
//!   loop, so a blocking readback deadlocks. Ops only record into xn's batch, so
//!   they go through the ordinary synchronous `Backend` trait; only readbacks are
//!   awaited, via [`xn::WebGpuDevice::tensor_to_vec`].
//! * **The eos flag must not force a round trip mid-step.** `generate_step_parts`
//!   returns the raw eos logit as a tensor instead of a `bool`, so a step records
//!   the sampling *and* the Mimi decode before anything is read back. The frame's
//!   two readbacks then land on one flush.

use std::cell::RefCell;
use std::collections::HashMap;

use half::f16;
use ptts::flow_lm::StepInput;
use ptts::mimi::MimiDecoderState;
use ptts::tts_model::{TTSConfig, TTSModel, TTSState};
use wasm_bindgen::prelude::*;
use xn::nn::VB;
use xn::webgpu_backend::quantization::{Q80F16, Q80F32};
use xn::{BackendQ, Tensor, Unquantized, WebGpuDevice};

/// Mimi decoder context, matching the native examples.
const MIMI_CONTEXT_SIZE: usize = 250;
/// Spare KV positions on top of what an utterance is calculated to need.
const SEQ_BUDGET_SLACK: usize = 16;

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// `performance.now()` from whichever global this is running in.
///
/// The model runs in a Worker, which has no `window`, so `web_sys::window()` is
/// `None` there and every timing silently came back as zero.
fn now_ms() -> f64 {
    use wasm_bindgen::JsCast;
    let global = js_sys::global();
    let perf = match js_sys::Reflect::get(&global, &JsValue::from_str("performance")) {
        Ok(p) => p,
        Err(_) => return 0.0,
    };
    let now = match js_sys::Reflect::get(&perf, &JsValue::from_str("now")) {
        Ok(f) => f,
        Err(_) => return 0.0,
    };
    match now.dyn_ref::<js_sys::Function>().map(|f| f.call0(&perf)) {
        Some(Ok(v)) => v.as_f64().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Same weight-name rewrites the other frontends apply.
fn remap_key(name: &str) -> Option<String> {
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
        || name.contains("learnt_padding")
    {
        return None;
    }
    let mut name = name.to_string();
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");
    Some(name)
}

/// Generation needs a `Tokenizer` to build a `TTSModel`, but never calls it: the
/// page tokenizes in JS (SentencePiece unigram) and passes ids to `generate`.
struct JsTokenizer;

impl ptts::Tokenizer for JsTokenizer {
    fn encode(&self, _text: &str) -> xn::Result<Vec<u32>> {
        xn::bail!("tokenization happens in JS; pass token ids to `generate`")
    }
    fn decode(&self, _tokens: &[u32]) -> xn::Result<String> {
        xn::bail!("tokenization happens in JS")
    }
}

struct WasmRng {
    inner: Box<rand::rngs::StdRng>,
    distr: rand_distr::Normal<f32>,
}

impl WasmRng {
    fn new(temperature: f32, seed: u64) -> Self {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, temperature.sqrt()).unwrap();
        Self { inner: Box::new(rand::rngs::StdRng::seed_from_u64(seed)), distr }
    }
}

impl ptts::flow_lm::Rng for WasmRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

/// A loaded model plus its voice embeddings, for one choice of `Q`.
struct Engine<Q: BackendQ<B = WebGpuDevice>> {
    model: TTSModel<Q>,
    voices: HashMap<String, Tensor<Q::T, Q::B>>,
}

/// `Q` is chosen at load time from the requested dtype, so the engine is an enum
/// over the four the WebGPU backend supports.
enum AnyEngine {
    F32(Engine<Unquantized<f32, WebGpuDevice>>),
    F16(Engine<Unquantized<f16, WebGpuDevice>>),
    Q80F32(Engine<Q80F32>),
    Q80F16(Engine<Q80F16>),
}

struct Loaded {
    engine: AnyEngine,
    device: WebGpuDevice,
    dtype: String,
    sample_rate: u32,
    frame_size: u32,
    temperature: f32,
}

thread_local! {
    /// One model per worker. Taken out for the duration of a generation rather
    /// than borrowed, since a `RefCell` borrow may not be held across an await.
    static LOADED: RefCell<Option<Loaded>> = const { RefCell::new(None) };
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Does this browser expose WebGPU at all, and can xn get a device from it?
#[wasm_bindgen]
pub async fn probe() -> Result<JsValue, JsValue> {
    let dev = WebGpuDevice::new_async(0).await.map_err(err)?;
    let info = serde_json::json!({
        "device": xn::Backend::name(&dev),
        "f16": dev.supports_f16(),
    });
    Ok(JsValue::from_str(&info.to_string()))
}

/// Build the model from safetensors bytes.
///
/// `dtype` is `f32`, `f16`, `q8` or `q8f16`. The q8 paths quantize the f32
/// weights to `q8_0` on load, so they read the same file as `f32` does.
#[wasm_bindgen]
pub async fn load_model(
    model_bytes: Vec<u8>,
    config_json: Option<String>,
    dtype: String,
    temperature: f32,
) -> Result<JsValue, JsValue> {
    let t0 = now_ms();
    let device = WebGpuDevice::new_async(0).await.map_err(err)?;

    let mut cfg: TTSConfig = match config_json.as_deref() {
        Some(json) if !json.trim().is_empty() => serde_json::from_str(json).map_err(err)?,
        _ => TTSConfig::v202601(temperature),
    };
    cfg.temp = temperature;

    let dtype = dtype.to_ascii_lowercase();
    let wants_f16 = dtype == "f16" || dtype == "q8f16";
    if wants_f16 && !device.supports_f16() {
        return Err(err(format!(
            "dtype '{dtype}' needs WGSL shader-f16, which this adapter does not report"
        )));
    }

    let is_gguf = model_bytes.len() >= 4 && &model_bytes[..4] == b"GGUF";
    // A q8 run in a browser needs the blocks already quantized in the file. Given
    // dense weights, xn would have to read every weight back off the device to
    // quantize it, and a browser cannot block on a readback.
    if dtype.starts_with("q8") && !is_gguf {
        // Report the magic and length: the usual cause is the page having fetched
        // the wrong file, and "it wasn't a gguf" alone does not say which.
        let magic: String = model_bytes
            .iter()
            .take(8)
            .map(|b| if b.is_ascii_graphic() { *b as char } else { '.' })
            .collect();
        return Err(err(format!(
            "a q8 dtype needs gguf weights in a browser: quantizing from safetensors reads every \
             weight back to the host, which deadlocks here. Got {} bytes starting {magic:?} -- \
             load model.q8.gguf instead.",
            model_bytes.len(),
        )));
    }

    let vb = if is_gguf {
        VB::load_gguf_with_key_map(std::io::Cursor::new(model_bytes), device.clone(), remap_key)
            .map_err(err)?
    } else {
        VB::from_bytes_with_key_map(vec![model_bytes], device.clone(), remap_key).map_err(err)?
    };
    let root = vb.root();

    macro_rules! build {
        ($variant:ident, $q:ty) => {{
            let model = TTSModel::<$q>::load(&root, Box::new(JsTokenizer), &cfg).map_err(err)?;
            let sample_rate = model.sample_rate() as u32;
            let frame_size = (sample_rate as f64 / cfg.mimi.frame_rate).round() as u32;
            (AnyEngine::$variant(Engine { model, voices: HashMap::new() }), sample_rate, frame_size)
        }};
    }
    let (engine, sample_rate, frame_size) = match dtype.as_str() {
        "f32" => build!(F32, Unquantized<f32, WebGpuDevice>),
        "f16" => build!(F16, Unquantized<f16, WebGpuDevice>),
        "q8" | "q8_0" => build!(Q80F32, Q80F32),
        "q8f16" | "q8_0f16" => build!(Q80F16, Q80F16),
        other => return Err(err(format!("unknown dtype '{other}'"))),
    };

    // Weight upload and any q8 quantization are recorded, not yet executed.
    device.flush_async().await.map_err(err)?;

    let info = serde_json::json!({
        "device": xn::Backend::name(&device),
        "dtype": dtype,
        "f16": device.supports_f16(),
        "sample_rate": sample_rate,
        "frame_size": frame_size,
        "container": if is_gguf { "gguf" } else { "safetensors" },
        "load_ms": now_ms() - t0,
    });
    LOADED.with(|c| {
        *c.borrow_mut() =
            Some(Loaded { engine, device, dtype, sample_rate, frame_size, temperature });
    });
    Ok(JsValue::from_str(&info.to_string()))
}

/// Register a voice from a precomputed embedding safetensors file.
#[wasm_bindgen]
pub fn add_voice(name: String, bytes: Vec<u8>) -> Result<(), JsValue> {
    LOADED.with(|c| {
        let mut slot = c.borrow_mut();
        let loaded = slot.as_mut().ok_or_else(|| err("no model loaded"))?;
        let dev = loaded.device.clone();
        let tensors = xn::safetensors::load_from_buffer(&bytes, &dev).map_err(err)?;
        let (_, emb) = tensors.into_iter().next().ok_or_else(|| err("empty voice file"))?;
        let emb: Tensor<f32, WebGpuDevice> = emb.to::<f32>().map_err(err)?;
        let dims = emb.shape().dims().to_vec();
        let emb =
            if dims.len() == 2 { emb.reshape((1, dims[0], dims[1])).map_err(err)? } else { emb };
        macro_rules! ins {
            ($e:expr) => {{
                $e.voices.insert(name.clone(), emb.to().map_err(err)?);
            }};
        }
        match &mut loaded.engine {
            AnyEngine::F32(e) => ins!(e),
            AnyEngine::F16(e) => ins!(e),
            AnyEngine::Q80F32(e) => ins!(e),
            AnyEngine::Q80F16(e) => ins!(e),
        }
        Ok(())
    })
}

fn max_frames_for(num_tokens: usize) -> usize {
    ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize
}

/// The frame budget an utterance is allowed, so the page can size a waveform it
/// draws as frames arrive. Exported rather than reimplemented in JS to keep one
/// definition of the budget.
#[wasm_bindgen]
pub fn max_frames_for_tokens(num_tokens: usize) -> usize {
    max_frames_for(num_tokens)
}

/// This build is single-threaded, and there is nothing to configure.
///
/// Threading in xn lives entirely in its CPU backend: the WebGPU backend contains
/// no rayon call, so nothing here would read a worker count anyway. On top of
/// that the wasm32 rustflags carry no `+atomics`, so `std::thread` cannot spawn
/// and `num_cpus` reports one core. Reported rather than set, so the number is
/// derived from the build instead of asserted over it.
///
/// Deliberately no setter: `xn::set_num_threads` writes `RAYON_NUM_THREADS`, and
/// `std::env::set_var` is unsupported on `wasm32-unknown-unknown` -- calling it
/// traps the module.
#[wasm_bindgen]
pub fn threads_info() -> JsValue {
    let info = serde_json::json!({
        "threads": xn::get_num_threads(),
        "cpus": xn::get_num_cpus(),
    });
    JsValue::from_str(&info.to_string())
}

/// One utterance. `on_frame(Float32Array, frame_index)` is called per Mimi frame,
/// as it becomes available, so the page can play audio while the rest generates.
#[wasm_bindgen]
pub async fn generate(
    voice: String,
    tokens: Vec<u32>,
    frames_after_eos: usize,
    seed: f64,
    on_frame: js_sys::Function,
) -> Result<JsValue, JsValue> {
    // Taken out rather than borrowed: the borrow could not be held across the
    // awaits below, and taking it also makes a reentrant call fail cleanly.
    let mut loaded = LOADED
        .with(|c| c.borrow_mut().take())
        .ok_or_else(|| err("no model loaded (or a generation is already running)"))?;

    let res = run(&mut loaded, &voice, &tokens, frames_after_eos, seed as u64, &on_frame).await;
    LOADED.with(|c| *c.borrow_mut() = Some(loaded));
    res
}

async fn run(
    loaded: &mut Loaded,
    voice: &str,
    tokens: &[u32],
    frames_after_eos: usize,
    seed: u64,
    on_frame: &js_sys::Function,
) -> Result<JsValue, JsValue> {
    let (dtype, sample_rate, frame_size, temperature) =
        (loaded.dtype.clone(), loaded.sample_rate, loaded.frame_size, loaded.temperature);
    let device_name = xn::Backend::name(&loaded.device);

    let stats = match &mut loaded.engine {
        AnyEngine::F32(e) => {
            run_q(e, voice, tokens, frames_after_eos, seed, temperature, on_frame).await
        }
        AnyEngine::F16(e) => {
            run_q(e, voice, tokens, frames_after_eos, seed, temperature, on_frame).await
        }
        AnyEngine::Q80F32(e) => {
            run_q(e, voice, tokens, frames_after_eos, seed, temperature, on_frame).await
        }
        AnyEngine::Q80F16(e) => {
            run_q(e, voice, tokens, frames_after_eos, seed, temperature, on_frame).await
        }
    }
    .map_err(err)?;

    let audio_ms = stats.samples as f64 / sample_rate as f64 * 1e3;
    let out = serde_json::json!({
        "device": device_name,
        "dtype": dtype,
        "sample_rate": sample_rate,
        "frame_size": frame_size,
        "tokens": tokens.len(),
        "frames": stats.frame_ms.len(),
        "samples": stats.samples,
        "audio_ms": audio_ms,
        "total_ms": stats.total_ms,
        "ttfa_ms": stats.ttfa_ms,
        "prompt_ms": stats.prompt_ms,
        "rtf": if stats.total_ms > 0.0 { audio_ms / stats.total_ms } else { 0.0 },
        "frame_ms": stats.frame_ms,
    });
    Ok(JsValue::from_str(&out.to_string()))
}

struct RunStats {
    total_ms: f64,
    prompt_ms: f64,
    ttfa_ms: Option<f64>,
    samples: usize,
    frame_ms: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
async fn run_q<Q: BackendQ<B = WebGpuDevice>>(
    e: &mut Engine<Q>,
    voice: &str,
    tokens: &[u32],
    frames_after_eos: usize,
    seed: u64,
    temperature: f32,
    on_frame: &js_sys::Function,
) -> xn::Result<RunStats> {
    let t0 = now_ms();
    let dev = e.model.device().clone();
    let voice_emb = match e.voices.get(voice) {
        Some(v) => v,
        None => xn::bail!("unknown voice '{voice}'"),
    };

    let max_frames = max_frames_for(tokens.len());
    let seq_budget = voice_emb.dim(1usize)? + tokens.len() + max_frames + SEQ_BUDGET_SLACK;

    let mut state: TTSState<Q> = e.model.init_flow_lm_state(1, seq_budget)?;
    e.model.prompt_audio(&mut state, voice_emb)?;
    e.model.prompt_text(&mut state, tokens)?;
    let mut mimi_state: MimiDecoderState<f32, Q::B> =
        e.model.init_mimi_state(1, MIMI_CONTEXT_SIZE)?;
    // Conditioning is recorded, not executed; make it real before timing frames.
    dev.flush_async().await?;
    let prompt_ms = now_ms() - t0;

    let mut rng = WasmRng::new(temperature, seed);
    let mut prev: Option<Tensor<Q::T, Q::B>> = None;
    let mut eos_countdown: Option<usize> = None;
    let mut stats =
        RunStats { total_ms: 0.0, prompt_ms, ttfa_ms: None, samples: 0, frame_ms: Vec::new() };
    let mut last = now_ms();

    for frame in 0..max_frames {
        // The first step is stated as `Bos` rather than signalled with a NaN
        // latent, which would have to be read back off the device to be noticed.
        let input = match &prev {
            None => StepInput::Bos { batch: 1 },
            Some(t) => StepInput::Latent(t),
        };
        let (latent, eos_logit) = e.model.generate_step_parts(&mut state, input, &mut rng)?;
        // Recorded before either readback, so both resolve on a single flush.
        let pcm_t = e.model.decode_latent(&latent, &mut mimi_state)?;
        let pcm_t = pcm_t.narrow(0, ..1)?.contiguous()?;

        let eos_val = dev.tensor_to_vec(&eos_logit).await?;
        let pcm = dev.tensor_to_vec(&pcm_t).await?;

        let now = now_ms();
        stats.frame_ms.push(now - last);
        last = now;
        if stats.ttfa_ms.is_none() {
            stats.ttfa_ms = Some(now - t0);
        }
        stats.samples += pcm.len();

        let arr = js_sys::Float32Array::from(pcm.as_slice());
        if on_frame.call2(&JsValue::NULL, &arr, &JsValue::from_f64(frame as f64)).is_err() {
            break;
        }

        if e.model.eos_from_logit(&eos_val) && eos_countdown.is_none() {
            eos_countdown = Some(frames_after_eos);
        }
        if let Some(c) = &mut eos_countdown {
            if *c == 0 {
                break;
            }
            *c -= 1;
        }
        prev = Some(latent);
    }
    stats.total_ms = now_ms() - t0;
    Ok(stats)
}

/// `prepare_text_prompt` from `ptts`, exposed so the page applies exactly the
/// same normalisation before it tokenizes. Returns `[text, frames_after_eos]`.
#[wasm_bindgen]
pub fn prepare_text(text: &str) -> js_sys::Array {
    let (prepared, frames_after_eos) = ptts::tts_model::prepare_text_prompt(text);
    let out = js_sys::Array::new();
    out.push(&JsValue::from_str(&prepared));
    out.push(&JsValue::from_f64(frames_after_eos as f64));
    out
}
