//! The raw WebAssembly surface of the browser build.
//!
//! This is what `wasm-bindgen` exports, and it is deliberately low level: it takes bytes the
//! caller has already fetched and generates a few frames per call, because the worker it
//! runs in must yield to its event loop between calls to hear a cancel. The `phonon-tts` npm
//! package runs it in a worker and handles downloads, caching and voices by name. Most
//! callers want that, not this.
//!
//! One engine serves every backend. Loading, voices, text normalization, sentence chunking
//! and the end-of-speech rule are written once, generic over the device; what differs is how
//! a result comes back to the host (see [`Readback`]) and how many frames a call produces
//! (see [`CPU_FRAMES_PER_STEP`]). The CPU runs on one thread or, in the `threads` build, on
//! several (see `start_cpu_pool`); the `webgpu` feature adds the GPU.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

use ptts::flow_lm::{FlowLMState, NormalRng, StepInput};
use ptts::loader::{load_speaker_proj, load_voice_emb_from_bytes, remap_key};
use ptts::mimi::MimiDecoderState;
use ptts::plan::{self, EosPolicy};
use ptts::preprocess::{Normalize, Rules};
use ptts::tok::Tok;
use ptts::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use ptts::tts_model::{
    MAX_TOKENS_PER_CHUNK, TTSConfig, TTSModel, TTSState, prepare_text_prompt,
    split_into_best_sentences,
};
#[cfg(feature = "webgpu")]
use xn::WebGpuDevice;
use xn::nn::{Linear, Path, VB};
use xn::quantized::Q80F32;
use xn::{Backend, BackendQ, CPU, CpuDevice, Result, Tensor, TypedTensor, Unquantized};

/// The flow LM's transformer state on device `B`: what a voice prompt leaves behind, and
/// what every chunk of an utterance starts from.
type RawState<B> = StreamingTransformerState<f32, B>;

/// Slots a voice state already occupies: the voice prompt's frames. Every flow-LM layer
/// advances together, so the first one says it for all of them.
fn raw_len<B: Backend>(state: &RawState<B>) -> usize {
    state
        .layer_states
        .iter()
        .find_map(|layer| match layer {
            LayerAttentionState::FlowLm(mha) => Some(mha.current_end),
            _ => None,
        })
        .unwrap_or(0)
}

/// Frames a `generation_step` produces on the CPU. One: reading a frame back costs nothing
/// there, and one frame is the soonest audio can start.
const CPU_FRAMES_PER_STEP: usize = 1;

/// Frames a `generation_step` produces on WebGPU. A readback there is a round trip through
/// the browser's GPU process, answered only through the event loop. A frame's next input is
/// its latent, which never leaves the GPU, so several frames are sampled with no readback,
/// decoded as one batch (which also gives the vocoder's matmuls more than one row), and read
/// back together. The cost is up to this many frames past end-of-speech, which are dropped.
#[cfg(feature = "webgpu")]
const GPU_FRAMES_PER_STEP: usize = 8;

/// Spare KV slots for the warm-up's throwaway chunk.
#[cfg(feature = "webgpu")]
const WARM_UP_SLACK: usize = 16;

/// What the warm-up prompts, to build the text path's pipelines along with the frames'.
#[cfg(feature = "webgpu")]
const WARM_UP_TEXT: &str = "Hello.";

/// How a device brings a tensor back to the host: the one thing the backends do differently.
/// The CPU already has it. WebGPU has to wait for the GPU, which a browser reports only
/// through its event loop, so nothing on that path may block.
trait Readback: Backend {
    async fn read(&self, t: &Tensor<f32, Self>) -> Result<Vec<f32>>;
}

impl Readback for CpuDevice {
    async fn read(&self, t: &Tensor<f32, Self>) -> Result<Vec<f32>> {
        t.to_vec()
    }
}

#[cfg(feature = "webgpu")]
impl Readback for WebGpuDevice {
    async fn read(&self, t: &Tensor<f32, Self>) -> Result<Vec<f32>> {
        self.tensor_to_vec(t).await
    }
}

/// Quantization variants exposed to JS.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Quant {
    F32,
    Q8,
}

impl Quant {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "f32" => Ok(Self::F32),
            "q8" => Ok(Self::Q8),
            other => xn::bail!("unsupported quantization '{other}', expected 'f32' or 'q8'"),
        }
    }
}

/// One sentence-aligned piece of the text, ready to prompt.
struct ChunkPlan {
    tokens: Vec<u32>,
    frame_budget: usize,
    frames_after_eos: usize,
}

/// The chunk currently being generated.
struct ChunkState<Q: BackendQ<T = f32>> {
    tts_state: TTSState<Q>,
    mimi_state: MimiDecoderState<f32, Q::B>,
    prev_latent: Option<Tensor<f32, Q::B>>,
    frame_budget: usize,
    eos: EosPolicy,
    step: usize,
    /// Set once `eos` has run out: the frame that did it has been returned, and the next
    /// call ends the chunk.
    done: bool,
}

struct GenState<Q: BackendQ<T = f32>> {
    /// The voice state, resized to fit the longest chunk. Every chunk starts from a clone of
    /// it, as `ptts::synth` does: chunks run one after the other, so sharing the KV storage
    /// is safe, and each overwrites only what lies past the voice prompt.
    base: RawState<Q::B>,
    chunks: std::vec::IntoIter<ChunkPlan>,
    current: Option<ChunkState<Q>>,
    /// One noise source for the whole text, so a seed fixes every chunk.
    rng: NormalRng,
}

/// Frames that have been generated and decoded on the device but not yet read back.
struct Pending<B: Backend> {
    /// The frames' PCM followed by one end-of-speech logit per frame, in one tensor so that
    /// a step costs a single readback however many frames it holds.
    out: Tensor<f32, B>,
    /// Where the PCM ends and the logits start.
    pcm_len: usize,
    frames: usize,
}

/// Generates `frames` frames of `chunk` and decodes them as one batch, reading nothing back.
fn record_frames<Q: BackendQ<T = f32>>(
    model: &TTSModel<Q>,
    chunk: &mut ChunkState<Q>,
    rng: &mut NormalRng,
    frames: usize,
) -> Result<Pending<Q::B>> {
    let mut latents = Vec::with_capacity(frames);
    let mut eos_logits = Vec::with_capacity(frames);
    for _ in 0..frames {
        let input = match &chunk.prev_latent {
            None => StepInput::Bos { batch: 1 },
            Some(t) => StepInput::Latent(t),
        };
        let (latent, eos_logit) = model.generate_step_parts(&mut chunk.tts_state, input, rng)?;
        chunk.prev_latent = Some(latent.clone());
        latents.push(latent);
        eos_logits.push(eos_logit);
    }
    let batched = Tensor::cat(&latents.iter().collect::<Vec<_>>(), 1)?;
    let pcm = model.decode_latent(&batched, &mut chunk.mimi_state)?;
    let pcm = pcm.narrow(0, ..1)?.contiguous()?;
    let pcm_len = pcm.elem_count();
    let eos = Tensor::cat(&eos_logits.iter().collect::<Vec<_>>(), 0)?;
    let out = Tensor::cat(&[&pcm.reshape(pcm_len)?, &eos.reshape(frames)?], 0)?;
    chunk.step += frames;
    Ok(Pending { out, pcm_len, frames })
}

/// A loaded model on one device, and the generation in progress on it.
struct Engine<Q: BackendQ<T = f32>> {
    model: TTSModel<Q>,
    device: Q::B,
    speaker_proj: Option<Linear<f32, Q::B>>,
    voice_states: Vec<RawState<Q::B>>,
    gen_state: Option<GenState<Q>>,
    frames_per_step: usize,
}

impl<Q: BackendQ<T = f32>> Engine<Q>
where
    Q::B: Readback,
{
    fn load(
        root: &Path<Q::B>,
        tokenizer: Box<dyn ptts::Tokenizer + Send + Sync>,
        cfg: &TTSConfig,
        device: Q::B,
        frames_per_step: usize,
    ) -> Result<Self> {
        let speaker_proj = load_speaker_proj(root, cfg)?;
        let model = TTSModel::load(root, tokenizer, cfg)?;
        Ok(Self {
            model,
            device,
            speaker_proj,
            voice_states: vec![],
            gen_state: None,
            frames_per_step,
        })
    }

    fn add_voice(&mut self, bytes: &[u8], cfg: &TTSConfig) -> Result<usize> {
        let tensors = xn::safetensors::load_from_buffer(bytes, &self.device)?;
        let raw = if tensors.contains_key(&kv_cache_name(0)) {
            voice_from_kv_cache(&tensors, cfg)?
        } else {
            self.voice_from_emb(bytes, cfg)?
        };
        self.voice_states.push(raw);
        Ok(self.voice_states.len() - 1)
    }

    /// A voice stored as an embedding (`emb`, `audio_prompt`) or as speaker-Mimi latents
    /// (`speaker_wavs`), the formats every other frontend reads. It is run through the flow
    /// LM once, here, so generation can start from the resulting state.
    fn voice_from_emb(&self, bytes: &[u8], cfg: &TTSConfig) -> Result<RawState<Q::B>> {
        let model_ext = cfg.model_ext();
        let emb = load_voice_emb_from_bytes(
            bytes,
            model_ext.as_deref(),
            self.speaker_proj.as_ref(),
            &self.device,
        )?;
        let frames = emb.dim(1usize)?;
        let mut state = self.model.init_flow_lm_state(1, frames)?;
        self.model.prompt_audio(&mut state, &emb)?;
        Ok(state.flow_lm_state.transformer_state)
    }

    fn start_generation(
        &mut self,
        voice_index: usize,
        text: &str,
        temperature: f32,
        seed: u32,
        normalize: Normalize,
        cfg: &TTSConfig,
    ) -> Result<usize> {
        // Dropped before anything else can fail, so a caller that swallows the error cannot
        // go on stepping and quietly resume the *previous* utterance.
        self.gen_state = None;
        // Built here rather than after planning: a temperature that cannot produce a
        // distribution should be refused before any work is done.
        let rng = NormalRng::new(temperature, seed as u64)?;
        let Some(voice) = self.voice_states.get(voice_index) else {
            xn::bail!("invalid voice index: {voice_index}")
        };
        let chunks = self.plan_chunks(text, normalize, cfg)?;

        // The KV budget has to hold the voice prompt plus the longest chunk's text and audio.
        // This is `plan::seq_budget` with the voice's real length in place of its
        // `PROMPT_SEQ_HEADROOM` guess, the same bound `ptts::synth` checks a session against.
        // A step never runs past a chunk's frame budget, however many frames it holds.
        let voice_len = raw_len(voice);
        let seq_budget =
            chunks.iter().map(|c| voice_len + c.tokens.len() + c.frame_budget).max().unwrap_or(0);
        let base = voice.with_seq_budget(seq_budget)?;

        let num_chunks = chunks.len();
        self.gen_state = Some(GenState { base, chunks: chunks.into_iter(), current: None, rng });
        Ok(num_chunks)
    }

    /// Normalize the whole text, split it into sentence-aligned chunks and tokenize each,
    /// exactly as `ptts::synth` does: normalization first, because it rewrites the characters
    /// the splitter looks for.
    fn plan_chunks(
        &self,
        text: &str,
        normalize: Normalize,
        cfg: &TTSConfig,
    ) -> Result<Vec<ChunkPlan>> {
        let text = normalize.apply(text);
        let conditioner = &self.model.flow_lm.conditioner;
        let Some(tokenizer) = conditioner.tokenizer.as_deref() else {
            xn::bail!("this model was loaded without a tokenizer")
        };
        let texts = split_into_best_sentences(tokenizer, &text, Some(MAX_TOKENS_PER_CHUNK))?;
        let mut chunks = Vec::with_capacity(texts.len());
        for text in texts {
            let (prepared, frames_after_eos) = prepare_text_prompt(&text);
            let tokens = conditioner.tokenize(&prepared)?;
            let frame_budget = plan::frame_budget(tokens.len(), cfg.mimi.frame_rate);
            chunks.push(ChunkPlan { tokens, frame_budget, frames_after_eos });
        }
        if chunks.is_empty() {
            xn::bail!("nothing to synthesize: the text is empty")
        }
        Ok(chunks)
    }

    fn next_chunk(&mut self) -> Result<Option<usize>> {
        let Some(gen_state) = self.gen_state.as_mut() else { return Ok(None) };
        gen_state.current = None;
        let Some(chunk) = gen_state.chunks.next() else {
            self.gen_state = None;
            return Ok(None);
        };
        let transformer_state = gen_state.base.clone();
        let mut tts_state = TTSState { flow_lm_state: FlowLMState { transformer_state } };
        self.model.prompt_text(&mut tts_state, &chunk.tokens)?;
        let mimi_state = self.model.init_mimi_state(1)?;
        gen_state.current = Some(ChunkState {
            tts_state,
            mimi_state,
            prev_latent: None,
            frame_budget: chunk.frame_budget,
            eos: EosPolicy::new(chunk.frames_after_eos),
            step: 0,
            done: false,
        });
        Ok(Some(chunk.tokens.len()))
    }

    /// Up to `frames_per_step` frames of the current chunk as PCM, or `None` once the chunk
    /// is finished.
    async fn step(&mut self) -> Result<Option<Vec<f32>>> {
        let pending = {
            let Some(gen_state) = self.gen_state.as_mut() else { return Ok(None) };
            let Some(chunk) = gen_state.current.as_mut() else { return Ok(None) };
            if chunk.done || chunk.step >= chunk.frame_budget {
                gen_state.current = None;
                return Ok(None);
            }
            let frames = self.frames_per_step.min(chunk.frame_budget - chunk.step);
            record_frames(&self.model, chunk, &mut gen_state.rng, frames)?
        };
        let mut pcm = self.device.read(&pending.out).await?;
        let eos = pcm.split_off(pending.pcm_len);

        // Keep frames up to the one where the end-of-speech rule runs out. `should_stop` is
        // asked after each frame, so that frame itself is part of the output, and the next
        // call reports the end of the chunk.
        let Some(chunk) = self.gen_state.as_mut().and_then(|g| g.current.as_mut()) else {
            return Ok(None);
        };
        let per_frame = pcm.len() / pending.frames;
        let mut keep = pending.frames;
        for (i, logit) in eos.iter().enumerate().take(pending.frames) {
            if chunk.eos.should_stop(self.model.eos_from_logit(std::slice::from_ref(logit))) {
                chunk.done = true;
                keep = i + 1;
                break;
            }
        }
        pcm.truncate(keep * per_frame);
        Ok(Some(pcm))
    }

    /// Prompts a short text and runs one throwaway step after it, so a GPU driver compiles
    /// its pipelines during the load rather than inside the first real utterance, where they
    /// would hold up its first audio. The step has the shape generation uses, so the
    /// vocoder's kernels are built at that shape.
    #[cfg(feature = "webgpu")]
    async fn warm_up(&self) -> Result<()> {
        let frames = self.frames_per_step;
        let tokens = self.model.flow_lm.conditioner.tokenize(WARM_UP_TEXT)?;
        let mut tts_state =
            self.model.init_flow_lm_state(1, tokens.len() + frames + WARM_UP_SLACK)?;
        self.model.prompt_text(&mut tts_state, &tokens)?;
        let mut chunk = ChunkState {
            tts_state,
            mimi_state: self.model.init_mimi_state(1)?,
            prev_latent: None,
            frame_budget: frames,
            eos: EosPolicy::new(0),
            step: 0,
            done: false,
        };
        let mut rng = NormalRng::new(0.3, 0)?;
        let pending = record_frames(&self.model, &mut chunk, &mut rng, frames)?;
        // Read something back, so this waits for the work rather than just queuing it.
        self.device.read(&pending.out).await?;
        Ok(())
    }
}

/// A voice stored as the flow LM's KV cache after the voice prompt, the format the
/// `embeddings_v2/` voices use: `transformer.layers.{i}.self_attn/cache`, shaped
/// `[2, 1, seq, heads, head_dim]`, for each layer. Nothing to run.
fn voice_from_kv_cache<B: Backend>(
    tensors: &std::collections::HashMap<String, TypedTensor<B>>,
    cfg: &TTSConfig,
) -> Result<RawState<B>> {
    let num_layers = cfg.flow_lm.num_layers;
    let mut layer_states = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let cache_name = kv_cache_name(i);
        let cache = match tensors.get(&cache_name) {
            Some(TypedTensor::F32(t)) => t,
            _ => xn::bail!("expected f32 tensor: {cache_name}"),
        };
        let (two, batch, seq_len, num_heads, head_dim) = cache.dims5()?;
        if two != 2 {
            xn::bail!("{cache_name}: expected a first dim of size 2, got {two}");
        }
        let kv = |i: usize| -> Result<Tensor<f32, B>> {
            cache.narrow(0, i..i + 1)?.contiguous()?.reshape((batch, seq_len, num_heads, head_dim))
        };
        layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState {
            k_cache: kv(0)?,
            v_cache: kv(1)?,
            current_end: seq_len,
        }));
    }
    Ok(StreamingTransformerState { layer_states })
}

/// The engine for whichever backend and weight format was loaded.
// There is one per loaded model and it never moves, so boxing the larger variants would
// only add an indirection.
#[allow(clippy::large_enum_variant)]
enum AnyEngine {
    CpuF32(Engine<Unquantized<f32, CpuDevice>>),
    CpuQ8(Engine<Q80F32>),
    #[cfg(feature = "webgpu")]
    WebGpuQ8(Engine<xn::webgpu_backend::quantization::Q8F32>),
}

/// Run a block against the loaded engine. Within the block, `$e` is the `Engine<Q>`.
macro_rules! with_engine {
    ($engine:expr, |$e:ident| $body:expr) => {
        match $engine {
            AnyEngine::CpuF32($e) => $body,
            AnyEngine::CpuQ8($e) => $body,
            #[cfg(feature = "webgpu")]
            AnyEngine::WebGpuQ8($e) => $body,
        }
    };
}

/// Everything a `Model` holds.
struct Loaded {
    engine: AnyEngine,
    cfg: TTSConfig,
    /// How `start_generation` normalizes, named by the page when it loaded the model.
    normalize: Normalize,
    device: &'static str,
}

impl Loaded {
    async fn load(
        model_weights: Vec<u8>,
        tokenizer_json: &[u8],
        config_json: Option<Vec<u8>>,
        quant: &str,
        lang: &str,
        rewrites: Option<&str>,
        device: &str,
    ) -> Result<Self> {
        let quant = Quant::parse(quant)?;
        let rules = match rewrites {
            Some(rewrites) => Rules::parse(rewrites)?,
            None => Rules::ALL,
        };
        let normalize = Normalize::parse(lang)?.with_rules(rules);
        let cfg = match config_json {
            Some(json) => match serde_json::from_slice(&json) {
                Ok(cfg) => cfg,
                Err(e) => xn::bail!("cannot parse config.json: {e}"),
            },
            // `temp` is not read by the runtime: sampling temperature reaches the model
            // through `start_generation`.
            None => TTSConfig::v202601(0.3),
        };
        let tokenizer: Box<dyn ptts::Tokenizer + Send + Sync> =
            Box::new(Tok::from_bytes(tokenizer_json)?);
        let is_gguf = model_weights.len() >= 4 && &model_weights[..4] == b"GGUF";
        console_log!("[phonon] loading model with quant={quant:?} on {device}");

        let (engine, device) = match device {
            "cpu" => {
                let vb = if is_gguf {
                    VB::load_gguf_with_key_map(std::io::Cursor::new(model_weights), CPU, remap_key)?
                } else {
                    VB::from_bytes_with_key_map(vec![model_weights], CPU, remap_key)?
                };
                let root = vb.root();
                let engine = match quant {
                    Quant::F32 => AnyEngine::CpuF32(Engine::load(
                        &root,
                        tokenizer,
                        &cfg,
                        CPU,
                        CPU_FRAMES_PER_STEP,
                    )?),
                    Quant::Q8 => AnyEngine::CpuQ8(Engine::load(
                        &root,
                        tokenizer,
                        &cfg,
                        CPU,
                        CPU_FRAMES_PER_STEP,
                    )?),
                };
                (engine, "cpu")
            }
            #[cfg(feature = "webgpu")]
            "webgpu" => {
                // q8 weights go to the GPU as they are. Quantizing dense weights would read
                // every one back to the host, which a browser cannot do.
                if quant != Quant::Q8 || !is_gguf {
                    xn::bail!("WebGPU needs q8 weights in a GGUF file")
                }
                let dev = WebGpuDevice::new_async(0).await?;
                let vb = VB::load_gguf_with_key_map(
                    std::io::Cursor::new(model_weights),
                    dev.clone(),
                    remap_key,
                )?;
                let engine =
                    Engine::load(&vb.root(), tokenizer, &cfg, dev.clone(), GPU_FRAMES_PER_STEP)?;
                // The weight upload is recorded, not yet executed.
                dev.flush_async().await?;
                engine.warm_up().await?;
                (AnyEngine::WebGpuQ8(engine), "webgpu")
            }
            #[cfg(not(feature = "webgpu"))]
            "webgpu" => xn::bail!("this build has no WebGPU support"),
            other => xn::bail!("unknown device '{other}', expected 'cpu' or 'webgpu'"),
        };
        Ok(Self { engine, cfg, normalize, device })
    }
}

fn kv_cache_name(layer: usize) -> String {
    format!("transformer.layers.{layer}.self_attn/cache")
}

fn js_err(e: xn::Error) -> JsError {
    JsError::new(&e.to_string())
}

/// A loaded model, on the CPU or on WebGPU.
///
/// The state sits behind an `Rc<RefCell<Option<_>>>` because `generation_step` is async and a
/// `RefCell` borrow may not be held across an await: the step takes the state out for its
/// duration and puts it back after. A call that arrives meanwhile finds it missing and is
/// refused, so calls must not overlap; `phonon-tts`'s worker awaits each one.
#[wasm_bindgen]
pub struct Model {
    loaded: Rc<RefCell<Option<Loaded>>>,
}

impl Model {
    fn with<T>(&self, f: impl FnOnce(&mut Loaded) -> Result<T>) -> Result<T> {
        match self.loaded.borrow_mut().as_mut() {
            Some(loaded) => f(loaded),
            None => xn::bail!("the model is busy: a generation step is still running"),
        }
    }

    /// A failed step leaves a chunk half prompted or half generated. Dropping the generation
    /// makes every later call report the end instead, so a caller that swallows the error
    /// cannot go on and silently skip a sentence. `start_generation` does the same.
    fn drop_generation_on_error<T>(loaded: &mut Loaded, result: &Result<T>) {
        if result.is_err() {
            with_engine!(&mut loaded.engine, |e| e.gen_state = None);
        }
    }
}

#[wasm_bindgen]
impl Model {
    /// Loads a model. `model_weights` is a safetensors or GGUF checkpoint, `tokenizer_json`
    /// the contents of the `tokenizer.json` for its vocabulary, and `config_json` its
    /// `config.json`, or `undefined` for the original Pocket TTS architecture.
    ///
    /// Two things a config cannot ask this build for. Its `temp` is not read: the sampling
    /// temperature reaches the model through `start_generation`. And there is no
    /// classifier-free guidance here -- guidance is a caller's option in `ptts::synth`
    /// (`SynthOpts::cfg_coef`), not a field of the config, and the browser build never turns
    /// it on, so `cfg_null_audio_empty` is inert. Everything else -- the flow LM and Mimi
    /// shapes, `lsd_decode_steps`, `eos_threshold`, `model_id`, `speaker_mimi` -- is honored.
    ///
    /// `quant` is `"f32"` or `"q8"`.
    ///
    /// `lang` is required: the language text is normalized as before it is
    /// tokenized, one of `"en"`, `"fr"`, `"de"`, `"es"`, `"pt"`, or `"none"`
    /// to hand text to the tokenizer as written. The spoken forms of `@`, `+`
    /// and `=` differ per language, so there is nothing safe to default to.
    ///
    /// `rewrites` picks which word rewrites run on the normalized text: `"all"`, `"none"`,
    /// or a comma-separated list of rule names, of which there is one today, `"numbers"`.
    /// `undefined` means `"all"`.
    ///
    /// `device` is `"cpu"` or `"webgpu"`. WebGPU needs `q8` weights in a GGUF file, and a
    /// build with the `webgpu` feature.
    pub async fn load(
        model_weights: Vec<u8>,
        tokenizer_json: Vec<u8>,
        config_json: Option<Vec<u8>>,
        quant: String,
        lang: String,
        rewrites: Option<String>,
        device: String,
    ) -> std::result::Result<Model, JsError> {
        let loaded = Loaded::load(
            model_weights,
            &tokenizer_json,
            config_json,
            &quant,
            &lang,
            rewrites.as_deref(),
            &device,
        )
        .await
        .map_err(js_err)?;
        Ok(Model { loaded: Rc::new(RefCell::new(Some(loaded))) })
    }

    /// Registers a voice from a safetensors file and returns its index for
    /// `start_generation`. Either a precomputed KV cache (`embeddings_v2/`) or a voice
    /// embedding (`emb`, `audio_prompt` or `speaker_wavs`), told apart by tensor name.
    pub fn add_voice(&self, voice: &[u8]) -> std::result::Result<usize, JsError> {
        self.with(|l| with_engine!(&mut l.engine, |e| e.add_voice(voice, &l.cfg))).map_err(js_err)
    }

    /// Normalizes `text`, splits it into sentence-aligned chunks and tokenizes them. Returns
    /// the number of chunks. Runs no model: call `next_chunk` to start the first one.
    pub fn start_generation(
        &self,
        voice_index: usize,
        text: &str,
        temperature: f32,
        seed: u32,
    ) -> std::result::Result<usize, JsError> {
        self.with(|l| {
            let (normalize, cfg) = (l.normalize, &l.cfg);
            with_engine!(&mut l.engine, |e| e.start_generation(
                voice_index,
                text,
                temperature,
                seed,
                normalize,
                cfg
            ))
        })
        .map_err(js_err)
    }

    /// Prompts the model with the next chunk's text and returns its token count, or
    /// `undefined` once every chunk has been generated.
    pub fn next_chunk(&self) -> std::result::Result<Option<usize>, JsError> {
        self.with(|l| {
            let result = with_engine!(&mut l.engine, |e| e.next_chunk());
            Self::drop_generation_on_error(l, &result);
            result
        })
        .map_err(js_err)
    }

    /// Generates and decodes the next frames of the current chunk: a promise of mono PCM at
    /// `sample_rate`, a whole number of `frame_size`-sample frames, or of `undefined` when the
    /// chunk is finished. One frame per call on the CPU, several on WebGPU.
    pub fn generation_step(&self) -> js_sys::Promise {
        let cell = Rc::clone(&self.loaded);
        wasm_bindgen_futures::future_to_promise(async move {
            let Some(mut loaded) = cell.borrow_mut().take() else {
                return Err(
                    JsError::new("the model is busy: a generation step is still running").into()
                );
            };
            let result = with_engine!(&mut loaded.engine, |e| e.step().await);
            Self::drop_generation_on_error(&mut loaded, &result);
            *cell.borrow_mut() = Some(loaded);
            match result {
                Ok(Some(pcm)) => Ok(js_sys::Float32Array::from(pcm.as_slice()).into()),
                Ok(None) => Ok(JsValue::UNDEFINED),
                Err(e) => Err(js_err(e).into()),
            }
        })
    }

    /// Drops the generation in progress, if any.
    pub fn stop_generation(&self) {
        if let Some(l) = self.loaded.borrow_mut().as_mut() {
            with_engine!(&mut l.engine, |e| e.gen_state = None);
        }
    }

    pub fn sample_rate(&self) -> std::result::Result<usize, JsError> {
        self.with(|l| Ok(with_engine!(&l.engine, |e| e.model.sample_rate()))).map_err(js_err)
    }

    /// Samples in one 80 ms frame: `generation_step` returns whole frames.
    pub fn frame_size(&self) -> std::result::Result<usize, JsError> {
        self.with(|l| {
            let sample_rate = with_engine!(&l.engine, |e| e.model.sample_rate());
            Ok((sample_rate as f64 / l.cfg.mimi.frame_rate).round() as usize)
        })
        .map_err(js_err)
    }

    /// `"cpu"` or `"webgpu"`: where this model runs.
    pub fn device(&self) -> std::result::Result<String, JsError> {
        self.with(|l| Ok(l.device.to_string())).map_err(js_err)
    }
}

/// CPU SIMD features the wasm module was compiled with. The relevant one
/// for browser builds is `simd128`; `avx`/`neon`/`f16c` are reported for
/// completeness so it's clear which native-target builds enabled them.
#[wasm_bindgen]
pub fn cpu_features() -> js_sys::Object {
    let obj = js_sys::Object::new();
    let set = |k: &str, v: bool| {
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(k), &JsValue::from_bool(v));
    };
    set("avx", xn::with_avx());
    set("neon", xn::with_neon());
    set("simd128", xn::with_simd128());
    set("f16c", xn::with_f16c());
    obj
}

// ---- threads ----
//
// Only in the `threads` build. It is a separate module because wasm threads need shared
// memory, which needs std rebuilt with atomics and a page that is cross-origin isolated, and
// a page that is not would fail to load it at all. The JS side picks this build only when
// the page can run it, and the single-threaded one otherwise.
//
// Two steps, from the worker that owns the `Model`, before it loads one:
// `initThreadPool(workers)` gives rayon its Web Workers, then `start_cpu_pool(workers)`
// parks xn's CPU pool on them. A rayon fork/join wakes a parked Web Worker per operator,
// which costs more than most of a frame's operators take; xn's pool workers spin between
// operators instead, so a dispatch is cheap and small operators are worth splitting.

#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Iterations a pool worker spins before parking: long enough to bridge the gaps between
/// operators inside a frame, short enough that an idle page stops burning its cores soon
/// after an utterance ends.
#[cfg(feature = "threads")]
const POOL_SPIN_BUDGET: u32 = 4_000_000;

/// The smallest operator, in multiply-adds, that the pool splits across its workers. Below
/// it, handing work out costs more than doing it on one thread.
#[cfg(feature = "threads")]
const MIN_PARALLEL_WORK: usize = 256 << 10;

/// Runs xn's CPU pool on `workers` of rayon's Web Workers, for good: they never return, so
/// nothing else may go through rayon afterwards. Call it once, after `initThreadPool` and
/// before loading a model, from a worker rather than the page, since a dispatch can block.
/// Returns how many threads now share the work: the workers plus the calling one.
#[cfg(feature = "threads")]
#[wasm_bindgen]
pub fn start_cpu_pool(workers: usize) -> usize {
    if workers == 0 {
        return 1;
    }
    // A worker rayon cannot schedule would never take a job, and the first dispatch would
    // wait for it forever.
    let workers = workers.min(rayon::current_num_threads());
    let size = xn::threadpool::start_pool_with(
        xn::threadpool::PoolConfig { workers, spin_budget: Some(POOL_SPIN_BUDGET) },
        |job| rayon::spawn(job),
    );
    xn::threadpool::set_min_parallel_work(MIN_PARALLEL_WORK);
    size
}
