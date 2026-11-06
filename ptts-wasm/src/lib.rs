//! The raw WebAssembly surface of the browser build.
//!
//! This is what `wasm-bindgen` exports, and it is deliberately low level: it takes bytes the
//! caller has already fetched. The `phonon-tts` npm package runs it in a worker and handles
//! downloads, caching and voices by name. Most callers want that, not this.
//!
//! Two things are exported:
//!
//! - [`Model`] runs the whole model on the CPU, one 80 ms frame per call, because the worker it
//!   runs in must yield to its event loop between calls to hear a cancel. It runs on one
//!   thread or, in the `threads` build, on several (see `start_cpu_pool`).
//! - [`Preparer`] is everything the package's WebGPU engine (`js/gpu/` in the package) needs
//!   from Rust, without the model: text normalization, sentence chunking and tokenization,
//!   the voice files, and the conditioner sum. The GPU engine reads the transformer and Mimi
//!   weights from the GGUF itself, so they never enter this module's memory.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
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
use ptts::loader::{load_conditions, load_speaker_proj, load_voice_emb_from_bytes, remap_key};
use ptts::mimi::MimiDecoderState;
use ptts::plan::{self, EosPolicy};
use ptts::preprocess::{Normalize, Rules};
use ptts::tok::Tok;
use ptts::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use ptts::tts_model::{MAX_TOKENS_PER_CHUNK, TTSConfig, TTSModel, TTSState};
use xn::nn::{Linear, Path, VB};
use xn::quantized::Q80F32;
use xn::{Backend, BackendQ, CPU, CpuDevice, Result, Tensor, TypedTensor, Unquantized};

/// The flow LM's transformer state: what a voice prompt leaves behind, and what every chunk of
/// an utterance starts from.
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

/// What every frontend reads from the page's settings: the config, how text is normalized, and
/// the conditioner values.
struct Settings {
    cfg: TTSConfig,
    normalize: Normalize,
    conditions: HashMap<String, String>,
}

impl Settings {
    fn parse(
        config_json: &[u8],
        lang: &str,
        rewrites: Option<&str>,
        conditions: &[String],
    ) -> Result<Self> {
        let rules = match rewrites {
            Some(rewrites) => rewrites.parse::<Rules>()?,
            None => Rules::DEFAULT,
        };
        let normalize = lang.parse::<Normalize>()?.with_rules(rules);
        let mut values = HashMap::new();
        for condition in conditions {
            let Some((name, value)) = condition.split_once('=') else {
                xn::bail!("a condition is NAME=VALUE, got {condition:?}")
            };
            values.insert(name.to_string(), value.to_string());
        }
        let cfg: TTSConfig = serde_json::from_slice(config_json)
            .map_err(|e| xn::Error::msg(format!("cannot parse config.json: {e}")))?;
        // Such a checkpoint speaks only its own voices, each with its own conditioning, and
        // voices here come from files.
        if !cfg.voices.is_empty() {
            xn::bail!(
                "this checkpoint has baked-in voices, which the browser build does not support"
            )
        }
        Ok(Self { cfg, normalize, conditions: values })
    }
}

/// The same chunks `ptts::synth` makes, from [`plan::chunks`].
fn plan_chunks(
    tokenizer: &dyn ptts::Tokenizer,
    text: &str,
    normalize: Normalize,
    cfg: &TTSConfig,
) -> Result<Vec<plan::Chunk>> {
    let frame_rate = cfg.mimi.frame_rate;
    let chunks = plan::chunks(tokenizer, text, normalize, MAX_TOKENS_PER_CHUNK, frame_rate)?;
    Ok(plan::fit_or_error(chunks, plan::MAX_FIT_TOKENS, tokenizer, frame_rate)?)
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
    chunks: std::vec::IntoIter<plan::Chunk>,
    current: Option<ChunkState<Q>>,
    /// One noise source for the whole text, so a seed fixes every chunk.
    rng: NormalRng,
}

/// A loaded model on the CPU, and the generation in progress on it.
struct Engine<Q: BackendQ<T = f32>> {
    model: TTSModel<Q>,
    device: Q::B,
    speaker_proj: Option<Linear<f32, Q::B>>,
    voice_states: Vec<RawState<Q::B>>,
    gen_state: Option<GenState<Q>>,
}

impl<Q: BackendQ<T = f32>> Engine<Q> {
    fn load(
        root: &Path<Q::B>,
        tokenizer: Box<dyn ptts::Tokenizer + Send + Sync>,
        cfg: &TTSConfig,
        conditions: &HashMap<String, String>,
        device: Q::B,
    ) -> Result<Self> {
        let speaker_proj = load_speaker_proj(root, cfg)?;
        let model = TTSModel::load(root, tokenizer, cfg, conditions)?;
        Ok(Self { model, device, speaker_proj, voice_states: vec![], gen_state: None })
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
        let emb = load_voice_emb_from_bytes(
            bytes,
            cfg.embeddings_ext.as_deref(),
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
        seed: u64,
        normalize: Normalize,
        cfg: &TTSConfig,
    ) -> Result<usize> {
        // Dropped before anything else can fail, so a caller that swallows the error cannot
        // go on stepping and quietly resume the *previous* utterance.
        self.gen_state = None;
        // Built here rather than after planning: a temperature that cannot produce a
        // distribution should be refused before any work is done.
        let rng = NormalRng::new(temperature, seed)?;
        let Some(voice) = self.voice_states.get(voice_index) else {
            xn::bail!("invalid voice index: {voice_index}")
        };
        let Some(tokenizer) = self.model.flow_lm.conditioner.tokenizer.as_deref() else {
            xn::bail!("this model was loaded without a tokenizer")
        };
        let chunks = plan_chunks(tokenizer, text, normalize, cfg)?;

        // The KV budget has to hold the voice prompt plus the longest chunk's text and audio.
        // This is `plan::seq_budget` with the voice's real length in place of its
        // `PROMPT_SEQ_HEADROOM` guess, the same bound `ptts::synth` checks a session against.
        let voice_len = raw_len(voice);
        let seq_budget =
            chunks.iter().map(|c| voice_len + c.tokens.len() + c.frame_budget).max().unwrap_or(0);
        let base = voice.with_seq_budget(seq_budget)?;

        let num_chunks = chunks.len();
        self.gen_state = Some(GenState { base, chunks: chunks.into_iter(), current: None, rng });
        Ok(num_chunks)
    }

    fn next_chunk(&mut self) -> Result<Option<usize>> {
        let Some(gen_state) = self.gen_state.as_mut() else { return Ok(None) };
        gen_state.current = None;
        let Some(chunk) = gen_state.chunks.next() else {
            self.gen_state = None;
            return Ok(None);
        };
        let transformer_state = gen_state.base.clone();
        let mut tts_state =
            TTSState { flow_lm_state: FlowLMState { transformer_state, conditions: None } };
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

    /// The next frame of the current chunk as PCM, or `None` once the chunk is finished.
    fn step(&mut self) -> Result<Option<Vec<f32>>> {
        let Self { model, gen_state, .. } = self;
        let Some(GenState { current, rng, .. }) = gen_state.as_mut() else { return Ok(None) };
        let Some(chunk) = current.as_mut() else { return Ok(None) };
        if chunk.done || chunk.step >= chunk.frame_budget {
            *current = None;
            return Ok(None);
        }
        let input = match &chunk.prev_latent {
            None => StepInput::Bos { batch: 1 },
            Some(t) => StepInput::Latent(t),
        };
        let (latent, eos_logit) = model.generate_step_parts(&mut chunk.tts_state, input, rng)?;
        let pcm = model.decode_latent(&latent, &mut chunk.mimi_state)?;
        let pcm = pcm.narrow(0, ..1)?.contiguous()?;
        let pcm = pcm.reshape(pcm.elem_count())?.to_vec()?;
        chunk.prev_latent = Some(latent);
        chunk.step += 1;
        // `should_stop` is asked after each frame, so that frame itself is part of the
        // output, and the next call reports the end of the chunk.
        if chunk.eos.should_stop(model.eos_from_logit(&eos_logit.to_vec()?)) {
            chunk.done = true;
        }
        Ok(Some(pcm))
    }
}

/// A voice stored as the flow LM's KV cache after the voice prompt, the format the
/// `embeddings_v2/` voices use: `transformer.layers.{i}.self_attn/cache`, shaped
/// `[2, 1, seq, heads, head_dim]`, for each layer. Nothing to run.
fn voice_from_kv_cache<B: Backend>(
    tensors: &HashMap<String, TypedTensor<B>>,
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

/// The engine for whichever weight format was loaded.
// There is one per loaded model and it never moves, so boxing the larger variant would only
// add an indirection.
#[allow(clippy::large_enum_variant)]
enum AnyEngine {
    CpuF32(Engine<Unquantized<f32, CpuDevice>>),
    CpuQ8(Engine<Q80F32>),
}

/// Run a block against the loaded engine. Within the block, `$e` is the `Engine<Q>`.
macro_rules! with_engine {
    ($engine:expr, |$e:ident| $body:expr) => {
        match $engine {
            AnyEngine::CpuF32($e) => $body,
            AnyEngine::CpuQ8($e) => $body,
        }
    };
}

/// Everything a `Model` holds.
struct Loaded {
    engine: AnyEngine,
    cfg: TTSConfig,
    /// How `start_generation` normalizes, named by the page when it loaded the model.
    normalize: Normalize,
}

impl Loaded {
    #[allow(clippy::too_many_arguments)]
    fn load(
        model_weights: Vec<u8>,
        tokenizer_json: &[u8],
        config_json: &[u8],
        quant: &str,
        lang: &str,
        rewrites: Option<&str>,
        conditions: &[String],
        device: &str,
    ) -> Result<Self> {
        match device {
            "cpu" => {}
            "webgpu" => xn::bail!(
                "this module runs on the CPU only: WebGPU runs in phonon-tts's own engine \
                 (`PhononTTS.load({{ device: 'webgpu' }})`), which takes the weights itself"
            ),
            other => xn::bail!("unknown device '{other}', expected 'cpu'"),
        }
        let quant = Quant::parse(quant)?;
        let Settings { cfg, normalize, conditions } =
            Settings::parse(config_json, lang, rewrites, conditions)?;
        let tokenizer: Box<dyn ptts::Tokenizer + Send + Sync> =
            Box::new(Tok::from_bytes(tokenizer_json)?);
        let is_gguf = model_weights.len() >= 4 && &model_weights[..4] == b"GGUF";
        console_log!("[phonon] loading model with quant={quant:?} on cpu");
        let vb = if is_gguf {
            VB::load_gguf_with_key_map(std::io::Cursor::new(model_weights), CPU, remap_key)?
        } else {
            VB::from_bytes_with_key_map(vec![model_weights], CPU, remap_key)?
        };
        let root = vb.root();
        let engine = match quant {
            Quant::F32 => {
                AnyEngine::CpuF32(Engine::load(&root, tokenizer, &cfg, &conditions, CPU)?)
            }
            Quant::Q8 => AnyEngine::CpuQ8(Engine::load(&root, tokenizer, &cfg, &conditions, CPU)?),
        };
        Ok(Self { engine, cfg, normalize })
    }
}

fn kv_cache_name(layer: usize) -> String {
    format!("transformer.layers.{layer}.self_attn/cache")
}

fn js_err(e: xn::Error) -> JsError {
    JsError::new(&e.to_string())
}

/// A loaded model, on the CPU.
///
/// The state sits behind an `Rc<RefCell<Option<_>>>` because `generation_step` returns a
/// promise and a `RefCell` borrow may not be held across one: the step takes the state out for
/// its duration and puts it back after. A call that arrives meanwhile finds it missing and is
/// refused, so calls must not overlap; `phonon-tts`'s worker awaits each one. The exception
/// is `stop_generation`, which the step applies when it puts the state back.
#[wasm_bindgen]
pub struct Model {
    loaded: Rc<RefCell<Option<Loaded>>>,
    /// Set by `stop_generation` while a step holds the model, so the stop is applied when
    /// the step hands it back instead of being lost.
    stop_requested: Rc<Cell<bool>>,
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
    /// `config.json`. All three are required.
    ///
    /// One thing a config cannot ask this build for: classifier-free guidance. Guidance is a
    /// caller's option in `ptts::synth` (`SpeechOptions::cfg_coef`), not a field of the
    /// config, and the browser build never turns it on, so `cfg_null_audio_empty` is inert.
    /// A config with baked-in `voices` is refused: voices here come from files. Everything
    /// else -- the flow LM and Mimi shapes, `lsd_decode_steps`, `eos_threshold`, `model_id`,
    /// `speaker_mimi`, `conditioners` -- is honored.
    ///
    /// `quant` is `"f32"` or `"q8"`.
    ///
    /// `lang` is required: the language text is normalized as before it is
    /// tokenized, one of `"en"`, `"fr"`, `"de"`, `"es"`, `"pt"`, `"ar"`, `"hi"`,
    /// `"ko"`, `"pl"`, `"sv"`, `"fi"`, `"no"`, `"it"`, `"nl"`, `"ja"`, `"zh"`,
    /// `"da"`, `"tr"`, or `"none"`
    /// to hand text to the tokenizer as written. The spoken forms of `@`, `+`
    /// and `=` differ per language, so there is nothing safe to default to.
    ///
    /// `rewrites` is optional and picks which word rewrites run on the
    /// normalized text: `"default"` (numbers, currency, dashed-digits, emails,
    /// urls, abbreviations, elongations), `"all"` (those and phones, times,
    /// dates), `"none"`, or a comma-separated list of rule names. The rules
    /// only read `"en"`, `"fr"`, `"de"`, `"es"` and `"pt"`.
    ///
    /// `conditions` is optional and sets the config's conditioners, each as `"NAME=VALUE"`,
    /// e.g. `["duration_delta=0.5"]`. Those left out take their defaults.
    ///
    /// `device` must be `"cpu"`. WebGPU is not in this module: `phonon-tts` runs its own
    /// engine for it, with a [`Preparer`] for the text and the voices.
    #[allow(clippy::too_many_arguments)]
    pub async fn load(
        model_weights: Vec<u8>,
        tokenizer_json: Vec<u8>,
        config_json: Vec<u8>,
        quant: String,
        lang: String,
        rewrites: Option<String>,
        conditions: Option<Vec<String>>,
        device: String,
    ) -> std::result::Result<Model, JsError> {
        let loaded = Loaded::load(
            model_weights,
            &tokenizer_json,
            &config_json,
            &quant,
            &lang,
            rewrites.as_deref(),
            &conditions.unwrap_or_default(),
            &device,
        )
        .map_err(js_err)?;
        Ok(Model { loaded: Rc::new(RefCell::new(Some(loaded))), stop_requested: Rc::default() })
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
        seed: u64,
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

    /// Generates and decodes the next frame of the current chunk: a promise of mono PCM at
    /// `sample_rate`, one `frame_size`-sample frame, or of `undefined` when the chunk is
    /// finished.
    pub fn generation_step(&self) -> js_sys::Promise {
        let cell = Rc::clone(&self.loaded);
        let stop_requested = Rc::clone(&self.stop_requested);
        wasm_bindgen_futures::future_to_promise(async move {
            let Some(mut loaded) = cell.borrow_mut().take() else {
                return Err(
                    JsError::new("the model is busy: a generation step is still running").into()
                );
            };
            let result = with_engine!(&mut loaded.engine, |e| e.step());
            Self::drop_generation_on_error(&mut loaded, &result);
            if stop_requested.take() {
                with_engine!(&mut loaded.engine, |e| e.gen_state = None);
            }
            *cell.borrow_mut() = Some(loaded);
            match result {
                Ok(Some(pcm)) => Ok(js_sys::Float32Array::from(pcm.as_slice()).into()),
                Ok(None) => Ok(JsValue::UNDEFINED),
                Err(e) => Err(js_err(e).into()),
            }
        })
    }

    /// Kept for callers written when this module also ran on WebGPU, where a voice's prompt
    /// was only recorded until flushed. The CPU has nothing queued: resolves at once.
    pub fn flush(&self) -> js_sys::Promise {
        js_sys::Promise::resolve(&JsValue::UNDEFINED)
    }

    /// Drops the generation in progress, if any. Called while a `generation_step` is pending,
    /// it takes effect when that step returns: the step still delivers its frame, and the
    /// next one reports the end.
    pub fn stop_generation(&self) {
        match self.loaded.borrow_mut().as_mut() {
            Some(l) => with_engine!(&mut l.engine, |e| e.gen_state = None),
            // A step holds the model: it drops the generation when it hands the model back.
            None => self.stop_requested.set(true),
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

    /// The languages the model was trained for, as listed in its config.
    pub fn languages(&self) -> std::result::Result<Vec<String>, JsError> {
        self.with(|l| Ok(l.cfg.languages.clone())).map_err(js_err)
    }

    /// `"cpu"`: where this model runs.
    pub fn device(&self) -> std::result::Result<String, JsError> {
        Ok("cpu".to_string())
    }
}

/// The parts of a checkpoint the WebGPU engine needs from Rust, without the model itself: the
/// tokenizer and the text normalization ([`plan::chunks`], as every other frontend splits
/// text), the voice files, and the sum of the config's conditioners.
///
/// Its tensors are the few the conditioners and the speaker projection need, which the caller
/// takes out of the checkpoint (`flow_lm.condition_provider.conditioners.*` in the GGUF, under
/// their GGUF names, as a safetensors file). The transformer and Mimi weights never come here.
#[wasm_bindgen]
pub struct Preparer {
    tok: Tok,
    cfg: TTSConfig,
    normalize: Normalize,
    conditions: Vec<f32>,
    speaker_proj: Option<Linear<f32, CpuDevice>>,
}

#[wasm_bindgen]
impl Preparer {
    /// `tokenizer_json`, `config_json`, `lang`, `rewrites` and `conditions` as for
    /// [`Model::load`]; `tensors` a safetensors file with the conditioner tensors.
    #[wasm_bindgen(constructor)]
    pub fn new(
        tokenizer_json: &[u8],
        config_json: &[u8],
        tensors: Vec<u8>,
        lang: &str,
        rewrites: Option<String>,
        conditions: Option<Vec<String>>,
    ) -> std::result::Result<Preparer, JsError> {
        Self::build(tokenizer_json, config_json, tensors, lang, rewrites, conditions)
            .map_err(js_err)
    }

    /// The vector every frame's input adds: the conditioners at their values, `d_model` long
    /// (zeros for a checkpoint without conditioners).
    pub fn conditions(&self) -> Vec<f32> {
        self.conditions.clone()
    }

    /// `text` as the model is prompted with it: an array of chunks, each
    /// `{ tokens: Uint32Array, framesAfterEos, frameBudget }`.
    pub fn plan(&self, text: &str) -> std::result::Result<js_sys::Array, JsError> {
        let chunks = plan_chunks(&self.tok, text, self.normalize, &self.cfg).map_err(js_err)?;
        // The text table's rows, as `LUTConditioner::embed_tokens` checks them: an id past it is
        // a tokenizer from another checkpoint.
        let rows = self.cfg.flow_lm.n_bins + 1;
        let out = js_sys::Array::new();
        for c in chunks {
            if let Some(&id) = c.tokens.iter().find(|&&id| id as usize >= rows) {
                return Err(JsError::new(&format!(
                    "token id {id} is past this checkpoint's {rows}-entry embedding table: the \
                     tokenizer.json does not belong to this checkpoint"
                )));
            }
            let o = js_sys::Object::new();
            let set = |k: &str, v: JsValue| js_sys::Reflect::set(&o, &k.into(), &v).map(|_| ());
            set("tokens", js_sys::Uint32Array::from(c.tokens.as_slice()).into())
                .and_then(|_| set("framesAfterEos", (c.frames_after_eos as f64).into()))
                .and_then(|_| set("frameBudget", (c.frame_budget as f64).into()))
                .map_err(|_| JsError::new("cannot build a chunk"))?;
            out.push(&o);
        }
        Ok(out)
    }

    /// A voice file's prompt: `{ kind: 'rows', frames, rows }`, the rows the flow LM is
    /// prompted with (`[frames, d_model]`, for an `emb` voice or `speaker_wavs` latents through
    /// the speaker projection), or `{ kind: 'kv', frames, k, v }` for a voice stored as the KV
    /// cache (`embeddings_v2/`): one `[frames, heads * head_dim]` array per layer in `k` and `v`.
    pub fn voice(&self, bytes: &[u8]) -> std::result::Result<js_sys::Object, JsError> {
        self.voice_inner(bytes).map_err(js_err)
    }

    /// The languages the model was trained for, as listed in its config.
    pub fn languages(&self) -> Vec<String> {
        self.cfg.languages.clone()
    }

    pub fn sample_rate(&self) -> usize {
        self.cfg.mimi.sample_rate
    }

    /// Samples in one 80 ms frame.
    pub fn frame_size(&self) -> usize {
        (self.cfg.mimi.sample_rate as f64 / self.cfg.mimi.frame_rate).round() as usize
    }
}

impl Preparer {
    fn build(
        tokenizer_json: &[u8],
        config_json: &[u8],
        tensors: Vec<u8>,
        lang: &str,
        rewrites: Option<String>,
        conditions: Option<Vec<String>>,
    ) -> Result<Self> {
        let Settings { cfg, normalize, conditions } = Settings::parse(
            config_json,
            lang,
            rewrites.as_deref(),
            &conditions.unwrap_or_default(),
        )?;
        let tok = Tok::from_bytes(tokenizer_json)?;
        let vb = VB::from_bytes_with_key_map(vec![tensors], CPU, remap_key)?;
        let root = vb.root();
        let summed: Option<Tensor<f32, CpuDevice>> = load_conditions(&root, &cfg, &conditions)?;
        let conditions = match summed {
            Some(t) => t.flatten_all()?.to_vec()?,
            None => vec![0.0; cfg.flow_lm.d_model],
        };
        let speaker_proj = load_speaker_proj(&root, &cfg)?;
        Ok(Self { tok, cfg, normalize, conditions, speaker_proj })
    }

    fn voice_inner(&self, bytes: &[u8]) -> Result<js_sys::Object> {
        let o = js_sys::Object::new();
        let set = |k: &str, v: JsValue| {
            js_sys::Reflect::set(&o, &k.into(), &v)
                .map(|_| ())
                .map_err(|_| xn::Error::msg("cannot build the voice"))
        };
        let tensors = xn::safetensors::load_from_buffer(bytes, &CPU)?;
        if tensors.contains_key(&kv_cache_name(0)) {
            let state = voice_from_kv_cache(&tensors, &self.cfg)?;
            let (k, v) = (js_sys::Array::new(), js_sys::Array::new());
            for layer in &state.layer_states {
                let LayerAttentionState::FlowLm(m) = layer else {
                    xn::bail!("unexpected attention state")
                };
                k.push(&js_sys::Float32Array::from(m.k_cache.flatten_all()?.to_vec()?.as_slice()));
                v.push(&js_sys::Float32Array::from(m.v_cache.flatten_all()?.to_vec()?.as_slice()));
            }
            set("kind", "kv".into())?;
            set("frames", (raw_len(&state) as f64).into())?;
            set("k", k.into())?;
            set("v", v.into())?;
        } else {
            let emb = load_voice_emb_from_bytes(
                bytes,
                self.cfg.embeddings_ext.as_deref(),
                self.speaker_proj.as_ref(),
                &CPU,
            )?;
            set("kind", "rows".into())?;
            set("frames", (emb.dim(1usize)? as f64).into())?;
            set(
                "rows",
                js_sys::Float32Array::from(emb.flatten_all()?.to_vec()?.as_slice()).into(),
            )?;
        }
        Ok(o)
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
