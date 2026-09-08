use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use ptts::tok::Tok;
use ptts::tts_model::{MimiEnc, TTSConfig, TTSModel, TTSState};
use pyo3::prelude::*;
use std::sync::Arc;
use xn::nn::VB;
use xn::{BackendQ, Tensor, Unquantized, error::Context};

struct StdRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl StdRng {
    fn new(temperature: f32, seed: u64) -> Self {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, temperature.sqrt()).unwrap();
        let inner = rand::rngs::StdRng::seed_from_u64(seed);
        Self { inner, distr }
    }
}

impl ptts::flow_lm::Rng for StdRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

trait PyRes<R> {
    fn w(self) -> PyResult<R>;
}

impl<R, E: Into<xn::Error>> PyRes<R> for Result<R, E> {
    fn w(self) -> PyResult<R> {
        self.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.into().to_string()))
    }
}

#[macro_export]
macro_rules! py_bail {
    ($msg:literal $(,)?) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($msg)))
    };
    ($err:expr $(,)?) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($err)))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($fmt, $($arg)*)))
    };
}

const POCKET_TTS_VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

/// Flatten a numpy conditioning-embedding array of shape `[T, dim]` or `[1, T, dim]`
/// into a contiguous `(data, T, dim)` triple.
fn parse_embeddings(arr: &PyReadonlyArrayDyn<'_, f32>) -> PyResult<(Vec<f32>, usize, usize)> {
    let (n_frames, dim) = match arr.shape() {
        [t, d] => (*t, *d),
        [1, t, d] => (*t, *d),
        shape => {
            py_bail!("expected embeddings of shape [T, dim] or [1, T, dim], got {shape:?}")
        }
    };
    let data: Vec<f32> = arr.as_array().iter().copied().collect();
    Ok((data, n_frames, dim))
}

struct ModelB<Q: BackendQ> {
    inner: Arc<TTSModel<Q>>,
    mimi_enc: Option<MimiEnc<Q>>,
    voices: std::collections::HashMap<String, Tensor<Q::T, Q::B>>,
    audio_prompt_min_duration: f32,
    audio_prompt_max_duration: f32,
    cfg_null_audio_empty: bool,
    speaker_sample_rate: usize,
}

impl<Q: BackendQ> ModelB<Q> {
    fn get_state_for_audio(
        &self,
        audio_prompt: &[f32],
        cfg_coef: Option<f32>,
        max_seq_len: usize,
    ) -> xn::Result<ModelStateB<Q>> {
        let sample_rate = self.speaker_sample_rate;
        let mimi_enc = match self.mimi_enc.as_ref() {
            Some(enc) => enc,
            None => xn::bail!("model does not support audio prompts"),
        };
        let min_len = (sample_rate as f32 * self.audio_prompt_min_duration).round() as usize;
        let max_len = (sample_rate as f32 * self.audio_prompt_max_duration).round() as usize;
        if audio_prompt.len() < min_len || audio_prompt.len() > max_len {
            xn::bail!(
                "audio_prompt must have between {min_len} and {max_len} samples ({}s-{}s at {sample_rate}Hz), got {}",
                self.audio_prompt_min_duration,
                self.audio_prompt_max_duration,
                audio_prompt.len()
            );
        }
        let dev = self.inner.device();
        let mut state = self.inner.init_flow_lm_state(1, max_seq_len)?;
        let pcm = if audio_prompt.is_empty() {
            None
        } else {
            let pcm = xn::Tensor::from_vec(audio_prompt.to_vec(), (1, 1, ()), dev)?.to::<Q::T>()?;
            let voice_emb = mimi_enc.encode_audio(&pcm)?;
            self.inner.prompt_audio(&mut state, &voice_emb)?;
            Some(pcm)
        };
        let cfg_state = if let Some(cfg_coef) = cfg_coef {
            let mut null_state = self.inner.init_flow_lm_state(1, max_seq_len)?;
            if !self.cfg_null_audio_empty
                && let Some(pcm) = pcm.as_ref()
            {
                let null_pcm = pcm.zeros_like()?;
                let null_emb = mimi_enc.encode_audio(&null_pcm)?;
                self.inner.prompt_audio(&mut null_state, &null_emb)?;
            }
            Some((cfg_coef, null_state))
        } else {
            None
        };
        Ok(ModelStateB { model: Arc::clone(&self.inner), state, cfg_state })
    }

    fn get_state_for_voice(&self, voice: &str, max_seq_len: usize) -> xn::Result<ModelStateB<Q>> {
        let voice_emb = match self.voices.get(voice) {
            Some(emb) => emb,
            None => {
                let available: Vec<_> = self.voices.keys().collect();
                xn::bail!("unknown voice '{voice}'. Available voices: {available:?}")
            }
        };
        let mut state = self.inner.init_flow_lm_state(1, max_seq_len)?;
        self.inner.prompt_audio(&mut state, voice_emb)?;
        Ok(ModelStateB { model: Arc::clone(&self.inner), state, cfg_state: None })
    }

    /// Build a generation state from a precomputed conditioning embedding (the mimi
    /// encoder output, shape `[T, dim]`). When `cfg_coef` is set, a CFG null state is
    /// also prepared; if the model uses non-empty null audio conditioning, the encoder
    /// output of silence must be supplied via `null_embeddings`.
    fn get_state_for_embeddings(
        &self,
        embeddings: Vec<f32>,
        n_frames: usize,
        dim: usize,
        cfg_coef: Option<f32>,
        null_embeddings: Option<(Vec<f32>, usize, usize)>,
        max_seq_len: usize,
    ) -> xn::Result<ModelStateB<Q>> {
        let dev = self.inner.device();
        let emb = Tensor::from_vec(embeddings, (1, n_frames, dim), dev)?.to::<Q::T>()?;
        let mut state = self.inner.init_flow_lm_state(1, max_seq_len)?;
        self.inner.prompt_audio(&mut state, &emb)?;
        let cfg_state = if let Some(cfg_coef) = cfg_coef {
            let mut null_state = self.inner.init_flow_lm_state(1, max_seq_len)?;
            if !self.cfg_null_audio_empty {
                match null_embeddings {
                    Some((null, n, d)) => {
                        let null_emb = Tensor::from_vec(null, (1, n, d), dev)?.to::<Q::T>()?;
                        self.inner.prompt_audio(&mut null_state, &null_emb)?;
                    }
                    None => xn::bail!(
                        "this model uses non-empty CFG null audio conditioning \
                         (cfg_null_audio_empty=false); pass `null_embeddings` (e.g. the mimi \
                         encoder output of silence of the same length) to use CFG with \
                         precomputed embeddings"
                    ),
                }
            }
            Some((cfg_coef, null_state))
        } else {
            None
        };
        Ok(ModelStateB { model: Arc::clone(&self.inner), state, cfg_state })
    }

    fn voices(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    /// Load the voice embedding stored at `voice_path` and register it under
    /// `name`, overwriting any existing voice with the same name.
    fn add_voice(&mut self, name: &str, voice_path: &std::path::Path) -> xn::Result<()> {
        let dev = self.inner.device().clone();
        let voice_emb = load_voice_embedding(voice_path, &dev)?.to::<Q::T>()?;
        self.voices.insert(name.to_string(), voice_emb);
        Ok(())
    }

    fn sample_rate(&self) -> usize {
        self.inner.sample_rate()
    }
}

// Poor man's type erasure, that's especially painful with pyo3 where
// the [pyclass] attribute is on a struct to get a proper object. The CPU
// backend supports unquantized f32 as well as the GGML quantized weight
// formats applied to the flow_lm transformer linear weights; the optional
// GPU backends (CUDA, Vulkan, Metal) stay unquantized.
enum ModelV {
    Cpu(ModelB<Unquantized<f32, xn::CpuDevice>>),
    Q80(ModelB<xn::quantized::Q80F32>),
    Q81(ModelB<xn::quantized::Q81F32>),
    Q8k(ModelB<xn::quantized::Q8kF32>),
    Q6k(ModelB<xn::quantized::Q6kF32>),
    Q50(ModelB<xn::quantized::Q50F32>),
    Q51(ModelB<xn::quantized::Q51F32>),
    Q5k(ModelB<xn::quantized::Q5kF32>),
    Q40(ModelB<xn::quantized::Q40F32>),
    Q41(ModelB<xn::quantized::Q41F32>),
    Q4k(ModelB<xn::quantized::Q4kF32>),
    #[cfg(feature = "cuda")]
    Cuda(ModelB<Unquantized<f32, xn::CudaDevice>>),
    #[cfg(feature = "vulkan")]
    Vulkan(ModelB<Unquantized<f32, xn::VulkanDevice>>),
    #[cfg(feature = "metal")]
    Metal(ModelB<Unquantized<f32, xn::MetalDevice>>),
}

/// Dispatch over a `&ModelV`, binding the inner `ModelB<Q>` to the supplied
/// identifier and evaluating `$body` for the matching backend.
macro_rules! on_model {
    ($e:expr, |$m:ident| $body:expr) => {
        match $e {
            ModelV::Cpu($m) => $body,
            ModelV::Q80($m) => $body,
            ModelV::Q81($m) => $body,
            ModelV::Q8k($m) => $body,
            ModelV::Q6k($m) => $body,
            ModelV::Q50($m) => $body,
            ModelV::Q51($m) => $body,
            ModelV::Q5k($m) => $body,
            ModelV::Q40($m) => $body,
            ModelV::Q41($m) => $body,
            ModelV::Q4k($m) => $body,
            #[cfg(feature = "cuda")]
            ModelV::Cuda($m) => $body,
            #[cfg(feature = "vulkan")]
            ModelV::Vulkan($m) => $body,
            #[cfg(feature = "metal")]
            ModelV::Metal($m) => $body,
        }
    };
}

/// Like [`on_model`] but wraps `$body` in the `ModelStateV` variant matching
/// the backend, so methods that build a new state stay backend-correct.
macro_rules! on_model_to_state {
    ($e:expr, |$m:ident| $body:expr) => {
        match $e {
            ModelV::Cpu($m) => ModelStateV::Cpu($body),
            ModelV::Q80($m) => ModelStateV::Q80($body),
            ModelV::Q81($m) => ModelStateV::Q81($body),
            ModelV::Q8k($m) => ModelStateV::Q8k($body),
            ModelV::Q6k($m) => ModelStateV::Q6k($body),
            ModelV::Q50($m) => ModelStateV::Q50($body),
            ModelV::Q51($m) => ModelStateV::Q51($body),
            ModelV::Q5k($m) => ModelStateV::Q5k($body),
            ModelV::Q40($m) => ModelStateV::Q40($body),
            ModelV::Q41($m) => ModelStateV::Q41($body),
            ModelV::Q4k($m) => ModelStateV::Q4k($body),
            #[cfg(feature = "cuda")]
            ModelV::Cuda($m) => ModelStateV::Cuda($body),
            #[cfg(feature = "vulkan")]
            ModelV::Vulkan($m) => ModelStateV::Vulkan($body),
            #[cfg(feature = "metal")]
            ModelV::Metal($m) => ModelStateV::Metal($body),
        }
    };
}

#[pyclass]
struct Model(ModelV);

#[pymethods]
impl Model {
    #[pyo3(signature = (audio_prompt, cfg_coef=None, max_seq_len=2048))]
    fn get_state_for_audio(
        &self,
        audio_prompt: PyReadonlyArray1<'_, f32>,
        cfg_coef: Option<f32>,
        max_seq_len: usize,
    ) -> PyResult<ModelState> {
        let audio_prompt = audio_prompt.as_slice()?;
        let inner = on_model_to_state!(&self.0, |m| m
            .get_state_for_audio(audio_prompt, cfg_coef, max_seq_len)
            .w()?);
        Ok(ModelState(inner))
    }

    #[pyo3(signature = (voice, max_seq_len=2048))]
    fn get_state_for_voice(&self, voice: &str, max_seq_len: usize) -> PyResult<ModelState> {
        let inner =
            on_model_to_state!(&self.0, |m| m.get_state_for_voice(voice, max_seq_len).w()?);
        Ok(ModelState(inner))
    }

    /// Build a generation state from a precomputed conditioning embedding (the mimi
    /// encoder output), with optional classifier-free guidance.
    ///
    /// `embeddings` must have shape `[T, dim]` or `[1, T, dim]`. To use CFG, pass
    /// `cfg_coef`; for models with non-empty null audio conditioning, also pass
    /// `null_embeddings` (the encoder output of silence of matching length).
    #[pyo3(signature = (embeddings, cfg_coef=None, null_embeddings=None, max_seq_len=2048))]
    fn get_state_for_embeddings(
        &self,
        embeddings: PyReadonlyArrayDyn<'_, f32>,
        cfg_coef: Option<f32>,
        null_embeddings: Option<PyReadonlyArrayDyn<'_, f32>>,
        max_seq_len: usize,
    ) -> PyResult<ModelState> {
        let (embeddings, n_frames, dim) = parse_embeddings(&embeddings)?;
        let null = match null_embeddings.as_ref() {
            Some(arr) => {
                let (data, n, d) = parse_embeddings(arr)?;
                if d != dim {
                    py_bail!(
                        "null_embeddings last dim ({d}) must match embeddings last dim ({dim})"
                    );
                }
                Some((data, n, d))
            }
            None => None,
        };
        let inner = on_model_to_state!(&self.0, |m| m
            .get_state_for_embeddings(embeddings, n_frames, dim, cfg_coef, null, max_seq_len)
            .w()?);
        Ok(ModelState(inner))
    }

    fn voices(&self) -> Vec<String> {
        on_model!(&self.0, |m| m.voices())
    }

    /// Load the voice embedding at `path` (a precomputed voice safetensors) and
    /// register it under `name` so it can be used with `get_state_for_voice`.
    fn add_voice(&mut self, name: &str, path: &str) -> PyResult<()> {
        let path = std::path::Path::new(path);
        on_model!(&mut self.0, |m| m.add_voice(name, path).w())
    }

    fn sample_rate(&self) -> usize {
        on_model!(&self.0, |m| m.sample_rate())
    }
}

struct ModelStateB<Q: BackendQ> {
    model: Arc<TTSModel<Q>>,
    state: TTSState<Q>,
    cfg_state: Option<(f32, TTSState<Q>)>,
}

enum ModelStateV {
    Cpu(ModelStateB<Unquantized<f32, xn::CpuDevice>>),
    Q80(ModelStateB<xn::quantized::Q80F32>),
    Q81(ModelStateB<xn::quantized::Q81F32>),
    Q8k(ModelStateB<xn::quantized::Q8kF32>),
    Q6k(ModelStateB<xn::quantized::Q6kF32>),
    Q50(ModelStateB<xn::quantized::Q50F32>),
    Q51(ModelStateB<xn::quantized::Q51F32>),
    Q5k(ModelStateB<xn::quantized::Q5kF32>),
    Q40(ModelStateB<xn::quantized::Q40F32>),
    Q41(ModelStateB<xn::quantized::Q41F32>),
    Q4k(ModelStateB<xn::quantized::Q4kF32>),
    #[cfg(feature = "cuda")]
    Cuda(ModelStateB<Unquantized<f32, xn::CudaDevice>>),
    #[cfg(feature = "vulkan")]
    Vulkan(ModelStateB<Unquantized<f32, xn::VulkanDevice>>),
    #[cfg(feature = "metal")]
    Metal(ModelStateB<Unquantized<f32, xn::MetalDevice>>),
}

/// Dispatch over a `&ModelStateV`, binding the inner `ModelStateB<Q>` to the
/// supplied identifier and evaluating `$body`.
macro_rules! on_state {
    ($e:expr, |$s:ident| $body:expr) => {
        match $e {
            ModelStateV::Cpu($s) => $body,
            ModelStateV::Q80($s) => $body,
            ModelStateV::Q81($s) => $body,
            ModelStateV::Q8k($s) => $body,
            ModelStateV::Q6k($s) => $body,
            ModelStateV::Q50($s) => $body,
            ModelStateV::Q51($s) => $body,
            ModelStateV::Q5k($s) => $body,
            ModelStateV::Q40($s) => $body,
            ModelStateV::Q41($s) => $body,
            ModelStateV::Q4k($s) => $body,
            #[cfg(feature = "cuda")]
            ModelStateV::Cuda($s) => $body,
            #[cfg(feature = "vulkan")]
            ModelStateV::Vulkan($s) => $body,
            #[cfg(feature = "metal")]
            ModelStateV::Metal($s) => $body,
        }
    };
}

/// Like [`on_state`] but wraps `$body` back in the matching `ModelStateV`
/// variant (used by `clone`).
macro_rules! on_state_to_state {
    ($e:expr, |$s:ident| $body:expr) => {
        match $e {
            ModelStateV::Cpu($s) => ModelStateV::Cpu($body),
            ModelStateV::Q80($s) => ModelStateV::Q80($body),
            ModelStateV::Q81($s) => ModelStateV::Q81($body),
            ModelStateV::Q8k($s) => ModelStateV::Q8k($body),
            ModelStateV::Q6k($s) => ModelStateV::Q6k($body),
            ModelStateV::Q50($s) => ModelStateV::Q50($body),
            ModelStateV::Q51($s) => ModelStateV::Q51($body),
            ModelStateV::Q5k($s) => ModelStateV::Q5k($body),
            ModelStateV::Q40($s) => ModelStateV::Q40($body),
            ModelStateV::Q41($s) => ModelStateV::Q41($body),
            ModelStateV::Q4k($s) => ModelStateV::Q4k($body),
            #[cfg(feature = "cuda")]
            ModelStateV::Cuda($s) => ModelStateV::Cuda($body),
            #[cfg(feature = "vulkan")]
            ModelStateV::Vulkan($s) => ModelStateV::Vulkan($body),
            #[cfg(feature = "metal")]
            ModelStateV::Metal($s) => ModelStateV::Metal($body),
        }
    };
}

#[pyclass]
struct ModelState(ModelStateV);

/// Returns true if a python signal has been raised, e.g. on an interrupt.
fn check_py_interrupt() -> xn::Result<()> {
    if Python::attach(|py| py.check_signals().is_err()) {
        xn::bail!("interrupted by python signal");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_generate<Q: BackendQ>(
    model: Arc<TTSModel<Q>>,
    mut state: TTSState<Q>,
    mut cfg_state: Option<(f32, TTSState<Q>)>,
    tokens: Vec<u32>,
    temperature: f32,
    seed: u64,
    frames_after_eos: usize,
    check_py_interrupts_every: Option<usize>,
) -> Result<Vec<f32>, xn::Error> {
    let device = model.device();
    let num_tokens = tokens.len();
    let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
    let mut rng = StdRng::new(temperature, seed);
    let mut mimi_state = model.init_mimi_state(1, 250)?;

    model.prompt_text(&mut state, &tokens)?;
    if let Some((_, cfg_state)) = cfg_state.as_mut() {
        model.prompt_text_null(cfg_state)?
    }

    let ldim = model.flow_lm.ldim;
    let nan_data = vec![f32::NAN; ldim];
    let mut prev_latent: Tensor<Q::T, Q::B> =
        Tensor::from_vec(nan_data, (1, 1, ldim), device)?.to::<Q::T>()?;

    let (latent_tx, latent_rx) = std::sync::mpsc::channel::<Tensor<Q::T, Q::B>>();

    let decode_model = Arc::clone(&model);
    let decode_handle = std::thread::spawn(move || -> Result<Tensor<f32, Q::B>, xn::Error> {
        let mut audio_chunks = Vec::new();
        while let Ok(latent) = latent_rx.recv() {
            let audio_chunk = decode_model.decode_latent(&latent, &mut mimi_state)?;
            audio_chunks.push(audio_chunk);
        }
        let audio_refs: Vec<_> = audio_chunks.iter().collect();
        let audio = Tensor::cat(&audio_refs, 2)?;
        let audio = audio.narrow(0, ..1)?.contiguous()?;
        Ok(audio)
    });

    let mut eos_countdown: Option<usize> = None;
    for step in 0..max_frames {
        let (next_latent, is_eos) = match cfg_state.as_mut() {
            Some((cfg_coef, cfg_state)) => {
                model.generate_step_cfg(&mut state, cfg_state, *cfg_coef, &prev_latent, &mut rng)?
            }
            None => model.generate_step(&mut state, &prev_latent, &mut rng)?,
        };
        // Rather than raising an error when the channel is closed, we break out of the loop
        // so as to get a proper error message from the decode thread.
        if latent_tx.send(next_latent.clone()).is_err() {
            break;
        }

        if is_eos && eos_countdown.is_none() {
            eos_countdown = Some(frames_after_eos);
        }
        if let Some(ref mut countdown) = eos_countdown {
            if *countdown == 0 {
                break;
            }
            *countdown -= 1;
        }
        prev_latent = next_latent;
        if let Some(check_step) = check_py_interrupts_every.as_ref()
            && step % check_step == 0
        {
            check_py_interrupt()?
        }
    }
    drop(latent_tx);

    let audio = decode_handle.join().map_err(|_| xn::Error::msg("decode thread panicked"))??;
    let pcm = audio.to_vec()?;
    Ok(pcm)
}

/// Do not implement Clone on this as it would result in switching is_done
/// if one of the clone drops.
struct Recv {
    pcm_rx: std::sync::mpsc::Receiver<Vec<f32>>,
    is_done: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for Recv {
    fn drop(&mut self) {
        self.is_done.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_generate_background<Q: BackendQ>(
    model: Arc<TTSModel<Q>>,
    mut state: TTSState<Q>,
    mut cfg_state: Option<(f32, TTSState<Q>)>,
    tokens: Vec<u32>,
    temperature: f32,
    seed: u64,
    frames_after_eos: usize,
) -> Result<Recv, xn::Error> {
    let device = model.device();
    let num_tokens = tokens.len();
    let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
    let mut rng = StdRng::new(temperature, seed);
    let mut mimi_state = model.init_mimi_state(1, 250)?;

    model.prompt_text(&mut state, &tokens)?;
    if let Some((_, cfg_state)) = cfg_state.as_mut() {
        model.prompt_text_null(cfg_state)?
    }

    let ldim = model.flow_lm.ldim;
    let nan_data = vec![f32::NAN; ldim];
    let mut prev_latent: Tensor<Q::T, Q::B> =
        Tensor::from_vec(nan_data, (1, 1, ldim), device)?.to::<Q::T>()?;

    let (latent_tx, latent_rx) = std::sync::mpsc::channel::<Tensor<Q::T, Q::B>>();
    let (pcm_tx, pcm_rx) = std::sync::mpsc::channel::<Vec<f32>>();

    let decode_model = Arc::clone(&model);
    let is_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _decode_handle = std::thread::spawn(move || -> Result<(), xn::Error> {
        while let Ok(latent) = latent_rx.recv() {
            let audio_chunk = decode_model.decode_latent(&latent, &mut mimi_state)?;
            let pcm = audio_chunk.narrow(0, ..1)?.contiguous()?.to_vec()?;
            pcm_tx.send(pcm).map_err(|_| xn::Error::msg("pcm channel closed"))?;
        }
        Ok(())
    });

    let recv = Recv { pcm_rx, is_done: is_done.clone() };

    let _backbone_handle = std::thread::spawn(move || -> Result<(), xn::Error> {
        let mut eos_countdown: Option<usize> = None;
        for _step in 0..max_frames {
            if is_done.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            let (next_latent, is_eos) = match cfg_state.as_mut() {
                Some((cfg_coef, cfg_state)) => model.generate_step_cfg(
                    &mut state,
                    cfg_state,
                    *cfg_coef,
                    &prev_latent,
                    &mut rng,
                )?,
                None => model.generate_step(&mut state, &prev_latent, &mut rng)?,
            };
            // Rather than raising an error when the channel is closed, we break out of the loop
            // so as to get a proper error message from the decode thread.
            if latent_tx.send(next_latent.clone()).is_err() {
                break;
            }

            if is_eos && eos_countdown.is_none() {
                eos_countdown = Some(frames_after_eos);
            }
            if let Some(ref mut countdown) = eos_countdown {
                if *countdown == 0 {
                    break;
                }
                *countdown -= 1;
            }
            prev_latent = next_latent;
        }
        drop(latent_tx);
        Ok(())
    });
    Ok(recv)
}

/// Streaming handle over a background generation, iterable from Python. The
/// pcm receiver sits behind a mutex so the pyclass is `Sync`; `is_done` is
/// kept outside it so `cancel` never blocks on a pending `__next__`.
#[pyclass]
struct Receiver {
    inner: std::sync::Mutex<Recv>,
    is_done: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Receiver {
    fn new(recv: Recv) -> Self {
        let is_done = recv.is_done.clone();
        Self { inner: std::sync::Mutex::new(recv), is_done }
    }
}

#[pymethods]
impl Receiver {
    /// Stop the background generation threads. Chunks already decoded can
    /// still be drained via `__next__`.
    fn cancel(&self) {
        self.is_done.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let inner = &self.inner;
        let pcm = py.detach(move || {
            let inner = match inner.lock() {
                Ok(inner) => inner,
                Err(_) => return None,
            };
            inner.pcm_rx.recv().ok()
        });
        Ok(pcm.map(|pcm| PyArray1::from_vec(py, pcm)))
    }
}

impl<Q: BackendQ> ModelStateB<Q> {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            state: self.state.clone(),
            cfg_state: self.cfg_state.clone(),
        }
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        self.model.flow_lm.conditioner.tokenize(text).w()
    }

    fn generate_audio<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        temperature: f32,
        seed: u64,
        pad_to: Option<usize>,
        check_py_interrupts_every: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = Arc::clone(&self.model);
        let state = self.state.clone();
        let cfg_state = self.cfg_state.clone();
        let text = text.to_string();

        let pcm = py
            .detach(move || -> xn::Result<Vec<f32>> {
                let (text, frames_after_eos) = ptts::tts_model::prepare_text_prompt(&text);
                let mut tokens = model.flow_lm.conditioner.tokenize(&text)?;
                if let Some(pad_to) = pad_to
                    && tokens.len() < pad_to
                {
                    let learnt_padding_id = match model.flow_lm.conditioner.learnt_padding_id() {
                        Some(id) => id,
                        None => {
                            xn::bail!("model does not have a learnt padding token, cannot pad to {pad_to} tokens")
                        }
                    };
                    tokens.resize(pad_to, learnt_padding_id);
                }
                run_generate(model, state, cfg_state, tokens, temperature, seed, frames_after_eos, check_py_interrupts_every)
            })
            .w()?;

        Ok(PyArray1::from_vec(py, pcm))
    }

    fn generate_bt(
        &self,
        py: Python<'_>,
        text: &str,
        temperature: f32,
        seed: u64,
        pad_to: Option<usize>,
    ) -> PyResult<Receiver> {
        let model = Arc::clone(&self.model);
        let state = self.state.clone();
        let cfg_state = self.cfg_state.clone();
        let text = text.to_string();

        let recv = py
            .detach(move || -> xn::Result<Recv> {
                let (text, frames_after_eos) = ptts::tts_model::prepare_text_prompt(&text);
                let mut tokens = model.flow_lm.conditioner.tokenize(&text)?;
                if let Some(pad_to) = pad_to
                    && tokens.len() < pad_to
                {
                    let learnt_padding_id = match model.flow_lm.conditioner.learnt_padding_id() {
                        Some(id) => id,
                        None => {
                            xn::bail!("model does not have a learnt padding token, cannot pad to {pad_to} tokens")
                        }
                    };
                    tokens.resize(pad_to, learnt_padding_id);
                }
                run_generate_background(model, state, cfg_state, tokens, temperature, seed, frames_after_eos)
            })
            .w()?;

        Ok(Receiver::new(recv))
    }

    fn generate_audio_for_tokens<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<i32>,
        temperature: f32,
        seed: u64,
        frames_after_eos: usize,
        check_py_interrupts_every: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = Arc::clone(&self.model);
        let state = self.state.clone();
        let cfg_state = self.cfg_state.clone();
        let mut new_tokens = Vec::with_capacity(tokens.len());
        for token in tokens {
            if token < 0 {
                let token = match model.flow_lm.conditioner.learnt_padding_id() {
                    Some(id) => id,
                    None => py_bail!(
                        "model does not have a learnt padding token, cannot use negative token value {token}"
                    ),
                };
                new_tokens.push(token);
            } else {
                new_tokens.push(token as u32);
            }
        }
        let pcm = py
            .detach(move || -> xn::Result<Vec<f32>> {
                run_generate(
                    model,
                    state,
                    cfg_state,
                    new_tokens,
                    temperature,
                    seed,
                    frames_after_eos,
                    check_py_interrupts_every,
                )
            })
            .w()?;

        Ok(PyArray1::from_vec(py, pcm))
    }
}

#[pymethods]
impl ModelState {
    fn clone(&self) -> Self {
        ModelState(on_state_to_state!(&self.0, |s| s.clone()))
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        on_state!(&self.0, |s| s.tokenize(text))
    }

    #[pyo3(signature = (text, temperature=0.5, seed=4242424242424242, pad_to=None, check_py_interrupts_every=5))]
    fn generate_audio<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        temperature: f32,
        seed: u64,
        pad_to: Option<usize>,
        check_py_interrupts_every: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        on_state!(&self.0, |s| s.generate_audio(
            py,
            text,
            temperature,
            seed,
            pad_to,
            check_py_interrupts_every
        ))
    }

    /// Generate audio on background threads, returning a `Receiver` that can
    /// be iterated over to get the audio chunks as they get decoded, or
    /// cancelled to stop the generation early.
    #[pyo3(signature = (text, temperature=0.5, seed=4242424242424242, pad_to=None))]
    fn generate_bt(
        &self,
        py: Python<'_>,
        text: &str,
        temperature: f32,
        seed: u64,
        pad_to: Option<usize>,
    ) -> PyResult<Receiver> {
        on_state!(&self.0, |s| s.generate_bt(py, text, temperature, seed, pad_to))
    }

    #[pyo3(signature = (tokens, temperature=0.5, seed=4242424242424242, frames_after_eos=1, check_py_interrupts_every=5))]
    fn generate_audio_for_tokens<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<i32>,
        temperature: f32,
        seed: u64,
        frames_after_eos: usize,
        check_py_interrupts_every: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        on_state!(&self.0, |s| s.generate_audio_for_tokens(
            py,
            tokens,
            temperature,
            seed,
            frames_after_eos,
            check_py_interrupts_every
        ))
    }
}

fn remap_key(name: &str) -> Option<String> {
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
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

fn load_voice_embedding<B: xn::Backend>(
    voice_path: &std::path::Path,
    device: &B,
) -> Result<Tensor<f32, B>, xn::Error> {
    let voice_vb = VB::load(&[voice_path], device.clone())?;
    let voice_names = voice_vb.tensor_names();
    let voice_key = voice_names.first().context("no tensors found in voice embedding file")?;
    let voice_shape = voice_vb.shape(voice_key).context("voice tensor not found")?;
    let voice_dims = voice_shape.dims();
    let voice_emb: Tensor<f32, B> = voice_vb.tensor(voice_key, voice_shape.clone())?;
    if voice_dims.len() == 2 {
        Ok(voice_emb.reshape((1, voice_dims[0], voice_dims[1]))?)
    } else {
        Ok(voice_emb)
    }
}

fn load_model_<Q: BackendQ>(
    temperature: f32,
    repo_id: String,
    model_file: String,
    config: Option<String>,
    eos_threshold: Option<f32>,
    dev: Q::B,
) -> xn::Result<ModelB<Q>> {
    let (model_path, tokenizer_path, cfg, voices) = match config {
        // A local config path: load the weights sitting next to it.
        Some(config_path)
            if std::path::Path::new(&config_path).is_file() || config_path.ends_with(".json") =>
        {
            let config_path = std::fs::canonicalize(&config_path).map_err(xn::Error::msg)?;
            let parent = config_path.parent().context("config path has no parent")?;
            // Prefer an unquantized safetensors checkpoint, falling back to a
            // pre-quantized GGUF if that's what sits next to the config.
            let model_path = if parent.join("model.safetensors").is_file() {
                parent.join("model.safetensors")
            } else {
                parent.join("model.q8.gguf")
            };
            let tokenizer_path = parent.join("tokenizer.json");
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| xn::Error::msg(e).with_path(&config_path))?;
            let cfg: TTSConfig = serde_json::from_str(&config_str)
                .map_err(|e| xn::Error::msg(e).with_path(&config_path))?;
            (model_path, tokenizer_path, cfg, std::collections::HashMap::new())
        }
        // A HuggingFace repo id: download `config.json`, the quantized
        // `model.q8.gguf`, the tokenizer and the `default-voice` embedding.
        Some(repo_id) => {
            use hf_hub::{Repo, RepoType, api::sync::Api};

            let api = Api::new().map_err(xn::Error::msg)?;
            let repo = api.repo(Repo::new(repo_id, RepoType::Model));

            let config_path = repo.get("config.json").map_err(xn::Error::msg)?;
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| xn::Error::msg(e).with_path(&config_path))?;
            let mut cfg: TTSConfig = serde_json::from_str(&config_str)
                .map_err(|e| xn::Error::msg(e).with_path(&config_path))?;
            cfg.temp = temperature;

            let model_path = repo.get("model.q8.gguf").map_err(xn::Error::msg)?;
            let tokenizer_path = repo.get("tokenizer.json").map_err(xn::Error::msg)?;

            (model_path, tokenizer_path, cfg, std::collections::HashMap::new())
        }
        None => {
            use hf_hub::{Repo, RepoType, api::sync::Api};

            let api = Api::new().map_err(xn::Error::msg)?;
            let repo = api.repo(Repo::new(repo_id, RepoType::Model));

            let model_path =
                repo.get(&model_file).map_err(|e| xn::Error::msg(e).with_path(&model_file))?;
            let tokenizer_path = repo.get("tokenizer.json").map_err(xn::Error::msg)?;

            let mut voices = std::collections::HashMap::new();
            for &voice in POCKET_TTS_VOICES {
                let voice_file = format!("embeddings/{voice}.safetensors");
                if let Ok(voice_path) = repo.get(&voice_file)
                    && let Ok(voice_emb) = load_voice_embedding(&voice_path, &dev)
                    && let Ok(voice_emb) = voice_emb.to::<Q::T>()
                {
                    voices.insert(voice.to_string(), voice_emb);
                }
            }

            let cfg = TTSConfig::v202601(temperature);
            (model_path, tokenizer_path, cfg, voices)
        }
    };

    let tokenizer = Tok::open(&tokenizer_path)?;

    // GGUF checkpoints carry weights already quantized; safetensors are
    // (re)quantized into `Q` on load.
    let vb = if model_path.extension().and_then(|v| v.to_str()) == Some("gguf") {
        let reader = std::fs::File::open(&model_path)
            .map_err(|e| xn::Error::msg(e).with_path(&model_path))?;
        let reader = std::io::BufReader::new(reader);
        VB::load_gguf_with_key_map(reader, dev, remap_key).map_err(|e| e.with_path(&model_path))?
    } else {
        VB::load_with_key_map(&[&model_path], dev, remap_key)
            .map_err(|e| e.with_path(&model_path))?
    }
    .root();
    let model = TTSModel::<Q>::load(&vb, Box::new(tokenizer), &cfg)?;
    let model = if let Some(eos_threshold) = eos_threshold {
        model.with_eos_threshold(eos_threshold)
    } else {
        model
    };
    // Probe under the speaker prefix: a dedicated speaker codec ships its
    // encoder as `<speaker_mimi.prefix>.encoder.*` with no `mimi.encoder.*`.
    let enc_probe = format!("{}.encoder.model.0.conv.weight", cfg.speaker_mimi_prefix());
    let mimi_enc =
        if vb.contains(&enc_probe) { Some(MimiEnc::<Q>::load(&vb, &cfg)?) } else { None };
    vb.check_all_used_with_ignore(|v| {
        v == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
            || v.starts_with("mimi.quantizer")
    })?;

    Ok(ModelB {
        inner: Arc::new(model),
        mimi_enc,
        voices,
        audio_prompt_min_duration: cfg.audio_prompt_min_duration,
        audio_prompt_max_duration: cfg.audio_prompt_max_duration,
        cfg_null_audio_empty: cfg.cfg_null_audio_empty,
        speaker_sample_rate: cfg.speaker_mimi_cfg().sample_rate,
    })
}

#[pyfunction]
#[pyo3(signature = (temperature=0.5, repo_id="kyutai/pocket-tts", model_file="tts_b6369a24.safetensors", config=None, eos_threshold=None, device=None, quant=None))]
#[allow(clippy::too_many_arguments)]
fn load_model(
    py: Python<'_>,
    temperature: f32,
    repo_id: &str,
    model_file: &str,
    config: Option<&str>,
    eos_threshold: Option<f32>,
    device: Option<&str>,
    quant: Option<&str>,
) -> PyResult<Model> {
    let repo_id = repo_id.to_string();
    let model_file = model_file.to_string();
    let config = config.map(|s| s.to_string());
    let quant = quant.map(|s| s.to_string());
    py.detach(move || {
        // `quant` selects a GGML weight format for the flow_lm transformer
        // linear weights; it is only supported on the CPU backend.
        macro_rules! load_cpu {
            ($variant:ident, $q:ty) => {{
                let model = load_model_::<$q>(
                    temperature,
                    repo_id,
                    model_file,
                    config,
                    eos_threshold,
                    xn::CpuDevice,
                )?;
                Ok(Model(ModelV::$variant(model)))
            }};
        }
        match device {
            None | Some("cpu") => match quant.as_deref() {
                None => load_cpu!(Cpu, Unquantized<f32, xn::CpuDevice>),
                Some("q8" | "q8_0") => load_cpu!(Q80, xn::quantized::Q80F32),
                Some("q8_1") => load_cpu!(Q81, xn::quantized::Q81F32),
                Some("q8k") => load_cpu!(Q8k, xn::quantized::Q8kF32),
                Some("q6k") => load_cpu!(Q6k, xn::quantized::Q6kF32),
                Some("q5" | "q5_0") => load_cpu!(Q50, xn::quantized::Q50F32),
                Some("q5_1") => load_cpu!(Q51, xn::quantized::Q51F32),
                Some("q5k") => load_cpu!(Q5k, xn::quantized::Q5kF32),
                Some("q4" | "q4_0") => load_cpu!(Q40, xn::quantized::Q40F32),
                Some("q4_1") => load_cpu!(Q41, xn::quantized::Q41F32),
                Some("q4k") => load_cpu!(Q4k, xn::quantized::Q4kF32),
                Some(other) => xn::bail!("unsupported quant '{other}'"),
            },
            #[cfg(feature = "cuda")]
            Some("cuda") => {
                if quant.is_some() {
                    return Err(xn::Error::msg(
                        "quant cannot be combined with cuda; quantization is CPU-only",
                    ));
                }
                let dev = xn::CudaDevice::new(0)?;
                let model = load_model_::<Unquantized<f32, xn::CudaDevice>>(
                    temperature,
                    repo_id,
                    model_file,
                    config,
                    eos_threshold,
                    dev,
                )?;
                Ok(Model(ModelV::Cuda(model)))
            }
            #[cfg(feature = "vulkan")]
            Some("vulkan") => {
                if quant.is_some() {
                    return Err(xn::Error::msg(
                        "quant cannot be combined with vulkan; quantization is CPU-only",
                    ));
                }
                let dev = xn::VulkanDevice::new(0)?;
                let model = load_model_::<Unquantized<f32, xn::VulkanDevice>>(
                    temperature,
                    repo_id,
                    model_file,
                    config,
                    eos_threshold,
                    dev,
                )?;
                Ok(Model(ModelV::Vulkan(model)))
            }
            #[cfg(feature = "metal")]
            Some("metal") => {
                if quant.is_some() {
                    return Err(xn::Error::msg(
                        "quant cannot be combined with metal; quantization is CPU-only",
                    ));
                }
                let dev = xn::MetalDevice::new(0)?;
                let model = load_model_::<Unquantized<f32, xn::MetalDevice>>(
                    temperature,
                    repo_id,
                    model_file,
                    config,
                    eos_threshold,
                    dev,
                )?;
                Ok(Model(ModelV::Metal(model)))
            }
            Some(d) => xn::bail!("unknown device '{d}'"),
        }
    })
    .w()
}

#[pyfunction]
fn get_num_threads() -> usize {
    xn::utils::get_num_threads()
}

#[pyfunction]
fn set_num_threads(num_threads: usize) {
    xn::utils::set_num_threads(num_threads);
}

#[pyfunction]
fn prepare_text_prompt(text: &str) -> String {
    ptts::tts_model::prepare_text_prompt(text).0
}

#[pyfunction]
fn cuda_available() -> bool {
    cfg!(feature = "cuda")
}

#[pyfunction]
fn vulkan_available() -> bool {
    cfg!(feature = "vulkan")
}

#[pyfunction]
fn metal_available() -> bool {
    cfg!(feature = "metal")
}

#[pymodule(name = "ptts")]
fn ptts_(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<ModelState>()?;
    m.add_class::<Receiver>()?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_text_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(vulkan_available, m)?)?;
    m.add_function(wrap_pyfunction!(metal_available, m)?)?;
    Ok(())
}
