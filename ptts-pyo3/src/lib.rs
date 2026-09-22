//! Python bindings for Pocket TTS.
//!
//! ```python
//! import ptts
//!
//! tts = ptts.TTS()
//! tts.save("out.wav", "Hello world")
//! ```
//!
//! The pipeline is `ptts::synth::Synth`, so this file is a translation layer:
//! locating a checkpoint, numpy in and out, releasing the GIL around the slow
//! parts, and letting Ctrl-C through between audio chunks.
//!
//! One invariant throughout: **never hold a lock across a GIL reacquisition**.
//! A thread parked in `py.detach` with a guard alive deadlocks against any
//! other Python thread wanting the same lock, and Ctrl-C reaches neither.

use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use ptts::synth::{DeviceKind, Quant, SpeechOptions, SpeechStream, Synth, SynthBuilder};
use ptts::tts_model::TTSConfig;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};

/// Voices the published checkpoint ships, used to name the files to fetch.
const POCKET_TTS_VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

const DEFAULT_REPO_ID: &str = "kyutai/pocket-tts";
const DEFAULT_MODEL_FILE: &str = "tts_b6369a24.safetensors";

/// Map a `ptts` error onto a Python exception. `ValueError` throughout: every
/// failure here is a bad argument, a missing file or a bad checkpoint.
trait IntoPy<R> {
    fn py(self) -> PyResult<R>;
}

impl<R, E: Into<xn::Error>> IntoPy<R> for Result<R, E> {
    fn py(self) -> PyResult<R> {
        self.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.into().to_string()))
    }
}

/// A checkpoint's files, located but not yet loaded.
struct Artifacts {
    cfg: TTSConfig,
    model_path: std::path::PathBuf,
    tokenizer_path: std::path::PathBuf,
    voices: Vec<(String, std::path::PathBuf)>,
}

/// Resolve `config` — a local `config.json`, a Hub repo id, or nothing for the
/// published checkpoint — into the files needed to load it.
fn resolve(config: Option<&str>, temperature: f32) -> xn::Result<Artifacts> {
    use xn::error::Context;

    match config {
        // A local config path: load the weights sitting next to it.
        Some(path) if std::path::Path::new(path).is_file() || path.ends_with(".json") => {
            let config_path = std::fs::canonicalize(path)
                .map_err(|e| xn::Error::msg(format!("cannot read config {path}: {e}")))?;
            let parent = config_path.parent().context("config path has no parent")?;
            // Prefer an unquantized safetensors checkpoint, falling back to a
            // pre-quantized GGUF if that is what sits next to the config.
            let model_path = if parent.join("model.safetensors").is_file() {
                parent.join("model.safetensors")
            } else {
                parent.join("model.q8.gguf")
            };
            let text = std::fs::read_to_string(&config_path)
                .map_err(|e| xn::Error::msg(e).with_path(&config_path))?;
            let mut cfg: TTSConfig = serde_json::from_str(&text)
                .map_err(|e| xn::Error::msg(e).with_path(&config_path))?;
            cfg.temp = temperature;
            let mut voices = vec![];
            collect_voices(&parent.join("voices"), &mut voices);
            Ok(Artifacts { cfg, model_path, tokenizer_path: parent.join("tokenizer.json"), voices })
        }
        // A Hub repo laid out with config.json and a quantized checkpoint.
        Some(repo_id) => {
            let repo = hub(repo_id)?;
            let config_path = hub_get(&repo, "config.json")?;
            let text = std::fs::read_to_string(&config_path)
                .map_err(|e| xn::Error::msg(e).with_path(&config_path))?;
            let mut cfg: TTSConfig = serde_json::from_str(&text)
                .map_err(|e| xn::Error::msg(e).with_path(&config_path))?;
            cfg.temp = temperature;
            Ok(Artifacts {
                cfg,
                model_path: hub_get(&repo, "model.q8.gguf")?,
                tokenizer_path: hub_get(&repo, "tokenizer.json")?,
                voices: vec![],
            })
        }
        // The published pocket-tts repo, whose voices sit under `embeddings/`.
        None => {
            let repo = hub(DEFAULT_REPO_ID)?;
            let model_path = hub_get(&repo, DEFAULT_MODEL_FILE)?;
            let tokenizer_path = hub_get(&repo, "tokenizer.json")?;
            let mut voices = vec![];
            for &voice in POCKET_TTS_VOICES {
                if let Ok(path) = hub_get(&repo, &format!("embeddings/{voice}.safetensors")) {
                    voices.push((voice.to_string(), path));
                }
            }
            Ok(Artifacts {
                cfg: TTSConfig::v202601(temperature),
                model_path,
                tokenizer_path,
                voices,
            })
        }
    }
}

type HubRepo = hf_hub::HFRepositorySync<hf_hub::repository::RepoTypeModel>;

fn hub(repo_id: &str) -> xn::Result<HubRepo> {
    let client = hf_hub::HFClientSync::new().map_err(xn::Error::msg)?;
    let (owner, name) = hf_hub::split_id(repo_id);
    Ok(client.model(owner, name))
}

/// Download `filename` from `repo`, or find it in the local cache.
fn hub_get(repo: &HubRepo, filename: &str) -> xn::Result<std::path::PathBuf> {
    repo.download_file()
        .filename(filename)
        .send()
        .map_err(|e| xn::Error::msg(e).with_path(filename))
}

/// Add every `*.safetensors` file in `dir` to `voices`, keyed by file stem.
/// A missing directory is not an error: voices are optional.
fn collect_voices(dir: &std::path::Path, voices: &mut Vec<(String, std::path::PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
            voices.push((name.to_string(), path));
        }
    }
}

/// Flatten a conditioning embedding of shape `[T, dim]` or `[1, T, dim]`.
fn embedding_dims(arr: &PyReadonlyArrayDyn<'_, f32>) -> PyResult<(Vec<f32>, usize, usize)> {
    let (frames, dim) = match arr.shape() {
        [t, d] => (*t, *d),
        [1, t, d] => (*t, *d),
        shape => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected an embedding of shape [T, dim] or [1, T, dim], got {shape:?}"
            )));
        }
    };
    Ok((arr.as_array().iter().copied().collect(), frames, dim))
}

/// A loaded Pocket TTS model.
#[pyclass(name = "TTS", module = "ptts")]
struct Tts {
    inner: Arc<Mutex<Synth>>,
    /// Voice for calls that name none. `SynthBuilder` picks its own during
    /// `build()`, which is before the voices below are registered.
    default_voice: Option<String>,
}

#[pymethods]
impl Tts {
    /// `TTS(config=None, device=None, quant=None, voice=None, temperature=0.5, seed=..., cfg_coef=None, eos_threshold=None)`
    #[new]
    #[pyo3(signature = (
        config = None,
        device = None,
        quant = None,
        voice = None,
        temperature = 0.5,
        seed = 4242424242424242,
        cfg_coef = None,
        eos_threshold = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        config: Option<String>,
        device: Option<&str>,
        quant: Option<&str>,
        voice: Option<String>,
        temperature: f32,
        seed: u64,
        cfg_coef: Option<f32>,
        eos_threshold: Option<f32>,
    ) -> PyResult<Self> {
        let device = match device {
            None => DeviceKind::Auto,
            Some(name) => DeviceKind::parse(name).py()?,
        };
        let quant = match quant {
            None => Quant::F32,
            Some(name) => Quant::parse(name).py()?,
        };
        // Both checks happen before resolve() downloads anything; `SynthBuilder`
        // would catch them, but only once the checkpoint is on disk.
        quant.check_device(device).py()?;
        let unavailable = match device {
            DeviceKind::Cuda if !cfg!(feature = "cuda") => Some("cuda"),
            DeviceKind::Vulkan if !cfg!(feature = "vulkan") => Some("vulkan"),
            DeviceKind::Metal if !cfg!(feature = "metal") => Some("metal"),
            _ => None,
        };
        if let Some(name) = unavailable {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "device '{name}' is not available in this build; available: {:?}",
                available_devices()
            )));
        }
        // Loading reads hundreds of megabytes and runs no Python.
        py.detach(move || {
            let artifacts = resolve(config.as_deref(), temperature).py()?;
            let mut builder = SynthBuilder::new(artifacts.cfg, &artifacts.model_path)
                .tokenizer_file(&artifacts.tokenizer_path)
                .device(device)
                .quant(quant)
                .temperature(temperature)
                .seed(seed);
            if let Some(cfg_coef) = cfg_coef {
                builder = builder.cfg_coef(cfg_coef);
            }
            if let Some(eos_threshold) = eos_threshold {
                builder = builder.eos_threshold(eos_threshold);
            }
            let mut synth = builder.build().py()?;
            // Registered after the build, not through it: the builder
            // propagates a bad voice file, and a checkpoint shipping one
            // unreadable voice should not stop the model from loading.
            for (name, path) in artifacts.voices.iter() {
                // Skipped rather than propagated, as before: `TTS.voices` shows
                // which ones made it.
                let _ = synth.add_voice_file(name, path);
            }
            if let Some(name) = voice.as_deref()
                && !synth.voices().iter().any(|v| v == name)
            {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown voice '{name}'; available voices are {:?}",
                    synth.voices()
                )));
            }
            Ok(Self { inner: Arc::new(Mutex::new(synth)), default_voice: voice })
        })
    }

    /// Sample rate of the audio this model produces, in Hz.
    #[getter]
    fn sample_rate(&self) -> PyResult<usize> {
        Ok(self.lock()?.sample_rate())
    }

    /// Names of the registered voices, sorted.
    #[getter]
    fn voices(&self) -> PyResult<Vec<String>> {
        Ok(self.lock()?.voices())
    }

    /// Device the model is running on, e.g. `"cpu"`.
    #[getter]
    fn device(&self) -> PyResult<String> {
        Ok(self.lock()?.device_name())
    }

    /// Weight format actually loaded, e.g. `"q8_0"`.
    #[getter]
    fn quant(&self) -> PyResult<&'static str> {
        Ok(self.lock()?.quant().as_str())
    }

    /// Sample rate `clone_voice` expects its PCM in, in Hz.
    #[getter]
    fn voice_prompt_sample_rate(&self) -> PyResult<usize> {
        Ok(self.lock()?.voice_prompt_sample_rate())
    }

    /// True if this checkpoint can clone voices from audio.
    #[getter]
    fn supports_voice_cloning(&self) -> PyResult<bool> {
        Ok(self.lock()?.supports_voice_cloning())
    }

    /// Synthesize `text` and return the waveform as a 1-D float32 array.
    #[pyo3(signature = (text, *, voice=None, temperature=None, seed=None, cfg_coef=None))]
    fn synth<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let opts = self.opts(voice, temperature, seed, cfg_coef);
        let stream = self.start(py, text, &opts)?;
        let pcm = drain(py, stream)?;
        Ok(PyArray1::from_vec(py, pcm))
    }

    /// Synthesize `text` straight to a mono 16-bit WAV file, returning its
    /// duration in seconds.
    #[pyo3(signature = (path, text, *, voice=None, temperature=None, seed=None, cfg_coef=None))]
    #[allow(clippy::too_many_arguments)]
    fn save(
        &self,
        py: Python<'_>,
        path: std::path::PathBuf,
        text: &str,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> PyResult<f64> {
        let opts = self.opts(voice, temperature, seed, cfg_coef);
        let stream = self.start(py, text, &opts)?;
        let sample_rate = stream.sample_rate();
        let pcm = drain(py, stream)?;
        let seconds = pcm.len() as f64 / sample_rate as f64;
        py.detach(|| ptts::wav::write_wav_file(&path, &pcm, sample_rate as u32).py())?;
        Ok(seconds)
    }

    /// Synthesize `text`, yielding float32 chunks as the decoder produces them.
    ///
    /// The returned object is an iterator; dropping it stops the generation.
    #[pyo3(signature = (text, *, voice=None, temperature=None, seed=None, cfg_coef=None))]
    fn stream(
        &self,
        py: Python<'_>,
        text: &str,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> PyResult<AudioStream> {
        let opts = self.opts(voice, temperature, seed, cfg_coef);
        let stream = self.start(py, text, &opts)?;
        Ok(AudioStream { inner: Mutex::new(Some(stream)) })
    }

    /// Register a voice from a precomputed embedding file.
    fn add_voice(&self, py: Python<'_>, name: &str, path: std::path::PathBuf) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        py.detach(move || inner.lock().map_err(|_| poisoned())?.add_voice_file(name, &path).py())
    }

    /// Register a voice from an in-memory conditioning embedding of shape
    /// `[T, dim]` or `[1, T, dim]`.
    ///
    /// `null_embedding` is the encoding of equal-length silence, which CFG
    /// needs on models whose null branch is conditioned on silence.
    #[pyo3(signature = (name, embedding, *, null_embedding=None))]
    fn add_voice_from_embedding(
        &self,
        name: &str,
        embedding: PyReadonlyArrayDyn<'_, f32>,
        null_embedding: Option<PyReadonlyArrayDyn<'_, f32>>,
    ) -> PyResult<()> {
        let (emb, frames, dim) = embedding_dims(&embedding)?;
        let null = match null_embedding.as_ref() {
            None => None,
            Some(arr) => Some(embedding_dims(arr)?.0),
        };
        self.lock()?.add_voice_from_embedding(name, &emb, frames, dim, null.as_deref()).py()
    }

    /// Clone a voice from ~10s of speech, given as float32 PCM at
    /// `voice_prompt_sample_rate`.
    fn clone_voice(
        &self,
        py: Python<'_>,
        name: &str,
        pcm: numpy::PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let pcm = pcm.as_slice()?.to_vec();
        let inner = Arc::clone(&self.inner);
        py.detach(move || inner.lock().map_err(|_| poisoned())?.add_voice_from_pcm(name, &pcm).py())
    }

    fn __repr__(&self) -> PyResult<String> {
        let synth = self.lock()?;
        Ok(format!(
            "TTS(device='{}', quant='{}', sample_rate={}, voices={:?})",
            synth.device_name(),
            synth.quant().as_str(),
            synth.sample_rate(),
            synth.voices()
        ))
    }
}

impl Tts {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Synth>> {
        self.inner.lock().map_err(|_| poisoned())
    }

    /// Per-call settings, with this model's default voice filled in.
    fn opts(
        &self,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> SpeechOptions {
        SpeechOptions {
            voice: voice.or_else(|| self.default_voice.clone()),
            temperature,
            seed,
            cfg_coef,
            max_tokens_per_chunk: None,
        }
    }

    /// Start a generation with the lock taken *inside* `py.detach`. Holding it
    /// across a GIL reacquisition — which `drain` does per chunk — deadlocks
    /// against any other Python thread calling in.
    fn start(&self, py: Python<'_>, text: &str, opts: &SpeechOptions) -> PyResult<SpeechStream> {
        let inner = Arc::clone(&self.inner);
        let mut opts = opts.clone();
        py.detach(move || {
            let synth = inner.lock().map_err(|_| poisoned())?;
            // Resolved here rather than at construction so a voice registered
            // afterwards -- `clone_voice` on a repo that ships none -- is used
            // without having to name it on every call.
            if opts.voice.is_none() {
                opts.voice = synth.voices().first().cloned();
            }
            synth.stream_with(text, &opts).py()
        })
    }
}

/// A panic inside a `&mut self` method would leave the model half-updated, so a
/// poisoned lock is reported rather than papered over.
fn poisoned() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("the model is unusable: a previous call panicked")
}

/// Collect a whole stream, letting Ctrl-C through between chunks.
///
/// The GIL is released while waiting on each chunk and reacquired to check for
/// signals, so a long generation stays interruptible without the caller passing
/// a polling interval.
fn drain(py: Python<'_>, mut stream: SpeechStream) -> PyResult<Vec<f32>> {
    let mut pcm = Vec::new();
    loop {
        match py.detach(|| stream.next()) {
            None => return Ok(pcm),
            Some(chunk) => pcm.extend_from_slice(&chunk.py()?),
        }
        py.check_signals()?;
    }
}

/// An in-progress generation. Iterate it for float32 chunks.
#[pyclass(module = "ptts")]
struct AudioStream {
    // `SpeechStream` owns an mpsc receiver, which is `Send` but not `Sync`,
    // while `#[pyclass]` wants both; the mutex bridges that. `None` after
    // `close`, so a closed stream iterates as empty rather than raising.
    inner: Mutex<Option<SpeechStream>>,
}

#[pymethods]
impl AudioStream {
    /// Sample rate of the chunks, in Hz.
    #[getter]
    fn sample_rate(&self) -> PyResult<usize> {
        match self.inner.lock().map_err(|_| poisoned())?.as_ref() {
            Some(stream) => Ok(stream.sample_rate()),
            None => Err(pyo3::exceptions::PyValueError::new_err("this stream is closed")),
        }
    }

    /// Stop generating and release the worker threads.
    fn close(&self) -> PyResult<()> {
        *self.inner.lock().map_err(|_| poisoned())? = None;
        Ok(())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let inner = &slf.inner;
        let next = py.detach(|| -> PyResult<Option<Result<Vec<f32>, ptts::Error>>> {
            let mut guard = inner.lock().map_err(|_| poisoned())?;
            Ok(guard.as_mut().and_then(|stream| stream.next()))
        })?;
        match next {
            None => Ok(None),
            Some(chunk) => Ok(Some(PyArray1::from_vec(py, chunk.py()?))),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Number of CPU threads used for tensor ops.
#[pyfunction]
fn get_num_threads() -> usize {
    xn::utils::get_num_threads()
}

/// Set the number of CPU threads used for tensor ops. Call before loading a
/// model: it sizes a global thread pool that is built once.
#[pyfunction]
fn set_num_threads(num_threads: usize) {
    xn::utils::set_num_threads(num_threads);
}

/// The device names this build accepts, most capable first. `"auto"` picks the
/// first of these.
#[pyfunction]
fn available_devices() -> Vec<&'static str> {
    let mut devices = vec![];
    if cfg!(feature = "cuda") {
        devices.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        devices.push("vulkan");
    }
    if cfg!(feature = "metal") {
        devices.push("metal");
    }
    devices.push("cpu");
    devices
}

/// The weight formats `quant=` accepts. All are CPU-only.
#[pyfunction]
fn available_quants() -> Vec<&'static str> {
    vec!["f32", "q8_0", "q8_1", "q8k", "q6k", "q5_0", "q5_1", "q5k", "q4_0", "q4_1", "q4k"]
}

/// The runtime's build configuration, for bug reports.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    info.set_item("version", env!("CARGO_PKG_VERSION"))?;
    info.set_item("devices", available_devices())?;
    info.set_item("avx", xn::with_avx())?;
    info.set_item("neon", xn::with_neon())?;
    info.set_item("f16c", xn::with_f16c())?;
    info.set_item("threads", xn::utils::get_num_threads())?;
    Ok(info)
}

#[pymodule(name = "ptts")]
fn ptts_(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Tts>()?;
    m.add_class::<AudioStream>()?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(available_devices, m)?)?;
    m.add_function(wrap_pyfunction!(available_quants, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    Ok(())
}
