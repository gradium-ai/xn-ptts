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
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};

/// Hub repo loaded when the caller names none. Which voices and weight files it holds is
/// `ptts::loader::ModelSource`'s to discover, not this file's to list.
const DEFAULT_REPO_ID: &str = "kyutai/pocket-tts";

/// Map a `ptts` error onto the Python exception its class calls for.
///
/// `ptts::ErrorKind` exists so a binding does not have to match twenty-odd variants, or flatten
/// them all onto one exception -- which is what this did before, and which made a gated
/// checkpoint and a misspelt voice both `ValueError`.
fn to_py_err(e: ptts::Error) -> PyErr {
    use ptts::ErrorKind::*;
    use pyo3::exceptions as exc;

    let msg = e.to_string();
    match e.kind() {
        InvalidArgument => exc::PyValueError::new_err(msg),
        // `LookupError` is the base of `KeyError`, and covers both "no such voice" and
        // "no such checkpoint file" without claiming either is a dict lookup.
        NotFound => exc::PyLookupError::new_err(msg),
        Unsupported => exc::PyNotImplementedError::new_err(msg),
        PermissionDenied => exc::PyPermissionError::new_err(msg),
        Network => exc::PyConnectionError::new_err(msg),
        // `Busy` is retryable and the others are not, but Python has no better shared base
        // than `RuntimeError` for either.
        InvalidData | Busy | Internal => exc::PyRuntimeError::new_err(msg),
        // `ErrorKind` is `#[non_exhaustive]`: a kind added upstream should not silently pick
        // one of the above.
        _ => exc::PyRuntimeError::new_err(msg),
    }
}

trait IntoPy<R> {
    fn py(self) -> PyResult<R>;
}

impl<R, E: Into<ptts::Error>> IntoPy<R> for Result<R, E> {
    fn py(self) -> PyResult<R> {
        self.map_err(|e| to_py_err(e.into()))
    }
}

/// Resolve `config` -- a local `config.json` or model directory, a Hub repo id, or nothing for
/// the published checkpoint -- into a located checkpoint.
///
/// `ptts::loader::ModelSource` does the searching; this only decides which of its two kinds the
/// string is. Routing through it is also what makes the typed errors reach Python: a gated repo
/// arrives as `PermissionError` rather than as a message inside a `RuntimeError`.
fn resolve(config: Option<&str>) -> ptts::Result<ptts::loader::Checkpoint> {
    use ptts::loader::ModelSource;

    let source = match config {
        None => ModelSource::hub(DEFAULT_REPO_ID),
        // A config file names its directory; a directory names itself.
        Some(path) if path.ends_with(".json") => {
            let config_path = std::path::Path::new(path);
            let parent = config_path.parent().filter(|p| !p.as_os_str().is_empty());
            ModelSource::dir(parent.unwrap_or(std::path::Path::new(".")))
        }
        Some(path) if std::path::Path::new(path).is_dir() => ModelSource::dir(path),
        Some(repo_id) => ModelSource::hub(repo_id),
    };
    source.resolve()
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
            let checkpoint = resolve(config.as_deref()).py()?;
            let mut cfg = checkpoint.config.clone();
            cfg.temp = temperature;
            let mut builder = SynthBuilder::new(cfg, &checkpoint.weights)
                .device(device)
                .quant(quant)
                .temperature(temperature)
                .seed(seed);
            if let Some(tokenizer) = checkpoint.tokenizer.as_ref() {
                builder = builder.tokenizer_file(tokenizer);
            }
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
            for (name, path) in checkpoint.voices.iter() {
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
