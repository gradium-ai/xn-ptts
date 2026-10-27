//! C interface to Phonon. The PhononTTS Swift package wraps it, and Android apps call it
//! through JNA (`android/README.md`).
//!
//! The same calls run on one of two engines. On Apple platforms, `coreml` drives the Core ML
//! driver, and `dir` is a bundle `export_coreml` wrote. Everywhere else, `cpu` runs
//! `ptts::synth::Synth` on the CPU, and `dir` is a checkpoint folder.
//!
//! `ptts_new` loads the model; it is the expensive call. `ptts_speak` may then be called
//! repeatedly, streaming the audio to a callback as it is decoded.
//!
//! Only the platform glue lives here. `include/ptts.h` declares this interface by hand, so any
//! change to a signature or to `PttsResult` has to be made there too.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

#[cfg(target_vendor = "apple")]
mod coreml;
#[cfg(target_vendor = "apple")]
use coreml::Engine;
#[cfg(not(target_vendor = "apple"))]
mod cpu;
#[cfg(not(target_vendor = "apple"))]
use cpu::Engine;

/// Where the flow LM runs. Only `PTTS_UNIT_CPU` exists off Apple.
pub const PTTS_UNIT_CPU: u32 = 0;
pub const PTTS_UNIT_ANE: u32 = 2;

pub struct PttsHandle {
    engine: Engine,
    /// The voice names as NUL-separated bytes, so `ptts_voices` can hand out a borrowed pointer.
    voice_blob: Vec<u8>,
    /// The same for `ptts_languages`.
    language_blob: Vec<u8>,
    last_error: Option<CString>,
}

/// One utterance's timings. The audio itself went to the callback.
#[repr(C)]
pub struct PttsResult {
    pub frames: u32,
    /// 24 kHz samples delivered.
    pub samples: usize,
    /// From the call to the first audio being delivered.
    pub ttfa_ms: f64,
    pub total_ms: f64,
}

/// Called with PCM as it is decoded, on the thread that called `ptts_speak`: one frame per call
/// on Core ML, one or more on the CPU. Return false to stop speaking: generation ends there and
/// `ptts_speak` returns normally with what was produced.
pub type PttsFrameFn = extern "C" fn(*const f32, usize, *mut c_void) -> bool;

thread_local! {
    /// The last `ptts_new` failure on this thread. Thread-local, so a failure elsewhere cannot
    /// free the string a caller is still reading.
    // Clippy asks for a `const` initializer on Android, where it already is one.
    #[cfg_attr(target_os = "android", allow(clippy::missing_const_for_thread_local))]
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn c_string(e: &str) -> CString {
    CString::new(e.replace('\0', " ")).expect("NULs were replaced")
}

/// Run `f`, turning a panic into an error: unwinding across the C interface aborts the app.
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let msg = p
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| p.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown".into());
        Err(format!("internal error: {msg}"))
    })
}

fn nul_separated(names: &[String]) -> Vec<u8> {
    let mut blob = Vec::new();
    for n in names {
        blob.extend_from_slice(n.as_bytes());
        blob.push(0);
    }
    blob.push(0);
    blob
}

fn open(dir: &Path, unit: u32, lang: &str) -> Result<PttsHandle, String> {
    let engine = Engine::open(dir, unit, lang)?;
    let voice_blob = nul_separated(engine.voices());
    let language_blob = nul_separated(engine.languages());
    Ok(PttsHandle { engine, voice_blob, language_blob, last_error: None })
}

/// Load the model in `dir` for `unit` (`PTTS_UNIT_ANE` or `PTTS_UNIT_CPU`), normalizing text as
/// `lang` (`en`, `fr`, `de`, `es`, `pt`, or `none`). On Apple, `dir` is an exported bundle,
/// compiled for this device first if it has not been (once per install); elsewhere it is a
/// checkpoint folder, and only `PTTS_UNIT_CPU` exists. Null on failure; then
/// `ptts_last_error(NULL)`, on the same thread, says why. A bundle without voices fails to load; a
/// checkpoint folder without voices loads, lists none, and speaks in no particular voice.
///
/// # Safety
/// `dir` and `lang` must be NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_new(
    dir: *const c_char,
    unit: u32,
    lang: *const c_char,
) -> *mut PttsHandle {
    let dir = unsafe { CStr::from_ptr(dir) }.to_string_lossy().to_string();
    let lang = unsafe { CStr::from_ptr(lang) }.to_string_lossy().to_string();
    match guarded(|| open(Path::new(&dir), unit, &lang)) {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            LAST_ERROR.with(|l| *l.borrow_mut() = Some(c_string(&e)));
            std::ptr::null_mut()
        }
    }
}

/// Speak `text`, calling `cb(pcm, n, user)` with each frame's audio as soon as it is decoded.
/// `out` then describes the whole utterance. Returns false on failure; see `ptts_last_error`.
///
/// # Safety
/// `h` must come from `ptts_new`; `text` must be NUL-terminated UTF-8; `cb` must be valid for
/// the duration of the call; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_speak(
    h: *mut PttsHandle,
    text: *const c_char,
    cb: PttsFrameFn,
    user: *mut c_void,
    out: *mut PttsResult,
) -> bool {
    let h = unsafe { &mut *h };
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy().to_string();
    let r = guarded(|| h.engine.speak(&text, &mut |pcm: &[f32]| cb(pcm.as_ptr(), pcm.len(), user)));
    match r {
        Ok(r) => {
            unsafe { *out = r };
            true
        }
        Err(e) => {
            h.last_error = Some(c_string(&e));
            false
        }
    }
}

/// Voice names, sorted, as one NUL-separated block ending in a second NUL. Borrowed.
///
/// # Safety
/// `h` must come from `ptts_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_voices(h: *const PttsHandle) -> *const c_char {
    unsafe { (*h).voice_blob.as_ptr() as *const c_char }
}

/// The languages the model was trained for, as its config lists them, in the same block format
/// as `ptts_voices`. Borrowed.
///
/// # Safety
/// `h` must come from `ptts_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_languages(h: *const PttsHandle) -> *const c_char {
    unsafe { (*h).language_blob.as_ptr() as *const c_char }
}

/// Switch to one of `ptts_voices`. Returns false on failure; see `ptts_last_error`.
///
/// # Safety
/// `h` must come from `ptts_new`; `name` must be NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_set_voice(h: *mut PttsHandle, name: *const c_char) -> bool {
    let h = unsafe { &mut *h };
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy().to_string();
    let r = guarded(|| {
        if !h.engine.voices().contains(&name) {
            return Err(format!("no voice named {name:?}"));
        }
        h.engine.set_voice(&name)
    });
    match r {
        Ok(()) => true,
        Err(e) => {
            h.last_error = Some(c_string(&e));
            false
        }
    }
}

/// Normalize text as `lang` (`en`, `fr`, `de`, `es`, `pt`, or `none`) from the next `ptts_speak`
/// on, as `ptts_new` does: switching languages needs no reload. Returns false on failure; see
/// `ptts_last_error`.
///
/// # Safety
/// `h` must come from `ptts_new`; `lang` must be NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_set_language(h: *mut PttsHandle, lang: *const c_char) -> bool {
    let h = unsafe { &mut *h };
    let lang = unsafe { CStr::from_ptr(lang) }.to_string_lossy().to_string();
    match guarded(|| h.engine.set_language(&lang)) {
        Ok(()) => true,
        Err(e) => {
            h.last_error = Some(c_string(&e));
            false
        }
    }
}

/// The last error on `h`, or, when `h` is null, the last `ptts_new` failure on this thread.
/// Borrowed until the next failing call.
///
/// # Safety
/// `h` must be null or come from `ptts_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_last_error(h: *const PttsHandle) -> *const c_char {
    if h.is_null() {
        // The string lives in this thread's slot until the next failure on this thread.
        return LAST_ERROR.with(|l| l.borrow().as_ref().map_or(std::ptr::null(), |c| c.as_ptr()));
    }
    unsafe { (*h).last_error.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()) }
}

/// # Safety
/// `h` must come from `ptts_new` and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_free(h: *mut PttsHandle) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}
