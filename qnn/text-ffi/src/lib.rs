//! C interface over `ptts`'s text front end: normalization, sentence chunking, prompt
//! preparation and tokenization, so a C or C++ runner prompts the model with exactly the token
//! ids `ptts::synth::Synth::say` / `stream` would.
//!
//! Nothing is reimplemented here. [`ptts_text_split`] is [`ptts::plan::chunks`], the function
//! `Synth` calls on every request (through its private `plan_chunks`), with the same defaults:
//! [`ptts::tts_model::MAX_TOKENS_PER_CHUNK`] tokens per chunk and the codec's 12.5 Hz frame rate.
//! `include/ptts_text.h` declares this interface by hand, so a change to a signature or to a
//! `#[repr(C)]` struct has to be made there too.
//!
//! A handle is immutable once built, so one may be shared between threads. Errors come back
//! through an out parameter as strings the library allocates; free them with
//! [`ptts_text_string_free`].

use ptts::plan::Chunk;
use ptts::preprocess::{Normalize, Rules};
use ptts::tok::Tok;
use ptts::tts_model::{MAX_TOKENS_PER_CHUNK, prepare_text_prompt};
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

/// The Mimi codec's frame rate in the published Phonon configs (`mimi.frame_rate`).
pub const DEFAULT_FRAME_RATE: f64 = 12.5;

/// A tokenizer plus a normalization policy.
pub struct PttsText {
    tokenizer: Tok,
    normalize: Normalize,
}

/// One chunk, as [`ptts::plan::Chunk`] holds it.
#[repr(C)]
pub struct PttsTextChunk {
    /// The chunk's normalized text, before `prepare_text_prompt` (`Chunk::text`).
    pub text: *const c_char,
    /// `prepare_text_prompt(text).0`: exactly the string that was tokenized.
    pub prepared: *const c_char,
    /// `Chunk::tokens`: the ids to prompt the model with, no special tokens added.
    pub tokens: *const u32,
    pub n_tokens: usize,
    /// `Chunk::frames_after_eos`: frames still generated after the model signals EOS
    /// (3 for a chunk of four words or fewer, 1 otherwise).
    pub frames_after_eos: u32,
    /// `Chunk::frame_budget`: the most frames this chunk may generate.
    pub frame_budget: u32,
    /// `Chunk::seq_budget()`: tokens + frame budget + 512 slots of headroom for the voice prompt.
    pub seq_budget: u32,
}

/// The chunks of one input, in order.
#[repr(C)]
pub struct PttsTextChunks {
    pub chunks: *const PttsTextChunk,
    pub n_chunks: usize,
}

/// What a `PttsTextChunks` pointer really points at: the public part first, then the storage
/// its pointers borrow from.
#[repr(C)]
struct Owned {
    public: PttsTextChunks,
    _strings: Vec<CString>,
    _tokens: Vec<Vec<u32>>,
    _chunks: Vec<PttsTextChunk>,
}

fn c_string(s: &str) -> CString {
    CString::new(s.replace('\0', " ")).expect("NULs were replaced")
}

/// Run `f`, turning a panic into an error: unwinding across the C interface is undefined.
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

/// Store `e` in `*error` when the caller asked for it.
unsafe fn set_error(error: *mut *mut c_char, e: &str) {
    if !error.is_null() {
        unsafe { *error = c_string(e).into_raw() };
    }
}

unsafe fn str_arg<'a>(p: *const c_char, what: &str) -> Result<&'a str, String> {
    if p.is_null() {
        return Err(format!("{what} is null"));
    }
    unsafe { CStr::from_ptr(p) }.to_str().map_err(|e| format!("{what} is not valid UTF-8: {e}"))
}

impl PttsText {
    pub fn open(tokenizer_json: &Path, lang: &str, rewrites: Option<&str>) -> Result<Self, String> {
        let mut normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        if let Some(rules) = rewrites {
            normalize = normalize.with_rules(rules.parse::<Rules>().map_err(|e| e.to_string())?);
        }
        let tokenizer = Tok::open(tokenizer_json).map_err(|e| e.to_string())?;
        Ok(Self { tokenizer, normalize })
    }

    /// Exactly what `Synth` plans for `text`.
    pub fn split(
        &self,
        text: &str,
        max_tokens: usize,
        frame_rate: f64,
    ) -> Result<Vec<Chunk>, String> {
        ptts::plan::chunks(&self.tokenizer, text, self.normalize, max_tokens, frame_rate)
            .map_err(|e| e.to_string())
    }

    pub fn normalize(&self, text: &str) -> String {
        self.normalize.apply(text).into_owned()
    }
}

fn to_c(chunks: Vec<Chunk>) -> Result<*mut PttsTextChunks, String> {
    let n = chunks.len();
    let mut strings = Vec::with_capacity(2 * n);
    let mut tokens = Vec::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    let narrow =
        |v: usize, what: &str| u32::try_from(v).map_err(|_| format!("{what} {v} overflows"));
    for chunk in chunks {
        let prepared = prepare_text_prompt(&chunk.text).0;
        let (text, prepared) = (c_string(&chunk.text), c_string(&prepared));
        // CString and Vec keep their heap buffers where they are when moved into the vectors.
        out.push(PttsTextChunk {
            text: text.as_ptr(),
            prepared: prepared.as_ptr(),
            tokens: chunk.tokens.as_ptr(),
            n_tokens: chunk.tokens.len(),
            frames_after_eos: narrow(chunk.frames_after_eos, "frames_after_eos")?,
            frame_budget: narrow(chunk.frame_budget, "frame_budget")?,
            seq_budget: narrow(chunk.seq_budget(), "seq_budget")?,
        });
        strings.push(text);
        strings.push(prepared);
        tokens.push(chunk.tokens);
    }
    let public = PttsTextChunks { chunks: out.as_ptr(), n_chunks: out.len() };
    let owned = Box::new(Owned { public, _strings: strings, _tokens: tokens, _chunks: out });
    Ok(Box::into_raw(owned) as *mut PttsTextChunks)
}

/// Open `tokenizer_json_path` and normalize as `lang` (`en`, `fr`, `de`, `es`, `pt`, or `none`)
/// with the default rewrite rules. Null on failure, with the reason in `*error` if `error` is
/// not null.
///
/// # Safety
/// The strings must be NUL-terminated; `error` must be null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_text_new(
    tokenizer_json_path: *const c_char,
    lang: *const c_char,
    error: *mut *mut c_char,
) -> *mut PttsText {
    unsafe { ptts_text_new_with_rewrites(tokenizer_json_path, lang, std::ptr::null(), error) }
}

/// [`ptts_text_new`] with the rewrite rules given as the frontends' `--rewrites` flag takes them:
/// `default`, `all`, `none`, or a comma-separated list of rule names. Null `rewrites` means
/// `default`.
///
/// # Safety
/// As [`ptts_text_new`]; `rewrites` must be null or NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_text_new_with_rewrites(
    tokenizer_json_path: *const c_char,
    lang: *const c_char,
    rewrites: *const c_char,
    error: *mut *mut c_char,
) -> *mut PttsText {
    let r = guarded(|| {
        let path = unsafe { str_arg(tokenizer_json_path, "tokenizer_json_path") }?;
        let lang = unsafe { str_arg(lang, "lang") }?;
        let rewrites =
            if rewrites.is_null() { None } else { Some(unsafe { str_arg(rewrites, "rewrites") }?) };
        PttsText::open(Path::new(path), lang, rewrites)
    });
    match r {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            unsafe { set_error(error, &e) };
            std::ptr::null_mut()
        }
    }
}

/// Split `text` into the chunks `Synth::say` / `Synth::stream` generate, in order. `max_tokens`
/// of 0 means `MAX_TOKENS_PER_CHUNK` (50, `Synth`'s default) and `frame_rate` of 0 or less means
/// 12.5. Returns null on failure (text empty after normalization, invalid UTF-8, ...), with the
/// reason in `*error` if `error` is not null. Free the result with [`ptts_text_chunks_free`].
///
/// # Safety
/// `h` must come from `ptts_text_new`; `text` must be NUL-terminated; `error` must be null or
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_text_split(
    h: *const PttsText,
    text: *const c_char,
    max_tokens: usize,
    frame_rate: f64,
    error: *mut *mut c_char,
) -> *mut PttsTextChunks {
    let r = guarded(|| {
        let h = unsafe { h.as_ref() }.ok_or("handle is null")?;
        let text = unsafe { str_arg(text, "text") }?;
        let max_tokens = if max_tokens == 0 { MAX_TOKENS_PER_CHUNK } else { max_tokens };
        let frame_rate = if frame_rate > 0.0 { frame_rate } else { DEFAULT_FRAME_RATE };
        to_c(h.split(text, max_tokens, frame_rate)?)
    });
    r.unwrap_or_else(|e| {
        unsafe { set_error(error, &e) };
        std::ptr::null_mut()
    })
}

/// The whole of `text` after normalization, the first thing `ptts_text_split` does. For
/// debugging; free it with [`ptts_text_string_free`]. Null on failure.
///
/// # Safety
/// As [`ptts_text_split`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_text_normalize(
    h: *const PttsText,
    text: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_char {
    let r = guarded(|| {
        let h = unsafe { h.as_ref() }.ok_or("handle is null")?;
        let text = unsafe { str_arg(text, "text") }?;
        Ok(c_string(&h.normalize(text)).into_raw())
    });
    r.unwrap_or_else(|e| {
        unsafe { set_error(error, &e) };
        std::ptr::null_mut()
    })
}

/// # Safety
/// `chunks` must be null or come from `ptts_text_split`, and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_text_chunks_free(chunks: *mut PttsTextChunks) {
    if !chunks.is_null() {
        drop(unsafe { Box::from_raw(chunks as *mut Owned) });
    }
}

/// Free an error string or a `ptts_text_normalize` result.
///
/// # Safety
/// `s` must be null or a string this library returned, and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_text_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// # Safety
/// `h` must be null or come from `ptts_text_new`, and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_text_free(h: *mut PttsText) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}
