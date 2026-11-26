//! The part of LiteRT's C API the engine uses, loaded at run time from `libLiteRt`.
//!
//! Nothing links against LiteRT: `ptts_new` opens the library with `dlopen`, so the crate
//! builds without a LiteRT SDK, and an app ships the `libLiteRt.so` of the LiteRT release its
//! models were compiled with. The declarations follow LiteRT 2.3's headers under `litert/c/`.
//! The structs passed by value are the ones those headers mark ABI stable.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::rc::Rc;

type Status = c_int;
type Handle = *mut c_void;

const STATUS_OK: Status = 0;

pub const ACCEL_CPU: c_int = 1;
pub const ACCEL_GPU: c_int = 1 << 1;
pub const ACCEL_NPU: c_int = 1 << 2;

const ELEMENT_FLOAT32: c_int = 1;
const ELEMENT_INT32: c_int = 2;
const ELEMENT_FLOAT16: c_int = 10;

const LOCK_READ: c_int = 0;
const LOCK_WRITE: c_int = 1;
const LOCK_READ_WRITE: c_int = 2;

const ENV_COMPILER_PLUGIN_DIR: c_int = 0;
const ENV_DISPATCH_DIR: c_int = 1;
const ENV_RUNTIME_DIR: c_int = 22;
const ENV_MIN_LOG_SEVERITY: c_int = 25;
const ANY_INT: c_int = 2;
const ANY_STRING: c_int = 8;
/// `kLiteRtLogSeverityWarning`: LiteRT's info lines are many and of no use to an app.
const LOG_WARNING: i64 = 2;

/// `LiteRtLayout`: `rank` is the low 7 bits of `bits`, `has_strides` the next one.
#[repr(C)]
#[derive(Clone, Copy)]
struct Layout {
    bits: u32,
    dimensions: [i32; 8],
    strides: [u32; 8],
}

/// `LiteRtRankedTensorType`.
#[repr(C)]
#[derive(Clone, Copy)]
struct RankedTensorType {
    element_type: c_int,
    layout: Layout,
}

/// `LiteRtAny`, for a string or an integer.
#[repr(C)]
struct Any {
    kind: c_int,
    value: AnyValue,
}

#[repr(C)]
union AnyValue {
    int: i64,
    string: *const c_char,
}

/// `LiteRtEnvOption`.
#[repr(C)]
struct EnvOption {
    tag: c_int,
    value: Any,
}

const _: () = assert!(std::mem::size_of::<RankedTensorType>() == 72);
const _: () = assert!(std::mem::size_of::<EnvOption>() == 24);

macro_rules! api {
    ($($name:ident: fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        #[allow(non_snake_case)]
        struct Fns {
            $($name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }

        impl Fns {
            /// Every symbol from `lib`, or the first one missing.
            unsafe fn load(lib: Handle) -> Result<Self, String> {
                Ok(Self {
                    $($name: unsafe {
                        let sym = libc::dlsym(lib, concat!(stringify!($name), "\0").as_ptr().cast());
                        if sym.is_null() {
                            return Err(format!(
                                "libLiteRt has no {}: it is not the LiteRT 2.3 C API",
                                stringify!($name)
                            ));
                        }
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) $(-> $ret)?>(sym)
                    },)*
                })
            }
        }
    };
}

api! {
    LiteRtGetStatusString: fn(Status) -> *const c_char;
    LiteRtCreateEnvironment: fn(c_int, *const EnvOption, *mut Handle) -> Status;
    LiteRtDestroyEnvironment: fn(Handle);
    LiteRtCreateModelFromFile: fn(Handle, *const c_char, *mut Handle) -> Status;
    LiteRtDestroyModel: fn(Handle);
    LiteRtGetNumModelSignatures: fn(Handle, *mut usize) -> Status;
    LiteRtGetModelSignature: fn(Handle, usize, *mut Handle) -> Status;
    LiteRtGetSignatureKey: fn(Handle, *mut *const c_char) -> Status;
    LiteRtGetNumSignatureInputs: fn(Handle, *mut usize) -> Status;
    LiteRtGetSignatureInputName: fn(Handle, usize, *mut *const c_char) -> Status;
    LiteRtGetSignatureInputTensorByIndex: fn(Handle, usize, *mut Handle) -> Status;
    LiteRtGetNumSignatureOutputs: fn(Handle, *mut usize) -> Status;
    LiteRtGetSignatureOutputName: fn(Handle, usize, *mut *const c_char) -> Status;
    LiteRtGetSignatureOutputTensorByIndex: fn(Handle, usize, *mut Handle) -> Status;
    LiteRtGetRankedTensorType: fn(Handle, *mut RankedTensorType) -> Status;
    LiteRtCreateOptions: fn(*mut Handle) -> Status;
    LiteRtDestroyOptions: fn(Handle);
    LiteRtSetOptionsHardwareAccelerators: fn(Handle, c_int) -> Status;
    LiteRtCreateOpaqueOptions: fn(*const c_char, *mut c_void, Option<unsafe extern "C" fn(*mut c_void)>, *mut Handle) -> Status;
    LiteRtAddOpaqueOptions: fn(Handle, Handle) -> Status;
    LiteRtCreateCompiledModel: fn(Handle, Handle, Handle, *mut Handle) -> Status;
    LiteRtDestroyCompiledModel: fn(Handle);
    LiteRtGetCompiledModelInputBufferRequirements: fn(Handle, usize, usize, *mut Handle) -> Status;
    LiteRtGetCompiledModelOutputBufferRequirements: fn(Handle, usize, usize, *mut Handle) -> Status;
    LiteRtCreateManagedTensorBufferFromRequirements: fn(Handle, *const RankedTensorType, Handle, *mut Handle) -> Status;
    LiteRtDestroyTensorBuffer: fn(Handle);
    LiteRtLockTensorBuffer: fn(Handle, *mut *mut c_void, c_int) -> Status;
    LiteRtUnlockTensorBuffer: fn(Handle) -> Status;
    LiteRtRunCompiledModel: fn(Handle, usize, usize, *mut Handle, usize, *mut Handle) -> Status;
}

/// The loaded library. Never unloaded: LiteRT keeps threads and accelerators that outlive any
/// one handle, and the library is loaded once per process.
struct Lib {
    fns: Fns,
    /// Where `libLiteRt` was found, where its accelerator and dispatch libraries are looked for.
    dir: Option<PathBuf>,
}

unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

fn dl_error() -> String {
    let e = unsafe { libc::dlerror() };
    if e.is_null() {
        "unknown error".into()
    } else {
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

#[cfg(target_vendor = "apple")]
const LIB_NAME: &str = "libLiteRt.dylib";
#[cfg(not(target_vendor = "apple"))]
const LIB_NAME: &str = "libLiteRt.so";

/// `libLiteRt` from `PTTS_LITERT_LIB_DIR` when it is set, else from wherever the dynamic linker
/// finds it: in an Android app, its native library folder.
fn open_lib() -> Result<Lib, String> {
    let dir = std::env::var_os("PTTS_LITERT_LIB_DIR").map(PathBuf::from);
    let path = dir.as_ref().map_or(PathBuf::from(LIB_NAME), |d| d.join(LIB_NAME));
    let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
    let lib = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if lib.is_null() {
        return Err(format!(
            "cannot load {}: {}. Ship LiteRT's {LIB_NAME} with the app, or set PTTS_LITERT_LIB_DIR",
            path.display(),
            dl_error()
        ));
    }
    let fns = unsafe { Fns::load(lib)? };
    let dir = dir.or_else(|| loaded_from(fns.LiteRtGetStatusString as *const c_void));
    Ok(Lib { fns, dir })
}

/// The folder of the library holding `sym`.
fn loaded_from(sym: *const c_void) -> Option<PathBuf> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    if unsafe { libc::dladdr(sym, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    let file = unsafe { CStr::from_ptr(info.dli_fname) }.to_string_lossy().into_owned();
    Path::new(&file).parent().filter(|d| d.is_dir()).map(Path::to_path_buf)
}

fn lib() -> Result<&'static Lib, String> {
    static LIB: std::sync::OnceLock<Result<Lib, String>> = std::sync::OnceLock::new();
    LIB.get_or_init(open_lib).as_ref().map_err(Clone::clone)
}

fn check(fns: &Fns, status: Status, what: &str) -> Result<(), String> {
    if status == STATUS_OK {
        return Ok(());
    }
    let s = unsafe { (fns.LiteRtGetStatusString)(status) };
    let s = if s.is_null() {
        format!("status {status}")
    } else {
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    };
    Err(format!("LiteRT: {what}: {s}"))
}

/// A LiteRT environment, with the folder the accelerator and NPU dispatch libraries are in.
pub struct Env {
    lib: &'static Lib,
    raw: Handle,
}

impl Env {
    pub fn new() -> Result<Rc<Self>, String> {
        let lib = lib()?;
        let dir = lib.dir.as_ref().map(|d| CString::new(d.to_string_lossy().as_bytes()));
        let dir = dir.transpose().map_err(|e| e.to_string())?;
        let mut opts: Vec<EnvOption> = dir
            .iter()
            .flat_map(|d| {
                [ENV_DISPATCH_DIR, ENV_COMPILER_PLUGIN_DIR, ENV_RUNTIME_DIR].map(|tag| EnvOption {
                    tag,
                    value: Any { kind: ANY_STRING, value: AnyValue { string: d.as_ptr() } },
                })
            })
            .collect();
        let severity = Any { kind: ANY_INT, value: AnyValue { int: LOG_WARNING } };
        opts.push(EnvOption { tag: ENV_MIN_LOG_SEVERITY, value: severity });
        let mut raw = std::ptr::null_mut();
        let f = &lib.fns;
        check(
            f,
            unsafe { (f.LiteRtCreateEnvironment)(opts.len() as c_int, opts.as_ptr(), &mut raw) },
            "creating the environment",
        )?;
        Ok(Rc::new(Self { lib, raw }))
    }

    fn fns(&self) -> &'static Fns {
        &self.lib.fns
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        unsafe { (self.fns().LiteRtDestroyEnvironment)(self.raw) }
    }
}

/// A tensor's element type and shape.
#[derive(Clone, Copy)]
pub struct TensorType(RankedTensorType);

impl TensorType {
    pub fn shape(&self) -> Vec<usize> {
        let rank = (self.0.layout.bits & 0x7f) as usize;
        self.0.layout.dimensions[..rank].iter().map(|&d| d.max(0) as usize).collect()
    }

    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    fn is_f32(&self) -> bool {
        self.0.element_type == ELEMENT_FLOAT32
    }

    fn is_i32(&self) -> bool {
        self.0.element_type == ELEMENT_INT32
    }

    /// Bytes per element of the types a bundle uses.
    pub fn elem_size(&self) -> usize {
        if self.0.element_type == ELEMENT_FLOAT16 { 2 } else { 4 }
    }

    pub fn bytes(&self) -> usize {
        self.len() * self.elem_size()
    }
}

/// A tensor buffer allocated by LiteRT in whatever memory the compiled model wants for it.
pub struct Buffer {
    env: Rc<Env>,
    raw: Handle,
    ty: TensorType,
    name: String,
}

impl Buffer {
    pub fn ty(&self) -> &TensorType {
        &self.ty
    }

    fn lock(&self, mode: c_int) -> Result<*mut c_void, String> {
        let f = self.env.fns();
        let mut p = std::ptr::null_mut();
        check(f, unsafe { (f.LiteRtLockTensorBuffer)(self.raw, &mut p, mode) }, &self.name)?;
        Ok(p)
    }

    fn unlock(&self) -> Result<(), String> {
        let f = self.env.fns();
        check(f, unsafe { (f.LiteRtUnlockTensorBuffer)(self.raw) }, &self.name)
    }

    /// The buffer's bytes, for reading and writing, whatever its element type.
    pub fn with_bytes<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> Result<R, String> {
        let p = self.lock(LOCK_READ_WRITE)?;
        let r = f(unsafe { std::slice::from_raw_parts_mut(p as *mut u8, self.ty.bytes()) });
        self.unlock()?;
        Ok(r)
    }

    /// Overwrite the buffer with `data`, which must be its size in bytes.
    pub fn write_bytes(&self, data: &[u8]) -> Result<(), String> {
        if data.len() != self.ty.bytes() {
            return Err(format!(
                "{} takes {} bytes, not {}",
                self.name,
                self.ty.bytes(),
                data.len()
            ));
        }
        let p = self.lock(LOCK_WRITE)?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut u8, data.len()) };
        self.unlock()
    }

    /// Copy the buffer's bytes into `out`, resized to fit.
    pub fn read_bytes(&self, out: &mut Vec<u8>) -> Result<(), String> {
        let p = self.lock(LOCK_READ)?;
        out.clear();
        out.extend_from_slice(unsafe {
            std::slice::from_raw_parts(p as *const u8, self.ty.bytes())
        });
        self.unlock()
    }

    /// Overwrite the buffer with `data`, which must be its size.
    pub fn write_f32(&self, data: &[f32]) -> Result<(), String> {
        if !self.ty.is_f32() || data.len() != self.ty.len() {
            return Err(format!("{} takes {} f32s, not {}", self.name, self.ty.len(), data.len()));
        }
        let p = self.lock(LOCK_WRITE)?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut f32, data.len()) };
        self.unlock()
    }

    pub fn write_i32(&self, data: &[i32]) -> Result<(), String> {
        if !self.ty.is_i32() || data.len() != self.ty.len() {
            return Err(format!("{} takes {} i32s, not {}", self.name, self.ty.len(), data.len()));
        }
        let p = self.lock(LOCK_WRITE)?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut i32, data.len()) };
        self.unlock()
    }

    /// Copy the buffer into `out`, resized to fit.
    pub fn read_f32(&self, out: &mut Vec<f32>) -> Result<(), String> {
        if !self.ty.is_f32() {
            return Err(format!("{} is not f32", self.name));
        }
        let p = self.lock(LOCK_READ)?;
        out.clear();
        out.extend_from_slice(unsafe {
            std::slice::from_raw_parts(p as *const f32, self.ty.len())
        });
        self.unlock()
    }

    /// Copy `from`, of the same type, into this buffer.
    pub fn copy_from(&self, from: &Buffer) -> Result<(), String> {
        if from.ty.len() != self.ty.len() || from.ty.0.element_type != self.ty.0.element_type {
            return Err(format!("{} and {} differ", from.name, self.name));
        }
        let src = from.lock(LOCK_READ)?;
        let dst = self.lock(LOCK_WRITE);
        let r = dst.map(|dst| unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, self.ty.bytes())
        });
        let u = if r.is_ok() { self.unlock() } else { Ok(()) };
        from.unlock()?;
        r.and(u)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { (self.env.fns().LiteRtDestroyTensorBuffer)(self.raw) }
    }
}

/// One signature of a compiled model, with a buffer for each input and output.
pub struct Signature {
    pub name: String,
    index: usize,
    inputs: Vec<(String, Rc<Buffer>)>,
    outputs: Vec<(String, Rc<Buffer>)>,
}

fn find<'a>(
    list: &'a [(String, Rc<Buffer>)],
    sig: &str,
    name: &str,
) -> Result<&'a Rc<Buffer>, String> {
    list.iter()
        .find(|(n, _)| n == name)
        .map(|(_, b)| b)
        .ok_or_else(|| format!("{sig} has no tensor {name}"))
}

impl Signature {
    pub fn input(&self, name: &str) -> Result<&Rc<Buffer>, String> {
        find(&self.inputs, &self.name, name)
    }

    pub fn output(&self, name: &str) -> Result<&Rc<Buffer>, String> {
        find(&self.outputs, &self.name, name)
    }

    pub fn input_names(&self) -> impl Iterator<Item = &str> {
        self.inputs.iter().map(|(n, _)| n.as_str())
    }

    /// Run this signature on `with` buffer in place of its own `name` input, from now on.
    pub fn bind_input(&mut self, name: &str, with: Rc<Buffer>) -> Result<(), String> {
        let slot = self.inputs.iter_mut().find(|(n, _)| n == name);
        let (_, b) = slot.ok_or_else(|| format!("{} has no input {name}", self.name))?;
        if b.ty.len() != with.ty.len() {
            return Err(format!("{} cannot take {} as {name}", self.name, with.name));
        }
        *b = with;
        Ok(())
    }
}

/// A model compiled for a set of accelerators.
pub struct Compiled {
    env: Rc<Env>,
    model: Handle,
    raw: Handle,
    pub signatures: Vec<Signature>,
}

/// Qualcomm's settings: the NPU clocked up while it runs, and only errors logged.
/// `PTTS_LITERT_QNN_OPTIONS`, `key = value` pairs split by `;`, adds settings or replaces these.
fn qualcomm_options() -> String {
    let mut set = vec![("htp_performance_mode", "2".to_string()), ("log_level", "1".to_string())];
    let extra = std::env::var("PTTS_LITERT_QNN_OPTIONS").unwrap_or_default();
    for (key, value) in extra.split(';').filter_map(|kv| kv.split_once('=')) {
        set.retain(|(k, _)| *k != key.trim());
        set.push((key.trim(), value.trim().to_string()));
    }
    set.iter().map(|(k, v)| format!("{k} = {v}\n")).collect()
}

impl Compiled {
    /// Load `path` and compile it for `accelerators` (`ACCEL_*`), running the CPU parts on
    /// `threads` threads.
    pub fn new(
        env: &Rc<Env>,
        path: &Path,
        accelerators: c_int,
        threads: usize,
    ) -> Result<Self, String> {
        let f = env.fns();
        let file = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let mut model = std::ptr::null_mut();
        let what = format!("loading {}", path.display());
        check(
            f,
            unsafe { (f.LiteRtCreateModelFromFile)(env.raw, file.as_ptr(), &mut model) },
            &what,
        )?;
        let mut me =
            Self { env: env.clone(), model, raw: std::ptr::null_mut(), signatures: Vec::new() };

        let mut options = std::ptr::null_mut();
        check(f, unsafe { (f.LiteRtCreateOptions)(&mut options) }, "creating options")?;
        let r = (|| {
            check(
                f,
                unsafe { (f.LiteRtSetOptionsHardwareAccelerators)(options, accelerators) },
                "accelerators",
            )?;
            add_opaque(f, options, c"xnnpack", &format!("num_threads = {threads}\n"))?;
            if accelerators & ACCEL_NPU != 0 {
                add_opaque(f, options, c"qualcomm", &qualcomm_options())?;
            }
            check(
                f,
                unsafe { (f.LiteRtCreateCompiledModel)(env.raw, model, options, &mut me.raw) },
                "compiling",
            )
        })();
        unsafe { (f.LiteRtDestroyOptions)(options) };
        r?;

        let mut n = 0;
        check(f, unsafe { (f.LiteRtGetNumModelSignatures)(model, &mut n) }, "signatures")?;
        for index in 0..n {
            let sig = me.signature(index)?;
            me.signatures.push(sig);
        }
        Ok(me)
    }

    fn signature(&self, index: usize) -> Result<Signature, String> {
        let f = self.env.fns();
        let mut sig = std::ptr::null_mut();
        check(f, unsafe { (f.LiteRtGetModelSignature)(self.model, index, &mut sig) }, "signature")?;
        let mut key = std::ptr::null();
        check(f, unsafe { (f.LiteRtGetSignatureKey)(sig, &mut key) }, "signature key")?;
        let name = unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned();

        let buffers = |output: bool| -> Result<Vec<(String, Rc<Buffer>)>, String> {
            let (count, get_name, get_tensor, get_reqs) = if output {
                (
                    f.LiteRtGetNumSignatureOutputs,
                    f.LiteRtGetSignatureOutputName,
                    f.LiteRtGetSignatureOutputTensorByIndex,
                    f.LiteRtGetCompiledModelOutputBufferRequirements,
                )
            } else {
                (
                    f.LiteRtGetNumSignatureInputs,
                    f.LiteRtGetSignatureInputName,
                    f.LiteRtGetSignatureInputTensorByIndex,
                    f.LiteRtGetCompiledModelInputBufferRequirements,
                )
            };
            let mut n = 0;
            check(f, unsafe { count(sig, &mut n) }, &name)?;
            let mut list = Vec::with_capacity(n);
            for i in 0..n {
                let mut p = std::ptr::null();
                check(f, unsafe { get_name(sig, i, &mut p) }, &name)?;
                let tname = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
                let what = format!("{name}.{tname}");
                let mut tensor = std::ptr::null_mut();
                check(f, unsafe { get_tensor(sig, i, &mut tensor) }, &what)?;
                let mut ty: RankedTensorType = unsafe { std::mem::zeroed() };
                check(f, unsafe { (f.LiteRtGetRankedTensorType)(tensor, &mut ty) }, &what)?;
                // The requirements stay the compiled model's.
                let mut reqs = std::ptr::null_mut();
                check(f, unsafe { get_reqs(self.raw, index, i, &mut reqs) }, &what)?;
                let mut raw = std::ptr::null_mut();
                check(
                    f,
                    unsafe {
                        (f.LiteRtCreateManagedTensorBufferFromRequirements)(
                            self.env.raw,
                            &ty,
                            reqs,
                            &mut raw,
                        )
                    },
                    &what,
                )?;
                let buf = Buffer { env: self.env.clone(), raw, ty: TensorType(ty), name: what };
                list.push((tname, Rc::new(buf)));
            }
            Ok(list)
        };
        let inputs = buffers(false)?;
        let outputs = buffers(true)?;
        Ok(Signature { name, index, inputs, outputs })
    }

    pub fn signature_named(&self, name: &str) -> Result<usize, String> {
        self.signatures
            .iter()
            .position(|s| s.name == name)
            .ok_or_else(|| format!("the model has no signature {name}"))
    }

    /// Run signature `i` on its buffers.
    pub fn run(&self, i: usize) -> Result<(), String> {
        let f = self.env.fns();
        let sig = &self.signatures[i];
        let mut ins: Vec<Handle> = sig.inputs.iter().map(|(_, b)| b.raw).collect();
        let mut outs: Vec<Handle> = sig.outputs.iter().map(|(_, b)| b.raw).collect();
        let status = unsafe {
            (f.LiteRtRunCompiledModel)(
                self.raw,
                sig.index,
                ins.len(),
                ins.as_mut_ptr(),
                outs.len(),
                outs.as_mut_ptr(),
            )
        };
        check(f, status, &format!("running {}", sig.name))
    }
}

impl Drop for Compiled {
    fn drop(&mut self) {
        // Buffers first: they belong to the compiled model's memory.
        self.signatures.clear();
        let f = self.env.fns();
        unsafe {
            if !self.raw.is_null() {
                (f.LiteRtDestroyCompiledModel)(self.raw);
            }
            (f.LiteRtDestroyModel)(self.model);
        }
    }
}

/// Options for one of LiteRT's accelerators, which it takes as a TOML payload under the
/// accelerator's name: `xnnpack` for the CPU, `qualcomm` for Qualcomm NPUs.
fn add_opaque(f: &Fns, options: Handle, id: &CStr, toml: &str) -> Result<(), String> {
    unsafe extern "C" fn free_payload(p: *mut c_void) {
        drop(unsafe { CString::from_raw(p as *mut c_char) });
    }
    let what = format!("{} options", id.to_string_lossy());
    let payload = CString::new(toml).map_err(|e| e.to_string())?.into_raw() as *mut c_void;
    let mut opaque = std::ptr::null_mut();
    let status = unsafe {
        (f.LiteRtCreateOpaqueOptions)(id.as_ptr(), payload, Some(free_payload), &mut opaque)
    };
    if status != STATUS_OK {
        unsafe { free_payload(payload) };
    }
    check(f, status, &what)?;
    // `options` owns `opaque` from here on.
    check(f, unsafe { (f.LiteRtAddOpaqueOptions)(options, opaque) }, &what)
}
