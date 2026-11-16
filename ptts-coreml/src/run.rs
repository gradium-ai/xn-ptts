//! Load and run a compiled ML Program through CoreML.
//!
//! The compiler ships with the OS: `MLModel.compileModel(at:)` turns a `.mlpackage` into a
//! `.mlmodelc`, so nothing beyond the OS is needed on the machine that runs this.

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_foundation::CFRetained;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_core_video::CVPixelBuffer;
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use std::collections::HashMap;
use std::path::Path;

/// Where CoreML may run a model. The flow LM wants the Neural Engine; Mimi, which is f32, runs
/// on the CPU either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compute {
    CpuOnly,
    CpuAndNeuralEngine,
}

impl Compute {
    fn units(self) -> MLComputeUnits {
        match self {
            Compute::CpuOnly => MLComputeUnits::CPUOnly,
            Compute::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
        }
    }
}

pub struct Model {
    inner: Retained<MLModel>,
    /// Declared element type per input, so callers can hand us f32 and we convert. Getting this
    /// wrong is silent: CoreML reinterprets the bytes and you get fp16 patterns read as f32.
    input_dtypes: HashMap<String, MLMultiArrayDataType>,
}

/// Apple documents `MLModel` prediction as thread-safe, and the fields beside it are read-only
/// after construction. That is what lets Mimi decode on one thread while the flow LM runs the
/// next frame on another.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

fn url(path: &Path) -> Retained<NSURL> {
    let s = NSString::from_str(path.to_str().expect("non-utf8 path"));
    NSURL::fileURLWithPath(&s)
}

fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    if from.is_dir() {
        std::fs::create_dir_all(to)?;
        for e in std::fs::read_dir(from)? {
            let e = e?;
            copy_tree(&e.path(), &to.join(e.file_name()))?;
        }
    } else {
        if let Some(p) = to.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::copy(from, to)?;
    }
    Ok(())
}

fn as_object<T>(v: &T) -> &AnyObject {
    unsafe { &*(v as *const T as *const AnyObject) }
}

/// An fp16 tensor in an IOSurface-backed pixel buffer, which the Neural Engine reads and writes
/// in place. Used for the KV cache.
///
/// A malloc-backed array is copied into ANE-addressable memory on every call, on the CPU: for
/// the cache that was ~1.4 ms a step on the phone, and it competed with Mimi for the cores.
/// Byte access goes through the pixel buffer's lock and never `dataPointer`: that call locks the
/// array, and CoreML then refuses to use it.
pub struct Buf {
    arr: Retained<MLMultiArray>,
    pb: CFRetained<CVPixelBuffer>,
}

/// Same reasoning as `Model`: only the predicting thread touches it.
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    /// Zero-filled, as a one-component half-float pixel buffer whose width is the innermost
    /// dimension and whose height is everything else.
    pub fn new(shape: &[usize]) -> Result<Self, String> {
        use objc2_core_foundation::{CFDictionary, CFString};
        use objc2_core_video::{
            CVPixelBufferCreate, kCVPixelBufferIOSurfacePropertiesKey,
            kCVPixelFormatType_OneComponent16Half,
        };
        let width = *shape.last().ok_or("empty shape")?;
        let height: usize = shape[..shape.len() - 1].iter().product();
        // An empty IOSurface-properties dictionary is what asks for IOSurface backing.
        let inner = CFDictionary::<CFString, CFString>::from_slices(&[], &[]);
        let key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
        let attrs = CFDictionary::<CFString, CFDictionary<CFString, CFString>>::from_slices(
            &[key],
            &[&inner],
        );
        let mut pb: *mut CVPixelBuffer = std::ptr::null_mut();
        let rc = unsafe {
            CVPixelBufferCreate(
                None,
                width,
                height,
                kCVPixelFormatType_OneComponent16Half,
                Some(attrs.as_opaque()),
                std::ptr::NonNull::from(&mut pb),
            )
        };
        let pb = std::ptr::NonNull::new(pb)
            .filter(|_| rc == 0)
            .ok_or_else(|| format!("CVPixelBufferCreate failed: {rc}"))?;
        let pb = unsafe { CFRetained::from_raw(pb) };
        let arr = unsafe {
            MLMultiArray::initWithPixelBuffer_shape(MLMultiArray::alloc(), &pb, &ns_shape(shape))
        };
        let b = Buf { arr, pb };
        b.with_bytes_mut(|p, n| unsafe { std::ptr::write_bytes(p, 0, n) });
        Ok(b)
    }

    /// Run `f` over the buffer's bytes under the pixel buffer's lock.
    pub fn with_bytes_mut(&self, f: impl FnOnce(*mut u8, usize)) {
        use objc2_core_video::{
            CVPixelBufferGetBaseAddress, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
            CVPixelBufferUnlockBaseAddress,
        };
        unsafe {
            CVPixelBufferLockBaseAddress(&self.pb, CVPixelBufferLockFlags(0));
            let p = CVPixelBufferGetBaseAddress(&self.pb) as *mut u8;
            if !p.is_null() {
                f(p, byte_extent(&self.arr));
            }
            CVPixelBufferUnlockBaseAddress(&self.pb, CVPixelBufferLockFlags(0));
        }
    }

    /// Copy the contents of another buffer of the same shape.
    pub fn copy_from(&self, other: &Buf) {
        other.with_bytes_mut(|src, ns| {
            self.with_bytes_mut(|dst, nd| unsafe {
                std::ptr::copy_nonoverlapping(src as *const u8, dst, ns.min(nd));
            });
        });
    }
}

/// A prediction's fixed parts, built once: the feature provider over prebound buffers and
/// preallocated small-input arrays. The per-call path was rebuilding ~80 Objective-C objects a
/// step, which on a phone was 1.3 ms of a 7.4 ms step.
pub struct Session {
    provider: Retained<MLDictionaryFeatureProvider>,
    small: Vec<(String, Retained<MLMultiArray>, Vec<usize>, MLMultiArrayDataType)>,
}

/// Same grounds as `Model` and `Buf`: only the predicting thread touches it.
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Model {
    /// Compile a `.mlpackage` into `compiled`, unless that is already there. Returns whether it
    /// had to do any work.
    ///
    /// Compiling takes ~30 s for the flow LM, and CoreML puts the result in a temp directory it
    /// later reclaims, so the result is kept.
    pub fn precompile(package: &Path, compiled: &Path) -> Result<bool, String> {
        if compiled.exists() {
            return Ok(false);
        }
        let tmp = compile(package)?;
        let from =
            std::path::PathBuf::from(tmp.path().ok_or("compiled model has no path")?.to_string());
        copy_tree(&from, compiled).map_err(|e| format!("caching {}: {e}", compiled.display()))?;
        Ok(true)
    }

    /// Load a compiled `.mlmodelc`, compiling `package` into it first if it is missing.
    /// `function` picks one function of a program that has several; `None` is its default.
    pub fn load(
        package: &Path,
        compiled: &Path,
        compute: Compute,
        function: Option<&str>,
    ) -> Result<Self, String> {
        Self::precompile(package, compiled)?;
        let cfg = unsafe { MLModelConfiguration::new() };
        unsafe { cfg.setComputeUnits(compute.units()) };
        if let Some(f) = function {
            unsafe { cfg.setFunctionName(Some(&NSString::from_str(f))) };
        }
        let inner =
            unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url(compiled), &cfg) }
                .map_err(|e| format!("{}: {e:?}", compiled.display()))?;
        let mut input_dtypes = HashMap::new();
        unsafe {
            let descs = inner.modelDescription().inputDescriptionsByName();
            for name in descs.allKeys().iter() {
                if let Some(c) = descs.objectForKey(&name).and_then(|d| d.multiArrayConstraint()) {
                    input_dtypes.insert(name.to_string(), c.dataType());
                }
            }
        }
        Ok(Self { inner, input_dtypes })
    }

    /// Build a session: `small` are the inputs whose values change each call, given by shape,
    /// whose arrays are allocated once and rewritten in place; `bufs` are bound once for good.
    pub fn session(
        &self,
        small: &[(&str, &[usize])],
        bufs: &[(&str, &Buf)],
    ) -> Result<Session, String> {
        let mut keys: Vec<Retained<NSString>> = Vec::new();
        let mut vals: Vec<Retained<MLFeatureValue>> = Vec::new();
        let mut arrays = Vec::new();
        for (name, shape) in small {
            let dt = self.input_dtypes.get(*name).copied().unwrap_or(MLMultiArrayDataType::Float32);
            let arr = unsafe {
                MLMultiArray::initWithShape_dataType_error(
                    MLMultiArray::alloc(),
                    &ns_shape(shape),
                    dt,
                )
            }
            .map_err(|e| format!("MLMultiArray alloc: {e:?}"))?;
            keys.push(NSString::from_str(name));
            vals.push(unsafe { MLFeatureValue::featureValueWithMultiArray(&arr) });
            arrays.push((name.to_string(), arr, shape.to_vec(), dt));
        }
        for (name, b) in bufs {
            keys.push(NSString::from_str(name));
            vals.push(unsafe { MLFeatureValue::featureValueWithMultiArray(&b.arr) });
        }
        let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
        let val_refs: Vec<&AnyObject> = vals.iter().map(|v| as_object(&**v)).collect();
        let dict = NSDictionary::from_slices(&key_refs, &val_refs);
        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
        }
        .map_err(|e| format!("feature provider: {e:?}"))?;
        Ok(Session { provider, small: arrays })
    }

    /// Predict through a session, rewriting only the small inputs' values. Outputs come back as
    /// f32 by name.
    pub fn predict(
        &self,
        s: &Session,
        values: &[(&str, &[f32])],
    ) -> Result<HashMap<String, Vec<f32>>, String> {
        for (name, data) in values {
            let Some((_, arr, shape, dt)) = s.small.iter().find(|(n, ..)| n == name) else {
                return Err(format!("session has no input named {name}"));
            };
            fill_array(arr, data, shape, *dt);
        }
        let fp = ProtocolObject::<dyn MLFeatureProvider>::from_ref(&*s.provider);
        let out = unsafe { self.inner.predictionFromFeatures_error(fp) }
            .map_err(|e| format!("predict: {e:?}"))?;
        let mut res = HashMap::new();
        for n in unsafe { out.featureNames() }.iter() {
            let Some(fv) = (unsafe { out.featureValueForName(&n) }) else { continue };
            let Some(arr) = (unsafe { fv.multiArrayValue() }) else { continue };
            res.insert(n.to_string(), read_f32(&arr));
        }
        Ok(res)
    }
}

fn ns_shape(shape: &[usize]) -> Retained<NSArray<NSNumber>> {
    let dims: Vec<Retained<NSNumber>> =
        shape.iter().map(|&d| NSNumber::new_isize(d as isize)).collect();
    let refs: Vec<&NSNumber> = dims.iter().map(|d| &**d).collect();
    NSArray::from_slice(&refs)
}

/// Compile a `.mlpackage`, waiting for the asynchronous API to finish.
///
/// The synchronous `compileModelAtURL:error:` is deprecated, and using it leaves the process in
/// a state where every later prediction returns NaN -- reproducibly, and only in the process that
/// did the compiling, which makes it look like a model bug rather than an API one. The async
/// entry point does not do that.
fn compile(package: &Path) -> Result<Retained<NSURL>, String> {
    let src = url(package);
    let (tx, rx) = std::sync::mpsc::channel::<Result<Retained<NSURL>, String>>();
    let handler = block2::RcBlock::new(move |out: *mut NSURL, err: *mut NSError| {
        let msg = if out.is_null() {
            Err(if err.is_null() {
                "compileModelAtURL: failed with no error".to_string()
            } else {
                format!("compileModelAtURL: {:?}", unsafe { &*err })
            })
        } else {
            Ok(unsafe { Retained::retain(out) }.expect("non-null compiled URL"))
        };
        let _ = tx.send(msg);
    });
    unsafe { MLModel::compileModelAtURL_completionHandler(&src, &handler) };
    rx.recv().map_err(|_| "compileModelAtURL: completion handler never ran".to_string())?
}

fn elem_size(dt: MLMultiArrayDataType) -> usize {
    if dt == MLMultiArrayDataType::Float16 { 2 } else { 4 }
}

/// Write `data` into an array, honouring its strides.
// `dataPointer` is deprecated for `getMutableBytesWithHandler`, a callback around the same
// pointer; these arrays are only ever touched by the predicting thread, between predictions.
#[allow(deprecated)]
fn fill_array(arr: &MLMultiArray, data: &[f32], shape: &[usize], dt: MLMultiArrayDataType) {
    let strides = strides_of(arr);
    let runs: Vec<usize> = run_offsets(shape, &strides);
    let inner = if runs.len() == 1 { data.len() } else { *shape.last().unwrap_or(&1) };
    unsafe {
        let p = arr.dataPointer().as_ptr();
        for (r, &base) in runs.iter().enumerate() {
            let src = &data[(r * inner).min(data.len())..((r + 1) * inner).min(data.len())];
            if dt == MLMultiArrayDataType::Float16 {
                let dst = (p as *mut half::f16).add(base);
                for (i, &v) in src.iter().enumerate() {
                    *dst.add(i) = half::f16::from_f32(v);
                }
            } else {
                std::ptr::copy_nonoverlapping(src.as_ptr(), (p as *mut f32).add(base), src.len());
            }
        }
    }
}

#[allow(deprecated)]
fn read_f32(arr: &MLMultiArray) -> Vec<f32> {
    let n = unsafe { arr.count() } as usize;
    let shape: Vec<usize> =
        unsafe { arr.shape() }.iter().map(|d| d.as_isize().max(0) as usize).collect();
    let runs = run_offsets(&shape, &strides_of(arr));
    let inner = if runs.len() == 1 { n } else { *shape.last().unwrap_or(&1) };
    let fp16 = unsafe { arr.dataType() } == MLMultiArrayDataType::Float16;
    let mut out = Vec::with_capacity(n);
    unsafe {
        let p = arr.dataPointer().as_ptr();
        for &base in &runs {
            if fp16 {
                let src = (p as *const half::f16).add(base);
                out.extend((0..inner).map(|i| (*src.add(i)).to_f32()));
            } else {
                out.extend_from_slice(std::slice::from_raw_parts(
                    (p as *const f32).add(base),
                    inner,
                ));
            }
        }
    }
    out
}

/// Element offsets of each contiguous run, in logical order: one run for a packed array, else
/// one per innermost row.
fn run_offsets(shape: &[usize], strides: &[usize]) -> Vec<usize> {
    if is_contiguous(shape, strides) {
        return vec![0];
    }
    let mut idx = vec![0usize; shape.len()];
    let mut v = Vec::new();
    loop {
        v.push(offset(&idx, strides));
        if !bump_outer(&mut idx, shape) {
            return v;
        }
    }
}

/// `MLMultiArray` pads the innermost dimension for alignment, so element `k` of a logically
/// contiguous buffer is not always at offset `k`. Anything that assumes packing has to check.
fn strides_of(arr: &MLMultiArray) -> Vec<usize> {
    unsafe { arr.strides() }.iter().map(|n| n.as_isize().max(0) as usize).collect()
}

/// True when the array is packed row-major, which is the common case and takes a bulk copy.
fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut want = 1usize;
    for d in (0..shape.len()).rev() {
        if strides[d] != want {
            return false;
        }
        want *= shape[d];
    }
    true
}

/// Bytes actually spanned by an array, which is not `count * elem_size` when it is padded.
///
/// The last logical element sits at `sum((dim - 1) * stride)`, past the end of a naive
/// `count`-sized region, so zeroing only `count` elements would leave the tail of a padded
/// buffer holding whatever was there. None of the shapes used here are padded -- measured, the
/// two figures agree for every cache buffer -- so this is a guard against a shape change rather
/// than a fix for anything observed.
fn byte_extent(arr: &MLMultiArray) -> usize {
    let shape: Vec<usize> =
        unsafe { arr.shape() }.iter().map(|d| d.as_isize().max(0) as usize).collect();
    let strides = strides_of(arr);
    let last: usize = shape.iter().zip(&strides).map(|(d, s)| d.saturating_sub(1) * s).sum();
    (last + 1) * elem_size(unsafe { arr.dataType() })
}

fn offset(idx: &[usize], strides: &[usize]) -> usize {
    idx.iter().zip(strides).map(|(i, s)| i * s).sum()
}

/// Advance every axis but the innermost, which the caller copies as one run.
fn bump_outer(idx: &mut [usize], shape: &[usize]) -> bool {
    if idx.len() < 2 {
        return false;
    }
    for d in (0..idx.len() - 1).rev() {
        idx[d] += 1;
        if idx[d] < shape[d] {
            return true;
        }
        idx[d] = 0;
    }
    false
}
