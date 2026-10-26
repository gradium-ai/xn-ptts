//! Read model weights out of a safetensors file as f32.
//!
//! The checkpoint stores the transformer in BF16, which safetensors exposes as raw bytes; the
//! conversion is a 16-bit left shift into f32.
use safetensors::SafeTensors;
use std::collections::HashMap;

pub struct Weights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let buf = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let st = SafeTensors::deserialize(&buf).map_err(|e| format!("safetensors: {e}"))?;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            let shape = view.shape().to_vec();
            let raw = view.data();
            let data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => {
                    raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
                }
                safetensors::Dtype::BF16 => raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
                    .collect(),
                safetensors::Dtype::F16 => raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::f16::from_le_bytes(*c).to_f32())
                    .collect(),
                d => return Err(format!("{name}: unsupported dtype {d:?}")),
            };
            tensors.insert(name.to_string(), (shape, data));
        }
        Ok(Self { tensors })
    }

    /// From named `(shape, data)` tensors, for tests.
    #[cfg(test)]
    pub(crate) fn from_tensors(tensors: HashMap<String, (Vec<usize>, Vec<f32>)>) -> Self {
        Self { tensors }
    }

    /// Load from a GGUF, dequantising every tensor to f32.
    ///
    /// The f32 checkpoint is not always around, and CoreML wants plain floats to quantise its
    /// own way, so q8_0 blocks are expanded here rather than passed through.
    #[cfg(feature = "gguf")]
    pub fn open_gguf(path: &std::path::Path) -> Result<Self, String> {
        use xn::quantized::gguf_file;
        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let content = gguf_file::Content::read(&mut f).map_err(|e| format!("gguf: {e}"))?;
        let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        let mut tensors = HashMap::new();
        for name in names {
            let qt = content.tensor(&mut f, &name).map_err(|e| format!("{name}: {e}"))?;
            let shape = qt.shape().dims().to_vec();
            let data = qt.dequantize().map_err(|e| format!("{name}: {e}"))?;
            tensors.insert(name, (shape, data));
        }
        Ok(Self { tensors })
    }

    /// Rename every tensor through `f`, dropping those it maps to `None` -- for reading a
    /// checkpoint under the names `ptts` uses (`ptts::loader::remap_key`).
    pub fn renamed(self, f: impl Fn(&str) -> Option<String>) -> Self {
        let tensors = self.tensors.into_iter().filter_map(|(k, v)| f(&k).map(|k| (k, v))).collect();
        Self { tensors }
    }

    pub fn get(&self, name: &str) -> Result<(&[usize], &[f32]), String> {
        self.tensors
            .get(name)
            .map(|(s, d)| (s.as_slice(), d.as_slice()))
            .ok_or_else(|| format!("missing tensor {name}"))
    }

    pub fn data(&self, name: &str) -> Result<&[f32], String> {
        Ok(self.get(name)?.1)
    }
}
