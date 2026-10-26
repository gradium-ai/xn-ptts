//! The "blob v2" weight file, reverse-engineered from a coremltools package.
//!
//! Layout, all offsets 64-byte aligned:
//!
//! ```text
//! 0    : u32 count, u32 version(=2), then zero padding to 64
//! then for each blob:
//!   u32 0xdeadbeef, u32 dtype, u64 size_in_bytes, u64 payload_offset, zero padding to 64
//!   payload
//! ```
//!
//! Weights live here rather than inline in the protobuf, which keeps the model spec small and
//! lets CoreML map them straight from the file.

/// Blob element types, which are numbered differently again from MIL and ArrayFeatureType.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlobDType {
    Fp16 = 1,
    Fp32 = 2,
}

const ALIGN: usize = 64;

pub struct BlobWriter {
    buf: Vec<u8>,
    count: u32,
}

impl Default for BlobWriter {
    // Not derived: an empty buffer lacks the header `finish` writes into.
    fn default() -> Self {
        Self::new()
    }
}

impl BlobWriter {
    pub fn new() -> Self {
        Self { buf: vec![0u8; ALIGN], count: 0 }
    }

    fn pad(&mut self) {
        let r = self.buf.len() % ALIGN;
        if r != 0 {
            self.buf.resize(self.buf.len() + (ALIGN - r), 0);
        }
    }

    /// Append a payload, returning the offset of its metadata record -- which is what
    /// `blobFileValue.offset` refers to, not the payload itself.
    pub fn add(&mut self, dtype: BlobDType, payload: &[u8]) -> u64 {
        self.pad();
        let meta_off = self.buf.len() as u64;
        let data_off = meta_off + ALIGN as u64;
        let mut meta = Vec::with_capacity(ALIGN);
        meta.extend_from_slice(&0xdead_beefu32.to_le_bytes());
        meta.extend_from_slice(&(dtype as u32).to_le_bytes());
        meta.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        meta.extend_from_slice(&data_off.to_le_bytes());
        meta.resize(ALIGN, 0);
        self.buf.extend_from_slice(&meta);
        self.buf.extend_from_slice(payload);
        self.count += 1;
        meta_off
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.pad();
        self.buf[0..4].copy_from_slice(&self.count.to_le_bytes());
        self.buf[4..8].copy_from_slice(&2u32.to_le_bytes());
        self.buf
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}
