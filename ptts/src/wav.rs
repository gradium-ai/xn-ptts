use std::io::prelude::*;

pub trait Sample {
    fn to_i16(&self) -> i16;
}

impl Sample for f32 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for f64 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for i16 {
    fn to_i16(&self) -> i16 {
        *self
    }
}

pub fn write_pcm_as_wav<W: Write, S: Sample>(
    w: &mut W,
    samples: &[S],
    sample_rate: u32,
    n_channels: u32,
) -> std::io::Result<()> {
    let len = 12u32; // header
    let len = len + 24u32; // fmt
    let len = len + samples.len() as u32 * 2 + 8; // data
    let bytes_per_second = sample_rate * 2 * n_channels;
    w.write_all(b"RIFF")?;
    w.write_all(&(len - 8).to_le_bytes())?; // total length minus 8 bytes
    w.write_all(b"WAVE")?;

    // Format block
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // bits per sample
    w.write_all(&1u16.to_le_bytes())?; // 1: PCM int, 2: PCM float
    w.write_all(&(n_channels as u16).to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&bytes_per_second.to_le_bytes())?;
    w.write_all(&2u16.to_le_bytes())?; // 2 bytes of data per sample
    w.write_all(&16u16.to_le_bytes())?; // bits per sample

    // Data block
    w.write_all(b"data")?;
    w.write_all(&(samples.len() as u32 * 2).to_le_bytes())?;
    for sample in samples.iter() {
        w.write_all(&sample.to_i16().to_le_bytes())?
    }
    Ok(())
}

/// Write mono PCM to a WAV file at `path`.
///
/// A convenience over [`write_pcm_as_wav`] for the common case of saving a
/// finished generation: open the file, wrap it in a `BufWriter`, write, flush.
pub fn write_wav_file<P: AsRef<std::path::Path>>(
    path: P,
    pcm: &[f32],
    sample_rate: u32,
) -> xn::Result<()> {
    let file = std::fs::File::create(path.as_ref())
        .map_err(|e| xn::Error::msg(format!("cannot create {}: {e}", path.as_ref().display())))?;
    let mut writer = std::io::BufWriter::new(file);
    write_pcm_as_wav(&mut writer, pcm, sample_rate, 1).map_err(xn::Error::wrap)?;
    writer.flush().map_err(xn::Error::wrap)?;
    Ok(())
}
