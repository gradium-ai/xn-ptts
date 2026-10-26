//! Loudness normalization for voice prompts: [`normalize_loudness`] brings a recording to a
//! fixed level before the speaker encoder embeds it.

use xn::Result;

/// Target loudness (in dB LUFS) used when normalizing voice-embedding input.
pub const LOUDNESS_HEADROOM_DB: f64 = 18.0;
/// RMS level below which the signal is considered too quiet to rescale.
pub const LOUDNESS_ENERGY_FLOOR: f32 = 2e-3;

/// Normalize `pcm` to a target loudness (in dB LUFS) following ITU-R BS.1770-4.
///
/// The signal is first centered (DC removed), then skipped entirely if its RMS
/// is below [`LOUDNESS_ENERGY_FLOOR`], otherwise rescaled so that its
/// integrated loudness equals `-LOUDNESS_HEADROOM_DB` LUFS.
pub fn normalize_loudness(pcm: &mut [f32], sample_rate: u32) -> Result<()> {
    if pcm.is_empty() {
        return Ok(());
    }
    // Remove DC offset.
    let mean = (pcm.iter().map(|&s| s as f64).sum::<f64>() / pcm.len() as f64) as f32;
    for s in pcm.iter_mut() {
        *s -= mean;
    }
    // Skip rescaling for signals below the energy floor (matches the Python
    // version which returns the input unchanged in that case).
    let rms = (pcm.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / pcm.len() as f64).sqrt()
        as f32;
    if rms < LOUDNESS_ENERGY_FLOOR {
        return Ok(());
    }
    let mut meter =
        ebur128::EbuR128::new(1, sample_rate, ebur128::Mode::I).map_err(xn::Error::wrap)?;
    meter.add_frames_f32(pcm).map_err(xn::Error::wrap)?;
    let input_loudness_db = match meter.loudness_global() {
        Ok(l) if l.is_finite() => l,
        // Audio too short or silent — leave it unchanged, matching the Python
        // fallback where `torchaudio.transforms.Loudness` raises `RuntimeError`.
        _ => return Ok(()),
    };
    let delta_loudness = -LOUDNESS_HEADROOM_DB - input_loudness_db;
    let gain = 10f64.powf(delta_loudness / 20.0) as f32;
    if !gain.is_finite() {
        return Ok(());
    }
    for s in pcm.iter_mut() {
        *s *= gain;
    }
    Ok(())
}
