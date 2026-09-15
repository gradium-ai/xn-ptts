//! Reading audio files, for voice cloning.
//!
//! [`crate::synth::SynthOf::add_voice_from_pcm`] wants mono `f32` at the
//! speaker codec's sample rate, which means decoding a file and resampling it.
//! Both examples that clone a voice need exactly that, and so does anything
//! outside this repo, so it lives here rather than in a private example module
//! no dependent can reach.
//!
//! Behind the `audio` feature: `symphonia` and `rubato` are large and neither
//! belongs in the wasm build.

use xn::Result;

/// Decode the first channel of an audio file to `f32`, returning the samples
/// and the file's own sample rate.
///
/// Format is detected from the stream, so anything `symphonia` supports works.
pub fn decode_file(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::CODEC_TYPE_NULL;

    let file = std::fs::File::open(path)
        .map_err(|e| xn::Error::msg(format!("cannot open {}: {e}", path.display())))?;
    let stream = symphonia::core::io::MediaSourceStream::new(Box::new(file), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &symphonia::core::probe::Hint::new(),
            stream,
            &Default::default(),
            &Default::default(),
        )
        .map_err(|e| xn::Error::msg(format!("cannot read {}: {e}", path.display())))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| xn::Error::msg(format!("no audio track in {}", path.display())))?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| xn::Error::msg(format!("{} declares no sample rate", path.display())))?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .map_err(|e| xn::Error::msg(format!("unsupported codec in {}: {e}", path.display())))?;

    let mut pcm = Vec::new();
    // `next_packet` returning Err is how symphonia signals end of stream, so a
    // clean finish and a truncated file look the same here.
    while let Ok(packet) = format.next_packet() {
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder
            .decode(&packet)
            .map_err(|e| xn::Error::msg(format!("cannot decode {}: {e}", path.display())))?;
        match decoded {
            AudioBufferRef::F32(buf) => pcm.extend(buf.chan(0)),
            AudioBufferRef::U8(b) => append(&mut pcm, b),
            AudioBufferRef::U16(b) => append(&mut pcm, b),
            AudioBufferRef::U24(b) => append(&mut pcm, b),
            AudioBufferRef::U32(b) => append(&mut pcm, b),
            AudioBufferRef::S8(b) => append(&mut pcm, b),
            AudioBufferRef::S16(b) => append(&mut pcm, b),
            AudioBufferRef::S24(b) => append(&mut pcm, b),
            AudioBufferRef::S32(b) => append(&mut pcm, b),
            AudioBufferRef::F64(b) => append(&mut pcm, b),
        }
    }
    if pcm.is_empty() {
        xn::bail!("{} decoded to no samples", path.display())
    }
    Ok((pcm, sample_rate))
}

fn append<T>(pcm: &mut Vec<f32>, buf: std::borrow::Cow<'_, symphonia::core::audio::AudioBuffer<T>>)
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    use symphonia::core::audio::Signal;
    use symphonia::core::conv::FromSample;
    pcm.extend(buf.chan(0).iter().map(|v| f32::from_sample(*v)))
}

/// Resample mono `f32` from `sr_in` to `sr_out`. A no-op when they match.
pub fn resample(pcm: &[f32], sr_in: usize, sr_out: usize) -> Result<Vec<f32>> {
    use rubato::Resampler;

    if sr_in == sr_out {
        return Ok(pcm.to_vec());
    }
    let mut out =
        Vec::with_capacity((pcm.len() as f64 * sr_out as f64 / sr_in as f64) as usize + 1024);
    let mut resampler =
        rubato::FftFixedInOut::<f32>::new(sr_in, sr_out, 1024, 1).map_err(xn::Error::wrap)?;
    let mut buf = resampler.output_buffer_allocate(true);

    let mut pos = 0;
    while pos + resampler.input_frames_next() < pcm.len() {
        let (used, made) = resampler
            .process_into_buffer(&[&pcm[pos..]], &mut buf, None)
            .map_err(xn::Error::wrap)?;
        pos += used;
        out.extend_from_slice(&buf[0][..made]);
    }
    if pos < pcm.len() {
        let (_, made) = resampler
            .process_partial_into_buffer(Some(&[&pcm[pos..]]), &mut buf, None)
            .map_err(xn::Error::wrap)?;
        out.extend_from_slice(&buf[0][..made]);
    }
    Ok(out)
}

/// Decode `path` to mono `f32` at `sample_rate` — [`decode_file`] then
/// [`resample`], which is what a voice prompt needs.
///
/// Pair with `voice_prompt_sample_rate`:
///
/// ```no_run
/// # fn main() -> xn::Result<()> {
/// # let mut tts: ptts::synth::Synth = todo!();
/// let pcm = ptts::audio::load_mono_at("voice.wav".as_ref(), tts.voice_prompt_sample_rate())?;
/// tts.add_voice_from_pcm("mine", &pcm)?;
/// # Ok(())
/// # }
/// ```
pub fn load_mono_at(path: &std::path::Path, sample_rate: usize) -> Result<Vec<f32>> {
    let (pcm, file_rate) = decode_file(path)?;
    tracing::info!(?path, samples = pcm.len(), rate = file_rate, "decoded audio");
    resample(&pcm, file_rate as usize, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_to_the_same_rate_is_a_copy() {
        let pcm = vec![0.1, -0.2, 0.3];
        assert_eq!(resample(&pcm, 24000, 24000).unwrap(), pcm);
    }

    #[test]
    fn resampling_scales_the_length() {
        // 2s at 16kHz to 24kHz should land near 2s again; the FFT resampler
        // works in blocks, so allow a block of slack.
        let pcm = vec![0f32; 32_000];
        let out = resample(&pcm, 16_000, 24_000).unwrap();
        assert!(
            (out.len() as f64 - 48_000.0).abs() < 2048.0,
            "expected ~48000 samples, got {}",
            out.len()
        );
    }

    #[test]
    fn a_missing_file_names_itself() {
        let err = decode_file("/nope/not-audio.wav".as_ref()).unwrap_err().to_string();
        assert!(err.contains("/nope/not-audio.wav"), "{err}");
    }
}
