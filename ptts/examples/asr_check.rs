//! Smoke test for `ptts::asr` against a wav file, on whichever backend is built in.
//!
//! ```bash
//! cargo run --release --features sp,webgpu --example asr_check -- audio.wav
//! ```
use anyhow::{Context, Result};
use ptts::asr::{AsrModel, Event, FRAME_SIZE, INITIAL_SILENCE_FRAMES, SAMPLE_RATE, remap_key};
use ptts::asr_lm::Config;
use xn::nn::VB;

#[path = "audio_helpers.rs"]
mod audio_helpers;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let audio: std::path::PathBuf = args.next().context("usage: asr_check <audio.wav>")?.into();
    let dir: std::path::PathBuf = args
        .next()
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/.cache/huggingface/hub/models--gr4d--asr-23b5a198.500/snapshots")
        })
        .into();
    let dir = if dir.join("config.json").is_file() {
        dir
    } else {
        std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.join("config.json").is_file())
            .context("no snapshot with config.json")?
    };
    println!("model dir: {}", dir.display());

    struct Run<'a> {
        dir: &'a std::path::Path,
        audio: &'a std::path::Path,
    }
    impl xn::WithQ for Run<'_> {
        type Output = ();
        fn run<Q: xn::BackendQ>(self, dev: Q::B) -> xn::Result<()> {
            let cfg: Config = serde_json::from_str(
                &std::fs::read_to_string(self.dir.join("config.json")).unwrap(),
            )
            .unwrap();
            let mimi_vb = VB::load_with_key_map(
                &[self.dir.join("mimi.safetensors")],
                dev.clone(),
                remap_key,
            )?
            .root();
            let lm_vb = VB::load_with_key_map(
                &[self.dir.join("model.safetensors")],
                dev.clone(),
                remap_key,
            )?
            .root();
            let model: AsrModel<Q> = AsrModel::load(&mimi_vb, &lm_vb, cfg, None)?;
            let tok = sentencepiece::SentencePieceProcessor::open(
                self.dir.join("tokenizer.model").to_str().unwrap(),
            )
            .map_err(xn::Error::wrap)?;

            let (pcm, sr) = audio_helpers::pcm_decode(self.audio).map_err(xn::Error::wrap)?;
            let pcm = if sr as usize == SAMPLE_RATE {
                pcm
            } else {
                audio_helpers::resample(&pcm, sr as usize, SAMPLE_RATE).map_err(xn::Error::wrap)?
            };
            let pcm = [
                vec![0.0; FRAME_SIZE * INITIAL_SILENCE_FRAMES],
                pcm,
                vec![0.0; FRAME_SIZE * model.delay_frames()],
            ]
            .concat();

            let mut state = model.init_state()?;
            let mut all: Vec<u32> = vec![];
            let t0 = std::time::Instant::now();
            let frames = pcm.len() / FRAME_SIZE;
            for i in 0..frames {
                for ev in model.step(&mut state, &pcm[i * FRAME_SIZE..(i + 1) * FRAME_SIZE], 0.0)? {
                    if let Event::Word { tokens, .. } = ev {
                        all.push(ptts::asr::TOKEN_PAD);
                        all.extend_from_slice(&tokens);
                    }
                }
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let audio_ms = frames as f64 / ptts::asr::FRAME_RATE * 1e3;
            println!("device {}  frames {frames}", xn::Backend::name(&dev));
            println!(
                "transcribe {ms:.0} ms for {audio_ms:.0} ms audio -> {:.2}x realtime",
                audio_ms / ms
            );
            println!("transcript: {}", tok.decode_piece_ids(&all).unwrap_or_default());
            Ok(())
        }
    }
    let dtype = match std::env::var("ASR_DTYPE").as_deref() {
        Ok("f16") => xn::DTypeQ::F16,
        Ok("bf16") => xn::DTypeQ::BF16,
        _ => xn::DTypeQ::F32,
    };
    xn::Runner::new().dtype(dtype).run(Run { dir: &dir, audio: &audio }, 0)?;
    Ok(())
}
