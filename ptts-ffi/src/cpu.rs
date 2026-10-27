//! The engine everywhere but Apple: [`ptts::synth::Synth`] on the CPU, with `dir` a checkpoint
//! folder as the other frontends read it.
//!
//! The folder holds `tokenizer.json`, the weights (q8_0 GGUF or f32 safetensors, named as in
//! [`WEIGHTS`]), its own `config.json` and its voices
//! in `voices/` or `embeddings/`, or as `default-voice.safetensors`.

use crate::{PTTS_UNIT_CPU, PttsResult};
use ptts::checkpoint::read_config;
use ptts::preprocess::Normalize;
use ptts::synth::{DeviceKind, Quant, SpeechOptions, Synth, SynthBuilder};
use std::path::Path;

/// Weight files tried in order, with the format each loads as.
const WEIGHTS: [(&str, Quant); 2] =
    [("model.q8.gguf", Quant::Q80), ("model.safetensors", Quant::F32)];

pub struct Engine {
    synth: Synth,
    voices: Vec<String>,
    languages: Vec<String>,
    voice: Option<String>,
    /// Applied here rather than by `synth`, which is built with `Normalize::OFF`, so that
    /// `set_language` can change it without a reload. `Synth` applies it first thing anyway.
    normalize: Normalize,
    /// Samples per frame, to count frames in what the stream hands back.
    frame_size: usize,
}

impl Engine {
    pub fn open(dir: &Path, unit: u32, lang: &str) -> Result<Self, String> {
        if unit != PTTS_UNIT_CPU {
            return Err(format!("compute unit {unit} is Apple only; use PTTS_UNIT_CPU"));
        }
        let normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        let config = dir.join("config.json");
        let cfg = read_config(&config).map_err(|e| e.to_string())?;
        let (weights, quant) =
            WEIGHTS.iter().map(|&(name, q)| (dir.join(name), q)).find(|(p, _)| p.is_file()).ok_or(
                format!(
                    "no weights in {}; expected one of {}",
                    dir.display(),
                    WEIGHTS.map(|(name, _)| name).join(", ")
                ),
            )?;
        let frame_rate = cfg.mimi.frame_rate;
        let mut builder = SynthBuilder::new(cfg, weights, Normalize::OFF)
            .tokenizer_file(dir.join("tokenizer.json"))
            .device(DeviceKind::Cpu)
            .quant(quant);
        for (name, path) in ptts::loader::checkpoint_voices(dir) {
            builder = builder.add_voice(name, path);
        }
        let synth = builder.build().map_err(|e| e.to_string())?;
        let frame_size = (synth.sample_rate() as f64 / frame_rate).round() as usize;
        Ok(Self {
            voices: synth.voices(),
            languages: synth.languages().to_vec(),
            voice: synth.default_voice(),
            normalize,
            synth,
            frame_size,
        })
    }

    pub fn voices(&self) -> &[String] {
        &self.voices
    }

    pub fn languages(&self) -> &[String] {
        &self.languages
    }

    /// Switch to one of [`Self::voices`].
    pub fn set_voice(&mut self, name: &str) -> Result<(), String> {
        self.voice = Some(name.to_string());
        Ok(())
    }

    /// Normalize text as `lang` from the next utterance on.
    pub fn set_language(&mut self, lang: &str) -> Result<(), String> {
        self.normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn speak(
        &mut self,
        text: &str,
        sink: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<PttsResult, String> {
        let start = std::time::Instant::now();
        let opts = SpeechOptions { voice: self.voice.clone(), ..Default::default() };
        let text = self.normalize.apply(text);
        let stream = self.synth.stream_with(&text, &opts).map_err(|e| e.to_string())?;
        let (mut samples, mut ttfa) = (0, None);
        // Dropping the stream on an early return stops generation.
        for pcm in stream {
            let pcm = pcm.map_err(|e| e.to_string())?;
            if pcm.is_empty() {
                continue;
            }
            ttfa.get_or_insert(start.elapsed());
            samples += pcm.len();
            if !sink(&pcm) {
                break;
            }
        }
        Ok(PttsResult {
            frames: samples.div_ceil(self.frame_size) as u32,
            samples,
            ttfa_ms: ttfa.unwrap_or_default().as_secs_f64() * 1e3,
            total_ms: start.elapsed().as_secs_f64() * 1e3,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PTTS_UNIT_ANE;

    fn open_error(unit: u32, lang: &str) -> String {
        match Engine::open(Path::new("no-such-checkpoint"), unit, lang) {
            Ok(_) => panic!("opened a checkpoint that does not exist"),
            Err(e) => e,
        }
    }

    // Each is refused before anything is loaded, so none needs weights.
    #[test]
    fn open_refuses_before_loading() {
        assert!(open_error(PTTS_UNIT_ANE, "en").contains("Apple only"));
        assert!(!open_error(PTTS_UNIT_CPU, "klingon").contains("no weights"));
        assert!(open_error(PTTS_UNIT_CPU, "en").contains("config.json"));
    }
}
