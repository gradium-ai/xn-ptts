//! The engine on Apple platforms: the Core ML driver, with `dir` an exported bundle.
//!
//! Text preparation, normalization and tokenization are `ptts`'s, and generation is
//! `ptts_coreml`'s.

use crate::{PTTS_UNIT_ANE, PTTS_UNIT_CPU, PttsResult};
use ptts::plan::Chunk;
use ptts::preprocess::Normalize;
use ptts_coreml::Weights;
use ptts_coreml::phonon::driver::{Config, Phonon, Voice};
use ptts_coreml::run::Compute;
use std::path::{Path, PathBuf};

pub struct Engine {
    phonon: Phonon,
    tokenizer: ptts::tok::Tok,
    normalize: Normalize,
    dir: PathBuf,
    voices: Vec<String>,
    languages: Vec<String>,
}

fn voice_names(dir: &Path) -> Vec<String> {
    ptts::loader::voices_in(&dir.join("voices")).into_iter().map(|(name, _)| name).collect()
}

/// A voice as the exporter writes it: an `emb` tensor of `[1, T, D]` or `[T, D]`, and for a
/// voice baked into the checkpoint, the `conditions` it is spoken with.
fn load_voice(dir: &Path, name: &str) -> Result<Voice, String> {
    let w = Weights::open(&dir.join("voices").join(format!("{name}.safetensors")))?;
    let (shape, data) = w.get("emb").map_err(|_| format!("voice {name} has no `emb` tensor"))?;
    let len = if shape.len() == 3 { shape[1] } else { shape[0] };
    let conditions = w.data("conditions").ok().map(<[f32]>::to_vec);
    Ok(Voice { emb: data.to_vec(), len, conditions })
}

/// Mimi's frame rate, for the per-chunk frame budget.
const FRAME_RATE: f64 = 12.5;

impl Engine {
    pub fn open(dir: &Path, unit: u32, lang: &str) -> Result<Self, String> {
        let flow_unit = match unit {
            PTTS_UNIT_ANE => Compute::CpuAndNeuralEngine,
            PTTS_UNIT_CPU => Compute::CpuOnly,
            u => return Err(format!("unknown compute unit {u}")),
        };
        let meta: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("bundle.json")).map_err(|e| format!("bundle.json: {e}"))?,
        )
        .map_err(|e| format!("bundle.json: {e}"))?;
        let get =
            |v: &serde_json::Value, k: &str| v[k].as_f64().ok_or(format!("bundle.json has no {k}"));
        let int = |k: &str| get(&meta, k).map(|v| v as usize);
        let dim = |k: &str| get(&meta["dims"], k).map(|v| v as usize);
        let cfg = Config {
            dims: ptts_coreml::phonon::flow_lm::Dims {
                d: dim("d")?,
                heads: dim("heads")?,
                // Bundles written before these were exported with their defaults.
                attn: dim("attn").or_else(|_| dim("d"))?,
                kv_heads: dim("kv_heads").or_else(|_| dim("heads"))?,
                layers: dim("layers")?,
                ff: dim("ff")?,
                ldim: dim("ldim")?,
                z: dim("z").ok(),
                flow_d: dim("flow_d")?,
                flow_blocks: dim("flow_blocks")?,
            },
            ctx: int("ctx")?,
            prefill_len: int("prefill_len")?,
            mimi: ptts_coreml::phonon::mimi::Dims::from_json(&meta["mimi"])?,
            mimi_window: int("mimi_window")?,
            max_frames: int("max_frames")?,
            eos_threshold: get(&meta, "eos_threshold")? as f32,
            temperature: get(&meta, "temperature")? as f32,
            seed: 0,
            flow_unit,
        };
        let voices = voice_names(dir);
        let languages = meta["languages"]
            .as_array()
            .map(|a| a.iter().filter_map(|l| l.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        let voice = load_voice(dir, voices.first().ok_or("no voices in the bundle")?)?;
        let tokenizer =
            ptts::tok::Tok::open(&dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
        let normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        Ok(Self {
            phonon: Phonon::load(dir, cfg, voice)?,
            tokenizer,
            normalize,
            dir: dir.to_path_buf(),
            voices,
            languages,
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
        self.phonon.set_voice(load_voice(&self.dir, name)?)
    }

    /// Normalize text as `lang` from the next utterance on.
    pub fn set_language(&mut self, lang: &str) -> Result<(), String> {
        self.normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Split `text` into chunks the prefill graph takes.
    ///
    /// The chunks every frontend makes, from [`ptts::plan::chunks`], with one addition the graph
    /// forces: it has a fixed number of rows, so a single sentence longer than that is cut by
    /// [`ptts::plan::fit`].
    fn plan(&self, text: &str) -> Result<Vec<Chunk>, String> {
        let max = self.phonon.max_tokens();
        // A bundle exported with a large `--max-tokens` would otherwise let through pieces
        // longer than the model says in one go.
        let fit_max = max.min(ptts::plan::MAX_FIT_TOKENS);
        let planned = ptts::plan::chunks(&self.tokenizer, text, self.normalize, max, FRAME_RATE)
            .and_then(|chunks| {
                ptts::plan::fit_with_word_limit(chunks, fit_max, max, &self.tokenizer, FRAME_RATE)
            })
            .map_err(|e| e.to_string())?;
        let mut chunks = Vec::new();
        for chunk in planned.into_iter().filter(|c| !c.tokens.is_empty()) {
            // Only a single character that is too long is left uncut.
            if chunk.tokens.len() > max {
                return Err(format!(
                    "one character is {} tokens, over the {max} the model takes",
                    chunk.tokens.len()
                ));
            }
            chunks.push(chunk);
        }
        Ok(chunks)
    }

    pub fn speak(
        &mut self,
        text: &str,
        sink: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<PttsResult, String> {
        let start = std::time::Instant::now();
        let chunks = self.plan(text)?;
        let (mut frames, mut samples, mut ttfa) = (0, 0, None);
        let mut pcm_buf = Vec::new();
        for chunk in chunks {
            let (tokens, budget) = (&chunk.tokens, chunk.frame_budget);
            let t = self.phonon.generate(
                tokens,
                chunk.frames_after_eos,
                budget,
                &mut |pcm: &[f32]| {
                    // Core ML has been seen to hand back a stray non-finite sample; silence it
                    // rather than send it to the speaker.
                    pcm_buf.clear();
                    pcm_buf.extend(pcm.iter().map(|v| if v.is_finite() { *v } else { 0.0 }));
                    sink(&pcm_buf)
                },
            )?;
            ttfa.get_or_insert(start.elapsed() - t.total + t.ttfa);
            frames += t.frames;
            samples += t.samples;
            if t.stopped {
                break;
            }
        }
        Ok(PttsResult {
            frames: frames as u32,
            samples,
            ttfa_ms: ttfa.unwrap_or_default().as_secs_f64() * 1e3,
            total_ms: start.elapsed().as_secs_f64() * 1e3,
        })
    }
}
