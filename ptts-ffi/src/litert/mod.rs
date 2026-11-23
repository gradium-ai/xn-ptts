//! The LiteRT engine: a bundle `ptts-litert export` wrote, run through LiteRT on the CPU, the
//! GPU or an NPU. Built with the `litert` feature, and used for any `dir` holding a
//! `model.tflite`.
//!
//! The bundle's model has five signatures (`ptts-litert/README.md`): `prefill`, `flow_step` and
//! `mimi_step` are the model, `prefill_tables` and `step_tables` turn positions into the RoPE
//! tables and masks those take. Everything else is done here, as `ptts-litert`'s `runner.py`
//! does it: the text front end and tokenizer are `ptts`'s, and the host keeps the flow LM's KV
//! cache, writing the rows each call returns into it, and Mimi's streaming state.
//!
//! For an NPU, `ptts-litert compile` writes one model per SoC into `npu/`. On Android the engine
//! picks the one for the SoC it runs on; with none, it hands `model.tflite` to LiteRT, which
//! compiles it on the device when the app ships the vendor's compiler plugin and runs it on the
//! CPU otherwise.

mod api;

use crate::{PTTS_UNIT_CPU, PTTS_UNIT_GPU, PTTS_UNIT_NPU, PttsResult};
use api::{Buffer, Compiled, Env};
use ptts::plan::Chunk;
use ptts::preprocess::Normalize;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

/// Sizes the host needs, from `bundle.json`.
struct Meta {
    prefill_len: usize,
    max_frames: usize,
    eos_threshold: f32,
    temperature: f32,
    frame_rate: f64,
    d: usize,
    ldim: usize,
}

/// What the host applies itself, from `host.safetensors`.
struct Host {
    /// `[vocab, d]`.
    text_emb: Vec<f32>,
    /// `input_linear`, `[d, ldim]`.
    input_linear: Vec<f32>,
    bos: Vec<f32>,
    /// The summed conditions the bundle was exported with, added to every frame's input.
    conditions: Vec<f32>,
}

/// Signature indices in the compiled model.
struct Sigs {
    prefill: usize,
    flow_step: usize,
    mimi_step: usize,
    prefill_tables: usize,
    step_tables: usize,
}

pub struct Engine {
    model: Compiled,
    sigs: Sigs,
    /// The flow LM's KV cache, `[2L, H, ctx, D]`: `flow_step`'s `kv` input, which `prefill` is
    /// bound to as well, so the host writes rows into it in place.
    kv: Rc<Buffer>,
    /// `[2L * H, ctx, D]` of `kv`.
    kv_dims: (usize, usize, usize),
    /// The cache just after the voice, restored before every chunk of text.
    voice_kv: Vec<f32>,
    voice_len: usize,
    /// Mimi's streaming state: each `mimi_step` input and the output that replaces it.
    mimi_states: Vec<(String, String)>,
    meta: Meta,
    host: Host,
    tokenizer: ptts::tok::Tok,
    normalize: Normalize,
    dir: PathBuf,
    voices: Vec<String>,
    languages: Vec<String>,
    temperature: f32,
    seed: u64,
}

/// Whether `dir` is a LiteRT bundle rather than what the platform's own engine takes.
pub fn is_bundle(dir: &Path) -> bool {
    dir.join("model.tflite").is_file()
}

fn read_f32(
    st: &safetensors::SafeTensors,
    name: &str,
    file: &str,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let t = st.tensor(name).map_err(|_| format!("{file} has no {name}"))?;
    if t.dtype() != safetensors::Dtype::F32 {
        return Err(format!("{file}: {name} is {:?}, not F32", t.dtype()));
    }
    let data = t.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    Ok((data, t.shape().to_vec()))
}

fn open_safetensors(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// A voice as the export writes it: `emb`, `[1, T, d]`.
fn load_voice(dir: &Path, name: &str, d: usize) -> Result<(Vec<f32>, usize), String> {
    let file = format!("voices/{name}.safetensors");
    let bytes = open_safetensors(&dir.join(&file))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| format!("{file}: {e}"))?;
    let (emb, shape) = read_f32(&st, "emb", &file)?;
    if shape.last() != Some(&d) {
        return Err(format!("{file} is {shape:?}, the model is {d} wide"));
    }
    Ok((emb, shape[shape.len() - 2]))
}

/// The model file for `unit`: for an NPU, the one compiled for this SoC when the bundle has it.
fn model_file(dir: &Path, meta: &serde_json::Value, unit: u32) -> PathBuf {
    if unit == PTTS_UNIT_NPU {
        if let Some(name) = std::env::var_os("PTTS_LITERT_MODEL") {
            return dir.join(name);
        }
        let compiled = meta["npu"].as_object();
        let soc = soc_name();
        let found = compiled.zip(soc).and_then(|(compiled, soc)| {
            compiled
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(&soc))
                .and_then(|(_, v)| v.as_str())
        });
        if let Some(file) = found {
            return dir.join(file);
        }
    }
    dir.join("model.tflite")
}

/// This device's SoC as `ptts-litert compile` names its models, `Qualcomm_SM8750` or
/// `MediaTek_MT6991`. Android only: elsewhere `PTTS_LITERT_MODEL` names the file.
#[cfg(target_os = "android")]
fn soc_name() -> Option<String> {
    fn prop(name: &std::ffi::CStr) -> Option<String> {
        let mut buf = [0 as std::ffi::c_char; 92];
        let n = unsafe { libc::__system_property_get(name.as_ptr(), buf.as_mut_ptr()) };
        (n > 0).then(|| {
            unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
        })
    }
    let vendor = match prop(c"ro.soc.manufacturer")?.to_ascii_lowercase().as_str() {
        "qti" | "qualcomm" => "Qualcomm",
        "mediatek" => "MediaTek",
        _ => return None,
    };
    Some(format!("{vendor}_{}", prop(c"ro.soc.model")?))
}

#[cfg(not(target_os = "android"))]
fn soc_name() -> Option<String> {
    None
}

impl Engine {
    pub fn open(dir: &Path, unit: u32, lang: &str) -> Result<Self, String> {
        let accelerators = match unit {
            PTTS_UNIT_CPU => api::ACCEL_CPU,
            PTTS_UNIT_GPU => api::ACCEL_GPU | api::ACCEL_CPU,
            PTTS_UNIT_NPU => api::ACCEL_NPU | api::ACCEL_CPU,
            u => return Err(format!("compute unit {u} is not one LiteRT has: CPU, GPU or NPU")),
        };
        let normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        let json: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("bundle.json")).map_err(|e| format!("bundle.json: {e}"))?,
        )
        .map_err(|e| format!("bundle.json: {e}"))?;
        let num =
            |v: &serde_json::Value, k: &str| v[k].as_f64().ok_or(format!("bundle.json has no {k}"));
        let meta = Meta {
            prefill_len: num(&json, "prefill_len")? as usize,
            max_frames: num(&json, "max_frames")? as usize,
            eos_threshold: num(&json, "eos_threshold")? as f32,
            temperature: num(&json, "temperature")? as f32,
            frame_rate: num(&json, "frame_rate")?,
            d: num(&json["dims"], "d")? as usize,
            ldim: num(&json["dims"], "ldim")? as usize,
        };
        let names = |k: &str| -> Vec<String> {
            json[k]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .unwrap_or_default()
        };
        let (voices, languages) = (names("voices"), names("languages"));

        let bytes = open_safetensors(&dir.join("host.safetensors"))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| format!("host.safetensors: {e}"))?;
        let get = |n: &str| read_f32(&st, n, "host.safetensors").map(|(v, _)| v);
        let host = Host {
            text_emb: get("flow_lm.conditioner.embed.weight")?,
            input_linear: get("flow_lm.input_linear.weight")?,
            bos: get("flow_lm.bos_emb")?,
            conditions: get("flow_lm.conditions")?,
        };
        let (d, ldim) = (meta.d, meta.ldim);
        if host.input_linear.len() != d * ldim
            || host.bos.len() != ldim
            || host.conditions.len() != d
        {
            return Err("host.safetensors does not match the sizes in bundle.json".into());
        }

        let threads = std::env::var("PTTS_LITERT_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get().min(4)));
        let env = Env::new()?;
        let mut model = Compiled::new(&env, &model_file(dir, &json, unit), accelerators, threads)?;
        let sigs = Sigs {
            prefill: model.signature_named("prefill")?,
            flow_step: model.signature_named("flow_step")?,
            mimi_step: model.signature_named("mimi_step")?,
            prefill_tables: model.signature_named("prefill_tables")?,
            step_tables: model.signature_named("step_tables")?,
        };
        let kv = model.signatures[sigs.flow_step].input("kv")?.clone();
        model.signatures[sigs.prefill].bind_input("kv", kv.clone())?;
        let kv_dims = match kv.ty().shape()[..] {
            [g, h, ctx, hd] => (g * h, ctx, hd),
            ref s => return Err(format!("flow_step's kv is {s:?}, not [2L, H, ctx, D]")),
        };
        let mimi = &model.signatures[sigs.mimi_step];
        let mimi_states = mimi
            .input_names()
            .filter(|n| !matches!(*n, "latent" | "cos" | "sin" | "mask"))
            .map(|n| (n.to_string(), format!("{n}_out")))
            .collect::<Vec<_>>();
        for (_, out) in &mimi_states {
            mimi.output(out)?;
        }

        let tokenizer =
            ptts::tok::Tok::open(&dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
        let mut me = Self {
            model,
            sigs,
            kv,
            kv_dims,
            voice_kv: Vec::new(),
            voice_len: 0,
            mimi_states,
            temperature: meta.temperature,
            seed: 0,
            meta,
            host,
            tokenizer,
            normalize,
            dir: dir.to_path_buf(),
            voices,
            languages,
        };
        let first = me.voices.first().ok_or("no voices in the bundle")?.clone();
        me.set_voice(&first)?;
        Ok(me)
    }

    pub fn voices(&self) -> &[String] {
        &self.voices
    }

    pub fn languages(&self) -> &[String] {
        &self.languages
    }

    /// Switch to one of [`Self::voices`], prefilling it now so the next utterance does not.
    pub fn set_voice(&mut self, name: &str) -> Result<(), String> {
        let (emb, len) = load_voice(&self.dir, name, self.meta.d)?;
        let room = self.kv_dims.1.saturating_sub(self.meta.prefill_len + self.meta.max_frames);
        if len > room {
            return Err(format!("voice {name} is {len} positions; the bundle has room for {room}"));
        }
        self.kv.with_f32(|kv| kv.fill(0.0))?;
        self.prefill(0, &emb)?;
        self.kv.read_f32(&mut self.voice_kv)?;
        self.voice_len = len;
        Ok(())
    }

    /// Normalize text as `lang` from the next utterance on.
    pub fn set_language(&mut self, lang: &str) -> Result<(), String> {
        self.normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Sample at `temperature`, already checked, from the next utterance on.
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    /// Seed the sampling from the next utterance on.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    fn run(&self, sig: usize) -> Result<(), String> {
        self.model.run(sig)
    }

    fn sig(&self, i: usize) -> &api::Signature {
        &self.model.signatures[i]
    }

    /// Write `t` rows of keys and values, `[2L, H, t, D]`, into the cache from slot `pos`.
    fn write_kv(&self, rows: &[f32], t: usize, n: usize, pos: usize) -> Result<(), String> {
        let (g, ctx, hd) = self.kv_dims;
        self.kv.with_f32(|kv| {
            for gi in 0..g {
                let src = &rows[gi * t * hd..][..n * hd];
                kv[(gi * ctx + pos) * hd..][..n * hd].copy_from_slice(src);
            }
        })
    }

    /// Push `rows` (`[n, d]`) through `prefill` from slot `pos`, `prefill_len` rows a call,
    /// padded; only the real rows' keys and values go into the cache.
    fn prefill(&self, mut pos: usize, rows: &[f32]) -> Result<usize, String> {
        let (p, d) = (self.meta.prefill_len, self.meta.d);
        let (tables, prefill) = (self.sig(self.sigs.prefill_tables), self.sig(self.sigs.prefill));
        let mut x = vec![0f32; p * d];
        let mut kv_new = Vec::new();
        for chunk in rows.chunks(p * d) {
            let n = chunk.len() / d;
            x[..chunk.len()].copy_from_slice(chunk);
            x[chunk.len()..].fill(0.0);
            tables.input("pos")?.write_i32(&[pos as i32])?;
            self.run(self.sigs.prefill_tables)?;
            for name in ["cos", "sin", "mask"] {
                prefill.input(name)?.copy_from(tables.output(name)?)?;
            }
            prefill.input("x")?.write_f32(&x)?;
            self.run(self.sigs.prefill)?;
            prefill.output("kv_new")?.read_f32(&mut kv_new)?;
            self.write_kv(&kv_new, p, n, pos)?;
            pos += n;
        }
        Ok(pos)
    }

    /// Split `text` into chunks the prefill takes, as the Core ML engine does: the chunks every
    /// frontend makes, with any sentence longer than the prefill's rows cut by `plan::fit`.
    fn plan(&self, text: &str) -> Result<Vec<Chunk>, String> {
        let max = self.meta.prefill_len;
        let group = max.min(ptts::tts_model::MAX_TOKENS_PER_CHUNK);
        let fit_max = max.min(ptts::plan::MAX_FIT_TOKENS);
        let fr = self.meta.frame_rate;
        let planned = ptts::plan::chunks(&self.tokenizer, text, self.normalize, group, fr)
            .and_then(|chunks| {
                ptts::plan::fit_with_word_limit(chunks, fit_max, max, &self.tokenizer, fr)
            })
            .map_err(|e| e.to_string())?;
        let mut chunks = Vec::new();
        for chunk in planned.into_iter().filter(|c| !c.tokens.is_empty()) {
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

    /// Speak one chunk; returns the frames made and whether `sink` asked to stop.
    fn generate(
        &self,
        chunk: &Chunk,
        sink: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<(usize, bool), String> {
        use rand::{Rng, SeedableRng};
        let (d, ldim) = (self.meta.d, self.meta.ldim);
        self.kv.write_f32(&self.voice_kv)?;
        let mut text = Vec::with_capacity(chunk.tokens.len() * d);
        for &tok in &chunk.tokens {
            let row = self.host.text_emb.get(tok as usize * d..(tok as usize + 1) * d);
            text.extend_from_slice(row.ok_or("token outside the vocabulary")?);
        }
        let mut pos = self.prefill(self.voice_len, &text)?;

        let (tables, flow, mimi) = (
            self.sig(self.sigs.step_tables),
            self.sig(self.sigs.flow_step),
            self.sig(self.sigs.mimi_step),
        );
        for (input, _) in &self.mimi_states {
            mimi.input(input)?.with_f32(|s| s.fill(0.0))?;
        }
        // The same generator and distribution as every other frontend, so a seed means the same.
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let normal = rand_distr::Normal::new(0f32, self.temperature.max(0.0).sqrt())
            .map_err(|e| e.to_string())?;
        let (mut latent, mut emb, mut noise) =
            (self.host.bos.clone(), vec![0f32; d], vec![0f32; ldim]);
        let (mut kv_new, mut eos, mut pcm) = (Vec::new(), Vec::new(), Vec::new());
        let max_frames = chunk.frame_budget.min(self.meta.max_frames);
        let mut countdown: Option<usize> = None;
        let mut frames = 0;
        for frame in 0..max_frames {
            // `input_linear(latent) + conditions`, on the host.
            for (o, e) in emb.iter_mut().enumerate() {
                let row = &self.host.input_linear[o * ldim..(o + 1) * ldim];
                *e = self.host.conditions[o]
                    + row.iter().zip(&latent).map(|(a, b)| a * b).sum::<f32>();
            }
            noise.iter_mut().for_each(|n| *n = rng.sample(normal));

            tables.input("pos")?.write_i32(&[pos as i32])?;
            tables.input("frame")?.write_i32(&[frame as i32])?;
            self.run(self.sigs.step_tables)?;
            for name in ["cos", "sin", "mask"] {
                flow.input(name)?.copy_from(tables.output(name)?)?;
                mimi.input(name)?.copy_from(tables.output(&format!("mimi_{name}"))?)?;
            }
            flow.input("emb")?.write_f32(&emb)?;
            flow.input("noise")?.write_f32(&noise)?;
            self.run(self.sigs.flow_step)?;
            flow.output("kv_new")?.read_f32(&mut kv_new)?;
            self.write_kv(&kv_new, 1, 1, pos)?;
            pos += 1;
            flow.output("eos")?.read_f32(&mut eos)?;
            flow.output("latent")?.read_f32(&mut latent)?;

            mimi.input("latent")?.write_f32(&latent)?;
            self.run(self.sigs.mimi_step)?;
            for (input, output) in &self.mimi_states {
                mimi.input(input)?.copy_from(mimi.output(output)?)?;
            }
            mimi.output("audio")?.read_f32(&mut pcm)?;
            frames += 1;
            if !sink(&pcm) {
                return Ok((frames, true));
            }
            if countdown.is_none() && eos.first().is_some_and(|&e| e > self.meta.eos_threshold) {
                countdown = Some(chunk.frames_after_eos);
            }
            if let Some(c) = countdown.as_mut() {
                if *c == 0 {
                    break;
                }
                *c -= 1;
            }
        }
        Ok((frames, false))
    }

    pub fn speak(
        &mut self,
        text: &str,
        sink: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<PttsResult, String> {
        let start = Instant::now();
        let (mut frames, mut samples, mut ttfa) = (0, 0, None);
        for chunk in self.plan(text)? {
            let (n, stopped) = self.generate(&chunk, &mut |pcm: &[f32]| {
                ttfa.get_or_insert_with(|| start.elapsed());
                samples += pcm.len();
                sink(pcm)
            })?;
            frames += n;
            if stopped {
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
