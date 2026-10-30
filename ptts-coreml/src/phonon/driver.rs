//! Drive the three CoreML graphs as one TTS pipeline.
//!
//! Voice conditioning runs once per voice and is kept as a snapshot of the KV cache. Each
//! utterance then restores that snapshot, pushes the text through the batched prefill graph,
//! and decodes frames until the EOS head fires, with Mimi decoding frame N on a worker thread
//! while the flow LM runs frame N+1.

use super::{flow_lm as fl, mimi};
use crate::run::{Buf, Compute, Model, Session};
use crate::weights::Weights;
use std::path::Path;
use std::time::{Duration, Instant};

pub struct Config {
    /// The flow LM's sizes, from the checkpoint's config.
    pub dims: fl::Dims,
    /// Slots of the flow LM's ring KV cache: voice, text and frames must fit.
    pub ctx: usize,
    /// Rows of the batched text prefill graph; a shorter prompt is padded and the padding
    /// masked out of every later step.
    pub prefill_len: usize,
    /// The Mimi decoder's sizes, from the checkpoint's config.
    pub mimi: mimi::Dims,
    /// Mimi's attention window, the reference's `transformer_context`: the positions each query
    /// sees, itself included.
    pub mimi_window: usize,
    pub max_frames: usize,
    pub eos_threshold: f32,
    /// Sampling temperature. The flow sampler wants noise drawn from `N(0, sqrt(temperature))`;
    /// the wrong distribution distorts the voice while leaving it intelligible.
    pub temperature: f32,
    pub seed: u64,
    /// Where the flow LM runs. Mimi is f32 and always runs on the CPU.
    pub flow_unit: Compute,
}

/// How one utterance went.
pub struct Timings {
    /// Frames generated.
    pub frames: usize,
    /// From the start of the text prefill to the first frame's audio being delivered.
    pub ttfa: Duration,
    pub total: Duration,
    pub samples: usize,
    /// Whether `on_frame` asked to stop.
    pub stopped: bool,
}

/// The flow LM's KV cache: stacked `[L, H, ctx, HD]` K and V in IOSurface buffers, in ring
/// order. The graph only reads it and returns each call's rows, which the host writes into slot
/// `written % ctx`. `valid` says which slots hold a real row: padding from the batched prefill
/// occupies slots but must never be attended to.
struct Ring {
    k: Buf,
    v: Buf,
    written: usize,
    valid: Vec<bool>,
}

impl Ring {
    fn new(dm: &fl::Dims, ctx: usize) -> Result<Self, String> {
        let shape = [dm.layers, dm.heads, ctx, dm.hd()];
        Ok(Ring { k: Buf::new(&shape)?, v: Buf::new(&shape)?, written: 0, valid: vec![false; ctx] })
    }

    fn restore(&mut self, from: &Ring) {
        self.k.copy_from(&from.k);
        self.v.copy_from(&from.v);
        self.written = from.written;
        self.valid.clone_from(&from.valid);
    }
}

/// A flow LM graph for one row count, with its session bound to the live ring.
struct Flow {
    model: Model,
    dm: fl::Dims,
    t: usize,
    session: Session,
}

impl Flow {
    fn new(model: Model, dm: fl::Dims, t: usize, ctx: usize, ring: &Ring) -> Result<Self, String> {
        let small: [(&str, &[usize]); 5] = [
            ("emb", &[1, t, dm.d]),
            ("cos", &[1, 1, t, dm.hd() / 2]),
            ("sin", &[1, 1, t, dm.hd() / 2]),
            ("mask", &[1, 1, t, ctx + t]),
            ("noise", &[1, dm.ldim]),
        ];
        let session = model.session(&small, &[("k_all_in", &ring.k), ("v_all_in", &ring.v)])?;
        Ok(Flow { model, dm, t, session })
    }

    /// One call of `t` rows at rope position `pos`, whose real rows are marked in `rows_valid`.
    /// Writes the rows into the ring and returns `(next_latent, eos)`.
    fn step(
        &self,
        ring: &mut Ring,
        pos: usize,
        emb: &[f32],
        noise: &[f32],
        rows_valid: &[bool],
    ) -> Result<(Vec<f32>, f32), String> {
        let (t, ctx) = (self.t, ring.valid.len());
        if ring.written + t > ctx {
            return Err(format!("sequence of {} does not fit ctx {ctx}", ring.written + t));
        }
        let (c, s) = rope(pos, t, self.dm.hd());
        let mask = ring_mask(&ring.valid, t, ctx);
        let mut out = self.model.predict(
            &self.session,
            &[("emb", emb), ("cos", &c), ("sin", &s), ("mask", &mask), ("noise", noise)],
        )?;
        let k = out.remove("k_new").ok_or("flow LM produced no k_new")?;
        let v = out.remove("v_new").ok_or("flow LM produced no v_new")?;
        ring_write(&self.dm, &ring.k, &k, ring.written, t, ctx);
        ring_write(&self.dm, &ring.v, &v, ring.written, t, ctx);
        for (r, ok) in rows_valid.iter().enumerate() {
            ring.valid[(ring.written + r) % ctx] = *ok;
        }
        ring.written += t;
        let next = out.remove("next_latent").ok_or("flow LM produced no latent")?;
        let eos = out.get("eos").and_then(|v| v.first().copied()).unwrap_or(0.0);
        Ok((next, eos))
    }
}

pub struct Phonon {
    decode: Flow,
    prefill: Flow,
    mimi: Model,
    ring: Ring,
    /// The ring just after voice conditioning, restored at the start of every utterance.
    voice_ring: Option<Ring>,
    /// `input_linear` weight, applied on the host so the graph always takes an embedding.
    il: Vec<f32>,
    /// What the flow LM adds to every frame's input: the checkpoint's conditioners summed, as
    /// the bundle was exported with them.
    conditions: Vec<f32>,
    bos: Vec<f32>,
    text_emb: Vec<f32>,
    voice: Voice,
    cfg: Config,
}

/// A speaker, as the exporter writes it into the bundle.
pub struct Voice {
    /// The voice prompt, `len` positions of `D`.
    pub emb: Vec<f32>,
    pub len: usize,
    /// What the flow LM adds to every frame's input for this voice, `D` wide, when the voice
    /// carries its own in place of the bundle's: those baked into the checkpoint do.
    pub conditions: Option<Vec<f32>>,
}

/// rope cos/sin for `n` positions from `start`, half width: the graph rotates the first `HD/2`
/// channels against the second half.
fn rope(start: usize, n: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let (mut c, mut s) = (vec![0f32; n * half], vec![0f32; n * half]);
    for t in 0..n {
        let pos = (start + t) as f32;
        for i in 0..half {
            let a = pos / 10000f32.powf(i as f32 / half as f32);
            c[t * half + i] = a.cos();
            s[t * half + i] = a.sin();
        }
    }
    (c, s)
}

/// Mimi rotates adjacent channel pairs rather than halves, so it needs one entry per channel
/// with each angle repeated across its pair.
fn rope_mimi(dm: &mimi::Dims, step: usize) -> (Vec<f32>, Vec<f32>) {
    let (hd, n, start) = (dm.hd(), dm.steps, step * dm.steps);
    let half = hd / 2;
    let (mut c, mut s) = (vec![0f32; n * hd], vec![0f32; n * hd]);
    for t in 0..n {
        let pos = (start + t) as f32;
        for i in 0..half {
            let a = pos / 10000f32.powf(i as f32 / half as f32);
            let (co, si) = (a.cos(), a.sin());
            c[t * hd + 2 * i] = co;
            c[t * hd + 2 * i + 1] = co;
            s[t * hd + 2 * i] = si;
            s[t * hd + 2 * i + 1] = si;
        }
    }
    (c, s)
}

/// Mask for `t` new rows against the ring, `[t, ctx + t]`: the ring's slots, valid or not, then
/// this call's rows, which attend causally among themselves.
fn ring_mask(valid: &[bool], t: usize, ctx: usize) -> Vec<f32> {
    let w = ctx + t;
    let mut m = vec![-1e4f32; t * w];
    for q in 0..t {
        for (j, ok) in valid.iter().enumerate() {
            if *ok {
                m[q * w + j] = 0.0;
            }
        }
        for k in 0..=q {
            m[q * w + ctx + k] = 0.0;
        }
    }
    m
}

/// Mimi's sliding window, as the reference applies it: the query at absolute position `p` sees
/// positions `p - window + 1 ..= p` and nothing older. The cache holds `window` past positions
/// and this call's `steps` rows, so slot `j` holds position `start - window + j`, and is empty
/// while that is negative.
fn mimi_mask(dm: &mimi::Dims, step: usize, window: usize) -> Vec<f32> {
    let (t, n) = (dm.steps, dm.cache_len(window));
    let start = step * t;
    let mut m = vec![-1e4f32; t * n];
    for q in 0..t {
        for j in q + 1..=window + q {
            if start + j >= window {
                m[q * n + j] = 0.0;
            }
        }
    }
    m
}

/// Write one call's rows, `[L, H, t, HD]` as f32, into the ring's slots as fp16.
fn ring_write(dm: &fl::Dims, buf: &Buf, rows: &[f32], written: usize, t: usize, ctx: usize) {
    let (layers, heads, hd) = (dm.layers, dm.heads, dm.hd());
    buf.with_bytes_mut(|p, n| {
        assert!(n >= layers * heads * ctx * hd * 2, "ring buffer is smaller than its shape");
        let dst = p as *mut half::f16;
        for i in 0..layers {
            for h in 0..heads {
                for r in 0..t {
                    let slot = (written + r) % ctx;
                    let src = &rows[((i * heads + h) * t + r) * hd..][..hd];
                    let base = ((i * heads + h) * ctx + slot) * hd;
                    for (k, v) in src.iter().enumerate() {
                        unsafe { *dst.add(base + k) = half::f16::from_f32(*v) };
                    }
                }
            }
        }
    });
}

fn load(dir: &Path, name: &str, unit: Compute) -> Result<Model, String> {
    let package = dir.join(format!("{name}.mlpackage"));
    Model::load(&package, &package.with_extension("mlmodelc"), unit)
}

/// Mimi for one utterance: its session, and the streaming state it carries between frames.
struct MimiStream<'a> {
    model: &'a Model,
    session: Session,
    spec: Vec<(String, Vec<usize>)>,
    state: Vec<Vec<f32>>,
    dims: mimi::Dims,
    window: usize,
}

impl<'a> MimiStream<'a> {
    fn new(model: &'a Model, dims: &mimi::Dims, window: usize) -> Result<Self, String> {
        let n = dims.cache_len(window);
        let spec = dims.state_spec(n);
        let latent_shape = [1, 1, dims.ldim];
        let rope_shape = [1, 1, dims.steps, dims.hd()];
        let mut small: Vec<(&str, &[usize])> =
            vec![("latent", &latent_shape), ("cos", &rope_shape), ("sin", &rope_shape)];
        let mask_shape = [1, 1, dims.steps, n];
        small.push(("mask", &mask_shape));
        small.extend(spec.iter().map(|(name, sh)| (name.as_str(), sh.as_slice())));
        let session = model.session(&small, &[])?;
        let state = spec.iter().map(|(_, sh)| vec![0f32; sh.iter().product()]).collect();
        Ok(MimiStream { model, session, spec, state, dims: dims.clone(), window })
    }

    /// Frame `i`: latent in, one frame of PCM out, streaming state advanced.
    fn decode(&mut self, i: usize, latent: &[f32]) -> Result<Vec<f32>, String> {
        let (rc, rs) = rope_mimi(&self.dims, i);
        let mask = mimi_mask(&self.dims, i, self.window);
        let mut ins: Vec<(&str, &[f32])> =
            vec![("latent", latent), ("cos", &rc), ("sin", &rs), ("mask", &mask)];
        ins.extend(self.spec.iter().zip(&self.state).map(|((n, _), s)| (n.as_str(), s.as_slice())));
        let mut out = self.model.predict(&self.session, &ins)?;
        for ((n, _), s) in self.spec.iter().zip(self.state.iter_mut()) {
            *s = out.remove(&format!("{n}_out")).ok_or_else(|| format!("mimi lost {n}"))?;
        }
        out.remove("pcm").ok_or_else(|| "mimi produced no pcm".into())
    }
}

impl Phonon {
    /// Load the graphs and host tensors from an exported bundle directory, speaking with
    /// `voice`.
    pub fn load(dir: &Path, cfg: Config, voice: Voice) -> Result<Self, String> {
        let wt = Weights::open(&dir.join("host.safetensors"))?;
        let dm = cfg.dims;
        let ring = Ring::new(&dm, cfg.ctx)?;
        let decode = load(dir, &fl::package_name(cfg.ctx, 1), cfg.flow_unit)?;
        let decode = Flow::new(decode, dm, 1, cfg.ctx, &ring)?;
        let prefill = load(dir, &fl::package_name(cfg.ctx, cfg.prefill_len), cfg.flow_unit)?;
        let prefill = Flow::new(prefill, dm, cfg.prefill_len, cfg.ctx, &ring)?;
        let mimi = load(dir, &cfg.mimi.package_name(cfg.mimi_window), Compute::CpuOnly)?;

        // Bundles exported before the conditioners came from the config call it `num_speakers`,
        // the one conditioner they knew.
        let conditions = wt
            .data("flow_lm.conditions")
            .or_else(|_| wt.data("flow_lm.num_speakers"))
            .map_err(|_| "host.safetensors has no `flow_lm.conditions`".to_string())?
            .to_vec();
        if conditions.len() != dm.d {
            let n = conditions.len();
            return Err(format!("the bundle's conditions are {n} wide, the model is {}", dm.d));
        }

        let mut me = Self {
            decode,
            prefill,
            mimi,
            ring,
            voice_ring: None,
            il: wt.data("flow_lm.input_linear.weight")?.to_vec(),
            conditions,
            bos: wt.data("flow_lm.bos_emb")?.to_vec(),
            text_emb: wt.data("flow_lm.conditioner.embed.weight")?.to_vec(),
            voice: Voice { emb: Vec::new(), len: 0, conditions: None },
            cfg,
        };
        me.set_voice(voice)?;
        // CoreML's first prediction on a model is slow (on the CPU, ~160 ms for these two), so
        // pay it here rather than in the first utterance. Conditioning has already warmed the
        // decode graph; the ring is restored from the voice snapshot before every utterance, so
        // what this writes into it is never read.
        let zero = vec![0f32; dm.ldim];
        let (plen, vlen) = (me.cfg.prefill_len, me.voice.len);
        me.prefill.step(&mut me.ring, vlen, &vec![0f32; plen * dm.d], &zero, &vec![false; plen])?;
        MimiStream::new(&me.mimi, &me.cfg.mimi, me.cfg.mimi_window)?.decode(0, &zero)?;
        Ok(me)
    }

    /// Sample at `temperature` from the next utterance on.
    pub fn set_temperature(&mut self, temperature: f32) {
        self.cfg.temperature = temperature;
    }

    /// Seed the sampling from the next utterance on.
    pub fn set_seed(&mut self, seed: u64) {
        self.cfg.seed = seed;
    }

    /// Swap the speaker, conditioning on it now (up to about 0.6 s on a phone) so the next
    /// utterance does not pay for it.
    pub fn set_voice(&mut self, voice: Voice) -> Result<(), String> {
        let budget = self.cfg.ctx.saturating_sub(self.cfg.prefill_len + self.cfg.max_frames);
        if voice.len > budget {
            return Err(format!("a {}-position voice does not fit: room for {budget}", voice.len));
        }
        let d = self.cfg.dims.d;
        if voice.emb.len() != voice.len * d {
            let n = voice.emb.len();
            return Err(format!("the voice has {n} values, not {} positions of {d}", voice.len));
        }
        if let Some(c) = voice.conditions.as_ref().filter(|c| c.len() != d) {
            return Err(format!("the voice's conditions are {} wide, the model is {d}", c.len()));
        }
        self.voice = voice;
        self.voice_ring = None;
        self.condition()
    }

    /// Longest prompt the prefill graph takes, in tokens.
    pub fn max_tokens(&self) -> usize {
        self.cfg.prefill_len
    }

    /// Condition on the voice and keep a snapshot of the result.
    fn condition(&mut self) -> Result<(), String> {
        let ring = &mut self.ring;
        ring.k.with_bytes_mut(|p, n| unsafe { std::ptr::write_bytes(p, 0, n) });
        ring.v.with_bytes_mut(|p, n| unsafe { std::ptr::write_bytes(p, 0, n) });
        ring.written = 0;
        ring.valid.fill(false);
        let (d, zero) = (self.cfg.dims.d, vec![0f32; self.cfg.dims.ldim]);
        // One position at a time. The prefill graph would take this from ~0.6 s to a few tens
        // of ms on the phone, but batching changes the attention's summation order and with it
        // every later frame, so conditioning stays exact and runs when the voice is set instead.
        for i in 0..self.voice.len {
            let emb = &self.voice.emb[i * d..(i + 1) * d];
            self.decode.step(ring, i, emb, &zero, &[true])?;
        }
        let mut snap = Ring::new(&self.cfg.dims, self.cfg.ctx)?;
        snap.restore(ring);
        self.voice_ring = Some(snap);
        Ok(())
    }

    /// Speak `tokens`, handing each frame's 24 kHz PCM to `on_frame` as soon as it is decoded;
    /// `on_frame` returns false to stop there.
    /// Stops `frames_after_eos` frames after the EOS head fires, or after `max_frames` (capped
    /// at the bundle's budget).
    pub fn generate(
        &mut self,
        tokens: &[u32],
        frames_after_eos: usize,
        max_frames: usize,
        on_frame: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<Timings, String> {
        let plen = self.cfg.prefill_len;
        if tokens.len() > plen {
            return Err(format!("{} tokens exceed the prefill graph's {plen} rows", tokens.len()));
        }
        self.ring.restore(self.voice_ring.as_ref().ok_or("no voice: set_voice failed")?);

        let t_start = Instant::now();
        // The text in one call, padded to the graph's rows; the padding is masked from now on.
        let d = self.cfg.dims.d;
        let mut temb = vec![0f32; plen * d];
        for (r, &tok) in tokens.iter().enumerate() {
            let o = tok as usize * d;
            let row = self.text_emb.get(o..o + d).ok_or("token outside the vocabulary")?;
            temb[r * d..(r + 1) * d].copy_from_slice(row);
        }
        let rows_valid: Vec<bool> = (0..plen).map(|r| r < tokens.len()).collect();
        let zero = vec![0f32; self.cfg.dims.ldim];
        self.prefill.step(&mut self.ring, self.voice.len, &temb, &zero, &rows_valid)?;

        // Same generator and distribution as the reference implementation.
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.cfg.seed);
        let distr = rand_distr::Normal::new(0f32, self.cfg.temperature.max(0.0).sqrt())
            .map_err(|e| format!("noise distribution: {e}"))?;

        let mut stream = MimiStream::new(&self.mimi, &self.cfg.mimi, self.cfg.mimi_window)?;
        let (decode, ring) = (&self.decode, &mut self.ring);
        let conditions = self.voice.conditions.as_ref().unwrap_or(&self.conditions);
        let (il, cfg) = (&self.il, &self.cfg);
        let pos0 = self.voice.len + tokens.len();
        let mut lat = self.bos.clone();
        let (mut frames, mut samples, mut ttfa) = (0usize, 0usize, None);
        let stopped = std::cell::Cell::new(false);

        std::thread::scope(|scope| -> Result<(), String> {
            let (lat_tx, lat_rx) = std::sync::mpsc::channel::<(usize, Vec<f32>)>();
            let (pcm_tx, pcm_rx) = std::sync::mpsc::channel::<Result<Vec<f32>, String>>();
            // Mimi owns its streaming state for the whole utterance, so frames stay in order.
            std::thread::Builder::new()
                .name("mimi".into())
                .spawn_scoped(scope, move || {
                    while let Ok((i, lat)) = lat_rx.recv() {
                        let r = stream.decode(i, &lat);
                        let failed = r.is_err();
                        if pcm_tx.send(r).is_err() || failed {
                            break;
                        }
                    }
                })
                .map_err(|e| e.to_string())?;

            let mut deliver = |pcm: Vec<f32>| {
                if !stopped.get() {
                    ttfa.get_or_insert_with(|| t_start.elapsed());
                    samples += pcm.len();
                    stopped.set(!on_frame(&pcm));
                }
                !stopped.get()
            };
            let mut countdown: Option<usize> = None;
            for i in 0..max_frames.min(cfg.max_frames) {
                let noise: Vec<f32> = (0..cfg.dims.ldim).map(|_| rng.sample(distr)).collect();
                // input_linear + conditions on the host: 32x768, microseconds, and it is what
                // lets one graph serve both prefill and decode.
                let mut emb = conditions.clone();
                for (o, e) in emb.iter_mut().enumerate() {
                    let row = &il[o * cfg.dims.ldim..(o + 1) * cfg.dims.ldim];
                    *e += row.iter().zip(lat.iter()).map(|(a, b)| a * b).sum::<f32>();
                }
                let (next, eos) = decode.step(ring, pos0 + i, &emb, &noise, &[true])?;
                lat_tx.send((i, next.clone())).map_err(|_| "mimi worker died")?;
                // Stay one frame ahead: collect frame i-1 so the queue cannot run away.
                if i > 0 && !deliver(pcm_rx.recv().map_err(|_| "mimi worker died")??) {
                    break;
                }
                frames += 1;
                if eos > cfg.eos_threshold && countdown.is_none() {
                    countdown = Some(frames_after_eos);
                }
                if let Some(c) = countdown.as_mut() {
                    if *c == 0 {
                        break;
                    }
                    *c -= 1;
                }
                lat = next;
            }
            drop(lat_tx);
            // The frame still in flight, dropped if the caller stopped.
            while let Ok(r) = pcm_rx.recv() {
                deliver(r?);
            }
            Ok(())
        })?;

        Ok(Timings {
            frames,
            ttfa: ttfa.unwrap_or_default(),
            total: t_start.elapsed(),
            samples,
            stopped: stopped.get(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mimi_mask_shows_each_query_the_window_ending_at_itself() {
        let dm = mimi::Dims::default();
        for window in [5, 50, 250] {
            let n = dm.cache_len(window);
            for step in 0..window / dm.steps + 3 {
                let start = step * dm.steps;
                let mask = mimi_mask(&dm, step, window);
                for q in 0..dm.steps {
                    let p = start + q;
                    let seen: Vec<usize> = (0..n)
                        .filter(|&j| mask[q * n + j] == 0.0)
                        .map(|j| start + j - window)
                        .collect();
                    let want: Vec<usize> = (p.saturating_sub(window - 1)..=p).collect();
                    assert_eq!(seen, want, "window {window}, step {step}, query {q}");
                }
            }
        }
    }
}
