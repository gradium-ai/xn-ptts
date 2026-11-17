//! Drive a speech-to-speech checkpoint's flow LM and Mimi decoder as Core ML graphs.
//!
//! The caller turns the input audio into the flow LM's per-frame conditions (ptts's
//! `s2s::SourceEncoder`, which has no graph here), and [`Session::step`] runs one flow LM step
//! for each, decoding its latent once the stream is past the checkpoint's delay.
//!
//! The flow LM's cache is the decode graph's slot ring: the voice prompt holds the first slots
//! and the frames cycle through the rest. When those are full, the oldest third of the frames
//! is dropped, as ptts's session drops it: their slots are masked out and taken by the next
//! frames, and the kept keys are rotated back by the dropped count, so positions stay within
//! the window the checkpoint was trained on. Nothing is copied.

use super::driver::{Flow, MimiStream, Ring, Voice, load, rope};
use super::{flow_lm as fl, mimi};
use crate::run::{Compute, Model};
use crate::weights::Weights;
use std::path::Path;
use std::time::{Duration, Instant};

pub struct Config {
    /// The flow LM's sizes, from the checkpoint's config.
    pub dims: fl::Dims,
    /// Slots of the flow LM's ring: the voice prompt's, then the frames kept in view.
    pub ctx: usize,
    /// The Mimi decoder's sizes, from the checkpoint's config.
    pub mimi: mimi::Dims,
    /// Mimi's attention window, the reference's `transformer_context`.
    pub mimi_window: usize,
    /// Frames the output lags the input by, the checkpoint's `audio_delay`.
    pub delay: usize,
    /// Sampling temperature: noise is drawn from `N(0, sqrt(temperature))`.
    pub temperature: f32,
    pub seed: u64,
    /// Where the flow LM runs. Mimi is f32 and always runs on the CPU.
    pub flow_unit: Compute,
}

/// Time a session has spent in each stage, summed over its frames.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimes {
    /// The flow LM, the host's embedding and the window's upkeep included.
    pub flow: Duration,
    /// The Mimi decoder.
    pub decode: Duration,
}

pub struct SpeechToSpeech {
    decode: Flow,
    mimi: Model,
    ring: Ring,
    /// The ring just after the voice prompt, restored at the start of every session.
    voice_ring: Ring,
    voice_len: usize,
    /// `input_linear` weight, applied on the host so the graph always takes an embedding.
    il: Vec<f32>,
    bos: Vec<f32>,
    cfg: Config,
}

impl SpeechToSpeech {
    /// Load the graphs and host tensors from an exported bundle directory, and run the voice
    /// prompt.
    pub fn load(dir: &Path, cfg: Config, voice: Voice) -> Result<Self, String> {
        let dm = cfg.dims;
        if voice.len + 3 > cfg.ctx {
            return Err(format!(
                "a {}-position voice leaves no room in ctx {}",
                voice.len, cfg.ctx
            ));
        }
        if voice.emb.len() != voice.len * dm.d {
            let n = voice.emb.len();
            return Err(format!(
                "the voice has {n} values, not {} positions of {}",
                voice.len, dm.d
            ));
        }
        let wt = Weights::open(&dir.join("host.safetensors"))?;
        let mut ring = Ring::new(&dm, cfg.ctx)?;
        let decode = load(dir, &fl::package_name(cfg.ctx), cfg.flow_unit, None)?;
        let decode = Flow::new(decode, dm, 1, cfg.ctx, &ring)?;
        let mimi = load(dir, &cfg.mimi.package_name(cfg.mimi_window), Compute::CpuOnly, None)?;

        // The voice prompt, one position at a time, as `driver::Phonon` conditions.
        let zero = vec![0f32; dm.ldim];
        for i in 0..voice.len {
            decode.step(&mut ring, i, &voice.emb[i * dm.d..(i + 1) * dm.d], &zero, &[true])?;
        }
        let mut voice_ring = Ring::new(&dm, cfg.ctx)?;
        voice_ring.restore(&ring);
        // Core ML's first prediction on a model is slow, so pay Mimi's here.
        MimiStream::new(&mimi, &cfg.mimi, cfg.mimi_window)?.decode(0, &zero)?;
        Ok(Self {
            decode,
            mimi,
            ring,
            voice_ring,
            voice_len: voice.len,
            il: wt.data("flow_lm.input_linear.weight")?.to_vec(),
            bos: wt.data("flow_lm.bos_emb")?.to_vec(),
            cfg,
        })
    }

    /// Start a stream in the loaded voice.
    pub fn session(&mut self) -> Result<Session<'_>, String> {
        use rand::SeedableRng;
        self.ring.restore(&self.voice_ring);
        let distr = rand_distr::Normal::new(0f32, self.cfg.temperature.max(0.0).sqrt())
            .map_err(|e| format!("noise distribution: {e}"))?;
        Ok(Session {
            flow: &self.decode,
            ring: &mut self.ring,
            mimi: MimiStream::new(&self.mimi, &self.cfg.mimi, self.cfg.mimi_window)?,
            il: &self.il,
            bos: &self.bos,
            cfg: &self.cfg,
            voice_len: self.voice_len,
            rng: rand::rngs::StdRng::seed_from_u64(self.cfg.seed),
            distr,
            steps: 0,
            in_view: 0,
            decoded: 0,
            previous: None,
            times: StageTimes::default(),
        })
    }
}

/// One stream through a [`SpeechToSpeech`], see the module docs.
pub struct Session<'a> {
    flow: &'a Flow,
    ring: &'a mut Ring,
    mimi: MimiStream<'a>,
    il: &'a [f32],
    bos: &'a [f32],
    cfg: &'a Config,
    voice_len: usize,
    rng: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
    /// Flow LM steps run, one per frame.
    steps: usize,
    /// Frames in the ring, after the voice prompt.
    in_view: usize,
    /// Frames handed to Mimi.
    decoded: usize,
    /// The last step's latent, the next one's input.
    previous: Option<Vec<f32>>,
    times: StageTimes,
}

impl Session<'_> {
    /// One frame: its condition, `d_model` wide, in; its output PCM out, once the stream is
    /// `delay` frames in. After the input ends, step `delay` more times with the last condition
    /// to get the rest.
    pub fn step(&mut self, condition: &[f32]) -> Result<Option<Vec<f32>>, String> {
        use rand::Rng;
        let (dm, cfg) = (self.cfg.dims, self.cfg);
        if condition.len() != dm.d {
            return Err(format!("a condition is {} wide, not {}", condition.len(), dm.d));
        }
        let start = Instant::now();
        let (v, context) = (self.voice_len, cfg.ctx - self.voice_len);
        if self.in_view == context {
            let drop = context / 3;
            let oldest = self.steps - self.in_view;
            for j in 0..drop {
                self.ring.valid[v + (oldest + j) % context] = false;
            }
            rotate_back(&dm, self.ring, v, drop);
            self.in_view -= drop;
        }
        // The first `delay + 1` steps start from BOS, as ptts's session does.
        let input = match &self.previous {
            Some(latent) if self.steps > cfg.delay => latent.as_slice(),
            _ => self.bos,
        };
        let mut emb = condition.to_vec();
        for (o, e) in emb.iter_mut().enumerate() {
            let row = &self.il[o * dm.ldim..(o + 1) * dm.ldim];
            *e += row.iter().zip(input).map(|(a, b)| a * b).sum::<f32>();
        }
        let noise: Vec<f32> = (0..dm.ldim).map(|_| self.rng.sample(self.distr)).collect();
        let slot = v + self.steps % context;
        let (latent, _eos) =
            self.flow.step_at(self.ring, v + self.in_view, slot, &emb, &noise, &[true])?;
        self.steps += 1;
        self.in_view += 1;
        self.times.flow += start.elapsed();

        let pcm = if self.steps > cfg.delay {
            let start = Instant::now();
            let pcm = self.mimi.decode(self.decoded, &latent)?;
            self.decoded += 1;
            self.times.decode += start.elapsed();
            Some(pcm)
        } else {
            None
        };
        self.previous = Some(latent);
        Ok(pcm)
    }

    /// Time spent in each stage so far.
    pub fn times(&self) -> StageTimes {
        self.times
    }
}

/// Rotate every frame's key in the ring, the voice prompt's first `prefix` slots aside, back by
/// `count` positions: the graph's rotation at position `count`, by the negated angle.
fn rotate_back(dm: &fl::Dims, ring: &Ring, prefix: usize, count: usize) {
    let (hd, ctx) = (dm.hd(), ring.valid.len());
    let half = hd / 2;
    let (cos, sin) = rope(count, 1, hd);
    let slots: Vec<usize> = (prefix..ctx).filter(|&s| ring.valid[s]).collect();
    ring.k.with_bytes_mut(|p, n| {
        assert!(n >= dm.layers * dm.heads * ctx * hd * 2, "ring buffer is smaller than its shape");
        let k = p as *mut half::f16;
        for lh in 0..dm.layers * dm.heads {
            for &s in &slots {
                let base = (lh * ctx + s) * hd;
                for i in 0..half {
                    unsafe {
                        let (lo, hi) = (k.add(base + i), k.add(base + half + i));
                        let (x, y) = ((*lo).to_f32(), (*hi).to_f32());
                        *lo = half::f16::from_f32(x * cos[i] + y * sin[i]);
                        *hi = half::f16::from_f32(y * cos[i] - x * sin[i]);
                    }
                }
            }
        }
    });
}
