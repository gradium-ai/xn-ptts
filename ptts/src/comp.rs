//! A small Rust DSP library: an `AudioModule` trait, a `Chain` that runs modules in
//! series, and a few mono `f32` modules — `Gain`, `Compressor`, `Limiter`. Chains
//! can be built in code or loaded from TOML.

mod compressor {
    use serde::Deserialize;

    use super::{AudioModule, time_const_coef};

    #[derive(Debug, Clone, Copy, Default, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Detector {
        /// Track instantaneous absolute value (peak).
        #[default]
        Peak,
        /// Track root-mean-square level. Attack/decay coefficients smooth
        /// the squared signal, so they double as the RMS averaging window.
        Rms,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct CompressorConfig {
        /// Threshold in dBFS (e.g. -20.0). Levels above this get attenuated.
        pub threshold_db: f32,
        /// Compression ratio. 1.0 = no compression, 4.0 = 4:1, etc.
        pub ratio: f32,
        /// Attack time in milliseconds.
        pub attack_ms: f32,
        /// Decay (release) time in milliseconds.
        pub decay_ms: f32,
        /// Optional lookahead in milliseconds. Output is delayed by this much,
        /// padded on the left with zeros until the delay line fills.
        #[serde(default)]
        pub lookahead_ms: f32,
        /// Level detector: peak (default) or RMS.
        #[serde(default)]
        pub detector: Detector,
    }

    pub struct Compressor {
        threshold_db: f32,
        ratio: f32,
        attack_coef: f32,
        decay_coef: f32,
        detector: Detector,
        /// In peak mode this holds |x|; in RMS mode it holds the smoothed x².
        envelope: f32,
        delay: Vec<f32>,
        delay_idx: usize,
    }

    impl Compressor {
        pub fn new(
            sample_rate: u32,
            threshold_db: f32,
            ratio: f32,
            attack_ms: f32,
            decay_ms: f32,
            lookahead_ms: f32,
            detector: Detector,
        ) -> Self {
            let sample_rate = sample_rate as f32;
            let attack_coef = time_const_coef(attack_ms, sample_rate);
            let decay_coef = time_const_coef(decay_ms, sample_rate);
            let lookahead_samples = (lookahead_ms * 0.001 * sample_rate).round().max(0.0) as usize;
            Self {
                threshold_db,
                ratio: ratio.max(1.0),
                attack_coef,
                decay_coef,
                detector,
                envelope: 0.0,
                delay: vec![0.0; lookahead_samples],
                delay_idx: 0,
            }
        }

        pub fn from_config(sample_rate: u32, cfg: &CompressorConfig) -> Self {
            Self::new(
                sample_rate,
                cfg.threshold_db,
                cfg.ratio,
                cfg.attack_ms,
                cfg.decay_ms,
                cfg.lookahead_ms,
                cfg.detector,
            )
        }
    }

    impl AudioModule for Compressor {
        fn process(&mut self, buffer: &mut [f32]) {
            let slope = 1.0 / self.ratio - 1.0;
            for sample in buffer.iter_mut() {
                let x = *sample;

                // Detector feeds the envelope follower with either |x| (peak)
                // or x² (RMS). In RMS mode the stored envelope is mean-square;
                // we sqrt it before converting to dB.
                let detect = match self.detector {
                    Detector::Peak => x.abs(),
                    Detector::Rms => x * x,
                };
                let coef = if detect > self.envelope { self.attack_coef } else { self.decay_coef };
                self.envelope = detect + coef * (self.envelope - detect);

                let level = match self.detector {
                    Detector::Peak => self.envelope,
                    Detector::Rms => self.envelope.max(0.0).sqrt(),
                };
                let env_db = 20.0 * (level + 1e-20).log10();
                let gain_db = if env_db > self.threshold_db {
                    (env_db - self.threshold_db) * slope
                } else {
                    0.0
                };
                let gain = 10f32.powf(gain_db * (1.0 / 20.0));

                // With lookahead, apply the just-computed gain to a sample
                // that is `lookahead_samples` in the past. The buffer starts
                // zeroed, so the first lookahead samples of output are 0.
                let target = if self.delay.is_empty() {
                    x
                } else {
                    let out = self.delay[self.delay_idx];
                    self.delay[self.delay_idx] = x;
                    self.delay_idx += 1;
                    if self.delay_idx == self.delay.len() {
                        self.delay_idx = 0;
                    }
                    out
                };

                *sample = target * gain;
            }
        }

        fn reset(&mut self) {
            self.envelope = 0.0;
            self.delay.fill(0.0);
            self.delay_idx = 0;
        }

        fn latency(&self) -> usize {
            self.delay.len()
        }
    }
}

mod gain {
    use serde::Deserialize;

    use super::AudioModule;

    #[derive(Debug, Clone, Deserialize)]
    pub struct GainConfig {
        /// Gain in dB. Positive values amplify, negative attenuate.
        pub gain_db: f32,
    }

    pub struct Gain {
        gain: f32,
    }

    impl Gain {
        pub fn new(gain_db: f32) -> Self {
            Self { gain: 10f32.powf(gain_db * (1.0 / 20.0)) }
        }

        pub fn from_config(_sample_rate: u32, cfg: &GainConfig) -> Self {
            Self::new(cfg.gain_db)
        }
    }

    impl AudioModule for Gain {
        fn process(&mut self, buffer: &mut [f32]) {
            for sample in buffer.iter_mut() {
                *sample *= self.gain;
            }
        }
    }
}

mod limiter {
    use serde::Deserialize;

    use super::{AudioModule, time_const_coef};

    #[derive(Debug, Clone, Deserialize)]
    pub struct LimiterConfig {
        /// Lookahead in milliseconds. Output is delayed by this much,
        /// padded on the left with zeros until the delay line fills.
        /// Larger lookahead gives a smoother pre-attenuation ramp into
        /// each peak instead of a hard click.
        pub lookahead_ms: f32,
        /// Headroom in dB below 0 dBFS. The ceiling is `-headroom_db`
        /// (e.g. headroom_db = 1.0 => ceiling at -1 dBFS).
        pub headroom_db: f32,
        /// Release time in ms — how fast the gain returns to unity after
        /// a peak has passed. Defaults to 50 ms.
        #[serde(default = "default_release_ms")]
        pub release_ms: f32,
    }

    fn default_release_ms() -> f32 {
        50.0
    }

    pub struct Limiter {
        ceiling: f32,
        release_coef: f32,
        gain: f32,
        delay: Vec<f32>,
        delay_idx: usize,
    }

    impl Limiter {
        pub fn new(sample_rate: u32, lookahead_ms: f32, headroom_db: f32, release_ms: f32) -> Self {
            let sample_rate = sample_rate as f32;
            let ceiling = 10f32.powf(-headroom_db.abs() * (1.0 / 20.0));
            let lookahead_samples = (lookahead_ms * 0.001 * sample_rate).round().max(0.0) as usize;
            Self {
                ceiling,
                release_coef: time_const_coef(release_ms, sample_rate),
                gain: 1.0,
                delay: vec![0.0; lookahead_samples],
                delay_idx: 0,
            }
        }

        pub fn from_config(sample_rate: u32, cfg: &LimiterConfig) -> Self {
            Self::new(sample_rate, cfg.lookahead_ms, cfg.headroom_db, cfg.release_ms)
        }
    }

    impl AudioModule for Limiter {
        fn process(&mut self, buffer: &mut [f32]) {
            for sample in buffer.iter_mut() {
                let x = *sample;

                // Pop the oldest delayed sample, push the new one.
                // After this, the delay buffer holds the lookahead window
                // (input[n-L+1 ..= n]) relative to the sample we just popped.
                let delayed = if self.delay.is_empty() {
                    x
                } else {
                    let d = self.delay[self.delay_idx];
                    self.delay[self.delay_idx] = x;
                    self.delay_idx += 1;
                    if self.delay_idx == self.delay.len() {
                        self.delay_idx = 0;
                    }
                    d
                };

                // Peak magnitude across the lookahead window (plus current sample
                // when there is no delay buffer to fall back on).
                let mut peak = if self.delay.is_empty() { x.abs() } else { 0.0 };
                for &s in &self.delay {
                    let a = s.abs();
                    if a > peak {
                        peak = a;
                    }
                }

                // Gain required so the loudest sample in the window hits exactly
                // the ceiling (or unity if everything is already below).
                let target = if peak > self.ceiling { self.ceiling / peak } else { 1.0 };

                // Snap down instantly when we see a new (lower) target — the
                // lookahead delay is what gives us the smooth pre-attenuation
                // ramp into the peak. Release smoothly back toward unity.
                if target < self.gain {
                    self.gain = target;
                } else {
                    self.gain = target + self.release_coef * (self.gain - target);
                }

                *sample = delayed * self.gain;
            }
        }

        fn reset(&mut self) {
            self.gain = 1.0;
            self.delay.fill(0.0);
            self.delay_idx = 0;
        }

        fn latency(&self) -> usize {
            self.delay.len()
        }
    }
}

use serde::Deserialize;

pub use compressor::{Compressor, CompressorConfig};
pub use gain::{Gain, GainConfig};
pub use limiter::{Limiter, LimiterConfig};

pub(crate) fn time_const_coef(time_ms: f32, sample_rate: f32) -> f32 {
    if time_ms <= 0.0 { 0.0 } else { (-1.0 / (time_ms * 0.001 * sample_rate)).exp() }
}

/// A streaming audio processing module. Operates on `f32` mono samples
/// in place. Implementations may hold internal state across calls, so
/// the same buffer length is not required from call to call.
pub trait AudioModule: Send {
    fn process(&mut self, buffer: &mut [f32]);

    fn reset(&mut self) {}

    /// How many samples the output lags the input by.
    fn latency(&self) -> usize {
        0
    }
}

/// A sequence of `AudioModule`s applied in order.
pub struct Chain {
    modules: Vec<Module>,
    sample_rate: u32,
}

impl Chain {
    pub fn new(sample_rate: u32) -> Self {
        Self { modules: Vec::new(), sample_rate }
    }

    pub fn push<M: Into<Module>>(&mut self, module: M) -> &mut Self {
        self.modules.push(module.into());
        self
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn from_config(cfg: &ChainConfig) -> Self {
        let mut chain = Chain::new(cfg.sample_rate);
        for m in &cfg.modules {
            chain.modules.push(m.build(cfg.sample_rate));
        }
        chain
    }

    pub fn from_modules(sample_rate: u32, modules: &[ModuleConfig]) -> Self {
        Self { modules: modules.iter().map(|m| m.build(sample_rate)).collect(), sample_rate }
    }

    pub fn gain_compressor_limiter(gain_db: f32, sample_rate: u32) -> Self {
        let mut chain = Chain::new(sample_rate);
        chain
            .push(Gain::new(gain_db))
            // .push(Compressor::new(sample_rate, -20.0, 3.0, 5.0, 100.0, 5.0, Default::default()))
            .push(Limiter::new(sample_rate, 3.0, 1.0, 100.0));
        chain
    }
}

impl AudioModule for Chain {
    fn process(&mut self, buffer: &mut [f32]) {
        for m in &mut self.modules {
            match m {
                Module::Compressor(c) => c.process(buffer),
                Module::Limiter(l) => l.process(buffer),
                Module::Gain(g) => g.process(buffer),
            }
        }
    }

    fn reset(&mut self) {
        for m in &mut self.modules {
            match m {
                Module::Compressor(c) => c.reset(),
                Module::Limiter(l) => l.reset(),
                Module::Gain(g) => g.reset(),
            }
        }
    }

    fn latency(&self) -> usize {
        self.modules
            .iter()
            .map(|m| match m {
                Module::Compressor(c) => c.latency(),
                Module::Limiter(l) => l.latency(),
                Module::Gain(g) => g.latency(),
            })
            .sum()
    }
}

#[derive(Debug, Deserialize)]
pub struct ChainConfig {
    pub sample_rate: u32,
    #[serde(default)]
    pub modules: Vec<ModuleConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModuleConfig {
    Compressor(CompressorConfig),
    Limiter(LimiterConfig),
    Gain(GainConfig),
}

pub enum Module {
    Compressor(Compressor),
    Limiter(Limiter),
    Gain(Gain),
}

impl ModuleConfig {
    pub fn build(&self, sample_rate: u32) -> Module {
        match self {
            ModuleConfig::Compressor(cfg) => {
                Module::Compressor(Compressor::from_config(sample_rate, cfg))
            }
            ModuleConfig::Limiter(cfg) => Module::Limiter(Limiter::from_config(sample_rate, cfg)),
            ModuleConfig::Gain(cfg) => Module::Gain(Gain::from_config(sample_rate, cfg)),
        }
    }
}

impl From<Limiter> for Module {
    fn from(limiter: Limiter) -> Self {
        Module::Limiter(limiter)
    }
}

impl From<Compressor> for Module {
    fn from(compressor: Compressor) -> Self {
        Module::Compressor(compressor)
    }
}

impl From<Gain> for Module {
    fn from(gain: Gain) -> Self {
        Module::Gain(gain)
    }
}
