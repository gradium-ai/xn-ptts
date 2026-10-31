//! The flow LM's per-frame step, replayed as a CUDA graph.
//!
//! Launching a step's few hundred small kernels one at a time costs the host more than the GPU
//! takes to run them. A graph records the step once, against buffers of its own, and from then
//! on launches it with one call per frame. [`GraphPool`] keeps the graphs from one generation to
//! the next, so recording happens once rather than per request. Off CUDA none is ever built,
//! and generation runs as before.

use crate::flow_lm::Rng;
use crate::tts_model::{TTSModel, TTSState};
use std::sync::Mutex;
use xn::{BackendQ, Result, Tensor};

#[cfg(feature = "cuda")]
pub use cuda::StepGraph;

/// Most step graphs a pool keeps: one per sequence budget and conditioning in use.
const MAX_STEP_GRAPHS: usize = 4;

/// Recorded steps, taken by a generation and given back when it ends.
pub struct GraphPool<Q: BackendQ> {
    steps: Mutex<Vec<StepGraph<Q>>>,
}

impl<Q: BackendQ> Default for GraphPool<Q> {
    fn default() -> Self {
        Self { steps: Mutex::new(Vec::new()) }
    }
}

impl<Q: BackendQ> GraphPool<Q> {
    /// A step graph for generations from `base`, recorded before if one fits it. `None` where
    /// there are no graphs.
    pub fn step(&self, model: &TTSModel<Q>, base: &TTSState<Q>) -> Result<Option<StepGraph<Q>>> {
        let reused = {
            let mut steps = self.steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            steps.iter().position(|g| g.fits(base)).map(|i| steps.swap_remove(i))
        };
        match reused {
            Some(graph) => Ok(Some(graph)),
            None => StepGraph::new(model, base),
        }
    }

    /// Give back a step graph, once the generation using it has ended without an error.
    pub fn put_step(&self, graph: StepGraph<Q>) {
        let mut steps = self.steps.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if steps.len() >= MAX_STEP_GRAPHS {
            steps.remove(0);
        }
        steps.push(graph);
    }
}

/// Never built without the `cuda` feature.
#[cfg(not(feature = "cuda"))]
pub struct StepGraph<Q: BackendQ>(std::convert::Infallible, std::marker::PhantomData<fn() -> Q>);

#[cfg(not(feature = "cuda"))]
impl<Q: BackendQ> StepGraph<Q> {
    fn new(_: &TTSModel<Q>, _: &TTSState<Q>) -> Result<Option<Self>> {
        Ok(None)
    }

    fn fits(&self, _: &TTSState<Q>) -> bool {
        match self.0 {}
    }

    pub fn begin(&mut self, _: &TTSModel<Q>, _: &TTSState<Q>, _: &[u32]) -> Result<()> {
        match self.0 {}
    }

    #[allow(clippy::type_complexity)]
    pub fn step(
        &mut self,
        _: &TTSModel<Q>,
        _: Option<&Tensor<Q::T, Q::B>>,
        _: &mut impl Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
        match self.0 {}
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::rope::RotaryEmbedding;
    use crate::transformer::{DeviceStep, LayerAttentionState, StreamingTransformerState};
    use xn::WithDTypeF;
    use xn::cuda_backend::{CaptureGraph, Device};

    /// Whether to replay steps as CUDA graphs where they can be. On unless
    /// `PTTS_CUDA_GRAPHS=0`, which is there to compare the two.
    fn enabled() -> bool {
        std::env::var("PTTS_CUDA_GRAPHS").map_or(true, |v| v != "0")
    }

    /// Record `body` once against the buffers it uses: a record pass that runs it, then a
    /// capture that only records it. The record pass is what lets the capture serve the uploads
    /// `body` makes from device copies.
    fn record(dev: &Device, mut body: impl FnMut() -> Result<()>) -> Result<CaptureGraph> {
        dev.capture_record_begin()?;
        let recorded = body();
        dev.capture_record_end()?;
        recorded?;
        dev.capture_begin()?;
        if let Err(e) = body() {
            dev.capture_abort();
            return Err(e);
        }
        dev.capture_end()
    }

    /// The filled length of a flow-LM state, its capacity and its batch size.
    fn positions<Q: BackendQ>(state: &TTSState<Q>) -> Option<(usize, usize, usize)> {
        state.flow_lm_state.transformer_state.layer_states.iter().find_map(|l| match l {
            LayerAttentionState::FlowLm(mha) => {
                let (b, kv) = (mha.k_cache.dim(0usize).ok()?, mha.k_cache.dim(1usize).ok()?);
                Some((mha.current_end, kv, b))
            }
            LayerAttentionState::Mimi(_) => None,
        })
    }

    /// The flow LM's recorded step, with a state of its own that every generation it serves
    /// loads its primed prefix into, and the buffers the step reads and writes.
    pub struct StepGraph<Q: BackendQ> {
        dev: Device,
        state: TTSState<Q>,
        /// `[1]`, the position the step writes.
        pos: Tensor<i64, Q::B>,
        latent_in: Tensor<Q::T, Q::B>,
        noise: Tensor<Q::T, Q::B>,
        latent_out: Tensor<Q::T, Q::B>,
        eos_out: Tensor<Q::T, Q::B>,
        /// `[1, kv]`, see [`DeviceStep`].
        mask: Tensor<Q::T, Q::B>,
        rope: RotaryEmbedding<Q::T, Q::B>,
        empty_text: Tensor<Q::T, Q::B>,
        kv: usize,
        graph: Option<CaptureGraph>,
        /// Whether `mask` matches the state's filled prefix. Prompting fills more.
        mask_ready: bool,
    }

    impl<Q: BackendQ> StepGraph<Q> {
        /// `None` off CUDA, for a batch of more than one, or when disabled.
        pub(super) fn new(model: &TTSModel<Q>, base: &TTSState<Q>) -> Result<Option<Self>> {
            let any: &dyn std::any::Any = model.device();
            let Some(dev) = any.downcast_ref::<Device>() else { return Ok(None) };
            let Some((_, kv, 1)) = positions(base) else { return Ok(None) };
            if !enabled() {
                return Ok(None);
            }
            let d = model.device();
            let ldim = model.flow_lm.ldim;
            let neg_inf = Q::T::from_f32(f32::NEG_INFINITY);
            let fl = &base.flow_lm_state;
            let state = TTSState {
                flow_lm_state: crate::flow_lm::FlowLMState {
                    transformer_state: fl.transformer_state.with_seq_budget(kv)?,
                    conditions: fl.conditions.as_ref().map(|c| c.copy()).transpose()?,
                },
            };
            Ok(Some(Self {
                dev: dev.clone(),
                state,
                pos: Tensor::zeros(1, d)?,
                latent_in: Tensor::zeros((1, 1, ldim), d)?,
                noise: Tensor::zeros((1, ldim), d)?,
                latent_out: Tensor::zeros((1, 1, ldim), d)?,
                eos_out: Tensor::zeros((1, 1), d)?,
                mask: Tensor::full(neg_inf, (1, kv), d)?,
                rope: model.flow_lm.transformer.rope_table(kv, d)?,
                empty_text: Tensor::zeros((1, 0, model.flow_lm.conditioner.output_dim), d)?,
                kv,
                graph: None,
                mask_ready: false,
            }))
        }

        /// Whether generations from `base` can run in this graph's state.
        pub(super) fn fits(&self, base: &TTSState<Q>) -> bool {
            let ours = self.state.flow_lm_state.conditions.as_ref().map(|c| c.dims().to_vec());
            let theirs = base.flow_lm_state.conditions.as_ref().map(|c| c.dims().to_vec());
            matches!(positions(base), Some((_, kv, 1)) if kv == self.kv) && ours == theirs
        }

        /// Start a chunk from the primed state `base`: load its prefix and conditioning into
        /// this graph's state and prompt it with the chunk's tokens.
        pub fn begin(
            &mut self,
            model: &TTSModel<Q>,
            base: &TTSState<Q>,
            tokens: &[u32],
        ) -> Result<()> {
            let (ours, theirs) = (&mut self.state.flow_lm_state, &base.flow_lm_state);
            load_prefix(&mut ours.transformer_state, &theirs.transformer_state)?;
            if let (Some(ours), Some(theirs)) = (&ours.conditions, &theirs.conditions) {
                ours.copy_(theirs)?;
            }
            model.prompt_text(&mut self.state, tokens)?;
            self.mask_ready = false;
            Ok(())
        }

        /// One step from the previous latent, or from BOS for the chunk's first, as
        /// [`TTSModel::generate_step_parts`] would take it: the latent and the raw eos logit.
        #[allow(clippy::type_complexity)]
        pub fn step(
            &mut self,
            model: &TTSModel<Q>,
            prev: Option<&Tensor<Q::T, Q::B>>,
            rng: &mut impl Rng,
        ) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
            let d = model.device();
            let Some((pos, kv, _)) = positions(&self.state) else {
                xn::bail!("the state has no flow LM layers")
            };
            if pos >= kv {
                xn::bail!("no room left: the step graph holds {kv} positions")
            }
            if !self.mask_ready {
                let zero = Q::T::from_f32(0.0);
                let neg_inf = Q::T::from_f32(f32::NEG_INFINITY);
                let mask = (0..kv).map(|j| if j < pos { zero } else { neg_inf }).collect();
                self.mask.copy_(&Tensor::from_vec(mask, (1, kv), d)?)?;
                self.mask_ready = true;
            }
            self.pos.copy_(&Tensor::from_vec(vec![pos as i64], 1, d)?)?;
            let ldim = model.flow_lm.ldim;
            let noise = model.flow_lm.sample_noise(1, rng);
            self.noise.copy_(&Tensor::from_vec(noise, (1, ldim), d)?)?;
            match prev {
                Some(prev) => self.latent_in.copy_(prev)?,
                None => self.latent_in.copy_(&model.flow_lm.bos(1)?)?,
            }
            match &self.graph {
                Some(graph) => graph.launch()?,
                None => {
                    let graph = record(&self.dev, || self.body(model))?;
                    self.graph = Some(graph);
                }
            }
            self.state.flow_lm_state.transformer_state.advance(1);
            Ok((self.latent_out.copy()?, self.eos_out.copy()?))
        }

        fn body(&self, model: &TTSModel<Q>) -> Result<()> {
            let heads = model.flow_lm.transformer.heads();
            let step = DeviceStep::new(&self.pos, &self.mask, &self.rope, (1, heads.0, heads.1))?;
            // The step writes the caches in place and leaves the host-side position alone, so a
            // copy of the state's handles is as good as the state.
            let mut state = self.state.flow_lm_state.clone();
            let (latent, eos) = model.flow_lm.sample_next_latent_step(
                &self.latent_in,
                &self.empty_text,
                &mut state,
                &step,
                &self.noise,
                model.lsd_decode_steps(),
            )?;
            self.latent_out.copy_(&latent)?;
            self.eos_out.copy_(&eos)
        }
    }

    /// Copy `src`'s filled prefix into `dst`'s caches, of at least that capacity.
    fn load_prefix<T: WithDTypeF, B: xn::Backend>(
        dst: &mut StreamingTransformerState<T, B>,
        src: &StreamingTransformerState<T, B>,
    ) -> Result<()> {
        for (d, s) in dst.layer_states.iter_mut().zip(src.layer_states.iter()) {
            let (LayerAttentionState::FlowLm(d), LayerAttentionState::FlowLm(s)) = (d, s) else {
                xn::bail!("loading a prefix into a Mimi layer")
            };
            let used = s.current_end;
            if used > 0 {
                d.k_cache.slice_set(&s.k_cache.narrow(1, 0..used)?.contiguous()?, 1usize, 0)?;
                d.v_cache.slice_set(&s.v_cache.narrow(1, 0..used)?.contiguous()?, 1usize, 0)?;
            }
            d.current_end = used;
        }
        Ok(())
    }
}
