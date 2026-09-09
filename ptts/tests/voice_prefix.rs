//! Guards the extract/inject round trip `VoicePrefix` performs on the flow LM's KV cache.
//!
//! The optimization it enables -- condition on a voice once, then seed every request's state
//! from the result -- is only sound if a seeded state is indistinguishable from one that ran
//! `prompt_audio` itself. These tests build cache states directly rather than loading weights,
//! so they assert exactly that property: the same numbers land in the same positions, the fill
//! marker agrees, and nothing past the prefix is disturbed.

use ptts::flow_lm::FlowLMState;
use ptts::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use ptts::tts_model::{TTSState, VoicePrefix, max_frames_for, seq_budget_for};
use xn::{CpuDevice, Result, Tensor};

type Q = xn::Unquantized<f32, CpuDevice>;

const LAYERS: usize = 4;
const HEADS: usize = 8;
const HEAD_DIM: usize = 16;
const VOICE_LEN: usize = 5;

fn noise(n: usize, seed: &mut u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            (*seed as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

/// A state of `capacity` positions per layer, with `filled` of them carrying `seed`-derived
/// noise -- what the cache looks like after conditioning on a voice of that length.
fn state(capacity: usize, filled: usize, seed: &mut u32) -> Result<TTSState<Q>> {
    let shape = (1, capacity, HEADS, HEAD_DIM);
    let mut layer_states = Vec::with_capacity(LAYERS);
    for _ in 0..LAYERS {
        let k_cache: Tensor<f32, CpuDevice> = Tensor::zeros(shape, &xn::CPU)?;
        let v_cache: Tensor<f32, CpuDevice> = Tensor::zeros(shape, &xn::CPU)?;
        if filled > 0 {
            let n = filled * HEADS * HEAD_DIM;
            let k = Tensor::from_vec(noise(n, seed), (1, filled, HEADS, HEAD_DIM), &xn::CPU)?;
            let v = Tensor::from_vec(noise(n, seed), (1, filled, HEADS, HEAD_DIM), &xn::CPU)?;
            k_cache.slice_set(&k, 1usize, 0)?;
            v_cache.slice_set(&v, 1usize, 0)?;
        }
        layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState {
            k_cache,
            v_cache,
            current_end: filled,
        }));
    }
    Ok(TTSState {
        flow_lm_state: FlowLMState {
            transformer_state: StreamingTransformerState { layer_states },
        },
    })
}

/// `Result::unwrap_err` needs the Ok type to be `Debug`, which a tensor-bearing prefix is not.
fn error_of<T>(r: Result<T>) -> String {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
}

fn caches(state: &TTSState<Q>) -> Vec<(Vec<f32>, Vec<f32>, usize)> {
    state
        .flow_lm_state
        .transformer_state
        .layer_states
        .iter()
        .map(|l| match l {
            LayerAttentionState::FlowLm(s) => {
                (s.k_cache.to_vec().unwrap(), s.v_cache.to_vec().unwrap(), s.current_end)
            }
            LayerAttentionState::Mimi(_) => panic!("unexpected mimi state"),
        })
        .collect()
}

/// The property the optimization rests on: a state seeded from a prefix is byte-identical to
/// the conditioned state the prefix came from, including the zeros past the prefix.
#[test]
fn seeding_reproduces_the_conditioned_state() -> Result<()> {
    let mut seed = 0x1234_5678;
    let conditioned = state(VOICE_LEN, VOICE_LEN, &mut seed)?;
    let prefix = VoicePrefix::from_state(&conditioned)?;
    assert_eq!(prefix.len(), VOICE_LEN);
    assert!(!prefix.is_empty());

    // Same capacity as the state it was taken from: the two must agree exactly.
    let mut seeded = state(VOICE_LEN, 0, &mut seed)?;
    prefix.apply(&mut seeded)?;
    assert_eq!(caches(&seeded), caches(&conditioned));
    Ok(())
}

/// The point of holding the prefix compactly: it seeds a state of any larger capacity, which is
/// what lets each request size its own cache instead of sharing one fixed budget.
#[test]
fn seeding_a_larger_state_leaves_the_tail_untouched() -> Result<()> {
    let mut seed = 0x9e37_79b9;
    let conditioned = state(VOICE_LEN, VOICE_LEN, &mut seed)?;
    let prefix = VoicePrefix::from_state(&conditioned)?;

    let capacity = 64;
    let mut seeded = state(capacity, 0, &mut seed)?;
    prefix.apply(&mut seeded)?;

    let want = caches(&conditioned);
    for (layer, (k, v, end)) in caches(&seeded).into_iter().enumerate() {
        assert_eq!(end, VOICE_LEN, "layer {layer} fill marker");
        let stride = HEADS * HEAD_DIM;
        let head = VOICE_LEN * stride;
        assert_eq!(k[..head], want[layer].0[..head], "layer {layer} keys");
        assert_eq!(v[..head], want[layer].1[..head], "layer {layer} values");
        // Everything the request has yet to write stays zero.
        assert!(k[head..].iter().all(|&x| x == 0.0), "layer {layer} key tail");
        assert!(v[head..].iter().all(|&x| x == 0.0), "layer {layer} value tail");
        assert_eq!(k.len(), capacity * stride);
    }
    Ok(())
}

#[test]
fn a_state_too_small_for_the_prefix_is_rejected() -> Result<()> {
    let mut seed = 0xdead_beef;
    let prefix = VoicePrefix::from_state(&state(VOICE_LEN, VOICE_LEN, &mut seed)?)?;
    let mut small = state(VOICE_LEN - 1, 0, &mut seed)?;
    let err = error_of(prefix.apply(&mut small));
    assert!(err.contains("too few"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn a_state_with_the_wrong_layer_count_is_rejected() -> Result<()> {
    let mut seed = 0x0bad_f00d;
    let prefix = VoicePrefix::from_state(&state(VOICE_LEN, VOICE_LEN, &mut seed)?)?;
    let mut wrong = state(VOICE_LEN, 0, &mut seed)?;
    wrong.flow_lm_state.transformer_state.layer_states.pop();
    let err = error_of(prefix.apply(&mut wrong));
    assert!(err.contains("layers"), "unexpected error: {err}");
    Ok(())
}

/// Layers are filled by one forward pass, so disagreement means the state was assembled by hand
/// and snapshotting it would silently truncate a layer.
#[test]
fn layers_disagreeing_on_the_fill_marker_are_rejected() -> Result<()> {
    let mut seed = 0xfeed_face;
    let mut s = state(VOICE_LEN, VOICE_LEN, &mut seed)?;
    match &mut s.flow_lm_state.transformer_state.layer_states[2] {
        LayerAttentionState::FlowLm(l) => l.current_end = VOICE_LEN - 1,
        LayerAttentionState::Mimi(_) => panic!("unexpected mimi state"),
    }
    let err = error_of(VoicePrefix::from_state(&s));
    assert!(err.contains("disagree"), "unexpected error: {err}");
    Ok(())
}

#[test]
fn an_unconditioned_state_yields_an_empty_prefix() -> Result<()> {
    let mut seed = 0x5eed_5eed;
    let prefix = VoicePrefix::from_state(&state(VOICE_LEN, 0, &mut seed)?)?;
    assert!(prefix.is_empty());
    assert_eq!(prefix.len(), 0);

    // Applying it is a no-op that still resets the fill marker to zero.
    let mut target = state(32, 0, &mut seed)?;
    prefix.apply(&mut target)?;
    for (_, _, end) in caches(&target) {
        assert_eq!(end, 0);
    }
    Ok(())
}

#[test]
fn seq_budget_covers_the_longest_chunk_and_is_capped() {
    // Voice, tokens and one position per frame the chunk may generate, plus slack.
    let one = seq_budget_for(100, [30]);
    assert_eq!(one, 100 + 30 + max_frames_for(30) + ptts::tts_model::SEQ_BUDGET_SLACK);

    // The largest chunk sets the budget, not the last or the sum.
    assert_eq!(seq_budget_for(100, [30, 10, 20]), one);

    // No chunks at all still leaves room for the voice itself.
    assert_eq!(seq_budget_for(100, []), 100 + ptts::tts_model::SEQ_BUDGET_SLACK);

    // A pathological input cannot ask for an unbounded cache.
    assert_eq!(seq_budget_for(100, [1_000_000]), ptts::tts_model::MAX_SEQ_BUDGET);
}
