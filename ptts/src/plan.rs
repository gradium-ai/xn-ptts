//! Generation policy: how many frames to run, how much KV budget to reserve, and
//! when to stop.
//!
//! These rules were inlined — as literals — in every frontend: the `pocket_tts`
//! and `bench` examples, `ptts-pyo3` (twice), `ptts-wasm` and `ptts-ws-server`
//! each carried their own copy of `((n / 3.0 + 2.0) * 12.5).ceil()` and their own
//! EOS countdown loop. They are pure functions of the token count and the config,
//! so they belong here, where they can be unit tested once.
//!
//! The two examples are converted; `ptts-pyo3`, `ptts-wasm` and `ptts-ws-server`
//! still carry their copies and are left for a follow-up, since each also has its
//! own generation loop to untangle.

/// Extra KV-cache entries reserved on top of the text tokens and the generated
/// frames, covering the voice-prompt frames (~125 at 12.5Hz for a 10s prompt)
/// plus slack.
pub const PROMPT_SEQ_HEADROOM: usize = 512;

/// Frames of audio to generate for a text prompt of `num_tokens` tokens.
///
/// Roughly three tokens per second of speech, plus two seconds of slack,
/// converted to frames at the codec frame rate. This is an upper bound: normal
/// generation stops earlier, when the model signals EOS (see [`EosPolicy`]).
///
/// `frame_rate` comes from `TTSConfig::mimi.frame_rate`. The frontends all
/// hardcoded `12.5`, which silently assumed the shipped config.
pub fn frame_budget(num_tokens: usize, frame_rate: f64) -> usize {
    ((num_tokens as f64 / 3.0 + 2.0) * frame_rate).ceil() as usize
}

/// KV-cache length to allocate for a single text chunk: its text tokens, the
/// frames it may generate, and [`PROMPT_SEQ_HEADROOM`] for the voice prompt.
///
/// A state is allocated once and reused across chunks, so callers with several
/// chunks should allocate the max over all of them.
pub fn seq_budget(num_tokens: usize, frame_budget: usize) -> usize {
    num_tokens + PROMPT_SEQ_HEADROOM + frame_budget
}

/// Tracks the tail of a generation: once the model reports EOS, a few more
/// frames are still emitted so the codec can close out the utterance cleanly.
///
/// `frames_after_eos` comes from [`crate::tts_model::prepare_text_prompt`] — 3
/// for very short prompts, 1 otherwise.
#[derive(Clone, Copy, Debug)]
pub struct EosPolicy {
    frames_after_eos: usize,
    countdown: Option<usize>,
}

impl EosPolicy {
    pub fn new(frames_after_eos: usize) -> Self {
        Self { frames_after_eos, countdown: None }
    }

    /// Records the EOS flag of the frame that was just generated and returns
    /// whether generation should stop now.
    ///
    /// Call this once per frame, *after* the frame has been handed to the
    /// decoder — the EOS frame itself is part of the output.
    pub fn should_stop(&mut self, is_eos: bool) -> bool {
        if is_eos && self.countdown.is_none() {
            self.countdown = Some(self.frames_after_eos);
        }
        match self.countdown.as_mut() {
            None => false,
            Some(0) => true,
            Some(countdown) => {
                *countdown -= 1;
                false
            }
        }
    }

    /// True once the model has signalled EOS, whether or not the tail has run out.
    pub fn saw_eos(&self) -> bool {
        self.countdown.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_budget_matches_the_frontends() {
        // The literal every frontend carried: ((n / 3 + 2) * 12.5).ceil().
        for n in [0usize, 1, 7, 12, 50, 137] {
            let expected = ((n as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
            assert_eq!(frame_budget(n, 12.5), expected, "n = {n}");
        }
        assert_eq!(frame_budget(0, 12.5), 25);
        assert_eq!(frame_budget(50, 12.5), 234);
    }

    #[test]
    fn frame_budget_follows_the_frame_rate() {
        assert_eq!(frame_budget(30, 12.5), frame_budget(30, 25.0) / 2);
    }

    #[test]
    fn seq_budget_covers_tokens_frames_and_headroom() {
        assert_eq!(seq_budget(40, 234), 40 + 512 + 234);
    }

    /// Reference implementation, transcribed from `pocket_tts.rs` before the
    /// refactor. `EosPolicy` must agree with it frame for frame.
    fn reference(eos_at: Option<usize>, frames_after_eos: usize, max_frames: usize) -> usize {
        let mut eos_countdown: Option<usize> = None;
        let mut emitted = 0;
        for step in 0..max_frames {
            emitted += 1;
            let is_eos = eos_at == Some(step);
            if is_eos && eos_countdown.is_none() {
                eos_countdown = Some(frames_after_eos);
            }
            if let Some(ref mut countdown) = eos_countdown {
                if *countdown == 0 {
                    break;
                }
                *countdown -= 1;
            }
        }
        emitted
    }

    fn under_test(eos_at: Option<usize>, frames_after_eos: usize, max_frames: usize) -> usize {
        let mut policy = EosPolicy::new(frames_after_eos);
        let mut emitted = 0;
        for step in 0..max_frames {
            emitted += 1;
            if policy.should_stop(eos_at == Some(step)) {
                break;
            }
        }
        emitted
    }

    #[test]
    fn eos_policy_matches_the_reference_loop() {
        for frames_after_eos in [0usize, 1, 3] {
            for eos_at in [None, Some(0), Some(1), Some(5), Some(19)] {
                for max_frames in [1usize, 6, 20] {
                    assert_eq!(
                        under_test(eos_at, frames_after_eos, max_frames),
                        reference(eos_at, frames_after_eos, max_frames),
                        "frames_after_eos = {frames_after_eos}, eos_at = {eos_at:?}, max = {max_frames}"
                    );
                }
            }
        }
    }

    #[test]
    fn eos_policy_emits_the_eos_frame_plus_the_tail() {
        // EOS on frame 5 with a 1-frame tail: frames 0..=6 are emitted.
        assert_eq!(under_test(Some(5), 1, 100), 7);
        // A 3-frame tail emits three more.
        assert_eq!(under_test(Some(5), 3, 100), 9);
        // No EOS: capped by max_frames.
        assert_eq!(under_test(None, 1, 100), 100);
    }

    #[test]
    fn eos_policy_reports_eos() {
        let mut policy = EosPolicy::new(1);
        assert!(!policy.saw_eos());
        policy.should_stop(false);
        assert!(!policy.saw_eos());
        policy.should_stop(true);
        assert!(policy.saw_eos());
    }
}
