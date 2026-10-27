//! Generation policy: how many frames to run, how much KV budget to reserve, and
//! when to stop.
//!
//! These rules were inlined — as literals — in every frontend: the `ptts`
//! and `bench` examples, `ptts-pyo3` (twice), `ptts-wasm` and `ptts-ws-server`
//! each carried their own copy of `((n / 3.0 + 2.0) * 12.5).ceil()` and their own
//! EOS countdown loop. They are pure functions of the token count and the config,
//! so they belong here, where they can be unit tested once.
//!
//! Every frontend is converted now. The `ptts` and `say` examples, `ptts-pyo3` and
//! `ptts-ws-server` reach these through [`crate::synth`], which applies them on
//! the caller's behalf; `bench` and `ptts-wasm` call them directly, because both
//! drive [`crate::tts_model::TTSModel`] themselves — `bench` to attribute
//! sampling and decoding time to the frame that caused them, `ptts-wasm` because
//! `Synth` generates on background threads and the browser has none.

use crate::Tokenizer;
use crate::preprocess::Normalize;
use crate::tts_model::{prepare_text_prompt, split_into_best_sentences};
use crate::{Error, Result};

/// Extra KV-cache entries reserved on top of the text tokens and the generated
/// frames, covering the voice-prompt frames (~125 at 12.5Hz for a 10s prompt)
/// plus slack.
pub const PROMPT_SEQ_HEADROOM: usize = 512;

/// The most text tokens a chunk may have after [`fit`] cuts an oversized sentence, whatever
/// room a fixed budget leaves. Normal sentence grouping targets 50 tokens by default; that
/// target does not split a sentence on its own.
///
/// A chunk much longer than this outruns the length of speech the model learnt to say in one
/// go: it stops near that length anyway and skips words to get there.
pub const MAX_FIT_TOKENS: usize = 200;

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

/// One piece of the text, ready to prompt the model with.
///
/// Built by [`chunks`] or [`Chunk::new`], so the budgets always match the tokens.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Chunk {
    /// The piece of normalized text, before [`crate::tts_model::prepare_text_prompt`]; empty
    /// for a chunk built from tokens alone.
    pub text: String,
    pub tokens: Vec<u32>,
    /// The tail [`EosPolicy`] runs after the model signals the end.
    pub frames_after_eos: usize,
    /// The most frames this chunk may generate, from [`frame_budget`].
    pub frame_budget: usize,
}

impl Chunk {
    /// Prepare and tokenize one piece of already normalized text.
    pub fn new(text: String, tokenizer: &dyn Tokenizer, frame_rate: f64) -> Result<Self> {
        let (prepared, frames_after_eos) = prepare_text_prompt(&text);
        let tokens = tokenizer.encode(&prepared)?;
        let frame_budget = frame_budget(tokens.len(), frame_rate);
        Ok(Self { text, tokens, frames_after_eos, frame_budget })
    }

    /// The KV-cache length this chunk needs, from [`seq_budget`].
    pub fn seq_budget(&self) -> usize {
        seq_budget(self.tokens.len(), self.frame_budget)
    }
}

/// Split `text` into chunks the way every frontend does.
///
/// The whole input is normalized first, since that rewrites the characters the sentence
/// splitter looks for. Then whole sentences are grouped into chunks of up to `max_tokens`
/// tokens ([`split_into_best_sentences`]); a single sentence longer than that stays one chunk.
/// Each chunk is then prepared and tokenized on its own.
pub fn chunks(
    tokenizer: &dyn Tokenizer,
    text: &str,
    normalize: Normalize,
    max_tokens: usize,
    frame_rate: f64,
) -> Result<Vec<Chunk>> {
    let text = normalize.apply(text);
    let chunks = split_into_best_sentences(tokenizer, &text, Some(max_tokens))?
        .into_iter()
        .map(|text| Chunk::new(text, tokenizer, frame_rate))
        .collect::<Result<Vec<_>>>()?;
    if chunks.is_empty() {
        return Err(Error::invalid_argument("nothing to synthesize: the text is empty"));
    }
    Ok(chunks)
}

/// Cut every chunk of more than `max` tokens until each piece fits, keeping their order.
///
/// [`chunks`] never cuts inside a sentence. A caller with a fixed KV budget, or a fixed number of
/// prompt rows, cuts what is left over with this, and should keep `max` at or under
/// [`MAX_FIT_TOKENS`].
///
/// A chunk is cut at the sentence end nearest its middle, if it holds more than one sentence.
/// Otherwise it is cut after the comma, semicolon or colon nearest its middle. Either way each
/// side must keep a quarter of `max`: a piece of a few words reads badly. A piece cut after a
/// comma keeps it, so [`prepare_text_prompt`] does not end it with a full stop, and the model does
/// not read it as a finished sentence. A sentence with no such mark is cut at the word nearest its
/// middle, and that piece does get the full stop. Every piece starts with a capital letter, but
/// otherwise holds the same words the sentence did. A single word longer than `max`, such as a
/// long run of digits, is cut between its characters. Only a single character longer than `max`
/// is left as it is, for the caller to refuse.
pub fn fit(
    chunks: Vec<Chunk>,
    max: usize,
    tokenizer: &dyn Tokenizer,
    frame_rate: f64,
) -> Result<Vec<Chunk>> {
    fit_with_word_limit(chunks, max, max, tokenizer, frame_rate)
}

/// Cut chunks toward `max` tokens while keeping a single word intact when it fits
/// `word_limit`. Use this when the model accepts a longer word than the preferred
/// chunk size. The result may contain single-word chunks over `max`, but never over
/// `word_limit` unless even one character is too large.
pub fn fit_with_word_limit(
    chunks: Vec<Chunk>,
    max: usize,
    word_limit: usize,
    tokenizer: &dyn Tokenizer,
    frame_rate: f64,
) -> Result<Vec<Chunk>> {
    let word_limit = word_limit.max(max);
    let mut fitted = Vec::with_capacity(chunks.len());
    let mut todo: Vec<Chunk> = chunks.into_iter().rev().collect();
    while let Some(chunk) = todo.pop() {
        if chunk.tokens.len() <= max {
            fitted.push(chunk);
            continue;
        }
        let words: Vec<&str> = chunk.text.split_whitespace().collect();
        let halves = if words.len() >= 2 {
            let (a, b) = words.split_at(cut(&words, max.div_ceil(4), tokenizer)?);
            [a.join(" "), b.join(" ")]
        } else {
            if chunk.tokens.len() <= word_limit {
                fitted.push(chunk);
                continue;
            }
            let word = chunk.text.trim();
            match word.char_indices().nth(word.chars().count() / 2) {
                Some((mid, _)) if mid > 0 => [word[..mid].to_string(), word[mid..].to_string()],
                _ => {
                    fitted.push(chunk);
                    continue;
                }
            }
        };
        for half in halves.into_iter().rev() {
            todo.push(Chunk::new(half, tokenizer, frame_rate)?);
        }
    }
    Ok(fitted)
}

/// Cut chunks to `max` tokens and reject a piece that cannot be split further.
///
/// Call this before sizing a KV cache from the chunks. [`fit`] leaves a single oversized character
/// untouched so callers with their own fixed prompt limit can report an appropriate error.
pub fn fit_or_error(
    chunks: Vec<Chunk>,
    max: usize,
    tokenizer: &dyn Tokenizer,
    frame_rate: f64,
) -> Result<Vec<Chunk>> {
    let chunks = fit(chunks, max, tokenizer, frame_rate)?;
    if let Some(chunk) = chunks.iter().find(|chunk| chunk.tokens.len() > max) {
        return Err(Error::invalid_argument(format!(
            "a text chunk is {} tokens and cannot fit the {max}-token limit; shorten or split it",
            chunk.tokens.len()
        )));
    }
    Ok(chunks)
}

/// Where to cut `words`, as the number of words before the cut: after the sentence end nearest
/// the middle in tokens that leaves `min_side` tokens on each side, else after such a clause mark,
/// else after the word nearest the middle. Takes at least two words, and always leaves at least
/// one on each side.
fn cut(words: &[&str], min_side: usize, tokenizer: &dyn Tokenizer) -> Result<usize> {
    // `ends[i]` is the number of tokens in `words[..=i]`.
    let mut ends = Vec::with_capacity(words.len());
    let mut total = 0;
    for word in words {
        total += tokenizer.encode(word)?.len();
        ends.push(total);
    }
    let inner = || ends[..words.len() - 1].iter().copied().enumerate();
    let imbalance = |&(_, end): &(usize, usize)| end.abs_diff(total - end);
    // A mark can sit inside closing quotes or brackets: `"yes,"`, `“yes,”`, `oui,»`.
    let closers = ['"', '\'', ')', ']', '”', '’', '»'];
    let after = |marks: &[char]| {
        inner()
            .filter(|&(i, end)| {
                words[i].trim_end_matches(closers).ends_with(marks)
                    && end >= min_side
                    && total - end >= min_side
            })
            .min_by_key(imbalance)
    };
    let (i, _) = after(&['.', '!', '?'])
        .or_else(|| after(&[',', ';', ':']))
        .or_else(|| inner().min_by_key(imbalance))
        .ok_or_else(|| Error::invalid_argument("nothing to cut: fewer than two words"))?;
    Ok(i + 1)
}

/// The most text tokens one chunk can have when `room` KV slots are left after the voice prompt:
/// the largest `n` with `n + frame_budget(n, frame_rate) <= room`.
pub fn max_tokens_for(room: usize, frame_rate: f64) -> usize {
    (0..=room).rev().find(|&n| n + frame_budget(n, frame_rate) <= room).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per word and per `.`, `!` or `?`, so sentence ends get their own ids as
    /// they do in the real tokenizer.
    #[derive(Default)]
    struct Words(std::cell::RefCell<Vec<String>>);

    impl Tokenizer for Words {
        fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
            let mut vocab = self.0.borrow_mut();
            let mut id = |piece: String| match vocab.iter().position(|v| *v == piece) {
                Some(i) => i as u32,
                None => {
                    vocab.push(piece);
                    vocab.len() as u32 - 1
                }
            };
            let mut ids = vec![];
            for word in text.split_whitespace() {
                let stem = word.trim_end_matches(['.', '!', '?']);
                if !stem.is_empty() {
                    ids.push(id(stem.to_string()));
                }
                ids.extend(word[stem.len()..].chars().map(|c| id(c.to_string())));
            }
            Ok(ids)
        }

        fn decode(&self, tokens: &[u32]) -> xn::Result<String> {
            let vocab = self.0.borrow();
            let mut text = String::new();
            for &t in tokens {
                let piece = &vocab[t as usize];
                if !text.is_empty() && !matches!(piece.as_str(), "." | "!" | "?") {
                    text.push(' ');
                }
                text.push_str(piece);
            }
            Ok(text)
        }
    }

    fn texts(chunks: &[Chunk]) -> Vec<&str> {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    #[test]
    fn chunks_group_whole_sentences_up_to_the_limit() {
        let tok = Words::default();
        let text = "one two. three four. five six.";
        // Each sentence is three tokens: two of them fit in nine, all three do not fit in eight.
        let two = chunks(&tok, text, Normalize::OFF, 8, 12.5).unwrap();
        assert_eq!(texts(&two), ["One two. three four.", "five six."]);
        let one = chunks(&tok, text, Normalize::OFF, 9, 12.5).unwrap();
        assert_eq!(texts(&one), ["One two. three four. five six."]);
    }

    #[test]
    fn a_sentence_over_the_limit_stays_one_chunk() {
        // Never cut inside a sentence: that is what the original splitter does.
        let tok = Words::default();
        let chunks = chunks(&tok, "a b c d e f g h", Normalize::OFF, 3, 12.5).unwrap();
        assert_eq!(texts(&chunks), ["A b c d e f g h."]);
        assert!(chunks[0].tokens.len() > 3);
    }

    #[test]
    fn each_chunk_is_prepared_and_budgeted_on_its_own() {
        let tok = Words::default();
        let chunks =
            chunks(&tok, "short one. and now a longer second sentence", Normalize::OFF, 4, 12.5)
                .unwrap();
        assert_eq!(chunks.len(), 2);
        for chunk in &chunks {
            let (prepared, frames_after_eos) = prepare_text_prompt(&chunk.text);
            assert_eq!(chunk.tokens, tok.encode(&prepared).unwrap());
            assert_eq!(chunk.frames_after_eos, frames_after_eos);
            assert_eq!(chunk.frame_budget, frame_budget(chunk.tokens.len(), 12.5));
            assert_eq!(chunk.seq_budget(), seq_budget(chunk.tokens.len(), chunk.frame_budget));
        }
    }

    #[test]
    fn chunks_normalize_the_whole_text_first() {
        let tok = Words::default();
        let en = Normalize::for_lang(crate::preprocess::Lang::En);
        let chunks = chunks(&tok, "mail a@b now", en, 50, 12.5).unwrap();
        assert_eq!(texts(&chunks), ["Mail a at b now."]);
    }

    #[test]
    fn empty_text_is_an_error() {
        let err = chunks(&Words::default(), "   ", Normalize::OFF, 50, 12.5).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    fn fit_text(text: &str, max: usize) -> Vec<Chunk> {
        let tok = Words::default();
        let chunk = Chunk::new(text.to_string(), &tok, 12.5).unwrap();
        fit(vec![chunk], max, &tok, 12.5).unwrap()
    }

    /// The pieces as the model reads them.
    fn prompts(chunks: &[Chunk]) -> Vec<String> {
        chunks.iter().map(|c| prepare_text_prompt(&c.text).0).collect()
    }

    #[test]
    fn fit_cuts_after_the_middle_comma_without_a_full_stop() {
        let chunks = fit_text("one two three four five, six seven eight nine ten eleven twelve", 8);
        assert_eq!(
            prompts(&chunks),
            ["One two three four five,", "Six seven eight nine ten eleven twelve."]
        );
    }

    #[test]
    fn fit_passes_over_a_comma_that_would_leave_a_tiny_piece() {
        let chunks = fit_text("one, two three four five six seven eight", 5);
        assert_eq!(prompts(&chunks), ["One, two three four.", "Five six seven eight."]);
    }

    #[test]
    fn fit_counts_a_comma_inside_closing_quotes() {
        let text = "one two three four \"five,\" six seven eight nine ten eleven twelve";
        assert_eq!(
            prompts(&fit_text(text, 8)),
            ["One two three four \"five,\"", "Six seven eight nine ten eleven twelve."]
        );
    }

    #[test]
    fn fit_prefers_a_sentence_end_to_a_comma() {
        let tok = Words::default();
        let text = "one two, three four five six. seven eight nine ten.";
        let planned = chunks(&tok, text, Normalize::OFF, 50, 12.5).unwrap();
        assert_eq!(planned.len(), 1);
        let fitted = fit(planned, 8, &tok, 12.5).unwrap();
        assert_eq!(prompts(&fitted), ["One two, three four five six.", "Seven eight nine ten."]);
    }

    #[test]
    fn fit_keeps_the_words_in_order_and_every_piece_fits() {
        let words: Vec<String> =
            (0..60).map(|i| if i % 7 == 6 { format!("w{i},") } else { format!("w{i}") }).collect();
        let text = words.join(" ");
        let chunks = fit_text(&text, 6);
        assert!(chunks.iter().all(|c| c.tokens.len() <= 6));
        assert_eq!(chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" "), text);
    }

    #[test]
    fn fit_leaves_chunks_that_fit_alone() {
        let tok = Words::default();
        let planned = chunks(&tok, "one two. three four.", Normalize::OFF, 3, 12.5).unwrap();
        let texts_before: Vec<String> = planned.iter().map(|c| c.text.clone()).collect();
        assert_eq!(texts(&fit(planned, 3, &tok, 12.5).unwrap()), texts_before);
    }

    /// One token per character, so a single word can be longer than the limit.
    struct Chars;

    impl Tokenizer for Chars {
        fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
            Ok(text.chars().map(|c| c as u32).collect())
        }

        fn decode(&self, tokens: &[u32]) -> xn::Result<String> {
            Ok(tokens.iter().filter_map(|&t| char::from_u32(t)).collect())
        }
    }

    #[test]
    fn fit_cuts_a_word_that_is_too_long_on_its_own() {
        let word = "abcdefghijkl";
        let chunk = Chunk::new(word.to_string(), &Chars, 12.5).unwrap();
        let fitted = fit(vec![chunk], 5, &Chars, 12.5).unwrap();
        assert_eq!(texts(&fitted), ["abc", "def", "ghi", "jkl"]);
        assert!(fitted.iter().all(|c| c.tokens.len() <= 5));

        let chunk = Chunk::new("é".repeat(9), &Chars, 12.5).unwrap();
        let fitted = fit(vec![chunk], 4, &Chars, 12.5).unwrap();
        assert_eq!(fitted.iter().map(|c| c.text.as_str()).collect::<String>(), "é".repeat(9));
    }

    #[test]
    fn fit_keeps_a_word_that_fits_the_models_larger_limit() {
        let chunk = Chunk::new("hi abcdefghijkl bye".to_string(), &Chars, 12.5).unwrap();
        let fitted = fit_with_word_limit(vec![chunk], 5, 13, &Chars, 12.5).unwrap();
        assert_eq!(texts(&fitted), ["hi", "abcdefghijkl", "bye"]);
        assert_eq!(prompts(&fitted), ["Hi.", "Abcdefghijkl.", "Bye."]);

        let chunk = Chunk::new("abcdefghijklmnop".to_string(), &Chars, 12.5).unwrap();
        let fitted = fit_with_word_limit(vec![chunk], 5, 13, &Chars, 12.5).unwrap();
        assert_eq!(texts(&fitted), ["abcdefgh", "ijklmnop"]);
        assert!(fitted.iter().all(|piece| piece.tokens.len() <= 13));
    }

    #[test]
    fn fit_or_error_accepts_a_long_word_between_sentences() {
        let text = format!("Your number is {}. Thank you.", "9".repeat(40));
        let planned = chunks(&Chars, &text, Normalize::OFF, 50, 12.5).unwrap();
        let fitted = fit_or_error(planned, 20, &Chars, 12.5).unwrap();
        assert!(fitted.iter().all(|c| c.tokens.len() <= 20));
        let spoken: String = fitted.iter().map(|c| c.text.replace(' ', "")).collect();
        assert_eq!(spoken, text.replace(' ', ""));
    }

    #[test]
    fn fit_or_error_bounds_a_long_sentence_before_cache_sizing() {
        let tok = Words::default();
        let text = std::iter::repeat_n("word", 225).collect::<Vec<_>>().join(" ");
        let planned = chunks(&tok, &text, Normalize::OFF, 50, 12.5).unwrap();
        assert!(planned[0].tokens.len() > MAX_FIT_TOKENS);
        let fitted = fit_or_error(planned, MAX_FIT_TOKENS, &tok, 12.5).unwrap();
        assert!(fitted.iter().all(|chunk| chunk.tokens.len() <= MAX_FIT_TOKENS));

        let single = chunks(&tok, "word", Normalize::OFF, 50, 12.5).unwrap();
        let err = fit_or_error(single, 1, &tok, 12.5).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn max_tokens_for_is_the_largest_chunk_that_fits() {
        assert_eq!(max_tokens_for(0, 12.5), 0);
        for room in [30usize, 100, 796, 3971] {
            let n = max_tokens_for(room, 12.5);
            assert!(n + frame_budget(n, 12.5) <= room, "room = {room}");
            assert!(n + 1 + frame_budget(n + 1, 12.5) > room, "room = {room}");
        }
    }

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

    /// Reference implementation, transcribed from `ptts.rs` before the
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
