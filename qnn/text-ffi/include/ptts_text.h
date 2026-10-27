/* C interface to ptts's text front end: normalization, sentence chunking, prompt preparation
 * and tokenization. The chunks and token ids are exactly those ptts::synth::Synth::say and
 * Synth::stream generate from (ptts::plan::chunks).
 *
 * All strings are NUL-terminated UTF-8. A handle is immutable once built and may be shared
 * between threads. Errors come back through `char** error` (pass NULL to ignore them) as
 * strings the library allocates; free them with ptts_text_string_free. */
#ifndef PTTS_TEXT_H
#define PTTS_TEXT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Synth's default token budget per chunk, and the codec frame rate of the Phonon configs. */
#define PTTS_TEXT_MAX_TOKENS_PER_CHUNK 50
#define PTTS_TEXT_FRAME_RATE 12.5

typedef struct ptts_text ptts_text;

typedef struct {
    /* The chunk's normalized text, before prepare_text_prompt. */
    const char* text;
    /* prepare_text_prompt(text): capitalized, whitespace collapsed, a full stop added after a
     * trailing letter or digit. Exactly the string that was tokenized. */
    const char* prepared;
    /* The ids to prompt the flow LM with (no special tokens added). */
    const uint32_t* tokens;
    size_t n_tokens;
    /* Frames still generated after the model signals EOS, the EOS frame itself not counted:
     * 3 for a chunk of four words or fewer, 1 otherwise. */
    uint32_t frames_after_eos;
    /* The most frames this chunk may generate: ceil((n_tokens / 3 + 2) * frame_rate). */
    uint32_t frame_budget;
    /* KV-cache length the chunk needs: n_tokens + frame_budget + 512 (voice prompt headroom). */
    uint32_t seq_budget;
} ptts_text_chunk;

typedef struct {
    const ptts_text_chunk* chunks;
    size_t n_chunks;
} ptts_text_chunks;

/* Open a tokenizer.json and normalize as `lang`: "en", "fr", "de", "es", "pt", or "none"
 * (text handed to the tokenizer as written). Uses the default rewrite rules. NULL on failure. */
ptts_text* ptts_text_new(const char* tokenizer_json_path, const char* lang, char** error);

/* As ptts_text_new, with the rewrite rules as the frontends' --rewrites flag takes them:
 * "default", "all", "none", or a comma-separated list of rule names. NULL means "default". */
ptts_text* ptts_text_new_with_rewrites(const char* tokenizer_json_path, const char* lang,
                                       const char* rewrites, char** error);

/* Split `text` into chunks, in generation order. max_tokens 0 means
 * PTTS_TEXT_MAX_TOKENS_PER_CHUNK; frame_rate <= 0 means PTTS_TEXT_FRAME_RATE. NULL on failure,
 * including text that is empty after normalization. Free with ptts_text_chunks_free. */
ptts_text_chunks* ptts_text_split(const ptts_text* h, const char* text, size_t max_tokens,
                                  double frame_rate, char** error);

/* The whole text after normalization (the first step of ptts_text_split), for debugging.
 * Free with ptts_text_string_free. NULL on failure. */
char* ptts_text_normalize(const ptts_text* h, const char* text, char** error);

void ptts_text_chunks_free(ptts_text_chunks* chunks);
void ptts_text_string_free(char* s);
void ptts_text_free(ptts_text* h);

#ifdef __cplusplus
}
#endif

#endif /* PTTS_TEXT_H */
