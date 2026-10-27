// C interface to Phonon: the Core ML driver on Apple, which the PhononTTS Swift package wraps,
// and the CPU everywhere else, which android/PhononTTS.kt wraps.
// Declared by hand: keep it in step with ../src/lib.rs, which documents each call in full.
#ifndef PTTS_H
#define PTTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PTTS_UNIT_CPU 0
#define PTTS_UNIT_ANE 2

typedef struct PttsHandle PttsHandle;

/// One utterance's timings. The audio itself went to the callback.
typedef struct {
    uint32_t frames;
    size_t samples;   // 24 kHz mono samples delivered
    double ttfa_ms;   // from the call to the first audio
    double total_ms;
} PttsResult;

/// Called with audio as it is decoded, on the thread that called ptts_speak: one frame per call on
/// Core ML, one or more on the CPU. Return false to stop.
typedef bool (*PttsFrameFn)(const float *pcm, size_t n, void *user);

/// Load a model. On Apple, `dir` is an exported bundle, compiled for this device first if needed
/// (once per install), and `unit` is PTTS_UNIT_ANE or PTTS_UNIT_CPU. Elsewhere, `dir` is a
/// checkpoint folder and `unit` is PTTS_UNIT_CPU. `lang` is en, fr, de, es, pt or none. NULL on
/// failure; then ptts_last_error(NULL), on the same thread, says why. A bundle without voices fails
/// to load; a checkpoint folder without voices loads, lists none, and speaks in no particular
/// voice.
PttsHandle *ptts_new(const char *dir, uint32_t unit, const char *lang);
bool ptts_speak(PttsHandle *h, const char *text, PttsFrameFn cb, void *user, PttsResult *out);
/// NUL-separated voice names, ending in a second NUL. Borrowed.
const char *ptts_voices(const PttsHandle *h);
bool ptts_set_voice(PttsHandle *h, const char *name);
/// Normalize text as `lang` (en, fr, de, es, pt or none) from the next ptts_speak on, with no reload.
bool ptts_set_language(PttsHandle *h, const char *lang);
/// The languages the model was trained for, NUL-separated like ptts_voices. Borrowed.
const char *ptts_languages(const PttsHandle *h);
/// The last error on `h`, or the last ptts_new failure on this thread when `h` is NULL.
const char *ptts_last_error(const PttsHandle *h);
void ptts_free(PttsHandle *h);

#endif
