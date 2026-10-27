// Speak a line of text into a WAV file through the C interface, ../include/ptts.h.
//
//     speak MODEL LANG TEXT OUT.wav [VOICE]
//
// MODEL is a checkpoint folder, or on Apple platforms a Core ML bundle. ../README.md shows how to
// build the library and this program.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ptts.h"

#define SAMPLE_RATE 24000

// Write `v` as `bytes` little-endian bytes, which WAV files use on every platform.
static void put(uint8_t *p, uint32_t v, int bytes) {
    for (int i = 0; i < bytes; i++) p[i] = (uint8_t)(v >> (8 * i));
}

// The 44-byte header of a 16-bit mono WAV file holding `samples` samples.
static void wav_header(uint8_t h[44], uint32_t samples) {
    memcpy(h, "RIFF", 4);
    put(h + 4, 36 + samples * 2, 4);
    memcpy(h + 8, "WAVEfmt ", 8);
    put(h + 16, 16, 4);               // the size of the format chunk
    put(h + 20, 1, 2);                // PCM
    put(h + 22, 1, 2);                // one channel
    put(h + 24, SAMPLE_RATE, 4);
    put(h + 28, SAMPLE_RATE * 2, 4);  // bytes per second
    put(h + 32, 2, 2);                // bytes per sample
    put(h + 34, 16, 2);               // bits per sample
    memcpy(h + 36, "data", 4);
    put(h + 40, samples * 2, 4);
}

// Called with the audio as it is made, on the thread that called ptts_speak. A player would queue
// it here; this appends it to the file as 16-bit samples. Returning false stops early.
static bool on_audio(const float *pcm, size_t n, void *user) {
    FILE *out = user;
    for (size_t i = 0; i < n; i++) {
        float s = pcm[i] < -1.0f ? -1.0f : pcm[i] > 1.0f ? 1.0f : pcm[i];
        uint8_t b[2];
        put(b, (uint16_t)(int16_t)(s * 32767.0f), 2);
        if (fwrite(b, 1, 2, out) != 2) return false;
    }
    return true;
}

// Print a list from ptts_voices or ptts_languages: names separated by NULs, ending in a second NUL.
static void print_names(const char *names) {
    for (const char *n = names; *n; n += strlen(n) + 1) fprintf(stderr, "  %s\n", n);
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: %s MODEL LANG TEXT OUT.wav [VOICE]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1], *lang = argv[2], *text = argv[3], *path = argv[4];

#ifdef __APPLE__
    uint32_t unit = PTTS_UNIT_ANE;  // the Neural Engine
#else
    uint32_t unit = PTTS_UNIT_CPU;  // the only unit elsewhere
#endif
    // The slow call. An app loads once and keeps the handle.
    PttsHandle *h = ptts_new(model, unit, lang);
    if (!h) {
        fprintf(stderr, "cannot load %s: %s\n", model, ptts_last_error(NULL));
        return 1;
    }
    if (argc == 6 && !ptts_set_voice(h, argv[5])) {
        fprintf(stderr, "%s. The voices are:\n", ptts_last_error(h));
        print_names(ptts_voices(h));
        ptts_free(h);
        return 1;
    }

    FILE *out = fopen(path, "wb");
    if (!out) {
        perror(path);
        ptts_free(h);
        return 1;
    }
    // The length is not known yet: write the header again once it is.
    uint8_t header[44];
    wav_header(header, 0);
    fwrite(header, 1, sizeof header, out);

    PttsResult result;
    bool ok = ptts_speak(h, text, on_audio, out, &result);
    if (!ok) {
        fprintf(stderr, "cannot speak: %s\n", ptts_last_error(h));
    } else {
        // This needs OUT.wav to be a file: a pipe cannot seek back to the header.
        wav_header(header, (uint32_t)result.samples);
        if (fseek(out, 0, SEEK_SET) != 0 || fwrite(header, 1, sizeof header, out) != sizeof header) {
            perror(path);
            ok = false;
        }
    }
    int write_error = ferror(out);
    if (fclose(out) != 0 || write_error) {
        perror(path);
        ok = false;
    }
    if (ok) printf("%s: %.1f s of audio\n", path, (double)result.samples / SAMPLE_RATE);
    ptts_free(h);
    return ok ? 0 : 1;
}
