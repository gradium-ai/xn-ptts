/* Print the chunks and token ids libptts_text produces for each line of an inputs file
 * (`\n`, `\r` and `\t` in a line are unescaped), in the format tests/parity.rs writes to
 * target/expected.txt.
 *
 *   dump <tokenizer.json> <lang> <inputs.txt>
 */
#include "ptts_text.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void unescape(char* s) {
    char* w = s;
    for (char* r = s; *r; r++) {
        if (r[0] == '\\' && (r[1] == 'n' || r[1] == 'r' || r[1] == 't')) {
            *w++ = r[1] == 'n' ? '\n' : r[1] == 'r' ? '\r' : '\t';
            r++;
        } else {
            *w++ = *r;
        }
    }
    *w = 0;
}

static void print_escaped(const char* key, const char* s) {
    printf("  %s=", key);
    for (; *s; s++) {
        switch (*s) {
            case '\\': fputs("\\\\", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default: putchar(*s);
        }
    }
    putchar('\n');
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <tokenizer.json> <lang> <inputs.txt>\n", argv[0]);
        return 2;
    }
    char* err = NULL;
    ptts_text* h = ptts_text_new(argv[1], argv[2], &err);
    if (!h) {
        fprintf(stderr, "ptts_text_new: %s\n", err);
        ptts_text_string_free(err);
        return 1;
    }
    FILE* f = fopen(argv[3], "r");
    if (!f) {
        perror(argv[3]);
        return 1;
    }
    static char line[1 << 20];
    int i = 0;
    while (fgets(line, sizeof line, f)) {
        size_t n = strlen(line);
        if (n && line[n - 1] == '\n') line[n - 1] = 0;
        unescape(line);
        ptts_text_chunks* out = ptts_text_split(h, line, 0, 0.0, &err);
        if (!out) {
            printf("#%d error: %s\n", i++, err);
            ptts_text_string_free(err);
            err = NULL;
            continue;
        }
        printf("#%d chunks=%zu\n", i++, out->n_chunks);
        for (size_t c = 0; c < out->n_chunks; c++) {
            const ptts_text_chunk* ch = &out->chunks[c];
            print_escaped("text", ch->text);
            print_escaped("prepared", ch->prepared);
            printf("  frames_after_eos=%u frame_budget=%u seq_budget=%u n_tokens=%zu\n",
                   ch->frames_after_eos, ch->frame_budget, ch->seq_budget, ch->n_tokens);
            printf("  ids=");
            for (size_t t = 0; t < ch->n_tokens; t++) printf(t ? " %u" : "%u", ch->tokens[t]);
            printf("\n");
        }
        ptts_text_chunks_free(out);
    }
    fclose(f);
    ptts_text_free(h);
    return 0;
}
