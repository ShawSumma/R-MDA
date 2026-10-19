#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_tokenizer.h"
#include "util.h"
#include "tokenizer.h"

mda_test_tokenizer_file_t mda_test_tokenizer_vocab(const mda_test_tokenizer_piece_t *multi) {
    mda_test_tokenizer_file_t file = {
        .buf = NULL,
        .size = 0,
    };
    FILE *w = open_memstream(&file.buf, &file.size);
    fprintf(w, "%u\n", 256u + MDA_TEST_TOKENIZER_PIECES);
    for (uint32_t b = 0; b < 256; b += 1) {
        char ch = (char)b;
        mda_tokenizer_write_entry(w, -8.0f, &ch, 1);
    }
    for (uint32_t i = 0; i < MDA_TEST_TOKENIZER_PIECES; i += 1) {
        mda_tokenizer_write_entry(w, -1.0f, multi[i].bytes, multi[i].len);
    }
    fclose(w);
    return file;
}

uint32_t mda_test_tokenizer_mismatches(const mda_tokenizer_t *t, const mda_test_tokenizer_piece_t *multi) {
    uint32_t bad = 0;
    for (uint32_t b = 0; b < 256; b += 1) {
        if (t->entries[b].len != 1 || (uint8_t)t->entries[b].bytes[0] != b) {
            bad += 1;
        }
    }
    for (uint32_t i = 0; i < MDA_TEST_TOKENIZER_PIECES; i += 1) {
        mda_tokenizer_entry_t *e = &t->entries[256 + i];
        if (e->len != multi[i].len || memcmp(e->bytes, multi[i].bytes, e->len) != 0) {
            bad += 1;
        }
    }
    return bad;
}

bool mda_test_tokenizer_vocab_ok(const mda_tokenizer_t *t, const mda_test_tokenizer_piece_t *multi) {
    if (t->count != 256 + MDA_TEST_TOKENIZER_PIECES) {
        return false;
    } else {
        return mda_test_tokenizer_mismatches(t, multi) == 0;
    }
}

bool mda_test_tokenizer_encode(const mda_tokenizer_t *t) {
    const char text[] = "x\0y\\n z\n\n the a\0b\0\0\\";
    uint32_t len = sizeof(text) - 1;
    mda_tokenizer_ids_t r = mda_tokenizer_encode(t, text, len);
    char *back = malloc(r.n * MDA_TOKENIZER_MAX_LEN + 1);
    uint32_t blen = 0;
    size_t i = 0;
    while (i < r.n && r.ids[i] < t->count) {
        mda_tokenizer_entry_t *e = &t->entries[r.ids[i]];
        memcpy(back + blen, e->bytes, e->len);
        blen += e->len;
        i += 1;
    }
    bool in_range = i == r.n;
    bool decoded = in_range && blen == len && memcmp(back, text, len) == 0;
    bool merged = r.n == len - 8;
    printf("encode: %zu tokens, in range %s, decodes back %s, multi-byte tokens used %s\n", r.n, mda_util_verdict(in_range), mda_util_verdict(decoded), mda_util_verdict(merged));
    free(back);
    free(r.ids);
    return decoded && merged;
}

bool mda_test_tokenizer_check(const mda_tokenizer_t *t, const mda_test_tokenizer_piece_t *multi) {
    if (t->entries == NULL) {
        return false;
    } else {
        bool vocab_ok = mda_test_tokenizer_vocab_ok(t, multi);
        printf("vocab round trip: %s\n", mda_util_verdict(vocab_ok));
        bool encode_ok = mda_test_tokenizer_encode(t);
        return vocab_ok && encode_ok;
    }
}

bool mda_test_tokenizer_main(void) {
    mda_test_tokenizer_piece_t multi[MDA_TEST_TOKENIZER_PIECES] = {
        {
            .bytes = "\n\n",
            .len = 2,
        },
        {
            .bytes = "\\n",
            .len = 2,
        },
        {
            .bytes = "a\0b",
            .len = 3,
        },
        {
            .bytes = " the",
            .len = 4,
        },
        {
            .bytes = "\0\0",
            .len = 2,
        },
    };
    mda_test_tokenizer_file_t file = mda_test_tokenizer_vocab(multi);
    mda_tokenizer_t t = mda_tokenizer_read(fmemopen(file.buf, file.size, "rb"), "test vocab");
    bool ok = mda_test_tokenizer_check(&t, multi);
    free(file.buf);
    return ok;
}

int main(void) {
    bool ok = mda_test_tokenizer_main();
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
