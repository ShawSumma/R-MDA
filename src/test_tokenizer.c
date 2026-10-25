#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_tokenizer.h"
#include "util.h"
#include "unicode.h"
#include "tokenizer.h"

mda_test_tokenizer_file_t mda_test_tokenizer_file(const mda_test_tokenizer_merge_t *merges) {
    mda_test_tokenizer_file_t file = {
        .buf = NULL,
        .size = 0,
    };
    FILE *w = open_memstream(&file.buf, &file.size);
    fprintf(w, "%u\n", 256u + MDA_TEST_TOKENIZER_MERGES);
    for (uint32_t b = 0; b < 256; b += 1) {
        char ch = (char)b;
        mda_tokenizer_write_entry(w, &ch, 1);
    }
    for (uint32_t i = 0; i < MDA_TEST_TOKENIZER_MERGES; i += 1) {
        mda_tokenizer_write_entry(w, merges[i].bytes, (uint32_t)strlen(merges[i].bytes));
    }
    fprintf(w, "%u\n", MDA_TEST_TOKENIZER_MERGES);
    for (uint32_t i = 0; i < MDA_TEST_TOKENIZER_MERGES; i += 1) {
        fprintf(w, "%u %u %u\n", merges[i].left, merges[i].right, 256u + i);
    }
    fclose(w);
    return file;
}

bool mda_test_tokenizer_vocab_ok(const mda_tokenizer_t *t, const mda_test_tokenizer_merge_t *merges) {
    uint32_t bad = 0;
    for (uint32_t b = 0; b < 256 && t->count == 256 + MDA_TEST_TOKENIZER_MERGES; b += 1) {
        bad += t->entries[b].len != 1 || (uint8_t)t->entries[b].bytes[0] != b || t->byte_id[b] != b;
    }
    for (uint32_t i = 0; i < MDA_TEST_TOKENIZER_MERGES && t->count == 256 + MDA_TEST_TOKENIZER_MERGES; i += 1) {
        mda_tokenizer_entry_t *e = &t->entries[256 + i];
        bad += e->len != strlen(merges[i].bytes) || memcmp(e->bytes, merges[i].bytes, e->len) != 0;
    }
    return t->count == 256 + MDA_TEST_TOKENIZER_MERGES && t->n_merges == MDA_TEST_TOKENIZER_MERGES && bad == 0;
}

bool mda_test_tokenizer_encode(const mda_tokenizer_t *t) {
    const char text[] = "the  the 12 ab's\n\nxAB'S(x \nx";
    uint32_t want[MDA_TEST_TOKENIZER_EXPECT] = {257, ' ', 258, ' ', '1', '2', ' ', 259, '\'', 's', '\n', '\n', 'x', 'A', 'B', 260, 261, 262, 'x'};
    mda_tokenizer_ids_t r = mda_tokenizer_encode(t, text, sizeof(text) - 1);
    bool same = r.n == MDA_TEST_TOKENIZER_EXPECT && memcmp(r.ids, want, sizeof(want)) == 0;
    printf("encode: %zu tokens, Qwen pre-tokenization and merge order %s\n", r.n, mda_util_verdict(same));
    free(r.ids);
    return same;
}

bool mda_test_tokenizer_unicode(void) {
    bool ok = mda_unicode_letter(0xe9) && mda_unicode_letter(0x4e2d) && mda_unicode_number(0x663) && mda_unicode_number(0xbd) && mda_unicode_mark(0x301) && mda_unicode_letter('1') == false && mda_unicode_number('a') == false && mda_unicode_letter('-') == false && mda_unicode_mark('a') == false;
    printf("unicode letter, number and mark classes: %s\n", mda_util_verdict(ok));
    return ok;
}

bool mda_test_tokenizer_main(void) {
    mda_test_tokenizer_merge_t merges[MDA_TEST_TOKENIZER_MERGES] = {
        {
            .bytes = "th",
            .left = 't',
            .right = 'h',
        },
        {
            .bytes = "the",
            .left = 256,
            .right = 'e',
        },
        {
            .bytes = " the",
            .left = ' ',
            .right = 257,
        },
        {
            .bytes = "ab",
            .left = 'a',
            .right = 'b',
        },
        {
            .bytes = "'S",
            .left = '\'',
            .right = 'S',
        },
        {
            .bytes = "(x",
            .left = '(',
            .right = 'x',
        },
        {
            .bytes = " \n",
            .left = ' ',
            .right = '\n',
        },
    };
    mda_test_tokenizer_file_t file = mda_test_tokenizer_file(merges);
    mda_tokenizer_t t = mda_tokenizer_read(fmemopen(file.buf, file.size, "rb"), "test tokenizer");
    bool vocab_ok = t.entries != NULL && mda_test_tokenizer_vocab_ok(&t, merges);
    printf("tokenizer file round trip: %s\n", mda_util_verdict(vocab_ok));
    bool encode_ok = vocab_ok && mda_test_tokenizer_encode(&t);
    bool unicode_ok = mda_test_tokenizer_unicode();
    free(file.buf);
    return vocab_ok && encode_ok && unicode_ok;
}

int main(void) {
    bool ok = mda_test_tokenizer_main();
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
