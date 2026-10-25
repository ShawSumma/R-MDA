#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check_qwen.h"
#include "vkc.h"
#include "util.h"
#include "tokenizer.h"
#include "qwen.h"

mda_tokenizer_ids_t mda_check_qwen_text(const mda_tokenizer_t *tok, const char *path) {
    FILE *f = fopen(path, "rb");
    char *text = malloc(MDA_CHECK_QWEN_TEXT);
    size_t got = 0;
    if (f == NULL) {
        perror(path);
    } else {
        got = fread(text, 1, MDA_CHECK_QWEN_TEXT, f);
        fclose(f);
    }
    mda_tokenizer_ids_t ids = mda_tokenizer_encode(tok, text, (uint32_t)got);
    free(text);
    return ids;
}

bool mda_check_qwen_take(void *ctx, mda_qwen_t *q, uint32_t row0, uint32_t rows) {
    mda_check_qwen_sink_t *s = ctx;
    float *l = mda_vkc_read(q->c, &q->sc.logits);
    memcpy(s->logits + (size_t)row0 * s->vocab, l, (size_t)rows * s->vocab * sizeof(float));
    free(l);
    return q->c->fail == false;
}

bool mda_check_qwen_write(const uint32_t *ids, const float *logits, uint32_t n, uint32_t vocab) {
    FILE *f = fopen(MDA_CHECK_QWEN_OUT, "wb");
    if (f == NULL) {
        perror(MDA_CHECK_QWEN_OUT);
        return false;
    } else {
        uint32_t hdr[2] = {n, vocab};
        size_t a = fwrite(hdr, sizeof(uint32_t), 2, f);
        size_t b = fwrite(ids, sizeof(uint32_t), n, f);
        size_t c = fwrite(logits, sizeof(float), (size_t)n * vocab, f);
        int32_t closed = fclose(f);
        return a == 2 && b == n && c == (size_t)n * vocab && closed == 0;
    }
}

bool mda_check_qwen_run(mda_vkc_t *c, const mda_tokenizer_ids_t *ids, const char *dir) {
    uint32_t n = MDA_CHECK_QWEN_LEN;
    mda_qwen_t *q = NULL;
    if (ids->n < n) {
        fprintf(stderr, "check_qwen: fewer than %u tokens of text\n", n);
    } else {
        q = mda_qwen_open(c, dir, n, n, n, n);
    }
    if (q == NULL) {
        return false;
    } else {
        mda_check_qwen_sink_t sink = {
            .logits = malloc((size_t)n * q->cfg.vocab * sizeof(float)),
            .vocab = q->cfg.vocab,
        };
        bool ok = mda_qwen_run(q, ids->ids, mda_check_qwen_take, &sink) && mda_check_qwen_write(ids->ids, sink.logits, n, sink.vocab);
        if (ok) {
            printf("%s: logits for %u tokens\n", MDA_CHECK_QWEN_OUT, n);
        }
        free(sink.logits);
        return ok;
    }
}

bool mda_check_qwen_main(uint32_t argc, char **argv) {
    const char *text = mda_util_arg(argc, argv, 1, MDA_TOKENIZER_CORPUS_PATH);
    const char *dir = mda_util_arg(argc, argv, 2, MDA_QWEN_DIR);
    mda_tokenizer_t tok = mda_tokenizer_load(MDA_QWEN_VOCAB_PATH);
    mda_vkc_t c = mda_vkc_init();
    if (tok.entries == NULL || c.fail) {
        return false;
    } else {
        mda_tokenizer_ids_t ids = mda_check_qwen_text(&tok, text);
        bool ok = mda_check_qwen_run(&c, &ids, dir);
        free(ids.ids);
        return ok;
    }
}

int main(int argc, char **argv) {
    bool ok = mda_check_qwen_main((uint32_t)argc, argv);
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
