#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check_teacher.h"
#include "vkc.h"
#include "util.h"
#include "tokenizer.h"
#include "data.h"
#include "model.h"
#include "teacher.h"
#include "import.h"

mda_tokenizer_ids_t mda_check_teacher_text(const mda_tokenizer_t *tok) {
    FILE *f = fopen(MDA_TOKENIZER_CORPUS_PATH, "rb");
    char *text = malloc(MDA_CHECK_TEACHER_TEXT);
    size_t got = 0;
    if (f == NULL) {
        perror(MDA_TOKENIZER_CORPUS_PATH);
    } else {
        got = fread(text, 1, MDA_CHECK_TEACHER_TEXT, f);
        fclose(f);
    }
    mda_tokenizer_ids_t ids = mda_tokenizer_encode(tok, text, (uint32_t)got);
    free(text);
    return ids;
}

bool mda_check_teacher_write(const uint32_t *ids, const float *logits, uint32_t n, uint32_t vocab) {
    FILE *f = fopen(MDA_CHECK_TEACHER_OUT, "wb");
    if (f == NULL) {
        perror(MDA_CHECK_TEACHER_OUT);
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

bool mda_check_teacher_run(mda_vkc_t *c, mda_model_t *m, const mda_tokenizer_ids_t *ids, const char *path) {
    uint32_t n = m->cfg.seqlen;
    mda_teacher_t *t = mda_teacher_load(m, path, n, false);
    mda_vkc_buf_t buf = mda_vkc_alloc_cached(c, n * sizeof(uint32_t));
    if (t == NULL || c->fail || ids->n < n) {
        fprintf(stderr, "check_teacher: no teacher, or fewer than %u tokens of text\n", n);
        return false;
    } else {
        uint32_t *p = buf.ptr;
        for (uint32_t i = 0; i < n; i += 1) {
            p[i] = ids->ids[i];
        }
        mda_vkc_begin(c);
        mda_teacher_rec(t, &buf);
        mda_vkc_flush(c);
        float *logits = mda_vkc_read(c, &t->logits);
        bool ok = c->fail == false && mda_check_teacher_write(ids->ids, logits, n, t->dims.vocab);
        free(logits);
        if (ok) {
            printf("%s: teacher logits for %u tokens\n", MDA_CHECK_TEACHER_OUT, n);
        }
        return ok;
    }
}

double mda_check_teacher_nll(const float *logits, const mda_data_batch_t *b, uint32_t vocab) {
    double nll = 0.0;
    for (uint32_t r = 0; r < b->batch * b->seqlen; r += 1) {
        const float *row = logits + (size_t)r * vocab;
        float mx = row[0];
        for (uint32_t v = 1; v < vocab; v += 1) {
            mx = fmaxf(mx, row[v]);
        }
        double se = 0.0;
        for (uint32_t v = 0; v < vocab; v += 1) {
            se += exp((double)row[v] - mx);
        }
        nll += (double)mx + log(se) - row[b->target[r]];
    }
    return nll;
}

bool mda_check_teacher_batches(mda_vkc_t *c, mda_teacher_t *t, const mda_data_corpus_t *cp, mda_data_batch_t *b, mda_vkc_buf_t *buf) {
    double nll = 0.0;
    double bytes = 0.0;
    for (uint32_t i = 0; i < MDA_CHECK_TEACHER_BATCHES && c->fail == false; i += 1) {
        mda_data_batch(b, cp, MDA_DATA_SPLIT_EVAL, MDA_DATA_SCHED_AR, i);
        memcpy(buf->ptr, b->ids, b->batch * b->seqlen * sizeof(uint32_t));
        mda_vkc_begin(c);
        mda_teacher_rec(t, buf);
        mda_vkc_flush(c);
        float *logits = mda_vkc_read(c, &t->logits);
        nll += mda_check_teacher_nll(logits, b, t->dims.vocab);
        bytes += mda_data_bytes(cp, b);
        free(logits);
    }
    if (c->fail == false) {
        printf("teacher held-out next-token NLL %.4f bits/byte over %u batches of %u sequences\n", nll / bytes / 0.6931472, MDA_CHECK_TEACHER_BATCHES, b->batch);
    }
    return c->fail == false;
}

bool mda_check_teacher_eval(mda_vkc_t *c, mda_model_t *m, const mda_data_corpus_t *cp, const char *path) {
    uint32_t n = MDA_CHECK_TEACHER_SEQS * m->cfg.seqlen;
    mda_teacher_t *t = mda_teacher_load(m, path, n, false);
    mda_vkc_buf_t buf = mda_vkc_alloc_cached(c, n * sizeof(uint32_t));
    if (t == NULL || c->fail) {
        return false;
    } else {
        mda_data_batch_t b = {
            .ids = malloc(n * sizeof(uint32_t)),
            .target = malloc(n * sizeof(uint32_t)),
            .mask = malloc(n * sizeof(float)),
            .weight = malloc(n * sizeof(float)),
            .batch = MDA_CHECK_TEACHER_SEQS,
            .seqlen = m->cfg.seqlen,
            .nclean = 0,
        };
        bool ok = mda_check_teacher_batches(c, t, cp, &b, &buf);
        free(b.ids);
        free(b.target);
        free(b.mask);
        free(b.weight);
        return ok;
    }
}

bool mda_check_teacher_main(uint32_t argc, char **argv) {
    const char *path = mda_util_arg(argc, argv, 1, MDA_IMPORT_WEIGHTS_PATH);
    mda_data_corpus_t cp = mda_data_load(MDA_TOKENIZER_CORPUS_PATH);
    mda_vkc_t c = mda_vkc_init();
    if (cp.ids == NULL || c.fail) {
        return false;
    } else {
        mda_model_cfg_t cfg = mda_model_cfg(cp.tok.count);
        cfg.batch = 1;
        mda_model_t *m = mda_model_init(&c, cfg);
        mda_tokenizer_ids_t ids = mda_check_teacher_text(&cp.tok);
        bool ok = m != NULL && mda_check_teacher_run(&c, m, &ids, path) && mda_check_teacher_eval(&c, m, &cp, path);
        free(ids.ids);
        return ok;
    }
}

int main(int argc, char **argv) {
    bool ok = mda_check_teacher_main((uint32_t)argc, argv);
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
