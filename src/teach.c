#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "teach.h"
#include "util.h"
#include "pc.h"
#include "tokenizer.h"
#include "model.h"

uint32_t *mda_teach_off(mda_teach_t *t, uint32_t j) {
    return t->tok_off + (size_t)j * (t->q->seqlen + 1);
}

uint32_t *mda_teach_rec(mda_teach_t *t, uint32_t j, uint32_t x) {
    return t->recs + ((size_t)j * MDA_TEACH_MAX_BYTES + x) * MDA_KD_K;
}

void mda_teach_tokens(mda_teach_t *t, uint32_t j) {
    uint32_t T = t->q->seqlen;
    mda_kd_window_t *w = &t->wins[j];
    mda_tokenizer_ids_t r = mda_tokenizer_encode(t->qt, (const char *)w->bytes, w->n_bytes);
    uint32_t n = (uint32_t)mda_util_min_size(r.n, T);
    uint32_t *off = mda_teach_off(t, j);
    off[0] = 0;
    for (uint32_t i = 0; i < T; i += 1) {
        uint32_t id = t->q->cfg.eos;
        if (i < n) {
            id = r.ids[i];
            off[i + 1] = off[i] + t->qt->entries[id].len;
        }
        t->ids[(size_t)j * T + i] = id;
    }
    t->n_tok[j] = n;
    free(r.ids);
}

void mda_teach_window(mda_teach_t *t, uint64_t w, uint32_t j) {
    mda_kd_hdr_t *h = &t->hdr;
    uint32_t per = h->accum * h->nclean;
    uint32_t s = h->from + (uint32_t)(w / per);
    uint32_t micro = (uint32_t)((w / h->nclean) % h->accum);
    uint32_t row = (uint32_t)(w % h->nclean);
    mda_data_batch(&t->batch, t->cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL, (uint64_t)(s - 1) * h->accum + micro + 1);
    const uint32_t *ids = t->batch.ids + (size_t)row * h->seqlen;
    uint32_t last = t->batch.target[(size_t)row * h->seqlen + h->seqlen - 1];
    mda_kd_window_fill(&t->wins[j], &t->cp->tok, ids, h->seqlen, last);
    mda_teach_tokens(t, j);
    memset(mda_teach_rec(t, j, 0), 0, (size_t)MDA_TEACH_MAX_BYTES * MDA_KD_K * sizeof(uint32_t));
}

void mda_teach_blank(mda_teach_t *t, uint32_t j) {
    uint32_t T = t->q->seqlen;
    for (uint32_t i = 0; i < T; i += 1) {
        t->ids[(size_t)j * T + i] = t->q->cfg.eos;
    }
    t->n_tok[j] = 0;
}

uint32_t mda_teach_depths(mda_teach_t *t, uint32_t j, uint32_t i) {
    uint32_t *off = mda_teach_off(t, j);
    if (i + 1 < t->n_tok[j]) {
        return mda_util_min_u32(off[i + 2] - off[i + 1], MDA_KD_DEPTH);
    } else {
        return 0;
    }
}

void mda_teach_conts(mda_teach_t *t, uint32_t row0, uint32_t rows) {
    uint32_t T = t->q->seqlen;
    uint32_t *cont = t->cont.ptr;
    for (uint32_t r = 0; r < rows; r += 1) {
        uint32_t j = (row0 + r) / T;
        uint32_t i = (row0 + r) % T;
        uint32_t nd = mda_teach_depths(t, j, i);
        uint32_t at = 0;
        if (nd > 0) {
            at = mda_teach_off(t, j)[i + 1];
        }
        mda_kd_cont(cont + (size_t)r * 5, t->wins[j].bytes, t->wins[j].n_bytes, at, nd);
    }
}

void mda_teach_collect(mda_teach_t *t, uint32_t row0, uint32_t rows) {
    uint32_t T = t->q->seqlen;
    const uint32_t *o = t->outp.ptr;
    for (uint32_t r = 0; r < rows; r += 1) {
        uint32_t j = (row0 + r) / T;
        uint32_t i = (row0 + r) % T;
        uint32_t nd = mda_teach_depths(t, j, i);
        for (uint32_t l = 0; l < nd; l += 1) {
            uint32_t x = mda_teach_off(t, j)[i + 1] + l;
            const uint32_t *e = o + ((size_t)r * MDA_KD_DEPTH + l) * MDA_KD_K;
            float p = mda_kd_prob(e[0]);
            if (x < MDA_TEACH_MAX_BYTES && j < t->n_win) {
                memcpy(mda_teach_rec(t, j, x), e, MDA_KD_K * sizeof(uint32_t));
            }
            if (p > 0.0f && j < t->n_win) {
                t->nll_bits -= log2(p);
                t->nll_bytes += 1.0;
            }
        }
    }
}

bool mda_teach_sink(void *ctx, mda_qwen_t *q, uint32_t row0, uint32_t rows) {
    mda_teach_t *t = ctx;
    mda_teach_conts(t, row0, rows);
    mda_pc_kd_bytes_t pc = {
        .rows = rows,
        .vocab = q->cfg.vocab,
    };
    mda_vkc_buf_t *bufs[4] = {
        &q->sc.logits,
        &t->vocab,
        &t->cont,
        &t->outp,
    };
    mda_vkc_run(q->c, &t->k_kd, bufs, rows, 1, 1, &pc);
    if (q->c->fail == false) {
        mda_teach_collect(t, row0, rows);
    }
    return q->c->fail == false;
}

bool mda_teach_write(mda_teach_t *t) {
    uint32_t bad = 0;
    for (uint32_t j = 0; j < t->n_win; j += 1) {
        uint32_t n = mda_util_min_u32(t->wins[j].n_bytes, MDA_TEACH_MAX_BYTES);
        t->index[t->w0 + j] = t->at;
        bad += fwrite(&n, sizeof(uint32_t), 1, t->out) != 1;
        bad += fwrite(mda_teach_rec(t, j, 0), sizeof(uint32_t) * MDA_KD_K, n, t->out) != n;
        t->at += sizeof(uint32_t) + (uint64_t)n * MDA_KD_K * sizeof(uint32_t);
    }
    return bad == 0;
}

bool mda_teach_pass(mda_teach_t *t, uint64_t w0, uint32_t n_win, uint32_t windows) {
    t->w0 = w0;
    t->n_win = n_win;
    uint64_t toks = 0;
    uint32_t full = 0;
    for (uint32_t j = 0; j < windows; j += 1) {
        if (j < n_win) {
            mda_teach_window(t, w0 + j, j);
            toks += t->n_tok[j];
            full += t->n_tok[j] == t->q->seqlen;
        } else {
            mda_teach_blank(t, j);
        }
    }
    fprintf(stderr, "teach: windows %llu..%llu, %.1f teacher tokens each, %u filled all %u\n", (unsigned long long)w0, (unsigned long long)(w0 + n_win), (double)toks / n_win, full, t->q->seqlen);
    bool ran = mda_qwen_run(t->q, t->ids, mda_teach_sink, t);
    return ran && mda_teach_write(t);
}

bool mda_teach_finish(mda_teach_t *t) {
    int32_t sought = fseek(t->out, sizeof(mda_kd_hdr_t), SEEK_SET);
    size_t wrote = fwrite(t->index, sizeof(uint64_t), t->hdr.n_windows, t->out);
    int32_t closed = fclose(t->out);
    return sought == 0 && wrote == t->hdr.n_windows && closed == 0;
}

bool mda_teach_all(mda_teach_t *t, uint32_t windows) {
    double t0 = mda_util_now();
    uint64_t n = t->hdr.n_windows;
    uint32_t bad = 0;
    for (uint64_t w0 = 0; w0 < n && bad == 0; w0 += windows) {
        uint32_t k = (uint32_t)mda_util_min_size(windows, n - w0);
        bool passed = mda_teach_pass(t, w0, k, windows);
        bad += passed == false;
        double el = mda_util_now() - t0;
        fprintf(stderr, "teach: %llu of %llu windows, teacher %.4f bits/byte, %.0f s, %.0f s left\n", (unsigned long long)(w0 + k), (unsigned long long)n, t->nll_bits / fmax(t->nll_bytes, 1.0), el, el / (double)(w0 + k) * (double)(n - w0 - k));
    }
    return bad == 0 && mda_teach_finish(t);
}

bool mda_teach_open_out(mda_teach_t *t, const char *path) {
    t->out = fopen(path, "wb");
    if (t->out == NULL) {
        perror(path);
        return false;
    } else {
        size_t a = fwrite(&t->hdr, sizeof(mda_kd_hdr_t), 1, t->out);
        size_t b = fwrite(t->index, sizeof(uint64_t), t->hdr.n_windows, t->out);
        t->at = sizeof(mda_kd_hdr_t) + t->hdr.n_windows * sizeof(uint64_t);
        return a == 1 && b == t->hdr.n_windows;
    }
}

mda_data_batch_t mda_teach_batch(mda_model_cfg_t cfg) {
    size_t n = (size_t)cfg.batch * cfg.seqlen;
    return (mda_data_batch_t) {
        .ids = malloc(n * sizeof(uint32_t)),
        .target = malloc(n * sizeof(uint32_t)),
        .mask = malloc(n * sizeof(float)),
        .weight = malloc(n * sizeof(float)),
        .batch = cfg.batch,
        .seqlen = cfg.seqlen,
        .nclean = mda_model_nclean(cfg),
    };
}

bool mda_teach_buffers(mda_teach_t *t, uint32_t windows) {
    mda_vkc_t *c = t->q->c;
    uint32_t hr = t->q->head_rows;
    uint32_t vocab = t->q->cfg.vocab;
    uint32_t *table = mda_kd_vocab(t->qt, vocab, 0, MDA_KD_QWEN_SPECIAL);
    t->vocab = mda_vkc_alloc(c, (VkDeviceSize)vocab * 5 * sizeof(uint32_t));
    mda_vkc_write(c, &t->vocab, table);
    free(table);
    t->cont = mda_vkc_alloc_cached(c, (VkDeviceSize)hr * 5 * sizeof(uint32_t));
    t->outp = mda_vkc_alloc_cached(c, (VkDeviceSize)hr * MDA_KD_ENTRY * sizeof(uint32_t));
    t->k_kd = mda_vkc_load(c, "build/kd_bytes.spv", 4, sizeof(mda_pc_kd_bytes_t));
    t->wins = malloc(windows * sizeof(mda_kd_window_t));
    for (uint32_t j = 0; j < windows; j += 1) {
        t->wins[j] = mda_kd_window_alloc(t->hdr.seqlen);
    }
    t->tok_off = calloc((size_t)windows * (t->q->seqlen + 1), sizeof(uint32_t));
    t->n_tok = calloc(windows, sizeof(uint32_t));
    t->ids = calloc((size_t)windows * t->q->seqlen, sizeof(uint32_t));
    t->recs = calloc((size_t)windows * MDA_TEACH_MAX_BYTES * MDA_KD_K, sizeof(uint32_t));
    t->index = calloc(t->hdr.n_windows, sizeof(uint64_t));
    return c->fail == false;
}

bool mda_teach_run(mda_vkc_t *c, const mda_data_corpus_t *cp, const mda_tokenizer_t *qt, mda_teach_args_t a) {
    mda_model_cfg_t cfg = mda_model_cfg_mode(cp->tok.count, a.mode, a.batch);
    uint32_t nclean = mda_model_nclean(cfg);
    uint32_t cw = mda_util_min_u32(a.windows, MDA_TEACH_CHUNK_WINDOWS);
    uint32_t hw = mda_util_min_u32(a.windows, MDA_TEACH_HEAD_WINDOWS);
    if (nclean == 0 || a.to <= a.from || a.from == 0 || a.windows == 0) {
        fprintf(stderr, "teach: needs a block mode (clean rows), steps 1 <= from < to, and windows > 0\n");
        return false;
    } else {
        mda_teach_t t;
        memset(&t, 0, sizeof(t));
        t.cp = cp;
        t.qt = qt;
        t.batch = mda_teach_batch(cfg);
        t.hdr = (mda_kd_hdr_t) {
            .magic = MDA_KD_MAGIC,
            .n_windows = (uint64_t)(a.to - a.from) * a.accum * nclean,
            .from = a.from,
            .to = a.to,
            .accum = a.accum,
            .nclean = nclean,
            .seqlen = cfg.seqlen,
            .depth = MDA_KD_DEPTH,
            .k = MDA_KD_K,
            .batch = cfg.batch,
        };
        t.q = mda_qwen_open(c, MDA_QWEN_DIR, a.windows * MDA_TEACH_SEQ, cw * MDA_TEACH_SEQ, MDA_TEACH_SEQ, hw * MDA_TEACH_SEQ);
        bool ok = t.q != NULL && mda_teach_buffers(&t, a.windows) && mda_teach_open_out(&t, a.out) && mda_teach_all(&t, a.windows);
        if (ok) {
            fprintf(stderr, "%s: KD targets for steps %u..%u, %llu windows\n", a.out, a.from, a.to, (unsigned long long)t.hdr.n_windows);
        }
        return ok;
    }
}

bool mda_teach_main(uint32_t argc, char **argv) {
    mda_teach_args_t a = {
        .from = (uint32_t)strtoul(mda_util_arg(argc, argv, 1, "1"), NULL, 10),
        .to = (uint32_t)strtoul(mda_util_arg(argc, argv, 2, "2"), NULL, 10),
        .out = mda_util_arg(argc, argv, 3, MDA_KD_PATH),
        .windows = (uint32_t)strtoul(mda_util_arg(argc, argv, 4, "32"), NULL, 10),
        .mode = mda_util_arg(argc, argv, 5, "block16"),
        .batch = (uint32_t)strtoul(mda_util_arg(argc, argv, 6, "2"), NULL, 10),
        .accum = (uint32_t)strtoul(mda_util_arg(argc, argv, 7, "8"), NULL, 10),
    };
    mda_data_corpus_t cp = mda_data_load(MDA_TOKENIZER_CORPUS_PATH);
    mda_tokenizer_t qt = mda_tokenizer_load(MDA_QWEN_VOCAB_PATH);
    mda_vkc_t c = mda_vkc_init();
    if (cp.ids == NULL || qt.entries == NULL || c.fail) {
        return false;
    } else {
        return mda_teach_run(&c, &cp, &qt, a);
    }
}

int main(int argc, char **argv) {
    bool ok = mda_teach_main((uint32_t)argc, argv);
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
