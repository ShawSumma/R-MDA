#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "generate.h"
#include "vkc.h"
#include "hash.h"
#include "util.h"
#include "sort.h"
#include "tokenizer.h"
#include "model.h"
#include "fwd.h"
#include "ckpt.h"

double mda_generate_repeat_rate(const uint32_t *s, uint32_t n) {
    uint32_t rep = 0;
    for (uint32_t i = 2; i < n; i += 1) {
        uint32_t j = 2;
        while (j < i && (s[j] != s[i] || s[j - 1] != s[i - 1] || s[j - 2] != s[i - 2])) {
            j += 1;
        }
        if (j < i) {
            rep += 1;
        }
    }
    return (double)rep / (n - 2);
}

uint32_t mda_generate_pick(uint64_t key, uint32_t i, uint32_t n, mda_generate_order_t order) {
    switch (order) {
    case MDA_GENERATE_LTR:
        return 0;
    default:
        return (uint32_t)(mda_hash_u64(MDA_HASH_GEN, key, 1 + i) % n);
    }
}

void mda_generate_reveal_order(uint32_t *ids, const uint32_t *tok, uint32_t *open, uint32_t n_open, uint32_t reveal, uint64_t key, uint32_t base, mda_generate_order_t order) {
    for (uint32_t r = 0; r < reveal; r += 1) {
        uint32_t j = r + mda_generate_pick(key, base + r, n_open - r, order);
        uint32_t p = open[j];
        open[j] = open[r];
        ids[p] = tok[p];
    }
}

void mda_generate_reveal_conf(uint32_t *ids, const uint32_t *tok, const float *conf, const uint32_t *open, uint32_t n_open, uint32_t reveal) {
    mda_sort_item_t *items = malloc(n_open * sizeof(mda_sort_item_t));
    for (uint32_t j = 0; j < n_open; j += 1) {
        items[j] = (mda_sort_item_t) {
            .key = -(double)conf[open[j]],
            .idx = open[j],
        };
    }
    mda_sort_items(items, n_open);
    for (uint32_t r = 0; r < reveal; r += 1) {
        ids[items[r].idx] = tok[items[r].idx];
    }
    free(items);
}

void mda_generate_reveal(uint32_t *ids, const uint32_t *tok, const float *conf, uint32_t *open, uint32_t n, uint32_t mask_id, uint64_t key, uint32_t remaining, uint32_t base, mda_generate_order_t order) {
    uint32_t n_open = 0;
    for (uint32_t p = 0; p < n; p += 1) {
        if (ids[p] == mask_id) {
            open[n_open] = p;
            n_open += 1;
        }
    }
    uint32_t reveal = (n_open + remaining - 1) / remaining;
    switch (order) {
    case MDA_GENERATE_CONF:
        mda_generate_reveal_conf(ids, tok, conf, open, n_open, reveal);
        break;
    default:
        mda_generate_reveal_order(ids, tok, open, n_open, reveal, key, base, order);
        break;
    }
}

void mda_generate_remask(uint32_t *ids, uint32_t n, uint32_t first, uint32_t mask_id, uint64_t key, uint32_t base, float eta) {
    for (uint32_t p = first; p < n; p += 1) {
        if (ids[p] != mask_id && mda_hash_unit(MDA_HASH_GEN, key, (1u << 24) + base + p) < eta) {
            ids[p] = mask_id;
        }
    }
}

uint32_t mda_generate_draw(const float *probs, uint32_t vocab, double u) {
    double acc = 0.0;
    uint32_t v = 0;
    while (v + 1 < vocab && acc + probs[v] <= u) {
        acc += probs[v];
        v += 1;
    }
    return v;
}

uint32_t mda_generate_residual(const float *p, const float *q, float *tmp, uint32_t vocab, double u) {
    double sum = 0.0;
    for (uint32_t v = 0; v < vocab; v += 1) {
        tmp[v] = fmaxf(p[v] - q[v], 0.0f);
        sum += tmp[v];
    }
    if (sum <= 0.0) {
        return mda_generate_draw(p, vocab, u);
    } else {
        for (uint32_t v = 0; v < vocab; v += 1) {
            tmp[v] = (float)(tmp[v] / sum);
        }
        return mda_generate_draw(tmp, vocab, u);
    }
}

void mda_generate_rows(mda_generate_ctx_t *g, float *dst, uint32_t row, uint32_t pos, uint32_t count) {
    mda_model_cfg_t cfg = g->m->cfg;
    const float *src = g->stage.ptr;
    memcpy(dst, src + ((size_t)row * cfg.seqlen + pos) * cfg.vocab, (size_t)count * cfg.vocab * sizeof(float));
}

void mda_generate_fill_spec(mda_generate_ctx_t *g, uint32_t lo) {
    uint32_t blk = g->m->cfg.seqlen;
    uint32_t *ids = g->sc->batch.ids;
    for (uint32_t s = 0; s < g->nseq; s += 1) {
        const mda_generate_spec_t *sp = &g->specs[s];
        const uint32_t *row = g->done + s * g->len;
        for (uint32_t p = 0; p < blk; p += 1) {
            uint32_t at = lo + p;
            uint32_t clean = g->mask_id;
            uint32_t noisy = g->mask_id;
            if (at < sp->n) {
                clean = row[at];
                noisy = row[at];
            } else if (at < sp->n + sp->k) {
                clean = sp->drafts[at - sp->n];
            }
            ids[s * blk + p] = clean;
            ids[(g->nseq + s) * blk + p] = noisy;
        }
    }
}

void mda_generate_fill_rows(mda_generate_ctx_t *g, uint32_t lo) {
    uint32_t blk = g->m->cfg.seqlen;
    uint32_t *ids = g->sc->batch.ids;
    for (uint32_t s = 0; s < g->nseq; s += 1) {
        const uint32_t *row = g->done + s * g->len + lo;
        for (uint32_t p = 0; p < blk; p += 1) {
            ids[s * blk + p] = row[p];
            ids[(g->nseq + s) * blk + p] = row[p];
        }
    }
}

uint32_t mda_generate_spec_accepts(const mda_generate_spec_t *sp, uint32_t vocab, uint64_t key, uint32_t s) {
    uint32_t j = 0;
    while (j < sp->k && mda_hash_unit(MDA_HASH_GEN, key, 3 * MDA_GENERATE_MAX_DRAFT * s + j) * sp->q[(size_t)j * vocab + sp->drafts[j]] < sp->p_rows[(size_t)j * vocab + sp->drafts[j]]) {
        j += 1;
    }
    return j;
}

void mda_generate_spec_redraft(mda_generate_ctx_t *g, mda_generate_spec_t *sp, uint32_t lo, uint32_t hi, uint64_t key, uint32_t s) {
    uint32_t vocab = g->m->cfg.vocab;
    uint32_t room = 0;
    if (sp->n < hi) {
        room = mda_util_min_u32(g->policy.draft, hi - sp->n);
    }
    if (room > 0) {
        mda_generate_rows(g, sp->q, g->nseq + s, sp->n - lo, room);
    }
    for (uint32_t i = 0; i < room; i += 1) {
        sp->drafts[i] = mda_generate_draw(sp->q + (size_t)i * vocab, vocab, mda_hash_unit(MDA_HASH_GEN, key, 3 * MDA_GENERATE_MAX_DRAFT * s + MDA_GENERATE_MAX_DRAFT + i));
    }
    sp->k = room;
}

void mda_generate_spec_step(mda_generate_ctx_t *g, uint32_t s, uint32_t lo, uint64_t key) {
    uint32_t vocab = g->m->cfg.vocab;
    uint32_t hi = lo + g->m->cfg.seqlen;
    mda_generate_spec_t *sp = &g->specs[s];
    uint32_t *row = g->done + s * g->len;
    uint32_t n = sp->n;
    mda_generate_rows(g, sp->p_rows, s, n - 1 - lo, sp->k + 1);
    uint32_t a = mda_generate_spec_accepts(sp, vocab, key, s);
    double u = mda_hash_unit(MDA_HASH_GEN, key, 3 * MDA_GENERATE_MAX_DRAFT * s + 2 * MDA_GENERATE_MAX_DRAFT);
    for (uint32_t j = 0; j < a; j += 1) {
        row[n + j] = sp->drafts[j];
    }
    uint32_t fix = 0;
    if (a < sp->k) {
        fix = mda_generate_residual(sp->p_rows + (size_t)a * vocab, sp->q + (size_t)a * vocab, g->tmp, vocab, u);
    } else {
        fix = mda_generate_draw(sp->p_rows + (size_t)a * vocab, vocab, u);
    }
    if (n + a < g->len) {
        row[n + a] = fix;
    }
    sp->accepted += a;
    sp->proposed += sp->k;
    sp->n = mda_util_min_u32(n + a + 1, g->len);
    mda_generate_spec_redraft(g, sp, lo, hi, key, s);
}

uint32_t mda_generate_spec_pending(const mda_generate_ctx_t *g, uint32_t hi) {
    uint32_t pending = 0;
    for (uint32_t s = 0; s < g->nseq; s += 1) {
        pending += g->specs[s].n < hi;
    }
    return pending;
}

uint64_t mda_generate_key(uint32_t round, uint32_t lo, uint32_t step) {
    return (uint64_t)round << 32 | (uint64_t)lo << 12 | step;
}

uint32_t mda_generate_block_spec(mda_generate_ctx_t *g, uint32_t lo, uint32_t round) {
    uint32_t hi = lo + g->m->cfg.seqlen;
    uint32_t forwards = 0;
    while (mda_generate_spec_pending(g, hi) > 0 && g->m->c->fail == false) {
        uint64_t key = mda_generate_key(round, lo, forwards);
        mda_generate_fill_spec(g, lo);
        mda_fwd_run(g->m, g->sc);
        for (uint32_t s = 0; s < g->nseq; s += 1) {
            if (g->specs[s].n < hi) {
                mda_generate_spec_step(g, s, lo, key);
            }
        }
        forwards += 1;
    }
    return forwards;
}

uint32_t mda_generate_block_fixed(mda_generate_ctx_t *g, uint32_t lo, uint32_t round) {
    uint32_t blk = g->m->cfg.seqlen;
    for (uint32_t step = 0; step < g->steps && g->m->c->fail == false; step += 1) {
        uint64_t key = mda_generate_key(round, lo, step);
        mda_generate_fill_rows(g, lo);
        g->sc->seed = (uint32_t)mda_hash_u64(MDA_HASH_GEN, key, 0);
        mda_fwd_run(g->m, g->sc);
        uint32_t *tok = g->sc->tok.ptr;
        float *conf = g->sc->conf.ptr;
        for (uint32_t s = 0; s < g->nseq; s += 1) {
            uint32_t *row = g->done + s * g->len + lo;
            mda_generate_reveal(row, tok + (g->nseq + s) * blk, conf + (g->nseq + s) * blk, g->open, blk, g->mask_id, key, g->steps - step, s * blk, g->policy.order);
            if (step + 1 < g->steps) {
                mda_generate_remask(row, blk, (uint32_t)(lo == 0), g->mask_id, key, s * blk, g->policy.remask);
            }
        }
    }
    return g->steps;
}

void mda_generate_next_first(mda_generate_ctx_t *g, uint32_t lo, uint32_t round) {
    uint32_t blk = g->m->cfg.seqlen;
    uint32_t vocab = g->m->cfg.vocab;
    uint64_t key = mda_generate_key(round, lo, 4095);
    for (uint32_t s = 0; s < g->nseq; s += 1) {
        mda_generate_spec_t *sp = &g->specs[s];
        if (sp->n == lo + blk) {
            mda_generate_rows(g, sp->p_rows, s, blk - 1, 1);
            g->done[s * g->len + lo + blk] = mda_generate_draw(sp->p_rows, vocab, mda_hash_unit(MDA_HASH_GEN, key, s));
            sp->n = lo + blk + 1;
        }
        sp->k = 0;
    }
}

uint32_t mda_generate_advance(mda_generate_ctx_t *g, uint32_t lo, uint32_t round) {
    mda_generate_fill_rows(g, lo);
    mda_fwd_run(g->m, g->sc);
    mda_fwd_cache_commit(g->cache);
    if (g->policy.order == MDA_GENERATE_SPEC) {
        mda_generate_next_first(g, lo, round);
    }
    return 1;
}

void mda_generate_start(mda_generate_ctx_t *g) {
    for (uint32_t i = 0; i < g->nseq * g->len; i += 1) {
        g->done[i] = g->mask_id;
    }
    for (uint32_t s = 0; s < g->nseq; s += 1) {
        g->done[s * g->len] = g->newline;
        g->specs[s].n = 1;
        g->specs[s].k = 0;
    }
}

uint32_t mda_generate_round_cached(mda_generate_ctx_t *g, uint32_t round) {
    uint32_t blk = g->m->cfg.seqlen;
    uint32_t forwards = 0;
    mda_generate_start(g);
    mda_fwd_cache_reset(g->m, g->cache);
    for (uint32_t lo = 0; lo < g->len && g->m->c->fail == false; lo += blk) {
        if (g->policy.order == MDA_GENERATE_SPEC) {
            forwards += mda_generate_block_spec(g, lo, round);
        } else {
            forwards += mda_generate_block_fixed(g, lo, round);
        }
        if (lo + blk < g->len) {
            forwards += mda_generate_advance(g, lo, round);
        }
    }
    return forwards;
}

uint32_t mda_generate_round_plain(mda_generate_ctx_t *g, uint32_t round) {
    mda_generate_start(g);
    for (uint32_t step = 0; step < g->steps && g->m->c->fail == false; step += 1) {
        uint64_t key = mda_generate_key(round, 0, step);
        memcpy(g->sc->batch.ids, g->done, g->nseq * g->len * sizeof(uint32_t));
        g->sc->seed = (uint32_t)mda_hash_u64(MDA_HASH_GEN, key, 0);
        mda_fwd_run(g->m, g->sc);
        uint32_t *tok = g->sc->tok.ptr;
        float *conf = g->sc->conf.ptr;
        for (uint32_t s = 0; s < g->nseq; s += 1) {
            uint32_t *row = g->done + s * g->len;
            mda_generate_reveal(row, tok + s * g->len, conf + s * g->len, g->open, g->len, g->mask_id, key, g->steps - step, s * g->len, g->policy.order);
            if (step + 1 < g->steps) {
                mda_generate_remask(row, g->len, 1, g->mask_id, key, s * g->len, g->policy.remask);
            }
        }
    }
    return g->steps;
}

void mda_generate_judge(mda_model_t *j, mda_fwd_scratch_t *js, const mda_tokenizer_t *tk, const uint32_t *gen, mda_generate_stats_t *st) {
    mda_model_cfg_t cfg = j->cfg;
    mda_data_batch_t *b = &js->batch;
    for (uint32_t i = 0; i < cfg.batch * cfg.seqlen; i += 1) {
        uint32_t p = i % cfg.seqlen;
        b->ids[i] = gen[i];
        b->weight[i] = 1.0f;
        if (p + 1 < cfg.seqlen) {
            b->target[i] = gen[i + 1];
            b->mask[i] = 1.0f;
        } else {
            b->target[i] = gen[i];
            b->mask[i] = 0.0f;
        }
    }
    mda_fwd_run(j, js);
    float *loss = js->loss.ptr;
    for (uint32_t i = 0; i < cfg.batch * cfg.seqlen && j->c->fail == false; i += 1) {
        st->nll += loss[i];
        st->nll_tokens += b->mask[i];
        st->nll_bytes += b->mask[i] * tk->entries[b->target[i]].len;
    }
}

void mda_generate_count(const mda_generate_ctx_t *g, mda_generate_stats_t *st) {
    for (uint32_t s = 0; s < g->nseq; s += 1) {
        st->rep += mda_generate_repeat_rate(g->done + s * g->len, g->len);
        st->seqs += 1;
        st->accepted += g->specs[s].accepted;
        st->proposed += g->specs[s].proposed;
    }
    for (uint32_t i = 0; i < g->nseq * g->len; i += 1) {
        st->counts[g->done[i]] += 1;
        st->tokens += 1;
    }
}

void mda_generate_print_text(const mda_tokenizer_t *tk, const mda_generate_ctx_t *g) {
    for (uint32_t i = 0; i < g->nseq * g->len; i += 1) {
        mda_tokenizer_entry_t *e = &tk->entries[g->done[i]];
        fwrite(e->bytes, 1, e->len, stdout);
        if (i % g->len == g->len - 1) {
            printf("\n---\n");
        }
    }
}

double mda_generate_entropy(const mda_generate_stats_t *st, uint32_t vocab) {
    double h = 0.0;
    for (uint32_t v = 0; v < vocab; v += 1) {
        if (st->counts[v] > 0) {
            double q = (double)st->counts[v] / st->tokens;
            h -= q * log2(q);
        }
    }
    return h;
}

void mda_generate_summary(const mda_generate_stats_t *st, uint32_t vocab, float temp) {
    double per_round = (double)st->forwards / MDA_GENERATE_ROUNDS;
    double per_seq = (double)st->tokens / st->seqs;
    printf("%u sequences, temp %g, %.1f forwards per sequence, %.2f tokens per forward, %.0f tok/s\n", st->seqs, temp, per_round, per_seq / per_round, st->tokens / st->seconds);
    printf("repeated trigrams %.1f%%, unigram entropy %.2f bits\n", 100.0 * st->rep / st->seqs, mda_generate_entropy(st, vocab));
    if (st->proposed > 0) {
        printf("speculation: %u of %u drafted tokens accepted (%.1f%%)\n", st->accepted, st->proposed, 100.0 * st->accepted / st->proposed);
    }
    if (st->nll_tokens > 0.0) {
        printf("judge: generative perplexity %.1f, %.4f bits/byte\n", exp(st->nll / st->nll_tokens), st->nll / st->nll_bytes / 0.6931472);
    }
}

uint32_t mda_generate_round(mda_generate_ctx_t *g, uint32_t round) {
    for (uint32_t s = 0; s < g->nseq; s += 1) {
        g->specs[s].accepted = 0;
        g->specs[s].proposed = 0;
    }
    if (g->cache != NULL) {
        return mda_generate_round_cached(g, round);
    } else {
        return mda_generate_round_plain(g, round);
    }
}

bool mda_generate_rounds(mda_generate_ctx_t *g, mda_model_t *j, mda_fwd_scratch_t *js, const mda_tokenizer_t *tk) {
    mda_generate_stats_t st = {
        .counts = calloc(g->m->cfg.vocab, sizeof(uint32_t)),
    };
    for (uint32_t r = 0; r < MDA_GENERATE_ROUNDS && g->m->c->fail == false; r += 1) {
        double t0 = mda_util_now();
        st.forwards += mda_generate_round(g, r);
        st.seconds += mda_util_now() - t0;
        mda_generate_count(g, &st);
        if (r == 0) {
            mda_generate_print_text(tk, g);
        }
        if (j != NULL) {
            mda_generate_judge(j, js, tk, g->done, &st);
        }
    }
    bool ok = g->m->c->fail == false;
    if (ok) {
        mda_generate_summary(&st, g->m->cfg.vocab, g->sc->temp);
    }
    free(st.counts);
    return ok;
}

uint32_t mda_generate_newline(const mda_tokenizer_t *tk) {
    uint32_t id = 0;
    for (uint32_t v = 0; v < tk->count; v += 1) {
        if (tk->entries[v].len == 1 && tk->entries[v].bytes[0] == '\n') {
            id = v;
        }
    }
    return id;
}

mda_model_t *mda_generate_load(mda_vkc_t *c, const mda_tokenizer_t *tk, const char *path, mda_model_cfg_t cfg, bool causal, uint32_t dec) {
    mda_model_t *m = mda_model_init(c, mda_ckpt_shape(path, cfg));
    if (m == NULL || mda_model_set_dec(m, dec) == false) {
        return NULL;
    } else {
        m->causal = causal;
        mda_ckpt_res_t loaded = mda_ckpt_load(m, mda_tokenizer_hash(tk), path, false);
        if (loaded.ok == false) {
            return NULL;
        } else {
            fprintf(stderr, "%s: step %u\n", path, loaded.step);
            return m;
        }
    }
}

mda_model_cfg_t mda_generate_cfg(uint32_t vocab, const char *mode) {
    mda_model_cfg_t base = mda_model_cfg(vocab);
    uint32_t block = mda_model_block_of(mode);
    if (strcmp(mode, "ar") == 0) {
        block = MDA_MODEL_BLOCK;
    }
    if (block > 0 && mda_model_dec_of(mode) > 0) {
        base.seqlen = block;
        base.batch = mda_util_max_u32(base.batch, 64 / block);
        return mda_model_cfg_block(base, block);
    } else if (block > 0) {
        base.seqlen = block;
        return mda_model_cfg_block(base, block);
    } else {
        return base;
    }
}

bool mda_generate_judged(mda_vkc_t *c, const mda_tokenizer_t *tk, mda_generate_ctx_t *g, const char *judge) {
    if (strlen(judge) == 0) {
        return mda_generate_rounds(g, NULL, NULL, tk);
    } else {
        mda_model_t *j = mda_generate_load(c, tk, judge, mda_model_cfg(tk->count), true, 0);
        if (j == NULL) {
            return false;
        } else {
            mda_fwd_scratch_t js = mda_fwd_alloc(j, 1);
            return mda_generate_rounds(g, j, &js, tk);
        }
    }
}

bool mda_generate_with(mda_vkc_t *c, const mda_tokenizer_t *tk, mda_model_t *m, const char *judge, uint32_t steps, float temp, mda_generate_policy_t policy) {
    m->infer = true;
    bool cached = m->cfg.block > 0;
    mda_fwd_scratch_t sc = mda_fwd_alloc(m, 1);
    mda_fwd_cache_t cache = { .cur = 0 };
    sc.temp = temp;
    uint32_t vocab = m->cfg.vocab;
    uint32_t rows = m->cfg.seqlen;
    mda_generate_ctx_t g = {
        .m = m,
        .sc = &sc,
        .cache = NULL,
        .policy = policy,
        .len = mda_model_cfg(vocab).seqlen,
        .nseq = m->cfg.batch,
        .mask_id = m->cfg.vocab,
        .newline = mda_generate_newline(tk),
        .steps = steps,
    };
    if (cached) {
        cache = mda_fwd_cache_alloc(m);
        sc.cache = &cache;
        g.cache = &cache;
        g.stage = mda_vkc_alloc_cached(c, sc.logits.sz);
        sc.stage = &g.stage;
        g.nseq = mda_model_nclean(m->cfg);
    }
    g.specs = calloc(g.nseq, sizeof(mda_generate_spec_t));
    for (uint32_t s = 0; s < g.nseq; s += 1) {
        g.specs[s].q = malloc((size_t)MDA_GENERATE_MAX_DRAFT * vocab * sizeof(float));
        g.specs[s].p_rows = malloc((size_t)rows * vocab * sizeof(float));
    }
    g.done = malloc(g.nseq * g.len * sizeof(uint32_t));
    g.open = malloc(g.len * sizeof(uint32_t));
    g.tmp = malloc(vocab * sizeof(float));
    return mda_generate_judged(c, tk, &g, judge);
}

mda_generate_policy_t mda_generate_policy(const char *name, const char *mode) {
    mda_generate_policy_t p = {
        .order = MDA_GENERATE_RANDOM,
        .draft = 0,
        .remask = 0.0f,
    };
    if (strcmp(mode, "ar") == 0) {
        p.order = MDA_GENERATE_SPEC;
    } else if (strcmp(name, "ltr") == 0) {
        p.order = MDA_GENERATE_LTR;
    } else if (strcmp(name, "conf") == 0) {
        p.order = MDA_GENERATE_CONF;
    } else if (strncmp(name, "remdm", 5) == 0) {
        p.remask = strtof(name + 5, NULL);
    } else if (strncmp(name, "spec", 4) == 0) {
        p.order = MDA_GENERATE_SPEC;
        p.draft = mda_util_min_u32((uint32_t)strtoul(name + 4, NULL, 10), MDA_GENERATE_MAX_DRAFT);
    }
    return p;
}

bool mda_generate_device(mda_vkc_t *c, const char *path, const char *judge, const char *mode, uint32_t steps, float temp, mda_generate_policy_t policy) {
    mda_tokenizer_t tk = mda_tokenizer_load(MDA_TOKENIZER_VOCAB_PATH);
    if (tk.entries == NULL) {
        return false;
    } else if (policy.order == MDA_GENERATE_SPEC && mda_model_block_of(mode) == 0 && strcmp(mode, "ar") != 0) {
        fprintf(stderr, "generate: spec needs block or ar mode\n");
        return false;
    } else {
        mda_model_t *m = mda_generate_load(c, &tk, path, mda_generate_cfg(tk.count, mode), false, mda_model_dec_of(mode));
        if (m == NULL) {
            return false;
        } else {
            return mda_generate_with(c, &tk, m, judge, steps, temp, policy);
        }
    }
}

bool mda_generate_main(uint32_t argc, char **argv) {
    uint32_t steps = (uint32_t)strtoul(mda_util_arg(argc, argv, 1, "16"), NULL, 10);
    const char *path = mda_util_arg(argc, argv, 2, MDA_CKPT_PATH);
    float temp = strtof(mda_util_arg(argc, argv, 3, "0.8"), NULL);
    const char *mode = mda_util_arg(argc, argv, 4, "diffusion");
    const char *judge = mda_util_arg(argc, argv, 5, "");
    mda_generate_policy_t policy = mda_generate_policy(mda_util_arg(argc, argv, 6, "random"), mode);
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_generate_device(&c, path, judge, mode, steps, temp, policy);
    }
}

int main(int argc, char **argv) {
    bool ok = mda_generate_main((uint32_t)argc, argv);
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
