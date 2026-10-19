#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vkc.h"
#include "hash.h"
#include "util.h"
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

void mda_generate_reveal(uint32_t *ids, const uint32_t *tok, uint32_t *open, uint32_t seqlen, uint32_t mask_id, uint32_t s, uint32_t remaining, uint32_t base) {
    uint32_t n_open = 0;
    for (uint32_t p = 0; p < seqlen; p += 1) {
        if (ids[p] == mask_id) {
            open[n_open] = p;
            n_open += 1;
        }
    }
    uint32_t reveal = (n_open + remaining - 1) / remaining;
    for (uint32_t r = 0; r < reveal; r += 1) {
        uint32_t j = r + (uint32_t)(mda_hash_u64(MDA_HASH_GEN, s, 1 + base + r) % (n_open - r));
        uint32_t p = open[j];
        open[j] = open[r];
        ids[p] = tok[p];
    }
}

void mda_generate_weights(const uint32_t *ids, float *weight, uint32_t seqlen, uint32_t mask_id) {
    uint32_t masked = 0;
    for (uint32_t p = 0; p < seqlen; p += 1) {
        masked += ids[p] == mask_id;
    }
    for (uint32_t p = 0; p < seqlen; p += 1) {
        weight[p] = (float)seqlen / (float)mda_util_max_u32(masked, 1);
    }
}

void mda_generate_print(const mda_tokenizer_t *tk, const uint32_t *ids, mda_model_cfg_t cfg, uint32_t steps, float temp, double dt) {
    uint32_t bt = cfg.batch * cfg.seqlen;
    printf("generated: %u steps, temp %g, %.1f ms, %.0f tok/s\n", steps, temp, dt * 1e3, bt / dt);
    double rep = 0.0;
    for (uint32_t b = 0; b < cfg.batch; b += 1) {
        const uint32_t *row = ids + b * cfg.seqlen;
        rep += mda_generate_repeat_rate(row, cfg.seqlen);
        for (uint32_t p = 0; p < cfg.seqlen; p += 1) {
            mda_tokenizer_entry_t *e = &tk->entries[row[p]];
            fwrite(e->bytes, 1, e->len, stdout);
        }
        printf("\n---\n");
    }
    printf("repeated trigrams: %.1f%% of all\n", 100.0 * rep / cfg.batch);
}

bool mda_generate_run(mda_model_t *m, mda_fwd_scratch_t *sc, const mda_tokenizer_t *tk, uint32_t steps) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t mask_id = tk->count;
    uint32_t *ids = sc->batch.ids;
    for (uint32_t i = 0; i < cfg.batch * cfg.seqlen; i += 1) {
        ids[i] = mask_id;
    }
    uint32_t *open = malloc(cfg.seqlen * sizeof(uint32_t));
    double t0 = mda_util_now();
    for (uint32_t s = 0; s < steps && m->c->fail == false; s += 1) {
        for (uint32_t b = 0; b < cfg.batch; b += 1) {
            mda_generate_weights(ids + b * cfg.seqlen, sc->batch.weight + b * cfg.seqlen, cfg.seqlen, mask_id);
        }
        sc->seed = (uint32_t)mda_hash_u64(MDA_HASH_GEN, s, 0);
        mda_fwd_run(m, sc);
        uint32_t *tok = sc->tok.ptr;
        for (uint32_t b = 0; b < cfg.batch && m->c->fail == false; b += 1) {
            mda_generate_reveal(ids + b * cfg.seqlen, tok + b * cfg.seqlen, open, cfg.seqlen, mask_id, s, steps - s, b * cfg.seqlen);
        }
    }
    double dt = mda_util_now() - t0;
    free(open);
    if (m->c->fail) {
        return false;
    } else {
        mda_generate_print(tk, ids, cfg, steps, sc->temp, dt);
        return true;
    }
}

bool mda_generate_scratch(mda_model_t *m, mda_fwd_scratch_t *sc, const mda_tokenizer_t *tk, const char *path, uint32_t steps) {
    mda_ckpt_res_t loaded = mda_ckpt_load(m, mda_tokenizer_hash(tk), path, false);
    if (loaded.ok == false) {
        return false;
    } else {
        if (loaded.step == 0) {
            fprintf(stderr, "no checkpoint: generating from the random init\n");
        } else {
            fprintf(stderr, "loaded checkpoint at step %u\n", loaded.step);
        }
        return mda_generate_run(m, sc, tk, steps);
    }
}

bool mda_generate_model(mda_vkc_t *c, const mda_tokenizer_t *tk, const char *path, uint32_t steps, float temp) {
    mda_model_t *m = mda_model_init(c, mda_model_cfg(tk->count));
    if (m == NULL) {
        return false;
    } else {
        m->infer = true;
        mda_fwd_scratch_t sc = mda_fwd_alloc(m, 1);
        sc.temp = temp;
        if (c->fail) {
            return false;
        } else {
            return mda_generate_scratch(m, &sc, tk, path, steps);
        }
    }
}

bool mda_generate_device(mda_vkc_t *c, const char *path, uint32_t steps, float temp) {
    mda_tokenizer_t tk = mda_tokenizer_load(MDA_TOKENIZER_VOCAB_PATH);
    if (tk.entries == NULL) {
        return false;
    } else {
        return mda_generate_model(c, &tk, path, steps, temp);
    }
}

bool mda_generate_main(uint32_t argc, char **argv) {
    uint32_t steps = (uint32_t)strtoul(mda_util_arg(argc, argv, 1, "16"), NULL, 10);
    const char *path = mda_util_arg(argc, argv, 2, MDA_CKPT_PATH);
    float temp = strtof(mda_util_arg(argc, argv, 3, "0.8"), NULL);
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_generate_device(&c, path, steps, temp);
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
