#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vkc.h"
#include "util.h"
#include "tokenizer.h"
#include "data.h"
#include "model.h"
#include "fwd.h"
#include "step.h"
#include "ckpt.h"

float mda_probe_acts_maxabs(mda_vkc_t *c, mda_vkc_buf_t *b, uint32_t n, mda_vkc_buf_t *stage) {
    mda_vkc_copy(c, stage, b, (VkDeviceSize)n * sizeof(float));
    const float *p = stage->ptr;
    float mx = 0.0f;
    uint32_t i = 0;
    while (i < n && isnan(p[i]) == 0) {
        mx = fmaxf(mx, fabsf(p[i]));
        i += 1;
    }
    if (i < n) {
        return INFINITY;
    } else {
        return mx;
    }
}

uint32_t mda_probe_acts_bad16(mda_vkc_t *c, mda_vkc_buf_t *b, uint32_t n, mda_vkc_buf_t *stage) {
    mda_vkc_copy(c, stage, b, (VkDeviceSize)n * sizeof(uint16_t));
    const uint16_t *p = stage->ptr;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; i += 1) {
        if ((p[i] & 0x7c00u) == 0x7c00u) {
            bad += 1;
        }
    }
    return bad;
}

uint32_t mda_probe_acts_first(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, uint32_t batches) {
    uint32_t first = 0;
    for (uint32_t i = 1; i <= batches && first == 0 && tr->m->c->fail == false; i += 1) {
        mda_data_batch(&tr->sc.batch, cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL, i);
        float loss = mda_fwd_run(tr->m, &tr->sc);
        if (tr->m->c->fail == false && isfinite(loss) == 0) {
            first = i;
        }
    }
    return first;
}

void mda_probe_acts_table(mda_step_trainer_t *tr, mda_vkc_buf_t *stage) {
    mda_vkc_t *c = tr->m->c;
    mda_model_cfg_t cfg = tr->m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    uint32_t ns = cfg.batch * cfg.n_head * cfg.seqlen / MDA_MODEL_WKV_CHUNK * 64 * 64;
    printf(
        "%5s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
        "layer",
        "x_in",
        "n1",
        "shift",
        "pv",
        "pr",
        "wkv_raw",
        "x_mid",
        "n2",
        "gate",
        "up",
        "h",
        "inf S l2r",
        "inf S r2l"
    );
    for (uint32_t l = 0; l < cfg.n_layer; l += 1) {
        mda_fwd_act_t *a = &tr->sc.acts[l];
        printf(
            "%5u %9.3g %9.3g %9.3g %9.3g %9.3g %9.3g %9.3g %9.3g %9.3g %9.3g %9.3g %9u %9u\n",
            l,
            mda_probe_acts_maxabs(c, &a->x_in, bt * d, stage),
            mda_probe_acts_maxabs(c, &a->n1, bt * d, stage),
            mda_probe_acts_maxabs(c, &a->shift, bt * d, stage),
            mda_probe_acts_maxabs(c, &a->pv, bt * d, stage),
            mda_probe_acts_maxabs(c, &a->pr, bt * d, stage),
            mda_probe_acts_maxabs(c, &a->wkv_raw, bt * d, stage),
            mda_probe_acts_maxabs(c, &a->x_mid, bt * d, stage),
            mda_probe_acts_maxabs(c, &a->n2, bt * d, stage),
            mda_probe_acts_maxabs(c, &a->gate_pre, bt * hid, stage),
            mda_probe_acts_maxabs(c, &a->up, bt * hid, stage),
            mda_probe_acts_maxabs(c, &a->h, bt * hid, stage),
            mda_probe_acts_bad16(c, &a->sstate, ns, stage),
            mda_probe_acts_bad16(c, &a->sstate2, ns, stage)
        );
    }
    printf("final xn %.3g\n", mda_probe_acts_maxabs(c, &tr->sc.xn, bt * d, stage));
}

bool mda_probe_acts_inspect(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    mda_model_cfg_t cfg = m->cfg;
    uint32_t ns = cfg.batch * cfg.n_head * cfg.seqlen / MDA_MODEL_WKV_CHUNK * 64 * 64;
    m->ref32 = true;
    m->dirty16 = true;
    float l32 = mda_fwd_run(m, &tr->sc);
    printf("fp32 reference forward on that batch: loss %g\n", l32);
    m->ref32 = false;
    m->dirty16 = true;
    mda_step_grad(tr);
    mda_vkc_buf_t stage = mda_vkc_alloc_cached(m->c, (VkDeviceSize)mda_util_max_u32(cfg.batch * cfg.seqlen * mda_util_max_u32(cfg.ffn_hidden, cfg.vocab), ns / 2) * sizeof(float));
    if (m->c->fail == false) {
        mda_probe_acts_table(tr, &stage);
    }
    return m->c->fail == false;
}

bool mda_probe_acts_search(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, uint32_t step, uint32_t batches) {
    uint32_t first = mda_probe_acts_first(tr, cp, batches);
    printf("checkpoint step %u: first non-finite fp16 loss at batch %u of %u\n", step, first, batches);
    if (tr->m->c->fail) {
        return false;
    } else if (first == 0) {
        return true;
    } else {
        return mda_probe_acts_inspect(tr);
    }
}

bool mda_probe_acts_trainer(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, const char *path, uint32_t batches) {
    mda_ckpt_res_t loaded = mda_ckpt_load(tr->m, mda_tokenizer_hash(&cp->tok), path, false);
    if (loaded.ok == false) {
        return false;
    } else if (loaded.step == 0) {
        fprintf(stderr, "%s: no checkpoint\n", path);
        return false;
    } else {
        return mda_probe_acts_search(tr, cp, loaded.step, batches);
    }
}

bool mda_probe_acts_model(mda_vkc_t *c, const mda_data_corpus_t *cp, const char *path, uint32_t batches) {
    mda_model_t *m = mda_model_init(c, mda_model_cfg(cp->tok.count));
    if (m == NULL) {
        return false;
    } else {
        mda_step_trainer_t tr = mda_step_init(m, 1.0f, 1, false);
        if (c->fail) {
            return false;
        } else {
            return mda_probe_acts_trainer(&tr, cp, path, batches);
        }
    }
}

bool mda_probe_acts_device(mda_vkc_t *c, const char *path, uint32_t batches) {
    mda_data_corpus_t cp = mda_data_load(MDA_TOKENIZER_CORPUS_PATH);
    if (cp.ids == NULL) {
        return false;
    } else {
        return mda_probe_acts_model(c, &cp, path, batches);
    }
}

bool mda_probe_acts_main(uint32_t argc, char **argv) {
    const char *path = mda_util_arg(argc, argv, 1, MDA_CKPT_OVERFLOW_PATH);
    uint32_t batches = (uint32_t)strtoul(mda_util_arg(argc, argv, 2, "500"), NULL, 10);
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_probe_acts_device(&c, path, batches);
    }
}

int main(int argc, char **argv) {
    bool ok = mda_probe_acts_main((uint32_t)argc, argv);
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
