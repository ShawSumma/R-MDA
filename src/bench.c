#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "vkc.h"
#include "hash.h"
#include "util.h"
#include "sort.h"
#include "data.h"
#include "model.h"
#include "fwd.h"
#include "step.h"

void mda_bench_run(mda_step_trainer_t *tr, mda_bench_phase_t ph, uint32_t step) {
    switch (ph) {
    case MDA_BENCH_FORWARD:
        mda_fwd_run(tr->m, &tr->sc);
        break;
    case MDA_BENCH_GRAD:
        mda_step_grad(tr);
        break;
    case MDA_BENCH_STEP:
        mda_step_train(tr, 1e-5f, step);
        break;
    }
}

void mda_bench_phase(mda_step_trainer_t *tr, mda_bench_phase_t ph, const char *name) {
    mda_sort_item_t ms[MDA_BENCH_ROUNDS];
    uint32_t step = 1;
    for (uint32_t i = 0; i < 2; i += 1) {
        mda_bench_run(tr, ph, step);
        step += 1;
    }
    for (uint32_t r = 0; r < MDA_BENCH_ROUNDS; r += 1) {
        double t0 = mda_util_now();
        for (uint32_t i = 0; i < MDA_BENCH_ITERS; i += 1) {
            mda_bench_run(tr, ph, step);
            step += 1;
        }
        ms[r] = (mda_sort_item_t) {
            .key = (mda_util_now() - t0) / MDA_BENCH_ITERS * 1e3,
            .idx = r,
        };
    }
    mda_sort_items(ms, MDA_BENCH_ROUNDS);
    uint32_t tokens = tr->m->cfg.batch * tr->m->cfg.seqlen;
    printf("%-8s %10.2f %10.2f %12.0f\n", name, ms[0].key, ms[MDA_BENCH_ROUNDS / 2].key, tokens / (ms[0].key / 1e3));
}

void mda_bench_profile(mda_step_trainer_t *tr) {
    mda_vkc_t *c = tr->m->c;
    mda_vkc_prof_begin(c);
    for (uint32_t i = 0; i < MDA_BENCH_PROF_BATCHES; i += 1) {
        mda_step_grad(tr);
        mda_vkc_begin(c);
        mda_step_opt_rec(tr, 1.0f);
        mda_vkc_flush(c);
    }
    if (c->fail == false) {
        printf("GPU time per training step:\n");
        mda_vkc_prof_report(c, MDA_BENCH_PROF_BATCHES);
    }
}

bool mda_bench_trainer(mda_step_trainer_t *tr) {
    uint32_t *cids = malloc(MDA_BENCH_CORPUS_LEN * sizeof(uint32_t));
    for (uint32_t i = 0; i < MDA_BENCH_CORPUS_LEN; i += 1) {
        cids[i] = (uint32_t)(mda_hash_u64(MDA_HASH_CORPUS, 0, i) % MDA_BENCH_VOCAB);
    }
    mda_data_corpus_t cp = {
        .ids = cids,
        .len = MDA_BENCH_CORPUS_LEN,
        .eval_start = MDA_BENCH_CORPUS_LEN,
        .mask_id = MDA_BENCH_VOCAB,
    };
    mda_data_batch(&tr->sc.batch, &cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL, 0);
    printf("%-8s %10s %10s %12s\n", "phase", "min ms", "median ms", "tok/s");
    mda_bench_phase(tr, MDA_BENCH_FORWARD, "forward");
    mda_bench_phase(tr, MDA_BENCH_GRAD, "fwd+bwd");
    mda_bench_phase(tr, MDA_BENCH_STEP, "step");
    if (tr->m->c->fail == false) {
        mda_bench_profile(tr);
    }
    free(cids);
    return tr->m->c->fail == false;
}

void mda_bench_decode_time(mda_model_t *m, mda_fwd_scratch_t *sc, const char *name) {
    mda_sort_item_t ms[MDA_BENCH_ROUNDS];
    for (uint32_t i = 0; i < 2; i += 1) {
        mda_fwd_run(m, sc);
    }
    for (uint32_t r = 0; r < MDA_BENCH_ROUNDS; r += 1) {
        double t0 = mda_util_now();
        for (uint32_t i = 0; i < MDA_BENCH_ITERS; i += 1) {
            mda_fwd_run(m, sc);
        }
        ms[r] = (mda_sort_item_t) {
            .key = (mda_util_now() - t0) / MDA_BENCH_ITERS * 1e3,
            .idx = r,
        };
    }
    mda_sort_items(ms, MDA_BENCH_ROUNDS);
    printf("%-8s %10s %10s %12s\n", "decode", "min ms", "median ms", "positions");
    printf("%-8s %10.2f %10.2f %12u\n", name, ms[0].key, ms[MDA_BENCH_ROUNDS / 2].key, m->cfg.batch * m->cfg.seqlen);
}

void mda_bench_decode_profile(mda_model_t *m, mda_fwd_scratch_t *sc, const char *name) {
    mda_vkc_t *c = m->c;
    mda_vkc_prof_begin(c);
    for (uint32_t i = 0; i < MDA_BENCH_PROF_BATCHES; i += 1) {
        mda_fwd_run(m, sc);
    }
    if (c->fail == false) {
        printf("GPU time per %s decode forward:\n", name);
        mda_vkc_prof_report(c, MDA_BENCH_PROF_BATCHES);
    }
}

bool mda_bench_decode(mda_vkc_t *c, mda_model_cfg_t cfg, const char *name, bool cached) {
    mda_model_t *m = mda_model_init(c, cfg);
    if (m == NULL) {
        return false;
    } else {
        m->infer = true;
        mda_fwd_scratch_t sc = mda_fwd_alloc(m, 1);
        mda_fwd_cache_t cache = { .cur = 0 };
        if (cached) {
            cache = mda_fwd_cache_alloc(m);
            sc.cache = &cache;
        }
        for (uint32_t i = 0; i < cfg.batch * cfg.seqlen && c->fail == false; i += 1) {
            sc.batch.ids[i] = (uint32_t)(mda_hash_u64(MDA_HASH_CORPUS, 1, i) % MDA_BENCH_VOCAB);
            sc.batch.t[i] = 0.5f;
        }
        mda_bench_decode_time(m, &sc, name);
        mda_bench_decode_profile(m, &sc, name);
        return c->fail == false;
    }
}

bool mda_bench_decoders(mda_vkc_t *c) {
    mda_model_cfg_t base = mda_model_cfg(MDA_BENCH_VOCAB);
    mda_model_cfg_t full = mda_model_cfg_block(base, MDA_MODEL_BLOCK);
    base.seqlen = MDA_MODEL_BLOCK;
    mda_model_cfg_t blk = mda_model_cfg_block(base, MDA_MODEL_BLOCK);
    if (mda_bench_decode(c, full, "full", false)) {
        return mda_bench_decode(c, blk, "cached", true);
    } else {
        return false;
    }
}

bool mda_bench_device(mda_vkc_t *c, bool muon) {
    mda_model_t *m = mda_model_init(c, mda_model_cfg(MDA_BENCH_VOCAB));
    if (m == NULL) {
        return false;
    } else {
        mda_step_trainer_t tr = mda_step_init(m, 1.0f, 1, muon);
        if (c->fail) {
            return false;
        } else if (mda_bench_trainer(&tr)) {
            return mda_bench_decoders(c);
        } else {
            return false;
        }
    }
}

bool mda_bench_main(uint32_t argc, char **argv) {
    bool muon = strcmp(mda_util_arg(argc, argv, 1, "adamw"), "muon") == 0;
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_bench_device(&c, muon);
    }
}

int main(int argc, char **argv) {
    bool ok = mda_bench_main((uint32_t)argc, argv);
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
