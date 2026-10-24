#include <math.h>
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

double mda_bench_mm_time(mda_vkc_t *c, mda_vkc_kernel_t *k, mda_vkc_buf_t **bufs, uint32_t gx, uint32_t gy, mda_pc_mm16_t *pc) {
    double best = 1e30;
    for (uint32_t r = 0; r < MDA_BENCH_ROUNDS; r += 1) {
        double t0 = mda_util_now();
        mda_vkc_begin(c);
        for (uint32_t i = 0; i < MDA_BENCH_MM_REPS; i += 1) {
            mda_vkc_run(c, k, bufs, gx, gy, 1, pc);
        }
        mda_vkc_flush(c);
        best = fmin(best, (mda_util_now() - t0) / MDA_BENCH_MM_REPS);
    }
    return best;
}

void mda_bench_mm_shape(mda_model_t *m, uint32_t rows, uint32_t n, uint32_t k, uint32_t ta, uint32_t tb) {
    mda_vkc_t *c = m->c;
    mda_vkc_buf_t a = mda_vkc_alloc(c, (VkDeviceSize)rows * k * sizeof(uint16_t));
    mda_vkc_buf_t b = mda_vkc_alloc(c, (VkDeviceSize)k * n * sizeof(uint16_t));
    mda_vkc_buf_t out = mda_vkc_alloc(c, (VkDeviceSize)rows * n * sizeof(float));
    mda_vkc_fill(c, &a, 0);
    mda_vkc_fill(c, &b, 0);
    mda_vkc_buf_t *bufs[6] = {
        &a,
        &b,
        &out,
        &m->amax,
        &out,
        &out,
    };
    mda_pc_mm16_t pc = {
        .m = rows,
        .n = n,
        .k = k,
        .sa = MDA_PC_NO_SLOT,
        .sb = MDA_PC_NO_SLOT,
        .ta = ta,
        .tb = tb,
        .store = MDA_FWD_STORE32,
        .alpha = 1.0f,
    };
    double t64 = mda_bench_mm_time(c, &m->k_mm16, bufs, n / 64, rows / 64, &pc);
    double t256 = mda_bench_mm_time(c, &m->k_mm16w, bufs, (n + 127) / 128, (rows + 127) / 128, &pc);
    double flops = 2.0 * rows * n * k;
    printf("%6u %6u %6u %2u %2u %9.3f %7.2f %9.3f %7.2f\n", rows, n, k, ta, tb, t64 * 1e3, flops / t64 / 1e12, t256 * 1e3, flops / t256 / 1e12);
    mda_vkc_free(c, &a);
    mda_vkc_free(c, &b);
    mda_vkc_free(c, &out);
}

void mda_bench_matmuls(mda_model_t *m) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    printf("%6s %6s %6s %2s %2s %9s %7s %9s %7s\n", "m", "n", "k", "ta", "tb", "64 ms", "TFLOPS", "128 ms", "TFLOPS");
    mda_bench_mm_shape(m, bt, d, d, 0, 0);
    mda_bench_mm_shape(m, bt, hid, d, 0, 0);
    mda_bench_mm_shape(m, bt, d, hid, 0, 0);
    mda_bench_mm_shape(m, bt, cfg.vocab, d, 0, 0);
    mda_bench_mm_shape(m, bt, d, cfg.vocab, 0, 0);
    mda_bench_mm_shape(m, d, d, bt, 1, 0);
    mda_bench_mm_shape(m, d, hid, bt, 1, 0);
    mda_bench_mm_shape(m, hid, d, bt, 1, 0);
    mda_bench_mm_shape(m, cfg.vocab, d, bt, 1, 0);
    mda_bench_mm_shape(m, d, cfg.vocab, bt, 1, 0);
    mda_bench_mm_shape(m, d, d, d, 0, 0);
    mda_bench_mm_shape(m, d, d, hid, 0, 1);
    mda_bench_mm_shape(m, d, d, hid, 1, 0);
    mda_bench_mm_shape(m, d, hid, d, 0, 0);
    mda_bench_mm_shape(m, hid, d, d, 0, 0);
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

bool mda_bench_decode(mda_vkc_t *c, mda_model_cfg_t cfg, const char *name, bool cached, uint32_t dec) {
    mda_model_t *m = mda_model_init(c, cfg);
    if (m == NULL || mda_model_set_dec(m, dec) == false) {
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
        }
        mda_bench_decode_time(m, &sc, name);
        mda_bench_decode_profile(m, &sc, name);
        return c->fail == false;
    }
}

bool mda_bench_decoders(mda_vkc_t *c, uint32_t dec) {
    mda_model_cfg_t base = mda_model_cfg(MDA_BENCH_VOCAB);
    if (dec > 0) {
        base.batch = mda_util_max_u32(base.batch, 64 / MDA_MODEL_BLOCK);
    }
    mda_model_cfg_t full = mda_model_cfg_block(base, MDA_MODEL_BLOCK);
    base.seqlen = MDA_MODEL_BLOCK;
    mda_model_cfg_t blk = mda_model_cfg_block(base, MDA_MODEL_BLOCK);
    if (mda_bench_decode(c, full, "full", false, dec)) {
        return mda_bench_decode(c, blk, "cached", true, dec);
    } else {
        return false;
    }
}

bool mda_bench_device(mda_vkc_t *c, bool muon, uint32_t batch, const char *mode) {
    mda_model_t *m = mda_model_init(c, mda_model_cfg_mode(MDA_BENCH_VOCAB, mode, batch));
    uint32_t dec = mda_model_dec_of(mode);
    if (m == NULL || mda_model_set_dec(m, dec) == false) {
        return false;
    } else {
        m->causal = strcmp(mode, "ar") == 0;
        mda_step_trainer_t tr = mda_step_init(m, 1.0f, 1, muon, NULL);
        printf("training uses %.2f GB of GPU memory\n", (double)c->dev_bytes / 1e9);
        mda_bench_matmuls(m);
        if (c->fail) {
            return false;
        } else if (mda_bench_trainer(&tr)) {
            return mda_bench_decoders(c, dec);
        } else {
            return false;
        }
    }
}

bool mda_bench_main(uint32_t argc, char **argv) {
    bool muon = strcmp(mda_util_arg(argc, argv, 1, "muon"), "muon") == 0;
    uint32_t batch = (uint32_t)strtoul(mda_util_arg(argc, argv, 2, "8"), NULL, 10);
    const char *mode = mda_util_arg(argc, argv, 3, "diffusion");
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_bench_device(&c, muon, batch, mode);
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
