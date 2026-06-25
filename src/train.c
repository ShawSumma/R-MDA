// train.c -- RWKV-7 + masked-diffusion denoiser training on Wikipedia bytes,
// entirely on the GPU via raw Vulkan compute. Forward + backward + AdamW.
// Time-based atomic checkpoints; designed to run overnight under an auto-restart
// wrapper that resumes from the latest checkpoint.
#include "model.h"
#include "fwd.h"
#include "step.h"
#include <time.h>

typedef struct adam_pc adam_pc_t;

struct adam_pc {
    uint32_t n;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float wd;
    float bc1;
    float bc2;
};

static void adamw_all(model_t *m, float lr, uint32_t step) {
    float b1 = 0.9f;
    float b2 = 0.999f;
    float bc1 = 1.0f - powf(b1, (float)step);
    float bc2 = 1.0f - powf(b2, (float)step);
    for (uint32_t i = 0; i < m->n_params; i++) {
        param_t *p = &m->params[i];
        adam_pc_t pc = (adam_pc_t) {
            .n = p->n,
            .lr = lr,
            .beta1 = b1,
            .beta2 = b2,
            .eps = 1e-8f,
            .wd = 0.0f,
            .bc1 = bc1,
            .bc2 = bc2,
        };
        vkc_buf_t *bufs[4] = {
            &p->w,
            &p->g,
            &p->m,
            &p->v,
        };
        vkc_run(m->c, &m->k_adamw, bufs, (p->n + 255) / 256, 1, 1, &pc);
    }
}

typedef struct sumsq_pc sumsq_pc_t;
struct sumsq_pc {
    uint32_t n;
    uint32_t op;
    float scale;
};

// global-norm gradient clipping (GPU-reduced). The 1/t diffusion weighting makes
// some batches produce huge gradients (norm 8-15) that kick the model out of its
// minimum -> recurring drift. Clip global norm to `clip`. The sum-of-squares is
// reduced ON THE GPU (a host loop over all 63M mapped grads cost ~14s/step!); only
// the small per-workgroup partials are read back. Returns the pre-clip norm.
static float clip_grads(model_t *m, vkc_buf_t *partials, float clip) {
    double ss = 0.0;
    for (uint32_t i = 0; i < m->n_params; i++) {
        uint32_t n = m->params[i].n;
        uint32_t ng = (n + 255) / 256;
        sumsq_pc_t pc = (sumsq_pc_t) { .n = n, .op = 0, .scale = 0.0f };
        vkc_buf_t *bufs[2] = { &m->params[i].g, partials };
        vkc_run(m->c, &m->k_sumsq, bufs, ng, 1, 1, &pc);
        float *p = partials->ptr;
        for (uint32_t w = 0; w < ng; w++) {
            ss += (double)p[w];
        }
    }
    float norm = (float)sqrt(ss);
    if (norm > clip) {
        float scale = clip / (norm + 1e-9f);
        for (uint32_t i = 0; i < m->n_params; i++) {
            uint32_t n = m->params[i].n;
            sumsq_pc_t pc = (sumsq_pc_t) { .n = n, .op = 1, .scale = scale };
            vkc_buf_t *bufs[2] = { &m->params[i].g, partials };
            vkc_run(m->c, &m->k_sumsq, bufs, (n + 255) / 256, 1, 1, &pc);
        }
    }
    return norm;
}

// atomic checkpoint: write every param's weights + moments + the step to a temp
// file, then rename over the real one (so a crash mid-write can't corrupt it).
static void checkpoint_save(model_t *m, uint32_t step) {
    const char *tmp = "models/ckpt.tmp";
    const char *dst = "models/ckpt.bin";
    FILE *f = fopen(tmp, "wb");
    if (f == NULL) {
        return;
    }
    fwrite(&step, sizeof(uint32_t), 1, f);
    for (uint32_t i = 0; i < m->n_params; i++) {
        param_t *p = &m->params[i];
        fwrite(p->w.ptr, sizeof(float), p->n, f);
        fwrite(p->m.ptr, sizeof(float), p->n, f);
        fwrite(p->v.ptr, sizeof(float), p->n, f);
    }
    fclose(f);
    rename(tmp, dst);
}

static uint32_t checkpoint_load(model_t *m) {
    FILE *f = fopen("models/ckpt.bin", "rb");
    if (f == NULL) {
        return 0;
    }
    uint32_t step = 0;
    if (fread(&step, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return 0;
    }
    for (uint32_t i = 0; i < m->n_params; i++) {
        param_t *p = &m->params[i];
        fread(p->w.ptr, sizeof(float), p->n, f);
        fread(p->m.ptr, sizeof(float), p->n, f);
        fread(p->v.ptr, sizeof(float), p->n, f);
    }
    fclose(f);
    fprintf(stderr, "resumed from checkpoint at step %u\n", step);
    return step;
}

int main(void) {
    srand(1234);
    vkc_t c = vkc_init();

    mcfg_t cfg = (mcfg_t) {
        .d = 512,
        .n_layer = 12,
        .n_head = 8,
        .seqlen = 64,
        .batch = 8,
        .ffn_hidden = 1365,
    };
    model_t *m = model_init(&c, cfg);
    trainer_t tr = trainer_make(m);
    corpus_t cp = corpus_load("data/wiki.txt");

    fprintf(stderr, "params %.2f M | d=%u L=%u H=%u seq=%u batch=%u\n",
        (double)model_param_count(m) / 1e6, cfg.d, cfg.n_layer, cfg.n_head,
        cfg.seqlen, cfg.batch);

    uint32_t start = checkpoint_load(m);
    vkc_buf_t clip_partials = vkc_alloc(&c, 8192 * sizeof(float));

    float lr_peak = 3e-5f;          // lower peak: model destabilizing even at 5e-5+decay
    uint32_t warmup = 50;           // linear warmup avoids early instability
    uint32_t max_steps = 1000000;
    double ckpt_interval = 120.0;   // checkpoint every 2 minutes
    struct timespec t0;
    struct timespec t_ckpt;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    t_ckpt = t0;

    double ema = 0.0;
    double ema_bpb = 0.0;   // bits/byte, the clean low-noise learning signal
    int ema_init = 0;
    // best ema_bpb ever checkpointed (true best-state ratchet). PERSISTED across
    // restarts (models/best_bpb.txt) so a post-divergence restart never overwrites the
    // all-time-best checkpoint with a worse recovery state -- the on-disk ckpt is the
    // genuine global best. Fresh start: high so the first healthy state saves.
    double best_bpb = 1e9;
    {
        FILE *bf = fopen("models/best_bpb.txt", "r");
        if (bf != NULL) {
            if (fscanf(bf, "%lf", &best_bpb) != 1) {
                best_bpb = 1e9;
            }
            fclose(bf);
        }
    }
    for (uint32_t s = start + 1; s <= max_steps; s++) {
        make_batch(&cp, cfg.batch, cfg.seqlen,
            tr.sc.ids.ptr, tr.sc.target.ptr, tr.sc.mask.ptr, tr.sc.weight.ptr);
        float loss = train_step(&tr);
        // warmup, then slow exponential decay so the model SETTLES into minima
        // instead of the optimizer overshooting near a low (the recurring "reach a
        // low then the ema floor creeps up" pattern -> needs decaying steps, not
        // just a lower constant lr or tighter clip).
        // CONSTANT lr after warmup (no decay): the decay drove lr so low the model
        // could not self-correct from wandering and diverged. Constant modest lr lets
        // it both keep learning and recover from perturbations.
        float lr = (s < warmup) ? lr_peak * (float)s / (float)warmup : lr_peak;
        float gnorm = clip_grads(m, &clip_partials, 2.0f);   // GPU-reduced norm clip
        adamw_all(m, lr, s);
        (void)gnorm;

        float bpb = tr.last_raw_ce / 0.6931472f;   // nats -> bits per masked byte

        // smoothed so the trend is readable through the 1/t-weighting noise
        if (ema_init == 0) {
            ema = loss;
            ema_bpb = bpb;
            ema_init = 1;
        } else {
            ema = 0.95 * ema + 0.05 * loss;
            ema_bpb = 0.95 * ema_bpb + 0.05 * bpb;
        }

        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double since_ckpt = (t1.tv_sec - t_ckpt.tv_sec) + (t1.tv_nsec - t_ckpt.tv_nsec) / 1e9;

        if (s % 5 == 0 || s == start + 1) {
            double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            printf("step %6u  loss %.4f ema %.4f  bits/byte %.4f ema %.4f  %.2f s/step\n",
                s, loss, ema, bpb, ema_bpb, dt / (s - start));
            fflush(stdout);
        }
        // Divergence protection: this architecture has a recurring late-training
        // instability that no lr/wd/clip tuning fully prevents. So (a) only checkpoint
        // when the model is HEALTHY (ema_bpb < 3.8) — never overwrite a good ckpt with
        // a degraded one — and (b) if it diverges (ema_bpb > 4.5 sustained), exit so the
        // auto-restart wrapper reloads the last good checkpoint (restart resets the
        // optimizer state, which reliably recovers it to ~3.0). This makes the
        // recover→drift→restart cycle automatic and keeps the run at a usable level.
        // TRUE best-state ratchet: only overwrite the checkpoint when ema_bpb is a NEW
        // BEST (not merely healthy) -> the on-disk ckpt monotonically improves, and a
        // divergence-restart always reloads the genuinely-best state, not the latest.
        if (since_ckpt >= ckpt_interval && ema_init && ema_bpb < best_bpb) {
            checkpoint_save(m, s);
            t_ckpt = t1;
            best_bpb = ema_bpb;
            // persist the all-time best so restarts preserve it (atomic: write+rename)
            FILE *bf = fopen("models/best_bpb.tmp", "w");
            if (bf != NULL) {
                fprintf(bf, "%.6f\n", best_bpb);
                fclose(bf);
                rename("models/best_bpb.tmp", "models/best_bpb.txt");
            }
            fprintf(stderr, "[checkpoint @ step %u  bpb %.3f (new best)]\n", s, ema_bpb);
        }
        if (s > start + 30 && ema_bpb > 4.5) {
            fprintf(stderr, "[diverged: ema bpb %.3f at step %u -> exit for restart from last good ckpt]\n", ema_bpb, s);
            return 1;   // nonzero -> wrapper restarts, reloading the healthy checkpoint
        }
    }
    return 0;
}
