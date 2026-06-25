// train.c -- RWKV-7 + masked-diffusion denoiser training on Wikipedia, entirely on
// the GPU via raw Vulkan compute. Forward + backward + AdamW. Time-based atomic
// checkpoints; designed to resume from the latest checkpoint after a restart.
#include "model.h"
#include "fwd.h"
#include "step.h"
#include <time.h>

static void adamw_all(model_t *m, float lr, uint32_t step) {
    float b1 = 0.9f;
    float b2 = 0.999f;
    float bc1 = 1.0f - powf(b1, (float)step);
    float bc2 = 1.0f - powf(b2, (float)step);
    for (uint32_t i = 0; i < m->n_params; i = i + 1) {
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

// global-norm gradient clipping (GPU-reduced). The 1/t diffusion weighting makes
// some batches produce huge gradients that kick the model out of its minimum. Clip
// the global norm to clip. The sum-of-squares is reduced on the GPU (a host loop
// over all the mapped grads is far too slow); only the small per-workgroup partials
// are read back. Returns the pre-clip norm.
static float clip_grads(model_t *m, vkc_buf_t *partials, float clip) {
    double ss = 0.0;
    for (uint32_t i = 0; i < m->n_params; i = i + 1) {
        uint32_t n = m->params[i].n;
        uint32_t ng = (n + 255) / 256;
        sumsq_pc_t pc = (sumsq_pc_t) {
            .n = n,
            .op = 0,
            .scale = 0.0f,
        };
        vkc_buf_t *bufs[2] = {
            &m->params[i].g,
            partials,
        };
        vkc_run(m->c, &m->k_sumsq, bufs, ng, 1, 1, &pc);
        float *p = partials->ptr;
        for (uint32_t w = 0; w < ng; w = w + 1) {
            ss = ss + (double)p[w];
        }
    }
    float norm = (float)sqrt(ss);
    if (norm <= clip) {
        return norm;
    }
    float scale = clip / (norm + 1e-9f);
    for (uint32_t i = 0; i < m->n_params; i = i + 1) {
        uint32_t n = m->params[i].n;
        sumsq_pc_t pc = (sumsq_pc_t) {
            .n = n,
            .op = 1,
            .scale = scale,
        };
        vkc_buf_t *bufs[2] = {
            &m->params[i].g,
            partials,
        };
        vkc_run(m->c, &m->k_sumsq, bufs, (n + 255) / 256, 1, 1, &pc);
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
    for (uint32_t i = 0; i < m->n_params; i = i + 1) {
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
    for (uint32_t i = 0; i < m->n_params; i = i + 1) {
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

    fprintf(stderr, "params %.2f M | d=%u L=%u H=%u seq=%u batch=%u\n", (double)model_param_count(m) / 1e6, cfg.d, cfg.n_layer, cfg.n_head, cfg.seqlen, cfg.batch);

    uint32_t start = checkpoint_load(m);
    vkc_buf_t clip_partials = vkc_alloc(&c, 8192 * sizeof(float));

    float lr_peak = 3e-5f;
    uint32_t warmup = 50;
    uint32_t max_steps = 1000000;
    double ckpt_interval = 120.0;
    struct timespec t0;
    struct timespec t_ckpt;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    t_ckpt = t0;

    double ema = 0.0;
    double ema_bpb = 0.0;
    int ema_init = 0;

    for (uint32_t s = start + 1; s <= max_steps; s = s + 1) {
        make_batch(
            &cp,
            cfg.batch,
            cfg.seqlen,
            tr.sc.ids.ptr,
            tr.sc.target.ptr,
            tr.sc.mask.ptr,
            tr.sc.weight.ptr);
        float loss = train_step(&tr);
        float lr = (s < warmup) ? lr_peak * (float)s / (float)warmup : lr_peak;
        float gnorm = clip_grads(m, &clip_partials, 2.0f);
        adamw_all(m, lr, s);
        (void)gnorm;

        float bpb = tr.last_raw_ce / 0.6931472f;

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
            double per_step = dt / (s - start);
            printf("step %6u  loss %.4f ema %.4f  bits/byte %.4f ema %.4f  %.2f s/step\n", s, loss, ema, bpb, ema_bpb, per_step);
            fflush(stdout);
        }
        if (since_ckpt >= ckpt_interval) {
            checkpoint_save(m, s);
            t_ckpt = t1;
            fprintf(stderr, "[checkpoint @ step %u  bpb %.3f]\n", s, ema_bpb);
        }
    }
    return 0;
}
