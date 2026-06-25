// train.c -- RWKV-7 + masked-diffusion denoiser training on Wikipedia bytes,
// entirely on the GPU via raw Vulkan compute. Forward + backward + AdamW.
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
            .wd = 0.01f,
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

int main(void) {
    srand(1234);
    vkc_t c = vkc_init();

    mcfg_t cfg = (mcfg_t) {
        .d = 512,
        .n_layer = 12,
        .n_head = 8,
        .seqlen = 128,
        .batch = 8,
        .ffn_hidden = 1365,
    };
    model_t *m = model_init(&c, cfg);
    trainer_t tr = trainer_make(m);
    corpus_t cp = corpus_load("data/wiki.txt");

    fprintf(stderr, "params %.2f M | d=%u L=%u H=%u seq=%u batch=%u\n",
        (double)model_param_count(m) / 1e6, cfg.d, cfg.n_layer, cfg.n_head,
        cfg.seqlen, cfg.batch);

    float lr = 3e-4f;
    uint32_t steps = 300;
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t s = 1; s <= steps; s++) {
        make_batch(&cp, cfg.batch, cfg.seqlen,
            tr.sc.ids.ptr, tr.sc.target.ptr, tr.sc.mask.ptr, tr.sc.weight.ptr);
        float loss = train_step(&tr);
        adamw_all(m, lr, s);
        if (s % 10 == 0 || s == 1) {
            struct timespec t1;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            printf("step %4u  loss %.4f  %.2f s/step\n", s, loss, dt / s);
            fflush(stdout);
        }
    }
    return 0;
}
