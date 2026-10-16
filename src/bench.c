// bench.c -- time the forward pass and report tokens/sec, so the optimization work
// is measured rather than guessed. Builds the model, runs a few warmup forwards,
// then times a batch of them. Reports ms/forward and tokens/sec at the configured
// batch and sequence length.
#include "model.h"
#include "fwd.h"
#include "step.h"
#include <time.h>

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(void) {
    srand(7);
    vkc_t c = vkc_init();
    corpus_t cp = corpus_load("data/wiki.txt");
    mcfg_t cfg = (mcfg_t) {
        .d = 640,
        .n_layer = 12,
        .n_head = 10,
        .seqlen = 64,
        .batch = 8,
        .ffn_hidden = 1728,
        .vocab = data_vocab,
        .vocab_in = data_vocab + 1,
    };
    model_t *m = model_init(&c, cfg);
    fwd_scratch_t sc = fwd_alloc(m, 1);

    make_batch(
        &cp,
        cfg.batch,
        cfg.seqlen,
        sc.ids.ptr,
        sc.target.ptr,
        sc.mask.ptr,
        sc.weight.ptr);

    uint32_t warmup = 3;
    uint32_t iters = 20;
    for (uint32_t i = 0; i < warmup; ++i) {
        model_forward(m, &sc);
    }
    double t0 = now_sec();
    for (uint32_t i = 0; i < iters; ++i) {
        model_forward(m, &sc);
    }
    double dt = (now_sec() - t0) / iters;

    uint32_t tokens = cfg.batch * cfg.seqlen;
    printf("forward: %.2f ms  -> %.0f tokens/sec  (%u tokens/forward)\n", dt * 1e3, tokens / dt, tokens);
    return 0;
}
