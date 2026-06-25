// bwd.h -- backward pass. Mirrors fwd.h in reverse, accumulating grads into each
// param's .g buffer. Where a verified kernel exists we use it; the elementwise
// glue (residual adds, sigmoid/silu derivative, pack/unpack) is done host-side in
// fp32 (fine at this size) to avoid unverified kernels. The forward saves the
// per-layer activations the backward needs into model->acts.
#ifndef BWD_H
#define BWD_H

#include "fwd.h"

// matmul backward helper: given dC, produce dA (mode 0) and accumulate dB into g.
static void mm_dA(
    model_t *m,
    vkc_buf_t *b,
    vkc_buf_t *dc,
    vkc_buf_t *da,
    uint32_t rows,
    uint32_t n,
    uint32_t k) {
    mm_bw_pc_t pc = {
        .m = rows,
        .n = n,
        .k = k,
        .mode = 0,
    };
    vkc_buf_t *bufs[4] = {
        da,
        b,
        dc,
        da,
    };
    vkc_run(m->c, &m->k_mm_bw, bufs, (k + 15) / 16, (rows + 15) / 16, 1, &pc);
}

static void mm_dB(
    model_t *m,
    vkc_buf_t *a,
    vkc_buf_t *dc,
    vkc_buf_t *db,
    uint32_t rows,
    uint32_t n,
    uint32_t k) {
    mm_bw_pc_t pc = {
        .m = rows,
        .n = n,
        .k = k,
        .mode = 1,
    };
    vkc_buf_t *bufs[4] = {
        a,
        a,
        dc,
        db,
    };
    vkc_run(m->c, &m->k_mm_bw, bufs, (n + 15) / 16, (k + 15) / 16, 1, &pc);
}

// accumulate src into dst->g (host add). used because mm_dB writes a fresh grad
// each call; for params used once per layer we can write directly, but tok/pos
// embeddings and shared params need accumulation. Here every weight is per-layer
// and used once, so we write the grad buffer directly (no cross-step accumulation).

static void host_add(vkc_buf_t *dst, vkc_buf_t *src, uint32_t n) {
    float *d = dst->ptr;
    float *s = src->ptr;
    for (uint32_t i = 0; i < n; ++i) {
        d[i] = d[i] + s[i];
    }
}

static void host_zero(vkc_buf_t *b, uint32_t n) {
    memset(b->ptr, 0, n * sizeof(float));
}

#endif
