// Finite-difference check of the WKV7 backward kernel.
// Loss = sum(dO .* O). dL/dO = dO (random). Compare kernel dK/dV/dR/dW/dA to
// numerical gradients of L wrt K/V/R/W/A.
#include "vkc.h"
#include <math.h>

typedef struct fwd_push fwd_push_t;
typedef struct bwd_push bwd_push_t;

struct fwd_push {
    uint32_t bsz;
    uint32_t hd;
    uint32_t t;
    uint32_t dh;
    uint32_t dir;
    uint32_t save;
};

struct bwd_push {
    uint32_t bsz;
    uint32_t hd;
    uint32_t t;
    uint32_t dh;
};

static vkc_t c;
static vkc_kernel_t kf;
static vkc_kernel_t kb;
static uint32_t g_bsz = 2;
static uint32_t g_hd = 3;
static uint32_t g_t = 6;
static uint32_t g_dh = 4;

static void run_fwd(vkc_buf_t *bk, vkc_buf_t *bv, vkc_buf_t *br, vkc_buf_t *bw,
    vkc_buf_t *ba, vkc_buf_t *bo, vkc_buf_t *bs, uint32_t save) {
    fwd_push_t push = (fwd_push_t) {
        .bsz = g_bsz,
        .hd = g_hd,
        .t = g_t,
        .dh = g_dh,
        .dir = 0,
        .save = save,
    };
    vkc_buf_t *b[7] = {
        bk,
        bv,
        br,
        bw,
        ba,
        bo,
        bs,
    };
    vkc_run(&c, &kf, b, g_bsz * g_hd, 1, 1, &push);
}

int main(void) {
    c = vkc_init();
    kf = vkc_load(&c, "build/wkv7_fwd.spv", 7, sizeof(fwd_push_t));
    kb = vkc_load(&c, "build/wkv7_bwd.spv", 12, sizeof(bwd_push_t));

    uint32_t n = g_bsz * g_hd * g_t * g_dh;
    uint32_t n2 = g_bsz * g_hd * g_t * g_dh * g_dh;

    vkc_buf_t bk = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bv = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t br = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bw = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t ba = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bo = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bs = vkc_alloc(&c, n2 * sizeof(float));
    vkc_buf_t bdo = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bdk = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bdv = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bdr = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bdw = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bda = vkc_alloc(&c, n * sizeof(float));

    float *k = bk.ptr;
    float *v = bv.ptr;
    float *r = br.ptr;
    float *w = bw.ptr;
    float *a = ba.ptr;
    float *o = bo.ptr;
    float *pdo = bdo.ptr;

    srand(3);
    for (uint32_t i = 0; i < n; i++) {
        k[i] = ((rand() % 200) / 100.0f - 1.0f) * 0.5f;
        v[i] = ((rand() % 200) / 100.0f - 1.0f);
        r[i] = ((rand() % 200) / 100.0f - 1.0f) * 0.5f;
        w[i] = 0.9f + 0.09f * ((rand() % 100) / 100.0f);
        a[i] = 0.3f;
        pdo[i] = ((rand() % 200) / 100.0f - 1.0f);
    }

    run_fwd(&bk, &bv, &br, &bw, &ba, &bo, &bs, 1);
    bwd_push_t pb = (bwd_push_t) {
        .bsz = g_bsz,
        .hd = g_hd,
        .t = g_t,
        .dh = g_dh,
    };
    vkc_buf_t *bb[12] = {
        &bk,
        &bv,
        &br,
        &bw,
        &ba,
        &bs,
        &bdo,
        &bdk,
        &bdv,
        &bdr,
        &bdw,
        &bda,
    };
    vkc_run(&c, &kb, bb, g_bsz * g_hd, 1, 1, &pb);

    float *arrs[5] = {
        k,
        v,
        r,
        w,
        a,
    };
    vkc_buf_t *dbuf[5] = {
        &bdk,
        &bdv,
        &bdr,
        &bdw,
        &bda,
    };
    const char *names[5] = {
        "dK",
        "dV",
        "dR",
        "dW",
        "dA",
    };

    double eps = 1e-3;
    double worst = 0.0;
    for (uint32_t p = 0; p < 5; p++) {
        float *arr = arrs[p];
        float *ana = dbuf[p]->ptr;
        double maxrel = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            float orig = arr[i];

            arr[i] = orig + (float)eps;
            run_fwd(&bk, &bv, &br, &bw, &ba, &bo, &bs, 0);
            double lp = 0.0;
            for (uint32_t j = 0; j < n; j++) {
                lp += pdo[j] * o[j];
            }

            arr[i] = orig - (float)eps;
            run_fwd(&bk, &bv, &br, &bw, &ba, &bo, &bs, 0);
            double lm = 0.0;
            for (uint32_t j = 0; j < n; j++) {
                lm += pdo[j] * o[j];
            }

            arr[i] = orig;
            double num = (lp - lm) / (2.0 * eps);
            if (p == 4 && (i % g_dh) != 0) {
                continue;
            }
            double rel = fabs(num - ana[i]) / (fabs(num) + fabs(ana[i]) + 1e-3);
            if (rel > maxrel) {
                maxrel = rel;
            }
        }
        printf("%s: max rel err = %g\n", names[p], maxrel);
        if (maxrel > worst) {
            worst = maxrel;
        }
    }
    printf("%s\n", worst < 2e-2 ? "WKV7 BWD OK" : "WKV7 BWD WRONG");
    return worst < 2e-2 ? 0 : 1;
}
