// Verify matmul forward (vs CPU) and backward (vs finite difference).
#include "vkc.h"
#include <math.h>

typedef struct mm_push mm_push_t;
typedef struct mmbw_push mmbw_push_t;

struct mm_push {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

struct mmbw_push {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t mode;
};

static vkc_t c;
static vkc_kernel_t kmm;
static vkc_kernel_t kbw;
static uint32_t g_m = 7;
static uint32_t g_n = 5;
static uint32_t g_k = 6;

static void run_mm(vkc_buf_t *ba, vkc_buf_t *bb, vkc_buf_t *bc) {
    mm_push_t push = (mm_push_t) {
        .m = g_m,
        .n = g_n,
        .k = g_k,
    };
    vkc_buf_t *b[3] = {
        ba,
        bb,
        bc,
    };
    vkc_run(&c, &kmm, b, (g_n + 15) / 16, (g_m + 15) / 16, 1, &push);
}

int main(void) {
    c = vkc_init();
    kmm = vkc_load(&c, "build/matmul.spv", 3, sizeof(mm_push_t));
    kbw = vkc_load(&c, "build/matmul_bw.spv", 4, sizeof(mmbw_push_t));

    vkc_buf_t ba = vkc_alloc(&c, g_m * g_k * sizeof(float));
    vkc_buf_t bb = vkc_alloc(&c, g_k * g_n * sizeof(float));
    vkc_buf_t bc = vkc_alloc(&c, g_m * g_n * sizeof(float));
    vkc_buf_t bdc = vkc_alloc(&c, g_m * g_n * sizeof(float));
    vkc_buf_t bda = vkc_alloc(&c, g_m * g_k * sizeof(float));
    vkc_buf_t bdb = vkc_alloc(&c, g_k * g_n * sizeof(float));

    float *a = ba.ptr;
    float *b = bb.ptr;
    float *cc = bc.ptr;
    float *dc = bdc.ptr;

    srand(1);
    for (uint32_t i = 0; i < g_m * g_k; i++) {
        a[i] = (float)((int)((i * 7) % 13) - 6) * 0.1f;
    }
    for (uint32_t i = 0; i < g_k * g_n; i++) {
        b[i] = (float)((int)((i * 5) % 11) - 5) * 0.1f;
    }
    for (uint32_t i = 0; i < g_m * g_n; i++) {
        dc[i] = ((rand() % 200) / 100.0f - 1.0f);
    }

    run_mm(&ba, &bb, &bc);

    double maxerr = 0.0;
    for (uint32_t i = 0; i < g_m; i++) {
        for (uint32_t j = 0; j < g_n; j++) {
            double acc = 0.0;
            for (uint32_t kk = 0; kk < g_k; kk++) {
                acc += a[i * g_k + kk] * b[kk * g_n + j];
            }
            double e = fabs(acc - cc[i * g_n + j]) / (fabs(acc) + 1e-4);
            if (e > maxerr) {
                maxerr = e;
            }
        }
    }
    printf("matmul fwd: max rel err = %g\n", maxerr);

    // analytic backward
    mmbw_push_t pa = (mmbw_push_t) {
        .m = g_m,
        .n = g_n,
        .k = g_k,
        .mode = 0,
    };
    vkc_buf_t *bua[4] = {
        &ba,
        &bb,
        &bdc,
        &bda,
    };
    vkc_run(&c, &kbw, bua, (g_k + 15) / 16, (g_m + 15) / 16, 1, &pa);

    // snapshot analytic dA before any further dispatches can touch it
    float da_snap[7 * 6];
    for (uint32_t i = 0; i < g_m * g_k; i++) {
        da_snap[i] = ((float *)bda.ptr)[i];
    }

    mmbw_push_t pbb = (mmbw_push_t) {
        .m = g_m,
        .n = g_n,
        .k = g_k,
        .mode = 1,
    };
    vkc_buf_t *bub[4] = {
        &ba,
        &bb,
        &bdc,
        &bdb,
    };
    vkc_run(&c, &kbw, bub, (g_n + 15) / 16, (g_k + 15) / 16, 1, &pbb);

    // finite difference: loss = sum(dc .* c)
    double eps = 1e-3;
    double worst = 0.0;
    float *da = da_snap;
    for (uint32_t i = 0; i < g_m * g_k; i++) {
        float orig = a[i];
        a[i] = orig + (float)eps;
        run_mm(&ba, &bb, &bc);
        double lp = 0.0;
        for (uint32_t j = 0; j < g_m * g_n; j++) {
            lp += dc[j] * cc[j];
        }
        a[i] = orig - (float)eps;
        run_mm(&ba, &bb, &bc);
        double lm = 0.0;
        for (uint32_t j = 0; j < g_m * g_n; j++) {
            lm += dc[j] * cc[j];
        }
        a[i] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double rel = fabs(num - da[i]) / (fabs(num) + fabs(da[i]) + 1e-3);
        if (rel > worst) {
            worst = rel;
        }
    }
    printf("matmul bwd dA: max rel err = %g\n", worst);

    int ok = maxerr < 1e-3 && worst < 2e-2;
    printf("%s\n", ok ? "MATMUL OK" : "MATMUL WRONG");
    return ok ? 0 : 1;
}
