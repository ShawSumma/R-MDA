// Verify RMSNorm forward (vs CPU) and backward dx, dg (vs finite difference).
#include "vkc.h"
#include <math.h>

typedef struct rms_push rms_push_t;

struct rms_push {
    uint32_t rows;
    uint32_t d;
    uint32_t op;
};

static vkc_t c;
static vkc_kernel_t kn;
static uint32_t g_rows = 4;
static uint32_t g_d = 33;

static void run_rms(vkc_buf_t *bx, vkc_buf_t *bg, vkc_buf_t *by,
    vkc_buf_t *brms, vkc_buf_t *bdg, uint32_t op) {
    rms_push_t push = (rms_push_t) {
        .rows = g_rows,
        .d = g_d,
        .op = op,
    };
    vkc_buf_t *b[5] = {
        bx,
        bg,
        by,
        brms,
        bdg,
    };
    vkc_run(&c, &kn, b, g_rows, 1, 1, &push);
}

// CPU forward into out, returns nothing
static void cpu_fwd(const float *x, const float *g, float *out) {
    for (uint32_t row = 0; row < g_rows; row++) {
        double ss = 0.0;
        for (uint32_t i = 0; i < g_d; i++) {
            ss += (double)x[row * g_d + i] * x[row * g_d + i];
        }
        float r = (float)sqrt(ss / (double)g_d + 1e-5);
        for (uint32_t i = 0; i < g_d; i++) {
            out[row * g_d + i] = x[row * g_d + i] / r * g[i];
        }
    }
}

int main(void) {
    c = vkc_init();
    kn = vkc_load(&c, "build/rmsnorm.spv", 5, sizeof(rms_push_t));

    uint32_t n = g_rows * g_d;
    vkc_buf_t bx = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bg = vkc_alloc(&c, g_d * sizeof(float));
    vkc_buf_t by = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t brms = vkc_alloc(&c, g_rows * sizeof(float));
    vkc_buf_t bdg = vkc_alloc(&c, g_d * sizeof(float));

    float *x = bx.ptr;
    float *g = bg.ptr;
    float *y = by.ptr;
    float *dg = bdg.ptr;

    float *x0 = malloc(n * sizeof(float));
    float *dy = malloc(n * sizeof(float));
    srand(9);
    for (uint32_t i = 0; i < n; i++) {
        x0[i] = ((rand() % 400) / 100.0f - 2.0f);
        x[i] = x0[i];
        dy[i] = ((rand() % 200) / 100.0f - 1.0f);
    }
    for (uint32_t i = 0; i < g_d; i++) {
        g[i] = 0.5f + (rand() % 100) / 100.0f;
    }
    float *g0 = malloc(g_d * sizeof(float));
    for (uint32_t i = 0; i < g_d; i++) {
        g0[i] = g[i];
    }

    // forward check
    run_rms(&bx, &bg, &by, &brms, &bdg, 0);
    float *ref = malloc(n * sizeof(float));
    cpu_fwd(x0, g0, ref);
    double ferr = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double e = fabs(y[i] - ref[i]);
        if (e > ferr) {
            ferr = e;
        }
    }

    // backward: y buffer = dy, dg zeroed, x = x0
    for (uint32_t i = 0; i < n; i++) {
        y[i] = dy[i];
        x[i] = x0[i];
    }
    for (uint32_t i = 0; i < g_d; i++) {
        dg[i] = 0.0f;
    }
    run_rms(&bx, &bg, &by, &brms, &bdg, 1);
    // x now holds dx, dg holds grad. snapshot.
    float *dx = malloc(n * sizeof(float));
    for (uint32_t i = 0; i < n; i++) {
        dx[i] = x[i];
    }
    float *dg_snap = malloc(g_d * sizeof(float));
    for (uint32_t i = 0; i < g_d; i++) {
        dg_snap[i] = dg[i];
    }

    // finite diff loss = sum(dy .* fwd(x,g))
    double eps = 1e-2;
    double dx_err = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        float orig = x0[i];
        x0[i] = orig + (float)eps;
        cpu_fwd(x0, g0, ref);
        double lp = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lp += dy[j] * ref[j];
        }
        x0[i] = orig - (float)eps;
        cpu_fwd(x0, g0, ref);
        double lm = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lm += dy[j] * ref[j];
        }
        x0[i] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double rel = fabs(num - dx[i]) / (fabs(num) + fabs(dx[i]) + 1e-3);
        if (rel > dx_err) {
            dx_err = rel;
        }
    }

    double dg_err = 0.0;
    for (uint32_t i = 0; i < g_d; i++) {
        float orig = g0[i];
        g0[i] = orig + (float)eps;
        cpu_fwd(x0, g0, ref);
        double lp = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lp += dy[j] * ref[j];
        }
        g0[i] = orig - (float)eps;
        cpu_fwd(x0, g0, ref);
        double lm = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lm += dy[j] * ref[j];
        }
        g0[i] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double rel = fabs(num - dg_snap[i]) / (fabs(num) + fabs(dg_snap[i]) + 1e-3);
        if (rel > dg_err) {
            dg_err = rel;
        }
    }

    printf("rmsnorm: fwd err = %g, dx rel = %g, dg rel = %g\n", ferr, dx_err, dg_err);
    int ok = ferr < 1e-4 && dx_err < 2e-2 && dg_err < 2e-2;
    printf("%s\n", ok ? "RMSNORM OK" : "RMSNORM WRONG");
    return ok ? 0 : 1;
}
