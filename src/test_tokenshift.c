// Verify token-shift forward (vs CPU) and backward dx, dmix (vs finite difference).
#include "vkc.h"
#include <math.h>

typedef struct ts_push ts_push_t;

struct ts_push {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
};

static vkc_t c;
static vkc_kernel_t kn;
static uint32_t g_b = 2;
static uint32_t g_seqlen = 6;
static uint32_t g_d = 5;

static void run_ts(vkc_buf_t *bx, vkc_buf_t *bmix, vkc_buf_t *bout,
    vkc_buf_t *bdx, vkc_buf_t *bdmix, uint32_t op) {
    uint32_t bt = g_b * g_seqlen;
    ts_push_t push = (ts_push_t) {
        .bt = bt,
        .d = g_d,
        .seqlen = g_seqlen,
        .op = op,
    };
    vkc_buf_t *b[5] = {
        bx,
        bmix,
        bout,
        bdx,
        bdmix,
    };
    vkc_run(&c, &kn, b, (bt * g_d + 255) / 256, 1, 1, &push);
}

static void cpu_fwd(const float *x, const float *mix, float *out, uint32_t bt) {
    for (uint32_t row = 0; row < bt; row++) {
        uint32_t t = row % g_seqlen;
        for (uint32_t col = 0; col < g_d; col++) {
            float m = mix[col];
            float cur = x[row * g_d + col];
            float prev = (t == 0) ? 0.0f : x[(row - 1) * g_d + col];
            out[row * g_d + col] = m * cur + (1.0f - m) * prev;
        }
    }
}

int main(void) {
    c = vkc_init();
    kn = vkc_load(&c, "build/tokenshift.spv", 5, sizeof(ts_push_t));

    uint32_t bt = g_b * g_seqlen;
    uint32_t n = bt * g_d;
    vkc_buf_t bx = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bmix = vkc_alloc(&c, g_d * sizeof(float));
    vkc_buf_t bout = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bdx = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bdmix = vkc_alloc(&c, g_d * sizeof(float));

    float *x = bx.ptr;
    float *mix = bmix.ptr;
    float *out = bout.ptr;
    float *dx = bdx.ptr;
    float *dmix = bdmix.ptr;

    float *x0 = malloc(n * sizeof(float));
    float *mix0 = malloc(g_d * sizeof(float));
    float *dy = malloc(n * sizeof(float));
    srand(4);
    for (uint32_t i = 0; i < n; i++) {
        x[i] = ((rand() % 200) / 100.0f - 1.0f);
        x0[i] = x[i];
        dy[i] = ((rand() % 200) / 100.0f - 1.0f);
    }
    for (uint32_t i = 0; i < g_d; i++) {
        mix[i] = (rand() % 100) / 100.0f;
        mix0[i] = mix[i];
    }

    run_ts(&bx, &bmix, &bout, &bdx, &bdmix, 0);
    float *ref = malloc(n * sizeof(float));
    cpu_fwd(x0, mix0, ref, bt);
    double ferr = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double e = fabs(out[i] - ref[i]);
        if (e > ferr) {
            ferr = e;
        }
    }

    // backward
    for (uint32_t i = 0; i < n; i++) {
        out[i] = dy[i];
        dx[i] = 0.0f;
    }
    for (uint32_t i = 0; i < g_d; i++) {
        dmix[i] = 0.0f;
    }
    run_ts(&bx, &bmix, &bout, &bdx, &bdmix, 1);
    float *dx_snap = malloc(n * sizeof(float));
    for (uint32_t i = 0; i < n; i++) {
        dx_snap[i] = dx[i];
    }
    float *dmix_snap = malloc(g_d * sizeof(float));
    for (uint32_t i = 0; i < g_d; i++) {
        dmix_snap[i] = dmix[i];
    }

    double eps = 1e-3;
    double dx_err = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        float orig = x0[i];
        x0[i] = orig + (float)eps;
        cpu_fwd(x0, mix0, ref, bt);
        double lp = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lp += dy[j] * ref[j];
        }
        x0[i] = orig - (float)eps;
        cpu_fwd(x0, mix0, ref, bt);
        double lm = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lm += dy[j] * ref[j];
        }
        x0[i] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double rel = fabs(num - dx_snap[i]) / (fabs(num) + fabs(dx_snap[i]) + 1e-3);
        if (rel > dx_err) {
            dx_err = rel;
        }
    }

    double dmix_err = 0.0;
    for (uint32_t i = 0; i < g_d; i++) {
        float orig = mix0[i];
        mix0[i] = orig + (float)eps;
        cpu_fwd(x0, mix0, ref, bt);
        double lp = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lp += dy[j] * ref[j];
        }
        mix0[i] = orig - (float)eps;
        cpu_fwd(x0, mix0, ref, bt);
        double lm = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lm += dy[j] * ref[j];
        }
        mix0[i] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double rel = fabs(num - dmix_snap[i]) / (fabs(num) + fabs(dmix_snap[i]) + 1e-3);
        if (rel > dmix_err) {
            dmix_err = rel;
        }
    }

    printf("tokenshift: fwd err = %g, dx rel = %g, dmix rel = %g\n", ferr, dx_err, dmix_err);
    int ok = ferr < 1e-5 && dx_err < 1e-2 && dmix_err < 1e-2;
    printf("%s\n", ok ? "TOKENSHIFT OK" : "TOKENSHIFT WRONG");
    return ok ? 0 : 1;
}
