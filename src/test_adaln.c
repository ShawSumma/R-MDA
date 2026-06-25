// Verify adaLN forward (vs CPU) and backward dx, dscale, dshift (vs finite diff).
#include "vkc.h"
#include <math.h>

typedef struct adaln_push adaln_push_t;

struct adaln_push {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
};

static vkc_t c;
static vkc_kernel_t kn;
static uint32_t g_b = 3;
static uint32_t g_seqlen = 4;
static uint32_t g_d = 6;

static void run_adaln(vkc_buf_t *bx, vkc_buf_t *bsc, vkc_buf_t *bsh,
    vkc_buf_t *bout, vkc_buf_t *bdx, vkc_buf_t *bdsc, vkc_buf_t *bdsh, uint32_t op) {
    uint32_t bt = g_b * g_seqlen;
    adaln_push_t push = (adaln_push_t) {
        .bt = bt,
        .d = g_d,
        .seqlen = g_seqlen,
        .op = op,
    };
    vkc_buf_t *b[7] = {
        bx,
        bsc,
        bsh,
        bout,
        bdx,
        bdsc,
        bdsh,
    };
    vkc_run(&c, &kn, b, (bt * g_d + 255) / 256, 1, 1, &push);
}

static void cpu_fwd(const float *x, const float *sc, const float *sh, float *out, uint32_t bt) {
    for (uint32_t row = 0; row < bt; row++) {
        uint32_t b = row / g_seqlen;
        for (uint32_t col = 0; col < g_d; col++) {
            uint32_t si = b * g_d + col;
            out[row * g_d + col] = x[row * g_d + col] * (1.0f + sc[si]) + sh[si];
        }
    }
}

int main(void) {
    c = vkc_init();
    kn = vkc_load(&c, "build/adaln.spv", 7, sizeof(adaln_push_t));

    uint32_t bt = g_b * g_seqlen;
    uint32_t n = bt * g_d;
    uint32_t ns = g_b * g_d;
    vkc_buf_t bx = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bsc = vkc_alloc(&c, ns * sizeof(float));
    vkc_buf_t bsh = vkc_alloc(&c, ns * sizeof(float));
    vkc_buf_t bout = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bdx = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bdsc = vkc_alloc(&c, ns * sizeof(float));
    vkc_buf_t bdsh = vkc_alloc(&c, ns * sizeof(float));

    float *x = bx.ptr;
    float *sc = bsc.ptr;
    float *sh = bsh.ptr;
    float *out = bout.ptr;
    float *dx = bdx.ptr;
    float *dsc = bdsc.ptr;
    float *dsh = bdsh.ptr;

    float *x0 = malloc(n * sizeof(float));
    float *sc0 = malloc(ns * sizeof(float));
    float *sh0 = malloc(ns * sizeof(float));
    float *dy = malloc(n * sizeof(float));
    srand(11);
    for (uint32_t i = 0; i < n; i++) {
        x[i] = ((rand() % 200) / 100.0f - 1.0f);
        x0[i] = x[i];
        dy[i] = ((rand() % 200) / 100.0f - 1.0f);
    }
    for (uint32_t i = 0; i < ns; i++) {
        sc[i] = ((rand() % 100) / 100.0f - 0.5f);
        sc0[i] = sc[i];
        sh[i] = ((rand() % 100) / 100.0f - 0.5f);
        sh0[i] = sh[i];
    }

    run_adaln(&bx, &bsc, &bsh, &bout, &bdx, &bdsc, &bdsh, 0);
    float *ref = malloc(n * sizeof(float));
    cpu_fwd(x0, sc0, sh0, ref, bt);
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
    for (uint32_t i = 0; i < ns; i++) {
        dsc[i] = 0.0f;
        dsh[i] = 0.0f;
    }
    run_adaln(&bx, &bsc, &bsh, &bout, &bdx, &bdsc, &bdsh, 1);

    float *dx_s = malloc(n * sizeof(float));
    float *dsc_s = malloc(ns * sizeof(float));
    float *dsh_s = malloc(ns * sizeof(float));
    for (uint32_t i = 0; i < n; i++) {
        dx_s[i] = dx[i];
    }
    for (uint32_t i = 0; i < ns; i++) {
        dsc_s[i] = dsc[i];
        dsh_s[i] = dsh[i];
    }

    double eps = 1e-3;
    double worst = 0.0;

    for (uint32_t i = 0; i < n; i++) {
        float orig = x0[i];
        x0[i] = orig + (float)eps;
        cpu_fwd(x0, sc0, sh0, ref, bt);
        double lp = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lp += dy[j] * ref[j];
        }
        x0[i] = orig - (float)eps;
        cpu_fwd(x0, sc0, sh0, ref, bt);
        double lm = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lm += dy[j] * ref[j];
        }
        x0[i] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double rel = fabs(num - dx_s[i]) / (fabs(num) + fabs(dx_s[i]) + 1e-3);
        if (rel > worst) {
            worst = rel;
        }
    }

    for (uint32_t i = 0; i < ns; i++) {
        float orig = sc0[i];
        sc0[i] = orig + (float)eps;
        cpu_fwd(x0, sc0, sh0, ref, bt);
        double lp = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lp += dy[j] * ref[j];
        }
        sc0[i] = orig - (float)eps;
        cpu_fwd(x0, sc0, sh0, ref, bt);
        double lm = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lm += dy[j] * ref[j];
        }
        sc0[i] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double rel = fabs(num - dsc_s[i]) / (fabs(num) + fabs(dsc_s[i]) + 1e-3);
        if (rel > worst) {
            worst = rel;
        }

        orig = sh0[i];
        sh0[i] = orig + (float)eps;
        cpu_fwd(x0, sc0, sh0, ref, bt);
        lp = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lp += dy[j] * ref[j];
        }
        sh0[i] = orig - (float)eps;
        cpu_fwd(x0, sc0, sh0, ref, bt);
        lm = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            lm += dy[j] * ref[j];
        }
        sh0[i] = orig;
        num = (lp - lm) / (2.0 * eps);
        rel = fabs(num - dsh_s[i]) / (fabs(num) + fabs(dsh_s[i]) + 1e-3);
        if (rel > worst) {
            worst = rel;
        }
    }

    printf("adaln: fwd err = %g, worst grad rel = %g\n", ferr, worst);
    int ok = ferr < 1e-5 && worst < 1e-2;
    printf("%s\n", ok ? "ADALN OK" : "ADALN WRONG");
    return ok ? 0 : 1;
}
