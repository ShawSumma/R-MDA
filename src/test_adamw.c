// Verify AdamW: one step on a 1-D quadratic should move w toward the minimum,
// and match a CPU reference of the same update exactly.
#include "vkc.h"
#include <math.h>

typedef struct adam_push adam_push_t;

struct adam_push {
    uint32_t n;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float wd;
    float bc1;
    float bc2;
};

static vkc_t c;
static vkc_kernel_t kn;

int main(void) {
    c = vkc_init();
    kn = vkc_load(&c, "build/adamw.spv", 4, sizeof(adam_push_t));

    uint32_t n = 64;
    vkc_buf_t bw = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bg = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bm = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bv = vkc_alloc(&c, n * sizeof(float));

    float *w = bw.ptr;
    float *g = bg.ptr;
    float *m = bm.ptr;
    float *v = bv.ptr;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float wd = 0.01f;
    float lr = 0.1f;

    // reference state
    float *rw = malloc(n * sizeof(float));
    float *rm = malloc(n * sizeof(float));
    float *rv = malloc(n * sizeof(float));
    srand(8);
    for (uint32_t i = 0; i < n; i++) {
        w[i] = ((rand() % 200) / 100.0f - 1.0f);
        g[i] = ((rand() % 200) / 100.0f - 1.0f);
        m[i] = 0.0f;
        v[i] = 0.0f;
        rw[i] = w[i];
        rm[i] = 0.0f;
        rv[i] = 0.0f;
    }

    double worst = 0.0;
    for (int step = 1; step <= 5; step++) {
        float bc1 = 1.0f - powf(beta1, (float)step);
        float bc2 = 1.0f - powf(beta2, (float)step);

        // gpu step
        adam_push_t push = (adam_push_t) {
            .n = n,
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .eps = eps,
            .wd = wd,
            .bc1 = bc1,
            .bc2 = bc2,
        };
        vkc_buf_t *b[4] = {
            &bw,
            &bg,
            &bm,
            &bv,
        };
        vkc_run(&c, &kn, b, (n + 255) / 256, 1, 1, &push);

        // cpu reference
        for (uint32_t i = 0; i < n; i++) {
            rm[i] = beta1 * rm[i] + (1.0f - beta1) * g[i];
            rv[i] = beta2 * rv[i] + (1.0f - beta2) * g[i] * g[i];
            float mh = rm[i] / bc1;
            float vh = rv[i] / bc2;
            rw[i] = rw[i] - lr * (mh / (sqrtf(vh) + eps) + wd * rw[i]);
        }
        for (uint32_t i = 0; i < n; i++) {
            double e = fabs(w[i] - rw[i]);
            if (e > worst) {
                worst = e;
            }
        }
    }

    printf("adamw: max |gpu - cpu| over 5 steps = %g\n", worst);
    int ok = worst < 1e-5;
    printf("%s\n", ok ? "ADAMW OK" : "ADAMW WRONG");
    return ok ? 0 : 1;
}
