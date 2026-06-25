// Verify silu and gelu forward (vs CPU) and backward (vs finite difference).
#include "vkc.h"
#include <math.h>

typedef struct elem_push elem_push_t;

struct elem_push {
    uint32_t n;
    uint32_t op;
};

static vkc_t c;
static vkc_kernel_t ke;
static uint32_t g_n = 257;

static void run_elem(vkc_buf_t *bx, vkc_buf_t *bpre, uint32_t op) {
    elem_push_t push = (elem_push_t) {
        .n = g_n,
        .op = op,
    };
    vkc_buf_t *b[2] = {
        bx,
        bpre,
    };
    vkc_run(&c, &ke, b, (g_n + 255) / 256, 1, 1, &push);
}

static float cpu_silu(float v) {
    return v * (1.0f / (1.0f + expf(-v)));
}

static float cpu_gelu(float v) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
}

static int check(uint32_t fwd_op, uint32_t bwd_op, float (*ref)(float), const char *name) {
    vkc_buf_t bx = vkc_alloc(&c, g_n * sizeof(float));
    vkc_buf_t bpre = vkc_alloc(&c, g_n * sizeof(float));
    float *x = bx.ptr;

    float *x0 = malloc(g_n * sizeof(float));
    for (uint32_t i = 0; i < g_n; i++) {
        x0[i] = ((rand() % 400) / 100.0f - 2.0f);
        x[i] = x0[i];
    }

    // forward
    run_elem(&bx, &bpre, fwd_op);
    double ferr = 0.0;
    for (uint32_t i = 0; i < g_n; i++) {
        double e = fabs(x[i] - ref(x0[i]));
        if (e > ferr) {
            ferr = e;
        }
    }

    // backward: dy = 1 everywhere, so dx should equal d(ref)/dx. compare to FD.
    // pre[] already holds x0 from the forward pass. Set x (now grad-in) to 1.
    for (uint32_t i = 0; i < g_n; i++) {
        x[i] = 1.0f;
    }
    run_elem(&bx, &bpre, bwd_op);

    double berr = 0.0;
    double eps = 1e-3;
    for (uint32_t i = 0; i < g_n; i++) {
        double num = (ref(x0[i] + (float)eps) - ref(x0[i] - (float)eps)) / (2.0 * eps);
        double rel = fabs(num - x[i]) / (fabs(num) + fabs(x[i]) + 1e-3);
        if (rel > berr) {
            berr = rel;
        }
    }
    free(x0);
    printf("%s: fwd err = %g, bwd rel err = %g\n", name, ferr, berr);
    return ferr < 1e-5 && berr < 1e-2;
}

int main(void) {
    c = vkc_init();
    ke = vkc_load(&c, "build/elem.spv", 2, sizeof(elem_push_t));
    srand(5);

    int ok = 1;
    ok &= check(0, 1, cpu_silu, "silu");
    ok &= check(2, 3, cpu_gelu, "gelu");
    printf("%s\n", ok ? "ELEM OK" : "ELEM WRONG");
    return ok ? 0 : 1;
}
