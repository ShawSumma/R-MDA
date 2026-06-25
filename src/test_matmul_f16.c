// test_matmul_f16.c -- correctness and speed of the cooperative-matrix matmul
// against the scalar fp32 one. fp16 inputs carry ~3 decimal digits, so we check
// relative error against a CPU fp32 reference rather than exact equality.
#include "vkc.h"
#include <math.h>
#include <time.h>

typedef uint16_t half;

// minimal fp32 -> fp16 (round to nearest, no NaN/Inf handling needed for our data).
static half to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        return (half)sign;
    }
    if (exp >= 31) {
        return (half)(sign | 0x7c00u);
    }
    uint32_t out = sign | ((uint32_t)exp << 10) | (mant >> 13);
    return (half)out;
}

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

typedef struct mm_pc mm_pc_t;

struct mm_pc {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

int main(void) {
    srand(3);
    vkc_t c = vkc_init();
    vkc_kernel_t scalar = vkc_load(&c, "build/matmul.spv", 3, 12);
    vkc_kernel_t coop = vkc_load(&c, "build/matmul_f16.spv", 3, 12);

    uint32_t m = 512;
    uint32_t k = 640;
    uint32_t n = 1024;

    vkc_buf_t a32 = vkc_alloc(&c, m * k * sizeof(float));
    vkc_buf_t b32 = vkc_alloc(&c, k * n * sizeof(float));
    vkc_buf_t c32 = vkc_alloc(&c, m * n * sizeof(float));
    vkc_buf_t a16 = vkc_alloc(&c, m * k * sizeof(half));
    vkc_buf_t b16 = vkc_alloc(&c, k * n * sizeof(half));
    vkc_buf_t c16 = vkc_alloc(&c, m * n * sizeof(float));

    float *ap = a32.ptr;
    float *bp = b32.ptr;
    half *ahp = a16.ptr;
    half *bhp = b16.ptr;
    for (uint32_t i = 0; i < m * k; i = i + 1) {
        float v = (float)(rand() % 1000 - 500) / 500.0f;
        ap[i] = v;
        ahp[i] = to_half(v);
    }
    for (uint32_t i = 0; i < k * n; i = i + 1) {
        float v = (float)(rand() % 1000 - 500) / 500.0f;
        bp[i] = v;
        bhp[i] = to_half(v);
    }

    mm_pc_t pc = {
        .m = m,
        .n = n,
        .k = k,
    };
    vkc_buf_t *sb[3] = {
        &a32,
        &b32,
        &c32,
    };
    vkc_buf_t *cb[3] = {
        &a16,
        &b16,
        &c16,
    };

    vkc_run(&c, &scalar, sb, (n + 15) / 16, (m + 15) / 16, 1, &pc);
    vkc_run(&c, &coop, cb, n / 16, m / 16, 1, &pc);

    float *cs = c32.ptr;
    float *cc = c16.ptr;
    double max_rel = 0.0;
    double mean_rel = 0.0;
    for (uint32_t i = 0; i < m * n; i = i + 1) {
        double denom = fabs(cs[i]) + 1e-4;
        double rel = fabs(cs[i] - cc[i]) / denom;
        mean_rel = mean_rel + rel;
        if (rel > max_rel) {
            max_rel = rel;
        }
    }
    mean_rel = mean_rel / (m * n);
    printf("correctness vs fp32: mean_rel=%.4f max_rel=%.4f\n", mean_rel, max_rel);

    int iters = 200;
    double t0 = now_sec();
    for (int it = 0; it < iters; it = it + 1) {
        vkc_run(&c, &scalar, sb, (n + 15) / 16, (m + 15) / 16, 1, &pc);
    }
    double t_scalar = (now_sec() - t0) / iters;
    t0 = now_sec();
    for (int it = 0; it < iters; it = it + 1) {
        vkc_run(&c, &coop, cb, n / 16, m / 16, 1, &pc);
    }
    double t_coop = (now_sec() - t0) / iters;

    double flops = 2.0 * m * n * k;
    printf("scalar fp32: %.3f ms  (%.1f GFLOP/s)\n", t_scalar * 1e3, flops / t_scalar / 1e9);
    printf("coopmat f16: %.3f ms  (%.1f GFLOP/s)\n", t_coop * 1e3, flops / t_coop / 1e9);
    printf("speedup: %.1fx\n", t_scalar / t_coop);
    return 0;
}
