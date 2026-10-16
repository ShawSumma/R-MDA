// test_matmul_f16.c -- correctness and speed of the cooperative-matrix matmuls (the
// 64x64-block kernel and the 256x64 wide one) against the scalar fp32 one. fp16 inputs
// carry ~3 decimal digits, so the check is max abs error relative to the largest
// reference value rather than exact equality. Exits nonzero on a mismatch.
#include "vkc.h"
#include <math.h>
#include <time.h>

typedef uint16_t half;

// minimal fp32 -> fp16 (truncating, no NaN/Inf handling needed for our data).
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

typedef struct mm_pc_t mm_pc_t;

struct mm_pc_t {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

// max |ref - got| / max |ref| over m*n outputs; prints it and returns pass/fail.
static int check(const char *name, const float *ref, const float *got, uint32_t count) {
    double max_err = 0.0;
    double max_ref = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        double err = fabs(ref[i] - got[i]);
        if (err > max_err) {
            max_err = err;
        }
        if (fabs(ref[i]) > max_ref) {
            max_ref = fabs(ref[i]);
        }
    }
    double rel = max_err / max_ref;
    int ok = rel < 1e-2;
    printf("%s vs fp32: max_err/max_ref=%.5f %s\n", name, rel, ok ? "ok" : "FAIL");
    return ok;
}

int main(void) {
    srand(3);
    vkc_t c = vkc_init();
    vkc_kernel_t scalar = vkc_load(&c, "build/matmul.spv", 3, 12);
    vkc_kernel_t coop = vkc_load(&c, "build/matmul_f16.spv", 3, 12);
    vkc_kernel_t wide = vkc_load(&c, "build/matmul_f16_wide.spv", 3, 12);

    // m a multiple of 256 (wide kernel rows), n of 64, k of 16
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
    for (uint32_t i = 0; i < m * k; ++i) {
        float v = (float)(rand() % 1000 - 500) / 500.0f;
        ap[i] = v;
        ahp[i] = to_half(v);
    }
    for (uint32_t i = 0; i < k * n; ++i) {
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
    int ok = 1;
    float *cc = c16.ptr;
    // poison the output so tiles a kernel fails to write show up as errors
    for (uint32_t i = 0; i < m * n; ++i) {
        cc[i] = 1e30f;
    }
    vkc_run(&c, &coop, cb, n / 64, m / 64, 1, &pc);
    ok = check("coopmat f16 64x64", c32.ptr, c16.ptr, m * n) && ok;
    for (uint32_t i = 0; i < m * n; ++i) {
        cc[i] = 1e30f;
    }
    vkc_run(&c, &wide, cb, n / 64, m / 256, 1, &pc);
    ok = check("coopmat f16 256x64", c32.ptr, c16.ptr, m * n) && ok;

    int iters = 200;
    double t0 = now_sec();
    for (int it = 0; it < iters; ++it) {
        vkc_run(&c, &scalar, sb, (n + 15) / 16, (m + 15) / 16, 1, &pc);
    }
    double t_scalar = (now_sec() - t0) / iters;
    t0 = now_sec();
    for (int it = 0; it < iters; ++it) {
        vkc_run(&c, &coop, cb, n / 64, m / 64, 1, &pc);
    }
    double t_coop = (now_sec() - t0) / iters;

    double flops = 2.0 * m * n * k;
    printf("scalar fp32: %.3f ms  (%.1f GFLOP/s)\n", t_scalar * 1e3, flops / t_scalar / 1e9);
    printf("coopmat f16: %.3f ms  (%.1f GFLOP/s)\n", t_coop * 1e3, flops / t_coop / 1e9);
    printf("speedup: %.1fx\n", t_scalar / t_coop);
    return ok ? 0 : 1;
}
