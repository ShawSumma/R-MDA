#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "vkc.h"
#include "pc.h"
#include "hash.h"
#include "util.h"

uint16_t mda_test_matmul_f16_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        return (uint16_t)sign;
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    } else {
        return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    }
}

bool mda_test_matmul_f16_check(const char *name, const float *ref, const float *got, uint32_t count) {
    double max_err = 0.0;
    double max_ref = 0.0;
    for (uint32_t i = 0; i < count; i += 1) {
        max_err = fmax(max_err, fabs(ref[i] - got[i]));
        max_ref = fmax(max_ref, fabs(ref[i]));
    }
    double rel = max_err / max_ref;
    bool ok = rel < 1e-2;
    printf("%s vs fp32: max_err/max_ref %.5f %s\n", name, rel, mda_util_verdict(ok));
    return ok;
}

void mda_test_matmul_f16_fill(mda_vkc_buf_t *b32, mda_vkc_buf_t *b16, uint32_t n, uint64_t key) {
    float *p = b32->ptr;
    uint16_t *h = b16->ptr;
    for (uint32_t i = 0; i < n; i += 1) {
        float v = (float)(mda_hash_u64(MDA_HASH_TEST, key, i) % 1000) / 500.0f - 1.0f;
        p[i] = v;
        h[i] = mda_test_matmul_f16_half(v);
    }
}

uint32_t mda_test_matmul_f16_poison(void) {
    float big = 1e30f;
    uint32_t bits;
    memcpy(&bits, &big, 4);
    return bits;
}

bool mda_test_matmul_f16_run(mda_vkc_t *c) {
    mda_vkc_kernel_t scalar = mda_vkc_load(c, "build/matmul.spv", 3, sizeof(mda_pc_mm_t));
    mda_vkc_kernel_t coop = mda_vkc_load(c, "build/matmul_f16.spv", 4, sizeof(mda_pc_mm16_t));
    mda_vkc_kernel_t wide = mda_vkc_load(c, "build/matmul_f16_wide.spv", 4, sizeof(mda_pc_mm16_t));
    mda_vkc_kernel_t absmax = mda_vkc_load(c, "build/absmax.spv", 2, sizeof(mda_pc_absmax_t));
    mda_vkc_kernel_t cast = mda_vkc_load(c, "build/cast_f16.spv", 3, sizeof(mda_pc_cast_t));
    uint32_t m = 512;
    uint32_t k = 640;
    uint32_t n = 1024;
    mda_vkc_buf_t a32 = mda_vkc_alloc(c, m * k * sizeof(float));
    mda_vkc_buf_t b32 = mda_vkc_alloc(c, k * n * sizeof(float));
    mda_vkc_buf_t c32 = mda_vkc_alloc(c, m * n * sizeof(float));
    mda_vkc_buf_t a16 = mda_vkc_alloc(c, m * k * sizeof(uint16_t));
    mda_vkc_buf_t b16 = mda_vkc_alloc(c, k * n * sizeof(uint16_t));
    mda_vkc_buf_t c16 = mda_vkc_alloc(c, m * n * sizeof(float));
    mda_vkc_buf_t amax = mda_vkc_alloc(c, sizeof(uint32_t));
    if (c->fail) {
        return false;
    } else {
        *(uint32_t *)amax.ptr = 0;
        mda_test_matmul_f16_fill(&a32, &a16, m * k, 0);
        mda_test_matmul_f16_fill(&b32, &b16, k * n, 1);
        mda_pc_mm_t pc = {
            .m = m,
            .n = n,
            .k = k,
        };
        mda_pc_mm16_t pc16 = {
            .m = m,
            .n = n,
            .k = k,
            .sa = MDA_PC_NO_SLOT,
            .sb = MDA_PC_NO_SLOT,
            .acc = 0,
        };
        mda_vkc_buf_t *sb[3] = {
            &a32,
            &b32,
            &c32,
        };
        mda_vkc_buf_t *cb[4] = {
            &a16,
            &b16,
            &c16,
            &amax,
        };
        mda_vkc_run(c, &scalar, sb, (n + 15) / 16, (m + 15) / 16, 1, &pc);
        mda_vkc_fill(c, &c16, mda_test_matmul_f16_poison());
        mda_vkc_run(c, &coop, cb, n / 64, m / 64, 1, &pc16);
        bool ok64 = mda_test_matmul_f16_check("coopmat f16 64x64", c32.ptr, c16.ptr, m * n);
        mda_vkc_fill(c, &c16, mda_test_matmul_f16_poison());
        mda_vkc_run(c, &wide, cb, n / 64, m / 256, 1, &pc16);
        bool ok256 = mda_test_matmul_f16_check("coopmat f16 256x64", c32.ptr, c16.ptr, m * n);

        float *ap = a32.ptr;
        for (uint32_t i = 0; i < m * k; i += 1) {
            ap[i] *= 1e5f;
        }
        mda_vkc_run(c, &scalar, sb, (n + 15) / 16, (m + 15) / 16, 1, &pc);
        mda_pc_absmax_t apc = {
            .n = m * k,
            .slot = 0,
        };
        mda_pc_cast_t cpc = {
            .n = m * k,
            .slot = 0,
        };
        mda_vkc_buf_t *ab[2] = {
            &a32,
            &amax,
        };
        mda_vkc_buf_t *castb[3] = {
            &a32,
            &a16,
            &amax,
        };
        mda_vkc_run(c, &absmax, ab, 256, 1, 1, &apc);
        mda_vkc_run(c, &cast, castb, (m * k + 255) / 256, 1, 1, &cpc);
        mda_pc_mm16_t pc16s = pc16;
        pc16s.sa = 0;
        mda_vkc_fill(c, &c16, mda_test_matmul_f16_poison());
        mda_vkc_run(c, &coop, cb, n / 64, m / 64, 1, &pc16s);
        bool ok_scaled = mda_test_matmul_f16_check("scaled coopmat f16 (|A| ~ 1e5)", c32.ptr, c16.ptr, m * n);

        uint32_t iters = 200;
        double t0 = mda_util_now();
        for (uint32_t it = 0; it < iters; it += 1) {
            mda_vkc_run(c, &scalar, sb, (n + 15) / 16, (m + 15) / 16, 1, &pc);
        }
        double t_scalar = (mda_util_now() - t0) / iters;
        double t1 = mda_util_now();
        for (uint32_t it = 0; it < iters; it += 1) {
            mda_vkc_run(c, &coop, cb, n / 64, m / 64, 1, &pc16);
        }
        double t_coop = (mda_util_now() - t1) / iters;
        double flops = 2.0 * m * n * k;
        printf("scalar fp32: %.3f ms (%.1f GFLOP/s)\n", t_scalar * 1e3, flops / t_scalar / 1e9);
        printf("coopmat f16: %.3f ms (%.1f GFLOP/s)\n", t_coop * 1e3, flops / t_coop / 1e9);
        printf("speedup: %.1fx\n", t_scalar / t_coop);
        return c->fail == false && ok64 && ok256 && ok_scaled;
    }
}

bool mda_test_matmul_f16_main(void) {
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_test_matmul_f16_run(&c);
    }
}

int main(void) {
    bool ok = mda_test_matmul_f16_main();
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
