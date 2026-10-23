#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

bool mda_test_matmul_f16_check_buf(mda_vkc_t *c, const char *name, const float *ref, mda_vkc_buf_t *b, uint32_t count) {
    float *got = mda_vkc_read(c, b);
    bool ok = mda_test_matmul_f16_check(name, ref, got, count);
    free(got);
    return ok;
}

float *mda_test_matmul_f16_values(uint32_t n, uint64_t key) {
    float *v = malloc(n * sizeof(float));
    for (uint32_t i = 0; i < n; i += 1) {
        v[i] = (float)(mda_hash_u64(MDA_HASH_TEST, key, i) % 1000) / 500.0f - 1.0f;
    }
    return v;
}

uint16_t *mda_test_matmul_f16_halves(const float *v, uint32_t n) {
    uint16_t *h = malloc(n * sizeof(uint16_t));
    for (uint32_t i = 0; i < n; i += 1) {
        h[i] = mda_test_matmul_f16_half(v[i]);
    }
    return h;
}

void mda_test_matmul_f16_upload(mda_vkc_t *c, mda_vkc_buf_t *b32, mda_vkc_buf_t *b16, const float *v, uint32_t n) {
    uint16_t *h = mda_test_matmul_f16_halves(v, n);
    mda_vkc_write(c, b32, v);
    mda_vkc_write(c, b16, h);
    free(h);
}

void mda_test_matmul_f16_upload_t(mda_vkc_t *c, mda_vkc_buf_t *dst, const float *v, uint32_t rows, uint32_t cols) {
    uint16_t *h = mda_test_matmul_f16_halves(v, rows * cols);
    uint16_t *t = malloc(rows * cols * sizeof(uint16_t));
    for (uint32_t r = 0; r < rows; r += 1) {
        for (uint32_t col = 0; col < cols; col += 1) {
            t[col * rows + r] = h[r * cols + col];
        }
    }
    mda_vkc_write(c, dst, t);
    free(h);
    free(t);
}

uint32_t mda_test_matmul_f16_poison(void) {
    float big = 1e30f;
    uint32_t bits;
    memcpy(&bits, &big, 4);
    return bits;
}

void mda_test_matmul_f16_dispatch(mda_vkc_t *c, mda_vkc_kernel_t *kern, uint32_t tile, mda_vkc_buf_t **bufs, mda_pc_mm16_t *pc) {
    mda_vkc_run(c, kern, bufs, (pc->n + tile - 1) / tile, (pc->m + tile - 1) / tile, 1, pc);
}

bool mda_test_matmul_f16_mode(mda_vkc_t *c, mda_vkc_kernel_t *kern, uint32_t tile, mda_vkc_buf_t **bufs, mda_pc_mm16_t pc, const float *ref, const char *name) {
    mda_vkc_fill(c, bufs[2], mda_test_matmul_f16_poison());
    mda_test_matmul_f16_dispatch(c, kern, tile, bufs, &pc);
    char label[96];
    snprintf(label, sizeof(label), "coopmat f16 %ux%u tiles, %ux%ux%u %s", tile, tile, pc.m, pc.n, pc.k, name);
    return mda_test_matmul_f16_check_buf(c, label, ref, bufs[2], pc.m * pc.n);
}

float mda_test_matmul_f16_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits = sign;
    if (exp != 0) {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

bool mda_test_matmul_f16_epi(mda_vkc_t *c, mda_vkc_kernel_t *kern, uint32_t tile, mda_vkc_buf_t **bufs, mda_pc_mm16_t pc, const float *ref) {
    uint32_t count = pc.m * pc.n;
    mda_vkc_write(c, bufs[5], ref);
    mda_vkc_fill(c, bufs[2], mda_test_matmul_f16_poison());
    mda_pc_mm16_t pe = pc;
    pe.alpha = 0.5f;
    pe.beta = 2.0f;
    pe.store = 3;
    mda_test_matmul_f16_dispatch(c, kern, tile, bufs, &pe);
    float *want = malloc(count * sizeof(float));
    float *half = malloc(count * sizeof(float));
    uint16_t *h = mda_vkc_read(c, bufs[4]);
    for (uint32_t i = 0; i < count; i += 1) {
        want[i] = 2.5f * ref[i];
        half[i] = mda_test_matmul_f16_float(h[i]);
    }
    char label[96];
    snprintf(label, sizeof(label), "coopmat f16 %ux%u tiles, 0.5 AB + 2 P, fp32 out", tile, tile);
    bool ok32 = mda_test_matmul_f16_check_buf(c, label, want, bufs[2], count);
    snprintf(label, sizeof(label), "coopmat f16 %ux%u tiles, 0.5 AB + 2 P, fp16 out", tile, tile);
    bool ok16 = mda_test_matmul_f16_check(label, want, half, count);
    free(want);
    free(half);
    free(h);
    return ok32 && ok16;
}

bool mda_test_matmul_f16_layouts(mda_vkc_t *c, mda_vkc_kernel_t *scalar, mda_vkc_kernel_t *kern, uint32_t tile, uint32_t m, uint32_t n, uint32_t k) {
    mda_vkc_buf_t a32 = mda_vkc_alloc(c, m * k * sizeof(float));
    mda_vkc_buf_t b32 = mda_vkc_alloc(c, k * n * sizeof(float));
    mda_vkc_buf_t c32 = mda_vkc_alloc(c, m * n * sizeof(float));
    mda_vkc_buf_t a16 = mda_vkc_alloc(c, m * k * sizeof(uint16_t));
    mda_vkc_buf_t b16 = mda_vkc_alloc(c, k * n * sizeof(uint16_t));
    mda_vkc_buf_t at16 = mda_vkc_alloc(c, m * k * sizeof(uint16_t));
    mda_vkc_buf_t bt16 = mda_vkc_alloc(c, k * n * sizeof(uint16_t));
    mda_vkc_buf_t c16 = mda_vkc_alloc(c, m * n * sizeof(float));
    mda_vkc_buf_t h16 = mda_vkc_alloc(c, m * n * sizeof(uint16_t));
    mda_vkc_buf_t p32 = mda_vkc_alloc(c, m * n * sizeof(float));
    mda_vkc_buf_t amax = mda_vkc_alloc(c, sizeof(uint32_t));
    if (c->fail) {
        return false;
    } else {
        float *va = mda_test_matmul_f16_values(m * k, 2);
        float *vb = mda_test_matmul_f16_values(k * n, 3);
        mda_test_matmul_f16_upload(c, &a32, &a16, va, m * k);
        mda_test_matmul_f16_upload(c, &b32, &b16, vb, k * n);
        mda_test_matmul_f16_upload_t(c, &at16, va, m, k);
        mda_test_matmul_f16_upload_t(c, &bt16, vb, k, n);
        free(va);
        free(vb);
        mda_pc_mm_t spc = {
            .m = m,
            .n = n,
            .k = k,
        };
        mda_vkc_buf_t *sb[3] = {
            &a32,
            &b32,
            &c32,
        };
        mda_vkc_run(c, scalar, sb, (n + 15) / 16, (m + 15) / 16, 1, &spc);
        float *ref = mda_vkc_read(c, &c32);
        mda_pc_mm16_t pc = {
            .m = m,
            .n = n,
            .k = k,
            .sa = MDA_PC_NO_SLOT,
            .sb = MDA_PC_NO_SLOT,
            .store = 1,
            .alpha = 1.0f,
        };
        mda_vkc_buf_t *plain[6] = {
            &a16,
            &b16,
            &c16,
            &amax,
            &h16,
            &c16,
        };
        mda_vkc_buf_t *ta[6] = {
            &at16,
            &b16,
            &c16,
            &amax,
            &h16,
            &c16,
        };
        mda_vkc_buf_t *tb[6] = {
            &a16,
            &bt16,
            &c16,
            &amax,
            &h16,
            &c16,
        };
        mda_vkc_buf_t *epi[6] = {
            &a16,
            &b16,
            &c16,
            &amax,
            &h16,
            &p32,
        };
        mda_pc_mm16_t pa = pc;
        pa.ta = 1;
        mda_pc_mm16_t pb = pc;
        pb.tb = 1;
        bool ok0 = mda_test_matmul_f16_mode(c, kern, tile, plain, pc, ref, "A B");
        bool ok1 = mda_test_matmul_f16_mode(c, kern, tile, ta, pa, ref, "At B");
        bool ok2 = mda_test_matmul_f16_mode(c, kern, tile, tb, pb, ref, "A Bt");
        bool ok3 = mda_test_matmul_f16_epi(c, kern, tile, epi, pc, ref);
        free(ref);
        return c->fail == false && ok0 && ok1 && ok2 && ok3;
    }
}

bool mda_test_matmul_f16_run(mda_vkc_t *c) {
    mda_vkc_kernel_t scalar = mda_vkc_load(c, "build/matmul.spv", 3, sizeof(mda_pc_mm_t));
    mda_vkc_kernel_t coop = mda_vkc_load(c, "build/matmul_f16.spv", 6, sizeof(mda_pc_mm16_t));
    mda_vkc_kernel_t wide = mda_vkc_load(c, "build/matmul_f16_wide.spv", 6, sizeof(mda_pc_mm16_t));
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
        mda_vkc_fill(c, &amax, 0);
        float *va = mda_test_matmul_f16_values(m * k, 0);
        float *vb = mda_test_matmul_f16_values(k * n, 1);
        mda_test_matmul_f16_upload(c, &a32, &a16, va, m * k);
        mda_test_matmul_f16_upload(c, &b32, &b16, vb, k * n);
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
            .store = 1,
            .alpha = 1.0f,
        };
        mda_vkc_buf_t *sb[3] = {
            &a32,
            &b32,
            &c32,
        };
        mda_vkc_buf_t *cb[6] = {
            &a16,
            &b16,
            &c16,
            &amax,
            &c16,
            &c16,
        };
        mda_vkc_run(c, &scalar, sb, (n + 15) / 16, (m + 15) / 16, 1, &pc);
        float *ref = mda_vkc_read(c, &c32);
        mda_vkc_fill(c, &c16, mda_test_matmul_f16_poison());
        mda_vkc_run(c, &coop, cb, n / 64, m / 64, 1, &pc16);
        bool ok64 = mda_test_matmul_f16_check_buf(c, "coopmat f16 64x64", ref, &c16, m * n);
        mda_vkc_fill(c, &c16, mda_test_matmul_f16_poison());
        mda_vkc_run(c, &wide, cb, n / 128, m / 128, 1, &pc16);
        bool ok128 = mda_test_matmul_f16_check_buf(c, "coopmat f16 128x128", ref, &c16, m * n);
        free(ref);

        for (uint32_t i = 0; i < m * k; i += 1) {
            va[i] *= 1e5f;
        }
        mda_vkc_write(c, &a32, va);
        free(va);
        free(vb);
        mda_vkc_run(c, &scalar, sb, (n + 15) / 16, (m + 15) / 16, 1, &pc);
        float *ref_big = mda_vkc_read(c, &c32);
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
        bool ok_scaled = mda_test_matmul_f16_check_buf(c, "scaled coopmat f16 (|A| ~ 1e5)", ref_big, &c16, m * n);
        free(ref_big);

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
        bool ok_t0 = mda_test_matmul_f16_layouts(c, &scalar, &coop, 64, 512, 1024, 640);
        bool ok_t1 = mda_test_matmul_f16_layouts(c, &scalar, &coop, 64, 576, 1728, 640);
        bool ok_t2 = mda_test_matmul_f16_layouts(c, &scalar, &coop, 64, 64, 2560, 640);
        bool ok_t3 = mda_test_matmul_f16_layouts(c, &scalar, &wide, 128, 512, 1024, 640);
        bool ok_t4 = mda_test_matmul_f16_layouts(c, &scalar, &wide, 128, 576, 1728, 640);
        return c->fail == false && ok64 && ok128 && ok_scaled && ok_t0 && ok_t1 && ok_t2 && ok_t3 && ok_t4;
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
