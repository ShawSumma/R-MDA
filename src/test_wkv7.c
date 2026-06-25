// Verify the WKV7 forward recurrence on GPU matches a CPU reference.
#include "vkc.h"
#include <math.h>

typedef struct wkv_push wkv_push_t;
typedef struct wkv_dims wkv_dims_t;

struct wkv_push {
    uint32_t bsz;
    uint32_t hd;
    uint32_t t;
    uint32_t dh;
    uint32_t dir;
    uint32_t save;
};

struct wkv_dims {
    uint32_t bsz;
    uint32_t hd;
    uint32_t t;
    uint32_t dh;
};

int main(void) {
    vkc_t c = vkc_init();

    wkv_dims_t d = (wkv_dims_t) {
        .bsz = 1,
        .hd = 1,
        .t = 8,
        .dh = 4,
    };
    uint32_t n = d.bsz * d.hd * d.t * d.dh;

    vkc_buf_t bk = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bv = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t br = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bw = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t ba = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bo = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bs = vkc_alloc(&c, 16 * sizeof(float));

    float *k = bk.ptr;
    float *v = bv.ptr;
    float *r = br.ptr;
    float *w = bw.ptr;
    float *a = ba.ptr;
    float *o = bo.ptr;

    srand(7);
    for (uint32_t i = 0; i < n; i++) {
        k[i] = ((rand() % 200) / 100.0f - 1.0f) * 0.5f;
        v[i] = ((rand() % 200) / 100.0f - 1.0f);
        r[i] = ((rand() % 200) / 100.0f - 1.0f) * 0.5f;
        w[i] = 0.9f + 0.09f * ((rand() % 100) / 100.0f);
        a[i] = 0.3f;
    }

    vkc_kernel_t kn = vkc_load(&c, "build/wkv7_fwd.spv", 7, sizeof(wkv_push_t));
    wkv_push_t push = (wkv_push_t) {
        .bsz = d.bsz,
        .hd = d.hd,
        .t = d.t,
        .dh = d.dh,
        .dir = 0,
        .save = 0,
    };
    vkc_buf_t *bufs[7] = {
        &bk,
        &bv,
        &br,
        &bw,
        &ba,
        &bo,
        &bs,
    };
    vkc_run(&c, &kn, bufs, d.bsz * d.hd, 1, 1, &push);

    float sref[16];
    for (uint32_t i = 0; i < 16; i++) {
        sref[i] = 0.0f;
    }
    double maxerr = 0.0;
    for (uint32_t t = 0; t < d.t; t++) {
        float sk[4];
        for (uint32_t j = 0; j < d.dh; j++) {
            float s = 0.0f;
            for (uint32_t i = 0; i < d.dh; i++) {
                s += sref[i * d.dh + j] * k[t * d.dh + i];
            }
            sk[j] = s;
        }
        float at = a[t * d.dh + 0];
        for (uint32_t i = 0; i < d.dh; i++) {
            for (uint32_t j = 0; j < d.dh; j++) {
                sref[i * d.dh + j] = w[t * d.dh + j] * sref[i * d.dh + j]
                    - at * k[t * d.dh + i] * sk[j]
                    + at * k[t * d.dh + i] * v[t * d.dh + j];
            }
        }
        for (uint32_t j = 0; j < d.dh; j++) {
            float oo = 0.0f;
            for (uint32_t i = 0; i < d.dh; i++) {
                oo += r[t * d.dh + i] * sref[i * d.dh + j];
            }
            double e = fabs(oo - o[t * d.dh + j]);
            if (e > maxerr) {
                maxerr = e;
            }
        }
    }
    printf("WKV7 fwd: max err vs CPU = %g\n", maxerr);
    printf("%s\n", maxerr < 1e-4 ? "WKV7 FWD OK" : "WKV7 FWD WRONG");
    return maxerr < 1e-4 ? 0 : 1;
}
