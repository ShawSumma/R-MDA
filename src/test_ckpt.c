#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vkc.h"
#include "hash.h"
#include "util.h"
#include "model.h"
#include "ckpt.h"

bool mda_test_ckpt_same(const float *a, const mda_vkc_buf_t *b, uint32_t n) {
    return memcmp(a, b->ptr, n * sizeof(float)) == 0;
}

bool mda_test_ckpt_zero(const mda_vkc_buf_t *b, uint32_t n) {
    const float *p = b->ptr;
    uint32_t i = 0;
    while (i < n && p[i] == 0.0f) {
        i += 1;
    }
    return i == n;
}

mda_ckpt_res_t mda_test_ckpt_reload(mda_model_t *m, char *buf, size_t size, bool moments) {
    mda_vkc_buf_t *arenas[3] = {
        &m->w_all,
        &m->m_all,
        &m->v_all,
    };
    for (uint32_t i = 0; i < 3; i += 1) {
        mda_vkc_fill(m->c, arenas[i], 0);
    }
    FILE *r = fmemopen(buf, size, "rb");
    mda_ckpt_res_t res = mda_ckpt_read(m, 77, r, moments, "memory");
    fclose(r);
    return res;
}

bool mda_test_ckpt_model(mda_model_t *m) {
    uint32_t n = m->n_all;
    float *ref[3];
    mda_vkc_buf_t *arenas[3] = {
        &m->w_all,
        &m->m_all,
        &m->v_all,
    };
    for (uint32_t i = 0; i < 3; i += 1) {
        ref[i] = malloc(n * sizeof(float));
        for (uint32_t j = 0; j < n; j += 1) {
            ref[i][j] = (float)mda_hash_unit(MDA_HASH_TEST, i, j) - 0.5f;
        }
        memcpy(arenas[i]->ptr, ref[i], n * sizeof(float));
    }
    char *buf = NULL;
    size_t size = 0;
    FILE *w = open_memstream(&buf, &size);
    bool wrote = mda_ckpt_write(m, 77, 123, w);
    fclose(w);

    mda_ckpt_res_t res_full = mda_test_ckpt_reload(m, buf, size, true);
    bool full = res_full.ok && res_full.step == 123 && mda_test_ckpt_same(ref[0], arenas[0], n) && mda_test_ckpt_same(ref[1], arenas[1], n) && mda_test_ckpt_same(ref[2], arenas[2], n);
    printf("round trip with moments: %s\n", mda_util_verdict(full));

    mda_ckpt_res_t res_weights = mda_test_ckpt_reload(m, buf, size, false);
    bool weights = res_weights.ok && res_weights.step == 123 && mda_test_ckpt_same(ref[0], arenas[0], n) && mda_test_ckpt_zero(arenas[1], n) && mda_test_ckpt_zero(arenas[2], n);
    printf("weights only: %s\n", mda_util_verdict(weights));

    free(buf);
    for (uint32_t i = 0; i < 3; i += 1) {
        free(ref[i]);
    }
    return m->c->fail == false && wrote && full && weights;
}

bool mda_test_ckpt_device(mda_vkc_t *c) {
    mda_model_cfg_t cfg = {
        .d = 64,
        .n_layer = 2,
        .n_head = 1,
        .seqlen = 32,
        .batch = 2,
        .ffn_hidden = 128,
        .vocab = 64,
    };
    mda_model_t *m = mda_model_init(c, cfg);
    if (m == NULL) {
        return false;
    } else {
        return mda_test_ckpt_model(m);
    }
}

bool mda_test_ckpt_main(void) {
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_test_ckpt_device(&c);
    }
}

int main(void) {
    bool ok = mda_test_ckpt_main();
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
