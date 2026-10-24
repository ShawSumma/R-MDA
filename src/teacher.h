#ifndef MDA_TEACHER_H
#define MDA_TEACHER_H

#include <stdbool.h>
#include <stdint.h>

#include "vkc.h"
#include "model.h"
#include "safet.h"

#define MDA_TEACHER_ROPE_THETA 100000.0f
#define MDA_TEACHER_ALIGN 128

typedef struct mda_teacher_dims_t mda_teacher_dims_t;
typedef struct mda_teacher_layer_t mda_teacher_layer_t;
typedef struct mda_teacher_t mda_teacher_t;

struct mda_teacher_dims_t {
    uint32_t d;
    uint32_t kv;
    uint32_t ffn;
    uint32_t vocab;
    uint32_t n_layer;
};

struct mda_teacher_layer_t {
    uint64_t ln1;
    uint64_t ln2;
    uint64_t wq;
    uint64_t wk;
    uint64_t wv;
    uint64_t wo;
    uint64_t gate;
    uint64_t up;
    uint64_t down;
};

struct mda_teacher_t {
    mda_model_t *m;
    mda_teacher_layer_t *layers;
    mda_vkc_buf_t w32;
    mda_vkc_buf_t w16;
    mda_vkc_buf_t embt16;
    mda_vkc_buf_t x;
    mda_vkc_buf_t h;
    mda_vkc_buf_t h16;
    mda_vkc_buf_t q;
    mda_vkc_buf_t k;
    mda_vkc_buf_t v;
    mda_vkc_buf_t att;
    mda_vkc_buf_t att16;
    mda_vkc_buf_t gate;
    mda_vkc_buf_t up;
    mda_vkc_buf_t act;
    mda_vkc_buf_t act16;
    mda_vkc_buf_t rms;
    mda_vkc_buf_t logits;
    mda_vkc_buf_t logz;
    mda_vkc_buf_t xs;
    mda_vkc_buf_t ao;
    mda_vkc_kernel_t k_rope;
    mda_vkc_kernel_t k_attn;
    mda_teacher_dims_t dims;
    float teach;
    uint64_t emb;
    uint64_t norm;
    uint64_t n_all;
    uint32_t rows;
    uint32_t seqlen;
    bool keep;
};

mda_teacher_t *mda_teacher_load(mda_model_t *m, const char *path, uint32_t rows, bool keep);
mda_vkc_buf_t mda_teacher_layer_view(mda_teacher_t *t, mda_vkc_buf_t *b, uint32_t l);
void mda_teacher_rec(mda_teacher_t *t, mda_vkc_buf_t *ids);

#endif
