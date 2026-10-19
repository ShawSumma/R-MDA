#ifndef MDA_MODEL_H
#define MDA_MODEL_H

#include <stdbool.h>
#include <stdint.h>

#include "vkc.h"

#define MDA_MODEL_MAX_PARAMS 256
#define MDA_MODEL_MAX_SLOTS 1024
#define MDA_MODEL_PARAM_ALIGN 128
#define MDA_MODEL_WKV_CHUNK 4

typedef struct mda_model_cfg_t mda_model_cfg_t;
typedef struct mda_model_param_t mda_model_param_t;
typedef struct mda_model_layer_t mda_model_layer_t;
typedef struct mda_model_t mda_model_t;

struct mda_model_cfg_t {
    uint32_t d;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t seqlen;
    uint32_t batch;
    uint32_t ffn_hidden;
    uint32_t vocab;
};

struct mda_model_param_t {
    mda_vkc_buf_t w;
    mda_vkc_buf_t g;
    mda_vkc_buf_t m;
    mda_vkc_buf_t v;
    mda_vkc_buf_t w16;
    mda_vkc_buf_t wt16;
    const char *name;
    uint32_t rows;
    uint32_t cols;
    uint32_t n;
    uint32_t off;
    float init_std;
    float init_const;
    bool tmirror;
    bool accum;
};

struct mda_model_layer_t {
    mda_model_param_t *wk;
    mda_model_param_t *wv;
    mda_model_param_t *wr;
    mda_model_param_t *ww;
    mda_model_param_t *wa;
    mda_model_param_t *wo;
    mda_model_param_t *shift_mix;
    mda_model_param_t *n1;
    mda_model_param_t *n2;
    mda_model_param_t *w_gate;
    mda_model_param_t *w_up;
    mda_model_param_t *w_down;
    mda_model_param_t *ada;
};

struct mda_model_t {
    mda_vkc_t *c;
    mda_vkc_kernel_t k_embed;
    mda_vkc_kernel_t k_tshift;
    mda_vkc_kernel_t k_wkv_f;
    mda_vkc_kernel_t k_wkv_b;
    mda_vkc_kernel_t k_mm;
    mda_vkc_kernel_t k_mm16;
    mda_vkc_kernel_t k_mm16w;
    mda_vkc_kernel_t k_cast;
    mda_vkc_kernel_t k_tr16;
    mda_vkc_kernel_t k_absmax;
    mda_vkc_kernel_t k_mm_bw;
    mda_vkc_kernel_t k_elem;
    mda_vkc_kernel_t k_rms;
    mda_vkc_kernel_t k_ce;
    mda_vkc_kernel_t k_argmax;
    mda_vkc_kernel_t k_adamw;
    mda_vkc_kernel_t k_muon;
    mda_vkc_kernel_t k_temb;
    mda_vkc_kernel_t k_adaln;
    mda_vkc_kernel_t k_vec;
    mda_vkc_kernel_t k_repack;
    mda_vkc_kernel_t k_sumsq;
    mda_vkc_kernel_t k_heads_f;
    mda_vkc_kernel_t k_heads_b;
    mda_vkc_kernel_t k_headnorm;
    mda_model_param_t *tok_emb;
    mda_model_param_t *mask_emb;
    mda_model_param_t *n_out;
    mda_model_layer_t *layers;
    mda_model_param_t params[MDA_MODEL_MAX_PARAMS];
    mda_vkc_buf_t w_all;
    mda_vkc_buf_t g_all;
    mda_vkc_buf_t m_all;
    mda_vkc_buf_t v_all;
    mda_vkc_buf_t w16_all;
    mda_vkc_buf_t amax;
    mda_model_cfg_t cfg;
    uint32_t n_params;
    uint32_t n_all;
    uint32_t n_slots;
    bool dirty16;
    bool infer;
    bool ref32;
    bool grad_acc;
};

mda_model_cfg_t mda_model_cfg(uint32_t vocab);
mda_model_t *mda_model_init(mda_vkc_t *c, mda_model_cfg_t cfg);
uint64_t mda_model_param_count(const mda_model_t *m);

#endif
