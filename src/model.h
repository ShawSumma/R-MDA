#ifndef MDA_MODEL_H
#define MDA_MODEL_H

#include <stdbool.h>
#include <stdint.h>

#include "vkc.h"

#define MDA_MODEL_MAX_PARAMS 1024
#define MDA_MODEL_MAX_SLOTS 4096
#define MDA_MODEL_PARAM_ALIGN 128
#define MDA_MODEL_WKV_CHUNK 4
#define MDA_MODEL_BLOCK 16
#define MDA_MODEL_LORAS 4

typedef uint32_t mda_model_init_t;
typedef uint32_t mda_model_act_t;
typedef struct mda_model_cfg_t mda_model_cfg_t;
typedef struct mda_model_param_t mda_model_param_t;
typedef struct mda_model_lora_t mda_model_lora_t;
typedef struct mda_model_layer_t mda_model_layer_t;
typedef struct mda_model_t mda_model_t;

enum {
    MDA_MODEL_INIT_NORMAL,
    MDA_MODEL_INIT_CONST,
    MDA_MODEL_INIT_DECAY,
};

enum {
    MDA_MODEL_ACT_NONE,
    MDA_MODEL_ACT_TANH,
    MDA_MODEL_ACT_SIGMOID,
};

enum {
    MDA_MODEL_LORA_W,
    MDA_MODEL_LORA_A,
    MDA_MODEL_LORA_V,
    MDA_MODEL_LORA_G,
};

struct mda_model_cfg_t {
    uint32_t d;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t seqlen;
    uint32_t batch;
    uint32_t ffn_hidden;
    uint32_t vocab;
    uint32_t block;
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
    mda_model_init_t init;
    float init_value;
    float lr_mul;
    bool tmirror;
    bool accum;
    bool frozen;
};

struct mda_model_lora_t {
    mda_model_param_t *a;
    mda_model_param_t *b;
    uint32_t rank;
    mda_model_act_t act;
};

struct mda_model_layer_t {
    mda_model_param_t *wk;
    mda_model_param_t *wv;
    mda_model_param_t *wr;
    mda_model_param_t *wo;
    mda_model_param_t *shift_mix;
    mda_model_param_t *n1;
    mda_model_param_t *n2;
    mda_model_param_t *w_gate;
    mda_model_param_t *w_up;
    mda_model_param_t *w_down;
    mda_model_param_t *ada;
    mda_model_param_t *w0;
    mda_model_param_t *a0;
    mda_model_param_t *v0;
    mda_model_param_t *kk;
    mda_model_param_t *ka;
    mda_model_param_t *rk;
    mda_model_lora_t lora[MDA_MODEL_LORAS];
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
    mda_vkc_kernel_t k_softmax;
    mda_vkc_kernel_t k_adamw;
    mda_vkc_kernel_t k_muon;
    mda_vkc_kernel_t k_adaln;
    mda_vkc_kernel_t k_vec;
    mda_vkc_kernel_t k_repack;
    mda_vkc_kernel_t k_sumsq;
    mda_vkc_kernel_t k_heads_f;
    mda_vkc_kernel_t k_heads_b;
    mda_vkc_kernel_t k_wout;
    mda_vkc_kernel_t k_lora;
    mda_vkc_kernel_t k_colsum;
    mda_vkc_kernel_t k_align;
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
    uint32_t live;
    uint32_t dec;
    bool dirty16;
    bool infer;
    bool ref32;
    bool grad_acc;
    bool causal;
};

mda_model_cfg_t mda_model_cfg(uint32_t vocab);
mda_model_cfg_t mda_model_cfg_block(mda_model_cfg_t cfg, uint32_t block);
uint32_t mda_model_block_of(const char *mode);
uint32_t mda_model_dec_of(const char *mode);
mda_model_cfg_t mda_model_cfg_mode(uint32_t vocab, const char *mode, uint32_t batch);
uint32_t mda_model_nclean(mda_model_cfg_t cfg);
float mda_model_ada_scale(mda_model_cfg_t cfg);
mda_model_t *mda_model_init(mda_vkc_t *c, mda_model_cfg_t cfg);
void mda_model_freeze_channel(mda_model_t *m);
bool mda_model_set_dec(mda_model_t *m, uint32_t dec);
uint64_t mda_model_param_count(const mda_model_t *m);

#endif
