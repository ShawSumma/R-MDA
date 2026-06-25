// model.h -- RWKV-7 + masked-diffusion denoiser assembled from the verified
// Vulkan kernels. Holds all weights/activations as GPU buffers, runs forward and
// backward by dispatching the kernels in order. fp32 throughout.
#ifndef MODEL_H
#define MODEL_H

#include "vkc.h"
#include "data.h"
#include <math.h>

// registry capacity (max distinct parameter tensors).
#define M_MAX_PARAMS 256

typedef struct mcfg mcfg_t;

struct mcfg {
    // model dim
    uint32_t d;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t seqlen;
    uint32_t batch;
    uint32_t ffn_hidden;
    // output vocab (subword count) and input vocab (vocab + 1 for the mask id).
    uint32_t vocab;
    uint32_t vocab_in;
};

// One trainable tensor: the weights plus its grad and AdamW moments, all same size.
typedef struct param param_t;

struct param {
    vkc_buf_t w;
    vkc_buf_t g;
    vkc_buf_t m;
    vkc_buf_t v;
    // fp16 mirror of w, refreshed once per step for the cooperative-matrix matmul.
    vkc_buf_t w16;
    // element count
    uint32_t n;
    const char *name;
};

// Push constants for the AdamW kernel: one update over n elements.
typedef struct adam_pc adam_pc_t;

struct adam_pc {
    uint32_t n;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float wd;
    float bc1;
    float bc2;
};

// Push constants for the sum-of-squares kernel: op 0 reduces, op 1 scales by scale.
typedef struct sumsq_pc sumsq_pc_t;

struct sumsq_pc {
    uint32_t n;
    uint32_t op;
    float scale;
};

// Push constants for the matmul-backward kernel: mode 0 makes dA, mode 1 makes dB.
typedef struct mm_bw_pc mm_bw_pc_t;

struct mm_bw_pc {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t mode;
};

typedef struct mlayer mlayer_t;

struct mlayer {
    param_t *wk;
    param_t *wv;
    param_t *wr;
    param_t *ww;
    param_t *wa;
    param_t *wo;
    param_t *shift_mix;
    param_t *n1;
    param_t *n2;
    param_t *w_gate;
    param_t *w_up;
    param_t *w_down;
    param_t *ada;
};

typedef struct model model_t;

struct model {
    vkc_t *c;
    mcfg_t cfg;

    vkc_kernel_t k_embed;
    vkc_kernel_t k_tshift;
    vkc_kernel_t k_wkv_f;
    vkc_kernel_t k_wkv_b;
    vkc_kernel_t k_mm;
    vkc_kernel_t k_mm16;
    vkc_kernel_t k_cast;
    vkc_kernel_t k_mm_bw;
    vkc_kernel_t k_elem;
    vkc_kernel_t k_rms;
    vkc_kernel_t k_adaln;
    vkc_kernel_t k_ce;
    vkc_kernel_t k_adamw;
    vkc_kernel_t k_vec;
    vkc_kernel_t k_repack;
    vkc_kernel_t k_sumsq;
    vkc_kernel_t k_normheads;

    param_t *tok_emb;
    param_t *pos_emb;
    param_t *t_proj;
    param_t *head;
    mlayer_t *layers;

    // flat registry of all params (for AdamW loop + counting)
    param_t params[M_MAX_PARAMS];
    uint32_t n_params;
    // set whenever fp32 weights change; the fp16 mirrors are re-cast lazily on the
    // next forward. Inference casts once; training sets it each optimizer step.
    int dirty16;
};

static void randn_fill(float *p, uint32_t n, float std) {
    for (uint32_t i = 0; i < n; i = i + 1) {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        p[i] = std * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
}

// register a new parameter (weights + grad + moments), returns a stable pointer.
static param_t *param_new(
    model_t *m,
    uint32_t n,
    float init_std,
    float init_const,
    const char *name) {
    param_t *p = &m->params[m->n_params];
    m->n_params = m->n_params + 1;
    p->n = n;
    p->name = name;
    p->w = vkc_alloc(m->c, n * sizeof(float));
    p->g = vkc_alloc(m->c, n * sizeof(float));
    p->m = vkc_alloc(m->c, n * sizeof(float));
    p->v = vkc_alloc(m->c, n * sizeof(float));
    p->w16 = vkc_alloc(m->c, n * sizeof(uint16_t));
    float *wp = p->w.ptr;
    if (init_std > 0.0f) {
        randn_fill(wp, n, init_std);
    } else {
        for (uint32_t i = 0; i < n; i = i + 1) {
            wp[i] = init_const;
        }
    }
    memset(p->m.ptr, 0, n * sizeof(float));
    memset(p->v.ptr, 0, n * sizeof(float));
    return p;
}

static model_t *model_init(vkc_t *c, mcfg_t cfg) {
    model_t *m = calloc(1, sizeof(model_t));
    m->c = c;
    m->cfg = cfg;

    m->k_embed = vkc_load(c, "build/embed.spv", 4, 16);
    m->k_tshift = vkc_load(c, "build/tokenshift.spv", 5, 16);
    m->k_wkv_f = vkc_load(c, "build/wkv7_fwd.spv", 7, 24);
    m->k_wkv_b = vkc_load(c, "build/wkv7_bwd.spv", 12, 16);
    m->k_mm = vkc_load(c, "build/matmul.spv", 3, 12);
    m->k_mm16 = vkc_load(c, "build/matmul_f16.spv", 3, 12);
    m->k_cast = vkc_load(c, "build/cast_f16.spv", 4, 8);
    m->k_mm_bw = vkc_load(c, "build/matmul_bw.spv", 4, 16);
    m->k_elem = vkc_load(c, "build/elem.spv", 2, 8);
    m->k_rms = vkc_load(c, "build/rmsnorm.spv", 5, 12);
    m->k_adaln = vkc_load(c, "build/adaln.spv", 7, 16);
    m->k_ce = vkc_load(c, "build/softmax_ce.spv", 5, 8);
    m->k_adamw = vkc_load(c, "build/adamw.spv", 4, 32);
    m->k_vec = vkc_load(c, "build/vecops.spv", 2, 8);
    m->k_repack = vkc_load(c, "build/repack.spv", 2, 20);
    m->k_sumsq = vkc_load(c, "build/sumsq.spv", 2, 12);
    m->k_normheads = vkc_load(c, "build/normheads.spv", 2, 12);

    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    float s = 0.02f;

    m->tok_emb = param_new(m, cfg.vocab_in * d, s, 0.0f, "tok_emb");
    m->pos_emb = param_new(m, cfg.seqlen * d, s, 0.0f, "pos_emb");
    m->t_proj = param_new(m, d * d, s, 0.0f, "t_proj");
    m->head = param_new(m, d * cfg.vocab, s, 0.0f, "head");

    m->layers = malloc(cfg.n_layer * sizeof(mlayer_t));
    for (uint32_t l = 0; l < cfg.n_layer; l = l + 1) {
        mlayer_t *ly = &m->layers[l];
        ly->wk = param_new(m, d * d, s, 0.0f, "wk");
        ly->wv = param_new(m, d * d, s, 0.0f, "wv");
        ly->wr = param_new(m, d * d, s, 0.0f, "wr");
        ly->ww = param_new(m, d * d, s, 0.0f, "ww");
        ly->wa = param_new(m, d * d, s, 0.0f, "wa");
        ly->wo = param_new(m, d * d, s, 0.0f, "wo");
        ly->shift_mix = param_new(m, d, 0.0f, 0.5f, "shift_mix");
        ly->n1 = param_new(m, d, 0.0f, 1.0f, "n1");
        ly->n2 = param_new(m, d, 0.0f, 1.0f, "n2");
        ly->w_gate = param_new(m, d * hid, s, 0.0f, "w_gate");
        ly->w_up = param_new(m, d * hid, s, 0.0f, "w_up");
        ly->w_down = param_new(m, hid * d, s, 0.0f, "w_down");
        ly->ada = param_new(m, d * 6 * d, 0.0f, 0.0f, "ada");
    }
    m->dirty16 = 1;
    return m;
}

static uint64_t model_param_count(const model_t *m) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < m->n_params; i = i + 1) {
        total = total + m->params[i].n;
    }
    return total;
}

#endif
