// model.h -- RWKV-7 + masked-diffusion denoiser assembled from the verified
// Vulkan kernels. Holds all weights/activations as GPU buffers, runs forward and
// backward by dispatching the kernels in order. fp32 throughout.
#ifndef MODEL_H
#define MODEL_H

#include "vkc.h"
#include "data.h"
#include <math.h>

#define M_VOCAB 256
#define M_VOCAB_IN 257       // + MASK id
#define M_DIFF_STEPS 1000
#define M_MAX_PARAMS 256     // registry capacity

typedef struct mcfg mcfg_t;

struct mcfg {
    uint32_t d;              // model dim
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t seqlen;
    uint32_t batch;
    uint32_t ffn_hidden;
};

// One trainable tensor: the weights plus its grad and AdamW moments, all same size.
typedef struct param param_t;

struct param {
    vkc_buf_t w;
    vkc_buf_t g;
    vkc_buf_t m;
    vkc_buf_t v;
    uint32_t n;             // element count
    const char *name;
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
    vkc_kernel_t k_mm_bw;
    vkc_kernel_t k_elem;
    vkc_kernel_t k_rms;
    vkc_kernel_t k_adaln;
    vkc_kernel_t k_ce;
    vkc_kernel_t k_adamw;

    param_t *tok_emb;
    param_t *pos_emb;
    param_t *t_proj;
    param_t *head;
    mlayer_t *layers;

    // flat registry of all params (for AdamW loop + counting)
    param_t params[M_MAX_PARAMS];
    uint32_t n_params;
};

static void randn_fill(float *p, uint32_t n, float std) {
    for (uint32_t i = 0; i < n; i++) {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        p[i] = std * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
}

// register a new parameter (weights + grad + moments), returns a stable pointer.
static param_t *param_new(model_t *m, uint32_t n, float init_std, float init_const,
    const char *name) {
    param_t *p = &m->params[m->n_params++];
    p->n = n;
    p->name = name;
    p->w = vkc_alloc(m->c, n * sizeof(float));
    p->g = vkc_alloc(m->c, n * sizeof(float));
    p->m = vkc_alloc(m->c, n * sizeof(float));
    p->v = vkc_alloc(m->c, n * sizeof(float));
    float *wp = p->w.ptr;
    if (init_std > 0.0f) {
        randn_fill(wp, n, init_std);
    } else {
        for (uint32_t i = 0; i < n; i++) {
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
    m->k_mm_bw = vkc_load(c, "build/matmul_bw.spv", 4, 16);
    m->k_elem = vkc_load(c, "build/elem.spv", 2, 8);
    m->k_rms = vkc_load(c, "build/rmsnorm.spv", 5, 12);
    m->k_adaln = vkc_load(c, "build/adaln.spv", 7, 16);
    m->k_ce = vkc_load(c, "build/softmax_ce.spv", 5, 8);
    m->k_adamw = vkc_load(c, "build/adamw.spv", 4, 32);

    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    float s = 0.02f;

    m->tok_emb = param_new(m, M_VOCAB_IN * d, s, 0.0f, "tok_emb");
    m->pos_emb = param_new(m, cfg.seqlen * d, s, 0.0f, "pos_emb");
    m->t_proj = param_new(m, d * d, s, 0.0f, "t_proj");
    m->head = param_new(m, d * M_VOCAB, s, 0.0f, "head");

    m->layers = malloc(cfg.n_layer * sizeof(mlayer_t));
    for (uint32_t l = 0; l < cfg.n_layer; l++) {
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
    return m;
}

static uint64_t model_param_count(const model_t *m) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < m->n_params; i++) {
        total += m->params[i].n;
    }
    return total;
}

#endif
