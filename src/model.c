#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "pc.h"
#include "hash.h"

mda_model_cfg_t mda_model_cfg(uint32_t vocab) {
    return (mda_model_cfg_t) {
        .d = 640,
        .n_layer = 12,
        .n_head = 10,
        .seqlen = 256,
        .batch = 2,
        .ffn_hidden = 1728,
        .vocab = vocab,
        .block = 0,
    };
}

mda_model_cfg_t mda_model_cfg_block(mda_model_cfg_t cfg, uint32_t block) {
    mda_model_cfg_t b = cfg;
    b.block = block;
    b.batch = 2 * cfg.batch;
    return b;
}

uint32_t mda_model_block_of(const char *mode) {
    if (strncmp(mode, "block", 5) != 0) {
        return 0;
    } else if (mode[5] == '\0') {
        return MDA_MODEL_BLOCK;
    } else {
        return (uint32_t)strtoul(mode + 5, NULL, 10);
    }
}

mda_model_cfg_t mda_model_cfg_mode(uint32_t vocab, const char *mode, uint32_t batch) {
    mda_model_cfg_t base = mda_model_cfg(vocab);
    base.batch = batch;
    uint32_t block = mda_model_block_of(mode);
    if (block > 0) {
        return mda_model_cfg_block(base, block);
    } else {
        return base;
    }
}

uint32_t mda_model_nclean(mda_model_cfg_t cfg) {
    if (cfg.block > 0) {
        return cfg.batch / 2;
    } else {
        return 0;
    }
}

float mda_model_ada_scale(mda_model_cfg_t cfg) {
    return sqrtf(0.5f * (float)cfg.d);
}

mda_model_param_t *mda_model_param(mda_model_t *m, uint32_t rows, uint32_t cols, float init_std, float init_const, const char *name) {
    mda_model_param_t *p = &m->params[m->n_params];
    m->n_params += 1;
    *p = (mda_model_param_t) {
        .name = name,
        .rows = rows,
        .cols = cols,
        .n = rows * cols,
        .off = m->n_all,
        .init_std = init_std,
        .init_const = init_const,
        .lr_mul = 1.0f,
    };
    m->n_all += (p->n + MDA_MODEL_PARAM_ALIGN - 1) / MDA_MODEL_PARAM_ALIGN * MDA_MODEL_PARAM_ALIGN;
    return p;
}

mda_model_param_t *mda_model_param_mat(mda_model_t *m, uint32_t rows, uint32_t cols, float init_std, const char *name) {
    mda_model_param_t *p = mda_model_param(m, rows, cols, init_std, 0.0f, name);
    p->tmirror = true;
    return p;
}

mda_model_param_t *mda_model_param_accum(mda_model_t *m, uint32_t rows, uint32_t cols, float init_std, float init_const, const char *name) {
    mda_model_param_t *p = mda_model_param(m, rows, cols, init_std, init_const, name);
    p->accum = true;
    return p;
}

float mda_model_init_value(const mda_model_param_t *p, uint32_t i, uint32_t j) {
    if (p->init_std > 0.0f) {
        return p->init_std * mda_hash_normal(MDA_HASH_INIT, i, j);
    } else {
        return p->init_const;
    }
}

void mda_model_views(mda_model_t *m) {
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        mda_model_param_t *p = &m->params[i];
        VkDeviceSize off = p->off;
        p->w = mda_vkc_view(&m->w_all, off * sizeof(float), p->n * sizeof(float));
        p->g = mda_vkc_view(&m->g_all, off * sizeof(float), p->n * sizeof(float));
        p->m = mda_vkc_view(&m->m_all, off * sizeof(float), p->n * sizeof(float));
        p->v = mda_vkc_view(&m->v_all, off * sizeof(float), p->n * sizeof(float));
        p->w16 = mda_vkc_view(&m->w16_all, off * sizeof(uint16_t), p->n * sizeof(uint16_t));
        if (p->tmirror) {
            p->wt16 = mda_vkc_alloc(m->c, p->n * sizeof(uint16_t));
        }
        float *wp = p->w.ptr;
        for (uint32_t j = 0; j < p->n; j += 1) {
            wp[j] = mda_model_init_value(p, i, j);
        }
    }
    m->dirty16 = true;
}

void mda_model_alloc(mda_model_t *m) {
    mda_vkc_t *c = m->c;
    VkDeviceSize n = m->n_all;
    m->amax = mda_vkc_alloc(c, MDA_MODEL_MAX_SLOTS * sizeof(uint32_t));
    m->w_all = mda_vkc_alloc(c, n * sizeof(float));
    m->g_all = mda_vkc_alloc(c, n * sizeof(float));
    m->m_all = mda_vkc_alloc(c, n * sizeof(float));
    m->v_all = mda_vkc_alloc(c, n * sizeof(float));
    m->w16_all = mda_vkc_alloc(c, n * sizeof(uint16_t));
    mda_vkc_begin(c);
    mda_vkc_fill(c, &m->w_all, 0);
    mda_vkc_fill(c, &m->g_all, 0);
    mda_vkc_fill(c, &m->m_all, 0);
    mda_vkc_fill(c, &m->v_all, 0);
    mda_vkc_flush(c);
    if (c->fail == false) {
        mda_model_views(m);
    }
}

bool mda_model_cfg_check(mda_model_cfg_t cfg) {
    uint32_t bad = 0;
    if (cfg.n_head == 0 || cfg.d != 64 * cfg.n_head) {
        fprintf(stderr, "config: d / n_head must be 64 (d=%u n_head=%u)\n", cfg.d, cfg.n_head);
        bad += 1;
    }
    if (cfg.seqlen % MDA_MODEL_WKV_CHUNK != 0) {
        fprintf(stderr, "config: seqlen must be a multiple of %u (%u)\n", MDA_MODEL_WKV_CHUNK, cfg.seqlen);
        bad += 1;
    }
    if (cfg.block > 0 && (cfg.seqlen % cfg.block != 0 || cfg.block % MDA_MODEL_WKV_CHUNK != 0 || cfg.batch % 2 != 0)) {
        fprintf(stderr, "config: block %u must divide seqlen %u, be a multiple of %u, and batch %u must be even\n", cfg.block, cfg.seqlen, MDA_MODEL_WKV_CHUNK, cfg.batch);
        bad += 1;
    }
    if (cfg.ffn_hidden % 64 != 0) {
        fprintf(stderr, "config: ffn_hidden must be a multiple of 64 (%u)\n", cfg.ffn_hidden);
        bad += 1;
    }
    if (cfg.vocab % 64 != 0) {
        fprintf(stderr, "config: vocab must be a multiple of 64 (%u)\n", cfg.vocab);
        bad += 1;
    }
    if ((cfg.batch * cfg.seqlen) % 64 != 0) {
        fprintf(stderr, "config: batch * seqlen must be a multiple of 64 (%u)\n", cfg.batch * cfg.seqlen);
        bad += 1;
    }
    return bad == 0;
}

void mda_model_kernels(mda_model_t *m) {
    mda_vkc_t *c = m->c;
    m->k_embed = mda_vkc_load(c, "build/embed.spv", 4, sizeof(mda_pc_emb_t));
    m->k_tshift = mda_vkc_load(c, "build/tokenshift.spv", 7, sizeof(mda_pc_ts_t));
    m->k_wkv_f = mda_vkc_load(c, "build/wkv7_fwd.spv", 10, sizeof(mda_pc_wkv_t));
    m->k_wkv_b = mda_vkc_load(c, "build/wkv7_bwd.spv", 13, sizeof(mda_pc_wkv_bwd_t));
    m->k_mm = mda_vkc_load(c, "build/matmul.spv", 3, sizeof(mda_pc_mm_t));
    m->k_mm16 = mda_vkc_load(c, "build/matmul_f16.spv", 6, sizeof(mda_pc_mm16_t));
    m->k_mm16w = mda_vkc_load(c, "build/matmul_f16_wide.spv", 6, sizeof(mda_pc_mm16_t));
    m->k_cast = mda_vkc_load(c, "build/cast_f16.spv", 3, sizeof(mda_pc_cast_t));
    m->k_tr16 = mda_vkc_load(c, "build/transpose_f16.spv", 3, sizeof(mda_pc_tr16_t));
    m->k_absmax = mda_vkc_load(c, "build/absmax.spv", 2, sizeof(mda_pc_absmax_t));
    m->k_mm_bw = mda_vkc_load(c, "build/matmul_bw.spv", 4, sizeof(mda_pc_mm_bw_t));
    m->k_elem = mda_vkc_load(c, "build/elem.spv", 5, sizeof(mda_pc_elem_t));
    m->k_rms = mda_vkc_load(c, "build/rmsnorm.spv", 6, sizeof(mda_pc_rms_t));
    m->k_ce = mda_vkc_load(c, "build/softmax_ce.spv", 9, sizeof(mda_pc_ce_t));
    m->k_argmax = mda_vkc_load(c, "build/argmax.spv", 2, sizeof(mda_pc_argmax_t));
    m->k_softmax = mda_vkc_load(c, "build/softmax.spv", 2, sizeof(mda_pc_softmax_t));
    m->k_adamw = mda_vkc_load(c, "build/adamw.spv", 6, sizeof(mda_pc_adam_t));
    m->k_muon = mda_vkc_load(c, "build/muon.spv", 6, sizeof(mda_pc_muon_t));
    m->k_adaln = mda_vkc_load(c, "build/adaln.spv", 5, sizeof(mda_pc_adaln_t));
    m->k_vec = mda_vkc_load(c, "build/vecops.spv", 2, sizeof(mda_pc_vec_t));
    m->k_repack = mda_vkc_load(c, "build/repack.spv", 2, sizeof(mda_pc_repack_t));
    m->k_sumsq = mda_vkc_load(c, "build/sumsq.spv", 3, sizeof(mda_pc_sumsq_t));
    m->k_heads_f = mda_vkc_load(c, "build/heads_fwd.spv", 11, sizeof(mda_pc_heads_t));
    m->k_heads_b = mda_vkc_load(c, "build/heads_bwd.spv", 20, sizeof(mda_pc_heads_t));
    m->k_headnorm = mda_vkc_load(c, "build/headnorm.spv", 4, sizeof(mda_pc_hn_t));
}

void mda_model_params(mda_model_t *m) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    float s = 0.02f;
    float s_out = s / sqrtf(2.0f * cfg.n_layer);
    m->tok_emb = mda_model_param_mat(m, cfg.vocab, d, s, "tok_emb");
    m->tok_emb->accum = true;
    m->mask_emb = mda_model_param_accum(m, 1, d, s, 0.0f, "mask_emb");
    m->n_out = mda_model_param_accum(m, 1, d, 0.0f, 1.0f, "n_out");
    m->layers = malloc(cfg.n_layer * sizeof(mda_model_layer_t));
    for (uint32_t l = 0; l < cfg.n_layer; l += 1) {
        mda_model_layer_t *ly = &m->layers[l];
        ly->wk = mda_model_param_mat(m, d, d, s, "wk");
        ly->wv = mda_model_param_mat(m, d, d, s, "wv");
        ly->wr = mda_model_param_mat(m, d, d, s, "wr");
        ly->ww = mda_model_param_mat(m, d, d, s, "ww");
        ly->wa = mda_model_param_mat(m, d, d, s, "wa");
        ly->wo = mda_model_param_mat(m, d, d, s_out, "wo");
        ly->shift_mix = mda_model_param_accum(m, 1, d, 0.0f, 0.5f, "shift_mix");
        ly->n1 = mda_model_param_accum(m, 1, d, 0.0f, 1.0f, "n1");
        ly->n2 = mda_model_param_accum(m, 1, d, 0.0f, 1.0f, "n2");
        ly->w_gate = mda_model_param_mat(m, d, hid, s, "w_gate");
        ly->w_up = mda_model_param_mat(m, d, hid, s, "w_up");
        ly->w_down = mda_model_param_mat(m, hid, d, s_out, "w_down");
        ly->ada = mda_model_param_accum(m, 1, 4 * d, 0.0f, 0.0f, "ada");
        ly->ada->lr_mul = mda_model_ada_scale(cfg);
    }
}

mda_model_t *mda_model_build(mda_vkc_t *c, mda_model_cfg_t cfg) {
    mda_model_t *m = calloc(1, sizeof(mda_model_t));
    m->c = c;
    m->cfg = cfg;
    mda_model_kernels(m);
    mda_model_params(m);
    mda_model_alloc(m);
    if (c->fail) {
        return NULL;
    } else {
        return m;
    }
}

mda_model_t *mda_model_init(mda_vkc_t *c, mda_model_cfg_t cfg) {
    bool valid = mda_model_cfg_check(cfg);
    if (c->fail || valid == false) {
        return NULL;
    } else {
        return mda_model_build(c, cfg);
    }
}

uint64_t mda_model_param_count(const mda_model_t *m) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        total += m->params[i].n;
    }
    return total;
}
