#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "pc.h"
#include "hash.h"
#include "util.h"
#include "tokenizer.h"

uint32_t mda_model_pad_rank(uint32_t r) {
    return (r + MDA_MODEL_LORA_ALIGN - 1) / MDA_MODEL_LORA_ALIGN * MDA_MODEL_LORA_ALIGN;
}

mda_model_cfg_t mda_model_cfg(uint32_t vocab) {
    return (mda_model_cfg_t) {
        .d = 2560,
        .n_layer = 32,
        .n_head = 40,
        .seqlen = 256,
        .batch = 2,
        .ffn_hidden = 10240,
        .vocab = mda_tokenizer_padded(vocab),
        .block = 0,
        .lora = {
            mda_model_pad_rank(96),
            mda_model_pad_rank(96),
            mda_model_pad_rank(64),
            mda_model_pad_rank(320),
        },
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

uint32_t mda_model_dec_of(const char *mode) {
    const char *d = strchr(mode, 'd');
    if (mda_model_block_of(mode) == 0 || d == NULL) {
        return 0;
    } else {
        return (uint32_t)strtoul(d + 1, NULL, 10);
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

uint32_t mda_model_lora_max(mda_model_cfg_t cfg) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < MDA_MODEL_LORAS; i += 1) {
        r = mda_util_max_u32(r, cfg.lora[i]);
    }
    return r;
}

mda_model_param_t *mda_model_param(mda_model_t *m, uint32_t rows, uint32_t cols, mda_model_init_t init, float init_value, const char *name) {
    mda_model_param_t *p = &m->params[m->n_params];
    m->n_params += 1;
    *p = (mda_model_param_t) {
        .name = name,
        .rows = rows,
        .cols = cols,
        .n = rows * cols,
        .off = m->n_all,
        .init = init,
        .init_value = init_value,
        .lr_mul = 1.0f,
    };
    m->n_all += (p->n + MDA_MODEL_PARAM_ALIGN - 1) / MDA_MODEL_PARAM_ALIGN * MDA_MODEL_PARAM_ALIGN;
    return p;
}

mda_model_param_t *mda_model_param_mat(mda_model_t *m, uint32_t rows, uint32_t cols, mda_model_init_t init, float init_value, const char *name) {
    mda_model_param_t *p = mda_model_param(m, rows, cols, init, init_value, name);
    p->tmirror = true;
    return p;
}

mda_model_param_t *mda_model_param_accum(mda_model_t *m, uint32_t n, mda_model_init_t init, float init_value, const char *name) {
    mda_model_param_t *p = mda_model_param(m, 1, n, init, init_value, name);
    p->accum = true;
    return p;
}

float mda_model_init_value(const mda_model_param_t *p, uint32_t i, uint32_t j) {
    switch (p->init) {
    case MDA_MODEL_INIT_NORMAL:
        return p->init_value * mda_hash_normal(MDA_HASH_INIT, i, j);
    case MDA_MODEL_INIT_DECAY:
        return -6.5f + 5.0f * powf((float)j / (float)(p->n - 1), 0.85f + sqrtf(p->init_value));
    default:
        return p->init_value;
    }
}

void mda_model_views(mda_model_t *m) {
    float *w = calloc(m->n_all, sizeof(float));
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
        for (uint32_t j = 0; j < p->n; j += 1) {
            w[p->off + j] = mda_model_init_value(p, i, j);
        }
    }
    mda_vkc_write(m->c, &m->w_all, w);
    free(w);
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
    for (uint32_t i = 0; i < MDA_MODEL_LORAS; i += 1) {
        if (cfg.lora[i] == 0 || cfg.lora[i] % MDA_MODEL_LORA_ALIGN != 0) {
            fprintf(stderr, "config: low-rank width %u must be a positive multiple of %u\n", cfg.lora[i], MDA_MODEL_LORA_ALIGN);
            bad += 1;
        }
    }
    return bad == 0;
}

void mda_model_kernels(mda_model_t *m) {
    mda_vkc_t *c = m->c;
    m->k_embed = mda_vkc_load(c, "build/embed.spv", 4, sizeof(mda_pc_emb_t));
    m->k_tshift = mda_vkc_load(c, "build/tokenshift.spv", 7, sizeof(mda_pc_ts_t));
    m->k_wkv_f = mda_vkc_load(c, "build/wkv7_fwd.spv", 7, sizeof(mda_pc_wkv_t));
    m->k_wkv_b = mda_vkc_load(c, "build/wkv7_bwd.spv", 6, sizeof(mda_pc_wkv_bwd_t));
    m->k_mm = mda_vkc_load(c, "build/matmul.spv", 3, sizeof(mda_pc_mm_t));
    m->k_mm16 = mda_vkc_load(c, "build/matmul_f16.spv", 6, sizeof(mda_pc_mm16_t));
    m->k_mm16w = mda_vkc_load(c, "build/matmul_f16_wide.spv", 6, sizeof(mda_pc_mm16_t));
    m->k_cast = mda_vkc_load(c, "build/cast_f16.spv", 3, sizeof(mda_pc_cast_t));
    m->k_tr16 = mda_vkc_load(c, "build/transpose_f16.spv", 3, sizeof(mda_pc_tr16_t));
    m->k_absmax = mda_vkc_load(c, "build/absmax.spv", 2, sizeof(mda_pc_absmax_t));
    m->k_mm_bw = mda_vkc_load(c, "build/matmul_bw.spv", 4, sizeof(mda_pc_mm_bw_t));
    m->k_elem = mda_vkc_load(c, "build/elem.spv", 3, sizeof(mda_pc_elem_t));
    m->k_ln = mda_vkc_load(c, "build/layernorm.spv", 8, sizeof(mda_pc_ln_t));
    m->k_ce = mda_vkc_load(c, "build/softmax_ce.spv", 10, sizeof(mda_pc_ce_t));
    m->k_kd = mda_vkc_load(c, "build/kd_student.spv", 10, sizeof(mda_pc_kd_student_t));
    m->k_argmax = mda_vkc_load(c, "build/argmax.spv", 3, sizeof(mda_pc_argmax_t));
    m->k_softmax = mda_vkc_load(c, "build/softmax.spv", 2, sizeof(mda_pc_softmax_t));
    m->k_adamw = mda_vkc_load(c, "build/adamw.spv", 6, sizeof(mda_pc_adam_t));
    m->k_muon = mda_vkc_load(c, "build/muon.spv", 6, sizeof(mda_pc_muon_t));
    m->k_vec = mda_vkc_load(c, "build/vecops.spv", 2, sizeof(mda_pc_vec_t));
    m->k_repack = mda_vkc_load(c, "build/repack.spv", 2, sizeof(mda_pc_repack_t));
    m->k_sumsq = mda_vkc_load(c, "build/sumsq.spv", 3, sizeof(mda_pc_sumsq_t));
    m->k_heads_f = mda_vkc_load(c, "build/heads_fwd.spv", 8, sizeof(mda_pc_heads_t));
    m->k_heads_b = mda_vkc_load(c, "build/heads_bwd.spv", 14, sizeof(mda_pc_heads_bwd_t));
    m->k_wout = mda_vkc_load(c, "build/wkv_out.spv", 14, sizeof(mda_pc_wout_t));
    m->k_lora = mda_vkc_load(c, "build/lora_act.spv", 3, sizeof(mda_pc_lora_t));
    m->k_colsum = mda_vkc_load(c, "build/colsum.spv", 2, sizeof(mda_pc_colsum_t));
}

mda_model_lora_t mda_model_lora(mda_model_t *m, uint32_t rank, mda_model_act_t act, const char *a_name, const char *b_name) {
    uint32_t d = m->cfg.d;
    mda_model_param_t *b = NULL;
    switch (act) {
    case MDA_MODEL_ACT_SIGMOID:
        b = mda_model_param_mat(m, rank, d, MDA_MODEL_INIT_CONST, 2.0f / (float)rank, b_name);
        break;
    default:
        b = mda_model_param_mat(m, rank, d, MDA_MODEL_INIT_NORMAL, 0.1f / sqrtf((float)d), b_name);
        break;
    }
    return (mda_model_lora_t) {
        .a = mda_model_param_mat(m, d, rank, MDA_MODEL_INIT_CONST, 0.0f, a_name),
        .b = b,
        .rank = rank,
        .act = act,
    };
}

void mda_model_time_params(mda_model_t *m, mda_model_layer_t *ly, uint32_t l) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t d = cfg.d;
    float s = 0.02f;
    float ratio = (float)l / (float)mda_util_max_u32(cfg.n_layer - 1, 1);
    ly->ln1_w = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 1.0f, "ln1_w");
    ly->ln1_b = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 0.0f, "ln1_b");
    ly->att_mix = mda_model_param_accum(m, MDA_MODEL_MIXES * d, MDA_MODEL_INIT_CONST, 0.5f, "att_mix");
    ly->wr = mda_model_param_mat(m, d, d, MDA_MODEL_INIT_NORMAL, s, "wr");
    ly->wk = mda_model_param_mat(m, d, d, MDA_MODEL_INIT_NORMAL, s, "wk");
    ly->wv = mda_model_param_mat(m, d, d, MDA_MODEL_INIT_NORMAL, s, "wv");
    ly->wo = mda_model_param_mat(m, d, d, MDA_MODEL_INIT_NORMAL, s / sqrtf(2.0f * cfg.n_layer), "wo");
    ly->lora[MDA_MODEL_LORA_W] = mda_model_lora(m, cfg.lora[MDA_MODEL_LORA_W], MDA_MODEL_ACT_TANH, "lora_w_a", "lora_w_b");
    ly->lora[MDA_MODEL_LORA_A] = mda_model_lora(m, cfg.lora[MDA_MODEL_LORA_A], MDA_MODEL_ACT_NONE, "lora_a_a", "lora_a_b");
    ly->lora[MDA_MODEL_LORA_V] = mda_model_lora(m, cfg.lora[MDA_MODEL_LORA_V], MDA_MODEL_ACT_NONE, "lora_v_a", "lora_v_b");
    ly->lora[MDA_MODEL_LORA_G] = mda_model_lora(m, cfg.lora[MDA_MODEL_LORA_G], MDA_MODEL_ACT_SIGMOID, "lora_g_a", "lora_g_b");
    ly->w0 = mda_model_param_accum(m, d, MDA_MODEL_INIT_DECAY, ratio, "w0");
    ly->a0 = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 0.0f, "a0");
    ly->v0 = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, -1.0f, "v0");
    ly->kk = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 0.85f, "kk");
    ly->ka = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 1.0f, "ka");
    ly->rk = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 0.0f, "rk");
    ly->lnx_w = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 1.0f, "lnx_w");
    ly->lnx_b = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 0.0f, "lnx_b");
}

void mda_model_channel_params(mda_model_t *m, mda_model_layer_t *ly) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t d = cfg.d;
    float s = 0.02f;
    ly->ln2_w = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 1.0f, "ln2_w");
    ly->ln2_b = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 0.0f, "ln2_b");
    ly->ffn_mix = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 0.5f, "ffn_mix");
    ly->w_key = mda_model_param_mat(m, d, cfg.ffn_hidden, MDA_MODEL_INIT_NORMAL, s, "w_key");
    ly->w_val = mda_model_param_mat(m, cfg.ffn_hidden, d, MDA_MODEL_INIT_NORMAL, s / sqrtf(2.0f * cfg.n_layer), "w_val");
}

void mda_model_params(mda_model_t *m) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t d = cfg.d;
    float s = 0.02f;
    m->tok_emb = mda_model_param(m, cfg.vocab, d, MDA_MODEL_INIT_NORMAL, s, "tok_emb");
    m->tok_emb->accum = true;
    m->mask_emb = mda_model_param_accum(m, d, MDA_MODEL_INIT_NORMAL, s, "mask_emb");
    m->ln0_w = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 1.0f, "ln0_w");
    m->ln0_b = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 0.0f, "ln0_b");
    m->layers = malloc(cfg.n_layer * sizeof(mda_model_layer_t));
    for (uint32_t l = 0; l < cfg.n_layer; l += 1) {
        mda_model_time_params(m, &m->layers[l], l);
        mda_model_channel_params(m, &m->layers[l]);
    }
    m->ln_out_w = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 1.0f, "ln_out_w");
    m->ln_out_b = mda_model_param_accum(m, d, MDA_MODEL_INIT_CONST, 0.0f, "ln_out_b");
    m->head = mda_model_param_mat(m, d, cfg.vocab, MDA_MODEL_INIT_NORMAL, s, "head");
}

mda_model_t *mda_model_build(mda_vkc_t *c, mda_model_cfg_t cfg) {
    mda_model_t *m = calloc(1, sizeof(mda_model_t));
    m->c = c;
    m->cfg = cfg;
    m->live = cfg.batch;
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

bool mda_model_set_dec(mda_model_t *m, uint32_t dec) {
    mda_model_cfg_t cfg = m->cfg;
    if (dec > 0 && (dec >= cfg.n_layer || (mda_model_nclean(cfg) * cfg.seqlen) % 64 != 0)) {
        fprintf(stderr, "config: decoder layers %u must be fewer than %u, and the clean rows a multiple of 64\n", dec, cfg.n_layer);
        return false;
    } else {
        m->dec = dec;
        return true;
    }
}

uint64_t mda_model_param_count(const mda_model_t *m) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        total += m->params[i].n;
    }
    return total;
}
