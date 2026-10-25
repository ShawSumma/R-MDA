#include <stdio.h>
#include <stdlib.h>

#include "fwd.h"
#include "util.h"
#include "bwd.h"

mda_fwd_lora_act_t mda_fwd_lora_alloc(mda_model_t *m, const mda_model_lora_t *l) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
    VkDeviceSize n = (VkDeviceSize)bt * l->rank;
    mda_fwd_lora_act_t a = {
        .h = mda_vkc_alloc(m->c, n * sizeof(float)),
        .hf16 = mda_vkc_alloc(m->c, n * sizeof(uint16_t)),
    };
    switch (l->act) {
    case MDA_MODEL_ACT_NONE:
        a.hf = a.h;
        break;
    default:
        a.hf = mda_vkc_alloc(m->c, n * sizeof(float));
        break;
    }
    return a;
}

mda_vkc_buf_t mda_fwd_section(mda_vkc_buf_t *b, mda_model_cfg_t cfg, uint32_t i) {
    VkDeviceSize n = (VkDeviceSize)cfg.batch * cfg.seqlen * cfg.d * sizeof(float);
    return mda_vkc_view(b, i * n, n);
}

mda_vkc_buf_t mda_fwd_section16(mda_vkc_buf_t *b, mda_model_cfg_t cfg, uint32_t i) {
    VkDeviceSize n = (VkDeviceSize)cfg.batch * cfg.seqlen * cfg.d * sizeof(uint16_t);
    return mda_vkc_view(b, i * n, n);
}

mda_fwd_act_t mda_fwd_act_alloc(mda_model_t *m, mda_model_layer_t *ly) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t hid = cfg.ffn_hidden;
    uint32_t dh = cfg.d / cfg.n_head;
    uint32_t ns = cfg.batch * cfg.n_head * cfg.seqlen / MDA_MODEL_WKV_CHUNK * dh * dh;
    VkDeviceSize n = (VkDeviceSize)bt * cfg.d * sizeof(float);
    VkDeviceSize n16 = (VkDeviceSize)bt * cfg.d * sizeof(uint16_t);
    VkDeviceSize nh = (VkDeviceSize)bt * hid;
    mda_vkc_t *c = m->c;
    mda_fwd_act_t a = {
        .x_in = mda_vkc_alloc(c, n),
        .st1 = mda_vkc_alloc(c, 2 * bt * sizeof(float)),
        .n1 = mda_vkc_alloc(c, n),
        .shift = mda_vkc_alloc(c, MDA_MODEL_MIXES * n),
        .shift16 = mda_vkc_alloc(c, MDA_MODEL_MIXES * n16),
        .kv = mda_vkc_alloc(c, 2 * n),
        .heads = mda_vkc_alloc(c, 6 * n),
        .kinv = mda_vkc_alloc(c, bt * cfg.n_head * sizeof(float)),
        .bs = mda_vkc_alloc(c, bt * cfg.n_head * sizeof(float)),
        .vg = mda_vkc_alloc(c, n),
        .g = mda_vkc_alloc(c, n),
        .sstate = mda_vkc_alloc(c, (VkDeviceSize)ns * sizeof(float)),
        .sstate2 = mda_vkc_alloc(c, (VkDeviceSize)ns * sizeof(float)),
        .sk = mda_vkc_alloc(c, n),
        .sk2 = mda_vkc_alloc(c, n),
        .wkv_raw = mda_vkc_alloc(c, n),
        .p = mda_vkc_alloc(c, n),
        .wkv = mda_vkc_alloc(c, n),
        .stx = mda_vkc_alloc(c, 2 * bt * cfg.n_head * sizeof(float)),
        .x_mid = mda_vkc_alloc(c, n),
        .st2 = mda_vkc_alloc(c, 2 * bt * sizeof(float)),
        .n2 = mda_vkc_alloc(c, n),
        .fshift = mda_vkc_alloc(c, n),
        .fshift16 = mda_vkc_alloc(c, n16),
        .kpre = mda_vkc_alloc(c, nh * sizeof(float)),
        .h = mda_vkc_alloc(c, nh * sizeof(float)),
        .wkv16 = mda_vkc_alloc(c, n16),
        .h16 = mda_vkc_alloc(c, nh * sizeof(uint16_t)),
    };
    for (uint32_t i = 0; i < MDA_MODEL_LORAS; i += 1) {
        a.lora[i] = mda_fwd_lora_alloc(m, &ly->lora[i]);
    }
    return a;
}

mda_data_batch_t mda_fwd_batch(mda_model_cfg_t cfg, mda_vkc_buf_t *ids, mda_vkc_buf_t *target, mda_vkc_buf_t *mask, mda_vkc_buf_t *weight) {
    return (mda_data_batch_t) {
        .ids = ids->ptr,
        .target = target->ptr,
        .mask = mask->ptr,
        .weight = weight->ptr,
        .batch = cfg.batch,
        .seqlen = cfg.seqlen,
        .nclean = mda_model_nclean(cfg),
    };
}

uint32_t mda_fwd_block_states(mda_model_cfg_t cfg) {
    if (cfg.block > 0) {
        return mda_model_nclean(cfg) * cfg.n_head * (cfg.seqlen / cfg.block);
    } else {
        return 0;
    }
}

VkDeviceSize mda_fwd_cache_state_bytes(mda_model_cfg_t cfg) {
    return (VkDeviceSize)mda_model_nclean(cfg) * cfg.n_head * 4096 * sizeof(float);
}

VkDeviceSize mda_fwd_cache_carry_bytes(mda_model_cfg_t cfg) {
    return (VkDeviceSize)mda_model_nclean(cfg) * cfg.d * sizeof(float);
}

void mda_fwd_cache_reset(mda_model_t *m, mda_fwd_cache_t *cache) {
    for (uint32_t i = 0; i < 2; i += 1) {
        mda_vkc_fill(m->c, &cache->state[i], 0);
        mda_vkc_fill(m->c, &cache->carry[i], 0);
    }
    cache->cur = 0;
}

mda_fwd_cache_t mda_fwd_cache_alloc(mda_model_t *m) {
    mda_model_cfg_t cfg = m->cfg;
    mda_fwd_cache_t cache;
    for (uint32_t i = 0; i < 2; i += 1) {
        cache.state[i] = mda_vkc_alloc(m->c, cfg.n_layer * mda_fwd_cache_state_bytes(cfg));
        cache.carry[i] = mda_vkc_alloc(m->c, 2 * cfg.n_layer * mda_fwd_cache_carry_bytes(cfg));
    }
    mda_fwd_cache_reset(m, &cache);
    return cache;
}

void mda_fwd_cache_commit(mda_fwd_cache_t *cache) {
    cache->cur = 1 - cache->cur;
}

mda_vkc_buf_t mda_fwd_cache_view(mda_fwd_scratch_t *sc, mda_vkc_buf_t *pair, mda_vkc_buf_t *fallback, uint32_t which, uint32_t l, VkDeviceSize per) {
    if (sc->cache == NULL) {
        return *fallback;
    } else {
        return mda_vkc_view(&pair[which], l * per, per);
    }
}

mda_fwd_scratch_t mda_fwd_alloc(mda_model_t *m, uint32_t n_acts) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    mda_vkc_t *c = m->c;
    mda_fwd_act_t *acts = malloc(n_acts * sizeof(mda_fwd_act_t));
    for (uint32_t i = 0; i < n_acts; i += 1) {
        acts[i] = mda_fwd_act_alloc(m, &m->layers[i]);
    }
    VkDeviceSize nstate = (VkDeviceSize)mda_util_max_u32(mda_fwd_block_states(cfg), 1) * 4096 * sizeof(float);
    VkDeviceSize n = (VkDeviceSize)bt * d * sizeof(float);
    uint32_t widest = mda_util_max_u32(hid, cfg.vocab);
    uint32_t nkd = mda_fwd_kd_rows(m);
    mda_vkc_buf_t ids = mda_vkc_alloc_cached(c, bt * sizeof(uint32_t));
    mda_vkc_buf_t target = mda_vkc_alloc_cached(c, bt * sizeof(uint32_t));
    mda_vkc_buf_t mask = mda_vkc_alloc_cached(c, bt * sizeof(float));
    mda_vkc_buf_t weight = mda_vkc_alloc_cached(c, bt * sizeof(float));
    mda_fwd_scratch_t sc = {
        .opt = mda_vkc_alloc_cached(c, sizeof(mda_fwd_opt_t)),
        .ids = ids,
        .target = target,
        .mask = mask,
        .weight = weight,
        .emb = mda_vkc_alloc(c, n),
        .st0 = mda_vkc_alloc(c, 2 * bt * sizeof(float)),
        .x = mda_vkc_alloc(c, n),
        .bstate = mda_vkc_alloc(c, nstate),
        .dbstate = mda_vkc_alloc(c, nstate),
        .stn = mda_vkc_alloc(c, 2 * bt * sizeof(float)),
        .dg_dummy = mda_vkc_alloc(c, d * sizeof(float)),
        .raw = mda_vkc_alloc(c, 6 * n),
        .v0 = mda_vkc_alloc(c, n),
        .po = mda_vkc_alloc(c, n),
        .po2 = mda_vkc_alloc(c, n),
        .xn = mda_vkc_alloc(c, n),
        .logits = mda_vkc_alloc(c, (VkDeviceSize)bt * cfg.vocab * sizeof(float)),
        .loss = mda_vkc_alloc_cached(c, bt * sizeof(float)),
        .logz = mda_vkc_alloc(c, bt * sizeof(float)),
        .tok = mda_vkc_alloc_cached(c, bt * sizeof(uint32_t)),
        .conf = mda_vkc_alloc_cached(c, bt * sizeof(float)),
        .dnorm = mda_vkc_alloc(c, n),
        .dtmp = mda_vkc_alloc(c, n),
        .dtmp2 = mda_vkc_alloc(c, n),
        .dh = mda_vkc_alloc(c, (VkDeviceSize)bt * hid * sizeof(float)),
        .dshift = mda_vkc_alloc(c, MDA_MODEL_MIXES * n),
        .dheads = {
            mda_vkc_alloc(c, 6 * n),
            mda_vkc_alloc(c, 6 * n),
        },
        .dv0 = mda_vkc_alloc(c, n),
        .con = mda_vkc_alloc(c, 3 * n),
        .dbs = mda_vkc_alloc(c, bt * cfg.n_head * sizeof(float)),
        .dvb = mda_vkc_alloc(c, n),
        .dgate = mda_vkc_alloc(c, n),
        .dln = mda_vkc_alloc(c, 2 * n),
        .dlora = mda_vkc_alloc(c, (VkDeviceSize)bt * mda_model_lora_max(cfg) * sizeof(float)),
        .dwkv = mda_vkc_alloc(c, n),
        .dxh_buf = mda_vkc_alloc(c, n),
        .a16 = mda_vkc_alloc(c, (VkDeviceSize)bt * d * sizeof(uint16_t)),
        .dc16 = mda_vkc_alloc(c, (VkDeviceSize)bt * widest * sizeof(uint16_t)),
        .kd_toks = mda_vkc_alloc(c, (VkDeviceSize)cfg.vocab * 5 * sizeof(uint32_t)),
        .kd_cont = mda_vkc_alloc_cached(c, (VkDeviceSize)mda_util_max_u32(nkd, 1) * 5 * sizeof(uint32_t)),
        .kd_tgt = mda_vkc_alloc_cached(c, (VkDeviceSize)mda_util_max_u32(nkd, 1) * MDA_KD_ENTRY * sizeof(uint32_t)),
        .dkd = mda_vkc_alloc(c, (VkDeviceSize)mda_util_max_u32(nkd, 1) * cfg.vocab * sizeof(float)),
        .kd_loss = mda_vkc_alloc_cached(c, (VkDeviceSize)mda_util_max_u32(nkd, 1) * sizeof(float)),
        .kd_bits = mda_vkc_alloc_cached(c, (VkDeviceSize)mda_util_max_u32(nkd, 1) * 2 * sizeof(float)),
        .batch = mda_fwd_batch(cfg, &ids, &target, &mask, &weight),
        .acts = acts,
        .n_acts = n_acts,
    };
    if (c->fail == false) {
        *(mda_fwd_opt_t *)sc.opt.ptr = (mda_fwd_opt_t) {
            .gmul = 1.0f,
        };
    }
    return sc;
}

void mda_fwd_vec_op(mda_model_t *m, mda_vkc_buf_t *a, mda_vkc_buf_t *b, uint32_t n, mda_pc_vec_op_t op) {
    mda_pc_vec_t pc = {
        .n = n,
        .op = op,
    };
    mda_vkc_buf_t *bufs[2] = {
        a,
        b,
    };
    mda_vkc_run(m->c, &m->k_vec, bufs, (n + 255) / 256, 1, 1, &pc);
}

uint32_t mda_fwd_rows(const mda_model_t *m) {
    return m->live * m->cfg.seqlen;
}

bool mda_fwd_split(const mda_model_t *m) {
    return m->cfg.block > 0 && m->causal == false && m->dec > 0 && m->dec < m->cfg.n_layer;
}

uint32_t mda_fwd_dec_start(const mda_model_t *m) {
    if (mda_fwd_split(m)) {
        return m->cfg.n_layer - m->dec;
    } else {
        return 0;
    }
}

bool mda_fwd_clean_only(const mda_model_t *m) {
    return m->cfg.block > 0 && m->live < m->cfg.batch;
}

void mda_fwd_set_live(mda_model_t *m, uint32_t l) {
    if (l < mda_fwd_dec_start(m)) {
        m->live = mda_model_nclean(m->cfg);
    } else {
        m->live = m->cfg.batch;
    }
}

uint32_t mda_fwd_first_row(const mda_model_t *m, uint32_t l) {
    if (l == 0) {
        return 0;
    } else if (l == mda_fwd_dec_start(m)) {
        return mda_model_nclean(m->cfg) * m->cfg.seqlen;
    } else {
        return UINT32_MAX;
    }
}

void mda_fwd_repack(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t op) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = mda_fwd_rows(m);
    mda_pc_repack_t pc = {
        .bt = bt,
        .hd = cfg.n_head,
        .dh = cfg.d / cfg.n_head,
        .seqlen = cfg.seqlen,
        .op = op,
    };
    mda_vkc_buf_t *bufs[2] = {
        src,
        dst,
    };
    mda_vkc_run(m->c, &m->k_repack, bufs, (bt * cfg.d + 255) / 256, 1, 1, &pc);
}

void mda_fwd_mm(mda_model_t *m, mda_vkc_buf_t *a, mda_vkc_buf_t *b, mda_vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k) {
    mda_pc_mm_t pc = {
        .m = rows,
        .n = n,
        .k = k,
    };
    mda_vkc_buf_t *bufs[3] = {
        a,
        b,
        out,
    };
    mda_vkc_run(m->c, &m->k_mm, bufs, (n + 15) / 16, (rows + 15) / 16, 1, &pc);
}

void mda_fwd_slots_begin(mda_model_t *m) {
    m->n_slots = 0;
    mda_vkc_fill(m->c, &m->amax, 0);
}

uint32_t mda_fwd_slot(mda_model_t *m) {
    if (m->n_slots == MDA_MODEL_MAX_SLOTS) {
        fprintf(stderr, "out of fp16 scale slots (MDA_MODEL_MAX_SLOTS)\n");
        m->c->fail = true;
        return MDA_PC_NO_SLOT;
    } else {
        uint32_t slot = m->n_slots;
        m->n_slots += 1;
        return slot;
    }
}

uint32_t mda_fwd_absmax(mda_model_t *m, mda_vkc_buf_t *a, uint32_t n) {
    uint32_t slot = mda_fwd_slot(m);
    mda_pc_absmax_t pc = {
        .n = n,
        .slot = slot,
    };
    mda_vkc_buf_t *bufs[2] = {
        a,
        &m->amax,
    };
    mda_vkc_run(m->c, &m->k_absmax, bufs, mda_util_min_u32((n + 255) / 256, 256), 1, 1, &pc);
    return slot;
}

void mda_fwd_cast(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t n, uint32_t slot) {
    mda_pc_cast_t pc = {
        .n = n,
        .slot = slot,
    };
    mda_vkc_buf_t *bufs[3] = {
        src,
        dst,
        &m->amax,
    };
    mda_vkc_run(m->c, &m->k_cast, bufs, (n + 255) / 256, 1, 1, &pc);
}

uint32_t mda_fwd_cast_act(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t n) {
    uint32_t slot = mda_fwd_absmax(m, src, n);
    mda_fwd_cast(m, src, dst, n, slot);
    return slot;
}

void mda_fwd_elem(mda_model_t *m, mda_vkc_buf_t *x, mda_vkc_buf_t *pre, uint32_t op, uint32_t slot) {
    uint32_t n = mda_fwd_rows(m) * m->cfg.ffn_hidden;
    mda_pc_elem_t pc = {
        .n = n,
        .op = op,
        .slot = slot,
    };
    mda_vkc_buf_t *bufs[3] = {
        x,
        pre,
        &m->amax,
    };
    mda_vkc_run(m->c, &m->k_elem, bufs, mda_util_min_u32((n + 255) / 256, MDA_FWD_ELEM_GROUPS), 1, 1, &pc);
}

void mda_fwd_transpose16(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t rows, uint32_t cols, uint32_t slot) {
    mda_pc_tr16_t pc = {
        .rows = rows,
        .cols = cols,
        .slot = slot,
    };
    mda_vkc_buf_t *bufs[3] = {
        src,
        dst,
        &m->amax,
    };
    mda_vkc_run(m->c, &m->k_tr16, bufs, cols / 32, rows / 32, 1, &pc);
}

void mda_fwd_refresh_wt16_rec(mda_model_t *m) {
    uint32_t last = 0;
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        if (m->params[i].tmirror) {
            last = i;
        }
    }
    mda_vkc_set_nobar(m->c, true);
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        mda_model_param_t *p = &m->params[i];
        if (p->tmirror) {
            mda_vkc_set_nobar(m->c, i != last);
            mda_fwd_transpose16(m, &p->w, &p->wt16, p->rows, p->cols, MDA_PC_NO_SLOT);
        }
    }
}

void mda_fwd_cast_all(mda_model_t *m) {
    for (uint64_t lo = 0; lo < m->n_all; lo += MDA_FWD_CHUNK) {
        uint32_t n = (uint32_t)mda_util_min_size(MDA_FWD_CHUNK, m->n_all - lo);
        mda_vkc_buf_t w = mda_vkc_view(&m->w_all, lo * sizeof(float), (VkDeviceSize)n * sizeof(float));
        mda_vkc_buf_t w16 = mda_vkc_view(&m->w16_all, lo * sizeof(uint16_t), (VkDeviceSize)n * sizeof(uint16_t));
        mda_fwd_cast(m, &w, &w16, n, MDA_PC_NO_SLOT);
    }
}

void mda_fwd_refresh_w16(mda_model_t *m) {
    if (m->dirty16) {
        mda_vkc_begin(m->c);
        mda_fwd_cast_all(m);
        mda_fwd_refresh_wt16_rec(m);
        mda_vkc_flush(m->c);
        m->dirty16 = false;
    }
}

void mda_fwd_mm16_epi(mda_model_t *m, mda_vkc_buf_t *x16, uint32_t sx, mda_vkc_buf_t *y16, uint32_t sy, mda_vkc_buf_t *out, mda_vkc_buf_t *out16, uint32_t rows, uint32_t n, uint32_t k, mda_fwd_layout_t layout, mda_fwd_epi_t epi) {
    mda_pc_mm16_t pc = {
        .m = rows,
        .n = n,
        .k = k,
        .sa = sx,
        .sb = sy,
        .ta = (uint32_t)(layout == MDA_FWD_AT),
        .tb = (uint32_t)(layout == MDA_FWD_BT),
        .store = epi.store,
        .alpha = epi.alpha,
        .beta = epi.beta,
    };
    mda_vkc_buf_t *bufs[6] = {
        x16,
        y16,
        out,
        &m->amax,
        out16,
        epi.prev,
    };
    uint32_t gx = (n + 127) / 128;
    uint32_t gy = (rows + 127) / 128;
    if (gx * gy >= MDA_FWD_WIDE_GROUPS) {
        mda_vkc_run(m->c, &m->k_mm16w, bufs, gx, gy, 1, &pc);
    } else {
        mda_vkc_run(m->c, &m->k_mm16, bufs, n / 64, rows / 64, 1, &pc);
    }
}

void mda_fwd_mm16_raw(mda_model_t *m, mda_vkc_buf_t *x16, uint32_t sx, mda_vkc_buf_t *y16, uint32_t sy, mda_vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k, bool acc, mda_fwd_layout_t layout) {
    mda_fwd_epi_t epi = {
        .prev = out,
        .alpha = 1.0f,
        .beta = (float)acc,
        .store = MDA_FWD_STORE32,
    };
    mda_fwd_mm16_epi(m, x16, sx, y16, sy, out, out, rows, n, k, layout, epi);
}

void mda_fwd_mm16_pre(mda_model_t *m, mda_vkc_buf_t *a, mda_vkc_buf_t *a16, uint32_t slot, mda_model_param_t *p, mda_vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k) {
    if (m->ref32) {
        mda_fwd_mm(m, a, &p->w, out, rows, n, k);
    } else {
        mda_fwd_mm16_raw(m, a16, slot, &p->w16, MDA_PC_NO_SLOT, out, rows, n, k, false, MDA_FWD_AB);
    }
}

void mda_fwd_mm16_res(mda_model_t *m, mda_vkc_buf_t *a, mda_vkc_buf_t *a16, uint32_t slot, mda_model_param_t *p, mda_vkc_buf_t *prev, mda_vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k) {
    if (m->ref32) {
        mda_fwd_mm(m, a, &p->w, out, rows, n, k);
        mda_fwd_vec_op(m, out, prev, rows * n, MDA_PC_VEC_ADD);
    } else {
        mda_fwd_epi_t epi = {
            .prev = prev,
            .alpha = 1.0f,
            .beta = 1.0f,
            .store = MDA_FWD_STORE32,
        };
        mda_fwd_mm16_epi(m, a16, slot, &p->w16, MDA_PC_NO_SLOT, out, out, rows, n, k, MDA_FWD_AB, epi);
    }
}

uint32_t mda_fwd_n_vec(const mda_model_t *m) {
    return mda_fwd_rows(m) * m->cfg.n_head;
}

void mda_fwd_layernorm(mda_model_t *m, mda_vkc_buf_t *x, mda_model_param_t *w, mda_model_param_t *b, mda_vkc_buf_t *y, mda_vkc_buf_t *stats, uint32_t rows, uint32_t op, mda_vkc_buf_t *dacc) {
    mda_pc_ln_t pc = {
        .rows = rows,
        .d = m->cfg.d,
        .op = op,
        .eps = MDA_FWD_LN_EPS,
    };
    mda_vkc_buf_t *bufs[8] = {
        x,
        &w->w,
        &b->w,
        y,
        stats,
        &w->g,
        &b->g,
        dacc,
    };
    mda_vkc_run(m->c, &m->k_ln, bufs, rows, 1, 1, &pc);
}

void mda_fwd_tshift(mda_model_t *m, mda_fwd_scratch_t *sc, mda_vkc_buf_t *x, mda_model_param_t *mix, mda_vkc_buf_t *out, uint32_t nmix, uint32_t op, uint32_t carry, uint32_t l, mda_vkc_buf_t *dx) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = mda_fwd_rows(m);
    mda_pc_ts_t pc = {
        .bt = bt,
        .d = cfg.d,
        .seqlen = cfg.seqlen,
        .op = op,
        .blk = cfg.block,
        .nclean = mda_model_nclean(cfg),
        .cache = sc->cache != NULL && op == 0,
        .nmix = nmix,
        .sec = cfg.batch * cfg.seqlen * cfg.d,
        .rows_per = MDA_FWD_SUM_ROWS,
    };
    uint32_t cur = 0;
    mda_vkc_buf_t *pair = &sc->dg_dummy;
    if (sc->cache != NULL) {
        cur = sc->cache->cur;
        pair = sc->cache->carry;
    }
    VkDeviceSize per = mda_fwd_cache_carry_bytes(cfg);
    mda_vkc_buf_t carry_in = mda_fwd_cache_view(sc, pair, &sc->dg_dummy, cur, 2 * l + carry, per);
    mda_vkc_buf_t carry_out = mda_fwd_cache_view(sc, pair, &sc->dg_dummy, 1 - cur, 2 * l + carry, per);
    mda_vkc_buf_t *bufs[7] = {
        x,
        &mix->w,
        out,
        dx,
        &mix->g,
        &carry_in,
        &carry_out,
    };
    uint32_t groups = (bt * cfg.d + 255) / 256;
    if (op == 2) {
        groups = (cfg.d * nmix * ((bt + MDA_FWD_SUM_ROWS - 1) / MDA_FWD_SUM_ROWS) + 255) / 256;
    }
    mda_vkc_run(m->c, &m->k_tshift, bufs, groups, 1, 1, &pc);
}

void mda_fwd_lora_act(mda_model_t *m, mda_vkc_buf_t *h, mda_vkc_buf_t *x, uint32_t n, uint32_t op, mda_model_act_t act, uint32_t slot) {
    mda_pc_lora_t pc = {
        .n = n,
        .op = op,
        .act = act,
        .slot = slot,
    };
    mda_vkc_buf_t *bufs[3] = {
        h,
        x,
        &m->amax,
    };
    mda_vkc_run(m->c, &m->k_lora, bufs, mda_util_min_u32((n + 255) / 256, MDA_FWD_ELEM_GROUPS), 1, 1, &pc);
}

mda_vkc_buf_t mda_fwd_layer_vecs(mda_model_t *m, const mda_model_layer_t *ly) {
    VkDeviceSize lo = (VkDeviceSize)ly->w0->off * sizeof(float);
    VkDeviceSize hi = ((VkDeviceSize)ly->lnx_b->off + ly->lnx_b->n) * sizeof(float);
    return mda_vkc_view(&m->w_all, lo, hi - lo);
}

uint32_t mda_fwd_vec_off(const mda_model_layer_t *ly, const mda_model_param_t *p) {
    return p->off - ly->w0->off;
}

void mda_fwd_heads(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, uint32_t first_row) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t n_vec = mda_fwd_n_vec(m);
    mda_vkc_buf_t vecs = mda_fwd_layer_vecs(m, ly);
    mda_pc_heads_t pc = {
        .n_vec = n_vec,
        .sec = cfg.batch * cfg.seqlen * cfg.d,
        .hd = cfg.n_head,
        .seqlen = cfg.seqlen,
        .first_row = first_row,
        .w0 = mda_fwd_vec_off(ly, ly->w0),
        .a0 = mda_fwd_vec_off(ly, ly->a0),
        .v0 = mda_fwd_vec_off(ly, ly->v0),
        .kk = mda_fwd_vec_off(ly, ly->kk),
        .ka = mda_fwd_vec_off(ly, ly->ka),
        .rk = mda_fwd_vec_off(ly, ly->rk),
    };
    mda_vkc_buf_t *bufs[8] = {
        &a->kv,
        &sc->raw,
        &sc->v0,
        &vecs,
        &a->heads,
        &a->vg,
        &a->kinv,
        &a->bs,
    };
    mda_vkc_run(m->c, &m->k_heads_f, bufs, (n_vec + 3) / 4, 1, 1, &pc);
}

void mda_fwd_wout(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, mda_vkc_buf_t *y, uint32_t op, uint32_t slot) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t n_vec = mda_fwd_n_vec(m);
    mda_vkc_buf_t vecs = mda_fwd_layer_vecs(m, ly);
    mda_pc_wout_t pc = {
        .n_vec = n_vec,
        .sec = cfg.batch * cfg.seqlen * cfg.d,
        .hd = cfg.n_head,
        .seqlen = cfg.seqlen,
        .op = op,
        .slot = slot,
        .lw = mda_fwd_vec_off(ly, ly->lnx_w),
        .lb = mda_fwd_vec_off(ly, ly->lnx_b),
        .eps = MDA_FWD_LNX_EPS,
    };
    mda_vkc_buf_t *bufs[14] = {
        &a->wkv_raw,
        &a->heads,
        &a->bs,
        &a->g,
        &a->p,
        y,
        &a->stx,
        &sc->dtmp2,
        &sc->dgate,
        &sc->dbs,
        &sc->dvb,
        &m->amax,
        &vecs,
        &sc->dln,
    };
    mda_vkc_run(m->c, &m->k_wout, bufs, (n_vec + 3) / 4, 1, 1, &pc);
}

mda_fwd_act_t *mda_fwd_layer_acts(mda_fwd_scratch_t *sc, uint32_t l, bool save) {
    if (save) {
        return &sc->acts[l];
    } else {
        return &sc->acts[0];
    }
}

mda_vkc_buf_t *mda_fwd_next(mda_model_t *m, mda_fwd_scratch_t *sc, uint32_t l, bool save) {
    if (l + 1 == m->cfg.n_layer) {
        return &sc->x;
    } else {
        return &mda_fwd_layer_acts(sc, l + 1, save)->x_in;
    }
}

void mda_fwd_embed(mda_model_t *m, mda_fwd_scratch_t *sc) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
    mda_pc_emb_t pc = {
        .bt = bt,
        .d = m->cfg.d,
        .op = 0,
        .vocab = m->cfg.vocab,
    };
    mda_vkc_buf_t *bufs[4] = {
        &sc->ids,
        &m->tok_emb->w,
        &m->mask_emb->w,
        &sc->emb,
    };
    mda_vkc_run(m->c, &m->k_embed, bufs, (bt * m->cfg.d + 255) / 256, 1, 1, &pc);
    mda_fwd_layernorm(m, &sc->emb, m->ln0_w, m->ln0_b, &sc->acts[0].x_in, &sc->st0, bt, 0, &sc->dg_dummy);
}

void mda_fwd_wkv_run(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, uint32_t l, uint32_t dir, bool save, uint32_t seq0, uint32_t count, uint32_t split) {
    mda_model_cfg_t cfg = m->cfg;
    mda_pc_wkv_t pc = {
        .bsz = cfg.batch,
        .hd = cfg.n_head,
        .t = cfg.seqlen,
        .dh = cfg.d / cfg.n_head,
        .dir = dir,
        .save = (uint32_t)save,
        .blk = cfg.block,
        .nclean = mda_model_nclean(cfg),
        .seq0 = seq0,
        .cache = sc->cache != NULL,
        .sec = cfg.batch * cfg.seqlen * cfg.d,
        .split = split,
    };
    uint32_t cur = 0;
    mda_vkc_buf_t *pair = &sc->bstate;
    if (sc->cache != NULL) {
        cur = sc->cache->cur;
        pair = sc->cache->state;
    }
    VkDeviceSize per = mda_fwd_cache_state_bytes(cfg);
    mda_vkc_buf_t cin = mda_fwd_cache_view(sc, pair, &sc->bstate, cur, l, per);
    mda_vkc_buf_t cout = mda_fwd_cache_view(sc, pair, &sc->bstate, 1 - cur, l, per);
    mda_vkc_buf_t *out = &sc->po;
    mda_vkc_buf_t *states = &a->sstate;
    mda_vkc_buf_t *sk = &a->sk;
    if (dir == 1) {
        out = &sc->po2;
        states = &a->sstate2;
        sk = &a->sk2;
    }
    mda_vkc_buf_t *bufs[7] = {
        &a->heads,
        out,
        states,
        &sc->bstate,
        &cin,
        &cout,
        sk,
    };
    mda_vkc_run(m->c, &m->k_wkv_f, bufs, count * cfg.n_head * split, 1, 1, &pc);
}

void mda_fwd_wkv_blocks(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, uint32_t l, bool save) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t nclean = mda_model_nclean(cfg);
    uint32_t split = cfg.seqlen / cfg.block;
    VkDeviceSize half = (VkDeviceSize)nclean * cfg.seqlen * cfg.d * sizeof(float);
    mda_fwd_wkv_run(m, sc, a, l, 0, save, 0, nclean, 1);
    mda_vkc_set_nobar(m->c, true);
    mda_fwd_wkv_run(m, sc, a, l, 0, save, nclean, nclean, split);
    mda_vkc_set_nobar(m->c, false);
    mda_fwd_wkv_run(m, sc, a, l, 1, save, nclean, nclean, split);
    mda_vkc_buf_t po = mda_vkc_view(&sc->po, half, half);
    mda_vkc_buf_t po2 = mda_vkc_view(&sc->po2, half, half);
    mda_fwd_vec_op(m, &po, &po2, nclean * cfg.seqlen * cfg.d, MDA_PC_VEC_AVG);
}

void mda_fwd_wkv(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, uint32_t l, bool save) {
    mda_model_cfg_t cfg = m->cfg;
    if (m->causal || mda_fwd_clean_only(m)) {
        mda_fwd_wkv_run(m, sc, a, l, 0, save, 0, m->live, 1);
    } else if (cfg.block > 0) {
        mda_fwd_wkv_blocks(m, sc, a, l, save);
    } else {
        mda_vkc_set_nobar(m->c, true);
        mda_fwd_wkv_run(m, sc, a, l, 0, save, 0, cfg.batch, 1);
        mda_vkc_set_nobar(m->c, false);
        mda_fwd_wkv_run(m, sc, a, l, 1, save, 0, cfg.batch, 1);
        mda_fwd_vec_op(m, &sc->po, &sc->po2, cfg.batch * cfg.seqlen * cfg.d, MDA_PC_VEC_AVG);
    }
}

uint32_t mda_fwd_lora_mix(uint32_t i) {
    switch (i) {
    case MDA_MODEL_LORA_W:
        return MDA_MODEL_MIX_W;
    case MDA_MODEL_LORA_A:
        return MDA_MODEL_MIX_A;
    case MDA_MODEL_LORA_V:
        return MDA_MODEL_MIX_V;
    default:
        return MDA_MODEL_MIX_G;
    }
}

void mda_fwd_proj(mda_model_t *m, mda_fwd_act_t *a, mda_model_param_t *p, uint32_t mix, mda_vkc_buf_t *out, uint32_t n) {
    mda_model_cfg_t cfg = m->cfg;
    mda_vkc_buf_t x = mda_fwd_section(&a->shift, cfg, mix);
    mda_vkc_buf_t x16 = mda_fwd_section16(&a->shift16, cfg, mix);
    mda_fwd_mm16_pre(m, &x, &x16, a->shift_slot[mix], p, out, mda_fwd_rows(m), n, cfg.d);
}

void mda_fwd_mix_proj(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t d = cfg.d;
    mda_vkc_buf_t k = mda_fwd_section(&a->kv, cfg, 0);
    mda_vkc_buf_t v = mda_fwd_section(&a->kv, cfg, 1);
    mda_vkc_buf_t r = mda_fwd_section(&sc->raw, cfg, 2);
    mda_vkc_set_nobar(m->c, true);
    mda_fwd_proj(m, a, ly->wk, MDA_MODEL_MIX_K, &k, d);
    mda_fwd_proj(m, a, ly->wv, MDA_MODEL_MIX_V, &v, d);
    mda_fwd_proj(m, a, ly->wr, MDA_MODEL_MIX_R, &r, d);
    for (uint32_t i = 0; i < MDA_MODEL_LORAS; i += 1) {
        mda_model_lora_t *l = &ly->lora[i];
        mda_vkc_set_nobar(m->c, i + 1 < MDA_MODEL_LORAS);
        mda_fwd_proj(m, a, l->a, mda_fwd_lora_mix(i), &a->lora[i].h, l->rank);
    }
}

void mda_fwd_lora_hidden(mda_model_t *m, mda_model_layer_t *ly, mda_fwd_act_t *a) {
    uint32_t bt = mda_fwd_rows(m);
    for (uint32_t i = 0; i < MDA_MODEL_LORAS; i += 1) {
        mda_model_lora_t *l = &ly->lora[i];
        mda_fwd_lora_act_t *la = &a->lora[i];
        uint32_t n = bt * l->rank;
        switch (l->act) {
        case MDA_MODEL_ACT_NONE:
            la->slot = mda_fwd_cast_act(m, &la->h, &la->hf16, n);
            break;
        default:
            la->slot = mda_fwd_slot(m);
            mda_fwd_lora_act(m, &la->h, &la->hf, n, 0, l->act, la->slot);
            mda_fwd_cast(m, &la->hf, &la->hf16, n, la->slot);
            break;
        }
    }
}

mda_vkc_buf_t mda_fwd_lora_dest(mda_fwd_scratch_t *sc, mda_fwd_act_t *a, mda_model_cfg_t cfg, uint32_t i) {
    switch (i) {
    case MDA_MODEL_LORA_G:
        return a->g;
    default:
        return mda_fwd_section(&sc->raw, cfg, 3 + i);
    }
}

void mda_fwd_lora_out(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = mda_fwd_rows(m);
    for (uint32_t i = 0; i < MDA_MODEL_LORAS; i += 1) {
        mda_model_lora_t *l = &ly->lora[i];
        mda_fwd_lora_act_t *la = &a->lora[i];
        mda_vkc_buf_t out = mda_fwd_lora_dest(sc, a, cfg, i);
        mda_vkc_set_nobar(m->c, i + 1 < MDA_MODEL_LORAS);
        mda_fwd_mm16_pre(m, &la->hf, &la->hf16, la->slot, l->b, &out, bt, cfg.d, l->rank);
    }
}

void mda_fwd_cast_shifts(mda_model_t *m, mda_fwd_act_t *a) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t n = mda_fwd_rows(m) * cfg.d;
    for (uint32_t s = 0; s < MDA_MODEL_MIXES; s += 1) {
        mda_vkc_buf_t x = mda_fwd_section(&a->shift, cfg, s);
        mda_vkc_buf_t x16 = mda_fwd_section16(&a->shift16, cfg, s);
        a->shift_slot[s] = mda_fwd_cast_act(m, &x, &x16, n);
    }
}

void mda_fwd_time_mix(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, uint32_t l, bool save) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = mda_fwd_rows(m);
    uint32_t d = cfg.d;
    mda_fwd_layernorm(m, &a->x_in, ly->ln1_w, ly->ln1_b, &a->n1, &a->st1, bt, 0, &sc->dg_dummy);
    mda_fwd_tshift(m, sc, &a->n1, ly->att_mix, &a->shift, MDA_MODEL_MIXES, 0, 0, l, &sc->dg_dummy);
    mda_fwd_cast_shifts(m, a);
    mda_fwd_mix_proj(m, sc, ly, a);
    mda_fwd_lora_hidden(m, ly, a);
    mda_fwd_lora_out(m, sc, ly, a);
    mda_fwd_heads(m, sc, ly, a, mda_fwd_first_row(m, l));
    mda_fwd_wkv(m, sc, a, l, save);
    mda_fwd_repack(m, &sc->po, &a->wkv_raw, 1);
    mda_fwd_wout(m, sc, ly, a, &a->wkv, 0, MDA_PC_NO_SLOT);
    a->wkv_slot = mda_fwd_cast_act(m, &a->wkv, &a->wkv16, bt * d);
    mda_fwd_mm16_res(m, &a->wkv, &a->wkv16, a->wkv_slot, ly->wo, &a->x_in, &a->x_mid, bt, d, d);
}

void mda_fwd_channel_mix(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, uint32_t l, mda_vkc_buf_t *next) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = mda_fwd_rows(m);
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    mda_fwd_layernorm(m, &a->x_mid, ly->ln2_w, ly->ln2_b, &a->n2, &a->st2, bt, 0, &sc->dg_dummy);
    mda_fwd_tshift(m, sc, &a->n2, ly->ffn_mix, &a->fshift, 1, 0, 1, l, &sc->dg_dummy);
    a->fshift_slot = mda_fwd_cast_act(m, &a->fshift, &a->fshift16, bt * d);
    mda_fwd_mm16_pre(m, &a->fshift, &a->fshift16, a->fshift_slot, ly->w_key, &a->kpre, bt, hid, d);
    a->h_slot = mda_fwd_slot(m);
    mda_fwd_elem(m, &a->h, &a->kpre, 0, a->h_slot);
    mda_fwd_cast(m, &a->h, &a->h16, bt * hid, a->h_slot);
    mda_fwd_mm16_res(m, &a->h, &a->h16, a->h_slot, ly->w_val, &a->x_mid, next, bt, d, hid);
}

void mda_fwd_head(mda_model_t *m, mda_fwd_scratch_t *sc) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    mda_fwd_layernorm(m, &sc->x, m->ln_out_w, m->ln_out_b, &sc->xn, &sc->stn, bt, 0, &sc->dg_dummy);
    if (m->ref32) {
        mda_fwd_mm(m, &sc->xn, &m->head->w, &sc->logits, bt, cfg.vocab, cfg.d);
    } else {
        sc->a16_slot = mda_fwd_cast_act(m, &sc->xn, &sc->a16, bt * cfg.d);
        mda_fwd_mm16_raw(m, &sc->a16, sc->a16_slot, &m->head->w16, MDA_PC_NO_SLOT, &sc->logits, bt, cfg.vocab, cfg.d, false, MDA_FWD_AB);
    }
}

void mda_fwd_noisy_input(mda_model_t *m, mda_fwd_scratch_t *sc, bool save) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t n = mda_model_nclean(cfg) * cfg.seqlen * cfg.d;
    if (save && mda_fwd_split(m)) {
        mda_vkc_buf_t src = mda_vkc_view(&sc->acts[0].x_in, (VkDeviceSize)n * sizeof(float), (VkDeviceSize)n * sizeof(float));
        mda_vkc_buf_t dst = mda_vkc_view(&sc->acts[mda_fwd_dec_start(m)].x_in, (VkDeviceSize)n * sizeof(float), (VkDeviceSize)n * sizeof(float));
        mda_fwd_vec_op(m, &dst, &src, n, MDA_PC_VEC_COPY);
    }
}

void mda_fwd_rec(mda_model_t *m, mda_fwd_scratch_t *sc, bool save) {
    mda_fwd_slots_begin(m);
    mda_fwd_embed(m, sc);
    mda_fwd_noisy_input(m, sc, save);
    for (uint32_t l = 0; l < m->cfg.n_layer; l += 1) {
        mda_model_layer_t *ly = &m->layers[l];
        mda_fwd_act_t *a = mda_fwd_layer_acts(sc, l, save);
        mda_fwd_set_live(m, l);
        mda_fwd_time_mix(m, sc, ly, a, l, save);
        mda_fwd_channel_mix(m, sc, ly, a, l, mda_fwd_next(m, sc, l, save));
    }
    m->live = m->cfg.batch;
    mda_fwd_head(m, sc);
}

void mda_fwd_ce(mda_model_t *m, mda_fwd_scratch_t *sc, uint32_t op, uint32_t slot) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
    mda_pc_ce_t pc = {
        .rows = bt,
        .vocab = m->cfg.vocab,
        .op = op,
        .slot = slot,
        .kd_rows = mda_fwd_kd_rows(m),
    };
    mda_vkc_buf_t *bufs[10] = {
        &sc->logits,
        &sc->target,
        &sc->mask,
        &sc->weight,
        &sc->loss,
        &sc->opt,
        &sc->logz,
        &sc->dc16,
        &m->amax,
        &sc->dkd,
    };
    mda_vkc_run(m->c, &m->k_ce, bufs, bt, 1, 1, &pc);
}

void mda_fwd_ce_rec(mda_model_t *m, mda_fwd_scratch_t *sc) {
    sc->ce_slot = mda_fwd_slot(m);
    mda_fwd_ce(m, sc, 0, sc->ce_slot);
}

uint32_t mda_fwd_kd_rows(const mda_model_t *m) {
    if (m->kd) {
        return mda_model_nclean(m->cfg) * m->cfg.seqlen;
    } else {
        return 0;
    }
}

void mda_fwd_kd_rec(mda_model_t *m, mda_fwd_scratch_t *sc) {
    uint32_t rows = mda_fwd_kd_rows(m);
    mda_pc_kd_student_t pc = {
        .rows = rows,
        .vocab = m->cfg.vocab,
        .slot = sc->ce_slot,
        .lambda = m->kd_weight,
    };
    mda_vkc_buf_t *bufs[10] = {
        &sc->logits,
        &sc->kd_toks,
        &sc->kd_cont,
        &sc->kd_tgt,
        &sc->logz,
        &sc->dkd,
        &sc->kd_loss,
        &m->amax,
        &sc->opt,
        &sc->kd_bits,
    };
    mda_vkc_run(m->c, &m->k_kd, bufs, rows, 1, 1, &pc);
}

void mda_fwd_argmax_rec(mda_model_t *m, mda_fwd_scratch_t *sc) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
    mda_pc_argmax_t pc = {
        .rows = bt,
        .vocab = m->cfg.vocab,
        .temp = sc->temp,
        .seed = sc->seed,
    };
    mda_vkc_buf_t *bufs[3] = {
        &sc->logits,
        &sc->tok,
        &sc->conf,
    };
    mda_vkc_run(m->c, &m->k_argmax, bufs, bt, 1, 1, &pc);
}

void mda_fwd_softmax_rec(mda_model_t *m, mda_fwd_scratch_t *sc) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
    mda_pc_softmax_t pc = {
        .rows = bt,
        .vocab = m->cfg.vocab,
        .temp = sc->temp,
    };
    mda_vkc_buf_t *bufs[2] = {
        &sc->logits,
        sc->stage,
    };
    mda_vkc_run(m->c, &m->k_softmax, bufs, bt, 1, 1, &pc);
}

float mda_fwd_loss_scale(mda_model_cfg_t cfg) {
    return 1.0f / (float)(cfg.batch * cfg.seqlen);
}

float mda_fwd_loss_sum(mda_model_t *m, mda_fwd_scratch_t *sc) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
    float *lp = sc->loss.ptr;
    double total = 0.0;
    for (uint32_t i = 0; i < bt; i += 1) {
        total += lp[i];
    }
    return (float)(total * mda_fwd_loss_scale(m->cfg));
}

float mda_fwd_run(mda_model_t *m, mda_fwd_scratch_t *sc) {
    mda_fwd_refresh_w16(m);
    mda_vkc_begin(m->c);
    mda_fwd_rec(m, sc, false);
    mda_fwd_argmax_rec(m, sc);
    if (m->infer == false) {
        mda_fwd_ce_rec(m, sc);
    }
    if (sc->stage != NULL) {
        mda_fwd_softmax_rec(m, sc);
    }
    mda_vkc_flush(m->c);
    if (m->infer || m->c->fail) {
        return 0.0f;
    } else {
        return mda_fwd_loss_sum(m, sc);
    }
}
