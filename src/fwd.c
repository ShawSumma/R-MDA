#include <stdio.h>
#include <stdlib.h>

#include "fwd.h"
#include "util.h"
#include "bwd.h"

mda_fwd_act_t mda_fwd_act_alloc(mda_model_t *m) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    uint32_t dh = d / cfg.n_head;
    uint32_t ns = cfg.batch * cfg.n_head * cfg.seqlen / MDA_MODEL_WKV_CHUNK * dh * dh;
    mda_vkc_t *c = m->c;
    return (mda_fwd_act_t) {
        .x_in = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .rms1 = mda_vkc_alloc(c, bt * sizeof(float)),
        .n1_pre = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .n1 = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .shift = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .pk = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .pv = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .pr = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .pw = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .pa = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .kinv = mda_vkc_alloc(c, bt * cfg.n_head * sizeof(float)),
        .sstate = mda_vkc_alloc(c, (VkDeviceSize)ns * sizeof(uint16_t)),
        .sstate2 = mda_vkc_alloc(c, (VkDeviceSize)ns * sizeof(uint16_t)),
        .wkv_raw = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .wkv = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .rmsx = mda_vkc_alloc(c, bt * cfg.n_head * sizeof(float)),
        .x_mid = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .rms2 = mda_vkc_alloc(c, bt * sizeof(float)),
        .n2_pre = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .n2 = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .gate_pre = mda_vkc_alloc(c, bt * hid * sizeof(float)),
        .up = mda_vkc_alloc(c, bt * hid * sizeof(float)),
        .h = mda_vkc_alloc(c, bt * hid * sizeof(float)),
        .shift16 = mda_vkc_alloc(c, bt * d * sizeof(uint16_t)),
        .wkv16 = mda_vkc_alloc(c, bt * d * sizeof(uint16_t)),
        .n2_16 = mda_vkc_alloc(c, bt * d * sizeof(uint16_t)),
        .h16 = mda_vkc_alloc(c, bt * hid * sizeof(uint16_t)),
    };
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
        cache.carry[i] = mda_vkc_alloc(m->c, cfg.n_layer * mda_fwd_cache_carry_bytes(cfg));
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
        acts[i] = mda_fwd_act_alloc(m);
    }
    VkDeviceSize nstate = (VkDeviceSize)mda_util_max_u32(mda_fwd_block_states(cfg), 1) * 4096 * sizeof(float);
    uint32_t widest = mda_util_max_u32(mda_util_max_u32(d, hid), cfg.vocab);
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
        .x = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .bstate = mda_vkc_alloc(c, nstate),
        .dbstate = mda_vkc_alloc(c, nstate),
        .dup = mda_vkc_alloc(c, bt * hid * sizeof(float)),
        .rms_stash = mda_vkc_alloc(c, bt * sizeof(float)),
        .dg_dummy = mda_vkc_alloc(c, d * sizeof(float)),
        .pk = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .pv = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .pr = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .pw = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .pa = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .po = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .po2 = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .xn = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .logits = mda_vkc_alloc(c, (VkDeviceSize)bt * cfg.vocab * sizeof(float)),
        .loss = mda_vkc_alloc_cached(c, bt * sizeof(float)),
        .logz = mda_vkc_alloc(c, bt * sizeof(float)),
        .tok = mda_vkc_alloc_cached(c, bt * sizeof(uint32_t)),
        .dnorm = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dtmp = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dtmp2 = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dh = mda_vkc_alloc(c, bt * hid * sizeof(float)),
        .dpk = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dpv = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dpr = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dpw = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dpa = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dpk2 = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dpv2 = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dpr2 = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dpw2 = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dpa2 = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dwkv = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .dxh_buf = mda_vkc_alloc(c, bt * d * sizeof(float)),
        .a16 = mda_vkc_alloc(c, bt * d * sizeof(uint16_t)),
        .dc16 = mda_vkc_alloc(c, (VkDeviceSize)bt * widest * sizeof(uint16_t)),
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

void mda_fwd_repack(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t op) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
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

void mda_fwd_elem(mda_model_t *m, mda_vkc_buf_t *x, mda_fwd_act_t *a, mda_vkc_buf_t *dup, uint32_t op, uint32_t slot, uint32_t slot2) {
    uint32_t n = m->cfg.batch * m->cfg.seqlen * m->cfg.ffn_hidden;
    mda_pc_elem_t pc = {
        .n = n,
        .op = op,
        .slot = slot,
        .slot2 = slot2,
    };
    mda_vkc_buf_t *bufs[5] = {
        x,
        &a->gate_pre,
        &a->up,
        dup,
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

void mda_fwd_refresh_w16(mda_model_t *m) {
    if (m->dirty16) {
        mda_vkc_begin(m->c);
        mda_fwd_cast(m, &m->w_all, &m->w16_all, m->n_all, MDA_PC_NO_SLOT);
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

void mda_fwd_heads(mda_model_t *m, mda_vkc_kernel_t *k, mda_vkc_buf_t **bufs, uint32_t slot) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t n_vec = cfg.batch * cfg.seqlen * cfg.n_head;
    mda_pc_heads_t pc = {
        .n_vec = n_vec,
        .hd = cfg.n_head,
        .seqlen = cfg.seqlen,
        .slot = slot,
    };
    mda_vkc_run(m->c, k, bufs, (n_vec + 3) / 4, 1, 1, &pc);
}

void mda_fwd_headnorm(mda_model_t *m, mda_vkc_buf_t *x, mda_vkc_buf_t *y, mda_vkc_buf_t *rms, mda_vkc_buf_t *dx, uint32_t op) {
    uint32_t n_vec = m->cfg.batch * m->cfg.seqlen * m->cfg.n_head;
    mda_pc_hn_t pc = {
        .n_vec = n_vec,
        .op = op,
    };
    mda_vkc_buf_t *bufs[4] = {
        x,
        y,
        rms,
        dx,
    };
    mda_vkc_run(m->c, &m->k_headnorm, bufs, (n_vec + 3) / 4, 1, 1, &pc);
}

void mda_fwd_rmsnorm(mda_model_t *m, mda_fwd_scratch_t *sc, mda_vkc_buf_t *x, mda_vkc_buf_t *g, mda_vkc_buf_t *out, mda_vkc_buf_t *stash, uint32_t rows, uint32_t d) {
    mda_pc_rms_t pc = {
        .rows = rows,
        .d = d,
        .op = 0,
    };
    mda_vkc_buf_t *bufs[6] = {
        x,
        g,
        out,
        stash,
        &sc->dg_dummy,
        &sc->dg_dummy,
    };
    mda_vkc_run(m->c, &m->k_rms, bufs, rows, 1, 1, &pc);
}

uint32_t mda_fwd_adaln_groups(mda_model_cfg_t cfg, uint32_t op) {
    uint32_t bt = cfg.batch * cfg.seqlen;
    switch (op) {
    case 0:
        return (bt * cfg.d + 255) / 256;
    default:
        return (cfg.d * ((bt + MDA_FWD_ADALN_ROWS - 1) / MDA_FWD_ADALN_ROWS) + 255) / 256;
    }
}

void mda_fwd_adaln(mda_model_t *m, mda_vkc_buf_t *x, mda_model_param_t *ada, mda_vkc_buf_t *y, mda_vkc_buf_t *dx, uint32_t op, uint32_t slot) {
    mda_model_cfg_t cfg = m->cfg;
    mda_pc_adaln_t pc = {
        .bt = cfg.batch * cfg.seqlen,
        .d = cfg.d,
        .op = op,
        .s_off = slot * cfg.d,
        .sh_off = (slot + 1) * cfg.d,
        .rows_per = MDA_FWD_ADALN_ROWS,
        .scale = mda_model_ada_scale(cfg),
    };
    mda_vkc_buf_t *bufs[5] = {
        x,
        &ada->w,
        y,
        dx,
        &ada->g,
    };
    mda_vkc_run(m->c, &m->k_adaln, bufs, mda_fwd_adaln_groups(cfg, op), 1, 1, &pc);
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
        &sc->acts[0].x_in,
    };
    mda_vkc_run(m->c, &m->k_embed, bufs, (bt * m->cfg.d + 255) / 256, 1, 1, &pc);
}

void mda_fwd_wkv_run(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, uint32_t l, uint32_t dir, bool save, uint32_t seq0, uint32_t count) {
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
    if (dir == 1) {
        out = &sc->po2;
        states = &a->sstate2;
    }
    mda_vkc_buf_t *bufs[10] = {
        &a->pk,
        &a->pv,
        &a->pr,
        &a->pw,
        &a->pa,
        out,
        states,
        &sc->bstate,
        &cin,
        &cout,
    };
    mda_vkc_run(m->c, &m->k_wkv_f, bufs, count * cfg.n_head, 1, 1, &pc);
}

void mda_fwd_wkv_blocks(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, uint32_t l, bool save) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t nclean = mda_model_nclean(cfg);
    VkDeviceSize half = (VkDeviceSize)nclean * cfg.seqlen * cfg.d * sizeof(float);
    mda_fwd_wkv_run(m, sc, a, l, 0, save, 0, nclean);
    mda_vkc_set_nobar(m->c, true);
    mda_fwd_wkv_run(m, sc, a, l, 0, save, nclean, nclean);
    mda_vkc_set_nobar(m->c, false);
    mda_fwd_wkv_run(m, sc, a, l, 1, save, nclean, nclean);
    mda_vkc_buf_t po = mda_vkc_view(&sc->po, half, half);
    mda_vkc_buf_t po2 = mda_vkc_view(&sc->po2, half, half);
    mda_fwd_vec_op(m, &po, &po2, nclean * cfg.seqlen * cfg.d, MDA_PC_VEC_AVG);
}

void mda_fwd_wkv(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, uint32_t l, bool save) {
    mda_model_cfg_t cfg = m->cfg;
    if (m->causal) {
        mda_fwd_wkv_run(m, sc, a, l, 0, save, 0, cfg.batch);
    } else if (cfg.block > 0) {
        mda_fwd_wkv_blocks(m, sc, a, l, save);
    } else {
        mda_vkc_set_nobar(m->c, true);
        mda_fwd_wkv_run(m, sc, a, l, 0, save, 0, cfg.batch);
        mda_vkc_set_nobar(m->c, false);
        mda_fwd_wkv_run(m, sc, a, l, 1, save, 0, cfg.batch);
        mda_fwd_vec_op(m, &sc->po, &sc->po2, cfg.batch * cfg.seqlen * cfg.d, MDA_PC_VEC_AVG);
    }
}

void mda_fwd_time_mix(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, uint32_t l, bool save) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    mda_fwd_rmsnorm(m, sc, &a->x_in, &ly->n1->w, &a->n1_pre, &a->rms1, bt, d);
    mda_fwd_adaln(m, &a->n1_pre, ly->ada, &a->n1, &a->n1, 0, 0);
    mda_pc_ts_t tpc = {
        .bt = bt,
        .d = d,
        .seqlen = cfg.seqlen,
        .op = 0,
        .blk = cfg.block,
        .nclean = mda_model_nclean(cfg),
        .cache = sc->cache != NULL,
    };
    uint32_t cur = 0;
    mda_vkc_buf_t *pair = &sc->dg_dummy;
    if (sc->cache != NULL) {
        cur = sc->cache->cur;
        pair = sc->cache->carry;
    }
    VkDeviceSize per = mda_fwd_cache_carry_bytes(cfg);
    mda_vkc_buf_t carry_in = mda_fwd_cache_view(sc, pair, &sc->dg_dummy, cur, l, per);
    mda_vkc_buf_t carry_out = mda_fwd_cache_view(sc, pair, &sc->dg_dummy, 1 - cur, l, per);
    mda_vkc_buf_t *tsb[7] = {
        &a->n1,
        &ly->shift_mix->w,
        &a->shift,
        &sc->pk,
        &sc->dg_dummy,
        &carry_in,
        &carry_out,
    };
    mda_vkc_run(m->c, &m->k_tshift, tsb, (bt * d + 255) / 256, 1, 1, &tpc);
    a->shift_slot = mda_fwd_cast_act(m, &a->shift, &a->shift16, bt * d);
    mda_vkc_set_nobar(m->c, true);
    mda_fwd_mm16_pre(m, &a->shift, &a->shift16, a->shift_slot, ly->wk, &sc->pk, bt, d, d);
    mda_fwd_mm16_pre(m, &a->shift, &a->shift16, a->shift_slot, ly->wv, &sc->pv, bt, d, d);
    mda_fwd_mm16_pre(m, &a->shift, &a->shift16, a->shift_slot, ly->wr, &sc->pr, bt, d, d);
    mda_fwd_mm16_pre(m, &a->shift, &a->shift16, a->shift_slot, ly->ww, &sc->pw, bt, d, d);
    mda_vkc_set_nobar(m->c, false);
    mda_fwd_mm16_pre(m, &a->shift, &a->shift16, a->shift_slot, ly->wa, &sc->pa, bt, d, d);
    mda_vkc_buf_t *hb[11] = {
        &sc->pk,
        &sc->pv,
        &sc->pr,
        &sc->pw,
        &sc->pa,
        &a->pk,
        &a->pv,
        &a->pr,
        &a->pw,
        &a->pa,
        &a->kinv,
    };
    mda_fwd_heads(m, &m->k_heads_f, hb, MDA_PC_NO_SLOT);
    mda_fwd_wkv(m, sc, a, l, save);
    mda_fwd_repack(m, &sc->po, &a->wkv_raw, 1);
    mda_fwd_headnorm(m, &a->wkv_raw, &a->wkv, &a->rmsx, &a->wkv, 0);
    a->wkv_slot = mda_fwd_cast_act(m, &a->wkv, &a->wkv16, bt * d);
    mda_fwd_mm16_res(m, &a->wkv, &a->wkv16, a->wkv_slot, ly->wo, &a->x_in, &a->x_mid, bt, d, d);
}

void mda_fwd_channel_mix(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, mda_vkc_buf_t *next) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    mda_fwd_rmsnorm(m, sc, &a->x_mid, &ly->n2->w, &a->n2_pre, &a->rms2, bt, d);
    mda_fwd_adaln(m, &a->n2_pre, ly->ada, &a->n2, &a->n2, 0, 2);
    a->n2_slot = mda_fwd_cast_act(m, &a->n2, &a->n2_16, bt * d);
    mda_vkc_set_nobar(m->c, true);
    mda_fwd_mm16_pre(m, &a->n2, &a->n2_16, a->n2_slot, ly->w_gate, &a->gate_pre, bt, hid, d);
    mda_vkc_set_nobar(m->c, false);
    mda_fwd_mm16_pre(m, &a->n2, &a->n2_16, a->n2_slot, ly->w_up, &a->up, bt, hid, d);
    a->h_slot = mda_fwd_slot(m);
    mda_fwd_elem(m, &a->h, a, &sc->dup, 0, a->h_slot, MDA_PC_NO_SLOT);
    mda_fwd_cast(m, &a->h, &a->h16, bt * hid, a->h_slot);
    mda_fwd_mm16_res(m, &a->h, &a->h16, a->h_slot, ly->w_down, &a->x_mid, next, bt, d, hid);
}

void mda_fwd_head(mda_model_t *m, mda_fwd_scratch_t *sc) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    mda_fwd_rmsnorm(m, sc, &sc->x, &m->n_out->w, &sc->xn, &sc->rms_stash, bt, cfg.d);
    if (m->ref32) {
        mda_bwd_mm_da(m, &m->tok_emb->w, &sc->xn, &sc->logits, bt, cfg.d, cfg.vocab, false);
    } else {
        sc->a16_slot = mda_fwd_cast_act(m, &sc->xn, &sc->a16, bt * cfg.d);
        mda_fwd_mm16_raw(m, &sc->a16, sc->a16_slot, &m->tok_emb->wt16, MDA_PC_NO_SLOT, &sc->logits, bt, cfg.vocab, cfg.d, false, MDA_FWD_AB);
    }
}

void mda_fwd_rec(mda_model_t *m, mda_fwd_scratch_t *sc, bool save) {
    mda_fwd_slots_begin(m);
    mda_fwd_embed(m, sc);
    for (uint32_t l = 0; l < m->cfg.n_layer; l += 1) {
        mda_model_layer_t *ly = &m->layers[l];
        mda_fwd_act_t *a = mda_fwd_layer_acts(sc, l, save);
        mda_fwd_time_mix(m, sc, ly, a, l, save);
        mda_fwd_channel_mix(m, sc, ly, a, mda_fwd_next(m, sc, l, save));
    }
    mda_fwd_head(m, sc);
}

void mda_fwd_ce(mda_model_t *m, mda_fwd_scratch_t *sc, uint32_t op, uint32_t slot) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
    mda_pc_ce_t pc = {
        .rows = bt,
        .vocab = m->cfg.vocab,
        .op = op,
        .slot = slot,
    };
    mda_vkc_buf_t *bufs[9] = {
        &sc->logits,
        &sc->target,
        &sc->mask,
        &sc->weight,
        &sc->loss,
        &sc->opt,
        &sc->logz,
        &sc->dc16,
        &m->amax,
    };
    mda_vkc_run(m->c, &m->k_ce, bufs, bt, 1, 1, &pc);
}

void mda_fwd_ce_rec(mda_model_t *m, mda_fwd_scratch_t *sc) {
    sc->ce_slot = mda_fwd_slot(m);
    mda_fwd_ce(m, sc, 0, sc->ce_slot);
}

void mda_fwd_argmax_rec(mda_model_t *m, mda_fwd_scratch_t *sc) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
    mda_pc_argmax_t pc = {
        .rows = bt,
        .vocab = m->cfg.vocab,
        .temp = sc->temp,
        .seed = sc->seed,
    };
    mda_vkc_buf_t *bufs[2] = {
        &sc->logits,
        &sc->tok,
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
