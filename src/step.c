#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "step.h"
#include "pc.h"
#include "bwd.h"
#include "util.h"

void mda_step_head_bwd(mda_model_t *m, mda_fwd_scratch_t *sc) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t vocab = cfg.vocab;
    mda_model_param_t *h = m->head;
    if (m->ref32) {
        mda_fwd_ce(m, sc, 1, MDA_PC_NO_SLOT);
        mda_bwd_mm_db(m, &sc->xn, &sc->logits, &h->g, bt, vocab, d);
        mda_bwd_mm_da(m, &h->w, &sc->logits, &sc->dnorm, bt, vocab, d, false);
    } else {
        mda_fwd_ce(m, sc, 2, sc->ce_slot);
        mda_vkc_set_nobar(m->c, true);
        mda_fwd_mm16_raw(m, &sc->dc16, sc->ce_slot, &h->wt16, MDA_PC_NO_SLOT, &sc->dnorm, bt, d, vocab, false, MDA_FWD_AB);
        mda_vkc_set_nobar(m->c, false);
        mda_fwd_mm16_raw(m, &sc->a16, sc->a16_slot, &sc->dc16, sc->ce_slot, &h->g, d, vocab, bt, m->grad_acc, MDA_FWD_AT);
    }
    mda_fwd_layernorm(m, &sc->x, m->ln_out_w, m->ln_out_b, &sc->dnorm, &sc->stn, bt, 1, &sc->dxh_buf);
}

void mda_step_tshift_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_vkc_buf_t *x, mda_model_param_t *mix, uint32_t nmix, uint32_t l) {
    mda_vkc_fill(m->c, &sc->dnorm, 0);
    mda_vkc_set_nobar(m->c, true);
    mda_fwd_tshift(m, sc, x, mix, &sc->dshift, nmix, 1, 0, l, &sc->dnorm);
    mda_vkc_set_nobar(m->c, false);
    mda_fwd_tshift(m, sc, x, mix, &sc->dshift, nmix, 2, 0, l, &sc->dnorm);
}

void mda_step_ffn_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, uint32_t l) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = mda_fwd_rows(m);
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    uint32_t sx = mda_fwd_absmax(m, &sc->dxh_buf, bt * d);
    mda_bwd_mm(m, sc, &a->h, &a->h16, a->h_slot, ly->w_val, &sc->dxh_buf, sx, &sc->dh, bt, d, hid, false);
    uint32_t sh = mda_fwd_slot(m);
    mda_fwd_elem(m, &sc->dh, &a->kpre, 1, sh);
    mda_vkc_buf_t dfs = mda_fwd_section(&sc->dshift, cfg, 0);
    mda_bwd_mm(m, sc, &a->fshift, &a->fshift16, a->fshift_slot, ly->w_key, &sc->dh, sh, &dfs, bt, hid, d, false);
    mda_step_tshift_bwd(m, sc, &a->n2, ly->ffn_mix, 1, l);
    mda_fwd_layernorm(m, &a->x_mid, ly->ln2_w, ly->ln2_b, &sc->dnorm, &a->st2, bt, 1, &sc->dxh_buf);
}

void mda_step_wkv_bwd_run(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, uint32_t dir, uint32_t seq0, uint32_t count, uint32_t split) {
    mda_model_cfg_t cfg = m->cfg;
    mda_pc_wkv_bwd_t pc = {
        .bsz = cfg.batch,
        .hd = cfg.n_head,
        .t = cfg.seqlen,
        .dh = cfg.d / cfg.n_head,
        .dir = dir,
        .blk = cfg.block,
        .nclean = mda_model_nclean(cfg),
        .seq0 = seq0,
        .split = split,
        .sec = cfg.batch * cfg.seqlen * cfg.d,
    };
    mda_vkc_buf_t *states = &a->sstate;
    mda_vkc_buf_t *sk = &a->sk;
    if (dir == 1) {
        states = &a->sstate2;
        sk = &a->sk2;
    }
    mda_vkc_buf_t *bufs[6] = {
        &a->heads,
        states,
        &sc->dwkv,
        &sc->dheads[dir],
        &sc->dbstate,
        sk,
    };
    mda_vkc_run(m->c, &m->k_wkv_b, bufs, count * cfg.n_head * split, 1, 1, &pc);
}

void mda_step_wkv_bwd_blocks(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t nclean = mda_model_nclean(cfg);
    uint32_t split = cfg.seqlen / cfg.block;
    uint32_t n = nclean * cfg.seqlen * cfg.d;
    mda_vkc_set_nobar(m->c, true);
    mda_step_wkv_bwd_run(m, sc, a, 0, nclean, nclean, split);
    mda_vkc_set_nobar(m->c, false);
    mda_step_wkv_bwd_run(m, sc, a, 1, nclean, nclean, split);
    mda_step_wkv_bwd_run(m, sc, a, 0, 0, nclean, 1);
    for (uint32_t i = 0; i < 6; i += 1) {
        mda_vkc_buf_t from = mda_fwd_section(&sc->dheads[0], cfg, i);
        mda_vkc_buf_t to = mda_fwd_section(&sc->dheads[1], cfg, i);
        mda_vkc_buf_t src = mda_vkc_view(&from, 0, (VkDeviceSize)n * sizeof(float));
        mda_vkc_buf_t dst = mda_vkc_view(&to, 0, (VkDeviceSize)n * sizeof(float));
        mda_vkc_set_nobar(m->c, i + 1 < 6);
        mda_fwd_vec_op(m, &dst, &src, n, MDA_PC_VEC_COPY);
    }
}

void mda_step_wkv_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, uint32_t l) {
    mda_model_cfg_t cfg = m->cfg;
    if (m->causal) {
        mda_step_wkv_bwd_run(m, sc, a, 0, 0, cfg.batch, 1);
    } else if (mda_fwd_clean_only(m)) {
        if (l + 1 == mda_fwd_dec_start(m)) {
            mda_vkc_fill(m->c, &sc->dbstate, 0);
        }
        mda_step_wkv_bwd_run(m, sc, a, 0, 0, m->live, 1);
    } else if (cfg.block > 0) {
        mda_step_wkv_bwd_blocks(m, sc, a);
    } else {
        mda_vkc_set_nobar(m->c, true);
        mda_step_wkv_bwd_run(m, sc, a, 0, 0, cfg.batch, 1);
        mda_vkc_set_nobar(m->c, false);
        mda_step_wkv_bwd_run(m, sc, a, 1, 0, cfg.batch, 1);
    }
}

uint32_t mda_step_heads_slots(mda_model_t *m) {
    uint32_t slot = mda_fwd_slot(m);
    for (uint32_t j = 1; j < 6; j += 1) {
        mda_fwd_slot(m);
    }
    return slot;
}

uint32_t mda_step_heads_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, uint32_t first_row) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t n_vec = mda_fwd_n_vec(m);
    uint32_t slot = mda_step_heads_slots(m);
    mda_vkc_buf_t vecs = mda_fwd_layer_vecs(m, ly);
    mda_pc_heads_bwd_t pc = {
        .n_vec = n_vec,
        .sec = cfg.batch * cfg.seqlen * cfg.d,
        .hd = cfg.n_head,
        .seqlen = cfg.seqlen,
        .first_row = first_row,
        .slot = slot,
        .kk = mda_fwd_vec_off(ly, ly->kk),
        .ka = mda_fwd_vec_off(ly, ly->ka),
        .rk = mda_fwd_vec_off(ly, ly->rk),
    };
    mda_vkc_buf_t *d1 = &sc->dheads[1];
    if (m->causal || mda_fwd_clean_only(m)) {
        d1 = &sc->dheads[0];
    }
    mda_vkc_buf_t *bufs[14] = {
        &sc->dheads[0],
        d1,
        &sc->dvb,
        &sc->dbs,
        &a->heads,
        &a->kinv,
        &a->kv,
        &a->vg,
        &sc->v0,
        &vecs,
        &sc->raw,
        &sc->con,
        &sc->dv0,
        &m->amax,
    };
    mda_vkc_run(m->c, &m->k_heads_b, bufs, (n_vec + 3) / 4, 1, 1, &pc);
    return slot;
}

void mda_step_colsum(mda_model_t *m, mda_vkc_buf_t *x, mda_model_param_t *p) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t rows = mda_fwd_rows(m);
    mda_pc_colsum_t pc = {
        .rows = rows,
        .d = cfg.d,
        .rows_per = MDA_FWD_SUM_ROWS,
    };
    mda_vkc_buf_t *bufs[2] = {
        x,
        &p->g,
    };
    mda_vkc_run(m->c, &m->k_colsum, bufs, (cfg.d * ((rows + MDA_FWD_SUM_ROWS - 1) / MDA_FWD_SUM_ROWS) + 255) / 256, 1, 1, &pc);
}

void mda_step_colsums(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly) {
    mda_model_cfg_t cfg = m->cfg;
    mda_model_param_t *ps[6] = {
        ly->w0,
        ly->a0,
        ly->v0,
        ly->kk,
        ly->ka,
        ly->rk,
    };
    mda_vkc_buf_t xs[6] = {
        mda_fwd_section(&sc->raw, cfg, 3),
        mda_fwd_section(&sc->raw, cfg, 4),
        mda_fwd_section(&sc->raw, cfg, 5),
        mda_fwd_section(&sc->con, cfg, 0),
        mda_fwd_section(&sc->con, cfg, 1),
        mda_fwd_section(&sc->con, cfg, 2),
    };
    for (uint32_t i = 0; i < 6; i += 1) {
        mda_vkc_set_nobar(m->c, i + 1 < 6);
        mda_step_colsum(m, &xs[i], ps[i]);
    }
}

void mda_step_lnx_sums(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly) {
    mda_model_cfg_t cfg = m->cfg;
    mda_vkc_buf_t dw = mda_fwd_section(&sc->dln, cfg, 0);
    mda_vkc_buf_t db = mda_fwd_section(&sc->dln, cfg, 1);
    mda_vkc_set_nobar(m->c, true);
    mda_step_colsum(m, &dw, ly->lnx_w);
    mda_vkc_set_nobar(m->c, false);
    mda_step_colsum(m, &db, ly->lnx_b);
}

void mda_step_proj_in(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, mda_model_param_t *p, mda_vkc_buf_t *dout, uint32_t sd, uint32_t mix, uint32_t n, bool acc) {
    mda_model_cfg_t cfg = m->cfg;
    mda_vkc_buf_t x = mda_fwd_section(&a->shift, cfg, mix);
    mda_vkc_buf_t x16 = mda_fwd_section16(&a->shift16, cfg, mix);
    mda_vkc_buf_t dx = mda_fwd_section(&sc->dshift, cfg, mix);
    mda_bwd_mm(m, sc, &x, &x16, a->shift_slot[mix], p, dout, sd, &dx, mda_fwd_rows(m), n, cfg.d, acc);
}

void mda_step_proj_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, uint32_t sd) {
    mda_model_cfg_t cfg = m->cfg;
    mda_vkc_buf_t dk = mda_fwd_section(&sc->raw, cfg, 0);
    mda_vkc_buf_t dv = mda_fwd_section(&sc->raw, cfg, 1);
    mda_vkc_buf_t dr = mda_fwd_section(&sc->raw, cfg, 2);
    mda_step_proj_in(m, sc, a, ly->wk, &dk, sd, MDA_MODEL_MIX_K, cfg.d, false);
    mda_step_proj_in(m, sc, a, ly->wv, &dv, sd + 1, MDA_MODEL_MIX_V, cfg.d, false);
    mda_step_proj_in(m, sc, a, ly->wr, &dr, sd + 2, MDA_MODEL_MIX_R, cfg.d, false);
}

mda_vkc_buf_t mda_step_lora_grad(mda_fwd_scratch_t *sc, mda_model_cfg_t cfg, uint32_t i) {
    switch (i) {
    case MDA_MODEL_LORA_G:
        return sc->dgate;
    default:
        return mda_fwd_section(&sc->raw, cfg, 3 + i);
    }
}

uint32_t mda_step_lora_slot(uint32_t i, uint32_t sd, uint32_t sg) {
    switch (i) {
    case MDA_MODEL_LORA_G:
        return sg;
    default:
        return sd + 3 + i;
    }
}

void mda_step_lora_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, uint32_t sd, uint32_t sg) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = mda_fwd_rows(m);
    for (uint32_t i = 0; i < MDA_MODEL_LORAS; i += 1) {
        mda_model_lora_t *l = &ly->lora[i];
        mda_fwd_lora_act_t *la = &a->lora[i];
        uint32_t n = bt * l->rank;
        mda_vkc_buf_t dout = mda_step_lora_grad(sc, cfg, i);
        uint32_t so = mda_step_lora_slot(i, sd, sg);
        mda_vkc_buf_t dhf = mda_vkc_view(&sc->dlora, 0, (VkDeviceSize)n * sizeof(float));
        mda_bwd_mm(m, sc, &la->hf, &la->hf16, la->slot, l->b, &dout, so, &dhf, bt, cfg.d, l->rank, false);
        uint32_t sh = MDA_PC_NO_SLOT;
        switch (l->act) {
        case MDA_MODEL_ACT_NONE:
            sh = mda_fwd_absmax(m, &dhf, n);
            break;
        default:
            sh = mda_fwd_slot(m);
            mda_fwd_lora_act(m, &la->h, &dhf, n, 1, l->act, sh);
            break;
        }
        mda_step_proj_in(m, sc, a, l->a, &dhf, sh, mda_fwd_lora_mix(i), l->rank, i == MDA_MODEL_LORA_V);
    }
}

void mda_step_time_mix_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, uint32_t l) {
    uint32_t bt = mda_fwd_rows(m);
    uint32_t d = m->cfg.d;
    uint32_t sx = mda_fwd_absmax(m, &sc->dxh_buf, bt * d);
    mda_bwd_mm(m, sc, &a->wkv, &a->wkv16, a->wkv_slot, ly->wo, &sc->dxh_buf, sx, &sc->dtmp, bt, d, d, false);
    uint32_t sg = mda_fwd_slot(m);
    mda_fwd_wout(m, sc, ly, a, &sc->dtmp, 1, sg);
    mda_step_lnx_sums(m, sc, ly);
    mda_fwd_repack(m, &sc->dtmp2, &sc->dwkv, 0);
    mda_step_wkv_bwd(m, sc, a, l);
    uint32_t sd = mda_step_heads_bwd(m, sc, ly, a, mda_fwd_first_row(m, l));
    mda_step_colsums(m, sc, ly);
    mda_step_proj_bwd(m, sc, ly, a, sd);
    mda_step_lora_bwd(m, sc, ly, a, sd, sg);
    mda_step_tshift_bwd(m, sc, &a->n1, ly->att_mix, MDA_MODEL_MIXES, l);
    mda_fwd_layernorm(m, &a->x_in, ly->ln1_w, ly->ln1_b, &sc->dnorm, &a->st1, bt, 1, &sc->dxh_buf);
}

void mda_step_embed_bwd(mda_model_t *m, mda_fwd_scratch_t *sc) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
    mda_vkc_fill(m->c, &sc->dnorm, 0);
    mda_fwd_layernorm(m, &sc->emb, m->ln0_w, m->ln0_b, &sc->dxh_buf, &sc->st0, bt, 1, &sc->dnorm);
    mda_pc_emb_t pc = {
        .bt = bt,
        .d = m->cfg.d,
        .op = 1,
        .vocab = m->cfg.vocab,
    };
    mda_vkc_buf_t *bufs[4] = {
        &sc->ids,
        &m->tok_emb->g,
        &m->mask_emb->g,
        &sc->dnorm,
    };
    mda_vkc_run(m->c, &m->k_embed, bufs, (bt * m->cfg.d + 255) / 256, 1, 1, &pc);
}

void mda_step_zero_grads(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    mda_vkc_set_nobar(m->c, true);
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        if (m->params[i].accum && m->grad_acc == false) {
            mda_vkc_fill(m->c, &m->params[i].g, 0);
        }
    }
    mda_vkc_set_nobar(m->c, false);
    mda_vkc_fill(m->c, &tr->sc.dxh_buf, 0);
    mda_vkc_fill(m->c, &tr->sc.dv0, 0);
}

void mda_step_rec(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    mda_fwd_scratch_t *sc = &tr->sc;
    mda_step_zero_grads(tr);
    mda_fwd_rec(m, sc, true);
    mda_fwd_ce_rec(m, sc);
    if (m->kd) {
        mda_fwd_kd_rec(m, sc);
    }
    mda_step_head_bwd(m, sc);
    for (uint32_t i = m->cfg.n_layer; i > 0; i -= 1) {
        mda_fwd_set_live(m, i - 1);
        mda_step_ffn_bwd(m, sc, &m->layers[i - 1], &sc->acts[i - 1], i - 1);
        mda_step_time_mix_bwd(m, sc, &m->layers[i - 1], &sc->acts[i - 1], i - 1);
    }
    m->live = m->cfg.batch;
    mda_step_embed_bwd(m, sc);
}

void mda_step_sumsq_rec(mda_step_trainer_t *tr, float clip) {
    mda_model_t *m = tr->m;
    mda_pc_sumsq_t spc = {
        .op = 0,
        .gscale = mda_fwd_loss_scale(m->cfg) / (float)tr->accum,
        .clip = clip,
        .growth = MDA_STEP_GMUL_GROWTH,
    };
    for (uint64_t lo = 0; lo < m->n_all; lo += MDA_FWD_CHUNK) {
        uint32_t n = (uint32_t)mda_util_min_size(MDA_FWD_CHUNK, m->n_all - lo);
        mda_vkc_buf_t g = mda_vkc_view(&m->g_all, lo * sizeof(float), (VkDeviceSize)n * sizeof(float));
        mda_vkc_buf_t *bufs[3] = {
            &g,
            &tr->partials,
            &tr->sc.opt,
        };
        spc.n = n;
        spc.base = (uint32_t)(lo / 256);
        mda_vkc_set_nobar(m->c, lo + MDA_FWD_CHUNK < m->n_all);
        mda_vkc_run(m->c, &m->k_sumsq, bufs, (n + 255) / 256, 1, 1, &spc);
    }
    mda_vkc_buf_t unused = mda_vkc_view(&m->g_all, 0, sizeof(float));
    mda_vkc_buf_t *rbufs[3] = {
        &unused,
        &tr->partials,
        &tr->sc.opt,
    };
    mda_pc_sumsq_t rpc = spc;
    rpc.n = (m->n_all + 255) / 256;
    rpc.op = 1;
    rpc.base = 0;
    mda_vkc_run(m->c, &m->k_sumsq, rbufs, 1, 1, 1, &rpc);
}

void mda_step_adam_run(mda_step_trainer_t *tr, mda_model_param_t *p) {
    mda_pc_adam_t pc = {
        .n = p->n,
        .beta1 = MDA_STEP_ADAM_B1,
        .beta2 = MDA_STEP_ADAM_B2,
        .eps = 1e-8f,
        .wd = 0.0f,
        .lr_mul = p->lr_mul,
    };
    mda_vkc_buf_t *bufs[6] = {
        &p->w,
        &p->g,
        &p->m,
        &p->v,
        &p->w16,
        &tr->sc.opt,
    };
    mda_vkc_run(tr->m->c, &tr->m->k_adamw, bufs, (p->n + 255) / 256, 1, 1, &pc);
}

bool mda_step_is_muon(const mda_step_trainer_t *tr, const mda_model_param_t *p) {
    uint32_t hits = 0;
    for (uint32_t j = 0; j < tr->n_mats; j += 1) {
        hits += tr->mats[j].p == p;
    }
    return hits > 0;
}

void mda_step_adam_rec(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    uint32_t last = 0;
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        if (mda_step_is_muon(tr, &m->params[i]) == false) {
            last = i;
        }
    }
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        mda_model_param_t *p = &m->params[i];
        mda_vkc_set_nobar(m->c, i != last);
        if (mda_step_is_muon(tr, p) == false) {
            mda_step_adam_run(tr, p);
        }
    }
    mda_vkc_set_nobar(m->c, false);
}

uint32_t mda_step_muon_groups(uint32_t op, uint32_t n) {
    switch (op) {
    case 1:
        return 1;
    default:
        return (n + 255) / 256;
    }
}

void mda_step_muon_op(mda_step_trainer_t *tr, mda_step_muon_t *u, uint32_t op, mda_vkc_buf_t *a, mda_vkc_buf_t *b, mda_vkc_buf_t *h, uint32_t n, float p0, float p1) {
    mda_pc_muon_t pc = {
        .n = n,
        .op = op,
        .aux = u->groups,
        .p0 = p0,
        .p1 = p1,
    };
    mda_vkc_buf_t *bufs[6] = {
        a,
        b,
        &u->x,
        h,
        &u->red,
        &tr->sc.opt,
    };
    mda_vkc_run(tr->m->c, &tr->m->k_muon, bufs, mda_step_muon_groups(op, n), 1, 1, &pc);
}

void mda_step_muon_gram(mda_model_t *m, mda_step_muon_t *u, mda_vkc_buf_t *x16) {
    mda_fwd_epi_t epi = {
        .prev = &u->a,
        .alpha = 1.0f,
        .beta = 0.0f,
        .store = MDA_FWD_STORE32 | MDA_FWD_STORE16 | MDA_FWD_STORE_SYM,
    };
    if (u->wide) {
        mda_fwd_mm16_epi(m, x16, MDA_PC_NO_SLOT, x16, MDA_PC_NO_SLOT, &u->a, &u->a16, u->rows, u->rows, u->cols, MDA_FWD_BT, epi);
    } else {
        mda_fwd_mm16_epi(m, x16, MDA_PC_NO_SLOT, x16, MDA_PC_NO_SLOT, &u->a, &u->a16, u->cols, u->cols, u->rows, MDA_FWD_AT, epi);
    }
}

void mda_step_muon_poly(mda_model_t *m, mda_step_muon_t *u) {
    mda_fwd_epi_t epi = {
        .prev = &u->a,
        .alpha = MDA_STEP_MUON_C,
        .beta = MDA_STEP_MUON_B,
        .store = MDA_FWD_STORE16 | MDA_FWD_STORE_SYM,
    };
    mda_fwd_mm16_epi(m, &u->a16, MDA_PC_NO_SLOT, &u->a16, MDA_PC_NO_SLOT, &u->a, &u->b16, u->side, u->side, u->side, MDA_FWD_AB, epi);
}

void mda_step_muon_iterate(mda_model_t *m, mda_step_muon_t *u, mda_vkc_buf_t *x16, mda_vkc_buf_t *next16) {
    mda_fwd_epi_t epi = {
        .prev = &u->x,
        .alpha = 1.0f,
        .beta = MDA_STEP_MUON_A,
        .store = MDA_FWD_STORE32 | MDA_FWD_STORE16,
    };
    if (u->wide) {
        mda_fwd_mm16_epi(m, &u->b16, MDA_PC_NO_SLOT, x16, MDA_PC_NO_SLOT, &u->x, next16, u->rows, u->cols, u->rows, MDA_FWD_AB, epi);
    } else {
        mda_fwd_mm16_epi(m, x16, MDA_PC_NO_SLOT, &u->b16, MDA_PC_NO_SLOT, &u->x, next16, u->rows, u->cols, u->cols, MDA_FWD_AB, epi);
    }
}

void mda_step_muon_stage(mda_step_trainer_t *tr, mda_step_muon_t *u, mda_step_muon_phase_t phase, uint32_t it) {
    mda_model_t *m = tr->m;
    mda_model_param_t *p = u->p;
    uint32_t n = u->rows * u->cols;
    switch (phase) {
    case MDA_STEP_MUON_PHASE_MOMENTUM:
        mda_step_muon_op(tr, u, 0, &p->g, &p->m, &u->x16[0], n, MDA_STEP_MUON_MOMENTUM, 0.0f);
        break;
    case MDA_STEP_MUON_PHASE_NORM:
        mda_step_muon_op(tr, u, 1, &p->g, &p->m, &u->x16[0], u->groups, 0.0f, 0.0f);
        break;
    case MDA_STEP_MUON_PHASE_NORMALIZE:
        mda_step_muon_op(tr, u, 2, &p->g, &p->m, &u->x16[0], n, 0.0f, 0.0f);
        break;
    case MDA_STEP_MUON_PHASE_GRAM:
        mda_step_muon_gram(m, u, &u->x16[it % 2]);
        break;
    case MDA_STEP_MUON_PHASE_POLY:
        mda_step_muon_poly(m, u);
        break;
    case MDA_STEP_MUON_PHASE_ITERATE:
        mda_step_muon_iterate(m, u, &u->x16[it % 2], &u->x16[(it + 1) % 2]);
        break;
    case MDA_STEP_MUON_PHASE_UPDATE:
        mda_step_muon_op(tr, u, 3, &p->w, &p->m, &p->w16, n, MDA_STEP_MUON_RMS * sqrtf((float)mda_util_max_u32(u->rows, u->cols)), 0.0f);
        break;
    }
}

void mda_step_muon_phase(mda_step_trainer_t *tr, mda_step_muon_phase_t phase, uint32_t it) {
    for (uint32_t j = 0; j < tr->n_mats; j += 1) {
        mda_vkc_set_nobar(tr->m->c, j + 1 < tr->n_mats);
        mda_step_muon_stage(tr, &tr->mats[j], phase, it);
    }
}

void mda_step_muon_rec(mda_step_trainer_t *tr) {
    mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_MOMENTUM, 0);
    mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_NORM, 0);
    mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_NORMALIZE, 0);
    for (uint32_t it = 0; it < MDA_STEP_MUON_ITERS; it += 1) {
        mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_GRAM, it);
        mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_POLY, it);
        mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_ITERATE, it);
    }
    mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_UPDATE, 0);
}

void mda_step_opt_rec(mda_step_trainer_t *tr, float clip) {
    mda_step_sumsq_rec(tr, clip);
    mda_step_adam_rec(tr);
    if (tr->muon) {
        mda_step_muon_rec(tr);
    }
    mda_fwd_refresh_wt16_rec(tr->m);
}

mda_step_muon_t mda_step_muon_alloc(mda_vkc_t *c, mda_model_param_t *p) {
    uint32_t side = mda_util_min_u32(p->rows, p->cols);
    uint32_t n = p->rows * p->cols;
    uint32_t groups = (n + 255) / 256;
    return (mda_step_muon_t) {
        .x = mda_vkc_alloc(c, n * sizeof(float)),
        .x16 = {
            mda_vkc_alloc(c, n * sizeof(uint16_t)),
            mda_vkc_alloc(c, n * sizeof(uint16_t)),
        },
        .a = mda_vkc_alloc(c, side * side * sizeof(float)),
        .a16 = mda_vkc_alloc(c, side * side * sizeof(uint16_t)),
        .b16 = mda_vkc_alloc(c, side * side * sizeof(uint16_t)),
        .red = mda_vkc_alloc(c, (groups + 1) * sizeof(float)),
        .p = p,
        .rows = p->rows,
        .cols = p->cols,
        .side = side,
        .groups = groups,
        .wide = p->rows <= p->cols,
    };
}

void mda_step_muon_init(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    tr->n_mats = MDA_STEP_MUON_PER_LAYER * m->cfg.n_layer;
    tr->mats = malloc(tr->n_mats * sizeof(mda_step_muon_t));
    for (uint32_t l = 0; l < m->cfg.n_layer; l += 1) {
        mda_model_layer_t *ly = &m->layers[l];
        mda_model_param_t *ps[MDA_STEP_MUON_PER_LAYER] = {
            ly->wr,
            ly->wk,
            ly->wv,
            ly->wo,
            ly->w_key,
            ly->w_val,
        };
        for (uint32_t j = 0; j < MDA_STEP_MUON_PER_LAYER; j += 1) {
            tr->mats[l * MDA_STEP_MUON_PER_LAYER + j] = mda_step_muon_alloc(m->c, ps[j]);
        }
    }
}

mda_step_trainer_t mda_step_init(mda_model_t *m, float clip, uint32_t accum, bool muon) {
    mda_step_trainer_t tr = {
        .m = m,
        .sc = mda_fwd_alloc(m, m->cfg.n_layer),
        .partials = mda_vkc_alloc(m->c, (m->n_all + 255) / 256 * sizeof(float)),
        .accum = accum,
        .muon = muon,
    };
    if (muon) {
        mda_step_muon_init(&tr);
    }
    tr.first = mda_vkc_record(m->c);
    mda_step_rec(&tr);
    mda_vkc_end(m->c);
    m->grad_acc = true;
    tr.more = mda_vkc_record(m->c);
    mda_step_rec(&tr);
    mda_vkc_end(m->c);
    m->grad_acc = false;
    tr.opt = mda_vkc_record(m->c);
    mda_step_opt_rec(&tr, clip);
    mda_vkc_end(m->c);
    return tr;
}

void mda_step_reduce_kd(mda_step_trainer_t *tr) {
    uint32_t n = mda_fwd_kd_rows(tr->m);
    const float *kl = tr->sc.kd_loss.ptr;
    const float *kb = tr->sc.kd_bits.ptr;
    for (uint32_t i = 0; i < n; i += 1) {
        tr->kd_sum += kl[i];
        tr->kd_student_bits += kb[2 * i];
        tr->kd_bytes += kb[2 * i + 1];
    }
    tr->kd_n += n;
}

void mda_step_reduce(mda_step_trainer_t *tr) {
    if (tr->m->c->fail == false) {
        uint32_t bt = tr->m->cfg.batch * tr->m->cfg.seqlen;
        float *lp = tr->sc.loss.ptr;
        double total = 0.0;
        for (uint32_t i = 0; i < bt; i += 1) {
            total += lp[i];
            tr->ce_sum += lp[i] / tr->sc.batch.weight[i];
            tr->masked += tr->sc.batch.mask[i];
        }
        tr->loss += total * mda_fwd_loss_scale(tr->m->cfg) / tr->accum;
        mda_step_reduce_kd(tr);
    }
}

void mda_step_print_nonfinite(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    float *g = mda_vkc_read(m->c, &m->g_all);
    for (uint32_t i = 0; i < m->n_params && m->c->fail == false; i += 1) {
        mda_model_param_t *p = &m->params[i];
        uint32_t bad = 0;
        for (uint32_t j = 0; j < p->n; j += 1) {
            bad += isfinite(g[p->off + j]) == false;
        }
        if (bad > 0) {
            fprintf(stderr, "param %u %s: %u of %u gradients not finite\n", i, p->name, bad, p->n);
        }
    }
    free(g);
}

void mda_step_clear(mda_step_trainer_t *tr) {
    tr->loss = 0.0;
    tr->ce_sum = 0.0;
    tr->masked = 0.0;
    tr->kd_sum = 0.0;
    tr->kd_n = 0.0;
    tr->kd_student_bits = 0.0;
    tr->kd_teacher_bits = 0.0;
    tr->kd_bytes = 0.0;
    tr->kd_teacher_bytes = 0.0;
}

float mda_step_grad(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    mda_fwd_refresh_w16(m);
    mda_vkc_begin(m->c);
    mda_step_rec(tr);
    mda_vkc_flush(m->c);
    mda_step_clear(tr);
    mda_step_reduce(tr);
    return (float)tr->loss;
}

void mda_step_begin(mda_step_trainer_t *tr, float lr, uint32_t step) {
    mda_fwd_refresh_w16(tr->m);
    mda_fwd_opt_t *o = tr->sc.opt.ptr;
    o->lr = lr;
    o->bc1 = 1.0f - powf(MDA_STEP_ADAM_B1, (float)step);
    o->bc2 = 1.0f - powf(MDA_STEP_ADAM_B2, (float)step);
    mda_step_clear(tr);
}

mda_vkc_cmd_t *mda_step_micro_cmd(mda_step_trainer_t *tr, uint32_t i) {
    if (i == 0) {
        return &tr->first;
    } else {
        return &tr->more;
    }
}

void mda_step_finish(mda_step_trainer_t *tr) {
    mda_fwd_opt_t *o = tr->sc.opt.ptr;
    mda_vkc_submit(tr->m->c, &tr->opt);
    tr->gnorm = o->norm;
    tr->skipped = o->skip != 0.0f;
}

void mda_step_micro(mda_step_trainer_t *tr, uint32_t i) {
    mda_vkc_submit(tr->m->c, mda_step_micro_cmd(tr, i));
    mda_step_reduce(tr);
    if (i + 1 == tr->accum) {
        mda_step_finish(tr);
    }
}

float mda_step_train(mda_step_trainer_t *tr, float lr, uint32_t step) {
    mda_step_begin(tr, lr, step);
    mda_step_micro(tr, 0);
    return (float)tr->loss;
}
