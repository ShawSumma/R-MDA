#include <math.h>
#include <stdlib.h>

#include "step.h"
#include "pc.h"
#include "bwd.h"
#include "util.h"

void mda_step_rmsnorm_bwd(mda_model_t *m, mda_vkc_buf_t *x, mda_vkc_buf_t *g, mda_vkc_buf_t *dy, mda_vkc_buf_t *stash, mda_vkc_buf_t *dg, mda_vkc_buf_t *dacc, uint32_t rows, uint32_t d) {
    mda_pc_rms_t pc = {
        .rows = rows,
        .d = d,
        .op = 1,
    };
    mda_vkc_buf_t *bufs[6] = {
        x,
        g,
        dy,
        stash,
        dg,
        dacc,
    };
    mda_vkc_run(m->c, &m->k_rms, bufs, rows, 1, 1, &pc);
}

void mda_step_head_bwd(mda_model_t *m, mda_fwd_scratch_t *sc) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t vocab = cfg.vocab;
    mda_model_param_t *e = m->tok_emb;
    if (m->ref32) {
        mda_fwd_mm(m, &sc->logits, &e->w, &sc->dnorm, bt, d, vocab);
        mda_bwd_mm_db(m, &sc->logits, &sc->xn, &e->g, bt, d, vocab);
    } else {
        uint32_t sd = mda_fwd_cast_act(m, &sc->logits, &sc->dc16, bt * vocab);
        mda_fwd_transpose16(m, &sc->logits, &sc->dlt16, bt, vocab, sd);
        uint32_t sx = mda_fwd_cast_act(m, &sc->xn, &sc->a16, bt * d);
        mda_vkc_set_nobar(m->c, true);
        mda_fwd_mm16_raw(m, &sc->dc16, sd, &e->w16, MDA_PC_NO_SLOT, &sc->dnorm, bt, d, vocab, false);
        mda_vkc_set_nobar(m->c, false);
        mda_fwd_mm16_raw(m, &sc->dlt16, sd, &sc->a16, sx, &e->g, vocab, d, bt, m->grad_acc);
    }
}

void mda_step_ffn_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, mda_vkc_buf_t *ada4) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    mda_bwd_mm_at(m, sc, &a->h, bt, hid);
    mda_bwd_mm(m, sc, &a->h, ly->w_down, &sc->dxh_buf, &sc->dh, bt, d, hid);
    mda_pc_elem_t pc = {
        .n = bt * hid,
        .op = 1,
    };
    mda_vkc_buf_t *bufs[5] = {
        &sc->dh,
        &a->gate_pre,
        &a->up,
        &a->gate,
        &sc->dup,
    };
    mda_vkc_run(m->c, &m->k_elem, bufs, (bt * hid + 255) / 256, 1, 1, &pc);
    mda_bwd_mm_at(m, sc, &a->n2, bt, d);
    mda_bwd_mm(m, sc, &a->n2, ly->w_up, &sc->dup, &sc->dnorm, bt, hid, d);
    mda_bwd_mm(m, sc, &a->n2, ly->w_gate, &sc->dh, &sc->dtmp2, bt, hid, d);
    mda_fwd_vec_op(m, &sc->dnorm, &sc->dtmp2, bt * d, MDA_PC_VEC_ADD);
    mda_fwd_adaln(m, &a->n2_pre, ada4, &sc->dnorm, &sc->dtmp, &sc->dada4, 1, 2);
    mda_step_rmsnorm_bwd(m, &a->x_mid, &ly->n2->w, &sc->dtmp, &a->rms2, &ly->n2->g, &sc->dxh_buf, bt, d);
}

void mda_step_heads_run(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, mda_vkc_buf_t **d2) {
    mda_vkc_buf_t *hb[19] = {
        &sc->dpk,
        &sc->dpv,
        &sc->dpr,
        &sc->dpw,
        &sc->dpa,
        d2[0],
        d2[1],
        d2[2],
        d2[3],
        d2[4],
        &a->pk,
        &a->kinv,
        &a->pw,
        &a->pa,
        &sc->pk,
        &sc->pv,
        &sc->pr,
        &sc->pw,
        &sc->pa,
    };
    mda_fwd_heads(m, &m->k_heads_b, hb);
}

void mda_step_heads_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a) {
    mda_vkc_buf_t *own[5] = {
        &sc->dpk,
        &sc->dpv,
        &sc->dpr,
        &sc->dpw,
        &sc->dpa,
    };
    mda_vkc_buf_t *other[5] = {
        &sc->dpk2,
        &sc->dpv2,
        &sc->dpr2,
        &sc->dpw2,
        &sc->dpa2,
    };
    if (m->causal) {
        mda_step_heads_run(m, sc, a, own);
    } else {
        mda_step_heads_run(m, sc, a, other);
    }
}

void mda_step_wkv_bwd_run(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a, uint32_t dir, uint32_t seq0, uint32_t count) {
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
    };
    mda_vkc_buf_t *fwd0[6] = {
        &a->sstate,
        &sc->dpk,
        &sc->dpv,
        &sc->dpr,
        &sc->dpw,
        &sc->dpa,
    };
    mda_vkc_buf_t *fwd1[6] = {
        &a->sstate2,
        &sc->dpk2,
        &sc->dpv2,
        &sc->dpr2,
        &sc->dpw2,
        &sc->dpa2,
    };
    mda_vkc_buf_t **d = fwd0;
    if (dir == 1) {
        d = fwd1;
    }
    mda_vkc_buf_t *bufs[13] = {
        &a->pk,
        &a->pv,
        &a->pr,
        &a->pw,
        &a->pa,
        d[0],
        &sc->dwkv,
        d[1],
        d[2],
        d[3],
        d[4],
        d[5],
        &sc->dbstate,
    };
    mda_vkc_run(m->c, &m->k_wkv_b, bufs, count * cfg.n_head, 1, 1, &pc);
}

void mda_step_wkv_bwd_blocks(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t nclean = mda_model_nclean(cfg);
    uint32_t n = nclean * cfg.seqlen * cfg.d;
    mda_vkc_set_nobar(m->c, true);
    mda_step_wkv_bwd_run(m, sc, a, 0, nclean, nclean);
    mda_vkc_set_nobar(m->c, false);
    mda_step_wkv_bwd_run(m, sc, a, 1, nclean, nclean);
    mda_step_wkv_bwd_run(m, sc, a, 0, 0, nclean);
    mda_vkc_buf_t *from[5] = {
        &sc->dpk,
        &sc->dpv,
        &sc->dpr,
        &sc->dpw,
        &sc->dpa,
    };
    mda_vkc_buf_t *to[5] = {
        &sc->dpk2,
        &sc->dpv2,
        &sc->dpr2,
        &sc->dpw2,
        &sc->dpa2,
    };
    for (uint32_t i = 0; i < 5; i += 1) {
        mda_vkc_buf_t src = mda_vkc_view(from[i], 0, (VkDeviceSize)n * sizeof(float));
        mda_vkc_buf_t dst = mda_vkc_view(to[i], 0, (VkDeviceSize)n * sizeof(float));
        mda_vkc_set_nobar(m->c, i + 1 < 5);
        mda_fwd_vec_op(m, &dst, &src, n, MDA_PC_VEC_COPY);
    }
}

void mda_step_wkv_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_fwd_act_t *a) {
    mda_model_cfg_t cfg = m->cfg;
    if (m->causal) {
        mda_step_wkv_bwd_run(m, sc, a, 0, 0, cfg.batch);
    } else if (cfg.block > 0) {
        mda_step_wkv_bwd_blocks(m, sc, a);
    } else {
        mda_vkc_set_nobar(m->c, true);
        mda_step_wkv_bwd_run(m, sc, a, 0, 0, cfg.batch);
        mda_vkc_set_nobar(m->c, false);
        mda_step_wkv_bwd_run(m, sc, a, 1, 0, cfg.batch);
    }
    mda_step_heads_bwd(m, sc, a);
}

void mda_step_time_mix_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, mda_vkc_buf_t *ada4) {
    mda_model_cfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    mda_bwd_mm_at(m, sc, &a->wkv, bt, d);
    mda_bwd_mm(m, sc, &a->wkv, ly->wo, &sc->dxh_buf, &sc->dtmp, bt, d, d);
    mda_fwd_headnorm(m, &a->wkv_raw, &sc->dtmp, &a->rmsx, &sc->dtmp2, 1);
    mda_fwd_repack(m, &sc->dtmp2, &sc->dwkv, 0);
    mda_step_wkv_bwd(m, sc, a);
    mda_vkc_buf_t *dps[5] = {
        &sc->pk,
        &sc->pv,
        &sc->pr,
        &sc->pw,
        &sc->pa,
    };
    mda_model_param_t *wps[5] = {
        ly->wk,
        ly->wv,
        ly->wr,
        ly->ww,
        ly->wa,
    };
    mda_bwd_mm_at(m, sc, &a->shift, bt, d);
    mda_bwd_mm(m, sc, &a->shift, wps[0], dps[0], &sc->dnorm, bt, d, d);
    for (uint32_t j = 1; j < 5; j += 1) {
        mda_bwd_mm(m, sc, &a->shift, wps[j], dps[j], &sc->dtmp2, bt, d, d);
        mda_fwd_vec_op(m, &sc->dnorm, &sc->dtmp2, bt * d, MDA_PC_VEC_ADD);
    }
    mda_vkc_fill(m->c, &sc->dtmp, 0);
    mda_pc_ts_t pc = {
        .bt = bt,
        .d = d,
        .seqlen = cfg.seqlen,
        .op = 1,
        .blk = cfg.block,
        .nclean = mda_model_nclean(cfg),
    };
    mda_vkc_buf_t *bufs[7] = {
        &a->n1,
        &ly->shift_mix->w,
        &sc->dnorm,
        &sc->dtmp,
        &ly->shift_mix->g,
        &sc->dg_dummy,
        &sc->dg_dummy,
    };
    mda_vkc_run(m->c, &m->k_tshift, bufs, (bt * d + 255) / 256, 1, 1, &pc);
    mda_fwd_adaln(m, &a->n1_pre, ada4, &sc->dtmp, &sc->dh, &sc->dada4, 1, 0);
    mda_step_rmsnorm_bwd(m, &a->x_in, &ly->n1->w, &sc->dh, &a->rms1, &ly->n1->g, &sc->dxh_buf, bt, d);
}

void mda_step_ada_bwd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly) {
    uint32_t crow = mda_fwd_cond_rows(m->cfg);
    uint32_t d = m->cfg.d;
    if (m->ref32) {
        mda_bwd_mm_db(m, &sc->temb, &sc->dada4, &ly->ada->g, crow, 4 * d, d);
    } else {
        mda_bwd_mm_at(m, sc, &sc->temb, crow, d);
        uint32_t sd = mda_fwd_cast_act(m, &sc->dada4, &sc->dc16, crow * 4 * d);
        mda_fwd_mm16_raw(m, &sc->at16, sc->at16_slot, &sc->dc16, sd, &ly->ada->g, d, 4 * d, crow, m->grad_acc);
    }
}

void mda_step_embed_bwd(mda_model_t *m, mda_fwd_scratch_t *sc) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
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
        &sc->dxh_buf,
    };
    mda_vkc_run(m->c, &m->k_embed, bufs, (bt * m->cfg.d + 255) / 256, 1, 1, &pc);
}

void mda_step_rec(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    mda_fwd_scratch_t *sc = &tr->sc;
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
    mda_vkc_set_nobar(m->c, true);
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        if (m->params[i].accum && m->grad_acc == false) {
            mda_vkc_fill(m->c, &m->params[i].g, 0);
        }
    }
    mda_vkc_set_nobar(m->c, false);
    mda_vkc_fill(m->c, &sc->dxh_buf, 0);
    mda_fwd_rec(m, sc, true);
    mda_fwd_ce_rec(m, sc);
    mda_step_head_bwd(m, sc);
    mda_step_rmsnorm_bwd(m, &sc->x, &m->n_out->w, &sc->dnorm, &sc->rms_stash, &m->n_out->g, &sc->dxh_buf, bt, m->cfg.d);
    for (uint32_t i = m->cfg.n_layer; i > 0; i -= 1) {
        mda_vkc_fill(m->c, &sc->dada4, 0);
        mda_step_ffn_bwd(m, sc, &m->layers[i - 1], &sc->acts[i - 1], &sc->ada4[i - 1]);
        mda_step_time_mix_bwd(m, sc, &m->layers[i - 1], &sc->acts[i - 1], &sc->ada4[i - 1]);
        mda_step_ada_bwd(m, sc, &m->layers[i - 1]);
    }
    mda_step_embed_bwd(m, sc);
}

void mda_step_sumsq_rec(mda_step_trainer_t *tr, float clip) {
    mda_model_t *m = tr->m;
    uint32_t groups = (m->n_all + 255) / 256;
    mda_pc_sumsq_t spc = {
        .n = m->n_all,
        .op = 0,
        .gscale = mda_fwd_loss_scale(m->cfg) / (float)tr->accum,
        .clip = clip,
        .growth = MDA_STEP_GMUL_GROWTH,
    };
    mda_vkc_buf_t *sbufs[3] = {
        &m->g_all,
        &tr->partials,
        &tr->sc.opt,
    };
    mda_vkc_run(m->c, &m->k_sumsq, sbufs, groups, 1, 1, &spc);
    mda_pc_sumsq_t rpc = spc;
    rpc.n = groups;
    rpc.op = 1;
    mda_vkc_run(m->c, &m->k_sumsq, sbufs, 1, 1, 1, &rpc);
}

void mda_step_adam_run(mda_step_trainer_t *tr, mda_vkc_buf_t *w, mda_vkc_buf_t *g, mda_vkc_buf_t *mom, mda_vkc_buf_t *v, mda_vkc_buf_t *w16, uint32_t n) {
    mda_pc_adam_t pc = {
        .n = n,
        .beta1 = MDA_STEP_ADAM_B1,
        .beta2 = MDA_STEP_ADAM_B2,
        .eps = 1e-8f,
        .wd = 0.0f,
    };
    mda_vkc_buf_t *bufs[6] = {
        w,
        g,
        mom,
        v,
        w16,
        &tr->sc.opt,
    };
    mda_vkc_run(tr->m->c, &tr->m->k_adamw, bufs, (n + 255) / 256, 1, 1, &pc);
}

bool mda_step_is_muon(const mda_step_trainer_t *tr, const mda_model_param_t *p) {
    uint32_t hits = 0;
    for (uint32_t j = 0; j < tr->n_mats; j += 1) {
        hits += tr->mats[j].p == p;
    }
    return hits > 0;
}

void mda_step_adam_rest_rec(mda_step_trainer_t *tr) {
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
            mda_step_adam_run(tr, &p->w, &p->g, &p->m, &p->v, &p->w16, p->n);
        }
    }
    mda_vkc_set_nobar(m->c, false);
}

void mda_step_adam_rec(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    if (tr->muon) {
        mda_step_adam_rest_rec(tr);
    } else {
        mda_step_adam_run(tr, &m->w_all, &m->g_all, &m->m_all, &m->v_all, &m->w16_all, m->n_all);
    }
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

void mda_step_muon_gram(mda_model_t *m, mda_step_muon_t *u) {
    if (u->wide) {
        mda_fwd_mm16_raw(m, &u->x16, MDA_PC_NO_SLOT, &u->xt16, MDA_PC_NO_SLOT, &u->a, u->rows, u->rows, u->cols, false);
    } else {
        mda_fwd_mm16_raw(m, &u->xt16, MDA_PC_NO_SLOT, &u->x16, MDA_PC_NO_SLOT, &u->a, u->cols, u->cols, u->rows, false);
    }
}

void mda_step_muon_bx(mda_model_t *m, mda_step_muon_t *u) {
    if (u->wide) {
        mda_fwd_mm16_raw(m, &u->b16, MDA_PC_NO_SLOT, &u->x16, MDA_PC_NO_SLOT, &u->bx, u->rows, u->cols, u->rows, false);
    } else {
        mda_fwd_mm16_raw(m, &u->x16, MDA_PC_NO_SLOT, &u->b16, MDA_PC_NO_SLOT, &u->bx, u->rows, u->cols, u->cols, false);
    }
}

void mda_step_muon_stage(mda_step_trainer_t *tr, mda_step_muon_t *u, mda_step_muon_phase_t phase) {
    mda_model_t *m = tr->m;
    mda_model_param_t *p = u->p;
    uint32_t n = u->rows * u->cols;
    uint32_t sq = u->side * u->side;
    switch (phase) {
    case MDA_STEP_MUON_PHASE_MOMENTUM:
        mda_step_muon_op(tr, u, 0, &p->g, &p->m, &u->x16, n, MDA_STEP_MUON_MOMENTUM, 0.0f);
        break;
    case MDA_STEP_MUON_PHASE_NORM:
        mda_step_muon_op(tr, u, 1, &p->g, &p->m, &u->x16, u->groups, 0.0f, 0.0f);
        break;
    case MDA_STEP_MUON_PHASE_NORMALIZE:
        mda_step_muon_op(tr, u, 2, &p->g, &p->m, &u->x16, n, 0.0f, 0.0f);
        break;
    case MDA_STEP_MUON_PHASE_TRANSPOSE:
        mda_fwd_transpose16(m, &u->x, &u->xt16, u->rows, u->cols, MDA_PC_NO_SLOT);
        break;
    case MDA_STEP_MUON_PHASE_GRAM:
        mda_step_muon_gram(m, u);
        break;
    case MDA_STEP_MUON_PHASE_CAST:
        mda_fwd_cast(m, &u->a, &u->a16, sq, MDA_PC_NO_SLOT);
        break;
    case MDA_STEP_MUON_PHASE_SQUARE:
        mda_fwd_mm16_raw(m, &u->a16, MDA_PC_NO_SLOT, &u->a16, MDA_PC_NO_SLOT, &u->aa, u->side, u->side, u->side, false);
        break;
    case MDA_STEP_MUON_PHASE_POLY:
        mda_step_muon_op(tr, u, 3, &u->a, &u->aa, &u->b16, sq, MDA_STEP_MUON_B, MDA_STEP_MUON_C);
        break;
    case MDA_STEP_MUON_PHASE_BX:
        mda_step_muon_bx(m, u);
        break;
    case MDA_STEP_MUON_PHASE_ITERATE:
        mda_step_muon_op(tr, u, 4, &u->bx, &u->aa, &u->x16, n, MDA_STEP_MUON_A, 0.0f);
        break;
    case MDA_STEP_MUON_PHASE_UPDATE:
        mda_step_muon_op(tr, u, 5, &p->w, &p->m, &p->w16, n, MDA_STEP_MUON_RMS * sqrtf((float)mda_util_max_u32(u->rows, u->cols)), 0.0f);
        break;
    }
}

void mda_step_muon_phase(mda_step_trainer_t *tr, mda_step_muon_phase_t phase) {
    for (uint32_t j = 0; j < tr->n_mats; j += 1) {
        mda_vkc_set_nobar(tr->m->c, j + 1 < tr->n_mats);
        mda_step_muon_stage(tr, &tr->mats[j], phase);
    }
}

void mda_step_muon_rec(mda_step_trainer_t *tr) {
    mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_MOMENTUM);
    mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_NORM);
    mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_NORMALIZE);
    for (uint32_t it = 0; it < MDA_STEP_MUON_ITERS; it += 1) {
        mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_TRANSPOSE);
        mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_GRAM);
        mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_CAST);
        mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_SQUARE);
        mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_POLY);
        mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_BX);
        mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_ITERATE);
    }
    mda_step_muon_phase(tr, MDA_STEP_MUON_PHASE_UPDATE);
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
        .x16 = mda_vkc_alloc(c, n * sizeof(uint16_t)),
        .xt16 = mda_vkc_alloc(c, n * sizeof(uint16_t)),
        .a = mda_vkc_alloc(c, side * side * sizeof(float)),
        .a16 = mda_vkc_alloc(c, side * side * sizeof(uint16_t)),
        .aa = mda_vkc_alloc(c, side * side * sizeof(float)),
        .b16 = mda_vkc_alloc(c, side * side * sizeof(uint16_t)),
        .bx = mda_vkc_alloc(c, n * sizeof(float)),
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
            ly->wk,
            ly->wv,
            ly->wr,
            ly->ww,
            ly->wa,
            ly->wo,
            ly->w_gate,
            ly->w_up,
            ly->w_down,
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
    }
}

void mda_step_clear(mda_step_trainer_t *tr) {
    tr->loss = 0.0;
    tr->ce_sum = 0.0;
    tr->masked = 0.0;
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
