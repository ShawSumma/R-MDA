// step.h -- one full training step: forward (saving activations) + backward (grads
// to every weight), recorded as a single GPU batch and submitted once. Every step,
// including the elementwise glue, runs on the GPU; the host only sums the per-token
// losses afterwards. Activations for layer l are in sc.acts[l].
#ifndef STEP_H
#define STEP_H

#include "bwd.h"

typedef struct trainer_t trainer_t;

struct trainer_t {
    model_t *m;
    fwd_scratch_t sc;
    uint32_t step;
    // unweighted mean CE on masked tokens (nats), low-noise metric
    float last_raw_ce;
};

static trainer_t trainer_make(model_t *m) {
    return (trainer_t) {
        .m = m,
        .sc = fwd_alloc(m, m->cfg.n_layer),
        .step = 0,
    };
}

// rmsnorm backward: given x, gain g, and dy (in the y buffer), produce dx (in x
// buffer) and accumulate dg. Uses the verified rmsnorm op=1.
static void rmsnorm_bwd(model_t *m, vkc_buf_t *x, vkc_buf_t *g, vkc_buf_t *dy,
    vkc_buf_t *stash, vkc_buf_t *dg, uint32_t rows, uint32_t d) {
    rms_pc_t pc = (rms_pc_t) {
        .rows = rows,
        .d = d,
        .op = 1,
    };
    // op1 expects: binding0 x(in)/dx(out), 1 g, 2 y=dy, 3 rms_stash, 4 dg
    vkc_buf_t *bufs[5] = {
        x,
        g,
        dy,
        stash,
        dg,
    };
    vkc_run(m->c, &m->k_rms, bufs, rows, 1, 1, &pc);
}

// adaLN backward for out = x * (1 + scale) + shift (scale/shift = ada6 slots s_slot,
// s_slot + 1): dx = dout * (1 + scale) is written, dscale/dshift are atomically added
// into the same slots of dada6.
static void adaln_bwd(model_t *m, vkc_buf_t *x, vkc_buf_t *ada6, vkc_buf_t *dout,
    vkc_buf_t *dx, vkc_buf_t *dada6, uint32_t s_slot) {
    mcfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    adaln_pc_t pc = (adaln_pc_t) {
        .bt = bt,
        .d = d,
        .seqlen = cfg.seqlen,
        .op = 1,
        .stride = 6 * d,
        .s_off = s_slot * d,
        .sh_off = (s_slot + 1) * d,
    };
    vkc_buf_t *bufs[5] = {
        x,
        ada6,
        dout,
        dx,
        dada6,
    };
    vkc_run(m->c, &m->k_adaln, bufs, (bt * d + 255) / 256, 1, 1, &pc);
}

// One training step: forward (saving acts) + backward (grads to every weight).
// Returns mean masked loss. Assumes batch inputs already in tr->sc.
static float train_step(trainer_t *tr) {
    model_t *m = tr->m;
    mcfg_t cfg = m->cfg;
    fwd_scratch_t *sc = &tr->sc;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hd = cfg.n_head;
    uint32_t dh = d / hd;
    uint32_t hid = cfg.ffn_hidden;
    uint32_t vocab = cfg.vocab;

    // the loss is the mean over masked tokens but the CE kernel produces dlogits for
    // the sum, so the backward scales by 1/cnt. The mask is host-written input, so cnt
    // is known before anything is recorded.
    float *mk = sc->mask.ptr;
    double cnt = 0.0;
    for (uint32_t i = 0; i < bt; ++i) {
        cnt += mk[i];
    }

    // refresh the fp16 weight mirrors (weights changed last optimizer step) so the
    // coopmat matmuls below see the current weights.
    refresh_w16(m);

    vkc_begin(m->c);

    // zero all grads (several are accumulated: the atomic ones and t_proj)
    for (uint32_t i = 0; i < m->n_params; ++i) {
        vec_scale(m, &m->params[i].g, m->params[i].n, 0.0f);
    }

    // ===== FORWARD (saving activations) =====
    forward_rec(m, sc, 1);
    // sc->logits now holds dlogits for the summed loss
    ce_rec(m, sc);

    // ===== BACKWARD =====
    vec_scale(m, &sc->logits, bt * vocab, (float)(1.0 / (cnt + 1e-6)));
    // head: logits = xn @ head, xn = rmsnorm(x_final, n_out)
    mm_dB(m, &sc->xn, &sc->logits, &m->head->g, bt, vocab, d);
    mm_dA(m, &m->head->w, &sc->logits, &sc->dnorm, bt, vocab, d);
    // sc->x still holds the final residual; the rmsnorm backward overwrites it with dx
    rmsnorm_bwd(m, &sc->x, &m->n_out->w, &sc->dnorm, &sc->rms_stash, &m->n_out->g, bt, d);
    vec_op(m, &sc->dxh_buf, &sc->x, bt * d, 5);
    // cond feeds every block; its grad is summed over blocks
    vec_scale(m, &sc->dcond, cfg.batch * d, 0.0f);

    for (int li = (int)cfg.n_layer - 1; li >= 0; --li) {
        mlayer_t *ly = &m->layers[li];
        act_t *a = &sc->acts[li];

        // adaLN modulation grads: the ffn fills slots 2,3, time mixing fills 0,1
        vec_scale(m, &sc->dada6, cfg.batch * 6 * d, 0.0f);

        // ffn backward. forward: n2 = adaLN(rmsnorm(x_mid)), h = silu(gate_pre) * up,
        // x = x_mid + h @ w_down. dxh_buf is the grad of x.
        vec_op(m, &sc->hbuf, &a->gate, bt * hid, 5);
        vec_op(m, &sc->hbuf, &a->up, bt * hid, 1);
        mm_dB(m, &sc->hbuf, &sc->dxh_buf, &ly->w_down->g, bt, d, hid);
        mm_dA(m, &ly->w_down->w, &sc->dxh_buf, &sc->dh, bt, d, hid);
        // d_gate = dh * up -> dh2 ; d_up = dh * gate -> pre
        vec_op(m, &sc->dh2, &sc->dh, bt * hid, 5);
        vec_op(m, &sc->dh2, &a->up, bt * hid, 1);
        vec_op(m, &sc->pre, &sc->dh, bt * hid, 5);
        vec_op(m, &sc->pre, &a->gate, bt * hid, 1);
        mm_dB(m, &a->n2, &sc->pre, &ly->w_up->g, bt, hid, d);
        mm_dA(m, &ly->w_up->w, &sc->pre, &sc->dnorm, bt, hid, d);
        // d_gate_pre = d_gate * silu'(gate_pre), in dh2
        {
            elem_pc_t silu_bwd_push = (elem_pc_t) { .n = bt * hid, .op = 1 };
            vkc_buf_t *silu_bwd_bufs[2] = { &sc->dh2, &a->gate_pre };
            vkc_run(m->c, &m->k_elem, silu_bwd_bufs, (bt * hid + 255) / 256, 1, 1, &silu_bwd_push);
        }
        mm_dB(m, &a->n2, &sc->dh2, &ly->w_gate->g, bt, hid, d);
        mm_dA(m, &ly->w_gate->w, &sc->dh2, &sc->dtmp2, bt, hid, d);
        // dnorm = d_n2
        vec_op(m, &sc->dnorm, &sc->dtmp2, bt * d, 0);
        // dtmp = d_n2_pre
        adaln_bwd(m, &a->n2_pre, &sc->ada6[li], &sc->dnorm, &sc->dtmp, &sc->dada6, 2);
        vec_op(m, &sc->tmp2, &a->x_mid, bt * d, 5);
        rmsnorm_bwd(m, &sc->tmp2, &ly->n2->w, &sc->dtmp, &a->rms2, &ly->n2->g, bt, d);
        // x_mid feeds both the residual and the ffn norm
        vec_op(m, &sc->dxh_buf, &sc->tmp2, bt * d, 0);

        // time-mix backward. forward: x_mid = x_in + wkv @ wo
        mm_dB(m, &a->wkv, &sc->dxh_buf, &ly->wo->g, bt, d, d);
        mm_dA(m, &ly->wo->w, &sc->dxh_buf, &sc->dtmp, bt, d, d);
        // the output was 0.5*(po+po2), so each direction's grad is half the packed dwkv
        gpu_repack(m, &sc->dtmp, &sc->dwkv, bt, hd, dh, cfg.seqlen, 0);
        vec_op(m, &sc->dwkv, &sc->dwkv, bt * d, 4);
        // the two directions are independent: no barrier between them
        wkv_bwd_pc_t wkv_bwd_push = (wkv_bwd_pc_t) {
            .bsz = cfg.batch,
            .hd = hd,
            .t = cfg.seqlen,
            .dh = dh,
            .dir = 0,
        };
        vkc_buf_t *wkv_bwd_bufs[12] = {
            &a->pk,
            &a->pv,
            &a->pr,
            &a->pw,
            &a->pa,
            &a->sstate,
            &sc->dwkv,
            &sc->dpk,
            &sc->dpv,
            &sc->dpr,
            &sc->dpw,
            &sc->dpa,
        };
        vkc_set_nobar(m->c, 1);
        vkc_run(m->c, &m->k_wkv_b, wkv_bwd_bufs, cfg.batch * hd, 1, 1, &wkv_bwd_push);
        vkc_set_nobar(m->c, 0);
        wkv_bwd_push.dir = 1;
        vkc_buf_t *wkv_bwd_bufs2[12] = {
            &a->pk,
            &a->pv,
            &a->pr,
            &a->pw,
            &a->pa,
            &a->sstate2,
            &sc->dwkv,
            &sc->dpk2,
            &sc->dpv2,
            &sc->dpr2,
            &sc->dpw2,
            &sc->dpa2,
        };
        vkc_run(m->c, &m->k_wkv_b, wkv_bwd_bufs2, cfg.batch * hd, 1, 1, &wkv_bwd_push);
        vec_op(m, &sc->dpk, &sc->dpk2, bt * d, 0);
        vec_op(m, &sc->dpv, &sc->dpv2, bt * d, 0);
        vec_op(m, &sc->dpr, &sc->dpr2, bt * d, 0);
        vec_op(m, &sc->dpw, &sc->dpw2, bt * d, 0);
        vec_op(m, &sc->dpa, &sc->dpa2, bt * d, 0);
        // back through k_hat = k/||k|| and the w, a sigmoids, while still packed (the
        // saved k_hat, kinv, w, a are packed)
        norm_heads(m, &sc->dpk, &a->kinv, &a->pk, bt * hd, dh, 1);
        vec_op(m, &sc->dpw, &a->pw, bt * d, 6);
        vec_op(m, &sc->dpa, &a->pa, bt * d, 6);
        // unpack into the forward projection scratch
        gpu_repack(m, &sc->dpk, &sc->pk, bt, hd, dh, cfg.seqlen, 1);
        gpu_repack(m, &sc->dpv, &sc->pv, bt, hd, dh, cfg.seqlen, 1);
        gpu_repack(m, &sc->dpr, &sc->pr, bt, hd, dh, cfg.seqlen, 1);
        gpu_repack(m, &sc->dpw, &sc->pw, bt, hd, dh, cfg.seqlen, 1);
        gpu_repack(m, &sc->dpa, &sc->pa, bt, hd, dh, cfg.seqlen, 1);
        // each projection p = shift @ wX: d_wX = shift^T dp ; d_shift += dp wX^T
        vec_scale(m, &sc->dnorm, bt * d, 0.0f);
        vkc_buf_t *dps[5] = { &sc->pk, &sc->pv, &sc->pr, &sc->pw, &sc->pa };
        param_t *wps[5] = { ly->wk, ly->wv, ly->wr, ly->ww, ly->wa };
        for (int j = 0; j < 5; ++j) {
            mm_dB(m, &a->shift, dps[j], &wps[j]->g, bt, d, d);
            mm_dA(m, &wps[j]->w, dps[j], &sc->dtmp2, bt, d, d);
            vec_op(m, &sc->dnorm, &sc->dtmp2, bt * d, 0);
        }
        // tokenshift backward: input n1, dout = d_shift -> dx (atomic, zeroed), dmix
        vec_scale(m, &sc->dtmp, bt * d, 0.0f);
        {
            ts_pc_t tshift_bwd_push = (ts_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 1 };
            vkc_buf_t *tshift_bwd_bufs[5] = { &a->n1, &ly->shift_mix->w, &sc->dnorm, &sc->dtmp, &ly->shift_mix->g };
            vkc_run(m->c, &m->k_tshift, tshift_bwd_bufs, (bt * d + 255) / 256, 1, 1, &tshift_bwd_push);
        }
        // dh = d_n1_pre
        adaln_bwd(m, &a->n1_pre, &sc->ada6[li], &sc->dtmp, &sc->dh, &sc->dada6, 0);
        // ada6 = cond @ ada (first batch rows)
        mm_dB(m, &sc->cond, &sc->dada6, &ly->ada->g, cfg.batch, 6 * d, d);
        mm_dA(m, &ly->ada->w, &sc->dada6, &sc->dtmp2, cfg.batch, 6 * d, d);
        vec_op(m, &sc->dcond, &sc->dtmp2, cfg.batch * d, 0);
        // n1_pre = rmsnorm(x_in): backward -> dx_in, added to the residual grad
        vec_op(m, &sc->tmp2, &a->x_in, bt * d, 5);
        rmsnorm_bwd(m, &sc->tmp2, &ly->n1->w, &sc->dh, &a->rms1, &ly->n1->g, bt, d);
        vec_op(m, &sc->dxh_buf, &sc->tmp2, bt * d, 0);
    }

    // cond = temb @ t_proj
    mm_dB(m, &sc->temb, &sc->dcond, &m->t_proj->g, cfg.batch, d, d);

    // embed backward: scatter the residual grad into the tok/pos grads.
    {
        emb_pc_t embed_bwd_push = (emb_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 1 };
        vkc_buf_t *embed_bwd_bufs[4] = { &sc->ids, &m->tok_emb->g, &m->pos_emb->g, &sc->dxh_buf };
        vkc_run(m->c, &m->k_embed, embed_bwd_bufs, (bt * d + 255) / 256, 1, 1, &embed_bwd_push);
    }

    vkc_flush(m->c);

    float *lp = sc->loss.ptr;
    float *wt = sc->weight.ptr;
    double total = 0.0;
    // unweighted CE on masked tokens (much less noisy than 1/t-weighted)
    double raw = 0.0;
    for (uint32_t i = 0; i < bt; ++i) {
        total += lp[i];
        // lp = mask*weight*CE -> /weight = mask*CE
        raw += lp[i] / (wt[i] + 1e-9);
    }
    // mean unweighted CE (nats/masked-token)
    tr->last_raw_ce = (float)(raw / (cnt + 1e-6));
    return (float)(total / (cnt + 1e-6));
}

#endif
