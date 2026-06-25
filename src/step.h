// step.h -- one full training step: forward (saving activations) + backward
// (grads to every weight) + returns loss. Built on the verified kernels; cheap
// elementwise glue runs host-side in fp32. Activations for layer l are stored in
// acts[l]; the residual stream snapshots let the backward reverse each block.
#ifndef STEP_H
#define STEP_H

#include "bwd.h"

// per-layer saved activations
typedef struct act_t act_t;
typedef struct trainer_t trainer_t;

struct act_t {
    // [bt*d] residual entering layer
    vkc_buf_t x_in;
    // rmsnorm(x_in)
    vkc_buf_t n1;
    // tokenshift(n1)  (input to k/v/r/w/a projections)
    vkc_buf_t shift;
    // packed normalized k  [bt*d]
    vkc_buf_t pk;
    vkc_buf_t pv;
    vkc_buf_t pr;
    // sigmoid'd, packed
    vkc_buf_t pw;
    // sigmoid'd, packed
    vkc_buf_t pa;
    // pre-sigmoid w (unpacked) for sigmoid' on bwd
    vkc_buf_t pw_pre;
    vkc_buf_t pa_pre;
    // unpacked wkv output [bt*d]
    vkc_buf_t wkv;
    // residual after time-mix
    vkc_buf_t x_mid;
    // rmsnorm(x_mid)
    vkc_buf_t n2;
    // pre-silu gate [bt*hid]
    vkc_buf_t gate_pre;
    // silu(gate)
    vkc_buf_t gate;
    // up proj [bt*hid]
    vkc_buf_t up;
    // saved wkv states (forward pass)
    vkc_buf_t sstate;
    // saved wkv states (backward pass, bidirectional)
    vkc_buf_t sstate2;
    // rms value of the time-mix norm [bt]
    vkc_buf_t rms1;
    // rms value of the ffn norm [bt]
    vkc_buf_t rms2;
    // 1/||k_pre|| per (b,h,t) [bt*hd]
    vkc_buf_t kinv;
    // rmsnorm(x) BEFORE adaLN modulation (time-mix)
    vkc_buf_t n1_pre;
    // rmsnorm(x_mid) BEFORE adaLN modulation (ffn)
    vkc_buf_t n2_pre;
    // [batch, 6d] this block's modulation projection
    vkc_buf_t ada6;
    // [batch, d] the timestep cond fed to this block's ada
    vkc_buf_t cond;
};

struct trainer_t {
    model_t *m;
    fwd_scratch_t sc;
    act_t *acts;
    vkc_buf_t x_final_n;
    uint32_t step;
    // unweighted mean CE on masked tokens (nats), low-noise metric
    float last_raw_ce;
};

static trainer_t trainer_make(model_t *m) {
    mcfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    uint32_t s2 = cfg.batch * cfg.n_head * cfg.seqlen * (d / cfg.n_head) * (d / cfg.n_head);
    trainer_t tr = (trainer_t) {
        .m = m,
        .sc = fwd_alloc(m),
        .acts = malloc(cfg.n_layer * sizeof(act_t)),
        .x_final_n = vkc_alloc(m->c, bt * d * sizeof(float)),
        .step = 0,
    };
    for (uint32_t l = 0; l < cfg.n_layer; ++l) {
        act_t *a = &tr.acts[l];
        a->x_in = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->n1 = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->shift = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->pk = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->pv = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->pr = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->pw = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->pa = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->pw_pre = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->pa_pre = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->wkv = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->x_mid = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->n2 = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->gate_pre = vkc_alloc(m->c, (bt * hid) * sizeof(float));
        a->gate = vkc_alloc(m->c, (bt * hid) * sizeof(float));
        a->up = vkc_alloc(m->c, (bt * hid) * sizeof(float));
        a->sstate = vkc_alloc(m->c, (s2) * sizeof(float));
        a->sstate2 = vkc_alloc(m->c, (s2) * sizeof(float));
        a->rms1 = vkc_alloc(m->c, (bt) * sizeof(float));
        a->rms2 = vkc_alloc(m->c, (bt) * sizeof(float));
        a->kinv = vkc_alloc(m->c, (bt * cfg.n_head) * sizeof(float));
        a->n1_pre = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->n2_pre = vkc_alloc(m->c, (bt * d) * sizeof(float));
        a->ada6 = vkc_alloc(m->c, (cfg.batch * 6 * d) * sizeof(float));
        a->cond = vkc_alloc(m->c, (cfg.batch * d) * sizeof(float));
    }
    return tr;
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

// Compute the timestep conditioning vector cond[b,:] for the diffusion denoiser.
// t is recovered per sequence as 1/weight[b*seqlen]. We sinusoidally embed t into
// d dims, then project by t_proj [d,d] -> cond [batch, d]. A diffusion denoiser
// needs to know the mask rate t so it can adapt to "mostly masked" vs "lightly
// masked" inputs. cond then drives adaLN modulation in each block.
static void compute_cond(trainer_t *tr) {
    model_t *m = tr->m;
    mcfg_t cfg = m->cfg;
    uint32_t d = cfg.d;
    float *wt = tr->sc.weight.ptr;
    // reuse tmp as [batch, d] embedding staging
    float *emb = tr->sc.tmp.ptr;
    for (uint32_t b = 0; b < cfg.batch; ++b) {
        float t = 1.0f / (wt[b * cfg.seqlen] + 1e-9f);
        float *row = emb + b * d;
        for (uint32_t i = 0; i < d; ++i) {
            float freq = expf(-logf(10000.0f) * (float)(i / 2) / (float)(d / 2));
            float ang = t * freq;
            row[i] = ((i % 2) == 0) ? sinf(ang) : cosf(ang);
        }
    }
    // cond = emb @ t_proj  ([batch, d] @ [d, d])
    mm(m, &tr->sc.tmp, &m->t_proj->w, &tr->sc.cond, cfg.batch, d, d);
}

// Extract a contiguous [batch, d] slice (slot `which` of 6) from ada6 [batch, 6d]
// into dst, for feeding the adaln kernel (which wants scale/shift as [batch,d]).
static void ada_slice(trainer_t *tr, uint32_t which, float *dst) {
    mcfg_t cfg = tr->m->cfg;
    uint32_t d = cfg.d;
    float *a6 = tr->sc.ada6.ptr;
    for (uint32_t b = 0; b < cfg.batch; ++b) {
        memcpy(dst + b * d, a6 + b * 6 * d + which * d, d * sizeof(float));
    }
}

// adaLN forward on a [bt,d] activation in `buf`: out = buf*(1+scale)+shift, where
// scale/shift come from slots (s_slot, sh_slot) of the per-block ada6 projection.
// Uses scratch sc->pre / sc->hbuf staging for the contiguous slices (batch*d each,
// tiny). Result written back into buf.
static void ada_apply(trainer_t *tr, vkc_buf_t *buf, vkc_buf_t *scale_b,
    vkc_buf_t *shift_b, uint32_t s_slot, uint32_t sh_slot) {
    model_t *m = tr->m;
    mcfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    ada_slice(tr, s_slot, scale_b->ptr);
    ada_slice(tr, sh_slot, shift_b->ptr);
    adaln_pc_t pc = (adaln_pc_t) {
        .bt = bt,
        .d = d,
        .seqlen = cfg.seqlen,
        .op = 0,
    };
    vkc_buf_t *bufs[7] = {
        buf,
        scale_b,
        shift_b,
        buf,
        buf,
        scale_b,
        shift_b,
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
    uint32_t nstate = cfg.batch * hd * cfg.seqlen * dh * dh;

    // refresh the fp16 weight mirrors (weights changed last optimizer step) so the
    // coopmat matmuls below see the current weights.
    refresh_w16(m);

    // zero all grads
    for (uint32_t i = 0; i < m->n_params; ++i) {
        host_zero(&m->params[i].g, m->params[i].n);
    }

    // ===== FORWARD (saving activations) =====
    emb_pc_t embed_push = (emb_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 0 };
    vkc_buf_t *embed_bufs[4] = { &sc->ids, &m->tok_emb->w, &m->pos_emb->w, &sc->x };
    vkc_run(m->c, &m->k_embed, embed_bufs, (bt * d + 255) / 256, 1, 1, &embed_push);

    // timestep conditioning vector (foundation ready; adaLN application pending
    // its backward -- task #37). Computed but not yet consumed.
    compute_cond(tr);

    for (uint32_t l = 0; l < cfg.n_layer; ++l) {
        mlayer_t *ly = &m->layers[l];
        act_t *a = &tr->acts[l];
        vec_op(m, &a->x_in, &sc->x, bt * d, 5);

        // project timestep cond -> this block's 6 modulation vectors [batch, 6d]
        mm(m, &sc->cond, &ly->ada->w, &sc->ada6, cfg.batch, 6 * d, d);
        vec_op(m, &a->ada6, &sc->ada6, cfg.batch * 6 * d, 5);
        vec_op(m, &a->cond, &sc->cond, cfg.batch * d, 5);
        // time-mix: n1 = rmsnorm(x), then adaLN modulate (scale=slot0, shift=slot1)
        rmsnorm_fwd(m, &sc->x, &ly->n1->w, &sc->tmp, &sc->rms_stash, bt, d);
        vec_op(m, &a->rms1, &sc->rms_stash, bt, 5);
        vec_op(m, &a->n1_pre, &sc->tmp, bt * d, 5);
        ada_apply(tr, &sc->tmp, &sc->ada_scale, &sc->ada_shift, 0, 1);
        vec_op(m, &a->n1, &sc->tmp, bt * d, 5);
        // tokenshift -> tmp2
        ts_pc_t tshift_push = (ts_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 0 };
        vkc_buf_t *tshift_bufs[5] = { &sc->tmp, &ly->shift_mix->w, &sc->tmp2, &sc->pk, &sc->dg_dummy };
        vkc_run(m->c, &m->k_tshift, tshift_bufs, (bt * d + 255) / 256, 1, 1, &tshift_push);
        vec_op(m, &a->shift, &sc->tmp2, bt * d, 5);

        // k,v,r,w,a projections
        mm16(m, sc, &sc->tmp2, ly->wk, &sc->pk, bt, d, d);
        mm16(m, sc, &sc->tmp2, ly->wv, &sc->pv, bt, d, d);
        mm16(m, sc, &sc->tmp2, ly->wr, &sc->pr, bt, d, d);
        mm16(m, sc, &sc->tmp2, ly->ww, &sc->pw, bt, d, d);
        mm16(m, sc, &sc->tmp2, ly->wa, &sc->pa, bt, d, d);
        vec_op(m, &a->pw_pre, &sc->pw, bt * d, 5);
        vec_op(m, &a->pa_pre, &sc->pa, bt * d, 5);
        // sigmoid on GPU
        vec_op(m, &sc->pw, &sc->pw, bt * d, 2);
        vec_op(m, &sc->pa, &sc->pa, bt * d, 2);

        // pack k,v,r,w,a to [b,head,t,dh] on GPU, normalize k
        gpu_repack(m, &sc->pk, &sc->tmp, bt, hd, dh, cfg.seqlen, 0);
        vec_op(m, &sc->pk, &sc->tmp, bt * d, 5);
        gpu_repack(m, &sc->pv, &sc->tmp, bt, hd, dh, cfg.seqlen, 0);
        vec_op(m, &sc->pv, &sc->tmp, bt * d, 5);
        gpu_repack(m, &sc->pr, &sc->tmp, bt, hd, dh, cfg.seqlen, 0);
        vec_op(m, &sc->pr, &sc->tmp, bt * d, 5);
        gpu_repack(m, &sc->pw, &sc->tmp, bt, hd, dh, cfg.seqlen, 0);
        vec_op(m, &sc->pw, &sc->tmp, bt * d, 5);
        gpu_repack(m, &sc->pa, &sc->tmp, bt, hd, dh, cfg.seqlen, 0);
        vec_op(m, &sc->pa, &sc->tmp, bt * d, 5);
        norm_heads(m, &sc->pk, &a->kinv, bt * hd, dh, 1);
        vec_op(m, &a->pk, &sc->pk, bt * d, 5);
        vec_op(m, &a->pv, &sc->pv, bt * d, 5);
        vec_op(m, &a->pr, &sc->pr, bt * d, 5);
        vec_op(m, &a->pw, &sc->pw, bt * d, 5);
        vec_op(m, &a->pa, &sc->pa, bt * d, 5);

        // wkv7 forward -- BIDIRECTIONAL: a left-to-right pass (dir=0) and a
        // right-to-left pass (dir=1). A diffusion denoiser must use both past and
        // future context to fill masked tokens, so we sum the two passes' outputs.
        wkv_pc_t wkv_push = (wkv_pc_t) { .bsz = cfg.batch, .hd = hd, .t = cfg.seqlen, .dh = dh, .dir = 0, .save = 1 };
        vkc_buf_t *wkv_bufs[7] = { &sc->pk, &sc->pv, &sc->pr, &sc->pw, &sc->pa, &sc->po, &sc->sstate };
        vkc_run(m->c, &m->k_wkv_f, wkv_bufs, cfg.batch * hd, 1, 1, &wkv_push);
        vec_op(m, &a->sstate, &sc->sstate, nstate, 5);

        wkv_pc_t wkv_push2 = (wkv_pc_t) { .bsz = cfg.batch, .hd = hd, .t = cfg.seqlen, .dh = dh, .dir = 1, .save = 1 };
        vkc_buf_t *wkv_bufs2[7] = { &sc->pk, &sc->pv, &sc->pr, &sc->pw, &sc->pa, &sc->po2, &sc->sstate2 };
        vkc_run(m->c, &m->k_wkv_f, wkv_bufs2, cfg.batch * hd, 1, 1, &wkv_push2);
        vec_op(m, &a->sstate2, &sc->sstate2, nstate, 5);

        // combine the two directions: po += po2 (sum), then unpack
        // average the two directions
        vec_op(m, &sc->po, &sc->po2, bt * d, 3);
        gpu_repack(m, &sc->po, &sc->tmp2, bt, hd, dh, cfg.seqlen, 1);
        vec_op(m, &a->wkv, &sc->tmp2, bt * d, 5);

        // wo projection + residual
        mm16(m, sc, &sc->tmp2, ly->wo, &sc->tmp, bt, d, d);
        // x += wo(wkv)
        vec_op(m, &sc->x, &sc->tmp, bt * d, 0);
        vec_op(m, &a->x_mid, &sc->x, bt * d, 5);

        // ffn: n2 = rmsnorm(x_mid), then adaLN modulate (scale2=slot2, shift2=slot3)
        rmsnorm_fwd(m, &sc->x, &ly->n2->w, &sc->tmp, &sc->rms_stash, bt, d);
        // pre-adaLN (for adaln bwd)
        vec_op(m, &a->n2_pre, &sc->tmp, bt * d, 5);
        ada_apply(tr, &sc->tmp, &sc->ada_scale, &sc->ada_shift, 2, 3);
        // post-adaLN (ffn gate/up input)
        vec_op(m, &a->n2, &sc->tmp, bt * d, 5);
        vec_op(m, &a->rms2, &sc->rms_stash, bt, 5);
        mm16(m, sc, &sc->tmp, ly->w_gate, &sc->hbuf, bt, hid, d);
        vec_op(m, &a->gate_pre, &sc->hbuf, bt * hid, 5);
        elem_pc_t silu_push = (elem_pc_t) { .n = bt * hid, .op = 0 };
        vkc_buf_t *silu_bufs[2] = { &sc->hbuf, &sc->pre };
        vkc_run(m->c, &m->k_elem, silu_bufs, (bt * hid + 255) / 256, 1, 1, &silu_push);
        vec_op(m, &a->gate, &sc->hbuf, bt * hid, 5);
        mm16(m, sc, &sc->tmp, ly->w_up, &sc->pre, bt, hid, d);
        vec_op(m, &a->up, &sc->pre, bt * hid, 5);
        // gate *= up
        vec_op(m, &sc->hbuf, &sc->pre, bt * hid, 1);
        mm16(m, sc, &sc->hbuf, ly->w_down, &sc->tmp2, bt, d, hid);
        // x += down(h)
        vec_op(m, &sc->x, &sc->tmp2, bt * d, 0);
    }

    // head
    rmsnorm_fwd(m, &sc->x, &m->layers[0].n1->w, &sc->tmp, &sc->rms_stash, bt, d);
    vec_op(m, &tr->x_final_n, &sc->tmp, bt * d, 5);
    mm16(m, sc, &sc->tmp, m->head, &sc->logits, bt, m->cfg.vocab, d);

    ce_pc_t ce_push = (ce_pc_t) { .rows = bt, .vocab = m->cfg.vocab };
    vkc_buf_t *ce_bufs[5] = { &sc->logits, &sc->target, &sc->mask, &sc->weight, &sc->loss };
    vkc_run(m->c, &m->k_ce, ce_bufs, bt, 1, 1, &ce_push);

    float *lp = sc->loss.ptr;
    float *mk = sc->mask.ptr;
    float *wt = sc->weight.ptr;
    double total = 0.0;
    double cnt = 0.0;
    // unweighted CE on masked tokens (much less noisy than 1/t-weighted)
    double raw = 0.0;
    for (uint32_t i = 0; i < bt; ++i) {
        total += lp[i];
        cnt += mk[i];
        // lp = mask*weight*CE -> /weight = mask*CE
        raw += lp[i] / (wt[i] + 1e-9);
    }
    float loss = (float)(total / (cnt + 1e-6));
    // mean unweighted CE (nats/masked-token)
    tr->last_raw_ce = (float)(raw / (cnt + 1e-6));

    // ===== BACKWARD =====
    // the loss is mean over masked tokens (total/cnt) but the CE kernel produced
    // dlogits for the SUM; scale by 1/cnt so the backward matches the mean loss.
    {
        float inv = 1.0f / (float)(cnt + 1e-6);
        vec_scale(m, &sc->logits, bt * m->cfg.vocab, inv);
    }
    // sc->logits now holds dlogits (the CE kernel wrote it).
    // head: logits = head(x_final_n).  d_head += x_final_n^T dlogits ; dx_n = dlogits head^T
    // a = x_final_n
    vec_op(m, &sc->tmp, &tr->x_final_n, bt * d, 5);
    mm_dB(m, &sc->tmp, &sc->logits, &m->head->g, bt, m->cfg.vocab, d);
    // dx_n in dx_head
    mm_dA(m, &m->head->w, &sc->logits, &sc->dx_head, bt, m->cfg.vocab, d);
    // rmsnorm before head backward: x stream grad. need x (final residual) -> recompute? we
    // have a->x at last layer's x_mid + ffn add == sc->x currently still holds final x.
    // dy = dx_head, x = sc->x (final), gain = layers[0].n1
    vec_op(m, &sc->dnorm, &sc->dx_head, bt * d, 5);
    rmsnorm_bwd(m, &sc->x, &m->layers[0].n1->w, &sc->dnorm, &sc->rms_stash, &sc->dg_dummy, bt, d);
    // sc->x now holds dx (grad of final residual stream). copy to dx accumulator.
    // dxh_buf = grad of residual stream
    memcpy(sc->dxh_buf.ptr, sc->x.ptr, bt * d * sizeof(float));

    for (int li = (int)cfg.n_layer - 1; li >= 0; --li) {
        mlayer_t *ly = &m->layers[li];
        act_t *a = &tr->acts[li];

        // per-block adaLN modulation-grad accumulator: ffn bwd fills slots 2,3,
        // time-mix bwd fills slots 0,1, then the ada/t_proj backward consumes it.
        vec_scale(m, &sc->dada6, cfg.batch * 6 * d, 0.0f);

        // ffn backward
        // forward: x_mid -> n2=rmsnorm -> gate=w_gate(n2), gate=silu ; up=w_up(n2);
        //          h = gate*up ; out = w_down(h) ; x = x_mid + out
        // dxh is grad of x (== grad of x_mid via residual, plus grad through ffn path).
        // grad to out == dxh. d_w_down += h^T dxh ; dh = dxh w_down^T
        // h_in = gate*up -> recompute
        vec_op(m, &sc->hbuf, &a->gate, bt * hid, 5);
        vec_op(m, &sc->hbuf, &a->up, bt * hid, 1);
        // dout = dxh
        vec_op(m, &sc->dtmp, &sc->dxh_buf, bt * d, 5);
        mm_dB(m, &sc->hbuf, &sc->dtmp, &ly->w_down->g, bt, d, hid);
        // dh [bt,hid]
        mm_dA(m, &ly->w_down->w, &sc->dtmp, &sc->dh, bt, d, hid);
        // h = gate*up : d_gate = dh*up ; d_up = dh*gate
        // d_gate = dh*up into dh2 ; d_up = dh*gate into pre (GPU)
        vec_op(m, &sc->dh2, &sc->dh, bt * hid, 5);
        vec_op(m, &sc->dh2, &a->up, bt * hid, 1);
        vec_op(m, &sc->pre, &sc->dh, bt * hid, 5);
        vec_op(m, &sc->pre, &a->gate, bt * hid, 1);
        // up = w_up(n2): d_w_up += n2^T dup ; dn2 += w_up^T dup
        vec_op(m, &sc->tmp, &a->n2, bt * d, 5);
        mm_dB(m, &sc->tmp, &sc->pre, &ly->w_up->g, bt, hid, d);
        // dn2 part1 -> dnorm
        mm_dA(m, &ly->w_up->w, &sc->pre, &sc->dnorm, bt, hid, d);
        // gate = silu(gate_pre): d_gate_pre = d_gate * silu'(gate_pre)
        // preact into hbuf
        vec_op(m, &sc->hbuf, &a->gate_pre, bt * hid, 5);
        // d_gate into pre (grad)
        vec_op(m, &sc->pre, &sc->dh2, bt * hid, 5);
        {
            elem_pc_t silu_bwd_push = (elem_pc_t) { .n = bt * hid, .op = 1 };
            vkc_buf_t *silu_bwd_bufs[2] = { &sc->pre, &sc->hbuf };
            vkc_run(m->c, &m->k_elem, silu_bwd_bufs, (bt * hid + 255) / 256, 1, 1, &silu_bwd_push);
        }
        // gate_pre = w_gate(n2): d_w_gate += n2^T d_gate_pre ; dn2 += w_gate^T d_gate_pre
        mm_dB(m, &sc->tmp, &sc->pre, &ly->w_gate->g, bt, hid, d);
        // dn2 part2
        mm_dA(m, &ly->w_gate->w, &sc->pre, &sc->dtmp2, bt, hid, d);
        // dn2 total (post-adaLN) in dnorm
        vec_op(m, &sc->dnorm, &sc->dtmp2, bt * d, 0);
        // adaLN backward for the ffn norm: n2 = n2_pre*(1+scale2)+shift2. dnorm holds
        // d_n2 (dout). Recover d_n2_pre (->dtmp) and scatter dscale2/dshift2 into the
        // per-block dada6 accumulator slots 2,3. (dada6 was zeroed at block start.)
        {
            float *a6 = a->ada6.ptr;
            float *scl = sc->ada_scale.ptr;
            for (uint32_t b = 0; b < cfg.batch; ++b) {
                memcpy(scl + b * d, a6 + b * 6 * d + 2 * d, d * sizeof(float));
            }
            vec_op(m, &sc->tmp, &a->n2_pre, bt * d, 5);
            // dscale2 out
            vec_scale(m, &sc->dtmp2, cfg.batch * d, 0.0f);
            // dshift2 out (dh2 is [bt,hid]>=d)
            vec_scale(m, &sc->dh2, cfg.batch * d, 0.0f);
            // d_n2_pre out (separate)
            vec_scale(m, &sc->dh, bt * d, 0.0f);
            adaln_pc_t adaln_push = (adaln_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 1 };
            vkc_buf_t *adaln_bufs[7] = { &sc->tmp, &sc->ada_scale, &sc->ada_scale, &sc->dnorm, &sc->dh, &sc->dtmp2, &sc->dh2 };
            vkc_run(m->c, &m->k_adaln, adaln_bufs, (bt * d + 255) / 256, 1, 1, &adaln_push);
            float *dd = sc->dada6.ptr;
            float *dsc = sc->dtmp2.ptr;
            float *dsh = sc->dh2.ptr;
            for (uint32_t b = 0; b < cfg.batch; ++b) {
                memcpy(dd + b * 6 * d + 2 * d, dsc + b * d, d * sizeof(float));
                memcpy(dd + b * 6 * d + 3 * d, dsh + b * d, d * sizeof(float));
            }
            // dnorm = d_n2_pre
            memcpy(sc->dnorm.ptr, sc->dh.ptr, bt * d * sizeof(float));
        }
        // n2 = rmsnorm(x_mid): backward -> dx_mid. need x_mid in a buffer.
        vec_op(m, &sc->tmp2, &a->x_mid, bt * d, 5);
        // dy
        vec_op(m, &sc->dtmp, &sc->dnorm, bt * d, 5);
        vec_op(m, &sc->rms_stash, &a->rms2, bt, 5);
        rmsnorm_bwd(m, &sc->tmp2, &ly->n2->w, &sc->dtmp, &sc->rms_stash, &ly->n2->g, bt, d);
        // dx from ffn-norm path; add to dxh (residual: x_mid feeds both) -- GPU
        vec_op(m, &sc->dxh_buf, &sc->tmp2, bt * d, 0);

        // time-mix backward
        // out_tm = wo(wkv_unpacked) ; x_mid = x_in + out_tm. grad to out_tm == dxh.
        // wkv_unpacked
        vec_op(m, &sc->tmp2, &a->wkv, bt * d, 5);
        // d_out_tm
        vec_op(m, &sc->dtmp, &sc->dxh_buf, bt * d, 5);
        mm_dB(m, &sc->tmp2, &sc->dtmp, &ly->wo->g, bt, d, d);
        // d wkv_unpacked [bt,d]
        mm_dA(m, &ly->wo->w, &sc->dtmp, &sc->dwkv, bt, d, d);
        gpu_repack(m, &sc->dwkv, &sc->tmp, bt, hd, dh, cfg.seqlen, 0);
        memcpy(sc->dwkv.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        // output was 0.5*(po+po2), so each direction's grad is 0.5*dwkv -> halve it
        vec_op(m, &sc->dwkv, &sc->dwkv, bt * d, 4);
        // wkv backward: needs packed k,v,r,w,a, saved states, dO -> dk,dv,dr,dw,da
        vec_op(m, &sc->pk, &a->pk, bt * d, 5);
        vec_op(m, &sc->pv, &a->pv, bt * d, 5);
        vec_op(m, &sc->pr, &a->pr, bt * d, 5);
        vec_op(m, &sc->pw, &a->pw, bt * d, 5);
        vec_op(m, &sc->pa, &a->pa, bt * d, 5);
        vec_op(m, &sc->sstate, &a->sstate, nstate, 5);
        wkv_bwd_pc_t wkv_bwd_push = (wkv_bwd_pc_t) {
            .bsz = cfg.batch,
            .hd = hd,
            .t = cfg.seqlen,
            .dh = dh,
            .dir = 0,
        };
        // dwkv is the grad of (po + po2), so it is the grad for BOTH directions.
        // dir=0 pass (uses sstate) -> dpk..dpa
        vkc_buf_t *wkv_bwd_bufs[12] = {
            &sc->pk,
            &sc->pv,
            &sc->pr,
            &sc->pw,
            &sc->pa,
            &sc->sstate,
            &sc->dwkv,
            &sc->dpk,
            &sc->dpv,
            &sc->dpr,
            &sc->dpw,
            &sc->dpa,
        };
        vkc_run(m->c, &m->k_wkv_b, wkv_bwd_bufs, cfg.batch * hd, 1, 1, &wkv_bwd_push);
        // dir=1 pass (uses sstate2) -> dpk2..dpa2, then sum into dpk..dpa
        wkv_bwd_push.dir = 1;
        vec_op(m, &sc->sstate, &a->sstate2, nstate, 5);
        vkc_buf_t *wkv_bwd_bufs2[12] = {
            &sc->pk,
            &sc->pv,
            &sc->pr,
            &sc->pw,
            &sc->pa,
            &sc->sstate,
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
        // k normalization backward (on PACKED dpk, before unpack): k_hat = k/||k||,
        //   dk = (dk_hat - k_hat*(k_hat . dk_hat)) / ||k||  per (b,h,t) head-vector.
        // packed layout: vector (b,h,t) is contiguous at ((b*hd+h)*seq+t)*dh; the
        // kinv index is the same (b*hd+h)*seq+t, so they align.
        {
            float *dpk = sc->dpk.ptr;
            // packed, unit-norm k
            float *khat = a->pk.ptr;
            uint32_t nvec = bt * hd;
            for (uint32_t vct = 0; vct < nvec; ++vct) {
                float *dv = dpk + vct * dh;
                float *kv = khat + vct * dh;
                float dot = 0.0f;
                for (uint32_t e = 0; e < dh; ++e) {
                    dot += kv[e] * dv[e];
                }
                float inv = ((float*)a->kinv.ptr)[vct];
                for (uint32_t e = 0; e < dh; ++e) {
                    dv[e] = (dv[e] - kv[e] * dot) * inv;
                }
            }
        }
        gpu_repack(m, &sc->dpk, &sc->tmp, bt, hd, dh, cfg.seqlen, 1);
        memcpy(sc->dpk.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        gpu_repack(m, &sc->dpv, &sc->tmp, bt, hd, dh, cfg.seqlen, 1);
        memcpy(sc->dpv.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        gpu_repack(m, &sc->dpr, &sc->tmp, bt, hd, dh, cfg.seqlen, 1);
        memcpy(sc->dpr.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        gpu_repack(m, &sc->dpw, &sc->tmp, bt, hd, dh, cfg.seqlen, 1);
        memcpy(sc->dpw.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        gpu_repack(m, &sc->dpa, &sc->tmp, bt, hd, dh, cfg.seqlen, 1);
        memcpy(sc->dpa.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        // sigmoid backward for w,a: d_pre = d_post * s*(1-s), s = a->pw/pa (post)
        // sigmoid-derivative multiply on the GPU (op 6): dp *= s*(1-s), s = saved sigmoid
        vec_op(m, &sc->dpw, &a->pw, bt * d, 6);
        vec_op(m, &sc->dpa, &a->pa, bt * d, 6);
        // each projection p = proj(shift): d_wX += shift^T dp ; d_shift += sum wX^T dp
        // shift input
        vec_op(m, &sc->tmp, &a->shift, bt * d, 5);
        // accumulate d_shift here
        vec_scale(m, &sc->dnorm, bt * d, 0.0f);
        vkc_buf_t *dps[5] = { &sc->dpk, &sc->dpv, &sc->dpr, &sc->dpw, &sc->dpa };
        param_t *wps[5] = { ly->wk, ly->wv, ly->wr, ly->ww, ly->wa };
        for (int j = 0; j < 5; ++j) {
            mm_dB(m, &sc->tmp, dps[j], &wps[j]->g, bt, d, d);
            mm_dA(m, &wps[j]->w, dps[j], &sc->dtmp2, bt, d, d);
            // accumulate d_shift on GPU
            vec_op(m, &sc->dnorm, &sc->dtmp2, bt * d, 0);
        }
        // tokenshift backward: input n1, mix shift_mix, dout=d_shift -> dx(n1), dmix
        // tokenshift input
        vec_op(m, &sc->tmp, &a->n1, bt * d, 5);
        // dout = d_shift
        vec_op(m, &sc->tmp2, &sc->dnorm, bt * d, 5);
        // dx output
        vec_scale(m, &sc->dtmp, bt * d, 0.0f);
        {
            ts_pc_t tshift_bwd_push = (ts_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 1 };
            vkc_buf_t *tshift_bwd_bufs[5] = { &sc->tmp, &ly->shift_mix->w, &sc->tmp2, &sc->dtmp, &ly->shift_mix->g };
            vkc_run(m->c, &m->k_tshift, tshift_bwd_bufs, (bt * d + 255) / 256, 1, 1, &tshift_bwd_push);
        }
        // adaLN backward: n1 = n1_pre*(1+scale1)+shift1. dtmp holds d_n1 (dout).
        {
            float *a6 = a->ada6.ptr;
            float *scl = sc->ada_scale.ptr;
            for (uint32_t b = 0; b < cfg.batch; ++b) {
                memcpy(scl + b * d, a6 + b * 6 * d + 0 * d, d * sizeof(float));
            }
            vec_op(m, &sc->tmp, &a->n1_pre, bt * d, 5);
            vec_scale(m, &sc->dnorm, cfg.batch * d, 0.0f);
            vec_scale(m, &sc->dtmp2, cfg.batch * d, 0.0f);
            vec_scale(m, &sc->dh, bt * d, 0.0f);
            adaln_pc_t adaln_push = (adaln_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 1 };
            vkc_buf_t *adaln_bufs[7] = { &sc->tmp, &sc->ada_scale, &sc->ada_scale, &sc->dtmp, &sc->dh, &sc->dnorm, &sc->dtmp2 };
            vkc_run(m->c, &m->k_adaln, adaln_bufs, (bt * d + 255) / 256, 1, 1, &adaln_push);
            // scatter into slots 0,1 (FFN already filled 2,3 -- do NOT re-zero dada6)
            float *dd = sc->dada6.ptr;
            float *dsc = sc->dnorm.ptr;
            float *dsh = sc->dtmp2.ptr;
            for (uint32_t b = 0; b < cfg.batch; ++b) {
                memcpy(dd + b * 6 * d + 0 * d, dsc + b * d, d * sizeof(float));
                memcpy(dd + b * 6 * d + 1 * d, dsh + b * d, d * sizeof(float));
            }
            vec_op(m, &sc->tmp2, &a->cond, cfg.batch * d, 5);
            mm_dB(m, &sc->tmp2, &sc->dada6, &ly->ada->g, cfg.batch, 6 * d, d);
            mm_dA(m, &ly->ada->w, &sc->dada6, &sc->dcond, cfg.batch, 6 * d, d);
            {
                float *emb = sc->tmp2.ptr;
                float *wt = sc->weight.ptr;
                for (uint32_t b = 0; b < cfg.batch; ++b) {
                    float t = 1.0f / (wt[b * cfg.seqlen] + 1e-9f);
                    float *row = emb + b * d;
                    for (uint32_t i = 0; i < d; ++i) {
                        float freq = expf(-logf(10000.0f) * (float)(i / 2) / (float)(d / 2));
                        float ang = t * freq;
                        row[i] = ((i % 2) == 0) ? sinf(ang) : cosf(ang);
                    }
                }
            }
            mm_dB(m, &sc->tmp2, &sc->dcond, &sc->dx_head, cfg.batch, d, d);
            vec_op(m, &m->t_proj->g, &sc->dx_head, d * d, 0);
            memcpy(sc->dtmp.ptr, sc->dh.ptr, bt * d * sizeof(float));
        }
        // dtmp = d_n1_pre (grad of rmsnorm output). rmsnorm backward -> dx_in.
        vec_op(m, &sc->tmp2, &a->x_in, bt * d, 5);
        vec_op(m, &sc->rms_stash, &a->rms1, bt, 5);
        rmsnorm_bwd(m, &sc->tmp2, &ly->n1->w, &sc->dtmp, &sc->rms_stash, &ly->n1->g, bt, d);
        vec_op(m, &sc->dxh_buf, &sc->tmp2, bt * d, 0);
    }

    // embed backward: dxh is grad of x after embed. scatter into tok/pos grads.
    memcpy(sc->x.ptr, sc->dxh_buf.ptr, bt * d * sizeof(float));
    vec_scale(m, &m->tok_emb->g, m->tok_emb->n, 0.0f);
    vec_scale(m, &m->pos_emb->g, m->pos_emb->n, 0.0f);
    {
        emb_pc_t embed_bwd_push = (emb_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 1 };
        vkc_buf_t *embed_bwd_bufs[4] = { &sc->ids, &m->tok_emb->g, &m->pos_emb->g, &sc->x };
        vkc_run(m->c, &m->k_embed, embed_bwd_bufs, (bt * d + 255) / 256, 1, 1, &embed_bwd_push);
    }

    return loss;
}

#endif

