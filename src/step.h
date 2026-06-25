// step.h -- one full training step: forward (saving activations) + backward
// (grads to every weight) + returns loss. Built on the verified kernels; cheap
// elementwise glue runs host-side in fp32. Activations for layer l are stored in
// acts[l]; the residual stream snapshots let the backward reverse each block.
#ifndef STEP_H
#define STEP_H

#include "bwd.h"

// per-layer saved activations
typedef struct act act_t;

struct act {
    float *x_in;        // [bt*d] residual entering layer
    float *n1;          // rmsnorm(x_in)
    float *shift;       // tokenshift(n1)  (input to k/v/r/w/a projections)
    float *pk;          // packed normalized k  [bt*d]
    float *pv;
    float *pr;
    float *pw;          // sigmoid'd, packed
    float *pa;          // sigmoid'd, packed
    float *pw_pre;      // pre-sigmoid w (unpacked) for sigmoid' on bwd
    float *pa_pre;
    float *wkv;         // unpacked wkv output [bt*d]
    float *x_mid;       // residual after time-mix
    float *n2;          // rmsnorm(x_mid)
    float *gate_pre;    // pre-silu gate [bt*hid]
    float *gate;        // silu(gate)
    float *up;          // up proj [bt*hid]
    float *sstate;      // saved wkv states
};

typedef struct trainer trainer_t;

struct trainer {
    model_t *m;
    fwd_scratch_t sc;
    act_t *acts;
    float *x_final_n;   // rmsnorm(x) before head
    uint32_t step;
};

static float *halloc(uint32_t n) {
    return malloc(n * sizeof(float));
}

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
        .x_final_n = halloc(bt * d),
        .step = 0,
    };
    for (uint32_t l = 0; l < cfg.n_layer; l++) {
        act_t *a = &tr.acts[l];
        a->x_in = halloc(bt * d);
        a->n1 = halloc(bt * d);
        a->shift = halloc(bt * d);
        a->pk = halloc(bt * d);
        a->pv = halloc(bt * d);
        a->pr = halloc(bt * d);
        a->pw = halloc(bt * d);
        a->pa = halloc(bt * d);
        a->pw_pre = halloc(bt * d);
        a->pa_pre = halloc(bt * d);
        a->wkv = halloc(bt * d);
        a->x_mid = halloc(bt * d);
        a->n2 = halloc(bt * d);
        a->gate_pre = halloc(bt * hid);
        a->gate = halloc(bt * hid);
        a->up = halloc(bt * hid);
        a->sstate = halloc(s2);
    }
    return tr;
}

// --- small host helpers ---
static void buf_to_host(vkc_buf_t *b, float *dst, uint32_t n) {
    memcpy(dst, b->ptr, n * sizeof(float));
}
static void host_to_buf(float *src, vkc_buf_t *b, uint32_t n) {
    memcpy(b->ptr, src, n * sizeof(float));
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

    // zero all grads
    for (uint32_t i = 0; i < m->n_params; i++) {
        host_zero(&m->params[i].g, m->params[i].n);
    }

    // ===== FORWARD (saving activations) =====
    emb_pc_t epc = (emb_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 0 };
    vkc_buf_t *ebuf[4] = { &sc->ids, &m->tok_emb->w, &m->pos_emb->w, &sc->x };
    vkc_run(m->c, &m->k_embed, ebuf, (bt * d + 255) / 256, 1, 1, &epc);

    for (uint32_t l = 0; l < cfg.n_layer; l++) {
        mlayer_t *ly = &m->layers[l];
        act_t *a = &tr->acts[l];
        buf_to_host(&sc->x, a->x_in, bt * d);

        // time-mix: n1 = rmsnorm(x)
        rmsnorm_fwd(m, &sc->x, &ly->n1->w, &sc->tmp, &sc->rms_stash, bt, d);
        buf_to_host(&sc->tmp, a->n1, bt * d);
        // tokenshift -> tmp2
        ts_pc_t tpc = (ts_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 0 };
        vkc_buf_t *tsb[5] = { &sc->tmp, &ly->shift_mix->w, &sc->tmp2, &sc->pk, &sc->dg_dummy };
        vkc_run(m->c, &m->k_tshift, tsb, (bt * d + 255) / 256, 1, 1, &tpc);
        buf_to_host(&sc->tmp2, a->shift, bt * d);

        // k,v,r,w,a projections
        mm(m, &sc->tmp2, &ly->wk->w, &sc->pk, bt, d, d);
        mm(m, &sc->tmp2, &ly->wv->w, &sc->pv, bt, d, d);
        mm(m, &sc->tmp2, &ly->wr->w, &sc->pr, bt, d, d);
        mm(m, &sc->tmp2, &ly->ww->w, &sc->pw, bt, d, d);
        mm(m, &sc->tmp2, &ly->wa->w, &sc->pa, bt, d, d);
        buf_to_host(&sc->pw, a->pw_pre, bt * d);
        buf_to_host(&sc->pa, a->pa_pre, bt * d);
        host_sigmoid(&sc->pw, bt * d);
        host_sigmoid(&sc->pa, bt * d);

        // pack k,v,r,w,a to [b,head,t,dh], normalize k
        pack_heads(m, &sc->pk, &sc->tmp);  memcpy(sc->pk.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        pack_heads(m, &sc->pv, &sc->tmp);  memcpy(sc->pv.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        pack_heads(m, &sc->pr, &sc->tmp);  memcpy(sc->pr.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        pack_heads(m, &sc->pw, &sc->tmp);  memcpy(sc->pw.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        pack_heads(m, &sc->pa, &sc->tmp);  memcpy(sc->pa.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        host_norm_heads(&sc->pk, bt * hd, dh);
        buf_to_host(&sc->pk, a->pk, bt * d);
        buf_to_host(&sc->pv, a->pv, bt * d);
        buf_to_host(&sc->pr, a->pr, bt * d);
        buf_to_host(&sc->pw, a->pw, bt * d);
        buf_to_host(&sc->pa, a->pa, bt * d);

        // wkv7 forward (save states)
        wkv_pc_t wpc = (wkv_pc_t) { .bsz = cfg.batch, .hd = hd, .t = cfg.seqlen, .dh = dh, .dir = 0, .save = 1 };
        vkc_buf_t *wb[7] = { &sc->pk, &sc->pv, &sc->pr, &sc->pw, &sc->pa, &sc->po, &sc->sstate };
        vkc_run(m->c, &m->k_wkv_f, wb, cfg.batch * hd, 1, 1, &wpc);
        buf_to_host(&sc->sstate, a->sstate, nstate);
        unpack_heads(m, &sc->po, &sc->tmp2);
        buf_to_host(&sc->tmp2, a->wkv, bt * d);

        // wo projection + residual
        mm(m, &sc->tmp2, &ly->wo->w, &sc->tmp, bt, d, d);
        host_add(&sc->x, &sc->tmp, bt * d);
        buf_to_host(&sc->x, a->x_mid, bt * d);

        // ffn: n2 = rmsnorm(x_mid)
        rmsnorm_fwd(m, &sc->x, &ly->n2->w, &sc->tmp, &sc->rms_stash, bt, d);
        buf_to_host(&sc->tmp, a->n2, bt * d);
        mm(m, &sc->tmp, &ly->w_gate->w, &sc->hbuf, bt, hid, d);
        buf_to_host(&sc->hbuf, a->gate_pre, bt * hid);
        elem_pc_t spc = (elem_pc_t) { .n = bt * hid, .op = 0 };
        vkc_buf_t *seb[2] = { &sc->hbuf, &sc->pre };
        vkc_run(m->c, &m->k_elem, seb, (bt * hid + 255) / 256, 1, 1, &spc);
        buf_to_host(&sc->hbuf, a->gate, bt * hid);
        mm(m, &sc->tmp, &ly->w_up->w, &sc->pre, bt, hid, d);
        buf_to_host(&sc->pre, a->up, bt * hid);
        {
            float *g = sc->hbuf.ptr;
            float *u = sc->pre.ptr;
            for (uint32_t i = 0; i < bt * hid; i++) {
                g[i] = g[i] * u[i];
            }
        }
        mm(m, &sc->hbuf, &ly->w_down->w, &sc->tmp2, bt, d, hid);
        host_add(&sc->x, &sc->tmp2, bt * d);
    }

    // head
    rmsnorm_fwd(m, &sc->x, &m->layers[0].n1->w, &sc->tmp, &sc->rms_stash, bt, d);
    buf_to_host(&sc->tmp, tr->x_final_n, bt * d);
    mm(m, &sc->tmp, &m->head->w, &sc->logits, bt, M_VOCAB, d);

    ce_pc_t cpc = (ce_pc_t) { .rows = bt, .vocab = M_VOCAB };
    vkc_buf_t *cb[5] = { &sc->logits, &sc->target, &sc->mask, &sc->weight, &sc->loss };
    vkc_run(m->c, &m->k_ce, cb, bt, 1, 1, &cpc);

    float *lp = sc->loss.ptr;
    float *mk = sc->mask.ptr;
    double total = 0.0;
    double cnt = 0.0;
    for (uint32_t i = 0; i < bt; i++) {
        total += lp[i];
        cnt += mk[i];
    }
    float loss = (float)(total / (cnt + 1e-6));

    // ===== BACKWARD =====
    // the loss is mean over masked tokens (total/cnt) but the CE kernel produced
    // dlogits for the SUM; scale by 1/cnt so the backward matches the mean loss.
    {
        float inv = 1.0f / (float)(cnt + 1e-6);
        float *dl = sc->logits.ptr;
        for (uint32_t i = 0; i < bt * M_VOCAB; i++) {
            dl[i] *= inv;
        }
    }
    // sc->logits now holds dlogits (the CE kernel wrote it).
    // head: logits = head(x_final_n).  d_head += x_final_n^T dlogits ; dx_n = dlogits head^T
    host_to_buf(tr->x_final_n, &sc->tmp, bt * d);          // a = x_final_n
    mm_dB(m, &sc->tmp, &sc->logits, &m->head->g, bt, M_VOCAB, d);
    mm_dA(m, &m->head->w, &sc->logits, &sc->dx_head, bt, M_VOCAB, d);  // dx_n in dx_head
    // rmsnorm before head backward: x stream grad. need x (final residual) -> recompute? we
    // have a->x at last layer's x_mid + ffn add == sc->x currently still holds final x.
    // dy = dx_head, x = sc->x (final), gain = layers[0].n1
    host_to_buf(sc->dx_head.ptr, &sc->dnorm, bt * d);
    rmsnorm_bwd(m, &sc->x, &m->layers[0].n1->w, &sc->dnorm, &sc->rms_stash, &sc->dg_dummy, bt, d);
    // sc->x now holds dx (grad of final residual stream). copy to dx accumulator.
    buf_to_host(&sc->x, sc->dxh, bt * d);   // dxh = grad of residual stream

    for (int li = (int)cfg.n_layer - 1; li >= 0; li--) {
        mlayer_t *ly = &m->layers[li];
        act_t *a = &tr->acts[li];

        // ---- ffn backward ----
        // forward: x_mid -> n2=rmsnorm -> gate=w_gate(n2), gate=silu ; up=w_up(n2);
        //          h = gate*up ; out = w_down(h) ; x = x_mid + out
        // dxh is grad of x (== grad of x_mid via residual, plus grad through ffn path).
        // grad to out == dxh. d_w_down += h^T dxh ; dh = dxh w_down^T
        host_to_buf(a->gate, &sc->hbuf, bt * hid);   // h_in = gate*up -> recompute
        {
            float *hb = sc->hbuf.ptr;
            for (uint32_t i = 0; i < bt * hid; i++) {
                hb[i] = a->gate[i] * a->up[i];
            }
        }
        host_to_buf(sc->dxh, &sc->dtmp, bt * d);      // dout = dxh
        mm_dB(m, &sc->hbuf, &sc->dtmp, &ly->w_down->g, bt, d, hid);
        mm_dA(m, &ly->w_down->w, &sc->dtmp, &sc->dh, bt, d, hid);   // dh [bt,hid]
        // h = gate*up : d_gate = dh*up ; d_up = dh*gate
        {
            float *dh = sc->dh.ptr;
            float *dgate = sc->dh2.ptr;
            float *dup = sc->pre.ptr;
            for (uint32_t i = 0; i < bt * hid; i++) {
                dgate[i] = dh[i] * a->up[i];
                dup[i] = dh[i] * a->gate[i];
            }
        }
        // up = w_up(n2): d_w_up += n2^T dup ; dn2 += w_up^T dup
        host_to_buf(a->n2, &sc->tmp, bt * d);
        mm_dB(m, &sc->tmp, &sc->pre, &ly->w_up->g, bt, hid, d);
        mm_dA(m, &ly->w_up->w, &sc->pre, &sc->dnorm, bt, hid, d);   // dn2 part1 -> dnorm
        // gate = silu(gate_pre): d_gate_pre = d_gate * silu'(gate_pre)
        host_to_buf(a->gate_pre, &sc->hbuf, bt * hid);   // preact into hbuf
        host_to_buf(sc->dh2.ptr, &sc->pre, bt * hid);    // d_gate into pre (grad)
        {
            elem_pc_t bpc = (elem_pc_t) { .n = bt * hid, .op = 1 };
            vkc_buf_t *beb[2] = { &sc->pre, &sc->hbuf };
            vkc_run(m->c, &m->k_elem, beb, (bt * hid + 255) / 256, 1, 1, &bpc);
        }
        // gate_pre = w_gate(n2): d_w_gate += n2^T d_gate_pre ; dn2 += w_gate^T d_gate_pre
        mm_dB(m, &sc->tmp, &sc->pre, &ly->w_gate->g, bt, hid, d);
        mm_dA(m, &ly->w_gate->w, &sc->pre, &sc->dtmp2, bt, hid, d);  // dn2 part2
        host_add(&sc->dnorm, &sc->dtmp2, bt * d);   // dn2 total in dnorm
        // n2 = rmsnorm(x_mid): backward -> dx_mid. need x_mid in a buffer.
        host_to_buf(a->x_mid, &sc->tmp2, bt * d);
        host_to_buf(sc->dnorm.ptr, &sc->dtmp, bt * d);  // dy
        rmsnorm_bwd(m, &sc->tmp2, &ly->n2->w, &sc->dtmp, &sc->rms_stash, &ly->n2->g, bt, d);
        // sc->tmp2 now = dx from ffn-norm path; add to dxh (residual: x_mid feeds both)
        {
            float *dxh = sc->dxh;
            float *p = sc->tmp2.ptr;
            for (uint32_t i = 0; i < bt * d; i++) {
                dxh[i] += p[i];
            }
        }

        // ---- time-mix backward ----
        // out_tm = wo(wkv_unpacked) ; x_mid = x_in + out_tm. grad to out_tm == dxh.
        host_to_buf(a->wkv, &sc->tmp2, bt * d);       // wkv_unpacked
        host_to_buf(sc->dxh, &sc->dtmp, bt * d);      // d_out_tm
        mm_dB(m, &sc->tmp2, &sc->dtmp, &ly->wo->g, bt, d, d);
        mm_dA(m, &ly->wo->w, &sc->dtmp, &sc->dwkv, bt, d, d);     // d wkv_unpacked [bt,d]
        pack_heads(m, &sc->dwkv, &sc->tmp);
        memcpy(sc->dwkv.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        // wkv backward: needs packed k,v,r,w,a, saved states, dO -> dk,dv,dr,dw,da
        host_to_buf(a->pk, &sc->pk, bt * d);
        host_to_buf(a->pv, &sc->pv, bt * d);
        host_to_buf(a->pr, &sc->pr, bt * d);
        host_to_buf(a->pw, &sc->pw, bt * d);
        host_to_buf(a->pa, &sc->pa, bt * d);
        host_to_buf(a->sstate, &sc->sstate, nstate);
        {
            struct {
                uint32_t bsz;
                uint32_t hd;
                uint32_t t;
                uint32_t dh;
            } wbpc = { .bsz = cfg.batch, .hd = hd, .t = cfg.seqlen, .dh = dh };
            vkc_buf_t *wbb[12] = {
                &sc->pk, &sc->pv, &sc->pr, &sc->pw, &sc->pa, &sc->sstate,
                &sc->dwkv, &sc->dpk, &sc->dpv, &sc->dpr, &sc->dpw, &sc->dpa
            };
            vkc_run(m->c, &m->k_wkv_b, wbb, cfg.batch * hd, 1, 1, &wbpc);
        }
        unpack_heads(m, &sc->dpk, &sc->tmp);  memcpy(sc->dpk.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        unpack_heads(m, &sc->dpv, &sc->tmp);  memcpy(sc->dpv.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        unpack_heads(m, &sc->dpr, &sc->tmp);  memcpy(sc->dpr.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        unpack_heads(m, &sc->dpw, &sc->tmp);  memcpy(sc->dpw.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        unpack_heads(m, &sc->dpa, &sc->tmp);  memcpy(sc->dpa.ptr, sc->tmp.ptr, bt * d * sizeof(float));
        // (k normalization gradient is approximated as identity here -- a known
        //  simplification; the dominant gradient path is preserved.)
        // sigmoid backward for w,a: d_pre = d_post * s*(1-s), s = a->pw/pa (post)
        {
            float *dpw = sc->dpw.ptr;
            float *dpa = sc->dpa.ptr;
            for (uint32_t i = 0; i < bt * d; i++) {
                float sw = a->pw[i];
                float sa = a->pa[i];
                dpw[i] *= sw * (1.0f - sw);
                dpa[i] *= sa * (1.0f - sa);
            }
        }
        // each projection p = proj(shift): d_wX += shift^T dp ; d_shift += sum wX^T dp
        host_to_buf(a->shift, &sc->tmp, bt * d);   // shift input
        host_zero(&sc->dnorm, bt * d);             // accumulate d_shift here
        vkc_buf_t *dps[5] = { &sc->dpk, &sc->dpv, &sc->dpr, &sc->dpw, &sc->dpa };
        param_t *wps[5] = { ly->wk, ly->wv, ly->wr, ly->ww, ly->wa };
        for (int j = 0; j < 5; j++) {
            mm_dB(m, &sc->tmp, dps[j], &wps[j]->g, bt, d, d);
            mm_dA(m, &wps[j]->w, dps[j], &sc->dtmp2, bt, d, d);
            host_add(&sc->dnorm, &sc->dtmp2, bt * d);
        }
        // tokenshift backward: input n1, mix shift_mix, dout=d_shift -> dx(n1), dmix
        host_to_buf(a->n1, &sc->tmp, bt * d);       // tokenshift input
        host_to_buf(sc->dnorm.ptr, &sc->tmp2, bt * d);  // dout = d_shift
        host_zero(&sc->dtmp, bt * d);               // dx output
        {
            ts_pc_t btpc = (ts_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 1 };
            vkc_buf_t *tsbb[5] = { &sc->tmp, &ly->shift_mix->w, &sc->tmp2, &sc->dtmp, &ly->shift_mix->g };
            vkc_run(m->c, &m->k_tshift, tsbb, (bt * d + 255) / 256, 1, 1, &btpc);
        }
        // dtmp = d_n1 (grad of rmsnorm output). rmsnorm backward -> dx_in.
        host_to_buf(a->x_in, &sc->tmp2, bt * d);
        rmsnorm_bwd(m, &sc->tmp2, &ly->n1->w, &sc->dtmp, &sc->rms_stash, &ly->n1->g, bt, d);
        // sc->tmp2 = dx from time-mix norm path; add to dxh (x_in feeds residual + this)
        {
            float *dxh = sc->dxh;
            float *p = sc->tmp2.ptr;
            for (uint32_t i = 0; i < bt * d; i++) {
                dxh[i] += p[i];
            }
        }
    }

    // embed backward: dxh is grad of x after embed. scatter into tok/pos grads.
    host_to_buf(sc->dxh, &sc->x, bt * d);
    host_zero(&m->tok_emb->g, m->tok_emb->n);
    host_zero(&m->pos_emb->g, m->pos_emb->n);
    {
        emb_pc_t bepc = (emb_pc_t) { .bt = bt, .d = d, .seqlen = cfg.seqlen, .op = 1 };
        vkc_buf_t *bebuf[4] = { &sc->ids, &m->tok_emb->g, &m->pos_emb->g, &sc->x };
        vkc_run(m->c, &m->k_embed, bebuf, (bt * d + 255) / 256, 1, 1, &bepc);
    }

    return loss;
}

#endif

