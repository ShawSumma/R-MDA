// fwd.h -- forward pass of the RWKV-7 diffusion denoiser. Returns total loss.
// Activations are kept in a scratch arena so the backward (later) can reuse them.
#ifndef FWD_H
#define FWD_H

#include "model.h"

// reshape helpers for k,v,r,w,a into [b, head, t, dh] layout expected by wkv7.
// Our projections output [bt, d] = [b*t, head*dh]; the wkv kernel wants, per
// (b, head): contiguous [t, dh]. We store the projected buffers already in
// [b, head, t, dh] order by doing the layout in a small repack kernel... but to
// avoid yet another kernel, we choose the projection output layout to match:
// the matmul writes [bt, d]; element (b,t,h,dh) lives at (b*seqlen+t)*d + h*dh+e.
// wkv7 indexes base=(b*head+h)*t*dh + ti*dh + e. These differ, so we repack on
// the host (small, fp32, fine for now). Done in wkv_pack/unpack below.

typedef struct fwd_scratch fwd_scratch_t;

struct fwd_scratch {
    // [bt, d] running residual stream
    vkc_buf_t x;
    // [b, d] timestep conditioning
    vkc_buf_t cond;
    // [bt, d] scratch
    vkc_buf_t tmp;
    // [bt, d] scratch
    vkc_buf_t tmp2;
    // [bt, hidden] elem stash / ffn scratch
    vkc_buf_t pre;
    // [bt, hidden] ffn hidden
    vkc_buf_t hbuf;
    // [bt] rms values
    vkc_buf_t rms_stash;
    // [d] unused grad sink for rmsnorm fwd
    vkc_buf_t dg_dummy;
    // wkv per-(b,head,t,dh) packed buffers
    vkc_buf_t pk;
    vkc_buf_t pv;
    vkc_buf_t pr;
    vkc_buf_t pw;
    vkc_buf_t pa;
    // wkv output packed (forward direction)
    vkc_buf_t po;
    // wkv output packed (backward direction, bidirectional)
    vkc_buf_t po2;
    // saved states for bwd (forward pass)
    vkc_buf_t sstate;
    // saved states for bwd (backward pass)
    vkc_buf_t sstate2;
    // logits + loss
    // [bt, vocab]
    vkc_buf_t logits;
    // [bt, vocab] pre-CE snapshot (for inference)
    vkc_buf_t logits_save;
    // [bt]
    vkc_buf_t loss;
    // batch inputs (as gpu buffers)
    vkc_buf_t ids;
    vkc_buf_t target;
    vkc_buf_t mask;
    vkc_buf_t weight;
    // backward scratch
    // [bt, d]
    vkc_buf_t dx_head;
    // [bt, d]
    vkc_buf_t dnorm;
    // [bt, d]
    vkc_buf_t dtmp;
    // [bt, d]
    vkc_buf_t dtmp2;
    // [bt, hid]
    vkc_buf_t dh;
    // [bt, hid]
    vkc_buf_t dh2;
    vkc_buf_t dpk;
    vkc_buf_t dpv;
    vkc_buf_t dpr;
    vkc_buf_t dpw;
    vkc_buf_t dpa;
    // second-direction WKV grads (bidirectional)
    vkc_buf_t dpk2;
    vkc_buf_t dpv2;
    vkc_buf_t dpr2;
    vkc_buf_t dpw2;
    vkc_buf_t dpa2;
    // [bt, d]
    vkc_buf_t dwkv;
    // host: grad of residual stream (legacy)
    float *dxh;
    // [bt, d] grad of residual stream, kept on GPU
    vkc_buf_t dxh_buf;
    // adaLN timestep conditioning scratch
    // [batch, 6d] projection of cond
    vkc_buf_t ada6;
    // [batch, 6d] its gradient
    vkc_buf_t dada6;
    // [batch, d] grad of cond
    vkc_buf_t dcond;
    // [batch, d] extracted scale slice
    vkc_buf_t ada_scale;
    // [batch, d] extracted shift slice
    vkc_buf_t ada_shift;
    // fp16 cast of a matmul's activation input, sized to the largest (bt * hidden).
    vkc_buf_t a16;
};

static fwd_scratch_t fwd_alloc(model_t *m) {
    mcfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    vkc_t *c = m->c;
    return (fwd_scratch_t) {
        .x = vkc_alloc(c, bt * d * sizeof(float)),
        .cond = vkc_alloc(c, cfg.batch * d * sizeof(float)),
        .tmp = vkc_alloc(c, bt * d * sizeof(float)),
        .tmp2 = vkc_alloc(c, bt * d * sizeof(float)),
        .pre = vkc_alloc(c, bt * hid * sizeof(float)),
        .hbuf = vkc_alloc(c, bt * hid * sizeof(float)),
        .rms_stash = vkc_alloc(c, bt * sizeof(float)),
        .dg_dummy = vkc_alloc(c, d * sizeof(float)),
        .pk = vkc_alloc(c, bt * d * sizeof(float)),
        .pv = vkc_alloc(c, bt * d * sizeof(float)),
        .pr = vkc_alloc(c, bt * d * sizeof(float)),
        .pw = vkc_alloc(c, bt * d * sizeof(float)),
        .pa = vkc_alloc(c, bt * d * sizeof(float)),
        .po = vkc_alloc(c, bt * d * sizeof(float)),
        .po2 = vkc_alloc(c, bt * d * sizeof(float)),
        .sstate = vkc_alloc(c, cfg.batch * cfg.n_head * cfg.seqlen
            * (d / cfg.n_head) * (d / cfg.n_head) * sizeof(float)),
        .sstate2 = vkc_alloc(c, cfg.batch * cfg.n_head * cfg.seqlen
            * (d / cfg.n_head) * (d / cfg.n_head) * sizeof(float)),
        .logits = vkc_alloc(c, bt * cfg.vocab * sizeof(float)),
        .logits_save = vkc_alloc(c, bt * cfg.vocab * sizeof(float)),
        .loss = vkc_alloc(c, bt * sizeof(float)),
        .ids = vkc_alloc(c, bt * sizeof(uint32_t)),
        .target = vkc_alloc(c, bt * sizeof(uint32_t)),
        .mask = vkc_alloc(c, bt * sizeof(float)),
        .weight = vkc_alloc(c, bt * sizeof(float)),
        .dx_head = vkc_alloc(c, bt * d * sizeof(float)),
        .dnorm = vkc_alloc(c, bt * d * sizeof(float)),
        .dtmp = vkc_alloc(c, bt * d * sizeof(float)),
        .dtmp2 = vkc_alloc(c, bt * d * sizeof(float)),
        .dh = vkc_alloc(c, bt * hid * sizeof(float)),
        .dh2 = vkc_alloc(c, bt * hid * sizeof(float)),
        .dpk = vkc_alloc(c, bt * d * sizeof(float)),
        .dpv = vkc_alloc(c, bt * d * sizeof(float)),
        .dpr = vkc_alloc(c, bt * d * sizeof(float)),
        .dpw = vkc_alloc(c, bt * d * sizeof(float)),
        .dpa = vkc_alloc(c, bt * d * sizeof(float)),
        .dpk2 = vkc_alloc(c, bt * d * sizeof(float)),
        .dpv2 = vkc_alloc(c, bt * d * sizeof(float)),
        .dpr2 = vkc_alloc(c, bt * d * sizeof(float)),
        .dpw2 = vkc_alloc(c, bt * d * sizeof(float)),
        .dpa2 = vkc_alloc(c, bt * d * sizeof(float)),
        .dwkv = vkc_alloc(c, bt * d * sizeof(float)),
        .dxh = malloc(bt * d * sizeof(float)),
        .dxh_buf = vkc_alloc(c, bt * d * sizeof(float)),
        .ada6 = vkc_alloc(c, cfg.batch * 6 * d * sizeof(float)),
        .dada6 = vkc_alloc(c, cfg.batch * 6 * d * sizeof(float)),
        .dcond = vkc_alloc(c, cfg.batch * d * sizeof(float)),
        .ada_scale = vkc_alloc(c, cfg.batch * d * sizeof(float)),
        .ada_shift = vkc_alloc(c, cfg.batch * d * sizeof(float)),
        .a16 = vkc_alloc(c, bt * cfg.ffn_hidden * sizeof(uint16_t)),
    };
}

// push-constant structs matching each shader
typedef struct mm_pc mm_pc_t;
struct mm_pc {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};
typedef struct cast_pc cast_pc_t;
struct cast_pc {
    uint32_t n;
    uint32_t op;
};
typedef struct normheads_pc normheads_pc_t;
struct normheads_pc {
    uint32_t n_vec;
    uint32_t dh;
    uint32_t op;
};
typedef struct emb_pc emb_pc_t;
struct emb_pc {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
};
typedef struct ts_pc ts_pc_t;
struct ts_pc {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
};
typedef struct wkv_pc wkv_pc_t;
struct wkv_pc {
    uint32_t bsz;
    uint32_t hd;
    uint32_t t;
    uint32_t dh;
    uint32_t dir;
    uint32_t save;
};
typedef struct elem_pc elem_pc_t;
struct elem_pc {
    uint32_t n;
    uint32_t op;
};
typedef struct rms_pc rms_pc_t;
struct rms_pc {
    uint32_t rows;
    uint32_t d;
    uint32_t op;
};
typedef struct ce_pc ce_pc_t;
struct ce_pc {
    uint32_t rows;
    uint32_t vocab;
};
typedef struct adaln_pc adaln_pc_t;
struct adaln_pc {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
};

typedef struct vec_pc vec_pc_t;
struct vec_pc {
    uint32_t n;
    uint32_t op;
};

// GPU elementwise: op 0 -> a += b, op 1 -> a *= b. Replaces host loops over
// mapped GPU memory (which are ~50000x slower than this dispatch).
static void vec_op(model_t *m, vkc_buf_t *a, vkc_buf_t *b, uint32_t n, uint32_t op) {
    vec_pc_t pc = (vec_pc_t) {
        .n = n,
        .op = op,
    };
    vkc_buf_t *bufs[2] = {
        a,
        b,
    };
    vkc_run(m->c, &m->k_vec, bufs, (n + 255) / 256, 1, 1, &pc);
}

typedef struct repack_pc repack_pc_t;
struct repack_pc {
    uint32_t bt;
    uint32_t hd;
    uint32_t dh;
    uint32_t seqlen;
    uint32_t op;
};

// GPU head repack: op 0 pack [b,t,h,dh]->[b,h,t,dh], op 1 unpack. src != dst.
static void gpu_repack(model_t *m, vkc_buf_t *src, vkc_buf_t *dst,
    uint32_t bt, uint32_t hd, uint32_t dh, uint32_t seqlen, uint32_t op) {
    repack_pc_t pc = (repack_pc_t) {
        .bt = bt,
        .hd = hd,
        .dh = dh,
        .seqlen = seqlen,
        .op = op,
    };
    vkc_buf_t *bufs[2] = {
        src,
        dst,
    };
    vkc_run(m->c, &m->k_repack, bufs, (bt * hd * dh + 255) / 256, 1, 1, &pc);
}

static void mm(model_t *m, vkc_buf_t *a, vkc_buf_t *b, vkc_buf_t *out,
    uint32_t rows, uint32_t n, uint32_t k) {
    mm_pc_t pc = (mm_pc_t) {
        .m = rows,
        .n = n,
        .k = k,
    };
    vkc_buf_t *bufs[3] = {
        a,
        b,
        out,
    };
    vkc_run(m->c, &m->k_mm, bufs, (n + 15) / 16, (rows + 15) / 16, 1, &pc);
}

// cast n fp32 values in src to fp16 in dst (op 0) or back (op 1).
static void cast_buf(model_t *m, vkc_buf_t *src, vkc_buf_t *dst, uint32_t n, uint32_t op) {
    cast_pc_t pc = (cast_pc_t) {
        .n = n,
        .op = op,
    };
    vkc_buf_t *bufs[4];
    if (op == 0) {
        bufs[0] = src;
        bufs[1] = dst;
        bufs[2] = dst;
        bufs[3] = src;
    } else {
        bufs[0] = dst;
        bufs[1] = src;
        bufs[2] = src;
        bufs[3] = dst;
    }
    vkc_run(m->c, &m->k_cast, bufs, (n + 255) / 256, 1, 1, &pc);
}

// refresh every weight's fp16 mirror from its fp32 master, but only when the fp32
// weights have changed since the last refresh (the dirty16 flag). Batched into one
// submission. Inference casts once; training marks dirty each optimizer step.
static void refresh_w16(model_t *m) {
    if (m->dirty16 == 0) {
        return;
    }
    vkc_begin(m->c);
    for (uint32_t i = 0; i < m->n_params; i = i + 1) {
        param_t *p = &m->params[i];
        cast_buf(m, &p->w, &p->w16, p->n, 0);
    }
    vkc_flush(m->c);
    m->dirty16 = 0;
}

// fp16 matmul: out[rows, n] = a[rows, k] @ p->w16[k, n], using the matrix cores.
// The activation a (fp32) is cast into the shared a16 scratch first; the weight is
// already mirrored in p->w16. Requires rows, n, k to be multiples of 16.
static void mm16(model_t *m, fwd_scratch_t *sc, vkc_buf_t *a, param_t *p,
    vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k) {
    cast_buf(m, a, &sc->a16, rows * k, 0);
    mm_pc_t pc = (mm_pc_t) {
        .m = rows,
        .n = n,
        .k = k,
    };
    vkc_buf_t *bufs[3] = {
        &sc->a16,
        &p->w16,
        out,
    };
    vkc_run(m->c, &m->k_mm16, bufs, n / 16, rows / 16, 1, &pc);
}

// pack [bt, d] (laid out b,t,head,dh) -> [b, head, t, dh]. The innermost dh run
// is contiguous in both layouts, so copy it as a block (mapped GPU memory is
// write-combined; block memcpy is far faster than element-wise access).
static void pack_heads(model_t *m, vkc_buf_t *src, vkc_buf_t *dst) {
    mcfg_t cfg = m->cfg;
    uint32_t d = cfg.d;
    uint32_t hd = cfg.n_head;
    uint32_t dh = d / hd;
    float *s = src->ptr;
    float *o = dst->ptr;
    for (uint32_t b = 0; b < cfg.batch; b = b + 1) {
        for (uint32_t h = 0; h < hd; h = h + 1) {
            for (uint32_t t = 0; t < cfg.seqlen; t = t + 1) {
                uint32_t src_i = (b * cfg.seqlen + t) * d + h * dh;
                uint32_t dst_i = ((b * hd + h) * cfg.seqlen + t) * dh;
                memcpy(o + dst_i, s + src_i, dh * sizeof(float));
            }
        }
    }
}

static void unpack_heads(model_t *m, vkc_buf_t *src, vkc_buf_t *dst) {
    mcfg_t cfg = m->cfg;
    uint32_t d = cfg.d;
    uint32_t hd = cfg.n_head;
    uint32_t dh = d / hd;
    float *s = src->ptr;
    float *o = dst->ptr;
    for (uint32_t b = 0; b < cfg.batch; b = b + 1) {
        for (uint32_t h = 0; h < hd; h = h + 1) {
            for (uint32_t t = 0; t < cfg.seqlen; t = t + 1) {
                uint32_t dst_i = (b * cfg.seqlen + t) * d + h * dh;
                uint32_t src_i = ((b * hd + h) * cfg.seqlen + t) * dh;
                memcpy(o + dst_i, s + src_i, dh * sizeof(float));
            }
        }
    }
}

// sigmoid/clamp a buffer in place on the host (for w in (0,1), a in (0,1))
static void host_sigmoid(vkc_buf_t *b, uint32_t n) {
    float *p = b->ptr;
    for (uint32_t i = 0; i < n; i = i + 1) {
        p[i] = 1.0f / (1.0f + expf(-p[i]));
    }
}

// L2-normalize each per-head dh-vector of a packed [b, head, t, dh] buffer.
// RWKV-7's delta rule needs a unit-norm removal key so a*k*k^T is a bounded
// projection (eigenvalues in [0, a]); without it the state recurrence diverges.
static void host_norm_heads(vkc_buf_t *buf, uint32_t n_vec, uint32_t dh) {
    float *p = buf->ptr;
    for (uint32_t v = 0; v < n_vec; v = v + 1) {
        float *vec = p + v * dh;
        float ss = 0.0f;
        for (uint32_t e = 0; e < dh; e = e + 1) {
            ss += vec[e] * vec[e];
        }
        float inv = 1.0f / (sqrtf(ss) + 1e-6f);
        for (uint32_t e = 0; e < dh; e = e + 1) {
            vec[e] *= inv;
        }
    }
}

// same, but also store each per-head vector's 1/||.|| into inv_norm[v]. The
// backward of k_hat = k/||k|| needs this scale: dk = (dk_hat - k_hat*(k_hat.dk_hat))/||k||.
static void host_norm_heads_save(float *p, float *inv_norm, uint32_t n_vec, uint32_t dh) {
    for (uint32_t v = 0; v < n_vec; v = v + 1) {
        float *vec = p + v * dh;
        float ss = 0.0f;
        for (uint32_t e = 0; e < dh; e = e + 1) {
            ss += vec[e] * vec[e];
        }
        float inv = 1.0f / (sqrtf(ss) + 1e-6f);
        inv_norm[v] = inv;
        for (uint32_t e = 0; e < dh; e = e + 1) {
            vec[e] *= inv;
        }
    }
}

// GPU per-head L2 normalization of `buf` (n_vec vectors of length dh), in place.
// op 1 also writes 1/||.|| per vector into inv; op 0 leaves inv untouched (pass any
// bound buffer). One workgroup per vector.
static void norm_heads(model_t *m, vkc_buf_t *buf, vkc_buf_t *inv,
    uint32_t n_vec, uint32_t dh, uint32_t op) {
    normheads_pc_t pc = (normheads_pc_t) {
        .n_vec = n_vec,
        .dh = dh,
        .op = op,
    };
    vkc_buf_t *bufs[2] = {
        buf,
        inv,
    };
    vkc_run(m->c, &m->k_normheads, bufs, n_vec, 1, 1, &pc);
}

// rmsnorm forward writing into `out`
static void rmsnorm_fwd(model_t *m, vkc_buf_t *x, vkc_buf_t *g, vkc_buf_t *out,
    vkc_buf_t *stash, uint32_t rows, uint32_t d) {
    rms_pc_t pc = (rms_pc_t) {
        .rows = rows,
        .d = d,
        .op = 0,
    };
    vkc_buf_t *bufs[5] = {
        x,
        g,
        out,
        stash,
        // dg sink, unused on fwd
        &m->layers[0].n1->w,
    };
    vkc_run(m->c, &m->k_rms, bufs, rows, 1, 1, &pc);
}

// returns the mean loss over masked tokens (host reduction)

static float model_forward(model_t *m, fwd_scratch_t *sc) {
    mcfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hd = cfg.n_head;
    uint32_t dh = d / hd;

    // mirror the current fp32 weights into their fp16 copies for the matrix cores.
    refresh_w16(m);

    // record the whole forward as one GPU batch (submitted once at the CE reduce).
    vkc_begin(m->c);

    // 1. embed: x = tok_emb[ids] + pos_emb
    emb_pc_t epc = (emb_pc_t) {
        .bt = bt,
        .d = d,
        .seqlen = cfg.seqlen,
        .op = 0,
    };
    vkc_buf_t *ebuf[4] = {
        &sc->ids,
        &m->tok_emb->w,
        &m->pos_emb->w,
        &sc->x,
    };
    vkc_run(m->c, &m->k_embed, ebuf, (bt * d + 255) / 256, 1, 1, &epc);

    // (timestep conditioning is folded into adaLN-zero which starts at identity;
    //  for the first training milestone cond is left at zero -> blocks act as
    //  plain pre-norm RWKV. The adaLN path is wired but cond=0 means no-op.)

    // 2. blocks
    for (uint32_t l = 0; l < cfg.n_layer; l = l + 1) {
        mlayer_t *ly = &m->layers[l];

        // --- time mixing ---
        // h = rmsnorm(x, n1) -> tmp
        rmsnorm_fwd(m, &sc->x, &ly->n1->w, &sc->tmp, &sc->rms_stash, bt, d);
        // token-shift tmp -> tmp2
        ts_pc_t tpc = (ts_pc_t) {
            .bt = bt,
            .d = d,
            .seqlen = cfg.seqlen,
            .op = 0,
        };
        vkc_buf_t *tsb[5] = {
            &sc->tmp,
            &ly->shift_mix->w,
            &sc->tmp2,
            &sc->pk,
            &sc->dg_dummy,
        };
        vkc_run(m->c, &m->k_tshift, tsb, (bt * d + 255) / 256, 1, 1, &tpc);

        // project to k,v,r,w,a from tmp2
        mm16(m, sc, &sc->tmp2, ly->wk, &sc->pk, bt, d, d);
        mm16(m, sc, &sc->tmp2, ly->wv, &sc->pv, bt, d, d);
        mm16(m, sc, &sc->tmp2, ly->wr, &sc->pr, bt, d, d);
        mm16(m, sc, &sc->tmp2, ly->ww, &sc->pw, bt, d, d);
        mm16(m, sc, &sc->tmp2, ly->wa, &sc->pa, bt, d, d);
        // w in (0,1), a in (0,1): sigmoid on the GPU (op 2), not a host loop.
        vec_op(m, &sc->pw, &sc->pw, bt * d, 2);
        vec_op(m, &sc->pa, &sc->pa, bt * d, 2);

        // pack k,v,r,w,a to [b,head,t,dh], run wkv7 forward, unpack output
        // (reuse tmp/tmp2 area? need distinct packed buffers -> pk..pa already are)
        // pack in place: we packed into pk..pa already? No -- they hold [bt,d].
        // repack each into itself via a temp. We'll pack into sc->tmp (reused).
        // For clarity, pack each buffer through host into the same buffer.
        // k:
        {
            // GPU repack (bit-identical to host pack_heads), copying back on the
            // GPU (op 5) instead of a host memcpy over mapped memory.
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
            // unit-norm the removal/replacement key per head on the GPU -> stable
            // delta rule (inv unused in forward; bind rms_stash as the slot).
            norm_heads(m, &sc->pk, &sc->rms_stash, cfg.batch * hd * cfg.seqlen, dh, 0);
        }
        wkv_pc_t wpc = (wkv_pc_t) {
            .bsz = cfg.batch,
            .hd = hd,
            .t = cfg.seqlen,
            .dh = dh,
            .dir = 0,
            .save = 0,
        };
        vkc_buf_t *wb[7] = {
            &sc->pk,
            &sc->pv,
            &sc->pr,
            &sc->pw,
            &sc->pa,
            &sc->po,
            &sc->sstate,
        };
        vkc_run(m->c, &m->k_wkv_f, wb, cfg.batch * hd, 1, 1, &wpc);
        // bidirectional: also run right-to-left and sum the outputs
        wkv_pc_t wpc2 = wpc;
        wpc2.dir = 1;
        vkc_buf_t *wb2[7] = {
            &sc->pk,
            &sc->pv,
            &sc->pr,
            &sc->pw,
            &sc->pa,
            &sc->po2,
            &sc->sstate2,
        };
        vkc_run(m->c, &m->k_wkv_f, wb2, cfg.batch * hd, 1, 1, &wpc2);
        // average the two directions
        vec_op(m, &sc->po, &sc->po2, bt * d, 3);
        // unpack po -> tmp2 ([bt, d])
        gpu_repack(m, &sc->po, &sc->tmp2, bt, hd, dh, cfg.seqlen, 1);
        // output projection wo, add residual on the GPU: x += wo(tmp2)
        mm16(m, sc, &sc->tmp2, ly->wo, &sc->tmp, bt, d, d);
        vec_op(m, &sc->x, &sc->tmp, bt * d, 0);

        // --- channel mixing (ffn) ---
        rmsnorm_fwd(m, &sc->x, &ly->n2->w, &sc->tmp, &sc->rms_stash, bt, d);
        mm16(m, sc, &sc->tmp, ly->w_gate, &sc->hbuf, bt, cfg.ffn_hidden, d);
        // silu(hbuf) in place
        elem_pc_t spc = (elem_pc_t) {
            .n = bt * cfg.ffn_hidden,
            .op = 0,
        };
        vkc_buf_t *seb[2] = {
            &sc->hbuf,
            &sc->pre,
        };
        vkc_run(m->c, &m->k_elem, seb, (bt * cfg.ffn_hidden + 255) / 256, 1, 1, &spc);
        // up projection, multiply (silu(gate) * up) on the GPU
        mm16(m, sc, &sc->tmp, ly->w_up, &sc->pre, bt, cfg.ffn_hidden, d);
        vec_op(m, &sc->hbuf, &sc->pre, bt * cfg.ffn_hidden, 1);
        // down projection -> tmp2, add residual on the GPU
        mm16(m, sc, &sc->hbuf, ly->w_down, &sc->tmp2, bt, d, cfg.ffn_hidden);
        vec_op(m, &sc->x, &sc->tmp2, bt * d, 0);
    }

    // 3. head: logits = head(rmsnorm(x))   (reuse n1[0] as a final gain? use ones)
    rmsnorm_fwd(m, &sc->x, &m->layers[0].n1->w, &sc->tmp, &sc->rms_stash, bt, d);
    mm16(m, sc, &sc->tmp, m->head, &sc->logits, bt, m->cfg.vocab, d);

    // snapshot logits before CE overwrites them with dlogits (for inference),
    // copied on the GPU instead of a large host memcpy over mapped memory.
    vec_op(m, &sc->logits_save, &sc->logits, bt * m->cfg.vocab, 5);

    // 4. masked CE loss (also writes dlogits into logits)
    ce_pc_t cpc = (ce_pc_t) {
        .rows = bt,
        .vocab = m->cfg.vocab,
    };
    vkc_buf_t *cb[5] = {
        &sc->logits,
        &sc->target,
        &sc->mask,
        &sc->weight,
        &sc->loss,
    };
    vkc_run(m->c, &m->k_ce, cb, bt, 1, 1, &cpc);

    // submit the whole recorded forward and wait, so the host can read the loss.
    vkc_flush(m->c);

    // reduce loss over masked tokens
    float *lp = sc->loss.ptr;
    float *mk = sc->mask.ptr;
    double total = 0.0;
    double cnt = 0.0;
    for (uint32_t i = 0; i < bt; i = i + 1) {
        total += lp[i];
        cnt += mk[i];
    }
    return (float)(total / (cnt + 1e-6));
}

#endif
