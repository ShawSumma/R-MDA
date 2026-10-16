// fwd.h -- forward pass of the RWKV-7 diffusion denoiser. One forward (forward_rec)
// serves both training and inference, so both always run the same network; training
// saves each layer's activations into an act_t for the backward.
#ifndef FWD_H
#define FWD_H

#include "model.h"

// The projections output [bt, d] (element (b,t,h,e) at (b*seqlen+t)*d + h*dh + e) but
// the wkv kernels want per (b, head) contiguous [t, dh] (at ((b*hd+h)*seqlen+t)*dh + e).
// The repack kernel converts between the two; "packed" below means the wkv layout.

// per-layer activations saved by the forward for the backward.
typedef struct act_t act_t;

struct act_t {
    // [bt, d] residual entering the layer
    vkc_buf_t x_in;
    // [bt, d] rmsnorm(x_in), before adaLN
    vkc_buf_t n1_pre;
    // [bt] rms of x_in
    vkc_buf_t rms1;
    // [bt, d] adaLN(n1_pre): the token-shift input
    vkc_buf_t n1;
    // [bt, d] tokenshift(n1): the k/v/r/w/a projection input
    vkc_buf_t shift;
    // packed unit-norm k
    vkc_buf_t pk;
    vkc_buf_t pv;
    vkc_buf_t pr;
    // packed, sigmoid'd
    vkc_buf_t pw;
    // packed, sigmoid'd
    vkc_buf_t pa;
    // [bt * hd] 1/||k|| per packed head vector
    vkc_buf_t kinv;
    // wkv pre-step states, left-to-right pass
    vkc_buf_t sstate;
    // wkv pre-step states, right-to-left pass
    vkc_buf_t sstate2;
    // [bt, d] unpacked wkv output (the wo input)
    vkc_buf_t wkv;
    // [bt, d] residual after time mixing
    vkc_buf_t x_mid;
    // [bt, d] rmsnorm(x_mid), before adaLN
    vkc_buf_t n2_pre;
    // [bt] rms of x_mid
    vkc_buf_t rms2;
    // [bt, d] adaLN(n2_pre): the ffn input
    vkc_buf_t n2;
    // [bt, hid] pre-silu gate
    vkc_buf_t gate_pre;
    // [bt, hid] silu(gate_pre)
    vkc_buf_t gate;
    // [bt, hid] up projection
    vkc_buf_t up;
};

// rows of cond and ada6: batch rounded up to the fp16 matmul's 64-row block.
static uint32_t cond_rows(mcfg_t cfg) {
    return (cfg.batch + 63) / 64 * 64;
}

typedef struct fwd_scratch_t fwd_scratch_t;

struct fwd_scratch_t {
    // [bt, d] running residual stream
    vkc_buf_t x;
    // [batch, d] sinusoidal embedding of the mask rate t
    vkc_buf_t temb;
    // [cond_rows, d] timestep conditioning, temb @ t_proj (rows >= batch unused)
    vkc_buf_t cond;
    // per layer [cond_rows, 6d] adaLN modulation, cond @ ada (rows >= batch unused)
    vkc_buf_t *ada6;
    // [bt, d] scratch
    vkc_buf_t tmp;
    // [bt, d] scratch
    vkc_buf_t tmp2;
    // [bt, hidden] ffn scratch
    vkc_buf_t pre;
    // [bt, hidden] ffn hidden
    vkc_buf_t hbuf;
    // [bt] rms of the final norm
    vkc_buf_t rms_stash;
    // [d] unused grad sink for tokenshift fwd
    vkc_buf_t dg_dummy;
    // [bt, d] unpacked k,v,r,w,a projections (and their grads in the backward)
    vkc_buf_t pk;
    vkc_buf_t pv;
    vkc_buf_t pr;
    vkc_buf_t pw;
    vkc_buf_t pa;
    // wkv output packed (forward direction)
    vkc_buf_t po;
    // wkv output packed (backward direction, bidirectional)
    vkc_buf_t po2;
    // [bt, d] rmsnorm(x_final, n_out): the head input
    vkc_buf_t xn;
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
    // [bt, d] packed grad of the wkv output
    vkc_buf_t dwkv;
    // [bt, d] grad of residual stream
    vkc_buf_t dxh_buf;
    // [batch, 6d] grad of a block's ada6
    vkc_buf_t dada6;
    // [batch, d] grad of cond, summed over blocks
    vkc_buf_t dcond;
    // fp16 cast of a matmul's activation input, sized to the largest (bt * hidden).
    vkc_buf_t a16;
    // saved activations: n_layer of them for training, 1 (reused by every layer) for
    // inference.
    act_t *acts;
    uint32_t n_acts;
};

static act_t act_alloc(model_t *m) {
    mcfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    uint32_t dh = d / cfg.n_head;
    uint32_t ns = cfg.batch * cfg.n_head * cfg.seqlen * dh * dh;
    vkc_t *c = m->c;
    return (act_t) {
        .x_in = vkc_alloc(c, bt * d * sizeof(float)),
        .n1_pre = vkc_alloc(c, bt * d * sizeof(float)),
        .rms1 = vkc_alloc(c, bt * sizeof(float)),
        .n1 = vkc_alloc(c, bt * d * sizeof(float)),
        .shift = vkc_alloc(c, bt * d * sizeof(float)),
        .pk = vkc_alloc(c, bt * d * sizeof(float)),
        .pv = vkc_alloc(c, bt * d * sizeof(float)),
        .pr = vkc_alloc(c, bt * d * sizeof(float)),
        .pw = vkc_alloc(c, bt * d * sizeof(float)),
        .pa = vkc_alloc(c, bt * d * sizeof(float)),
        .kinv = vkc_alloc(c, bt * cfg.n_head * sizeof(float)),
        .sstate = vkc_alloc(c, ns * sizeof(float)),
        .sstate2 = vkc_alloc(c, ns * sizeof(float)),
        .wkv = vkc_alloc(c, bt * d * sizeof(float)),
        .x_mid = vkc_alloc(c, bt * d * sizeof(float)),
        .n2_pre = vkc_alloc(c, bt * d * sizeof(float)),
        .rms2 = vkc_alloc(c, bt * sizeof(float)),
        .n2 = vkc_alloc(c, bt * d * sizeof(float)),
        .gate_pre = vkc_alloc(c, bt * hid * sizeof(float)),
        .gate = vkc_alloc(c, bt * hid * sizeof(float)),
        .up = vkc_alloc(c, bt * hid * sizeof(float)),
    };
}

static fwd_scratch_t fwd_alloc(model_t *m, uint32_t n_acts) {
    mcfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hid = cfg.ffn_hidden;
    vkc_t *c = m->c;
    uint32_t crow = cond_rows(cfg);
    act_t *acts = malloc(n_acts * sizeof(act_t));
    for (uint32_t i = 0; i < n_acts; ++i) {
        acts[i] = act_alloc(m);
    }
    vkc_buf_t *ada6 = malloc(cfg.n_layer * sizeof(vkc_buf_t));
    for (uint32_t l = 0; l < cfg.n_layer; ++l) {
        ada6[l] = vkc_alloc(c, crow * 6 * d * sizeof(float));
    }
    return (fwd_scratch_t) {
        .x = vkc_alloc(c, bt * d * sizeof(float)),
        .temb = vkc_alloc(c, cfg.batch * d * sizeof(float)),
        .cond = vkc_alloc(c, crow * d * sizeof(float)),
        .ada6 = ada6,
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
        .xn = vkc_alloc(c, bt * d * sizeof(float)),
        .logits = vkc_alloc(c, bt * cfg.vocab * sizeof(float)),
        .logits_save = vkc_alloc(c, bt * cfg.vocab * sizeof(float)),
        .loss = vkc_alloc(c, bt * sizeof(float)),
        .ids = vkc_alloc(c, bt * sizeof(uint32_t)),
        .target = vkc_alloc(c, bt * sizeof(uint32_t)),
        .mask = vkc_alloc(c, bt * sizeof(float)),
        .weight = vkc_alloc(c, bt * sizeof(float)),
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
        .dxh_buf = vkc_alloc(c, bt * d * sizeof(float)),
        .dada6 = vkc_alloc(c, cfg.batch * 6 * d * sizeof(float)),
        .dcond = vkc_alloc(c, cfg.batch * d * sizeof(float)),
        .a16 = vkc_alloc(c, bt * cfg.ffn_hidden * sizeof(uint16_t)),
        .acts = acts,
        .n_acts = n_acts,
    };
}

// push-constant structs matching each shader
typedef struct mm_pc_t mm_pc_t;
struct mm_pc_t {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};
typedef struct cast_pc_t cast_pc_t;
struct cast_pc_t {
    uint32_t n;
    uint32_t op;
};
typedef struct normheads_pc_t normheads_pc_t;
struct normheads_pc_t {
    uint32_t n_vec;
    uint32_t dh;
    uint32_t op;
};
typedef struct emb_pc_t emb_pc_t;
struct emb_pc_t {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
};
typedef struct temb_pc_t temb_pc_t;
struct temb_pc_t {
    uint32_t batch;
    uint32_t d;
    uint32_t seqlen;
};
typedef struct ts_pc_t ts_pc_t;
struct ts_pc_t {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
};
typedef struct wkv_pc_t wkv_pc_t;
struct wkv_pc_t {
    uint32_t bsz;
    uint32_t hd;
    uint32_t t;
    uint32_t dh;
    uint32_t dir;
    uint32_t save;
};
// wkv backward push constant: same fields as the forward minus `save`.
typedef struct wkv_bwd_pc_t wkv_bwd_pc_t;
struct wkv_bwd_pc_t {
    uint32_t bsz;
    uint32_t hd;
    uint32_t t;
    uint32_t dh;
    uint32_t dir;
};
typedef struct elem_pc_t elem_pc_t;
struct elem_pc_t {
    uint32_t n;
    uint32_t op;
};
typedef struct rms_pc_t rms_pc_t;
struct rms_pc_t {
    uint32_t rows;
    uint32_t d;
    uint32_t op;
};
typedef struct ce_pc_t ce_pc_t;
struct ce_pc_t {
    uint32_t rows;
    uint32_t vocab;
};
// argmax kernel push constant: per-row top-1 over the vocab logits.
typedef struct argmax_pc_t argmax_pc_t;
struct argmax_pc_t {
    uint32_t rows;
    uint32_t vocab;
};
typedef struct adaln_pc_t adaln_pc_t;
struct adaln_pc_t {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
    // row stride of the modulation buffer and the scale / shift column offsets
    uint32_t stride;
    uint32_t s_off;
    uint32_t sh_off;
};

typedef struct vec_pc_t vec_pc_t;
struct vec_pc_t {
    uint32_t n;
    uint32_t op;
    float scale;
};

// GPU elementwise: op 0 -> a += b, op 1 -> a *= b. Replaces host loops over
// mapped GPU memory (which are ~50000x slower than this dispatch).
static void vec_op(model_t *m, vkc_buf_t *a, vkc_buf_t *b, uint32_t n, uint32_t op) {
    vec_pc_t pc = (vec_pc_t) {
        .n = n,
        .op = op,
        .scale = 0.0f,
    };
    vkc_buf_t *bufs[2] = {
        a,
        b,
    };
    vkc_run(m->c, &m->k_vec, bufs, (n + 255) / 256, 1, 1, &pc);
}

// scalar multiply a *= scale on the GPU (op 7; b unused).
static void vec_scale(model_t *m, vkc_buf_t *a, uint32_t n, float scale) {
    vec_pc_t pc = (vec_pc_t) {
        .n = n,
        .op = 7,
        .scale = scale,
    };
    vkc_buf_t *bufs[2] = { a, a };
    vkc_run(m->c, &m->k_vec, bufs, (n + 255) / 256, 1, 1, &pc);
}

typedef struct repack_pc_t repack_pc_t;
struct repack_pc_t {
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
    for (uint32_t i = 0; i < m->n_params; ++i) {
        param_t *p = &m->params[i];
        cast_buf(m, &p->w, &p->w16, p->n, 0);
    }
    vkc_flush(m->c);
    m->dirty16 = 0;
}

// fp16 matmul out[rows, n] = a[rows, k] @ p->w16[k, n] on the matrix cores, from the
// fp32 activation a that the caller already cast into sc->a16. Requires rows, n
// multiples of 64 and k of 16 (mcfg_check enforces this for every model shape).
static void mm16_pre(model_t *m, fwd_scratch_t *sc, vkc_buf_t *a, param_t *p,
    vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k) {
    if (m->ref32) {
        mm(m, a, &p->w, out, rows, n, k);
        return;
    }
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
    // shape-dispatch: the wide 256x64 block maximizes reuse on large-n matmuls (the
    // vocab head), but starves on small per-layer matmuls (too few workgroups). Use it
    // only when n is large and rows is a multiple of 256; otherwise the 64x64 block.
    if (n >= 4096 && (rows % 256) == 0) {
        vkc_run(m->c, &m->k_mm16w, bufs, n / 64, rows / 256, 1, &pc);
    } else {
        vkc_run(m->c, &m->k_mm16, bufs, n / 64, rows / 64, 1, &pc);
    }
}

// fp16 matmul: casts the fp32 activation a into the shared a16 scratch, then runs.
// Use mm16_pre directly when several matmuls share one activation (cast once).
static void mm16(model_t *m, fwd_scratch_t *sc, vkc_buf_t *a, param_t *p,
    vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k) {
    cast_buf(m, a, &sc->a16, rows * k, 0);
    mm16_pre(m, sc, a, p, out, rows, n, k);
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
    for (uint32_t b = 0; b < cfg.batch; ++b) {
        for (uint32_t h = 0; h < hd; ++h) {
            for (uint32_t t = 0; t < cfg.seqlen; ++t) {
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
    for (uint32_t b = 0; b < cfg.batch; ++b) {
        for (uint32_t h = 0; h < hd; ++h) {
            for (uint32_t t = 0; t < cfg.seqlen; ++t) {
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
    for (uint32_t i = 0; i < n; ++i) {
        p[i] = 1.0f / (1.0f + expf(-p[i]));
    }
}

// L2-normalize each per-head dh-vector of a packed [b, head, t, dh] buffer.
// RWKV-7's delta rule needs a unit-norm removal key so a*k*k^T is a bounded
// projection (eigenvalues in [0, a]); without it the state recurrence diverges.
static void host_norm_heads(vkc_buf_t *buf, uint32_t n_vec, uint32_t dh) {
    float *p = buf->ptr;
    for (uint32_t v = 0; v < n_vec; ++v) {
        float *vec = p + v * dh;
        float ss = 0.0f;
        for (uint32_t e = 0; e < dh; ++e) {
            ss += vec[e] * vec[e];
        }
        float inv = 1.0f / (sqrtf(ss) + 1e-6f);
        for (uint32_t e = 0; e < dh; ++e) {
            vec[e] *= inv;
        }
    }
}

// same, but also store each per-head vector's 1/||.|| into inv_norm[v]. The
// backward of k_hat = k/||k|| needs this scale: dk = (dk_hat - k_hat*(k_hat.dk_hat))/||k||.
static void host_norm_heads_save(float *p, float *inv_norm, uint32_t n_vec, uint32_t dh) {
    for (uint32_t v = 0; v < n_vec; ++v) {
        float *vec = p + v * dh;
        float ss = 0.0f;
        for (uint32_t e = 0; e < dh; ++e) {
            ss += vec[e] * vec[e];
        }
        float inv = 1.0f / (sqrtf(ss) + 1e-6f);
        inv_norm[v] = inv;
        for (uint32_t e = 0; e < dh; ++e) {
            vec[e] *= inv;
        }
    }
}

// GPU per-head L2 normalization (n_vec vectors of length dh), one workgroup per vector.
// op 0: normalize buf in place and write 1/||.|| per vector into inv (khat unused,
// pass buf). op 1: backward -- buf holds dk_hat; khat and inv are the forward's
// unit vectors and scales.
static void norm_heads(model_t *m, vkc_buf_t *buf, vkc_buf_t *inv, vkc_buf_t *khat,
    uint32_t n_vec, uint32_t dh, uint32_t op) {
    normheads_pc_t pc = (normheads_pc_t) {
        .n_vec = n_vec,
        .dh = dh,
        .op = op,
    };
    vkc_buf_t *bufs[3] = {
        buf,
        inv,
        khat,
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

// adaLN forward: out = x * (1 + scale) + shift, with scale and shift read from slots
// s_slot and s_slot + 1 of the block's [batch, 6d] ada6.
static void adaln_fwd(model_t *m, vkc_buf_t *x, vkc_buf_t *ada6, vkc_buf_t *out,
    uint32_t s_slot) {
    mcfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    adaln_pc_t pc = (adaln_pc_t) {
        .bt = bt,
        .d = d,
        .seqlen = cfg.seqlen,
        .op = 0,
        .stride = 6 * d,
        .s_off = s_slot * d,
        .sh_off = (s_slot + 1) * d,
    };
    // dx / dada bindings are unused on fwd
    vkc_buf_t *bufs[5] = {
        x,
        ada6,
        out,
        out,
        ada6,
    };
    vkc_run(m->c, &m->k_adaln, bufs, (bt * d + 255) / 256, 1, 1, &pc);
}

// Record the whole forward into the open batch: embed, timestep conditioning, the
// blocks, then the head into sc->logits. With save set, layer l's activations land in
// sc->acts[l] for the backward; otherwise every layer reuses sc->acts[0].
static void forward_rec(model_t *m, fwd_scratch_t *sc, int save) {
    mcfg_t cfg = m->cfg;
    uint32_t bt = cfg.batch * cfg.seqlen;
    uint32_t d = cfg.d;
    uint32_t hd = cfg.n_head;
    uint32_t dh = d / hd;
    uint32_t hid = cfg.ffn_hidden;

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

    // 2. timestep conditioning: cond = temb(t) @ t_proj, with t recovered from the 1/t
    // weight. A denoiser needs the mask rate to adapt to "mostly masked" vs "lightly
    // masked" input; cond drives every block's adaLN.
    temb_pc_t tepc = (temb_pc_t) {
        .batch = cfg.batch,
        .d = d,
        .seqlen = cfg.seqlen,
    };
    vkc_buf_t *tebuf[2] = {
        &sc->weight,
        &sc->temb,
    };
    vkc_run(m->c, &m->k_temb, tebuf, (cfg.batch * d + 255) / 256, 1, 1, &tepc);
    mm(m, &sc->temb, &m->t_proj->w, &sc->cond, cfg.batch, d, d);
    // every block's modulation ada6[l] = cond @ ada_l [batch, 6d] (slots 0,1 time-mix
    // scale/shift, 2,3 ffn), up front from one fp16 cast of cond. cond's row count is
    // padded to the matmul block; the pad rows and their ada6 outputs are never read.
    uint32_t crow = cond_rows(cfg);
    cast_buf(m, &sc->cond, &sc->a16, crow * d, 0);
    for (uint32_t l = 0; l < cfg.n_layer; ++l) {
        mm16_pre(m, sc, &sc->cond, m->layers[l].ada, &sc->ada6[l], crow, 6 * d, d);
    }

    // 3. blocks
    for (uint32_t l = 0; l < cfg.n_layer; ++l) {
        mlayer_t *ly = &m->layers[l];
        act_t *a = &sc->acts[save ? l : 0];
        if (save) {
            vec_op(m, &a->x_in, &sc->x, bt * d, 5);
        }

        // time mixing: n1 = adaLN(rmsnorm(x, n1)), then token shift
        rmsnorm_fwd(m, &sc->x, &ly->n1->w, &a->n1_pre, &a->rms1, bt, d);
        adaln_fwd(m, &a->n1_pre, &sc->ada6[l], &a->n1, 0);
        ts_pc_t tpc = (ts_pc_t) {
            .bt = bt,
            .d = d,
            .seqlen = cfg.seqlen,
            .op = 0,
        };
        vkc_buf_t *tsb[5] = {
            &a->n1,
            &ly->shift_mix->w,
            &a->shift,
            &sc->pk,
            &sc->dg_dummy,
        };
        vkc_run(m->c, &m->k_tshift, tsb, (bt * d + 255) / 256, 1, 1, &tpc);

        // project to k,v,r,w,a: cast the shared activation to fp16 once.
        cast_buf(m, &a->shift, &sc->a16, bt * d, 0);
        mm16_pre(m, sc, &a->shift, ly->wk, &sc->pk, bt, d, d);
        mm16_pre(m, sc, &a->shift, ly->wv, &sc->pv, bt, d, d);
        mm16_pre(m, sc, &a->shift, ly->wr, &sc->pr, bt, d, d);
        mm16_pre(m, sc, &a->shift, ly->ww, &sc->pw, bt, d, d);
        mm16_pre(m, sc, &a->shift, ly->wa, &sc->pa, bt, d, d);
        // w in (0,1), a in (0,1)
        vec_op(m, &sc->pw, &sc->pw, bt * d, 2);
        vec_op(m, &sc->pa, &sc->pa, bt * d, 2);

        // pack k,v,r,w,a to [b,head,t,dh], then unit-norm k per head -> stable delta
        // rule
        gpu_repack(m, &sc->pk, &a->pk, bt, hd, dh, cfg.seqlen, 0);
        gpu_repack(m, &sc->pv, &a->pv, bt, hd, dh, cfg.seqlen, 0);
        gpu_repack(m, &sc->pr, &a->pr, bt, hd, dh, cfg.seqlen, 0);
        gpu_repack(m, &sc->pw, &a->pw, bt, hd, dh, cfg.seqlen, 0);
        gpu_repack(m, &sc->pa, &a->pa, bt, hd, dh, cfg.seqlen, 0);
        norm_heads(m, &a->pk, &a->kinv, &a->pk, bt * hd, dh, 0);

        // bidirectional wkv: a left-to-right pass (dir 0) and a right-to-left pass
        // (dir 1), averaged -- a denoiser must see both sides of a masked token. The
        // two passes are independent, so no barrier after the first; the second keeps
        // its barrier so the averaging waits for both.
        wkv_pc_t wpc = (wkv_pc_t) {
            .bsz = cfg.batch,
            .hd = hd,
            .t = cfg.seqlen,
            .dh = dh,
            .dir = 0,
            .save = save ? 1 : 0,
        };
        vkc_buf_t *wb[7] = {
            &a->pk,
            &a->pv,
            &a->pr,
            &a->pw,
            &a->pa,
            &sc->po,
            &a->sstate,
        };
        vkc_set_nobar(m->c, 1);
        vkc_run(m->c, &m->k_wkv_f, wb, cfg.batch * hd, 1, 1, &wpc);
        vkc_set_nobar(m->c, 0);
        wkv_pc_t wpc2 = wpc;
        wpc2.dir = 1;
        vkc_buf_t *wb2[7] = {
            &a->pk,
            &a->pv,
            &a->pr,
            &a->pw,
            &a->pa,
            &sc->po2,
            &a->sstate2,
        };
        vkc_run(m->c, &m->k_wkv_f, wb2, cfg.batch * hd, 1, 1, &wpc2);
        vec_op(m, &sc->po, &sc->po2, bt * d, 3);
        gpu_repack(m, &sc->po, &a->wkv, bt, hd, dh, cfg.seqlen, 1);
        // output projection wo, add residual: x += wo(wkv)
        mm16(m, sc, &a->wkv, ly->wo, &sc->tmp, bt, d, d);
        vec_op(m, &sc->x, &sc->tmp, bt * d, 0);
        if (save) {
            vec_op(m, &a->x_mid, &sc->x, bt * d, 5);
        }

        // channel mixing: n2 = adaLN(rmsnorm(x, n2)), x += down(silu(gate(n2)) * up(n2))
        rmsnorm_fwd(m, &sc->x, &ly->n2->w, &a->n2_pre, &a->rms2, bt, d);
        adaln_fwd(m, &a->n2_pre, &sc->ada6[l], &a->n2, 2);
        // gate and up share the same input: cast it to fp16 once for both matmuls.
        cast_buf(m, &a->n2, &sc->a16, bt * d, 0);
        mm16_pre(m, sc, &a->n2, ly->w_gate, &a->gate, bt, hid, d);
        // gate = silu(gate) in place, pre-activation stashed in gate_pre
        elem_pc_t spc = (elem_pc_t) {
            .n = bt * hid,
            .op = 0,
        };
        vkc_buf_t *seb[2] = {
            &a->gate,
            &a->gate_pre,
        };
        vkc_run(m->c, &m->k_elem, seb, (bt * hid + 255) / 256, 1, 1, &spc);
        mm16_pre(m, sc, &a->n2, ly->w_up, &a->up, bt, hid, d);
        // h = gate * up; the backward needs gate itself, so only multiply in place when
        // not saving.
        vkc_buf_t *h = &a->gate;
        if (save) {
            h = &sc->hbuf;
            vec_op(m, h, &a->gate, bt * hid, 5);
        }
        vec_op(m, h, &a->up, bt * hid, 1);
        mm16(m, sc, h, ly->w_down, &sc->tmp2, bt, d, hid);
        vec_op(m, &sc->x, &sc->tmp2, bt * d, 0);
    }

    // 4. head: logits = head(rmsnorm(x, n_out))
    rmsnorm_fwd(m, &sc->x, &m->n_out->w, &sc->xn, &sc->rms_stash, bt, d);
    mm16(m, sc, &sc->xn, m->head, &sc->logits, bt, cfg.vocab, d);
}

// Record the masked CE: loss[row] per position, and dlogits (for the summed loss)
// written over sc->logits.
static void ce_rec(model_t *m, fwd_scratch_t *sc) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;
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
}

// Inference forward (eval, generation, bench) as one GPU batch. Returns the mean
// loss over masked tokens; sc->logits_save holds the logits.
static float model_forward(model_t *m, fwd_scratch_t *sc) {
    uint32_t bt = m->cfg.batch * m->cfg.seqlen;

    // mirror the current fp32 weights into their fp16 copies for the matrix cores.
    refresh_w16(m);

    vkc_begin(m->c);
    forward_rec(m, sc, 0);

    // snapshot logits before CE overwrites them with dlogits (for inference),
    // copied on the GPU instead of a large host memcpy over mapped memory.
    vec_op(m, &sc->logits_save, &sc->logits, bt * m->cfg.vocab, 5);

    // inference (generation) only needs the logits -- skip the CE loss + host reduce.
    if (m->infer) {
        vkc_flush(m->c);
        return 0.0f;
    }

    ce_rec(m, sc);

    // submit the whole recorded forward and wait, so the host can read the loss.
    vkc_flush(m->c);

    // reduce loss over masked tokens
    float *lp = sc->loss.ptr;
    float *mk = sc->mask.ptr;
    double total = 0.0;
    double cnt = 0.0;
    for (uint32_t i = 0; i < bt; ++i) {
        total += lp[i];
        cnt += mk[i];
    }
    return (float)(total / (cnt + 1e-6));
}

#endif
