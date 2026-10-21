#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vkc.h"
#include "hash.h"
#include "util.h"
#include "data.h"
#include "model.h"
#include "fwd.h"
#include "step.h"

float mda_test_grad_loss(mda_model_t *m, mda_fwd_scratch_t *sc) {
    m->dirty16 = true;
    return mda_fwd_run(m, sc);
}

double mda_test_grad_fd(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_param_t *p, const float *u, float eps) {
    float *w = p->w.ptr;
    float *w0 = malloc(p->n * sizeof(float));
    memcpy(w0, w, p->n * sizeof(float));
    for (uint32_t j = 0; j < p->n; j += 1) {
        w[j] = w0[j] + eps * u[j];
    }
    float lp = mda_test_grad_loss(m, sc);
    for (uint32_t j = 0; j < p->n; j += 1) {
        w[j] = w0[j] - eps * u[j];
    }
    float lm = mda_test_grad_loss(m, sc);
    memcpy(w, w0, p->n * sizeof(float));
    m->dirty16 = true;
    free(w0);
    return ((double)lp - (double)lm) / (2.0 * eps);
}

void mda_test_grad_jitter(mda_model_param_t *p, float amount) {
    float *w = p->w.ptr;
    for (uint32_t j = 0; j < p->n; j += 1) {
        w[j] += amount * ((float)mda_hash_unit(MDA_HASH_TEST, p->off, j) - 0.5f);
    }
}

double mda_test_grad_ratio(double num, double den) {
    if (den > 0.0) {
        return num / den;
    } else {
        return num;
    }
}

double mda_test_grad_rel(double diff, double scale) {
    if (scale > 0.0) {
        return diff / scale;
    } else {
        return 0.0;
    }
}

float mda_test_grad_sign(uint64_t h) {
    if (h & 1) {
        return 1.0f;
    } else {
        return -1.0f;
    }
}

void mda_test_grad_perturb(mda_model_t *m) {
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        mda_model_param_t *p = &m->params[i];
        if (p->n >= m->cfg.d * m->cfg.d) {
            float *w = p->w.ptr;
            for (uint32_t j = 0; j < p->n; j += 1) {
                w[j] = 0.1f * mda_hash_normal(MDA_HASH_TEST, p->off, j);
            }
        }
    }
    for (uint32_t l = 0; l < m->cfg.n_layer; l += 1) {
        mda_model_layer_t *ly = &m->layers[l];
        mda_test_grad_jitter(ly->n1, 0.2f);
        mda_test_grad_jitter(ly->n2, 0.2f);
        mda_test_grad_jitter(ly->shift_mix, 0.4f);
    }
    mda_test_grad_jitter(m->n_out, 0.2f);
}

uint32_t mda_test_grad_fp16(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    float *g16 = malloc(m->n_all * sizeof(float));
    mda_step_grad(tr);
    memcpy(g16, m->g_all.ptr, m->n_all * sizeof(float));
    m->ref32 = true;
    mda_step_grad(tr);
    float *g32 = m->g_all.ptr;
    uint32_t fails = 0;
    for (uint32_t i = 0; i < m->n_params; i += 1) {
        mda_model_param_t *p = &m->params[i];
        double num = 0.0;
        double den = 0.0;
        for (uint32_t j = 0; j < p->n; j += 1) {
            double e = (double)g16[p->off + j] - g32[p->off + j];
            num += e * e;
            den += (double)g32[p->off + j] * g32[p->off + j];
        }
        double rel = sqrt(mda_test_grad_ratio(num, den));
        bool pass = rel <= 0.03;
        if (pass == false) {
            fails += 1;
        }
        printf("%u %s fp16 vs fp32 grad: rel err %.4f %s\n", i, p->name, rel, mda_util_verdict(pass));
    }
    free(g16);
    return fails;
}

float mda_test_grad_eps(double change, float loss, double analytic) {
    if (analytic > 0.0) {
        return (float)(change * fabs(loss) / analytic);
    } else {
        return 1e-3f;
    }
}

bool mda_test_grad_fd_param(mda_model_t *m, mda_fwd_scratch_t *sc, uint32_t i, float l_infer) {
    mda_model_param_t *p = &m->params[i];
    float gs = mda_fwd_loss_scale(m->cfg);
    float *g = p->g.ptr;
    float *u = malloc(p->n * sizeof(float));
    double analytic = 0.0;
    for (uint32_t j = 0; j < p->n; j += 1) {
        u[j] = mda_test_grad_sign(g[j] >= 0.0f);
        analytic += (double)g[j] * gs * u[j];
    }
    float eps[2] = {
        mda_test_grad_eps(1e-3, l_infer, analytic),
        mda_test_grad_eps(1e-4, l_infer, analytic),
    };
    double numeric[2];
    double best = INFINITY;
    for (uint32_t e = 0; e < 2; e += 1) {
        numeric[e] = mda_test_grad_fd(m, sc, p, u, eps[e]);
        double diff = fabs(analytic - numeric[e]);
        best = fmin(best, mda_test_grad_rel(diff, fmax(fabs(analytic), fabs(numeric[e]))));
    }
    free(u);
    bool pass = best <= 0.005;
    printf("%u %s analytic %+.6e numeric %+.6e %+.6e rel %.4f %s\n", i, p->name, analytic, numeric[0], numeric[1], best, mda_util_verdict(pass));
    return pass;
}

uint32_t mda_test_grad_fd_all(mda_step_trainer_t *tr, float l_infer) {
    uint32_t fails = 0;
    for (uint32_t i = 0; i < tr->m->n_params; i += 1) {
        bool pass = mda_test_grad_fd_param(tr->m, &tr->sc, i, l_infer);
        if (pass == false) {
            fails += 1;
        }
    }
    return fails;
}

bool mda_test_grad_opt(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    uint32_t n = m->n_all;
    float gs = mda_fwd_loss_scale(m->cfg);
    float *g0 = malloc(n * sizeof(float));
    float *w0 = malloc(n * sizeof(float));
    m->ref32 = false;
    mda_vkc_fill(m->c, &m->m_all, 0);
    mda_vkc_fill(m->c, &m->v_all, 0);
    mda_step_grad(tr);
    memcpy(g0, m->g_all.ptr, n * sizeof(float));
    memcpy(w0, m->w_all.ptr, n * sizeof(float));
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i += 1) {
        ss += (double)g0[i] * g0[i];
    }
    double norm = gs * sqrt(ss);
    double scale = gs * fmin(1.0, 1.0 / norm);
    float lr = 1e-3f;
    mda_step_train(tr, lr, 1);
    float *mg = m->m_all.ptr;
    float *vg = m->v_all.ptr;
    float *wg = m->w_all.ptr;
    double em = 0.0;
    double dm = 0.0;
    double ew = 0.0;
    double dw = 0.0;
    for (uint32_t i = 0; i < n; i += 1) {
        double mi = (1.0 - MDA_STEP_ADAM_B1) * g0[i] * scale;
        em += (mg[i] - mi) * (mg[i] - mi);
        dm += mi * mi;
        double upd = lr * (mg[i] / (1.0 - MDA_STEP_ADAM_B1)) / (sqrt(vg[i] / (1.0 - MDA_STEP_ADAM_B2)) + 1e-8);
        double wi = w0[i] - upd;
        ew += (wg[i] - wi) * (wg[i] - wi);
        dw += upd * upd;
    }
    free(g0);
    free(w0);
    double nrel = fabs(tr->gnorm - norm) / norm;
    double mrel = sqrt(em / dm);
    double wrel = sqrt(ew / dw);
    bool ok = nrel < 1e-3 && mrel < 1e-3 && wrel < 1e-3;
    printf("optimizer: grad norm %.5f (host %.5f), moment rel err %.2e, update rel err %.2e %s\n", tr->gnorm, norm, mrel, wrel, mda_util_verdict(ok));
    return ok;
}

double mda_test_grad_diff(const float *a, const float *b, uint32_t n) {
    double num = 0.0;
    double den = 0.0;
    for (uint32_t i = 0; i < n; i += 1) {
        num += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
        den += (double)b[i] * b[i];
    }
    return sqrt(mda_test_grad_ratio(num, den));
}

bool mda_test_grad_accum(mda_step_trainer_t *tr, mda_step_trainer_t *tr2, const mda_data_corpus_t *cp) {
    mda_model_t *m = tr->m;
    uint32_t n = m->n_all;
    float *sum = malloc(n * sizeof(float));
    float *gb = m->g_all.ptr;
    m->ref32 = false;
    mda_data_batch(&tr->sc.batch, cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL,1);
    mda_step_grad(tr);
    memcpy(sum, gb, n * sizeof(float));
    mda_data_batch(&tr->sc.batch, cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL,2);
    mda_step_grad(tr);
    for (uint32_t i = 0; i < n; i += 1) {
        sum[i] += gb[i];
    }
    mda_step_begin(tr2, 0.0f, 1);
    for (uint32_t i = 0; i < 2; i += 1) {
        mda_data_batch(&tr2->sc.batch, cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL,i + 1);
        mda_step_micro(tr2, i);
    }
    double rel = mda_test_grad_diff(gb, sum, n);
    free(sum);
    bool ok = rel < 1e-3;
    printf("accumulation: 2 micro-batches vs summed grads, rel err %.2e %s\n", rel, mda_util_verdict(ok));
    return ok;
}

void mda_test_grad_gram(const double *x, double *a, uint32_t r, uint32_t c) {
    uint32_t s = mda_util_min_u32(r, c);
    uint32_t len = mda_util_max_u32(r, c);
    for (uint32_t i = 0; i < s; i += 1) {
        for (uint32_t j = 0; j < s; j += 1) {
            double acc = 0.0;
            for (uint32_t k = 0; k < len; k += 1) {
                if (r <= c) {
                    acc += x[i * c + k] * x[j * c + k];
                } else {
                    acc += x[k * c + i] * x[k * c + j];
                }
            }
            a[i * s + j] = acc;
        }
    }
}

void mda_test_grad_poly(const double *a, double *b, uint32_t s) {
    for (uint32_t i = 0; i < s; i += 1) {
        for (uint32_t j = 0; j < s; j += 1) {
            double aa = 0.0;
            for (uint32_t k = 0; k < s; k += 1) {
                aa += a[i * s + k] * a[k * s + j];
            }
            b[i * s + j] = MDA_STEP_MUON_B * a[i * s + j] + MDA_STEP_MUON_C * aa;
        }
    }
}

void mda_test_grad_ns_step(double *x, const double *b, double *t, uint32_t r, uint32_t c) {
    uint32_t s = mda_util_min_u32(r, c);
    for (uint32_t i = 0; i < r; i += 1) {
        for (uint32_t j = 0; j < c; j += 1) {
            double acc = 0.0;
            for (uint32_t k = 0; k < s; k += 1) {
                if (r <= c) {
                    acc += b[i * s + k] * x[k * c + j];
                } else {
                    acc += x[i * c + k] * b[k * s + j];
                }
            }
            t[i * c + j] = acc;
        }
    }
    for (uint32_t i = 0; i < r * c; i += 1) {
        x[i] = MDA_STEP_MUON_A * x[i] + t[i];
    }
}

void mda_test_grad_ns(double *x, uint32_t r, uint32_t c) {
    uint32_t s = mda_util_min_u32(r, c);
    double *a = malloc(s * s * sizeof(double));
    double *b = malloc(s * s * sizeof(double));
    double *t = malloc(r * c * sizeof(double));
    for (uint32_t it = 0; it < MDA_STEP_MUON_ITERS; it += 1) {
        mda_test_grad_gram(x, a, r, c);
        mda_test_grad_poly(a, b, s);
        mda_test_grad_ns_step(x, b, t, r, c);
    }
    free(a);
    free(b);
    free(t);
}

bool mda_test_grad_muon_param(const mda_model_param_t *p, const float *w0, float scale, float lr) {
    uint32_t n = p->n;
    double *x = malloc(n * sizeof(double));
    const float *g = p->g.ptr;
    const float *w = p->w.ptr;
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i += 1) {
        x[i] = (1.0 + MDA_STEP_MUON_MOMENTUM) * g[i] * scale;
        ss += x[i] * x[i];
    }
    for (uint32_t i = 0; i < n; i += 1) {
        x[i] /= sqrt(ss);
    }
    mda_test_grad_ns(x, p->rows, p->cols);
    double f = lr * MDA_STEP_MUON_RMS * sqrt((double)mda_util_max_u32(p->rows, p->cols));
    double num = 0.0;
    double den = 0.0;
    for (uint32_t i = 0; i < n; i += 1) {
        double want = f * x[i];
        double got = (double)w0[p->off + i] - w[i];
        num += (got - want) * (got - want);
        den += want * want;
    }
    free(x);
    double rel = sqrt(num / den);
    bool ok = rel < 0.05;
    printf("muon %s [%u x %u]: update vs host Newton-Schulz, rel err %.4f %s\n", p->name, p->rows, p->cols, rel, mda_util_verdict(ok));
    return ok;
}

bool mda_test_grad_muon(mda_step_trainer_t *tr3, const mda_data_corpus_t *cp) {
    mda_model_t *m = tr3->m;
    float *w0 = malloc(m->n_all * sizeof(float));
    float lr = 1e-3f;
    memcpy(w0, m->w_all.ptr, m->n_all * sizeof(float));
    mda_vkc_fill(m->c, &m->m_all, 0);
    mda_data_batch(&tr3->sc.batch, cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL,3);
    mda_step_train(tr3, lr, 1);
    mda_fwd_opt_t *o = tr3->sc.opt.ptr;
    mda_model_layer_t *ly = &m->layers[0];
    bool wk = mda_test_grad_muon_param(ly->wk, w0, o->scale, lr);
    bool gate = mda_test_grad_muon_param(ly->w_gate, w0, o->scale, lr);
    bool down = mda_test_grad_muon_param(ly->w_down, w0, o->scale, lr);
    free(w0);
    return wk && gate && down;
}

bool mda_test_grad_skip(mda_step_trainer_t *tr) {
    mda_model_t *m = tr->m;
    mda_fwd_opt_t *o = tr->sc.opt.ptr;
    float *w0 = malloc(m->n_all * sizeof(float));
    o->gmul = 1e38f;
    memcpy(w0, m->w_all.ptr, m->n_all * sizeof(float));
    mda_step_train(tr, 1e-3f, 2);
    bool unchanged = memcmp(w0, m->w_all.ptr, m->n_all * sizeof(float)) == 0;
    free(w0);
    bool ok = tr->skipped && unchanged && o->gmul == 5e37f;
    printf("overflow: skipped %d, weights unchanged %d, loss scale %g %s\n", tr->skipped, unchanged, o->gmul, mda_util_verdict(ok));
    return ok;
}

bool mda_test_grad_causal(mda_step_trainer_t *tr, const mda_data_corpus_t *cp) {
    mda_model_t *m = tr->m;
    mda_fwd_opt_t *o = tr->sc.opt.ptr;
    o->gmul = 1.0f;
    m->causal = true;
    m->ref32 = true;
    mda_data_batch(&tr->sc.batch, cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_AR, 4);
    float l_train = mda_step_grad(tr);
    float l_infer = mda_test_grad_loss(m, &tr->sc);
    bool loss_ok = fabsf(l_train - l_infer) <= 1e-5f * fabsf(l_train);
    printf("causal loss: train %.6f infer %.6f %s\n", l_train, l_infer, mda_util_verdict(loss_ok));
    uint32_t fails = mda_test_grad_fd_all(tr, l_infer);
    m->causal = false;
    m->ref32 = false;
    return loss_ok && fails == 0;
}

bool mda_test_grad_blocks(mda_vkc_t *c, mda_model_cfg_t cfg, const mda_data_corpus_t *cp) {
    mda_model_t *m = mda_model_init(c, mda_model_cfg_block(cfg, MDA_MODEL_BLOCK));
    if (m == NULL) {
        return false;
    } else {
        mda_step_trainer_t tr = mda_step_init(m, 1.0f, 1, false);
        mda_test_grad_perturb(m);
        mda_data_batch(&tr.sc.batch, cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL, 5);
        m->dirty16 = true;
        float l_train = mda_step_grad(&tr);
        float l_infer = mda_test_grad_loss(m, &tr.sc);
        bool loss_ok = fabsf(l_train - l_infer) <= 1e-5f * fabsf(l_train);
        printf("block loss: train %.6f infer %.6f %s\n", l_train, l_infer, mda_util_verdict(loss_ok));
        uint32_t fp16_fails = mda_test_grad_fp16(&tr);
        uint32_t fd_fails = mda_test_grad_fd_all(&tr, l_infer);
        return c->fail == false && loss_ok && fp16_fails == 0 && fd_fails == 0;
    }
}

void mda_test_grad_copy_rows(mda_data_batch_t *dst, const mda_data_batch_t *src) {
    uint32_t n = src->batch * src->seqlen;
    uint32_t o = dst->nclean * dst->seqlen;
    memcpy(dst->ids + o, src->ids, n * sizeof(uint32_t));
    memcpy(dst->target + o, src->target, n * sizeof(uint32_t));
    memcpy(dst->mask + o, src->mask, n * sizeof(float));
    memcpy(dst->weight + o, src->weight, n * sizeof(float));
    memcpy(dst->t + o, src->t, n * sizeof(float));
    for (uint32_t i = 0; i < dst->nclean * dst->seqlen; i += 1) {
        dst->ids[i] = src->target[i];
        dst->target[i] = src->target[i];
        dst->mask[i] = 0.0f;
        dst->weight[i] = 1.0f;
        dst->t[i] = 0.0f;
    }
}

mda_model_t *mda_test_grad_model(mda_vkc_t *c, mda_model_cfg_t cfg) {
    mda_model_t *m = mda_model_init(c, cfg);
    if (m != NULL) {
        mda_test_grad_perturb(m);
        m->ref32 = true;
        m->dirty16 = true;
    }
    return m;
}

bool mda_test_grad_one_block(mda_vkc_t *c, mda_model_cfg_t cfg, const mda_data_corpus_t *cp) {
    mda_model_t *a = mda_test_grad_model(c, cfg);
    mda_model_t *b = mda_test_grad_model(c, mda_model_cfg_block(cfg, cfg.seqlen));
    if (a == NULL || b == NULL) {
        return false;
    } else {
        mda_fwd_scratch_t sa = mda_fwd_alloc(a, 1);
        mda_fwd_scratch_t sb = mda_fwd_alloc(b, 1);
        mda_data_batch(&sa.batch, cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL, 9);
        mda_test_grad_copy_rows(&sb.batch, &sa.batch);
        float la = mda_fwd_run(a, &sa);
        float lb = mda_fwd_run(b, &sb);
        bool ok = fabsf(la - 2.0f * lb) <= 1e-5f * fabsf(la);
        printf("one block = plain diffusion: loss %.6f vs %.6f %s\n", la, 2.0f * lb, mda_util_verdict(ok));
        return ok;
    }
}

float *mda_test_grad_losses(mda_model_t *m, mda_fwd_scratch_t *sc) {
    uint32_t n = m->cfg.batch * m->cfg.seqlen;
    float *out = malloc(n * sizeof(float));
    mda_fwd_run(m, sc);
    memcpy(out, sc->loss.ptr, n * sizeof(float));
    return out;
}

uint32_t mda_test_grad_changed(const float *x, const float *y, uint32_t lo, uint32_t hi) {
    uint32_t changed = 0;
    for (uint32_t i = lo; i < hi; i += 1) {
        changed += x[i] != y[i];
    }
    return changed;
}

bool mda_test_grad_causal_blocks(mda_vkc_t *c, mda_model_cfg_t cfg, const mda_data_corpus_t *cp) {
    mda_model_t *m = mda_test_grad_model(c, mda_model_cfg_block(cfg, cfg.seqlen / 2));
    if (m == NULL) {
        return false;
    } else {
        mda_fwd_scratch_t sc = mda_fwd_alloc(m, 1);
        mda_data_batch_t *b = &sc.batch;
        uint32_t half = cfg.seqlen / 2;
        uint32_t noisy = b->nclean * cfg.seqlen;
        mda_data_batch(b, cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL, 10);
        float *base = mda_test_grad_losses(m, &sc);
        b->ids[half + 3] = (b->ids[half + 3] + 1) % cfg.vocab;
        b->ids[noisy + half + 3] = (b->target[noisy + half + 3] + 1) % cfg.vocab;
        float *later = mda_test_grad_losses(m, &sc);
        b->ids[3] = (b->ids[3] + 1) % cfg.vocab;
        float *earlier = mda_test_grad_losses(m, &sc);
        uint32_t leak = mda_test_grad_changed(base, later, noisy, noisy + half);
        uint32_t seen = mda_test_grad_changed(later, earlier, noisy + half, noisy + cfg.seqlen);
        bool ok = leak == 0 && seen > 0;
        printf("block causality: block 0 losses changed by block 1 tokens %u, block 1 losses changed by a block 0 clean token %u %s\n", leak, seen, mda_util_verdict(ok));
        free(base);
        free(later);
        free(earlier);
        return ok;
    }
}

void mda_test_grad_slice(mda_data_batch_t *dst, const mda_data_batch_t *src, uint32_t lo) {
    for (uint32_t r = 0; r < dst->batch; r += 1) {
        for (uint32_t p = 0; p < dst->seqlen; p += 1) {
            uint32_t i = r * dst->seqlen + p;
            uint32_t j = r * src->seqlen + lo + p;
            dst->ids[i] = src->ids[j];
            dst->target[i] = src->target[j];
            dst->mask[i] = src->mask[j];
            dst->weight[i] = src->weight[j];
            dst->t[i] = src->t[j];
        }
    }
}

double mda_test_grad_slice_err(const float *full, const float *part, mda_model_cfg_t fc, mda_model_cfg_t pc, uint32_t lo) {
    double err = 0.0;
    double scale = 0.0;
    for (uint32_t r = 0; r < pc.batch; r += 1) {
        for (uint32_t p = 0; p < pc.seqlen; p += 1) {
            double a = full[r * fc.seqlen + lo + p];
            double b = part[r * pc.seqlen + p];
            err = fmax(err, fabs(a - b));
            scale = fmax(scale, fabs(a));
        }
    }
    return err / scale;
}

bool mda_test_grad_cache(mda_vkc_t *c, mda_model_cfg_t cfg, const mda_data_corpus_t *cp) {
    uint32_t blk = cfg.seqlen / 2;
    mda_model_cfg_t short_cfg = cfg;
    short_cfg.seqlen = blk;
    mda_model_t *f = mda_test_grad_model(c, mda_model_cfg_block(cfg, blk));
    mda_model_t *d = mda_test_grad_model(c, mda_model_cfg_block(short_cfg, blk));
    if (f == NULL || d == NULL) {
        return false;
    } else {
        mda_fwd_scratch_t sf = mda_fwd_alloc(f, 1);
        mda_fwd_scratch_t sd = mda_fwd_alloc(d, 1);
        mda_fwd_cache_t cache = mda_fwd_cache_alloc(d);
        sd.cache = &cache;
        mda_data_batch(&sf.batch, cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL, 11);
        float *full = mda_test_grad_losses(f, &sf);
        double worst = 0.0;
        for (uint32_t lo = 0; lo < cfg.seqlen; lo += blk) {
            mda_test_grad_slice(&sd.batch, &sf.batch, lo);
            float *part = mda_test_grad_losses(d, &sd);
            worst = fmax(worst, mda_test_grad_slice_err(full, part, f->cfg, d->cfg, lo));
            mda_fwd_cache_commit(&cache);
            free(part);
        }
        free(full);
        bool ok = worst < 1e-4;
        printf("cached block decoding vs full sequence: max rel loss err %.2e %s\n", worst, mda_util_verdict(ok));
        return ok;
    }
}

bool mda_test_grad_trainer(mda_step_trainer_t *tr, mda_step_trainer_t *tr2, mda_step_trainer_t *tr3) {
    mda_model_t *m = tr->m;
    mda_test_grad_perturb(m);
    uint32_t clen = 4096;
    uint32_t *cids = malloc(clen * sizeof(uint32_t));
    for (uint32_t i = 0; i < clen; i += 1) {
        cids[i] = (uint32_t)(mda_hash_u64(MDA_HASH_CORPUS, 0, i) % m->cfg.vocab);
    }
    mda_data_corpus_t cp = {
        .ids = cids,
        .len = clen,
        .eval_start = clen,
        .mask_id = m->cfg.vocab,
    };
    mda_data_batch(&tr->sc.batch, &cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL,0);
    m->dirty16 = true;
    float l_train = mda_step_grad(tr);
    float l_infer = mda_test_grad_loss(m, &tr->sc);
    bool loss_ok = fabsf(l_train - l_infer) <= 1e-5f * fabsf(l_train);
    printf("loss: train %.6f infer %.6f %s\n", l_train, l_infer, mda_util_verdict(loss_ok));
    uint32_t fp16_fails = mda_test_grad_fp16(tr);
    uint32_t fd_fails = mda_test_grad_fd_all(tr, l_infer);
    bool accum_ok = mda_test_grad_accum(tr, tr2, &cp);
    mda_data_batch(&tr->sc.batch, &cp, MDA_DATA_SPLIT_TRAIN, MDA_DATA_SCHED_BELL,0);
    bool opt_ok = mda_test_grad_opt(tr);
    bool muon_ok = mda_test_grad_muon(tr3, &cp);
    bool skip_ok = mda_test_grad_skip(tr);
    bool causal_ok = mda_test_grad_causal(tr, &cp);
    bool block_ok = mda_test_grad_blocks(m->c, m->cfg, &cp);
    bool one_ok = mda_test_grad_one_block(m->c, m->cfg, &cp);
    bool order_ok = mda_test_grad_causal_blocks(m->c, m->cfg, &cp);
    bool cache_ok = mda_test_grad_cache(m->c, m->cfg, &cp);
    free(cids);
    return m->c->fail == false && loss_ok && fp16_fails == 0 && fd_fails == 0 && accum_ok && opt_ok && muon_ok && skip_ok && causal_ok && block_ok && one_ok && order_ok && cache_ok;
}

bool mda_test_grad_device(mda_vkc_t *c) {
    mda_model_cfg_t cfg = {
        .d = 128,
        .n_layer = 2,
        .n_head = 2,
        .seqlen = 32,
        .batch = 2,
        .ffn_hidden = 192,
        .vocab = 64,
    };
    mda_model_t *m = mda_model_init(c, cfg);
    if (m == NULL) {
        return false;
    } else {
        mda_step_trainer_t tr = mda_step_init(m, 1.0f, 1, false);
        mda_step_trainer_t tr2 = mda_step_init(m, 1.0f, 2, false);
        mda_step_trainer_t tr3 = mda_step_init(m, 1.0f, 1, true);
        if (c->fail) {
            return false;
        } else {
            return mda_test_grad_trainer(&tr, &tr2, &tr3);
        }
    }
}

bool mda_test_grad_main(void) {
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_test_grad_device(&c);
    }
}

int main(void) {
    bool ok = mda_test_grad_main();
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
