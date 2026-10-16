// test_grad.c -- finite-difference check of the hand-written backward, plus a check
// that training and inference run the same network. A tiny model, with every param
// randomly initialized so adaLN, the gains and the token shift are all exercised, is
// checked per param tensor along a random +-1 direction u: the analytic g.u from
// train_step against (L(w + eps u) - L(w - eps u)) / (2 eps) from model_forward. The
// fp16 forward's quantization noise swamps the finite differences, so the model runs
// in the fp32 reference mode (ref32); the training/inference loss match is checked in
// the normal fp16 mode. Two step sizes are tried and a tensor passes if either agrees:
// the large step is limited by curvature, the small one by the float resolution of
// the loss, while a real gradient bug is wrong at both. Exits nonzero on a failure.
#include "model.h"
#include "fwd.h"
#include "step.h"

static float param_loss(model_t *m, fwd_scratch_t *sc) {
    m->dirty16 = 1;
    return model_forward(m, sc);
}

// central difference of the loss along u for param p; restores p's weights.
static double fd_grad(model_t *m, fwd_scratch_t *sc, param_t *p, const float *u, float eps) {
    float *w = p->w.ptr;
    float *w0 = malloc(p->n * sizeof(float));
    memcpy(w0, w, p->n * sizeof(float));
    for (uint32_t j = 0; j < p->n; ++j) {
        w[j] = w0[j] + eps * u[j];
    }
    float lp = param_loss(m, sc);
    for (uint32_t j = 0; j < p->n; ++j) {
        w[j] = w0[j] - eps * u[j];
    }
    float lm = param_loss(m, sc);
    memcpy(w, w0, p->n * sizeof(float));
    free(w0);
    return ((double)lp - (double)lm) / (2.0 * eps);
}

int main(void) {
    srand(11);
    vkc_t c = vkc_init();

    uint32_t vocab = 64;
    data_vocab = vocab;
    data_mask_id = vocab;
    mcfg_t cfg = (mcfg_t) {
        .d = 64,
        .n_layer = 2,
        .n_head = 1,
        .seqlen = 32,
        .batch = 2,
        .ffn_hidden = 128,
        .vocab = vocab,
        .vocab_in = vocab + 1,
    };
    model_t *m = model_init(&c, cfg);
    trainer_t tr = trainer_make(m);
    fwd_scratch_t *sc = &tr.sc;

    // near-unit-gain matrices (std ~ 1/sqrt(d)) so every block visibly moves the loss;
    // at the real 0.02 init many grads are too small for finite differences to resolve
    for (uint32_t i = 0; i < m->n_params; ++i) {
        param_t *p = &m->params[i];
        if (p->n >= cfg.d * cfg.d) {
            randn_fill(p->w.ptr, p->n, 0.1f);
        }
    }
    // move the zero/constant-initialized params off their special values
    for (uint32_t l = 0; l < cfg.n_layer; ++l) {
        mlayer_t *ly = &m->layers[l];
        param_t *gains[2] = { ly->n1, ly->n2 };
        for (int j = 0; j < 2; ++j) {
            float *w = gains[j]->w.ptr;
            for (uint32_t i = 0; i < gains[j]->n; ++i) {
                w[i] = w[i] + 0.2f * ((float)rand() / RAND_MAX - 0.5f);
            }
        }
        float *mix = ly->shift_mix->w.ptr;
        for (uint32_t i = 0; i < ly->shift_mix->n; ++i) {
            mix[i] = mix[i] + 0.4f * ((float)rand() / RAND_MAX - 0.5f);
        }
        randn_fill(ly->ada->w.ptr, ly->ada->n, 0.05f);
    }
    {
        float *w = m->n_out->w.ptr;
        for (uint32_t i = 0; i < m->n_out->n; ++i) {
            w[i] = w[i] + 0.2f * ((float)rand() / RAND_MAX - 0.5f);
        }
    }

    // a synthetic corpus of random tokens
    uint32_t clen = 4096;
    uint32_t *cids = malloc(clen * sizeof(uint32_t));
    for (uint32_t i = 0; i < clen; ++i) {
        cids[i] = (uint32_t)rand() % vocab;
    }
    corpus_t cp = {
        .ids = cids,
        .len = clen,
    };
    make_batch(&cp, cfg.batch, cfg.seqlen, sc->ids.ptr, sc->target.ptr, sc->mask.ptr, sc->weight.ptr);

    m->dirty16 = 1;
    float l_train = train_step(&tr);
    float l_infer = param_loss(m, sc);
    int ok = fabsf(l_train - l_infer) <= 1e-5f * fabsf(l_train);
    printf("loss: train_step %.6f  model_forward %.6f  %s\n", l_train, l_infer, ok ? "ok" : "FAIL");

    m->ref32 = 1;
    train_step(&tr);

    float eps[2] = { 1e-3f, 1e-4f };
    double tol = 0.02;
    for (uint32_t i = 0; i < m->n_params; ++i) {
        param_t *p = &m->params[i];
        float *g = p->g.ptr;
        float *u = malloc(p->n * sizeof(float));
        double analytic = 0.0;
        for (uint32_t j = 0; j < p->n; ++j) {
            u[j] = (rand() & 1) ? 1.0f : -1.0f;
            analytic = analytic + (double)g[j] * u[j];
        }
        double numeric[2];
        double best = INFINITY;
        for (int e = 0; e < 2; ++e) {
            numeric[e] = fd_grad(m, sc, p, u, eps[e]);
            double scale = fmax(fabs(analytic), fabs(numeric[e]));
            double rel = scale > 0.0 ? fabs(analytic - numeric[e]) / scale : 0.0;
            best = fmin(best, rel);
        }
        free(u);
        int pass = best <= tol;
        ok = ok && pass;
        printf("%-3u %-10s analytic %+.6e  numeric %+.6e %+.6e  rel %.4f %s\n", i, p->name, analytic, numeric[0], numeric[1], best, pass ? "ok" : "FAIL");
    }
    return ok ? 0 : 1;
}
