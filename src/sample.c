// sample.c -- load the latest checkpoint and show what the denoiser predicts on
// masked Wikipedia text. Runs independently of training (reads ckpt.bin). This is
// a qualitative window into what the model has learned.
#include "model.h"
#include "fwd.h"
#include "step.h"

static uint32_t load_ckpt(model_t *m) {
    FILE *f = fopen("models/ckpt.bin", "rb");
    if (f == NULL) {
        fprintf(stderr, "no checkpoint\n");
        return 0;
    }
    uint32_t step = 0;
    fread(&step, sizeof(uint32_t), 1, f);
    for (uint32_t i = 0; i < m->n_params; i++) {
        param_t *p = &m->params[i];
        fread(p->w.ptr, sizeof(float), p->n, f);
        fread(p->m.ptr, sizeof(float), p->n, f);
        fread(p->v.ptr, sizeof(float), p->n, f);
    }
    fclose(f);
    return step;
}

// forward up to logits only (no CE overwrite), so we can read predictions.
static void forward_logits(model_t *m, fwd_scratch_t *sc) {
    model_forward(m, sc);   // computes logits then CE -> but we re-read pre-CE below
}

int main(void) {
    srand(7);
    vkc_t c = vkc_init();
    mcfg_t cfg = (mcfg_t) {
        .d = 512, .n_layer = 12, .n_head = 8, .seqlen = 64, .batch = 8, .ffn_hidden = 1365,
    };
    model_t *m = model_init(&c, cfg);
    fwd_scratch_t sc = fwd_alloc(m);
    corpus_t cp = corpus_load("data/wiki.txt");
    uint32_t step = load_ckpt(m);
    fprintf(stderr, "loaded checkpoint @ step %u\n", step);

    // build a batch: mask 50% of positions
    make_batch(&cp, cfg.batch, cfg.seqlen,
        sc.ids.ptr, sc.target.ptr, sc.mask.ptr, sc.weight.ptr);

    // forward -> logits (then CE overwrites logits, so capture argmax inside a
    // custom mini-forward). Simplest: call model_forward but it ran CE; instead
    // recompute logits by copying the head output before CE. We re-run the head
    // path here by calling model_forward and reading the loss only is not enough.
    // Easiest correct approach: temporarily zero the mask so CE writes ~nothing,
    // but logits are still overwritten. So: do the forward, but read predictions
    // from a fresh logits buffer we fill ourselves.
    // quantitative: average bits/byte AND masked-fill accuracy over many held-out
    // batches (single-batch numbers are too noisy with ~14 masked positions).
    double raw = 0.0, cnt = 0.0;
    uint32_t acc_correct = 0, acc_total = 0;
    // bucket accuracy by mask rate t (recovered as 1/weight): low t = easy (few
    // masked), high t = hard (mostly masked). 3 buckets: t<0.33, 0.33-0.66, >0.66.
    uint32_t bc[3] = {0, 0, 0}, bt_[3] = {0, 0, 0};
    int n_batch = 8;
    for (int it = 0; it < n_batch; it++) {
        make_batch(&cp, cfg.batch, cfg.seqlen,
            sc.ids.ptr, sc.target.ptr, sc.mask.ptr, sc.weight.ptr);
        forward_logits(m, &sc);
        float *lp = sc.loss.ptr;
        float *mk = sc.mask.ptr;
        float *wt = sc.weight.ptr;
        uint32_t *bids = sc.ids.ptr;
        uint32_t *btgt = sc.target.ptr;
        float *blg = sc.logits_save.ptr;
        for (uint32_t i = 0; i < cfg.batch * cfg.seqlen; i++) {
            raw += lp[i] / (wt[i] + 1e-9);
            cnt += mk[i];
            if (bids[i] == DATA_MASK_ID) {
                float *row = blg + i * M_VOCAB;
                int best = 0;
                for (int v = 1; v < M_VOCAB; v++) {
                    if (row[v] > row[best]) {
                        best = v;
                    }
                }
                acc_total++;
                int hit = ((uint32_t)best == btgt[i]) ? 1 : 0;
                acc_correct += hit;
                float t = 1.0f / (wt[i] + 1e-9);
                int b = (t < 0.33f) ? 0 : (t < 0.66f ? 1 : 2);
                bt_[b]++;
                bc[b] += hit;
            }
        }
    }
    double bpb = (raw / (cnt + 1e-6)) / 0.6931472;
    printf("checkpoint step %u: held-out bits/byte = %.4f  (random = 8.0)\n", step, bpb);
    printf("accuracy by mask rate: low(<33%%)=%.1f%%  mid=%.1f%%  high(>66%%)=%.1f%%\n",
        100.0 * bc[0] / (bt_[0] + 1), 100.0 * bc[1] / (bt_[1] + 1), 100.0 * bc[2] / (bt_[2] + 1));
    printf("masked-fill accuracy over %d batches: %u/%u = %.1f%% (random 0.4%%)\n\n",
        n_batch, acc_correct, acc_total, 100.0 * acc_correct / (acc_total + 1));

    // re-run one batch for the qualitative sample below
    make_batch(&cp, cfg.batch, cfg.seqlen, sc.ids.ptr, sc.target.ptr, sc.mask.ptr, sc.weight.ptr);
    forward_logits(m, &sc);

    // qualitative: for the first sequence, show actual vs the model's prediction at
    // masked positions ('_' = a position the model had to fill).
    uint32_t *ids = sc.ids.ptr;
    uint32_t *tgt = sc.target.ptr;
    float *lg = sc.logits_save.ptr;
    uint32_t correct = 0, total = 0;
    printf("actual : ");
    for (uint32_t s = 0; s < cfg.seqlen; s++) {
        unsigned char ch = (unsigned char)tgt[s];
        putchar((ch >= 32 && ch < 127) ? ch : '.');
    }
    printf("\nmodel  : ");
    for (uint32_t s = 0; s < cfg.seqlen; s++) {
        if (ids[s] == DATA_MASK_ID) {
            float *row = lg + s * M_VOCAB;
            int best = 0;
            for (int v = 1; v < M_VOCAB; v++) {
                if (row[v] > row[best]) {
                    best = v;
                }
            }
            unsigned char ch = (unsigned char)best;
            putchar((ch >= 32 && ch < 127) ? ch : '.');
            total++;
            if ((uint32_t)best == tgt[s]) {
                correct++;
            }
        } else {
            putchar(' ');   // not masked -> model saw it
        }
    }
    printf("\n\nmasked-position accuracy: %u/%u = %.1f%%\n",
        correct, total, total ? 100.0 * correct / total : 0.0);
    return 0;
}
