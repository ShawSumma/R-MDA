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
    for (uint32_t i = 0; i < m->n_params; i = i + 1) {
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
    model_forward(m, sc);
}

// argmax over a vocab-sized row of logits.
static uint32_t argmax_row(float *row, uint32_t vocab) {
    uint32_t best = 0;
    for (uint32_t v = 1; v < vocab; v = v + 1) {
        if (row[v] > row[best]) {
            best = v;
        }
    }
    return best;
}

int main(void) {
    srand(7);
    vkc_t c = vkc_init();
    corpus_t cp = corpus_load("data/wiki.txt");
    mcfg_t cfg = (mcfg_t) {
        .d = 640,
        .n_layer = 12,
        .n_head = 10,
        .seqlen = 64,
        .batch = 8,
        .ffn_hidden = 1712,
        .vocab = data_vocab,
        .vocab_in = data_vocab + 1,
    };
    model_t *m = model_init(&c, cfg);
    fwd_scratch_t sc = fwd_alloc(m);
    uint32_t step = load_ckpt(m);
    m->dirty16 = 1;
    fprintf(stderr, "loaded checkpoint @ step %u\n", step);

    // quantitative: average bits/byte and masked-fill accuracy over many held-out
    // batches (single-batch numbers are too noisy with so few masked positions).
    // accuracy is also bucketed by mask rate t (recovered as 1/weight): low t is
    // easy (few positions masked), high t is hard (mostly masked).
    double raw = 0.0;
    double cnt = 0.0;
    uint32_t acc_correct = 0;
    uint32_t acc_total = 0;
    uint32_t bucket_correct[3] = {0, 0, 0};
    uint32_t bucket_total[3] = {0, 0, 0};
    int n_batch = 8;
    for (int it = 0; it < n_batch; it = it + 1) {
        make_batch(
            &cp,
            cfg.batch,
            cfg.seqlen,
            sc.ids.ptr,
            sc.target.ptr,
            sc.mask.ptr,
            sc.weight.ptr
        );
        forward_logits(m, &sc);
        float *loss = sc.loss.ptr;
        float *mask = sc.mask.ptr;
        float *weight = sc.weight.ptr;
        uint32_t *ids = sc.ids.ptr;
        uint32_t *target = sc.target.ptr;
        float *logits = sc.logits_save.ptr;
        for (uint32_t i = 0; i < cfg.batch * cfg.seqlen; i = i + 1) {
            // CE is per-token; convert to per-byte by spreading each token's loss
            // over its byte length, so bits/byte stays comparable across vocabs.
            uint32_t blen = cp.tok.entries[target[i]].len;
            raw = raw + loss[i] / (weight[i] + 1e-9);
            cnt = cnt + mask[i] * (double)blen;
            if (ids[i] != data_mask_id) {
                continue;
            }
            uint32_t best = argmax_row(logits + i * cfg.vocab, cfg.vocab);
            int hit = (best == target[i]) ? 1 : 0;
            acc_total = acc_total + 1;
            acc_correct = acc_correct + hit;
            float t = 1.0f / (weight[i] + 1e-9);
            int bucket = (t < 0.33f) ? 0 : (t < 0.66f ? 1 : 2);
            bucket_total[bucket] = bucket_total[bucket] + 1;
            bucket_correct[bucket] = bucket_correct[bucket] + hit;
        }
    }
    double bpb = (raw / (cnt + 1e-6)) / 0.6931472;
    double acc_low = 100.0 * bucket_correct[0] / (bucket_total[0] + 1);
    double acc_mid = 100.0 * bucket_correct[1] / (bucket_total[1] + 1);
    double acc_high = 100.0 * bucket_correct[2] / (bucket_total[2] + 1);
    double acc_all = 100.0 * acc_correct / (acc_total + 1);
    printf("checkpoint step %u: held-out bits/byte = %.4f  (random = 8.0)\n", step, bpb);
    printf("accuracy by mask rate: low(<33%%)=%.1f%%  mid=%.1f%%  high(>66%%)=%.1f%%\n", acc_low, acc_mid, acc_high);
    printf("masked-fill accuracy over %d batches: %u/%u = %.1f%% (random 0.4%%)\n\n", n_batch, acc_correct, acc_total, acc_all);

    // qualitative: for the first sequence, show actual vs the model's prediction at
    // masked positions (a blank means the model already saw that position).
    make_batch(
        &cp,
        cfg.batch,
        cfg.seqlen,
        sc.ids.ptr,
        sc.target.ptr,
        sc.mask.ptr,
        sc.weight.ptr
    );
    forward_logits(m, &sc);
    uint32_t *ids = sc.ids.ptr;
    uint32_t *target = sc.target.ptr;
    float *logits = sc.logits_save.ptr;
    uint32_t correct = 0;
    uint32_t total = 0;
    printf("actual : ");
    for (uint32_t s = 0; s < cfg.seqlen; s = s + 1) {
        tok_entry_t *e = &cp.tok.entries[target[s]];
        fwrite(e->bytes, 1, e->len, stdout);
    }
    printf("\nmodel  : ");
    for (uint32_t s = 0; s < cfg.seqlen; s = s + 1) {
        if (ids[s] != data_mask_id) {
            tok_entry_t *e = &cp.tok.entries[target[s]];
            for (uint32_t k = 0; k < e->len; k = k + 1) {
                putchar(' ');
            }
            continue;
        }
        uint32_t best = argmax_row(logits + s * cfg.vocab, cfg.vocab);
        tok_entry_t *e = &cp.tok.entries[best];
        fwrite(e->bytes, 1, e->len, stdout);
        total = total + 1;
        if (best == target[s]) {
            correct = correct + 1;
        }
    }
    double sample_acc = total ? 100.0 * correct / total : 0.0;
    printf("\n\nmasked-position accuracy: %u/%u = %.1f%%\n", correct, total, sample_acc);
    return 0;
}
