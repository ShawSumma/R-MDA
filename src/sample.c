#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "sample.h"
#include "vkc.h"
#include "util.h"
#include "tokenizer.h"
#include "data.h"
#include "model.h"
#include "fwd.h"
#include "ckpt.h"

double mda_sample_pct(uint32_t a, uint32_t b) {
    if (b == 0) {
        return 0.0;
    } else {
        return 100.0 * a / b;
    }
}

uint32_t mda_sample_bucket(float t) {
    if (t < 0.33f) {
        return 0;
    } else if (t < 0.66f) {
        return 1;
    } else {
        return 2;
    }
}

void mda_sample_count(mda_sample_stats_t *st, const mda_data_corpus_t *cp, mda_fwd_scratch_t *sc) {
    mda_data_batch_t *b = &sc->batch;
    float *loss = sc->loss.ptr;
    uint32_t *tok = sc->tok.ptr;
    st->bytes += mda_data_masked_bytes(cp, b);
    for (uint32_t i = 0; i < b->batch * b->seqlen; i += 1) {
        st->ce += loss[i] / b->weight[i];
        if (b->mask[i] != 0.0f) {
            uint32_t bucket = mda_sample_bucket(1.0f / b->weight[i]);
            st->masked += 1.0;
            st->total += 1;
            st->bucket_total[bucket] += 1;
            if (tok[i] == b->target[i]) {
                st->correct += 1;
                st->bucket_correct[bucket] += 1;
            }
        }
    }
}

void mda_sample_print_stats(const mda_sample_stats_t *st, uint32_t step, uint32_t vocab) {
    double bpb = st->ce / st->bytes / 0.6931472;
    double bpb_random = st->masked * log2((double)vocab) / st->bytes;
    printf("checkpoint step %u: held-out bits/byte %.4f (uniform guess %.2f)\n", step, bpb, bpb_random);
    printf("accuracy by mask rate: low (<33%%) %.1f%%, mid %.1f%%, high (>66%%) %.1f%%\n", mda_sample_pct(st->bucket_correct[0], st->bucket_total[0]), mda_sample_pct(st->bucket_correct[1], st->bucket_total[1]), mda_sample_pct(st->bucket_correct[2], st->bucket_total[2]));
    printf("masked-fill accuracy over %u batches: %u/%u = %.1f%% (uniform guess %.4f%%)\n\n", MDA_SAMPLE_BATCHES, st->correct, st->total, mda_sample_pct(st->correct, st->total), 100.0 / vocab);
}

void mda_sample_print_sequence(const mda_data_corpus_t *cp, mda_fwd_scratch_t *sc) {
    mda_data_batch_t *b = &sc->batch;
    uint32_t *tok = sc->tok.ptr;
    uint32_t correct = 0;
    uint32_t total = 0;
    printf("actual:\n");
    for (uint32_t s = 0; s < b->seqlen; s += 1) {
        mda_tokenizer_entry_t *e = &cp->tok.entries[b->target[s]];
        fwrite(e->bytes, 1, e->len, stdout);
    }
    printf("\nmodel:\n");
    for (uint32_t s = 0; s < b->seqlen; s += 1) {
        if (b->ids[s] != cp->mask_id) {
            for (uint32_t k = 0; k < cp->tok.entries[b->target[s]].len; k += 1) {
                putchar(' ');
            }
        } else {
            mda_tokenizer_entry_t *e = &cp->tok.entries[tok[s]];
            fwrite(e->bytes, 1, e->len, stdout);
            total += 1;
            if (tok[s] == b->target[s]) {
                correct += 1;
            }
        }
    }
    printf("\n\nmasked-position accuracy: %u/%u = %.1f%%\n", correct, total, mda_sample_pct(correct, total));
}

bool mda_sample_eval(mda_model_t *m, const mda_data_corpus_t *cp, mda_fwd_scratch_t *sc, uint32_t step) {
    mda_sample_stats_t st = {
        .ce = 0.0,
    };
    for (uint32_t it = 0; it < MDA_SAMPLE_BATCHES && m->c->fail == false; it += 1) {
        mda_data_batch(&sc->batch, cp, MDA_DATA_SPLIT_EVAL, it);
        mda_fwd_run(m, sc);
        if (m->c->fail == false) {
            mda_sample_count(&st, cp, sc);
        }
    }
    if (m->c->fail == false) {
        mda_sample_print_stats(&st, step, m->cfg.vocab);
        mda_data_batch(&sc->batch, cp, MDA_DATA_SPLIT_EVAL, MDA_SAMPLE_BATCHES);
        mda_fwd_run(m, sc);
    }
    if (m->c->fail == false) {
        mda_sample_print_sequence(cp, sc);
    }
    return m->c->fail == false;
}

bool mda_sample_scratch(mda_model_t *m, const mda_data_corpus_t *cp, mda_fwd_scratch_t *sc, const char *path) {
    mda_ckpt_res_t loaded = mda_ckpt_load(m, mda_tokenizer_hash(&cp->tok), path, false);
    if (loaded.ok == false) {
        return false;
    } else {
        if (loaded.step == 0) {
            fprintf(stderr, "no checkpoint: evaluating the random init\n");
        }
        return mda_sample_eval(m, cp, sc, loaded.step);
    }
}

bool mda_sample_model(mda_vkc_t *c, const mda_data_corpus_t *cp, const char *path) {
    mda_model_t *m = mda_model_init(c, mda_model_cfg(cp->tok.count));
    if (m == NULL) {
        return false;
    } else {
        mda_fwd_scratch_t sc = mda_fwd_alloc(m, 1);
        if (c->fail) {
            return false;
        } else {
            return mda_sample_scratch(m, cp, &sc, path);
        }
    }
}

bool mda_sample_device(mda_vkc_t *c, const char *path) {
    mda_data_corpus_t cp = mda_data_load(MDA_TOKENIZER_CORPUS_PATH);
    if (cp.ids == NULL) {
        return false;
    } else {
        return mda_sample_model(c, &cp, path);
    }
}

bool mda_sample_main(uint32_t argc, char **argv) {
    const char *path = mda_util_arg(argc, argv, 1, MDA_CKPT_PATH);
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_sample_device(&c, path);
    }
}

int main(int argc, char **argv) {
    bool ok = mda_sample_main((uint32_t)argc, argv);
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
