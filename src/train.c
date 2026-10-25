#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "train.h"
#include "vkc.h"
#include "util.h"
#include "tokenizer.h"
#include "data.h"
#include "model.h"
#include "step.h"
#include "ckpt.h"

uint32_t mda_train_max_steps(uint32_t stop) {
    if (stop != 0) {
        return stop;
    } else {
        return UINT32_MAX;
    }
}

uint32_t mda_train_decay_from(uint32_t stop) {
    uint32_t max_steps = mda_train_max_steps(stop);
    return max_steps - max_steps / 5;
}

float mda_train_lr(mda_train_args_t args, uint32_t s) {
    uint32_t decay_from = mda_train_decay_from(args.stop);
    if (s < MDA_TRAIN_WARMUP) {
        return args.lr_peak * (float)s / (float)MDA_TRAIN_WARMUP;
    } else if (args.stop != 0 && s > decay_from) {
        return args.lr_peak * (float)(args.stop - s) / (float)(args.stop - decay_from);
    } else {
        return args.lr_peak;
    }
}

void mda_train_record(mda_train_log_t *log, const mda_step_trainer_t *tr, float loss, double bpb) {
    log->n_logged += 1;
    if (tr->skipped) {
        log->skip_run += 1;
        log->n_skipped += 1;
    } else if (log->n_ema == 0) {
        log->skip_run = 0;
        log->ema = loss;
        log->ema_bpb = bpb;
        log->n_ema += 1;
    } else {
        log->skip_run = 0;
        log->ema = 0.95 * log->ema + 0.05 * loss;
        log->ema_bpb = 0.95 * log->ema_bpb + 0.05 * bpb;
        log->n_ema += 1;
    }
    if (tr->skipped == false && tr->gnorm > MDA_TRAIN_CLIP) {
        log->n_clipped += 1;
    }
}

void mda_train_report(mda_train_log_t *log, const mda_step_trainer_t *tr, uint32_t s, float lr, float loss, double bpb, double now) {
    if (s % 5 == 0 || s == log->start + 1) {
        double per_step = (now - log->t0) / (s - log->start);
        printf("step %u lr %.2e loss %.4f ema %.4f bits/byte %.4f ema %.4f gnorm %.3f clip %.0f%% skip %u kd %.4f student %.4f teacher %.4f %.3f s/step\n", s, lr, loss, log->ema, bpb, log->ema_bpb, tr->gnorm, 100.0 * log->n_clipped / log->n_logged, log->n_skipped, tr->kd_sum / fmax(tr->kd_n, 1.0), tr->kd_student_bits / fmax(tr->kd_bytes, 1.0), tr->kd_teacher_bits / fmax(tr->kd_teacher_bytes, 1.0), per_step);
        fflush(stdout);
        log->n_clipped = 0;
        log->n_skipped = 0;
        log->n_logged = 0;
    }
}

void mda_train_checkpoint(mda_train_log_t *log, mda_model_t *m, uint64_t vocab_hash, mda_train_args_t args, uint32_t s, double now) {
    if (now - log->t_ckpt >= MDA_TRAIN_CKPT_INTERVAL || s == args.stop) {
        bool saved = mda_ckpt_save(m, vocab_hash, s, args.ckpt_path);
        if (saved) {
            fprintf(stderr, "checkpoint at step %u, bpb %.3f\n", s, log->ema_bpb);
        }
        log->t_ckpt = now;
    }
}

mda_kd_rec_t mda_train_kd_rec(const mda_train_kd_t *kd, uint32_t s, uint32_t micro, uint32_t row) {
    mda_kd_rec_t none = {
        .entries = NULL,
        .n_bytes = 0,
    };
    if (s >= kd->f.hdr.from && s < kd->f.hdr.to) {
        return mda_kd_record(&kd->f, mda_kd_window_index(&kd->f.hdr, s, micro, row));
    } else {
        return none;
    }
}

void mda_train_kd_row(mda_step_trainer_t *tr, mda_train_kd_t *kd, mda_kd_rec_t rec, uint32_t c) {
    uint32_t seqlen = tr->m->cfg.seqlen;
    uint32_t *cont = tr->sc.kd_cont.ptr;
    uint32_t *tgt = tr->sc.kd_tgt.ptr;
    for (uint32_t p = 0; p < seqlen; p += 1) {
        uint32_t row = c * seqlen + p;
        uint32_t at = kd->w.ends[p];
        uint32_t nd = 0;
        if (rec.entries != NULL) {
            nd = mda_util_min_u32(kd->w.ends[p + 1] - at, MDA_KD_DEPTH);
        }
        mda_kd_cont(cont + (size_t)row * 5, kd->w.bytes, kd->w.n_bytes, at, nd);
        for (uint32_t l = 0; l < nd && at + l < rec.n_bytes; l += 1) {
            const uint32_t *e = rec.entries + (size_t)(at + l) * MDA_KD_K;
            float p = mda_kd_prob(e[0]);
            memcpy(tgt + ((size_t)row * MDA_KD_DEPTH + l) * MDA_KD_K, e, MDA_KD_K * sizeof(uint32_t));
            if (p > 0.0f) {
                tr->kd_teacher_bits -= log2(p);
                tr->kd_teacher_bytes += 1.0;
            }
        }
    }
}

void mda_train_kd_fill(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, mda_train_kd_t *kd, uint32_t s, uint32_t micro) {
    uint32_t seqlen = tr->m->cfg.seqlen;
    uint32_t nclean = mda_model_nclean(tr->m->cfg);
    const mda_data_batch_t *b = &tr->sc.batch;
    memset(tr->sc.kd_tgt.ptr, 0, (size_t)mda_fwd_kd_rows(tr->m) * MDA_KD_ENTRY * sizeof(uint32_t));
    for (uint32_t c = 0; c < nclean; c += 1) {
        mda_kd_window_fill(&kd->w, &cp->tok, b->ids + (size_t)c * seqlen, seqlen, b->target[(size_t)c * seqlen + seqlen - 1]);
        mda_train_kd_row(tr, kd, mda_train_kd_rec(kd, s, micro, c), c);
    }
}

double mda_train_micros(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, mda_train_kd_t *kd, mda_data_sched_t sched, uint32_t s) {
    double bytes = 0.0;
    for (uint32_t i = 0; i < tr->accum; i += 1) {
        mda_data_batch(&tr->sc.batch, cp, MDA_DATA_SPLIT_TRAIN, sched, (uint64_t)(s - 1) * tr->accum + i + 1);
        if (kd->on) {
            mda_train_kd_fill(tr, cp, kd, s, i);
        }
        mda_step_micro(tr, i);
        bytes += mda_data_masked_bytes(cp, &tr->sc.batch);
    }
    return bytes;
}

void mda_train_step(mda_train_log_t *log, mda_step_trainer_t *tr, const mda_data_corpus_t *cp, mda_train_kd_t *kd, uint64_t vocab_hash, mda_train_args_t args, uint32_t s) {
    float lr = mda_train_lr(args, s);
    mda_step_begin(tr, lr, s);
    double bytes = mda_train_micros(tr, cp, kd, args.sched, s);
    float loss = (float)tr->loss;
    if (tr->skipped && log->reported == false) {
        fprintf(stderr, "step %u skipped, the first in this run:\n", s);
        mda_step_print_nonfinite(tr);
        log->reported = true;
    }
    if (tr->m->c->fail == false) {
        double bpb = tr->ce_sum / bytes / 0.6931472;
        mda_train_record(log, tr, loss, bpb);
        double now = mda_util_now();
        if (log->skip_run < MDA_TRAIN_MAX_SKIPS) {
            mda_train_report(log, tr, s, lr, loss, bpb, now);
            mda_train_checkpoint(log, tr->m, vocab_hash, args, s, now);
        }
    }
}

bool mda_train_steps(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, mda_train_kd_t *kd, uint64_t vocab_hash, mda_train_args_t args, uint32_t start) {
    mda_train_log_t log = {
        .t0 = mda_util_now(),
        .t_ckpt = mda_util_now(),
        .start = start,
    };
    uint32_t max_steps = mda_train_max_steps(args.stop);
    uint32_t s = start + 1;
    while (s <= max_steps && tr->m->c->fail == false && log.skip_run < MDA_TRAIN_MAX_SKIPS) {
        mda_train_step(&log, tr, cp, kd, vocab_hash, args, s);
        s += 1;
    }
    if (log.skip_run == MDA_TRAIN_MAX_SKIPS) {
        fprintf(stderr, "step %u: %u steps in a row overflowed; stopping, weights in %s\n", s - 1, MDA_TRAIN_MAX_SKIPS, MDA_CKPT_OVERFLOW_PATH);
        mda_ckpt_save(tr->m, vocab_hash, s - 1, MDA_CKPT_OVERFLOW_PATH);
        return false;
    } else {
        return tr->m->c->fail == false;
    }
}

bool mda_train_resume(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, mda_train_kd_t *kd, mda_train_args_t args) {
    uint64_t vocab_hash = mda_tokenizer_hash(&cp->tok);
    mda_ckpt_res_t loaded = mda_ckpt_load(tr->m, vocab_hash, args.ckpt_path, true);
    if (loaded.ok == false) {
        return false;
    } else {
        if (loaded.found) {
            fprintf(stderr, "resumed from %s at step %u\n", args.ckpt_path, loaded.step);
        }
        if (args.stop != 0) {
            fprintf(stderr, "lr %g, decaying to 0 over steps %u..%u\n", args.lr_peak, mda_train_decay_from(args.stop), args.stop);
        } else {
            fprintf(stderr, "lr %g\n", args.lr_peak);
        }
        return mda_train_steps(tr, cp, kd, vocab_hash, args, loaded.step);
    }
}

void mda_train_kd_vocab(mda_step_trainer_t *tr, const mda_data_corpus_t *cp) {
    uint32_t *table = mda_kd_vocab(&cp->tok, tr->m->cfg.vocab, 1, cp->tok.count);
    mda_vkc_write(tr->m->c, &tr->sc.kd_toks, table);
    free(table);
}

bool mda_train_trainer(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, mda_train_kd_t *kd, mda_train_args_t args) {
    mda_model_cfg_t cfg = tr->m->cfg;
    if (kd->on) {
        mda_train_kd_vocab(tr, cp);
    }
    if (tr->m->c->fail) {
        return false;
    } else {
        fprintf(stderr, "params %.2f M, d=%u L=%u H=%u seq=%u batch=%u accum=%u muon=%u, %.2f GB of GPU memory\n", (double)mda_model_param_count(tr->m) / 1e6, cfg.d, cfg.n_layer, cfg.n_head, cfg.seqlen, cfg.batch, tr->accum, tr->muon, (double)tr->m->c->dev_bytes / 1e9);
        return mda_train_resume(tr, cp, kd, args);
    }
}

bool mda_train_kd_fits(const mda_train_kd_t *kd, mda_model_cfg_t cfg, mda_train_args_t args) {
    const mda_kd_hdr_t *h = &kd->f.hdr;
    bool ok = kd->on == false || (h->accum == args.accum && h->nclean == mda_model_nclean(cfg) && h->seqlen == cfg.seqlen && h->batch == cfg.batch);
    if (ok == false) {
        fprintf(stderr, "%s: KD targets for accum %u, %u clean rows of %u tokens, batch %u do not fit this run\n", args.kd_path, h->accum, h->nclean, h->seqlen, h->batch);
    } else if (kd->on) {
        fprintf(stderr, "%s: KD targets for steps %u..%u, weight %g\n", args.kd_path, h->from, h->to, args.kd_weight);
    }
    return ok;
}

bool mda_train_model(mda_vkc_t *c, const mda_data_corpus_t *cp, mda_train_kd_t *kd, mda_train_args_t args) {
    mda_model_cfg_t cfg = mda_ckpt_shape(args.ckpt_path, mda_model_cfg_mode(cp->tok.count, args.mode, args.batch));
    mda_model_t *m = mda_model_init(c, cfg);
    if (m == NULL || mda_model_set_dec(m, mda_model_dec_of(args.mode)) == false || mda_train_kd_fits(kd, cfg, args) == false) {
        return false;
    } else {
        m->causal = args.sched == MDA_DATA_SCHED_AR;
        m->kd = kd->on;
        m->kd_weight = args.kd_weight;
        mda_step_trainer_t tr = mda_step_init(m, MDA_TRAIN_CLIP, args.accum, args.muon);
        return mda_train_trainer(&tr, cp, kd, args);
    }
}

mda_train_kd_t mda_train_kd_open(const char *path, uint32_t seqlen) {
    mda_train_kd_t kd;
    memset(&kd, 0, sizeof(kd));
    if (path != NULL) {
        kd.f = mda_kd_open(path);
        kd.on = kd.f.map != NULL;
        kd.w = mda_kd_window_alloc(seqlen);
    }
    return kd;
}

bool mda_train_device(mda_vkc_t *c, mda_train_args_t args) {
    mda_data_corpus_t cp = mda_data_load(MDA_TOKENIZER_CORPUS_PATH);
    mda_train_kd_t kd = mda_train_kd_open(args.kd_path, mda_model_cfg(cp.tok.count).seqlen);
    if (cp.ids == NULL || (args.kd_path != NULL && kd.on == false)) {
        return false;
    } else {
        return mda_train_model(c, &cp, &kd, args);
    }
}

mda_data_sched_t mda_train_sched(const char *name) {
    if (strcmp(name, "ar") == 0) {
        return MDA_DATA_SCHED_AR;
    } else {
        return MDA_DATA_SCHED_BELL;
    }
}

bool mda_train_main(uint32_t argc, char **argv) {
    mda_train_args_t args = {
        .ckpt_path = mda_util_arg(argc, argv, 2, MDA_CKPT_PATH),
        .lr_peak = strtof(mda_util_arg(argc, argv, 1, "3e-4"), NULL),
        .stop = (uint32_t)strtoul(mda_util_arg(argc, argv, 3, "0"), NULL, 10),
        .accum = (uint32_t)strtoul(mda_util_arg(argc, argv, 4, "2"), NULL, 10),
        .batch = (uint32_t)strtoul(mda_util_arg(argc, argv, 7, "8"), NULL, 10),
        .muon = strcmp(mda_util_arg(argc, argv, 5, "muon"), "muon") == 0,
        .mode = mda_util_arg(argc, argv, 6, "diffusion"),
        .sched = mda_train_sched(mda_util_arg(argc, argv, 6, "diffusion")),
        .kd_path = mda_util_arg(argc, argv, 8, NULL),
        .kd_weight = strtof(mda_util_arg(argc, argv, 9, "1"), NULL),
    };
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_train_device(&c, args);
    }
}

int main(int argc, char **argv) {
    bool ok = mda_train_main((uint32_t)argc, argv);
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
