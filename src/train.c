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
        printf("step %u lr %.2e loss %.4f ema %.4f bits/byte %.4f ema %.4f gnorm %.3f clip %.0f%% skip %u %.3f s/step\n", s, lr, loss, log->ema, bpb, log->ema_bpb, tr->gnorm, 100.0 * log->n_clipped / log->n_logged, log->n_skipped, per_step);
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

double mda_train_micros(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, uint32_t s) {
    double bytes = 0.0;
    for (uint32_t i = 0; i < tr->accum; i += 1) {
        mda_data_batch(&tr->sc.batch, cp, MDA_DATA_SPLIT_TRAIN, (uint64_t)(s - 1) * tr->accum + i + 1);
        mda_step_micro(tr, i);
        bytes += mda_data_masked_bytes(cp, &tr->sc.batch);
    }
    return bytes;
}

void mda_train_step(mda_train_log_t *log, mda_step_trainer_t *tr, const mda_data_corpus_t *cp, uint64_t vocab_hash, mda_train_args_t args, uint32_t s) {
    float lr = mda_train_lr(args, s);
    mda_step_begin(tr, lr, s);
    double bytes = mda_train_micros(tr, cp, s);
    float loss = (float)tr->loss;
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

bool mda_train_steps(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, uint64_t vocab_hash, mda_train_args_t args, uint32_t start) {
    mda_train_log_t log = {
        .t0 = mda_util_now(),
        .t_ckpt = mda_util_now(),
        .start = start,
    };
    uint32_t max_steps = mda_train_max_steps(args.stop);
    uint32_t s = start + 1;
    while (s <= max_steps && tr->m->c->fail == false && log.skip_run < MDA_TRAIN_MAX_SKIPS) {
        mda_train_step(&log, tr, cp, vocab_hash, args, s);
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

bool mda_train_resume(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, mda_train_args_t args) {
    uint64_t vocab_hash = mda_tokenizer_hash(&cp->tok);
    mda_ckpt_res_t loaded = mda_ckpt_load(tr->m, vocab_hash, args.ckpt_path, true);
    if (loaded.ok == false) {
        return false;
    } else {
        if (loaded.step != 0) {
            fprintf(stderr, "resumed from checkpoint at step %u\n", loaded.step);
        }
        if (args.stop != 0) {
            fprintf(stderr, "lr %g, decaying to 0 over steps %u..%u\n", args.lr_peak, mda_train_decay_from(args.stop), args.stop);
        } else {
            fprintf(stderr, "lr %g\n", args.lr_peak);
        }
        return mda_train_steps(tr, cp, vocab_hash, args, loaded.step);
    }
}

bool mda_train_trainer(mda_step_trainer_t *tr, const mda_data_corpus_t *cp, mda_train_args_t args) {
    mda_model_cfg_t cfg = tr->m->cfg;
    if (tr->m->c->fail) {
        return false;
    } else {
        fprintf(stderr, "params %.2f M, d=%u L=%u H=%u seq=%u batch=%u accum=%u muon=%u\n", (double)mda_model_param_count(tr->m) / 1e6, cfg.d, cfg.n_layer, cfg.n_head, cfg.seqlen, cfg.batch, tr->accum, tr->muon);
        return mda_train_resume(tr, cp, args);
    }
}

bool mda_train_model(mda_vkc_t *c, const mda_data_corpus_t *cp, mda_train_args_t args) {
    mda_model_cfg_t cfg = mda_model_cfg(cp->tok.count);
    mda_model_t *m = mda_model_init(c, cfg);
    if (m == NULL) {
        return false;
    } else {
        mda_step_trainer_t tr = mda_step_init(m, MDA_TRAIN_CLIP, args.accum, args.muon);
        return mda_train_trainer(&tr, cp, args);
    }
}

bool mda_train_device(mda_vkc_t *c, mda_train_args_t args) {
    mda_data_corpus_t cp = mda_data_load(MDA_TOKENIZER_CORPUS_PATH);
    if (cp.ids == NULL) {
        return false;
    } else {
        return mda_train_model(c, &cp, args);
    }
}

bool mda_train_main(uint32_t argc, char **argv) {
    mda_train_args_t args = {
        .ckpt_path = mda_util_arg(argc, argv, 2, MDA_CKPT_PATH),
        .lr_peak = strtof(mda_util_arg(argc, argv, 1, "1e-4"), NULL),
        .stop = (uint32_t)strtoul(mda_util_arg(argc, argv, 3, "0"), NULL, 10),
        .accum = (uint32_t)strtoul(mda_util_arg(argc, argv, 4, "1"), NULL, 10),
        .muon = strcmp(mda_util_arg(argc, argv, 5, "adamw"), "muon") == 0,
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
