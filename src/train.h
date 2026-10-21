#ifndef MDA_TRAIN_H
#define MDA_TRAIN_H

#include <stdbool.h>
#include <stdint.h>

#define MDA_TRAIN_CKPT_INTERVAL 120.0
#define MDA_TRAIN_CLIP 1.0f
#define MDA_TRAIN_MAX_SKIPS 100
#define MDA_TRAIN_WARMUP 50

typedef struct mda_train_args_t mda_train_args_t;
typedef struct mda_train_log_t mda_train_log_t;

struct mda_train_args_t {
    const char *ckpt_path;
    const char *mode;
    float lr_peak;
    uint32_t stop;
    uint32_t accum;
    uint32_t sched;
    bool muon;
};

struct mda_train_log_t {
    double ema;
    double ema_bpb;
    double t0;
    double t_ckpt;
    uint32_t start;
    uint32_t n_ema;
    uint32_t n_clipped;
    uint32_t n_skipped;
    uint32_t skip_run;
    uint32_t n_logged;
};

#endif
