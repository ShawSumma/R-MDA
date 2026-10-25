#ifndef MDA_TRAIN_H
#define MDA_TRAIN_H

#include <stdbool.h>
#include <stdint.h>

#include "kd.h"

#define MDA_TRAIN_CKPT_INTERVAL 1800.0
#define MDA_TRAIN_CLIP 1.0f
#define MDA_TRAIN_MAX_SKIPS 100
#define MDA_TRAIN_WARMUP 50

typedef struct mda_train_args_t mda_train_args_t;
typedef struct mda_train_log_t mda_train_log_t;
typedef struct mda_train_kd_t mda_train_kd_t;

struct mda_train_args_t {
    const char *ckpt_path;
    const char *mode;
    const char *kd_path;
    float lr_peak;
    float kd_weight;
    uint32_t stop;
    uint32_t accum;
    uint32_t batch;
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
    bool reported;
};

struct mda_train_kd_t {
    mda_kd_file_t f;
    mda_kd_window_t w;
    bool on;
};

#endif
