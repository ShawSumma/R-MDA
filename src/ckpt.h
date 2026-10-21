#ifndef MDA_CKPT_H
#define MDA_CKPT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "model.h"

#define MDA_CKPT_MAGIC 0x37504b4341444d52ull
#define MDA_CKPT_OVERFLOW_PATH "models/ckpt-overflow.bin"
#define MDA_CKPT_PATH "models/ckpt.bin"

typedef struct mda_ckpt_hdr_t mda_ckpt_hdr_t;
typedef struct mda_ckpt_res_t mda_ckpt_res_t;

struct mda_ckpt_hdr_t {
    uint64_t magic;
    mda_model_cfg_t cfg;
    uint32_t step;
    uint64_t vocab_hash;
    uint64_t n_all;
};

struct mda_ckpt_res_t {
    uint32_t step;
    bool ok;
};

bool mda_ckpt_write(mda_model_t *m, uint64_t vocab_hash, uint32_t step, FILE *f);
mda_ckpt_res_t mda_ckpt_read(mda_model_t *m, uint64_t vocab_hash, FILE *f, bool moments, const char *path);
bool mda_ckpt_save(mda_model_t *m, uint64_t vocab_hash, uint32_t step, const char *path);
mda_ckpt_res_t mda_ckpt_load(mda_model_t *m, uint64_t vocab_hash, const char *path, bool moments);

#endif
