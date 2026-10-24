#ifndef MDA_GENERATE_H
#define MDA_GENERATE_H

#include <stdbool.h>
#include <stdint.h>

#include "vkc.h"
#include "model.h"
#include "fwd.h"

#define MDA_GENERATE_ROUNDS 8
#define MDA_GENERATE_MAX_DRAFT 16

typedef uint32_t mda_generate_order_t;
typedef struct mda_generate_policy_t mda_generate_policy_t;
typedef struct mda_generate_spec_t mda_generate_spec_t;
typedef struct mda_generate_stats_t mda_generate_stats_t;
typedef struct mda_generate_ctx_t mda_generate_ctx_t;

enum {
    MDA_GENERATE_RANDOM,
    MDA_GENERATE_LTR,
    MDA_GENERATE_CONF,
    MDA_GENERATE_SPEC,
};

struct mda_generate_policy_t {
    mda_generate_order_t order;
    uint32_t draft;
    float remask;
};

struct mda_generate_spec_t {
    float *q;
    float *p_rows;
    uint32_t drafts[MDA_GENERATE_MAX_DRAFT];
    uint32_t n;
    uint32_t k;
    uint32_t accepted;
    uint32_t proposed;
};

struct mda_generate_stats_t {
    uint32_t *counts;
    double nll;
    double nll_bytes;
    double nll_tokens;
    double rep;
    double seconds;
    uint32_t tokens;
    uint32_t seqs;
    uint32_t forwards;
    uint32_t accepted;
    uint32_t proposed;
};

struct mda_generate_ctx_t {
    mda_model_t *m;
    mda_fwd_scratch_t *sc;
    mda_fwd_cache_t *cache;
    mda_generate_spec_t *specs;
    uint32_t *done;
    uint32_t *open;
    float *tmp;
    mda_vkc_buf_t stage;
    mda_generate_policy_t policy;
    uint32_t len;
    uint32_t nseq;
    uint32_t mask_id;
    uint32_t newline;
    uint32_t steps;
};

#endif
