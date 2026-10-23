#ifndef MDA_DATA_H
#define MDA_DATA_H

#include <stddef.h>
#include <stdint.h>

#include "tokenizer.h"

#define MDA_DATA_EVAL_FRAC 0.01
#define MDA_DATA_IDS_MAGIC 0x3153444941444d52ull
#define MDA_DATA_T_EPS 1e-4f
#define MDA_DATA_BELL_MEAN 0.5f
#define MDA_DATA_BELL_STD 0.1f
#define MDA_DATA_TOK_WINDOW 8192

typedef uint32_t mda_data_split_t;
typedef uint32_t mda_data_sched_t;
typedef struct mda_data_corpus_t mda_data_corpus_t;
typedef struct mda_data_batch_t mda_data_batch_t;
typedef struct mda_data_ids_hdr_t mda_data_ids_hdr_t;
typedef struct mda_data_job_t mda_data_job_t;

enum {
    MDA_DATA_SPLIT_TRAIN,
    MDA_DATA_SPLIT_EVAL,
};

enum {
    MDA_DATA_SCHED_BELL,
    MDA_DATA_SCHED_ELBO,
    MDA_DATA_SCHED_AR,
};

struct mda_data_corpus_t {
    uint32_t *ids;
    size_t len;
    size_t eval_start;
    mda_tokenizer_t tok;
    uint32_t mask_id;
};

struct mda_data_batch_t {
    uint32_t *ids;
    uint32_t *target;
    float *mask;
    float *weight;
    uint32_t batch;
    uint32_t seqlen;
    uint32_t nclean;
};

struct mda_data_ids_hdr_t {
    uint64_t magic;
    uint64_t vocab_hash;
    uint64_t corpus_bytes;
    uint64_t n;
};

struct mda_data_job_t {
    const mda_tokenizer_t *tok;
    const char *bytes;
    size_t lo;
    size_t hi;
    uint32_t *ids;
    size_t n;
};

mda_data_corpus_t mda_data_load(const char *path);
void mda_data_batch(mda_data_batch_t *b, const mda_data_corpus_t *cp, mda_data_split_t split, mda_data_sched_t sched, uint64_t key);
double mda_data_bytes(const mda_data_corpus_t *cp, const mda_data_batch_t *b);
double mda_data_masked_bytes(const mda_data_corpus_t *cp, const mda_data_batch_t *b);

#endif
