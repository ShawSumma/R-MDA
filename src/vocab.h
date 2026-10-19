#ifndef MDA_VOCAB_H
#define MDA_VOCAB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tokenizer.h"

#define MDA_VOCAB_EM_CHUNK 4096
#define MDA_VOCAB_EM_ITERS 3
#define MDA_VOCAB_EM_SAMPLE_BYTES (16 * 1024 * 1024)
#define MDA_VOCAB_NEG_INF (-1e30)
#define MDA_VOCAB_PRUNE_KEEP 0.75
#define MDA_VOCAB_SAMPLE_PIECE (1024 * 1024)
#define MDA_VOCAB_SEED_KEEP 200000
#define MDA_VOCAB_SEED_MAX_LEN 16
#define MDA_VOCAB_SEED_SAMPLE_BYTES (64 * 1024 * 1024)
#define MDA_VOCAB_TARGET 16000

_Static_assert(MDA_VOCAB_SEED_MAX_LEN <= 16 && MDA_VOCAB_SEED_MAX_LEN <= MDA_TOKENIZER_MAX_LEN, "keys must fit 16 bytes and the runtime");

typedef struct mda_vocab_text_t mda_vocab_text_t;
typedef struct mda_vocab_cand_t mda_vocab_cand_t;
typedef struct mda_vocab_cands_t mda_vocab_cands_t;
typedef struct mda_vocab_key_t mda_vocab_key_t;
typedef struct mda_vocab_map_t mda_vocab_map_t;

struct mda_vocab_text_t {
    uint8_t *bytes;
    size_t len;
};

struct mda_vocab_cand_t {
    double logp;
    double expected;
    double score;
    char bytes[MDA_VOCAB_SEED_MAX_LEN];
    uint32_t len;
};

struct mda_vocab_cands_t {
    mda_vocab_cand_t *v;
    uint32_t n;
};

struct mda_vocab_key_t {
    uint64_t lo;
    uint64_t hi;
};

struct mda_vocab_map_t {
    mda_vocab_key_t *key;
    uint32_t *val;
    uint8_t *len;
    uint32_t cap;
    uint32_t used;
};

mda_vocab_text_t mda_vocab_sample(const char *path, size_t n);
mda_vocab_cands_t mda_vocab_seed(const mda_vocab_text_t *t);
void mda_vocab_em(mda_vocab_cands_t *c, const mda_vocab_text_t *t);
void mda_vocab_prune(mda_vocab_cands_t *c, uint32_t target);
bool mda_vocab_write(const char *path, const mda_vocab_cands_t *c);

#endif
