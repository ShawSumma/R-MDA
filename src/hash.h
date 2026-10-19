#ifndef MDA_HASH_H
#define MDA_HASH_H

#include <stdint.h>

typedef uint64_t mda_hash_stream_t;

enum {
    MDA_HASH_INIT,
    MDA_HASH_BATCH,
    MDA_HASH_GEN,
    MDA_HASH_CORPUS,
    MDA_HASH_TEST,
};

uint64_t mda_hash_mix(uint64_t x);
uint64_t mda_hash_u64(mda_hash_stream_t s, uint64_t key, uint64_t i);
double mda_hash_unit(mda_hash_stream_t s, uint64_t key, uint64_t i);
float mda_hash_normal(mda_hash_stream_t s, uint64_t key, uint64_t i);

#endif
