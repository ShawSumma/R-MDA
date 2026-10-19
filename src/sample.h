#ifndef MDA_SAMPLE_H
#define MDA_SAMPLE_H

#include <stdint.h>

#define MDA_SAMPLE_BATCHES 128

typedef struct mda_sample_stats_t mda_sample_stats_t;

struct mda_sample_stats_t {
    double ce;
    double bytes;
    double masked;
    uint32_t bucket_correct[3];
    uint32_t bucket_total[3];
    uint32_t correct;
    uint32_t total;
};

#endif
