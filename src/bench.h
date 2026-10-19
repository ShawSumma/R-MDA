#ifndef MDA_BENCH_H
#define MDA_BENCH_H

#include <stdint.h>

#define MDA_BENCH_CORPUS_LEN (1u << 20)
#define MDA_BENCH_ITERS 5
#define MDA_BENCH_PROF_BATCHES 3
#define MDA_BENCH_ROUNDS 7
#define MDA_BENCH_VOCAB 16000

typedef uint32_t mda_bench_phase_t;

enum {
    MDA_BENCH_FORWARD,
    MDA_BENCH_GRAD,
    MDA_BENCH_STEP,
};

#endif
