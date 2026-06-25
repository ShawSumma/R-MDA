// data.h -- byte-level corpus loader + masked-diffusion batch builder. No Python.
// Vocab is the 256 byte values plus one MASK id at index 256.
#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DATA_VOCAB 256
#define DATA_MASK_ID 256

typedef struct corpus corpus_t;

struct corpus {
    unsigned char *bytes;
    size_t len;
};

static corpus_t corpus_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(n);
    fread(b, 1, n, f);
    fclose(f);
    return (corpus_t) {
        .bytes = b,
        .len = (size_t)n,
    };
}

static float frand(void) {
    return (float)((rand() % 1000000) / 1000000.0);
}

// Build one masked-diffusion batch.
//   ids_in[b*seqlen]   : corrupted token ids fed to the denoiser (mask id where masked)
//   target[b*seqlen]   : the true byte at each position
//   mask[b*seqlen]     : 1.0 where the position was masked (loss is counted there)
//   weight[b*seqlen]   : 1/t loss weight (per sequence, broadcast over positions)
// t is sampled per sequence in [t_min, 1]; each token masked independently w.p. t.
static void make_batch(
    const corpus_t *cp,
    uint32_t batch,
    uint32_t seqlen,
    uint32_t *ids_in,
    uint32_t *target,
    float *mask,
    float *weight) {
    float t_min = 0.02f;
    for (uint32_t b = 0; b < batch; b = b + 1) {
        size_t start = (size_t)(frand() * (float)(cp->len - seqlen - 1));
        float t = t_min + (1.0f - t_min) * frand();
        // 1/t loss weight, uncapped; the gradient clip handles the variance.
        float w = 1.0f / t;
        for (uint32_t s = 0; s < seqlen; s = s + 1) {
            uint32_t pos = b * seqlen + s;
            unsigned char byte = cp->bytes[start + s];
            target[pos] = byte;
            weight[pos] = w;
            if (frand() < t) {
                ids_in[pos] = DATA_MASK_ID;
                mask[pos] = 1.0f;
            } else {
                ids_in[pos] = byte;
                mask[pos] = 0.0f;
            }
        }
    }
}

#endif
