// data.h -- corpus loader + masked-diffusion batch builder over Unigram subword
// tokens (see tokenizer.h). The corpus is tokenized once at load into a flat array
// of token ids; batches sample token windows and mask token positions. No Python.
// Vocab is the trained subword set plus one MASK id at index DATA_VOCAB.
#ifndef DATA_H
#define DATA_H

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// the vocab size and mask id are set when the corpus is tokenized; the model reads
// DATA_VOCAB to size its embedding and head. DATA_MASK_ID is the absorbing token.
extern uint32_t data_vocab;
extern uint32_t data_mask_id;

typedef struct corpus corpus_t;

struct corpus {
    uint32_t *ids;
    size_t len;
    tokenizer_t tok;
};

uint32_t data_vocab = 0;
uint32_t data_mask_id = 0;

static corpus_t corpus_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *bytes = malloc(n);
    fread(bytes, 1, n, f);
    fclose(f);

    tokenizer_t tok = tokenizer_load("models/vocab.txt");
    data_vocab = tok.count;
    data_mask_id = tok.count;

    // tokenize the whole corpus once. Encoding 60+ MB through Viterbi is slow if
    // done in one call (the back-pointer arrays scale with length), so segment it
    // in fixed windows; windows are large enough that boundary effects are tiny.
    uint32_t *ids = malloc((size_t)n * sizeof(uint32_t));
    size_t total = 0;
    uint32_t window = 8192;
    uint32_t *tmp = malloc(window * sizeof(uint32_t));
    size_t pos = 0;
    while (pos < (size_t)n) {
        uint32_t chunk = window;
        if (pos + chunk > (size_t)n) {
            chunk = (uint32_t)((size_t)n - pos);
        }
        uint32_t got = tokenizer_encode(&tok, bytes + pos, chunk, tmp, window);
        for (uint32_t i = 0; i < got; i = i + 1) {
            ids[total] = tmp[i];
            total = total + 1;
        }
        pos = pos + chunk;
    }
    free(tmp);
    free(bytes);

    fprintf(stderr, "tokenized corpus: %ld bytes -> %zu tokens (%.2f bytes/token), vocab %u\n", n, total, (double)n / total, tok.count);

    corpus_t cp = {
        .ids = ids,
        .len = total,
        .tok = tok,
    };
    return cp;
}

static float frand(void) {
    return (float)((rand() % 1000000) / 1000000.0);
}

// Build one masked-diffusion batch over token positions.
//   ids_in[b*seqlen] : corrupted token ids fed to the denoiser (mask id where masked)
//   target[b*seqlen] : the true token id at each position
//   mask[b*seqlen]   : 1.0 where the position was masked (loss is counted there)
//   weight[b*seqlen] : 1/t loss weight (per sequence, broadcast over positions)
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
            uint32_t token = cp->ids[start + s];
            target[pos] = token;
            weight[pos] = w;
            if (frand() < t) {
                ids_in[pos] = data_mask_id;
                mask[pos] = 1.0f;
            } else {
                ids_in[pos] = token;
                mask[pos] = 0.0f;
            }
        }
    }
}

#endif
