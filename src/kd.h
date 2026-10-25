#ifndef MDA_KD_H
#define MDA_KD_H

#include <stdbool.h>
#include <stdint.h>

#include "tokenizer.h"

#define MDA_KD_MAGIC 0x3144424b41444d52ull
#define MDA_KD_PATH "models/kd.bin"
#define MDA_KD_DEPTH 16
#define MDA_KD_K 8
#define MDA_KD_ENTRY (MDA_KD_DEPTH * MDA_KD_K)
#define MDA_KD_QWEN_SPECIAL 248044

typedef struct mda_kd_hdr_t mda_kd_hdr_t;
typedef struct mda_kd_window_t mda_kd_window_t;
typedef struct mda_kd_file_t mda_kd_file_t;
typedef struct mda_kd_rec_t mda_kd_rec_t;

struct mda_kd_hdr_t {
    uint64_t magic;
    uint64_t n_windows;
    uint32_t from;
    uint32_t to;
    uint32_t accum;
    uint32_t nclean;
    uint32_t seqlen;
    uint32_t depth;
    uint32_t k;
    uint32_t batch;
};

struct mda_kd_window_t {
    uint8_t *bytes;
    uint32_t *ends;
    uint32_t n_bytes;
};

struct mda_kd_file_t {
    uint8_t *map;
    const uint64_t *index;
    uint64_t size;
    mda_kd_hdr_t hdr;
};

struct mda_kd_rec_t {
    const uint32_t *entries;
    uint32_t n_bytes;
};

float mda_kd_prob(uint32_t entry);
uint32_t *mda_kd_vocab(const mda_tokenizer_t *tok, uint32_t vocab, uint32_t first, uint32_t end);
uint32_t mda_kd_window_cap(uint32_t seqlen);
mda_kd_window_t mda_kd_window_alloc(uint32_t seqlen);
void mda_kd_window_fill(mda_kd_window_t *w, const mda_tokenizer_t *tok, const uint32_t *row_ids, uint32_t seqlen, uint32_t last);
uint64_t mda_kd_window_index(const mda_kd_hdr_t *h, uint32_t step, uint32_t micro, uint32_t row);
void mda_kd_cont(uint32_t *dst, const uint8_t *bytes, uint32_t n_bytes, uint32_t at, uint32_t nd);
mda_kd_file_t mda_kd_open(const char *path);
mda_kd_rec_t mda_kd_record(const mda_kd_file_t *f, uint64_t w);

#endif
