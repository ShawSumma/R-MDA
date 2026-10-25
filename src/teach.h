#ifndef MDA_TEACH_H
#define MDA_TEACH_H

#include <stdint.h>
#include <stdio.h>

#include "vkc.h"
#include "data.h"
#include "kd.h"
#include "qwen.h"

#define MDA_TEACH_SEQ 384
#define MDA_TEACH_CHUNK_WINDOWS 32
#define MDA_TEACH_HEAD_WINDOWS 4
#define MDA_TEACH_MAX_BYTES 8192

typedef struct mda_teach_args_t mda_teach_args_t;
typedef struct mda_teach_t mda_teach_t;

struct mda_teach_args_t {
    const char *out;
    const char *mode;
    uint32_t from;
    uint32_t to;
    uint32_t windows;
    uint32_t batch;
    uint32_t accum;
};

struct mda_teach_t {
    mda_qwen_t *q;
    const mda_data_corpus_t *cp;
    const mda_tokenizer_t *qt;
    mda_kd_window_t *wins;
    uint32_t *tok_off;
    uint32_t *n_tok;
    uint32_t *ids;
    uint32_t *recs;
    uint64_t *index;
    FILE *out;
    mda_vkc_buf_t vocab;
    mda_vkc_buf_t cont;
    mda_vkc_buf_t outp;
    mda_vkc_kernel_t k_kd;
    mda_data_batch_t batch;
    mda_kd_hdr_t hdr;
    double nll_bits;
    double nll_bytes;
    uint64_t at;
    uint64_t w0;
    uint32_t n_win;
};

#endif
