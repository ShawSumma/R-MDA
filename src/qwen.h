#ifndef MDA_QWEN_H
#define MDA_QWEN_H

#include <stdbool.h>
#include <stdint.h>

#include "vkc.h"
#include "safet.h"

#define MDA_QWEN_DIR "models/qwen38fn"
#define MDA_QWEN_VOCAB_PATH "models/qwen.txt"
#define MDA_QWEN_PREFIX "model.language_model."
#define MDA_QWEN_MAX_SLOTS 256
#define MDA_QWEN_TILE 128
#define MDA_QWEN_WIDE_GROUPS 64
#define MDA_QWEN_EPS 1e-6f
#define MDA_QWEN_THETA 10000000.0f
#define MDA_QWEN_DENSE_CAP (128ull << 20)
#define MDA_QWEN_VEC_CAP (1u << 20)
#define MDA_QWEN_HC 4
#define MDA_QWEN_HC_PAD 384
#define MDA_QWEN_MOE_PAD 128
#define MDA_QWEN_NGRAM_MAX 4
#define MDA_QWEN_PLE_SHARDS 1024
#define MDA_QWEN_THREADS 16

typedef struct mda_qwen_cfg_t mda_qwen_cfg_t;
typedef struct mda_qwen_stage_t mda_qwen_stage_t;
typedef struct mda_qwen_res_t mda_qwen_res_t;
typedef struct mda_qwen_layer_t mda_qwen_layer_t;
typedef struct mda_qwen_bank_t mda_qwen_bank_t;
typedef struct mda_qwen_job_t mda_qwen_job_t;
typedef struct mda_qwen_ple_t mda_qwen_ple_t;
typedef struct mda_qwen_scratch_t mda_qwen_scratch_t;
typedef struct mda_qwen_t mda_qwen_t;
typedef struct mda_qwen_copy_t mda_qwen_copy_t;
typedef struct mda_qwen_gather_t mda_qwen_gather_t;
typedef bool (*mda_qwen_sink_t)(void *ctx, mda_qwen_t *q, uint32_t row0, uint32_t rows);

struct mda_qwen_cfg_t {
    uint32_t d;
    uint32_t n_layer;
    uint32_t vocab;
    uint32_t hc_rank;
    uint32_t k_heads;
    uint32_t v_heads;
    uint32_t lin_head;
    uint32_t conv;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rot;
    uint32_t experts;
    uint32_t topk;
    uint32_t inter;
    uint32_t shared;
    uint32_t attn_every;
    uint32_t ple_layer;
    uint32_t ngram;
    uint32_t ngram_heads;
    uint32_t ple_conv;
    uint32_t ple_dil;
    uint32_t eos;
};

struct mda_qwen_stage_t {
    mda_vkc_buf_t host;
    uint64_t used;
};

struct mda_qwen_res_t {
    uint64_t down;
    uint64_t up;
    uint32_t norm;
};

struct mda_qwen_layer_t {
    mda_qwen_res_t ahc;
    mda_qwen_res_t mhc;
    uint64_t mix;
    uint64_t out;
    uint64_t moe;
    uint64_t sdown;
    uint64_t ple;
    uint32_t conv;
    uint32_t a_log;
    uint32_t dt_bias;
    uint32_t gnorm;
    uint32_t qnorm;
    uint32_t knorm;
    uint32_t pkey;
    uint32_t pquery;
    uint32_t pnorm;
    uint32_t pconv;
    bool attn;
    bool ple_on;
};

struct mda_qwen_bank_t {
    mda_qwen_stage_t dense;
    mda_qwen_stage_t gu;
    mda_qwen_stage_t down;
    mda_qwen_stage_t vec;
    mda_qwen_layer_t ly;
    mda_qwen_res_t mixer;
};

struct mda_qwen_ple_t {
    mda_safet_tensor_t shards[MDA_QWEN_PLE_SHARDS];
    uint64_t mult[MDA_QWEN_NGRAM_MAX];
    uint64_t sizes[MDA_QWEN_NGRAM_MAX * 16];
    uint64_t offs[MDA_QWEN_NGRAM_MAX * 16];
    uint32_t n_shards;
    uint32_t shard_rows;
    uint32_t dim;
};

struct mda_qwen_scratch_t {
    mda_vkc_buf_t hn;
    mda_vkc_buf_t t;
    mda_vkc_buf_t u;
    mda_vkc_buf_t up;
    mda_vkc_buf_t x;
    mda_vkc_buf_t p;
    mda_vkc_buf_t s1;
    mda_vkc_buf_t s2;
    mda_vkc_buf_t s3;
    mda_vkc_buf_t y;
    mda_vkc_buf_t a16;
    mda_vkc_buf_t idx;
    mda_vkc_buf_t wt;
    mda_vkc_buf_t slot;
    mda_vkc_buf_t count;
    mda_vkc_buf_t off;
    mda_vkc_buf_t fill;
    mda_vkc_buf_t tiles;
    mda_vkc_buf_t xp;
    mda_vkc_buf_t g;
    mda_vkc_buf_t hp;
    mda_vkc_buf_t hp16;
    mda_vkc_buf_t ye;
    mda_vkc_buf_t hs;
    mda_vkc_buf_t sh;
    mda_vkc_buf_t e;
    mda_vkc_buf_t e_host;
    mda_vkc_buf_t logits;
    uint32_t moe_rows;
};

struct mda_qwen_t {
    mda_vkc_t *c;
    mda_safet_set_t set;
    mda_qwen_bank_t bank[2];
    mda_vkc_buf_t wd;
    mda_vkc_buf_t wgu;
    mda_vkc_buf_t wdown;
    mda_vkc_buf_t wvec;
    mda_vkc_buf_t h;
    mda_vkc_buf_t ids;
    mda_vkc_buf_t amax;
    mda_qwen_scratch_t sc;
    mda_qwen_ple_t ple;
    mda_qwen_layer_t ly;
    mda_qwen_res_t mixer;
    mda_vkc_kernel_t k_mm16;
    mda_vkc_kernel_t k_mm16w;
    mda_vkc_kernel_t k_moe;
    mda_vkc_kernel_t k_cast;
    mda_vkc_kernel_t k_absmax;
    mda_vkc_kernel_t k_bf16;
    mda_vkc_kernel_t k_embed;
    mda_vkc_kernel_t k_norm;
    mda_vkc_kernel_t k_hc;
    mda_vkc_kernel_t k_elem;
    mda_vkc_kernel_t k_conv;
    mda_vkc_kernel_t k_gdn;
    mda_vkc_kernel_t k_attn;
    mda_vkc_kernel_t k_ple;
    mda_vkc_kernel_t k_route;
    mda_vkc_kernel_t k_scan;
    mda_vkc_kernel_t k_scatter;
    mda_vkc_kernel_t k_combine;
    mda_qwen_cfg_t cfg;
    const uint32_t *host_ids;
    uint32_t rows;
    uint32_t chunk;
    uint32_t seqlen;
    uint32_t head_rows;
    uint32_t n_slots;
};

struct mda_qwen_job_t {
    mda_qwen_t *q;
    mda_qwen_bank_t *b;
    uint32_t item;
};

struct mda_qwen_copy_t {
    uint8_t *dst;
    const uint8_t *src;
    uint64_t n;
};

struct mda_qwen_gather_t {
    const mda_qwen_t *q;
    float *e;
    uint32_t row0;
    uint32_t lo;
    uint32_t hi;
};

mda_qwen_t *mda_qwen_open(mda_vkc_t *c, const char *dir, uint32_t rows, uint32_t chunk, uint32_t seqlen, uint32_t head_rows);
bool mda_qwen_run(mda_qwen_t *q, const uint32_t *ids, mda_qwen_sink_t sink, void *ctx);

#endif
