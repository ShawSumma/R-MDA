#ifndef MDA_PC_H
#define MDA_PC_H

#include <stdint.h>

#define MDA_PC_NO_SLOT 0xffffffffu

typedef uint32_t mda_pc_vec_op_t;
typedef struct mda_pc_emb_t mda_pc_emb_t;
typedef struct mda_pc_rms_t mda_pc_rms_t;
typedef struct mda_pc_ts_t mda_pc_ts_t;
typedef struct mda_pc_absmax_t mda_pc_absmax_t;
typedef struct mda_pc_cast_t mda_pc_cast_t;
typedef struct mda_pc_tr16_t mda_pc_tr16_t;
typedef struct mda_pc_mm_t mda_pc_mm_t;
typedef struct mda_pc_mm16_t mda_pc_mm16_t;
typedef struct mda_pc_mm_bw_t mda_pc_mm_bw_t;
typedef struct mda_pc_heads_t mda_pc_heads_t;
typedef struct mda_pc_heads_bwd_t mda_pc_heads_bwd_t;
typedef struct mda_pc_wkv_t mda_pc_wkv_t;
typedef struct mda_pc_wkv_bwd_t mda_pc_wkv_bwd_t;
typedef struct mda_pc_vec_t mda_pc_vec_t;
typedef struct mda_pc_repack_t mda_pc_repack_t;
typedef struct mda_pc_wout_t mda_pc_wout_t;
typedef struct mda_pc_lora_t mda_pc_lora_t;
typedef struct mda_pc_colsum_t mda_pc_colsum_t;
typedef struct mda_pc_rope_t mda_pc_rope_t;
typedef struct mda_pc_attn_t mda_pc_attn_t;
typedef struct mda_pc_align_t mda_pc_align_t;
typedef struct mda_pc_elem_t mda_pc_elem_t;
typedef struct mda_pc_ce_t mda_pc_ce_t;
typedef struct mda_pc_argmax_t mda_pc_argmax_t;
typedef struct mda_pc_softmax_t mda_pc_softmax_t;
typedef struct mda_pc_sumsq_t mda_pc_sumsq_t;
typedef struct mda_pc_adam_t mda_pc_adam_t;
typedef struct mda_pc_muon_t mda_pc_muon_t;
typedef struct mda_pc_adaln_t mda_pc_adaln_t;

enum {
    MDA_PC_VEC_ADD,
    MDA_PC_VEC_MUL,
    MDA_PC_VEC_AVG,
    MDA_PC_VEC_COPY,
};

struct mda_pc_emb_t {
    uint32_t bt;
    uint32_t d;
    uint32_t op;
    uint32_t vocab;
};

struct mda_pc_rms_t {
    uint32_t rows;
    uint32_t d;
    uint32_t op;
};

struct mda_pc_ts_t {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
    uint32_t blk;
    uint32_t nclean;
    uint32_t cache;
};

struct mda_pc_absmax_t {
    uint32_t n;
    uint32_t slot;
};

struct mda_pc_cast_t {
    uint32_t n;
    uint32_t slot;
};

struct mda_pc_tr16_t {
    uint32_t rows;
    uint32_t cols;
    uint32_t slot;
};

struct mda_pc_mm_t {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

struct mda_pc_mm16_t {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t sa;
    uint32_t sb;
    uint32_t ta;
    uint32_t tb;
    uint32_t store;
    float alpha;
    float beta;
};

struct mda_pc_mm_bw_t {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t mode;
    uint32_t acc;
};

struct mda_pc_heads_t {
    uint32_t n_vec;
    uint32_t sec;
    uint32_t hd;
    uint32_t seqlen;
    uint32_t first_row;
    uint32_t w0;
    uint32_t a0;
    uint32_t v0;
    uint32_t kk;
    uint32_t ka;
    uint32_t rk;
};

struct mda_pc_heads_bwd_t {
    uint32_t n_vec;
    uint32_t sec;
    uint32_t hd;
    uint32_t seqlen;
    uint32_t first_row;
    uint32_t slot;
    uint32_t kk;
    uint32_t ka;
    uint32_t rk;
};

struct mda_pc_wkv_t {
    uint32_t bsz;
    uint32_t hd;
    uint32_t t;
    uint32_t dh;
    uint32_t dir;
    uint32_t save;
    uint32_t blk;
    uint32_t nclean;
    uint32_t seq0;
    uint32_t cache;
    uint32_t sec;
    uint32_t split;
};

struct mda_pc_wkv_bwd_t {
    uint32_t bsz;
    uint32_t hd;
    uint32_t t;
    uint32_t dh;
    uint32_t dir;
    uint32_t blk;
    uint32_t nclean;
    uint32_t seq0;
    uint32_t split;
    uint32_t sec;
};

struct mda_pc_vec_t {
    uint32_t n;
    mda_pc_vec_op_t op;
};

struct mda_pc_repack_t {
    uint32_t bt;
    uint32_t hd;
    uint32_t dh;
    uint32_t seqlen;
    uint32_t op;
};

struct mda_pc_wout_t {
    uint32_t n_vec;
    uint32_t sec;
    uint32_t hd;
    uint32_t seqlen;
    uint32_t op;
    uint32_t slot;
};

struct mda_pc_lora_t {
    uint32_t n;
    uint32_t op;
    uint32_t act;
    uint32_t slot;
};

struct mda_pc_colsum_t {
    uint32_t rows;
    uint32_t d;
    uint32_t rows_per;
};

struct mda_pc_rope_t {
    uint32_t rows;
    uint32_t heads;
    uint32_t seqlen;
    float theta;
};

struct mda_pc_attn_t {
    uint32_t seqlen;
    uint32_t heads;
    uint32_t kv_heads;
};

struct mda_pc_align_t {
    uint32_t n;
    uint32_t d;
    uint32_t slot;
};

struct mda_pc_elem_t {
    uint32_t n;
    uint32_t op;
    uint32_t slot;
    uint32_t slot2;
};

struct mda_pc_ce_t {
    uint32_t rows;
    uint32_t vocab;
    uint32_t op;
    uint32_t slot;
    uint32_t soft;
    float teach;
};

struct mda_pc_argmax_t {
    uint32_t rows;
    uint32_t vocab;
    float temp;
    uint32_t seed;
};

struct mda_pc_softmax_t {
    uint32_t rows;
    uint32_t vocab;
    float temp;
};

struct mda_pc_sumsq_t {
    uint32_t n;
    uint32_t op;
    float gscale;
    float clip;
    float growth;
};

struct mda_pc_adam_t {
    uint32_t n;
    float beta1;
    float beta2;
    float eps;
    float wd;
    float lr_mul;
};

struct mda_pc_muon_t {
    uint32_t n;
    uint32_t op;
    uint32_t aux;
    float p0;
    float p1;
};

struct mda_pc_adaln_t {
    uint32_t bt;
    uint32_t d;
    uint32_t op;
    uint32_t s_off;
    uint32_t sh_off;
    uint32_t rows_per;
    float scale;
};

#endif
