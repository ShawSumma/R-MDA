#ifndef MDA_PC_H
#define MDA_PC_H

#include <stdint.h>

#define MDA_PC_NO_SLOT 0xffffffffu

typedef uint32_t mda_pc_vec_op_t;
typedef struct mda_pc_emb_t mda_pc_emb_t;
typedef struct mda_pc_ln_t mda_pc_ln_t;
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
typedef struct mda_pc_elem_t mda_pc_elem_t;
typedef struct mda_pc_ce_t mda_pc_ce_t;
typedef struct mda_pc_argmax_t mda_pc_argmax_t;
typedef struct mda_pc_softmax_t mda_pc_softmax_t;
typedef struct mda_pc_sumsq_t mda_pc_sumsq_t;
typedef struct mda_pc_adam_t mda_pc_adam_t;
typedef struct mda_pc_muon_t mda_pc_muon_t;
typedef struct mda_pc_qwen_bf16_t mda_pc_qwen_bf16_t;
typedef struct mda_pc_qwen_embed_t mda_pc_qwen_embed_t;
typedef struct mda_pc_qwen_norm_t mda_pc_qwen_norm_t;
typedef struct mda_pc_qwen_hc_t mda_pc_qwen_hc_t;
typedef struct mda_pc_qwen_elem_t mda_pc_qwen_elem_t;
typedef struct mda_pc_qwen_conv_t mda_pc_qwen_conv_t;
typedef struct mda_pc_qwen_gdn_t mda_pc_qwen_gdn_t;
typedef struct mda_pc_qwen_attn_t mda_pc_qwen_attn_t;
typedef struct mda_pc_qwen_ple_t mda_pc_qwen_ple_t;
typedef struct mda_pc_moe_route_t mda_pc_moe_route_t;
typedef struct mda_pc_moe_scan_t mda_pc_moe_scan_t;
typedef struct mda_pc_moe_scatter_t mda_pc_moe_scatter_t;
typedef struct mda_pc_moe_mm_t mda_pc_moe_mm_t;
typedef struct mda_pc_moe_combine_t mda_pc_moe_combine_t;
typedef struct mda_pc_kd_bytes_t mda_pc_kd_bytes_t;
typedef struct mda_pc_kd_student_t mda_pc_kd_student_t;

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

struct mda_pc_ln_t {
    uint32_t rows;
    uint32_t d;
    uint32_t op;
    float eps;
};

struct mda_pc_ts_t {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
    uint32_t blk;
    uint32_t nclean;
    uint32_t cache;
    uint32_t nmix;
    uint32_t sec;
    uint32_t rows_per;
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
    uint32_t lw;
    uint32_t lb;
    float eps;
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

struct mda_pc_elem_t {
    uint32_t n;
    uint32_t op;
    uint32_t slot;
};

struct mda_pc_ce_t {
    uint32_t rows;
    uint32_t vocab;
    uint32_t op;
    uint32_t slot;
    uint32_t kd_rows;
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
    uint32_t base;
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

struct mda_pc_qwen_bf16_t {
    uint32_t n;
};

struct mda_pc_qwen_embed_t {
    uint32_t rows;
    uint32_t d;
    uint32_t hc;
};

struct mda_pc_qwen_norm_t {
    uint32_t rows;
    uint32_t groups;
    uint32_t gsize;
    uint32_t in_stride;
    uint32_t in_off;
    uint32_t in_gstride;
    uint32_t out_stride;
    uint32_t out_off;
    uint32_t out_gstride;
    uint32_t w_off;
    uint32_t wstride;
    uint32_t op;
    uint32_t rot;
    uint32_t seqlen;
    uint32_t z_stride;
    uint32_t z_off;
    uint32_t z_gstride;
    float eps;
    float scale;
    float theta;
};

struct mda_pc_qwen_hc_t {
    uint32_t rows;
    uint32_t d;
    uint32_t hc;
    uint32_t op;
    uint32_t inj_stride;
    uint32_t inj_off;
};

struct mda_pc_qwen_elem_t {
    uint32_t rows;
    uint32_t n;
    uint32_t astride;
    uint32_t ao;
    uint32_t bo;
    uint32_t op;
    float scale;
};

struct mda_pc_qwen_conv_t {
    uint32_t rows;
    uint32_t ch;
    uint32_t seqlen;
    uint32_t k;
    uint32_t dil;
    uint32_t in_stride;
    uint32_t in_off;
    uint32_t w_off;
};

struct mda_pc_qwen_gdn_t {
    uint32_t seqlen;
    uint32_t vheads;
    uint32_t kheads;
    uint32_t qkv_stride;
    uint32_t k_off;
    uint32_t v_off;
    uint32_t ps;
    uint32_t boff;
    uint32_t aoff;
    uint32_t a_log;
    uint32_t dt_bias;
};

struct mda_pc_qwen_attn_t {
    uint32_t seqlen;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t p_stride;
    uint32_t v_off;
    uint32_t g_off;
    uint32_t g_hstride;
    float scale;
};

struct mda_pc_qwen_ple_t {
    uint32_t rows;
    uint32_t d;
    uint32_t hc;
    uint32_t pstride;
    uint32_t voff;
};

struct mda_pc_moe_route_t {
    uint32_t rows;
    uint32_t experts;
    uint32_t topk;
    uint32_t stride;
};

struct mda_pc_moe_scan_t {
    uint32_t experts;
    uint32_t tile;
};

struct mda_pc_moe_scatter_t {
    uint32_t rows;
    uint32_t topk;
    uint32_t d8;
};

struct mda_pc_moe_mm_t {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t sa;
    uint32_t sb;
};

struct mda_pc_moe_combine_t {
    uint32_t rows;
    uint32_t d;
    uint32_t topk;
    uint32_t pstride;
    uint32_t sg_off;
    uint32_t inj_stride;
    uint32_t inj_off;
    uint32_t hc;
};

struct mda_pc_kd_bytes_t {
    uint32_t rows;
    uint32_t vocab;
};

struct mda_pc_kd_student_t {
    uint32_t rows;
    uint32_t vocab;
    uint32_t slot;
    float lambda;
};

#endif
