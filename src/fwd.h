#ifndef MDA_FWD_H
#define MDA_FWD_H

#include <stdbool.h>
#include <stdint.h>

#include "vkc.h"
#include "pc.h"
#include "data.h"
#include "model.h"

typedef struct mda_fwd_act_t mda_fwd_act_t;
typedef struct mda_fwd_opt_t mda_fwd_opt_t;
typedef struct mda_fwd_scratch_t mda_fwd_scratch_t;

struct mda_fwd_act_t {
    mda_vkc_buf_t x_in;
    mda_vkc_buf_t rms1;
    mda_vkc_buf_t n1_pre;
    mda_vkc_buf_t n1;
    mda_vkc_buf_t shift;
    mda_vkc_buf_t pk;
    mda_vkc_buf_t pv;
    mda_vkc_buf_t pr;
    mda_vkc_buf_t pw;
    mda_vkc_buf_t pa;
    mda_vkc_buf_t kinv;
    mda_vkc_buf_t sstate;
    mda_vkc_buf_t sstate2;
    mda_vkc_buf_t wkv_raw;
    mda_vkc_buf_t wkv;
    mda_vkc_buf_t rmsx;
    mda_vkc_buf_t x_mid;
    mda_vkc_buf_t rms2;
    mda_vkc_buf_t n2_pre;
    mda_vkc_buf_t n2;
    mda_vkc_buf_t gate_pre;
    mda_vkc_buf_t gate;
    mda_vkc_buf_t up;
    mda_vkc_buf_t h;
};

struct mda_fwd_opt_t {
    float lr;
    float bc1;
    float bc2;
    float norm;
    float scale;
    float gmul;
    float good;
    float skip;
};

struct mda_fwd_scratch_t {
    mda_vkc_buf_t opt;
    mda_vkc_buf_t ids;
    mda_vkc_buf_t target;
    mda_vkc_buf_t mask;
    mda_vkc_buf_t weight;
    mda_vkc_buf_t x;
    mda_vkc_buf_t temb;
    mda_vkc_buf_t dada4;
    mda_vkc_buf_t tmp;
    mda_vkc_buf_t tmp2;
    mda_vkc_buf_t dup;
    mda_vkc_buf_t rms_stash;
    mda_vkc_buf_t dg_dummy;
    mda_vkc_buf_t pk;
    mda_vkc_buf_t pv;
    mda_vkc_buf_t pr;
    mda_vkc_buf_t pw;
    mda_vkc_buf_t pa;
    mda_vkc_buf_t po;
    mda_vkc_buf_t po2;
    mda_vkc_buf_t xn;
    mda_vkc_buf_t logits;
    mda_vkc_buf_t loss;
    mda_vkc_buf_t tok;
    mda_vkc_buf_t dnorm;
    mda_vkc_buf_t dtmp;
    mda_vkc_buf_t dtmp2;
    mda_vkc_buf_t dh;
    mda_vkc_buf_t dpk;
    mda_vkc_buf_t dpv;
    mda_vkc_buf_t dpr;
    mda_vkc_buf_t dpw;
    mda_vkc_buf_t dpa;
    mda_vkc_buf_t dpk2;
    mda_vkc_buf_t dpv2;
    mda_vkc_buf_t dpr2;
    mda_vkc_buf_t dpw2;
    mda_vkc_buf_t dpa2;
    mda_vkc_buf_t dwkv;
    mda_vkc_buf_t dxh_buf;
    mda_vkc_buf_t a16;
    mda_vkc_buf_t dc16;
    mda_vkc_buf_t dlt16;
    mda_vkc_buf_t at16;
    mda_data_batch_t batch;
    mda_fwd_act_t *acts;
    mda_vkc_buf_t *ada4;
    float temp;
    uint32_t seed;
    uint32_t a16_slot;
    uint32_t at16_slot;
    uint32_t n_acts;
};

mda_fwd_scratch_t mda_fwd_alloc(mda_model_t *m, uint32_t n_acts);
void mda_fwd_vec_op(mda_model_t *m, mda_vkc_buf_t *a, mda_vkc_buf_t *b, uint32_t n, mda_pc_vec_op_t op);
void mda_fwd_repack(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t op);
void mda_fwd_mm(mda_model_t *m, mda_vkc_buf_t *a, mda_vkc_buf_t *b, mda_vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k);
uint32_t mda_fwd_absmax(mda_model_t *m, mda_vkc_buf_t *a, uint32_t n);
void mda_fwd_cast(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t n, uint32_t slot);
uint32_t mda_fwd_cast_act(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t n);
void mda_fwd_transpose16(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t rows, uint32_t cols, uint32_t slot);
void mda_fwd_refresh_wt16_rec(mda_model_t *m);
void mda_fwd_refresh_w16(mda_model_t *m);
void mda_fwd_mm16_raw(mda_model_t *m, mda_vkc_buf_t *x16, uint32_t sx, mda_vkc_buf_t *y16, uint32_t sy, mda_vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k, bool acc);
void mda_fwd_heads(mda_model_t *m, mda_vkc_kernel_t *k, mda_vkc_buf_t **bufs);
void mda_fwd_headnorm(mda_model_t *m, mda_vkc_buf_t *x, mda_vkc_buf_t *y, mda_vkc_buf_t *rms, mda_vkc_buf_t *dx, uint32_t op);
void mda_fwd_rec(mda_model_t *m, mda_fwd_scratch_t *sc, bool save);
void mda_fwd_ce_rec(mda_model_t *m, mda_fwd_scratch_t *sc);
float mda_fwd_loss_scale(mda_model_cfg_t cfg);
uint32_t mda_fwd_cond_rows(mda_model_cfg_t cfg);
void mda_fwd_adaln(mda_model_t *m, mda_vkc_buf_t *x, mda_vkc_buf_t *ada4, mda_vkc_buf_t *y, mda_vkc_buf_t *dx, mda_vkc_buf_t *dada4, uint32_t op, uint32_t slot);
float mda_fwd_run(mda_model_t *m, mda_fwd_scratch_t *sc);

#endif
