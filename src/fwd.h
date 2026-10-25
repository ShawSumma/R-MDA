#ifndef MDA_FWD_H
#define MDA_FWD_H

#include <stdbool.h>
#include <stdint.h>

#include "vkc.h"
#include "pc.h"
#include "data.h"
#include "model.h"
#include "kd.h"

#define MDA_FWD_SUM_ROWS 64
#define MDA_FWD_ELEM_GROUPS 1024
#define MDA_FWD_WIDE_GROUPS 64
#define MDA_FWD_LN_EPS 1e-5f
#define MDA_FWD_LNX_EPS 64e-5f
#define MDA_FWD_CHUNK (1u << 28)

typedef uint32_t mda_fwd_layout_t;
typedef struct mda_fwd_epi_t mda_fwd_epi_t;
typedef struct mda_fwd_lora_act_t mda_fwd_lora_act_t;
typedef struct mda_fwd_act_t mda_fwd_act_t;
typedef struct mda_fwd_cache_t mda_fwd_cache_t;
typedef struct mda_fwd_opt_t mda_fwd_opt_t;
typedef struct mda_fwd_scratch_t mda_fwd_scratch_t;

enum {
    MDA_FWD_AB,
    MDA_FWD_AT,
    MDA_FWD_BT,
};

enum {
    MDA_FWD_STORE32 = 1,
    MDA_FWD_STORE16 = 2,
    MDA_FWD_STORE_SYM = 4,
};

struct mda_fwd_epi_t {
    mda_vkc_buf_t *prev;
    float alpha;
    float beta;
    uint32_t store;
};

struct mda_fwd_lora_act_t {
    mda_vkc_buf_t h;
    mda_vkc_buf_t hf;
    mda_vkc_buf_t hf16;
    uint32_t slot;
};

struct mda_fwd_act_t {
    mda_vkc_buf_t x_in;
    mda_vkc_buf_t st1;
    mda_vkc_buf_t n1;
    mda_vkc_buf_t shift;
    mda_vkc_buf_t shift16;
    mda_vkc_buf_t kv;
    mda_vkc_buf_t heads;
    mda_vkc_buf_t kinv;
    mda_vkc_buf_t bs;
    mda_vkc_buf_t vg;
    mda_vkc_buf_t g;
    mda_vkc_buf_t sstate;
    mda_vkc_buf_t sstate2;
    mda_vkc_buf_t sk;
    mda_vkc_buf_t sk2;
    mda_vkc_buf_t wkv_raw;
    mda_vkc_buf_t p;
    mda_vkc_buf_t wkv;
    mda_vkc_buf_t stx;
    mda_vkc_buf_t x_mid;
    mda_vkc_buf_t st2;
    mda_vkc_buf_t n2;
    mda_vkc_buf_t fshift;
    mda_vkc_buf_t fshift16;
    mda_vkc_buf_t kpre;
    mda_vkc_buf_t h;
    mda_vkc_buf_t wkv16;
    mda_vkc_buf_t h16;
    mda_fwd_lora_act_t lora[MDA_MODEL_LORAS];
    uint32_t shift_slot[MDA_MODEL_MIXES];
    uint32_t fshift_slot;
    uint32_t wkv_slot;
    uint32_t h_slot;
};

struct mda_fwd_cache_t {
    mda_vkc_buf_t state[2];
    mda_vkc_buf_t carry[2];
    uint32_t cur;
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
    mda_vkc_buf_t emb;
    mda_vkc_buf_t st0;
    mda_vkc_buf_t x;
    mda_vkc_buf_t bstate;
    mda_vkc_buf_t dbstate;
    mda_vkc_buf_t stn;
    mda_vkc_buf_t dg_dummy;
    mda_vkc_buf_t raw;
    mda_vkc_buf_t v0;
    mda_vkc_buf_t po;
    mda_vkc_buf_t po2;
    mda_vkc_buf_t xn;
    mda_vkc_buf_t logits;
    mda_vkc_buf_t loss;
    mda_vkc_buf_t logz;
    mda_vkc_buf_t tok;
    mda_vkc_buf_t conf;
    mda_vkc_buf_t dnorm;
    mda_vkc_buf_t dtmp;
    mda_vkc_buf_t dtmp2;
    mda_vkc_buf_t dh;
    mda_vkc_buf_t dshift;
    mda_vkc_buf_t dheads[2];
    mda_vkc_buf_t dv0;
    mda_vkc_buf_t con;
    mda_vkc_buf_t dbs;
    mda_vkc_buf_t dvb;
    mda_vkc_buf_t dgate;
    mda_vkc_buf_t dln;
    mda_vkc_buf_t dlora;
    mda_vkc_buf_t dwkv;
    mda_vkc_buf_t dxh_buf;
    mda_vkc_buf_t a16;
    mda_vkc_buf_t dc16;
    mda_vkc_buf_t kd_toks;
    mda_vkc_buf_t kd_cont;
    mda_vkc_buf_t kd_tgt;
    mda_vkc_buf_t dkd;
    mda_vkc_buf_t kd_loss;
    mda_vkc_buf_t kd_bits;
    mda_data_batch_t batch;
    mda_fwd_act_t *acts;
    mda_fwd_cache_t *cache;
    mda_vkc_buf_t *stage;
    float temp;
    uint32_t seed;
    uint32_t a16_slot;
    uint32_t ce_slot;
    uint32_t n_acts;
};

mda_fwd_scratch_t mda_fwd_alloc(mda_model_t *m, uint32_t n_acts);
mda_fwd_cache_t mda_fwd_cache_alloc(mda_model_t *m);
void mda_fwd_cache_reset(mda_model_t *m, mda_fwd_cache_t *cache);
void mda_fwd_cache_commit(mda_fwd_cache_t *cache);
mda_vkc_buf_t mda_fwd_cache_view(mda_fwd_scratch_t *sc, mda_vkc_buf_t *pair, mda_vkc_buf_t *fallback, uint32_t which, uint32_t l, VkDeviceSize per);
VkDeviceSize mda_fwd_cache_carry_bytes(mda_model_cfg_t cfg);
void mda_fwd_vec_op(mda_model_t *m, mda_vkc_buf_t *a, mda_vkc_buf_t *b, uint32_t n, mda_pc_vec_op_t op);
uint32_t mda_fwd_rows(const mda_model_t *m);
bool mda_fwd_split(const mda_model_t *m);
uint32_t mda_fwd_dec_start(const mda_model_t *m);
bool mda_fwd_clean_only(const mda_model_t *m);
void mda_fwd_set_live(mda_model_t *m, uint32_t l);
uint32_t mda_fwd_first_row(const mda_model_t *m, uint32_t l);
void mda_fwd_repack(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t op);
void mda_fwd_mm(mda_model_t *m, mda_vkc_buf_t *a, mda_vkc_buf_t *b, mda_vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k);
void mda_fwd_slots_begin(mda_model_t *m);
uint32_t mda_fwd_slot(mda_model_t *m);
uint32_t mda_fwd_absmax(mda_model_t *m, mda_vkc_buf_t *a, uint32_t n);
void mda_fwd_cast(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t n, uint32_t slot);
uint32_t mda_fwd_cast_act(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t n);
void mda_fwd_elem(mda_model_t *m, mda_vkc_buf_t *x, mda_vkc_buf_t *pre, uint32_t op, uint32_t slot);
void mda_fwd_transpose16(mda_model_t *m, mda_vkc_buf_t *src, mda_vkc_buf_t *dst, uint32_t rows, uint32_t cols, uint32_t slot);
void mda_fwd_refresh_wt16_rec(mda_model_t *m);
void mda_fwd_refresh_w16(mda_model_t *m);
void mda_fwd_mm16_epi(mda_model_t *m, mda_vkc_buf_t *x16, uint32_t sx, mda_vkc_buf_t *y16, uint32_t sy, mda_vkc_buf_t *out, mda_vkc_buf_t *out16, uint32_t rows, uint32_t n, uint32_t k, mda_fwd_layout_t layout, mda_fwd_epi_t epi);
void mda_fwd_mm16_raw(mda_model_t *m, mda_vkc_buf_t *x16, uint32_t sx, mda_vkc_buf_t *y16, uint32_t sy, mda_vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k, bool acc, mda_fwd_layout_t layout);
mda_vkc_buf_t mda_fwd_section(mda_vkc_buf_t *b, mda_model_cfg_t cfg, uint32_t i);
mda_vkc_buf_t mda_fwd_section16(mda_vkc_buf_t *b, mda_model_cfg_t cfg, uint32_t i);
uint32_t mda_fwd_n_vec(const mda_model_t *m);
mda_vkc_buf_t mda_fwd_layer_vecs(mda_model_t *m, const mda_model_layer_t *ly);
uint32_t mda_fwd_vec_off(const mda_model_layer_t *ly, const mda_model_param_t *p);
uint32_t mda_fwd_lora_mix(uint32_t i);
void mda_fwd_layernorm(mda_model_t *m, mda_vkc_buf_t *x, mda_model_param_t *w, mda_model_param_t *b, mda_vkc_buf_t *y, mda_vkc_buf_t *stats, uint32_t rows, uint32_t op, mda_vkc_buf_t *dacc);
void mda_fwd_tshift(mda_model_t *m, mda_fwd_scratch_t *sc, mda_vkc_buf_t *x, mda_model_param_t *mix, mda_vkc_buf_t *out, uint32_t nmix, uint32_t op, uint32_t carry, uint32_t l, mda_vkc_buf_t *dx);
void mda_fwd_lora_act(mda_model_t *m, mda_vkc_buf_t *h, mda_vkc_buf_t *x, uint32_t n, uint32_t op, mda_model_act_t act, uint32_t slot);
void mda_fwd_wout(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, mda_vkc_buf_t *y, uint32_t op, uint32_t slot);
mda_vkc_buf_t mda_fwd_lora_dest(mda_fwd_scratch_t *sc, mda_fwd_act_t *a, mda_model_cfg_t cfg, uint32_t i);
void mda_fwd_time_mix(mda_model_t *m, mda_fwd_scratch_t *sc, mda_model_layer_t *ly, mda_fwd_act_t *a, uint32_t l, bool save);
void mda_fwd_rec(mda_model_t *m, mda_fwd_scratch_t *sc, bool save);
void mda_fwd_ce(mda_model_t *m, mda_fwd_scratch_t *sc, uint32_t op, uint32_t slot);
void mda_fwd_ce_rec(mda_model_t *m, mda_fwd_scratch_t *sc);
uint32_t mda_fwd_kd_rows(const mda_model_t *m);
void mda_fwd_kd_rec(mda_model_t *m, mda_fwd_scratch_t *sc);
float mda_fwd_loss_scale(mda_model_cfg_t cfg);
float mda_fwd_run(mda_model_t *m, mda_fwd_scratch_t *sc);

#endif
