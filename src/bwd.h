#ifndef MDA_BWD_H
#define MDA_BWD_H

#include <stdbool.h>
#include <stdint.h>

#include "vkc.h"
#include "model.h"
#include "fwd.h"

void mda_bwd_mm_da(mda_model_t *m, mda_vkc_buf_t *b, mda_vkc_buf_t *dc, mda_vkc_buf_t *da, uint32_t rows, uint32_t n, uint32_t k, bool acc);
void mda_bwd_mm_db(mda_model_t *m, mda_vkc_buf_t *a, mda_vkc_buf_t *dc, mda_vkc_buf_t *db, uint32_t rows, uint32_t n, uint32_t k);
void mda_bwd_mm(mda_model_t *m, mda_fwd_scratch_t *sc, mda_vkc_buf_t *a, mda_vkc_buf_t *a16, uint32_t slot, mda_model_param_t *p, mda_vkc_buf_t *dout, uint32_t sd, mda_vkc_buf_t *da, uint32_t rows, uint32_t n, uint32_t k, bool acc);

#endif
