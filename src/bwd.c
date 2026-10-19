#include "bwd.h"
#include "pc.h"

void mda_bwd_mm_da(mda_model_t *m, mda_vkc_buf_t *b, mda_vkc_buf_t *dc, mda_vkc_buf_t *da, uint32_t rows, uint32_t n, uint32_t k) {
    mda_pc_mm_bw_t pc = {
        .m = rows,
        .n = n,
        .k = k,
        .mode = 0,
    };
    mda_vkc_buf_t *bufs[4] = {
        b,
        b,
        dc,
        da,
    };
    mda_vkc_run(m->c, &m->k_mm_bw, bufs, (k + 15) / 16, (rows + 15) / 16, 1, &pc);
}

void mda_bwd_mm_db(mda_model_t *m, mda_vkc_buf_t *a, mda_vkc_buf_t *dc, mda_vkc_buf_t *db, uint32_t rows, uint32_t n, uint32_t k) {
    mda_pc_mm_bw_t pc = {
        .m = rows,
        .n = n,
        .k = k,
        .mode = 1,
    };
    mda_vkc_buf_t *bufs[4] = {
        a,
        a,
        dc,
        db,
    };
    mda_vkc_run(m->c, &m->k_mm_bw, bufs, (n + 15) / 16, (k + 15) / 16, 1, &pc);
}

void mda_bwd_mm_at(mda_model_t *m, mda_fwd_scratch_t *sc, mda_vkc_buf_t *a, uint32_t rows, uint32_t k) {
    if (m->ref32 == false) {
        sc->at16_slot = mda_fwd_absmax(m, a, rows * k);
        mda_fwd_transpose16(m, a, &sc->at16, rows, k, sc->at16_slot);
    }
}

void mda_bwd_mm(mda_model_t *m, mda_fwd_scratch_t *sc, mda_vkc_buf_t *a, mda_model_param_t *p, mda_vkc_buf_t *dout, mda_vkc_buf_t *da, uint32_t rows, uint32_t n, uint32_t k) {
    if (m->ref32) {
        mda_bwd_mm_db(m, a, dout, &p->g, rows, n, k);
        mda_bwd_mm_da(m, &p->w, dout, da, rows, n, k);
    } else {
        uint32_t sd = mda_fwd_cast_act(m, dout, &sc->dc16, rows * n);
        mda_vkc_set_nobar(m->c, true);
        mda_fwd_mm16_raw(m, &sc->at16, sc->at16_slot, &sc->dc16, sd, &p->g, k, n, rows, m->grad_acc);
        mda_vkc_set_nobar(m->c, false);
        mda_fwd_mm16_raw(m, &sc->dc16, sd, &p->wt16, MDA_PC_NO_SLOT, da, rows, k, n, false);
    }
}
