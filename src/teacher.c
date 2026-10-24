#include <stdio.h>
#include <stdlib.h>

#include "teacher.h"
#include "pc.h"
#include "fwd.h"
#include "util.h"

uint64_t mda_teacher_round(uint64_t n) {
    return (n + MDA_TEACHER_ALIGN - 1) / MDA_TEACHER_ALIGN * MDA_TEACHER_ALIGN;
}

uint32_t mda_teacher_count_layers(const mda_safet_file_t *f) {
    uint32_t n = 0;
    char name[MDA_SAFET_MAX_KEY];
    snprintf(name, sizeof(name), "model.layers.%u.input_layernorm.weight", n);
    while (mda_safet_has(f, name)) {
        n += 1;
        snprintf(name, sizeof(name), "model.layers.%u.input_layernorm.weight", n);
    }
    return n;
}

mda_teacher_dims_t mda_teacher_dims(const mda_safet_file_t *f) {
    mda_safet_tensor_t emb = mda_safet_tensor(f, "model.embed_tokens.weight");
    mda_safet_tensor_t k = mda_safet_tensor(f, "model.layers.0.self_attn.k_proj.weight");
    mda_safet_tensor_t gate = mda_safet_tensor(f, "model.layers.0.mlp.gate_proj.weight");
    return (mda_teacher_dims_t) {
        .d = emb.cols,
        .kv = k.rows,
        .ffn = gate.rows,
        .vocab = emb.rows,
        .n_layer = mda_teacher_count_layers(f),
    };
}

void mda_teacher_layout(mda_teacher_t *t) {
    mda_teacher_dims_t dm = t->dims;
    uint64_t d = dm.d;
    uint64_t off = 0;
    t->emb = off;
    off += mda_teacher_round((uint64_t)dm.vocab * d);
    t->norm = off;
    off += mda_teacher_round(d);
    t->layers = malloc(dm.n_layer * sizeof(mda_teacher_layer_t));
    for (uint32_t l = 0; l < dm.n_layer; l += 1) {
        mda_teacher_layer_t *ly = &t->layers[l];
        ly->ln1 = off;
        off += mda_teacher_round(d);
        ly->ln2 = off;
        off += mda_teacher_round(d);
        ly->wq = off;
        off += mda_teacher_round(d * d);
        ly->wk = off;
        off += mda_teacher_round(d * dm.kv);
        ly->wv = off;
        off += mda_teacher_round(d * dm.kv);
        ly->wo = off;
        off += mda_teacher_round(d * d);
        ly->gate = off;
        off += mda_teacher_round(d * dm.ffn);
        ly->up = off;
        off += mda_teacher_round(d * dm.ffn);
        ly->down = off;
        off += mda_teacher_round(d * dm.ffn);
    }
    t->n_all = off;
}

bool mda_teacher_linear(float *w, const mda_safet_file_t *f, uint32_t l, const char *suffix, uint64_t off, uint32_t in, uint32_t out) {
    char name[MDA_SAFET_MAX_KEY];
    snprintf(name, sizeof(name), "model.layers.%u.%s", l, suffix);
    return mda_safet_linear(w + off, in, out, mda_safet_tensor(f, name));
}

bool mda_teacher_plain(float *w, const mda_safet_file_t *f, uint32_t l, const char *suffix, uint64_t off, uint32_t n) {
    char name[MDA_SAFET_MAX_KEY];
    snprintf(name, sizeof(name), "model.layers.%u.%s", l, suffix);
    return mda_safet_plain(w + off, n, mda_safet_tensor(f, name));
}

uint32_t mda_teacher_fill_layer(float *w, const mda_teacher_t *t, const mda_safet_file_t *f, uint32_t l) {
    const mda_teacher_layer_t *ly = &t->layers[l];
    mda_teacher_dims_t dm = t->dims;
    uint32_t bad = 0;
    bad += mda_teacher_plain(w, f, l, "input_layernorm.weight", ly->ln1, dm.d) == false;
    bad += mda_teacher_plain(w, f, l, "post_attention_layernorm.weight", ly->ln2, dm.d) == false;
    bad += mda_teacher_linear(w, f, l, "self_attn.q_proj.weight", ly->wq, dm.d, dm.d) == false;
    bad += mda_teacher_linear(w, f, l, "self_attn.k_proj.weight", ly->wk, dm.d, dm.kv) == false;
    bad += mda_teacher_linear(w, f, l, "self_attn.v_proj.weight", ly->wv, dm.d, dm.kv) == false;
    bad += mda_teacher_linear(w, f, l, "self_attn.o_proj.weight", ly->wo, dm.d, dm.d) == false;
    bad += mda_teacher_linear(w, f, l, "mlp.gate_proj.weight", ly->gate, dm.d, dm.ffn) == false;
    bad += mda_teacher_linear(w, f, l, "mlp.up_proj.weight", ly->up, dm.d, dm.ffn) == false;
    bad += mda_teacher_linear(w, f, l, "mlp.down_proj.weight", ly->down, dm.ffn, dm.d) == false;
    return bad;
}

uint32_t mda_teacher_fill(float *w, const mda_teacher_t *t, const mda_safet_file_t *f) {
    mda_teacher_dims_t dm = t->dims;
    uint32_t bad = 0;
    bad += mda_safet_plain(w + t->emb, dm.vocab * dm.d, mda_safet_tensor(f, "model.embed_tokens.weight")) == false;
    bad += mda_safet_plain(w + t->norm, dm.d, mda_safet_tensor(f, "model.norm.weight")) == false;
    for (uint32_t l = 0; l < dm.n_layer; l += 1) {
        bad += mda_teacher_fill_layer(w, t, f, l);
    }
    return bad;
}

void mda_teacher_alloc(mda_teacher_t *t) {
    mda_vkc_t *c = t->m->c;
    mda_teacher_dims_t dm = t->dims;
    VkDeviceSize r = t->rows;
    t->w32 = mda_vkc_alloc(c, t->n_all * sizeof(float));
    t->w16 = mda_vkc_alloc(c, t->n_all * sizeof(uint16_t));
    t->embt16 = mda_vkc_alloc(c, (VkDeviceSize)dm.vocab * dm.d * sizeof(uint16_t));
    t->x = mda_vkc_alloc(c, r * dm.d * sizeof(float));
    t->h = mda_vkc_alloc(c, r * dm.d * sizeof(float));
    t->h16 = mda_vkc_alloc(c, r * dm.d * sizeof(uint16_t));
    t->q = mda_vkc_alloc(c, r * dm.d * sizeof(float));
    t->k = mda_vkc_alloc(c, r * dm.kv * sizeof(float));
    t->v = mda_vkc_alloc(c, r * dm.kv * sizeof(float));
    t->att = mda_vkc_alloc(c, r * dm.d * sizeof(float));
    t->att16 = mda_vkc_alloc(c, r * dm.d * sizeof(uint16_t));
    t->gate = mda_vkc_alloc(c, r * dm.ffn * sizeof(float));
    t->up = mda_vkc_alloc(c, r * dm.ffn * sizeof(float));
    t->act = mda_vkc_alloc(c, r * dm.ffn * sizeof(float));
    t->act16 = mda_vkc_alloc(c, r * dm.ffn * sizeof(uint16_t));
    t->rms = mda_vkc_alloc(c, r * sizeof(float));
    t->logits = mda_vkc_alloc(c, r * dm.vocab * sizeof(float));
    t->logz = mda_vkc_alloc(c, r * sizeof(float));
    if (t->keep) {
        t->xs = mda_vkc_alloc(c, dm.n_layer * r * dm.d * sizeof(float));
        t->ao = mda_vkc_alloc(c, dm.n_layer * r * dm.d * sizeof(float));
    }
    t->k_rope = mda_vkc_load(c, "build/rope.spv", 1, sizeof(mda_pc_rope_t));
    t->k_attn = mda_vkc_load(c, "build/attn.spv", 4, sizeof(mda_pc_attn_t));
}

mda_vkc_buf_t mda_teacher_w32(mda_teacher_t *t, uint64_t off, uint64_t n) {
    return mda_vkc_view(&t->w32, off * sizeof(float), n * sizeof(float));
}

mda_vkc_buf_t mda_teacher_w16(mda_teacher_t *t, uint64_t off, uint64_t n) {
    return mda_vkc_view(&t->w16, off * sizeof(uint16_t), n * sizeof(uint16_t));
}

bool mda_teacher_upload(mda_teacher_t *t, const float *w) {
    mda_model_t *m = t->m;
    mda_teacher_dims_t dm = t->dims;
    mda_vkc_write(m->c, &t->w32, w);
    mda_vkc_begin(m->c);
    mda_fwd_cast(m, &t->w32, &t->w16, (uint32_t)t->n_all, MDA_PC_NO_SLOT);
    mda_vkc_buf_t emb = mda_teacher_w32(t, t->emb, (uint64_t)dm.vocab * dm.d);
    mda_fwd_transpose16(m, &emb, &t->embt16, dm.vocab, dm.d, MDA_PC_NO_SLOT);
    mda_vkc_flush(m->c);
    return m->c->fail == false;
}

bool mda_teacher_build(mda_teacher_t *t, const mda_safet_file_t *f) {
    mda_teacher_dims_t dm = t->dims;
    if (dm.vocab != t->m->cfg.vocab || dm.d % 64 != 0 || dm.kv % 64 != 0 || dm.n_layer == 0 || t->rows % t->seqlen != 0) {
        fprintf(stderr, "teacher: vocab %u d %u kv %u layers %u does not fit the student's vocab %u\n", dm.vocab, dm.d, dm.kv, dm.n_layer, t->m->cfg.vocab);
        return false;
    } else {
        mda_teacher_layout(t);
        mda_teacher_alloc(t);
        float *w = calloc(t->n_all, sizeof(float));
        uint32_t bad = mda_teacher_fill(w, t, f);
        bool ok = bad == 0 && t->m->c->fail == false && mda_teacher_upload(t, w);
        free(w);
        return ok;
    }
}

mda_teacher_t *mda_teacher_from(mda_model_t *m, const mda_safet_file_t *f, uint32_t rows, bool keep) {
    mda_teacher_t *t = calloc(1, sizeof(mda_teacher_t));
    t->m = m;
    t->rows = rows;
    t->keep = keep;
    t->seqlen = m->cfg.seqlen;
    t->teach = 1.0f;
    t->dims = mda_teacher_dims(f);
    if (mda_teacher_build(t, f)) {
        fprintf(stderr, "teacher: %u layers, d %u, kv %u, ffn %u, %u rows\n", t->dims.n_layer, t->dims.d, t->dims.kv, t->dims.ffn, rows);
        return t;
    } else {
        free(t);
        return NULL;
    }
}

mda_teacher_t *mda_teacher_load(mda_model_t *m, const char *path, uint32_t rows, bool keep) {
    mda_safet_file_t f = mda_safet_read(path);
    if (f.bytes == NULL) {
        return NULL;
    } else {
        mda_teacher_t *t = mda_teacher_from(m, &f, rows, keep);
        mda_safet_free(&f);
        return t;
    }
}

void mda_teacher_rmsnorm(mda_teacher_t *t, uint64_t g_off) {
    mda_model_t *m = t->m;
    mda_vkc_buf_t g = mda_teacher_w32(t, g_off, t->dims.d);
    mda_pc_rms_t pc = {
        .rows = t->rows,
        .d = t->dims.d,
        .op = 0,
    };
    mda_vkc_buf_t *bufs[6] = {
        &t->x,
        &g,
        &t->h,
        &t->rms,
        &t->rms,
        &t->rms,
    };
    mda_vkc_run(m->c, &m->k_rms, bufs, t->rows, 1, 1, &pc);
}

void mda_teacher_mm(mda_teacher_t *t, mda_vkc_buf_t *x16, uint32_t sx, uint64_t off, uint32_t n, uint32_t k, mda_vkc_buf_t *out) {
    mda_vkc_buf_t w = mda_teacher_w16(t, off, (uint64_t)k * n);
    mda_fwd_mm16_raw(t->m, x16, sx, &w, MDA_PC_NO_SLOT, out, t->rows, n, k, false, MDA_FWD_AB);
}

void mda_teacher_residual(mda_teacher_t *t, mda_vkc_buf_t *x16, uint32_t sx, uint64_t off, uint32_t k) {
    mda_vkc_buf_t w = mda_teacher_w16(t, off, (uint64_t)k * t->dims.d);
    mda_fwd_epi_t epi = {
        .prev = &t->x,
        .alpha = 1.0f,
        .beta = 1.0f,
        .store = MDA_FWD_STORE32,
    };
    mda_fwd_mm16_epi(t->m, x16, sx, &w, MDA_PC_NO_SLOT, &t->x, &t->x, t->rows, t->dims.d, k, MDA_FWD_AB, epi);
}

void mda_teacher_rope(mda_teacher_t *t, mda_vkc_buf_t *x, uint32_t heads) {
    mda_pc_rope_t pc = {
        .rows = t->rows,
        .heads = heads,
        .seqlen = t->seqlen,
        .theta = MDA_TEACHER_ROPE_THETA,
    };
    mda_vkc_buf_t *bufs[1] = {
        x,
    };
    mda_vkc_run(t->m->c, &t->k_rope, bufs, (t->rows * heads * 32 + 255) / 256, 1, 1, &pc);
}

void mda_teacher_attn(mda_teacher_t *t) {
    mda_pc_attn_t pc = {
        .seqlen = t->seqlen,
        .heads = t->dims.d / 64,
        .kv_heads = t->dims.kv / 64,
    };
    mda_vkc_buf_t *bufs[4] = {
        &t->q,
        &t->k,
        &t->v,
        &t->att,
    };
    mda_vkc_run(t->m->c, &t->k_attn, bufs, t->rows / t->seqlen * pc.heads, (t->seqlen + 63) / 64, 1, &pc);
}

void mda_teacher_ffn(mda_teacher_t *t, const mda_teacher_layer_t *ly) {
    mda_model_t *m = t->m;
    mda_teacher_dims_t dm = t->dims;
    mda_teacher_rmsnorm(t, ly->ln2);
    uint32_t sh = mda_fwd_cast_act(m, &t->h, &t->h16, t->rows * dm.d);
    mda_vkc_set_nobar(m->c, true);
    mda_teacher_mm(t, &t->h16, sh, ly->gate, dm.ffn, dm.d, &t->gate);
    mda_vkc_set_nobar(m->c, false);
    mda_teacher_mm(t, &t->h16, sh, ly->up, dm.ffn, dm.d, &t->up);
    uint32_t n = t->rows * dm.ffn;
    uint32_t sa = mda_fwd_slot(m);
    mda_pc_elem_t pc = {
        .n = n,
        .op = 0,
        .slot = sa,
        .slot2 = MDA_PC_NO_SLOT,
    };
    mda_vkc_buf_t *bufs[5] = {
        &t->act,
        &t->gate,
        &t->up,
        &t->act,
        &m->amax,
    };
    mda_vkc_run(m->c, &m->k_elem, bufs, mda_util_min_u32((n + 255) / 256, MDA_FWD_ELEM_GROUPS), 1, 1, &pc);
    mda_fwd_cast(m, &t->act, &t->act16, n, sa);
    mda_teacher_residual(t, &t->act16, sa, ly->down, dm.ffn);
}

mda_vkc_buf_t mda_teacher_layer_view(mda_teacher_t *t, mda_vkc_buf_t *b, uint32_t l) {
    VkDeviceSize n = (VkDeviceSize)t->rows * t->dims.d * sizeof(float);
    return mda_vkc_view(b, l * n, n);
}

void mda_teacher_attn_out(mda_teacher_t *t, const mda_teacher_layer_t *ly, uint32_t l, uint32_t sa) {
    mda_model_t *m = t->m;
    mda_teacher_dims_t dm = t->dims;
    if (t->keep) {
        mda_vkc_buf_t ao = mda_teacher_layer_view(t, &t->ao, l);
        mda_teacher_mm(t, &t->att16, sa, ly->wo, dm.d, dm.d, &ao);
        mda_fwd_vec_op(m, &t->x, &ao, t->rows * dm.d, MDA_PC_VEC_ADD);
    } else {
        mda_teacher_residual(t, &t->att16, sa, ly->wo, dm.d);
    }
}

void mda_teacher_layer(mda_teacher_t *t, const mda_teacher_layer_t *ly, uint32_t l) {
    mda_model_t *m = t->m;
    mda_teacher_dims_t dm = t->dims;
    if (t->keep) {
        mda_vkc_buf_t xs = mda_teacher_layer_view(t, &t->xs, l);
        mda_fwd_vec_op(m, &xs, &t->x, t->rows * dm.d, MDA_PC_VEC_COPY);
    }
    mda_teacher_rmsnorm(t, ly->ln1);
    uint32_t sh = mda_fwd_cast_act(m, &t->h, &t->h16, t->rows * dm.d);
    mda_vkc_set_nobar(m->c, true);
    mda_teacher_mm(t, &t->h16, sh, ly->wq, dm.d, dm.d, &t->q);
    mda_teacher_mm(t, &t->h16, sh, ly->wk, dm.kv, dm.d, &t->k);
    mda_vkc_set_nobar(m->c, false);
    mda_teacher_mm(t, &t->h16, sh, ly->wv, dm.kv, dm.d, &t->v);
    mda_vkc_set_nobar(m->c, true);
    mda_teacher_rope(t, &t->q, dm.d / 64);
    mda_vkc_set_nobar(m->c, false);
    mda_teacher_rope(t, &t->k, dm.kv / 64);
    mda_teacher_attn(t);
    uint32_t sa = mda_fwd_cast_act(m, &t->att, &t->att16, t->rows * dm.d);
    mda_teacher_attn_out(t, ly, l, sa);
    mda_teacher_ffn(t, ly);
}

void mda_teacher_head(mda_teacher_t *t) {
    mda_model_t *m = t->m;
    mda_teacher_rmsnorm(t, t->norm);
    uint32_t sx = mda_fwd_cast_act(m, &t->h, &t->h16, t->rows * t->dims.d);
    mda_fwd_mm16_raw(m, &t->h16, sx, &t->embt16, MDA_PC_NO_SLOT, &t->logits, t->rows, t->dims.vocab, t->dims.d, false, MDA_FWD_AB);
}

void mda_teacher_embed(mda_teacher_t *t, mda_vkc_buf_t *ids) {
    mda_model_t *m = t->m;
    mda_vkc_buf_t emb = mda_teacher_w32(t, t->emb, (uint64_t)t->dims.vocab * t->dims.d);
    mda_pc_emb_t pc = {
        .bt = t->rows,
        .d = t->dims.d,
        .op = 0,
        .vocab = t->dims.vocab,
    };
    mda_vkc_buf_t *bufs[4] = {
        ids,
        &emb,
        &emb,
        &t->x,
    };
    mda_vkc_run(m->c, &m->k_embed, bufs, (t->rows * t->dims.d + 255) / 256, 1, 1, &pc);
}

void mda_teacher_rec(mda_teacher_t *t, mda_vkc_buf_t *ids) {
    mda_model_t *m = t->m;
    m->n_slots = 0;
    mda_vkc_fill(m->c, &m->amax, 0);
    mda_teacher_embed(t, ids);
    for (uint32_t l = 0; l < t->dims.n_layer; l += 1) {
        mda_teacher_layer(t, &t->layers[l], l);
    }
    if (t->keep == false) {
        mda_teacher_head(t);
    }
}
