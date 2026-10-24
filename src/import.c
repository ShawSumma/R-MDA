#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "import.h"
#include "vkc.h"
#include "util.h"
#include "safet.h"
#include "tokenizer.h"
#include "model.h"
#include "ckpt.h"

bool mda_import_one(float *w, const mda_safet_file_t *f, const char *name, mda_model_param_t *p, bool linear) {
    mda_safet_tensor_t t = mda_safet_tensor(f, name);
    bool ok = false;
    if (linear) {
        ok = mda_safet_linear(w + p->off, p->rows, p->cols, t);
    } else {
        ok = mda_safet_plain(w + p->off, p->n, t);
    }
    if (t.ok && ok == false) {
        fprintf(stderr, "import: %s: shape %u x %u does not fit %s\n", name, t.rows, t.cols, p->name);
    }
    return ok;
}

uint32_t mda_import_layer(float *w, const mda_safet_file_t *f, mda_model_layer_t *ly, uint32_t l) {
    mda_import_map_t maps[MDA_IMPORT_LAYER_MAPS] = {
        {"self_attn.q_proj.weight", ly->wr, true},
        {"self_attn.k_proj.weight", ly->wk, true},
        {"self_attn.v_proj.weight", ly->wv, true},
        {"self_attn.o_proj.weight", ly->wo, true},
        {"mlp.gate_proj.weight", ly->w_gate, true},
        {"mlp.up_proj.weight", ly->w_up, true},
        {"mlp.down_proj.weight", ly->w_down, true},
        {"input_layernorm.weight", ly->n1, false},
        {"post_attention_layernorm.weight", ly->n2, false},
    };
    uint32_t bad = 0;
    for (uint32_t i = 0; i < MDA_IMPORT_LAYER_MAPS; i += 1) {
        char name[MDA_SAFET_MAX_KEY];
        snprintf(name, sizeof(name), "model.layers.%u.%s", l, maps[i].suffix);
        bad += mda_import_one(w, f, name, maps[i].p, maps[i].linear) == false;
    }
    for (uint32_t j = 0; j < ly->shift_mix->n; j += 1) {
        w[ly->shift_mix->off + j] = 1.0f;
    }
    return bad;
}

uint32_t mda_import_weights(float *w, const mda_safet_file_t *f, mda_model_t *m) {
    uint32_t bad = 0;
    bad += mda_import_one(w, f, "model.embed_tokens.weight", m->tok_emb, false) == false;
    bad += mda_import_one(w, f, "model.norm.weight", m->n_out, false) == false;
    for (uint32_t l = 0; l < m->cfg.n_layer; l += 1) {
        bad += mda_import_layer(w, f, &m->layers[l], l);
    }
    return bad;
}

bool mda_import_save(mda_model_t *m, const mda_tokenizer_t *tok, float *w, const char *out) {
    mda_vkc_write(m->c, &m->w_all, w);
    m->dirty16 = true;
    bool saved = mda_ckpt_save(m, mda_tokenizer_hash(tok), 0, out);
    if (saved) {
        fprintf(stderr, "%s: %.2f M params, %u layers imported\n", out, (double)mda_model_param_count(m) / 1e6, m->cfg.n_layer);
    }
    return saved;
}

bool mda_import_model(mda_vkc_t *c, const mda_tokenizer_t *tok, const mda_safet_file_t *f, const char *out) {
    mda_model_t *m = mda_model_init(c, mda_model_cfg(tok->count));
    if (m == NULL) {
        return false;
    } else {
        float *w = mda_vkc_read(c, &m->w_all);
        uint32_t bad = mda_import_weights(w, f, m);
        bool ok = bad == 0 && mda_import_save(m, tok, w, out);
        free(w);
        return ok;
    }
}

bool mda_import_device(mda_vkc_t *c, const char *src, const char *out) {
    mda_tokenizer_t tok = mda_tokenizer_load(MDA_TOKENIZER_VOCAB_PATH);
    mda_safet_file_t f = mda_safet_read(src);
    if (tok.entries == NULL || f.bytes == NULL) {
        return false;
    } else {
        bool ok = mda_import_model(c, &tok, &f, out);
        mda_safet_free(&f);
        return ok;
    }
}

bool mda_import_main(uint32_t argc, char **argv) {
    const char *src = mda_util_arg(argc, argv, 1, MDA_IMPORT_WEIGHTS_PATH);
    const char *out = mda_util_arg(argc, argv, 2, MDA_IMPORT_OUT_PATH);
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_import_device(&c, src, out);
    }
}

int main(int argc, char **argv) {
    bool ok = mda_import_main((uint32_t)argc, argv);
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
