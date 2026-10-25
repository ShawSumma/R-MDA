#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "import.h"
#include "vkc.h"
#include "util.h"
#include "safet.h"
#include "tokenizer.h"
#include "model.h"
#include "ckpt.h"

mda_safet_tensor_t mda_import_tensor(const mda_safet_file_t *f, uint32_t l, const char *suffix) {
    char name[MDA_SAFET_MAX_KEY];
    snprintf(name, sizeof(name), "blocks.%u.%s", l, suffix);
    return mda_safet_tensor(f, name);
}

bool mda_import_one(float *w, mda_safet_tensor_t t, const char *name, mda_model_param_t *p, bool linear) {
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

bool mda_import_padded(float *w, mda_safet_tensor_t t, const char *name, mda_model_param_t *p) {
    bool ok = t.ok && t.rows <= p->rows && t.cols <= p->cols;
    if (ok) {
        memset(w + p->off, 0, (size_t)p->n * sizeof(float));
        for (uint32_t r = 0; r < t.rows; r += 1) {
            for (uint32_t c = 0; c < t.cols; c += 1) {
                w[p->off + (uint64_t)r * p->cols + c] = mda_safet_bf16(t.data, (uint64_t)r * t.cols + c);
            }
        }
    } else if (t.ok) {
        fprintf(stderr, "import: %s: shape %u x %u does not fit %s\n", name, t.rows, t.cols, p->name);
    }
    return ok;
}

uint32_t mda_import_mixes(float *w, const mda_safet_file_t *f, mda_model_layer_t *ly, uint32_t l) {
    const char *names[MDA_MODEL_MIXES] = {
        "att.x_r",
        "att.x_w",
        "att.x_k",
        "att.x_v",
        "att.x_a",
        "att.x_g",
    };
    uint32_t d = ly->att_mix->n / MDA_MODEL_MIXES;
    uint32_t bad = 0;
    for (uint32_t s = 0; s < MDA_MODEL_MIXES; s += 1) {
        mda_safet_tensor_t t = mda_import_tensor(f, l, names[s]);
        bad += (t.ok && mda_safet_plain(w + ly->att_mix->off + s * d, d, t)) == false;
    }
    return bad;
}

uint32_t mda_import_loras(float *w, const mda_safet_file_t *f, mda_model_layer_t *ly, uint32_t l) {
    const char *names[MDA_MODEL_LORAS][2] = {
        {"att.w1", "att.w2"},
        {"att.a1", "att.a2"},
        {"att.v1", "att.v2"},
        {"att.g1", "att.g2"},
    };
    uint32_t bad = 0;
    for (uint32_t i = 0; i < MDA_MODEL_LORAS; i += 1) {
        bad += mda_import_padded(w, mda_import_tensor(f, l, names[i][0]), names[i][0], ly->lora[i].a) == false;
        bad += mda_import_padded(w, mda_import_tensor(f, l, names[i][1]), names[i][1], ly->lora[i].b) == false;
    }
    return bad;
}

uint32_t mda_import_layer(float *w, const mda_safet_file_t *f, mda_model_layer_t *ly, uint32_t l) {
    mda_import_map_t maps[MDA_IMPORT_LAYER_MAPS] = {
        {"ln1.weight", ly->ln1_w, false},
        {"ln1.bias", ly->ln1_b, false},
        {"att.receptance.weight", ly->wr, true},
        {"att.key.weight", ly->wk, true},
        {"att.value.weight", ly->wv, true},
        {"att.output.weight", ly->wo, true},
        {"att.ln_x.weight", ly->lnx_w, false},
        {"att.ln_x.bias", ly->lnx_b, false},
        {"ln2.weight", ly->ln2_w, false},
        {"ln2.bias", ly->ln2_b, false},
        {"ffn.x_k", ly->ffn_mix, false},
        {"ffn.key.weight", ly->w_key, true},
        {"ffn.value.weight", ly->w_val, true},
    };
    mda_import_map_t vecs[6] = {
        {"att.w0", ly->w0, false},
        {"att.a0", ly->a0, false},
        {"att.v0", ly->v0, false},
        {"att.k_k", ly->kk, false},
        {"att.k_a", ly->ka, false},
        {"att.r_k", ly->rk, false},
    };
    uint32_t bad = 0;
    for (uint32_t i = 0; i < MDA_IMPORT_LAYER_MAPS; i += 1) {
        bad += mda_import_one(w, mda_import_tensor(f, l, maps[i].suffix), maps[i].suffix, maps[i].p, maps[i].linear) == false;
    }
    for (uint32_t i = 0; i < 6; i += 1) {
        bad += mda_import_one(w, mda_import_tensor(f, l, vecs[i].suffix), vecs[i].suffix, vecs[i].p, false) == false;
    }
    return bad + mda_import_mixes(w, f, ly, l) + mda_import_loras(w, f, ly, l);
}

uint32_t mda_import_weights(float *w, const mda_safet_file_t *f, mda_model_t *m) {
    uint32_t bad = 0;
    bad += mda_import_one(w, mda_safet_tensor(f, "emb.weight"), "emb.weight", m->tok_emb, false) == false;
    bad += mda_import_one(w, mda_safet_tensor(f, "blocks.0.ln0.weight"), "ln0.weight", m->ln0_w, false) == false;
    bad += mda_import_one(w, mda_safet_tensor(f, "blocks.0.ln0.bias"), "ln0.bias", m->ln0_b, false) == false;
    bad += mda_import_one(w, mda_safet_tensor(f, "ln_out.weight"), "ln_out.weight", m->ln_out_w, false) == false;
    bad += mda_import_one(w, mda_safet_tensor(f, "ln_out.bias"), "ln_out.bias", m->ln_out_b, false) == false;
    bad += mda_import_one(w, mda_safet_tensor(f, "head.weight"), "head.weight", m->head, true) == false;
    for (uint32_t l = 0; l < m->cfg.n_layer; l += 1) {
        bad += mda_import_layer(w, f, &m->layers[l], l);
    }
    return bad;
}

uint32_t mda_import_layers(const mda_safet_file_t *f) {
    uint32_t n = 0;
    char name[MDA_SAFET_MAX_KEY];
    snprintf(name, sizeof(name), "blocks.%u.ln1.weight", n);
    while (mda_safet_has(f, name)) {
        n += 1;
        snprintf(name, sizeof(name), "blocks.%u.ln1.weight", n);
    }
    return n;
}

uint32_t mda_import_rank(const mda_safet_file_t *f, const char *suffix) {
    mda_safet_tensor_t t = mda_import_tensor(f, 0, suffix);
    return (t.cols + MDA_MODEL_LORA_ALIGN - 1) / MDA_MODEL_LORA_ALIGN * MDA_MODEL_LORA_ALIGN;
}

mda_model_cfg_t mda_import_cfg(const mda_safet_file_t *f, uint32_t vocab) {
    mda_model_cfg_t cfg = mda_model_cfg(vocab);
    mda_safet_tensor_t emb = mda_safet_tensor(f, "emb.weight");
    mda_safet_tensor_t key = mda_import_tensor(f, 0, "ffn.key.weight");
    cfg.d = emb.cols;
    cfg.n_head = emb.cols / 64;
    cfg.vocab = emb.rows;
    cfg.n_layer = mda_import_layers(f);
    cfg.ffn_hidden = key.rows;
    cfg.lora[MDA_MODEL_LORA_W] = mda_import_rank(f, "att.w1");
    cfg.lora[MDA_MODEL_LORA_A] = mda_import_rank(f, "att.a1");
    cfg.lora[MDA_MODEL_LORA_V] = mda_import_rank(f, "att.v1");
    cfg.lora[MDA_MODEL_LORA_G] = mda_import_rank(f, "att.g1");
    return cfg;
}

bool mda_import_save(mda_model_t *m, const mda_tokenizer_t *tok, float *w, const char *out) {
    mda_vkc_write(m->c, &m->w_all, w);
    m->dirty16 = true;
    bool saved = mda_ckpt_save(m, mda_tokenizer_hash(tok), 0, out);
    if (saved) {
        fprintf(stderr, "%s: %.2f M params, d %u, %u layers, ffn %u, vocab %u\n", out, (double)mda_model_param_count(m) / 1e6, m->cfg.d, m->cfg.n_layer, m->cfg.ffn_hidden, m->cfg.vocab);
    }
    return saved;
}

bool mda_import_model(mda_vkc_t *c, const mda_tokenizer_t *tok, const mda_safet_file_t *f, const char *out) {
    mda_model_t *m = mda_model_init(c, mda_import_cfg(f, tok->count));
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
