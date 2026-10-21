#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ckpt.h"
#include "util.h"

mda_ckpt_hdr_t mda_ckpt_hdr(const mda_model_t *m, uint64_t vocab_hash, uint32_t step) {
    mda_ckpt_hdr_t h;
    memset(&h, 0, sizeof(h));
    h.magic = MDA_CKPT_MAGIC;
    h.cfg = m->cfg;
    h.step = step;
    h.vocab_hash = vocab_hash;
    h.n_all = m->n_all;
    return h;
}

mda_ckpt_res_t mda_ckpt_bad(void) {
    return (mda_ckpt_res_t) {
        .step = 0,
        .ok = false,
    };
}

bool mda_ckpt_write_arenas(mda_model_t *m, FILE *f, mda_vkc_buf_t *stage) {
    VkDeviceSize bytes = (VkDeviceSize)m->n_all * sizeof(float);
    mda_vkc_buf_t *arenas[3] = {
        &m->w_all,
        &m->m_all,
        &m->v_all,
    };
    uint32_t i = 0;
    size_t wrote = m->n_all;
    while (i < 3 && m->c->fail == false && wrote == m->n_all) {
        mda_vkc_copy(m->c, stage, arenas[i], bytes);
        wrote = fwrite(stage->ptr, sizeof(float), m->n_all, f);
        i += 1;
    }
    return m->c->fail == false && wrote == m->n_all;
}

bool mda_ckpt_write(mda_model_t *m, uint64_t vocab_hash, uint32_t step, FILE *f) {
    mda_ckpt_hdr_t h = mda_ckpt_hdr(m, vocab_hash, step);
    size_t wrote = fwrite(&h, sizeof(h), 1, f);
    if (wrote != 1) {
        return false;
    } else {
        mda_vkc_buf_t stage = mda_vkc_alloc_cached(m->c, (VkDeviceSize)m->n_all * sizeof(float));
        bool ok = mda_ckpt_write_arenas(m, f, &stage);
        mda_vkc_free(m->c, &stage);
        return ok;
    }
}

bool mda_ckpt_same_shape(mda_model_cfg_t a, mda_model_cfg_t b) {
    return a.d == b.d && a.n_layer == b.n_layer && a.n_head == b.n_head && a.ffn_hidden == b.ffn_hidden && a.vocab == b.vocab;
}

const char *mda_ckpt_mismatch(mda_ckpt_hdr_t h, mda_ckpt_hdr_t want, size_t got, int64_t size, VkDeviceSize bytes) {
    if (got != 1 || h.magic != want.magic) {
        return "not a checkpoint of this format";
    } else if (mda_ckpt_same_shape(h.cfg, want.cfg) == false || h.n_all != want.n_all) {
        return "checkpoint is for a different model config";
    } else if (h.vocab_hash != want.vocab_hash) {
        return "checkpoint was trained with a different vocab";
    } else if ((uint64_t)size != sizeof(h) + 3 * bytes) {
        return "wrong size (truncated?)";
    } else {
        return NULL;
    }
}

uint32_t mda_ckpt_arenas(bool moments) {
    if (moments) {
        return 3;
    } else {
        return 1;
    }
}

bool mda_ckpt_read_arenas(mda_model_t *m, FILE *f, bool moments, const char *path) {
    VkDeviceSize bytes = (VkDeviceSize)m->n_all * sizeof(float);
    mda_vkc_buf_t stage = mda_vkc_alloc_cached(m->c, bytes);
    mda_vkc_buf_t *arenas[3] = {
        &m->w_all,
        &m->m_all,
        &m->v_all,
    };
    uint32_t n = mda_ckpt_arenas(moments);
    uint32_t i = 0;
    size_t got = m->n_all;
    while (i < n && m->c->fail == false && got == m->n_all) {
        got = fread(stage.ptr, sizeof(float), m->n_all, f);
        if (got == m->n_all) {
            mda_vkc_copy(m->c, arenas[i], &stage, bytes);
        }
        i += 1;
    }
    mda_vkc_free(m->c, &stage);
    if (got != m->n_all) {
        fprintf(stderr, "%s: short read\n", path);
    }
    return m->c->fail == false && got == m->n_all;
}

mda_ckpt_res_t mda_ckpt_read_body(mda_model_t *m, FILE *f, bool moments, const char *path, uint32_t step) {
    bool read = mda_ckpt_read_arenas(m, f, moments, path);
    if (read) {
        m->dirty16 = true;
        return (mda_ckpt_res_t) {
            .step = step,
            .ok = true,
        };
    } else {
        return mda_ckpt_bad();
    }
}

mda_ckpt_res_t mda_ckpt_read(mda_model_t *m, uint64_t vocab_hash, FILE *f, bool moments, const char *path) {
    mda_ckpt_hdr_t want = mda_ckpt_hdr(m, vocab_hash, 0);
    mda_ckpt_hdr_t h;
    fseek(f, 0, SEEK_END);
    int64_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t got = fread(&h, sizeof(h), 1, f);
    const char *why = mda_ckpt_mismatch(h, want, got, size, (VkDeviceSize)m->n_all * sizeof(float));
    if (why != NULL) {
        fprintf(stderr, "%s: %s\n", path, why);
        return mda_ckpt_bad();
    } else {
        return mda_ckpt_read_body(m, f, moments, path, h.step);
    }
}

bool mda_ckpt_save_file(mda_model_t *m, uint64_t vocab_hash, uint32_t step, FILE *f, const char *tmp, const char *path) {
    bool wrote = mda_ckpt_write(m, vocab_hash, step, f);
    int32_t flushed = fflush(f);
    int32_t synced = fsync(fileno(f));
    int32_t closed = fclose(f);
    if (wrote == false || flushed != 0 || synced != 0 || closed != 0) {
        perror(tmp);
        remove(tmp);
        return false;
    } else {
        int32_t moved = rename(tmp, path);
        if (moved != 0) {
            perror(tmp);
            remove(tmp);
        }
        return moved == 0;
    }
}

bool mda_ckpt_save_open(mda_model_t *m, uint64_t vocab_hash, uint32_t step, const char *tmp, const char *path) {
    FILE *f = fopen(tmp, "wb");
    if (f == NULL) {
        perror(tmp);
        return false;
    } else {
        return mda_ckpt_save_file(m, vocab_hash, step, f, tmp, path);
    }
}

bool mda_ckpt_save(mda_model_t *m, uint64_t vocab_hash, uint32_t step, const char *path) {
    char *tmp = mda_util_path_suffix(path, ".tmp");
    bool saved = mda_ckpt_save_open(m, vocab_hash, step, tmp, path);
    free(tmp);
    return saved;
}

mda_ckpt_res_t mda_ckpt_load(mda_model_t *m, uint64_t vocab_hash, const char *path, bool moments) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return (mda_ckpt_res_t) {
            .step = 0,
            .ok = true,
        };
    } else {
        mda_ckpt_res_t r = mda_ckpt_read(m, vocab_hash, f, moments, path);
        fclose(f);
        return r;
    }
}
