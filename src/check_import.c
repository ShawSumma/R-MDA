#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "check_import.h"
#include "vkc.h"
#include "util.h"
#include "tokenizer.h"
#include "model.h"
#include "fwd.h"
#include "ckpt.h"
#include "import.h"

mda_tokenizer_ids_t mda_check_import_text(const mda_tokenizer_t *tok, const char *path) {
    FILE *f = fopen(path, "rb");
    char *text = malloc(MDA_CHECK_IMPORT_TEXT);
    size_t got = 0;
    if (f == NULL) {
        perror(path);
    } else {
        got = fread(text, 1, MDA_CHECK_IMPORT_TEXT, f);
        fclose(f);
    }
    mda_tokenizer_ids_t ids = mda_tokenizer_encode(tok, text, (uint32_t)got);
    free(text);
    return ids;
}

bool mda_check_import_write(const uint32_t *ids, const float *logits, uint32_t n, uint32_t vocab) {
    FILE *f = fopen(MDA_CHECK_IMPORT_OUT, "wb");
    if (f == NULL) {
        perror(MDA_CHECK_IMPORT_OUT);
        return false;
    } else {
        uint32_t hdr[2] = {n, vocab};
        size_t a = fwrite(hdr, sizeof(uint32_t), 2, f);
        size_t b = fwrite(ids, sizeof(uint32_t), n, f);
        size_t c = fwrite(logits, sizeof(float), (size_t)n * vocab, f);
        int32_t closed = fclose(f);
        return a == 2 && b == n && c == (size_t)n * vocab && closed == 0;
    }
}

bool mda_check_import_run(mda_model_t *m, const mda_tokenizer_ids_t *ids) {
    uint32_t n = m->cfg.seqlen;
    mda_fwd_scratch_t sc = mda_fwd_alloc(m, 1);
    if (m->c->fail || ids->n < n) {
        fprintf(stderr, "check_import: fewer than %u tokens of text\n", n);
        return false;
    } else {
        for (uint32_t i = 0; i < n; i += 1) {
            sc.batch.ids[i] = ids->ids[i];
            sc.batch.target[i] = ids->ids[i];
            sc.batch.mask[i] = 0.0f;
            sc.batch.weight[i] = 1.0f;
        }
        m->infer = true;
        m->ref32 = true;
        m->dirty16 = true;
        mda_fwd_run(m, &sc);
        float *logits = mda_vkc_read(m->c, &sc.logits);
        bool ok = m->c->fail == false && mda_check_import_write(ids->ids, logits, n, m->cfg.vocab);
        free(logits);
        if (ok) {
            printf("%s: logits for %u tokens\n", MDA_CHECK_IMPORT_OUT, n);
        }
        return ok;
    }
}

bool mda_check_import_model(mda_vkc_t *c, const mda_tokenizer_t *tok, const char *path, const char *text) {
    mda_model_cfg_t base = mda_model_cfg(tok->count);
    base.batch = 1;
    base.seqlen = MDA_CHECK_IMPORT_LEN;
    mda_model_t *m = mda_model_init(c, mda_ckpt_shape(path, base));
    if (m == NULL) {
        return false;
    } else {
        m->causal = true;
        mda_ckpt_res_t loaded = mda_ckpt_load(m, mda_tokenizer_hash(tok), path, false);
        mda_tokenizer_ids_t ids = mda_check_import_text(tok, text);
        bool ok = loaded.ok && loaded.found && mda_check_import_run(m, &ids);
        free(ids.ids);
        return ok;
    }
}

bool mda_check_import_main(uint32_t argc, char **argv) {
    const char *path = mda_util_arg(argc, argv, 1, MDA_IMPORT_OUT_PATH);
    const char *text = mda_util_arg(argc, argv, 2, MDA_TOKENIZER_CORPUS_PATH);
    mda_tokenizer_t tok = mda_tokenizer_load(MDA_TOKENIZER_VOCAB_PATH);
    mda_vkc_t c = mda_vkc_init();
    if (tok.entries == NULL || c.fail) {
        return false;
    } else {
        return mda_check_import_model(&c, &tok, path, text);
    }
}

int main(int argc, char **argv) {
    bool ok = mda_check_import_main((uint32_t)argc, argv);
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
