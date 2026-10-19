#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "tokenizer.h"
#include "vocab.h"

bool mda_train_tokenizer_cands(mda_vocab_cands_t *cands) {
    mda_vocab_text_t em = mda_vocab_sample(MDA_TOKENIZER_CORPUS_PATH, MDA_VOCAB_EM_SAMPLE_BYTES);
    if (em.bytes == NULL) {
        return false;
    } else {
        mda_vocab_em(cands, &em);
        while (cands->n > MDA_VOCAB_TARGET) {
            mda_vocab_prune(cands, MDA_VOCAB_TARGET);
            fprintf(stderr, "pruned to %u\n", cands->n);
            mda_vocab_em(cands, &em);
        }
        free(em.bytes);
        return mda_vocab_write(MDA_TOKENIZER_VOCAB_PATH, cands);
    }
}

bool mda_train_tokenizer_main(void) {
    mda_vocab_text_t seed = mda_vocab_sample(MDA_TOKENIZER_CORPUS_PATH, MDA_VOCAB_SEED_SAMPLE_BYTES);
    if (seed.bytes == NULL) {
        return false;
    } else {
        mda_vocab_cands_t cands = mda_vocab_seed(&seed);
        free(seed.bytes);
        fprintf(stderr, "seeded %u candidates\n", cands.n);
        return mda_train_tokenizer_cands(&cands);
    }
}

int main(void) {
    bool ok = mda_train_tokenizer_main();
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
