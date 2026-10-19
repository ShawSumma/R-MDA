#ifndef MDA_TOKENIZER_H
#define MDA_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define MDA_TOKENIZER_CORPUS_PATH "data/corpus.txt"
#define MDA_TOKENIZER_MAX_LEN 32
#define MDA_TOKENIZER_VOCAB_PATH "models/vocab.txt"

typedef struct mda_tokenizer_entry_t mda_tokenizer_entry_t;
typedef struct mda_tokenizer_node_t mda_tokenizer_node_t;
typedef struct mda_tokenizer_t mda_tokenizer_t;
typedef struct mda_tokenizer_ids_t mda_tokenizer_ids_t;

struct mda_tokenizer_entry_t {
    char *bytes;
    uint32_t len;
    float logp;
};

struct mda_tokenizer_node_t {
    uint32_t child[256];
    int32_t token;
};

struct mda_tokenizer_t {
    mda_tokenizer_entry_t *entries;
    mda_tokenizer_node_t *nodes;
    uint32_t count;
    uint32_t node_count;
};

struct mda_tokenizer_ids_t {
    uint32_t *ids;
    size_t n;
};

void mda_tokenizer_write_entry(FILE *f, float logp, const char *bytes, uint32_t len);
mda_tokenizer_t mda_tokenizer_read(FILE *f, const char *path);
mda_tokenizer_t mda_tokenizer_load(const char *path);
uint64_t mda_tokenizer_hash(const mda_tokenizer_t *t);
mda_tokenizer_ids_t mda_tokenizer_encode(const mda_tokenizer_t *t, const char *text, uint32_t len);

#endif
