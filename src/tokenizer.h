#ifndef MDA_TOKENIZER_H
#define MDA_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define MDA_TOKENIZER_CORPUS_PATH "data/corpus.txt"
#define MDA_TOKENIZER_MAX_LEN 1024
#define MDA_TOKENIZER_VOCAB_PATH "models/world.txt"
#define MDA_TOKENIZER_PAD 64

typedef uint32_t mda_tokenizer_class_t;
typedef struct mda_tokenizer_entry_t mda_tokenizer_entry_t;
typedef struct mda_tokenizer_merge_t mda_tokenizer_merge_t;
typedef struct mda_tokenizer_t mda_tokenizer_t;
typedef struct mda_tokenizer_ids_t mda_tokenizer_ids_t;
typedef struct mda_tokenizer_char_t mda_tokenizer_char_t;
typedef struct mda_tokenizer_match_t mda_tokenizer_match_t;

enum {
    MDA_TOKENIZER_SPACE,
    MDA_TOKENIZER_LETTER,
    MDA_TOKENIZER_NUMBER,
    MDA_TOKENIZER_MARK,
    MDA_TOKENIZER_OTHER,
};

struct mda_tokenizer_entry_t {
    char *bytes;
    uint32_t len;
};

struct mda_tokenizer_merge_t {
    uint64_t key;
    uint32_t rank;
    uint32_t id;
};

struct mda_tokenizer_t {
    mda_tokenizer_entry_t *entries;
    mda_tokenizer_merge_t *merges;
    uint64_t *trie_keys;
    uint32_t *trie_next;
    uint32_t *trie_tok;
    uint32_t byte_id[256];
    uint32_t count;
    uint32_t n_merges;
    uint32_t cap;
    uint32_t trie_cap;
    uint32_t trie_nodes;
};

struct mda_tokenizer_ids_t {
    uint32_t *ids;
    size_t n;
};

struct mda_tokenizer_char_t {
    uint32_t cp;
    uint32_t len;
};

struct mda_tokenizer_match_t {
    uint32_t len;
    uint32_t tok;
};

void mda_tokenizer_write_entry(FILE *f, const char *bytes, uint32_t len);
mda_tokenizer_t mda_tokenizer_read(FILE *f, const char *path);
mda_tokenizer_t mda_tokenizer_load(const char *path);
uint64_t mda_tokenizer_hash(const mda_tokenizer_t *t);
uint32_t mda_tokenizer_padded(uint32_t count);
mda_tokenizer_ids_t mda_tokenizer_encode(const mda_tokenizer_t *t, const char *text, uint32_t len);

#endif
