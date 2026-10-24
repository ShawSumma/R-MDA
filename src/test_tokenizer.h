#ifndef MDA_TEST_TOKENIZER_H
#define MDA_TEST_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define MDA_TEST_TOKENIZER_MERGES 4
#define MDA_TEST_TOKENIZER_EXPECT 13

typedef struct mda_test_tokenizer_merge_t mda_test_tokenizer_merge_t;
typedef struct mda_test_tokenizer_file_t mda_test_tokenizer_file_t;

struct mda_test_tokenizer_merge_t {
    const char *bytes;
    uint32_t left;
    uint32_t right;
};

struct mda_test_tokenizer_file_t {
    char *buf;
    size_t size;
};

#endif
