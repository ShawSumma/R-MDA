#ifndef MDA_TEST_TOKENIZER_H
#define MDA_TEST_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define MDA_TEST_TOKENIZER_PIECES 5

typedef struct mda_test_tokenizer_piece_t mda_test_tokenizer_piece_t;
typedef struct mda_test_tokenizer_file_t mda_test_tokenizer_file_t;

struct mda_test_tokenizer_piece_t {
    const char *bytes;
    uint32_t len;
};

struct mda_test_tokenizer_file_t {
    char *buf;
    size_t size;
};

#endif
