#ifndef MDA_SAFET_H
#define MDA_SAFET_H

#include <stdbool.h>
#include <stdint.h>

#define MDA_SAFET_MAX_KEY 256

typedef struct mda_safet_file_t mda_safet_file_t;
typedef struct mda_safet_tensor_t mda_safet_tensor_t;

struct mda_safet_file_t {
    uint8_t *bytes;
    char *header;
    const uint8_t *data;
    uint64_t size;
};

struct mda_safet_tensor_t {
    const uint8_t *data;
    uint32_t rows;
    uint32_t cols;
    bool ok;
};

mda_safet_file_t mda_safet_read(const char *path);
void mda_safet_free(mda_safet_file_t *f);
bool mda_safet_has(const mda_safet_file_t *f, const char *name);
mda_safet_tensor_t mda_safet_tensor(const mda_safet_file_t *f, const char *name);
float mda_safet_bf16(const uint8_t *data, uint64_t i);
bool mda_safet_plain(float *dst, uint32_t n, mda_safet_tensor_t t);
bool mda_safet_linear(float *dst, uint32_t in, uint32_t out, mda_safet_tensor_t t);

#endif
