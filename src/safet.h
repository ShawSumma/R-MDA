#ifndef MDA_SAFET_H
#define MDA_SAFET_H

#include <stdbool.h>
#include <stdint.h>

#define MDA_SAFET_MAX_KEY 256
#define MDA_SAFET_MAX_PATH 1024

typedef uint32_t mda_safet_dtype_t;
typedef struct mda_safet_file_t mda_safet_file_t;
typedef struct mda_safet_tensor_t mda_safet_tensor_t;
typedef struct mda_safet_set_t mda_safet_set_t;

enum {
    MDA_SAFET_BF16,
    MDA_SAFET_I64,
};

struct mda_safet_file_t {
    uint8_t *bytes;
    char *header;
    const uint8_t *data;
    uint64_t size;
    uint64_t mapped;
};

struct mda_safet_tensor_t {
    const uint8_t *data;
    uint32_t rows;
    uint32_t cols;
    mda_safet_dtype_t dtype;
    bool ok;
};

struct mda_safet_set_t {
    mda_safet_file_t *files;
    uint32_t n;
};

mda_safet_file_t mda_safet_read(const char *path);
void mda_safet_free(mda_safet_file_t *f);
bool mda_safet_has(const mda_safet_file_t *f, const char *name);
mda_safet_tensor_t mda_safet_tensor(const mda_safet_file_t *f, const char *name);
float mda_safet_bf16(const uint8_t *data, uint64_t i);
int64_t mda_safet_i64(mda_safet_tensor_t t, uint64_t i);
bool mda_safet_plain(float *dst, uint32_t n, mda_safet_tensor_t t);
bool mda_safet_linear(float *dst, uint32_t in, uint32_t out, mda_safet_tensor_t t);
mda_safet_set_t mda_safet_open_set(const char *dir);
void mda_safet_free_set(mda_safet_set_t *s);
uint32_t mda_safet_set_file(const mda_safet_set_t *s, const char *name);
bool mda_safet_set_has(const mda_safet_set_t *s, const char *name);
mda_safet_tensor_t mda_safet_find(const mda_safet_set_t *s, const char *name);

#endif
