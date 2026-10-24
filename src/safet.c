#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "safet.h"

mda_safet_file_t mda_safet_none(void) {
    return (mda_safet_file_t) {
        .bytes = NULL,
        .header = NULL,
        .data = NULL,
        .size = 0,
    };
}

uint64_t mda_safet_header_len(const uint8_t *bytes, uint64_t size) {
    uint64_t n = UINT64_MAX;
    if (size >= 8) {
        memcpy(&n, bytes, 8);
    }
    return n;
}

mda_safet_file_t mda_safet_parse(uint8_t *bytes, uint64_t size, const char *path) {
    uint64_t n = mda_safet_header_len(bytes, size);
    if (n > size - 8) {
        fprintf(stderr, "%s: not a safetensors file\n", path);
        free(bytes);
        return mda_safet_none();
    } else {
        char *header = malloc(n + 1);
        memcpy(header, bytes + 8, n);
        header[n] = '\0';
        return (mda_safet_file_t) {
            .bytes = bytes,
            .header = header,
            .data = bytes + 8 + n,
            .size = size - 8 - n,
        };
    }
}

mda_safet_file_t mda_safet_read_file(FILE *f, const char *path) {
    fseek(f, 0, SEEK_END);
    uint64_t size = (uint64_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *bytes = malloc(size);
    size_t got = fread(bytes, 1, size, f);
    fclose(f);
    if (got != size) {
        fprintf(stderr, "%s: short read\n", path);
        free(bytes);
        return mda_safet_none();
    } else {
        return mda_safet_parse(bytes, size, path);
    }
}

mda_safet_file_t mda_safet_read(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        return mda_safet_none();
    } else {
        return mda_safet_read_file(f, path);
    }
}

void mda_safet_free(mda_safet_file_t *f) {
    free(f->bytes);
    free(f->header);
}

mda_safet_tensor_t mda_safet_bad(const char *name, const char *why) {
    fprintf(stderr, "safetensors: %s: %s\n", name, why);
    return (mda_safet_tensor_t) {
        .data = NULL,
        .rows = 0,
        .cols = 0,
        .ok = false,
    };
}

mda_safet_tensor_t mda_safet_shaped(const mda_safet_file_t *f, const char *name, const char *shape, const char *offs) {
    char *p = NULL;
    uint32_t d0 = (uint32_t)strtoul(shape + 9, &p, 10);
    uint32_t d1 = 0;
    if (*p == ',') {
        d1 = (uint32_t)strtoul(p + 1, NULL, 10);
    }
    uint64_t a = strtoull(offs + 16, &p, 10);
    uint64_t b = strtoull(p + 1, NULL, 10);
    uint32_t rows = 1;
    uint32_t cols = d0;
    if (d1 > 0) {
        rows = d0;
        cols = d1;
    }
    if (b > f->size || b < a || b - a != (uint64_t)rows * cols * 2) {
        return mda_safet_bad(name, "bad data offsets");
    } else {
        return (mda_safet_tensor_t) {
            .data = f->data + a,
            .rows = rows,
            .cols = cols,
            .ok = true,
        };
    }
}

mda_safet_tensor_t mda_safet_fields(const mda_safet_file_t *f, const char *name, const char *obj) {
    const char *end = strchr(obj, '}');
    const char *dtype = strstr(obj, "\"dtype\":\"");
    const char *shape = strstr(obj, "\"shape\":[");
    const char *offs = strstr(obj, "\"data_offsets\":[");
    if (end == NULL || dtype == NULL || shape == NULL || offs == NULL || dtype > end || shape > end || offs > end) {
        return mda_safet_bad(name, "malformed header entry");
    } else if (strncmp(dtype + 9, "BF16\"", 5) != 0) {
        return mda_safet_bad(name, "not bf16");
    } else {
        return mda_safet_shaped(f, name, shape, offs);
    }
}

bool mda_safet_has(const mda_safet_file_t *f, const char *name) {
    char key[MDA_SAFET_MAX_KEY];
    snprintf(key, sizeof(key), "\"%s\":{", name);
    return strstr(f->header, key) != NULL;
}

mda_safet_tensor_t mda_safet_tensor(const mda_safet_file_t *f, const char *name) {
    char key[MDA_SAFET_MAX_KEY];
    snprintf(key, sizeof(key), "\"%s\":{", name);
    const char *obj = strstr(f->header, key);
    if (obj == NULL) {
        return mda_safet_bad(name, "missing");
    } else {
        return mda_safet_fields(f, name, obj + strlen(key));
    }
}

float mda_safet_bf16(const uint8_t *data, uint64_t i) {
    uint32_t u = ((uint32_t)data[2 * i] | (uint32_t)data[2 * i + 1] << 8) << 16;
    float x;
    memcpy(&x, &u, 4);
    return x;
}

bool mda_safet_plain(float *dst, uint32_t n, mda_safet_tensor_t t) {
    bool ok = t.ok && (uint64_t)t.rows * t.cols == n;
    for (uint32_t j = 0; j < n && ok; j += 1) {
        dst[j] = mda_safet_bf16(t.data, j);
    }
    return ok;
}

bool mda_safet_linear(float *dst, uint32_t in, uint32_t out, mda_safet_tensor_t t) {
    bool ok = t.ok && t.cols == in && t.rows > 0 && out % t.rows == 0 && t.rows % 64 == 0;
    uint32_t group = 1;
    if (ok) {
        group = out / t.rows;
    }
    for (uint32_t o = 0; o < out && ok; o += 1) {
        uint64_t src = (uint64_t)((o / 64 / group) * 64 + o % 64) * t.cols;
        for (uint32_t i = 0; i < in; i += 1) {
            dst[(uint64_t)i * out + o] = mda_safet_bf16(t.data, src + i);
        }
    }
    return ok;
}
