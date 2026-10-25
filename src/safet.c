#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "safet.h"

mda_safet_file_t mda_safet_none(void) {
    return (mda_safet_file_t) {
        .bytes = NULL,
        .header = NULL,
        .data = NULL,
        .size = 0,
        .mapped = 0,
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
        munmap(bytes, size);
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
            .mapped = size,
        };
    }
}

mda_safet_file_t mda_safet_map(int32_t fd, const char *path) {
    struct stat st;
    int32_t got = fstat(fd, &st);
    void *p = MAP_FAILED;
    if (got == 0 && st.st_size >= 8) {
        p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    }
    close(fd);
    if (p == MAP_FAILED) {
        fprintf(stderr, "%s: cannot map\n", path);
        return mda_safet_none();
    } else {
        return mda_safet_parse(p, (uint64_t)st.st_size, path);
    }
}

mda_safet_file_t mda_safet_read(const char *path) {
    int32_t fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        return mda_safet_none();
    } else {
        return mda_safet_map(fd, path);
    }
}

void mda_safet_free(mda_safet_file_t *f) {
    if (f->bytes != NULL) {
        munmap(f->bytes, f->mapped);
    }
    free(f->header);
}

mda_safet_tensor_t mda_safet_bad(const char *name, const char *why) {
    fprintf(stderr, "safetensors: %s: %s\n", name, why);
    return (mda_safet_tensor_t) {
        .data = NULL,
        .rows = 0,
        .cols = 0,
        .dtype = MDA_SAFET_BF16,
        .ok = false,
    };
}

uint64_t mda_safet_elem(mda_safet_dtype_t dtype) {
    switch (dtype) {
    case MDA_SAFET_I64:
        return 8;
    default:
        return 2;
    }
}

mda_safet_tensor_t mda_safet_shaped(const mda_safet_file_t *f, const char *name, const char *shape, const char *offs, mda_safet_dtype_t dtype) {
    char *p = NULL;
    uint32_t rows = 1;
    uint32_t cols = (uint32_t)strtoul(shape + 9, &p, 10);
    while (*p == ',') {
        rows *= cols;
        cols = (uint32_t)strtoul(p + 1, &p, 10);
    }
    uint64_t a = strtoull(offs + 16, &p, 10);
    uint64_t b = strtoull(p + 1, NULL, 10);
    if (b > f->size || b < a || b - a != (uint64_t)rows * cols * mda_safet_elem(dtype)) {
        return mda_safet_bad(name, "bad data offsets");
    } else {
        return (mda_safet_tensor_t) {
            .data = f->data + a,
            .rows = rows,
            .cols = cols,
            .dtype = dtype,
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
    } else if (strncmp(dtype + 9, "BF16\"", 5) == 0) {
        return mda_safet_shaped(f, name, shape, offs, MDA_SAFET_BF16);
    } else if (strncmp(dtype + 9, "I64\"", 4) == 0) {
        return mda_safet_shaped(f, name, shape, offs, MDA_SAFET_I64);
    } else {
        return mda_safet_bad(name, "not bf16 or i64");
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

int64_t mda_safet_i64(mda_safet_tensor_t t, uint64_t i) {
    int64_t x;
    memcpy(&x, t.data + 8 * i, 8);
    return x;
}

bool mda_safet_plain(float *dst, uint32_t n, mda_safet_tensor_t t) {
    bool ok = t.ok && t.dtype == MDA_SAFET_BF16 && (uint64_t)t.rows * t.cols == n;
    for (uint32_t j = 0; j < n && ok; j += 1) {
        dst[j] = mda_safet_bf16(t.data, j);
    }
    return ok;
}

bool mda_safet_linear(float *dst, uint32_t in, uint32_t out, mda_safet_tensor_t t) {
    bool ok = t.ok && t.dtype == MDA_SAFET_BF16 && t.cols == in && t.rows > 0 && out % t.rows == 0 && t.rows % 64 == 0;
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

char *mda_safet_slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    char *text = NULL;
    if (f == NULL) {
        perror(path);
    } else {
        fseek(f, 0, SEEK_END);
        size_t n = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        text = calloc(n + 1, 1);
        size_t got = fread(text, 1, n, f);
        fclose(f);
        text[got] = '\0';
    }
    return text;
}

uint32_t mda_safet_shard_count(const char *dir) {
    char path[MDA_SAFET_MAX_PATH];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    char *index = mda_safet_slurp(path);
    uint32_t n = 0;
    const char *of = NULL;
    if (index != NULL) {
        of = strstr(index, "-of-");
    }
    if (of != NULL) {
        n = (uint32_t)strtoul(of + 4, NULL, 10);
    }
    free(index);
    return n;
}

void mda_safet_free_set(mda_safet_set_t *s) {
    for (uint32_t i = 0; i < s->n; i += 1) {
        mda_safet_free(&s->files[i]);
    }
    free(s->files);
    s->files = NULL;
    s->n = 0;
}

mda_safet_set_t mda_safet_open_set(const char *dir) {
    uint32_t n = mda_safet_shard_count(dir);
    mda_safet_set_t s = {
        .files = calloc(n + 1, sizeof(mda_safet_file_t)),
        .n = n,
    };
    uint32_t opened = 0;
    for (uint32_t i = 0; i < n; i += 1) {
        char path[MDA_SAFET_MAX_PATH];
        snprintf(path, sizeof(path), "%s/model-%05u-of-%05u.safetensors", dir, i + 1, n);
        s.files[i] = mda_safet_read(path);
        opened += s.files[i].bytes != NULL;
    }
    if (n == 0 || opened < n) {
        fprintf(stderr, "%s: no complete set of safetensors shards\n", dir);
        mda_safet_free_set(&s);
    }
    return s;
}

uint32_t mda_safet_set_file(const mda_safet_set_t *s, const char *name) {
    uint32_t i = 0;
    while (i < s->n && mda_safet_has(&s->files[i], name) == false) {
        i += 1;
    }
    return i;
}

bool mda_safet_set_has(const mda_safet_set_t *s, const char *name) {
    return mda_safet_set_file(s, name) < s->n;
}

mda_safet_tensor_t mda_safet_find(const mda_safet_set_t *s, const char *name) {
    uint32_t i = mda_safet_set_file(s, name);
    if (i == s->n) {
        return mda_safet_bad(name, "missing");
    } else {
        return mda_safet_tensor(&s->files[i], name);
    }
}
