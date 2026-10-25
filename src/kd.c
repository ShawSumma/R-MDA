#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kd.h"

uint32_t *mda_kd_vocab(const mda_tokenizer_t *tok, uint32_t vocab, uint32_t first, uint32_t end) {
    uint32_t *t = calloc((size_t)vocab * 5, sizeof(uint32_t));
    for (uint32_t i = first; i < end && i < tok->count && i < vocab; i += 1) {
        const mda_tokenizer_entry_t *e = &tok->entries[i];
        uint32_t n = e->len;
        if (n > 16) {
            n = 16;
        }
        memcpy(&t[(size_t)i * 5], e->bytes, n);
        t[(size_t)i * 5 + 4] = e->len;
    }
    return t;
}

float mda_kd_prob(uint32_t entry) {
    uint32_t h = entry >> 16;
    uint32_t e = (h >> 10) & 0x1f;
    float m = (float)(h & 0x3ff);
    if (e == 0) {
        return ldexpf(m, -24);
    } else {
        return ldexpf(m + 1024.0f, (int32_t)e - 25);
    }
}

uint32_t mda_kd_window_cap(uint32_t seqlen) {
    return (seqlen + 1) * MDA_TOKENIZER_MAX_LEN;
}

mda_kd_window_t mda_kd_window_alloc(uint32_t seqlen) {
    return (mda_kd_window_t) {
        .bytes = calloc(mda_kd_window_cap(seqlen) + 16, 1),
        .ends = calloc(seqlen + 1, sizeof(uint32_t)),
        .n_bytes = 0,
    };
}

void mda_kd_window_fill(mda_kd_window_t *w, const mda_tokenizer_t *tok, const uint32_t *row_ids, uint32_t seqlen, uint32_t last) {
    uint32_t at = 0;
    for (uint32_t p = 0; p <= seqlen; p += 1) {
        uint32_t id = last;
        if (p < seqlen) {
            id = row_ids[p];
        }
        const mda_tokenizer_entry_t *e = &tok->entries[id];
        memcpy(w->bytes + at, e->bytes, e->len);
        at += e->len;
        w->ends[p] = at;
    }
    memset(w->bytes + at, 0, 16);
    w->n_bytes = at;
}

uint64_t mda_kd_window_index(const mda_kd_hdr_t *h, uint32_t step, uint32_t micro, uint32_t row) {
    return ((uint64_t)(step - h->from) * h->accum + micro) * h->nclean + row;
}

void mda_kd_cont(uint32_t *dst, const uint8_t *bytes, uint32_t n_bytes, uint32_t at, uint32_t nd) {
    uint8_t buf[16];
    memset(buf, 0, sizeof(buf));
    for (uint32_t i = 0; i < 16 && at + i < n_bytes; i += 1) {
        buf[i] = bytes[at + i];
    }
    memcpy(dst, buf, 16);
    dst[4] = nd;
}

mda_kd_file_t mda_kd_none(void) {
    mda_kd_file_t f;
    memset(&f, 0, sizeof(f));
    return f;
}

mda_kd_file_t mda_kd_check(mda_kd_file_t f, const char *path) {
    memcpy(&f.hdr, f.map, sizeof(mda_kd_hdr_t));
    uint64_t need = sizeof(mda_kd_hdr_t) + f.hdr.n_windows * sizeof(uint64_t);
    if (f.hdr.magic != MDA_KD_MAGIC || f.hdr.depth != MDA_KD_DEPTH || f.hdr.k != MDA_KD_K || need > f.size) {
        fprintf(stderr, "%s: not a KD target file of this format\n", path);
        munmap(f.map, f.size);
        return mda_kd_none();
    } else {
        f.index = (const uint64_t *)(f.map + sizeof(mda_kd_hdr_t));
        return f;
    }
}

mda_kd_file_t mda_kd_map(int32_t fd, const char *path) {
    struct stat st;
    int32_t got = fstat(fd, &st);
    void *p = MAP_FAILED;
    if (got == 0 && (uint64_t)st.st_size >= sizeof(mda_kd_hdr_t)) {
        p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    }
    close(fd);
    if (p == MAP_FAILED) {
        fprintf(stderr, "%s: cannot map\n", path);
        return mda_kd_none();
    } else {
        mda_kd_file_t f = mda_kd_none();
        f.map = p;
        f.size = (uint64_t)st.st_size;
        return mda_kd_check(f, path);
    }
}

mda_kd_file_t mda_kd_open(const char *path) {
    int32_t fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        return mda_kd_none();
    } else {
        return mda_kd_map(fd, path);
    }
}

mda_kd_rec_t mda_kd_record(const mda_kd_file_t *f, uint64_t w) {
    mda_kd_rec_t r = {
        .entries = NULL,
        .n_bytes = 0,
    };
    uint64_t at = 0;
    if (w < f->hdr.n_windows) {
        at = f->index[w];
    }
    if (at > 0 && at + sizeof(uint32_t) <= f->size) {
        memcpy(&r.n_bytes, f->map + at, sizeof(uint32_t));
        r.entries = (const uint32_t *)(f->map + at + sizeof(uint32_t));
    }
    if (r.entries != NULL && at + sizeof(uint32_t) + (uint64_t)r.n_bytes * MDA_KD_K * sizeof(uint32_t) > f->size) {
        r.entries = NULL;
        r.n_bytes = 0;
    }
    return r;
}
