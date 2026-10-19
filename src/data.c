#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "data.h"
#include "hash.h"
#include "util.h"

void *mda_data_job_run(void *arg) {
    mda_data_job_t *j = arg;
    j->ids = malloc((j->hi - j->lo) * sizeof(uint32_t));
    j->n = 0;
    for (size_t pos = j->lo; pos < j->hi; pos += MDA_DATA_TOK_WINDOW) {
        uint32_t chunk = (uint32_t)mda_util_min_size(j->hi - pos, MDA_DATA_TOK_WINDOW);
        mda_tokenizer_ids_t r = mda_tokenizer_encode(j->tok, j->bytes + pos, chunk);
        memcpy(j->ids + j->n, r.ids, r.n * sizeof(uint32_t));
        j->n += r.n;
        free(r.ids);
    }
    return NULL;
}

mda_tokenizer_ids_t mda_data_ids_none(void) {
    return (mda_tokenizer_ids_t) {
        .ids = NULL,
        .n = 0,
    };
}

mda_tokenizer_ids_t mda_data_tokenize_bytes(const mda_tokenizer_t *tok, const char *path, const char *bytes, size_t n) {
    size_t nt = (size_t)sysconf(_SC_NPROCESSORS_ONLN);
    size_t windows = (n + MDA_DATA_TOK_WINDOW - 1) / MDA_DATA_TOK_WINDOW;
    size_t per = (windows + nt - 1) / nt * MDA_DATA_TOK_WINDOW;
    fprintf(stderr, "%s: tokenizing %zu bytes on %zu threads...\n", path, n, nt);
    mda_data_job_t *jobs = malloc(nt * sizeof(mda_data_job_t));
    pthread_t *threads = malloc(nt * sizeof(pthread_t));
    for (size_t i = 0; i < nt; i += 1) {
        size_t lo = mda_util_min_size(i * per, n);
        jobs[i] = (mda_data_job_t) {
            .tok = tok,
            .bytes = bytes,
            .lo = lo,
            .hi = mda_util_min_size(lo + per, n),
        };
        pthread_create(&threads[i], NULL, mda_data_job_run, &jobs[i]);
    }
    size_t total = 0;
    for (size_t i = 0; i < nt; i += 1) {
        pthread_join(threads[i], NULL);
        total += jobs[i].n;
    }
    uint32_t *ids = malloc(total * sizeof(uint32_t));
    size_t at = 0;
    for (size_t i = 0; i < nt; i += 1) {
        memcpy(ids + at, jobs[i].ids, jobs[i].n * sizeof(uint32_t));
        at += jobs[i].n;
        free(jobs[i].ids);
    }
    free(jobs);
    free(threads);
    return (mda_tokenizer_ids_t) {
        .ids = ids,
        .n = total,
    };
}

mda_tokenizer_ids_t mda_data_tokenize_file(const mda_tokenizer_t *tok, const char *path, FILE *f, size_t n) {
    char *bytes = malloc(n);
    size_t got = fread(bytes, 1, n, f);
    fclose(f);
    if (got != n) {
        fprintf(stderr, "%s: short read\n", path);
        free(bytes);
        return mda_data_ids_none();
    } else {
        mda_tokenizer_ids_t ids = mda_data_tokenize_bytes(tok, path, bytes, n);
        free(bytes);
        return ids;
    }
}

mda_tokenizer_ids_t mda_data_tokenize(const mda_tokenizer_t *tok, const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        return mda_data_ids_none();
    } else {
        return mda_data_tokenize_file(tok, path, f, n);
    }
}

mda_data_ids_hdr_t mda_data_ids_hdr(const mda_tokenizer_t *tok, size_t bytes, size_t n) {
    return (mda_data_ids_hdr_t) {
        .magic = MDA_DATA_IDS_MAGIC,
        .vocab_hash = mda_tokenizer_hash(tok),
        .corpus_bytes = bytes,
        .n = n,
    };
}

mda_tokenizer_ids_t mda_data_ids_read_body(FILE *f, size_t n) {
    uint32_t *ids = malloc(n * sizeof(uint32_t));
    size_t got = fread(ids, sizeof(uint32_t), n, f);
    fclose(f);
    if (got != n) {
        free(ids);
        return mda_data_ids_none();
    } else {
        return (mda_tokenizer_ids_t) {
            .ids = ids,
            .n = n,
        };
    }
}

mda_tokenizer_ids_t mda_data_ids_read_file(FILE *f, mda_data_ids_hdr_t want) {
    mda_data_ids_hdr_t h;
    size_t got = fread(&h, sizeof(h), 1, f);
    if (got != 1 || h.magic != want.magic || h.vocab_hash != want.vocab_hash || h.corpus_bytes != want.corpus_bytes) {
        fclose(f);
        return mda_data_ids_none();
    } else {
        return mda_data_ids_read_body(f, h.n);
    }
}

mda_tokenizer_ids_t mda_data_ids_read(const char *path, mda_data_ids_hdr_t want) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return mda_data_ids_none();
    } else {
        return mda_data_ids_read_file(f, want);
    }
}

void mda_data_ids_write_file(FILE *f, const char *tmp, const char *path, mda_data_ids_hdr_t h, const uint32_t *ids) {
    size_t wrote_hdr = fwrite(&h, sizeof(h), 1, f);
    size_t wrote_ids = fwrite(ids, sizeof(uint32_t), h.n, f);
    int32_t closed = fclose(f);
    if (wrote_hdr != 1 || wrote_ids != h.n || closed != 0) {
        perror(tmp);
        remove(tmp);
    } else {
        int32_t moved = rename(tmp, path);
        if (moved != 0) {
            perror(tmp);
            remove(tmp);
        }
    }
}

void mda_data_ids_write(const char *path, mda_data_ids_hdr_t h, const uint32_t *ids) {
    char *tmp = mda_util_path_suffix(path, ".tmp");
    FILE *f = fopen(tmp, "wb");
    if (f == NULL) {
        perror(tmp);
    } else {
        mda_data_ids_write_file(f, tmp, path, h, ids);
    }
    free(tmp);
}

mda_tokenizer_ids_t mda_data_ids_fresh(const mda_tokenizer_t *tok, const char *path, const char *cache, size_t bytes) {
    mda_tokenizer_ids_t fresh = mda_data_tokenize(tok, path, bytes);
    if (fresh.ids != NULL) {
        mda_data_ids_write(cache, mda_data_ids_hdr(tok, bytes, fresh.n), fresh.ids);
    }
    return fresh;
}

mda_tokenizer_ids_t mda_data_ids(const mda_tokenizer_t *tok, const char *path, size_t bytes) {
    char *cache = mda_util_path_suffix(path, ".ids");
    mda_tokenizer_ids_t cached = mda_data_ids_read(cache, mda_data_ids_hdr(tok, bytes, 0));
    if (cached.ids != NULL) {
        free(cache);
        return cached;
    } else {
        mda_tokenizer_ids_t fresh = mda_data_ids_fresh(tok, path, cache, bytes);
        free(cache);
        return fresh;
    }
}

mda_data_corpus_t mda_data_corpus_none(void) {
    return (mda_data_corpus_t) {
        .ids = NULL,
        .len = 0,
        .eval_start = 0,
        .mask_id = 0,
    };
}

mda_data_corpus_t mda_data_corpus(mda_tokenizer_t tok, mda_tokenizer_ids_t ids, size_t bytes) {
    if (ids.ids == NULL) {
        return mda_data_corpus_none();
    } else {
        size_t eval_start = ids.n - (size_t)((double)ids.n * MDA_DATA_EVAL_FRAC);
        fprintf(stderr, "corpus: %zu bytes -> %zu tokens (%.2f bytes/token), vocab %u, %zu held out\n", bytes, ids.n, (double)bytes / ids.n, tok.count, ids.n - eval_start);
        return (mda_data_corpus_t) {
            .ids = ids.ids,
            .len = ids.n,
            .eval_start = eval_start,
            .tok = tok,
            .mask_id = tok.count,
        };
    }
}

mda_data_corpus_t mda_data_load_file(mda_tokenizer_t tok, const char *path, FILE *f) {
    fseek(f, 0, SEEK_END);
    size_t bytes = (size_t)ftell(f);
    fclose(f);
    return mda_data_corpus(tok, mda_data_ids(&tok, path, bytes), bytes);
}

mda_data_corpus_t mda_data_load_text(mda_tokenizer_t tok, const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        return mda_data_corpus_none();
    } else {
        return mda_data_load_file(tok, path, f);
    }
}

mda_data_corpus_t mda_data_load(const char *path) {
    mda_tokenizer_t tok = mda_tokenizer_load(MDA_TOKENIZER_VOCAB_PATH);
    if (tok.entries == NULL) {
        return mda_data_corpus_none();
    } else {
        return mda_data_load_text(tok, path);
    }
}

size_t mda_data_split_lo(const mda_data_corpus_t *cp, mda_data_split_t split) {
    switch (split) {
    case MDA_DATA_SPLIT_EVAL:
        return cp->eval_start;
    default:
        return 0;
    }
}

size_t mda_data_split_hi(const mda_data_corpus_t *cp, mda_data_split_t split) {
    switch (split) {
    case MDA_DATA_SPLIT_EVAL:
        return cp->len;
    default:
        return cp->eval_start;
    }
}

void mda_data_batch(mda_data_batch_t *b, const mda_data_corpus_t *cp, mda_data_split_t split, uint64_t key) {
    size_t lo = mda_data_split_lo(cp, split);
    size_t hi = mda_data_split_hi(cp, split);
    double u0 = mda_hash_unit(MDA_HASH_BATCH, key, 0);
    for (uint32_t s = 0; s < b->batch; s += 1) {
        size_t start = lo + (size_t)(mda_hash_u64(MDA_HASH_BATCH, key, 1 + s) % (hi - lo - b->seqlen + 1));
        double u = fmod(u0 + (double)s / b->batch, 1.0);
        float t = MDA_DATA_T_MIN + (1.0f - MDA_DATA_T_MIN) * (float)u;
        float w = 1.0f / t;
        for (uint32_t p = 0; p < b->seqlen; p += 1) {
            uint32_t pos = s * b->seqlen + p;
            uint32_t token = cp->ids[start + p];
            b->target[pos] = token;
            b->weight[pos] = w;
            if (mda_hash_unit(MDA_HASH_BATCH, key, 1 + b->batch + pos) < t) {
                b->ids[pos] = cp->mask_id;
                b->mask[pos] = 1.0f;
            } else {
                b->ids[pos] = token;
                b->mask[pos] = 0.0f;
            }
        }
    }
}

double mda_data_masked_bytes(const mda_data_corpus_t *cp, const mda_data_batch_t *b) {
    double bytes = 0.0;
    for (uint32_t i = 0; i < b->batch * b->seqlen; i += 1) {
        bytes += b->mask[i] * (double)cp->tok.entries[b->target[i]].len;
    }
    return bytes;
}
