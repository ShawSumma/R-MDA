#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vocab.h"
#include "hash.h"
#include "util.h"
#include "sort.h"

mda_vocab_text_t mda_vocab_none(void) {
    return (mda_vocab_text_t) {
        .bytes = NULL,
        .len = 0,
    };
}

size_t mda_vocab_pieces(size_t len, size_t n) {
    if (len > n) {
        return n / MDA_VOCAB_SAMPLE_PIECE;
    } else {
        return 1;
    }
}

size_t mda_vocab_piece(size_t len, size_t n) {
    if (len > n) {
        return MDA_VOCAB_SAMPLE_PIECE;
    } else {
        return len;
    }
}

mda_vocab_text_t mda_vocab_read_pieces(FILE *f, const char *path, size_t n) {
    fseek(f, 0, SEEK_END);
    size_t len = (size_t)ftell(f);
    size_t pieces = mda_vocab_pieces(len, n);
    size_t piece = mda_vocab_piece(len, n);
    size_t stride = len / pieces;
    uint8_t *b = malloc(pieces * piece);
    for (size_t i = 0; i < pieces; i += 1) {
        int32_t sought = fseek(f, (int64_t)(i * stride), SEEK_SET);
        size_t got = fread(b + i * piece, 1, piece, f);
        if (sought != 0 || got != piece) {
            fprintf(stderr, "%s: short read\n", path);
            fclose(f);
            free(b);
            return mda_vocab_none();
        }
    }
    fclose(f);
    fprintf(stderr, "sample: %zu bytes from %zu pieces of %s (%zu bytes)\n", pieces * piece, pieces, path, len);
    return (mda_vocab_text_t) {
        .bytes = b,
        .len = pieces * piece,
    };
}

mda_vocab_text_t mda_vocab_sample(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        return mda_vocab_none();
    } else {
        return mda_vocab_read_pieces(f, path, n);
    }
}

uint32_t mda_vocab_byte_class(uint8_t b) {
    if (b == ' ' || b == '\n' || b == '\t' || b == '\r') {
        return 0;
    } else if (b >= '0' && b <= '9') {
        return 1;
    } else if (((b | 32) >= 'a' && (b | 32) <= 'z') || b >= 0x80) {
        return 2;
    } else {
        return 3;
    }
}

uint32_t mda_vocab_lead(const uint8_t *t, uint32_t max) {
    if (max > 1 && t[0] == ' ' && mda_vocab_byte_class(t[1]) != 0) {
        return 1;
    } else {
        return 0;
    }
}

uint32_t mda_vocab_shape_len(const uint8_t *t, uint32_t max) {
    uint32_t s = mda_vocab_lead(t, max);
    uint32_t c = mda_vocab_byte_class(t[s]);
    uint32_t l = s + 1;
    while (l < max && mda_vocab_byte_class(t[l]) == c) {
        l += 1;
    }
    return l;
}

mda_vocab_map_t mda_vocab_map_make(uint32_t cap) {
    return (mda_vocab_map_t) {
        .key = malloc((size_t)cap * sizeof(mda_vocab_key_t)),
        .val = malloc((size_t)cap * sizeof(uint32_t)),
        .len = calloc(cap, 1),
        .cap = cap,
        .used = 0,
    };
}

void mda_vocab_map_free(mda_vocab_map_t *t) {
    free(t->key);
    free(t->val);
    free(t->len);
}

mda_vocab_key_t mda_vocab_key(const char *s, uint32_t len) {
    mda_vocab_key_t k = {
        .lo = 0,
        .hi = 0,
    };
    memcpy(&k, s, len);
    return k;
}

uint32_t mda_vocab_map_slot0(const mda_vocab_map_t *t, mda_vocab_key_t k, uint32_t len) {
    uint64_t x = mda_hash_mix(k.lo + (uint64_t)len * 0x9e3779b97f4a7c15ull);
    return (uint32_t)mda_hash_mix(x ^ k.hi) & (t->cap - 1);
}

uint32_t mda_vocab_map_find(const mda_vocab_map_t *t, mda_vocab_key_t k, uint32_t len) {
    uint32_t slot = mda_vocab_map_slot0(t, k, len);
    while (t->len[slot] != 0 && (t->len[slot] != len || t->key[slot].lo != k.lo || t->key[slot].hi != k.hi)) {
        slot = (slot + 1) & (t->cap - 1);
    }
    return slot;
}

void mda_vocab_map_grow(mda_vocab_map_t *t) {
    mda_vocab_map_t g = mda_vocab_map_make(t->cap * 2);
    for (uint32_t i = 0; i < t->cap; i += 1) {
        if (t->len[i] != 0) {
            uint32_t slot = mda_vocab_map_find(&g, t->key[i], t->len[i]);
            g.key[slot] = t->key[i];
            g.val[slot] = t->val[i];
            g.len[slot] = t->len[i];
        }
    }
    g.used = t->used;
    mda_vocab_map_free(t);
    *t = g;
}

uint32_t *mda_vocab_map_put(mda_vocab_map_t *t, const char *s, uint32_t len) {
    if ((t->used + 1) * 2 > t->cap) {
        mda_vocab_map_grow(t);
    }
    mda_vocab_key_t k = mda_vocab_key(s, len);
    uint32_t slot = mda_vocab_map_find(t, k, len);
    if (t->len[slot] == 0) {
        t->key[slot] = k;
        t->val[slot] = 0;
        t->len[slot] = (uint8_t)len;
        t->used += 1;
    }
    return &t->val[slot];
}

int32_t mda_vocab_map_get(const mda_vocab_map_t *t, const char *s, uint32_t len) {
    uint32_t slot = mda_vocab_map_find(t, mda_vocab_key(s, len), len);
    if (t->len[slot] == 0) {
        return -1;
    } else {
        return (int32_t)t->val[slot];
    }
}

mda_vocab_map_t mda_vocab_count(const mda_vocab_text_t *t) {
    mda_vocab_map_t h = mda_vocab_map_make(1u << 22);
    fprintf(stderr, "seeding: counting substrings (len 2..%u) over %zu bytes...\n", MDA_VOCAB_SEED_MAX_LEN, t->len);
    for (size_t i = 0; i < t->len; i += 1) {
        uint32_t top = mda_vocab_shape_len(t->bytes + i, (uint32_t)mda_util_min_size(t->len - i, MDA_VOCAB_SEED_MAX_LEN));
        for (uint32_t l = 2; l <= top; l += 1) {
            uint32_t *count = mda_vocab_map_put(&h, (const char *)t->bytes + i, l);
            *count += 1;
        }
    }
    return h;
}

bool mda_vocab_seen_twice(const mda_vocab_map_t *h, uint32_t slot) {
    return h->len[slot] != 0 && h->val[slot] >= 2;
}

uint32_t mda_vocab_pool_size(const mda_vocab_map_t *h) {
    uint32_t n = 0;
    for (uint32_t slot = 0; slot < h->cap; slot += 1) {
        if (mda_vocab_seen_twice(h, slot)) {
            n += 1;
        }
    }
    return n;
}

mda_sort_item_t *mda_vocab_pool(const mda_vocab_map_t *h, uint32_t n) {
    mda_sort_item_t *pool = malloc((size_t)n * sizeof(mda_sort_item_t));
    uint32_t k = 0;
    for (uint32_t slot = 0; slot < h->cap; slot += 1) {
        if (mda_vocab_seen_twice(h, slot)) {
            pool[k] = (mda_sort_item_t) {
                .key = -(double)h->val[slot] * h->len[slot],
                .idx = slot,
            };
            k += 1;
        }
    }
    mda_sort_items(pool, n);
    return pool;
}

mda_vocab_cands_t mda_vocab_seed(const mda_vocab_text_t *t) {
    mda_vocab_map_t h = mda_vocab_count(t);
    uint32_t pool_n = mda_vocab_pool_size(&h);
    fprintf(stderr, "seeding: %u distinct substrings, %u seen twice or more, keeping top %u + 256 bytes\n", h.used, pool_n, MDA_VOCAB_SEED_KEEP);
    mda_sort_item_t *pool = mda_vocab_pool(&h, pool_n);
    uint32_t keep = mda_util_min_u32(pool_n, MDA_VOCAB_SEED_KEEP);
    mda_vocab_cand_t *v = malloc((MDA_VOCAB_SEED_KEEP + 256) * sizeof(mda_vocab_cand_t));
    for (uint32_t b = 0; b < 256; b += 1) {
        v[b] = (mda_vocab_cand_t) {
            .logp = 0.0,
            .expected = 1.0,
            .score = 0.0,
            .len = 1,
        };
        v[b].bytes[0] = (char)b;
    }
    for (uint32_t i = 0; i < keep; i += 1) {
        uint32_t slot = pool[i].idx;
        v[256 + i] = (mda_vocab_cand_t) {
            .logp = 0.0,
            .expected = h.val[slot],
            .score = -pool[i].key,
            .len = h.len[slot],
        };
        memcpy(v[256 + i].bytes, &h.key[slot], h.len[slot]);
    }
    free(pool);
    mda_vocab_map_free(&h);
    uint32_t n = 256 + keep;
    double total = 0.0;
    for (uint32_t i = 0; i < n; i += 1) {
        total += v[i].expected;
    }
    for (uint32_t i = 0; i < n; i += 1) {
        v[i].logp = log(v[i].expected / total);
    }
    return (mda_vocab_cands_t) {
        .v = v,
        .n = n,
    };
}

mda_vocab_map_t mda_vocab_lookup(const mda_vocab_cands_t *c) {
    mda_vocab_map_t lk = mda_vocab_map_make(1u << 16);
    for (uint32_t i = 0; i < c->n; i += 1) {
        *mda_vocab_map_put(&lk, c->v[i].bytes, c->v[i].len) = i;
    }
    return lk;
}

bool mda_vocab_reached(double x) {
    return x > MDA_VOCAB_NEG_INF + 1.0;
}

double mda_vocab_logaddexp(double a, double b) {
    if (mda_vocab_reached(a)) {
        double hi = fmax(a, b);
        return hi + log(exp(a - hi) + exp(b - hi));
    } else {
        return b;
    }
}

int32_t mda_vocab_chunk_id(const mda_vocab_map_t *lk, const uint8_t *text, uint32_t top, uint32_t l) {
    if (l <= top) {
        return mda_vocab_map_get(lk, (const char *)text, l);
    } else {
        return -1;
    }
}

int32_t *mda_vocab_chunk_ids(const mda_vocab_map_t *lk, const uint8_t *text, uint32_t len) {
    int32_t *ids = malloc((size_t)len * MDA_VOCAB_SEED_MAX_LEN * sizeof(int32_t));
    for (uint32_t i = 0; i < len; i += 1) {
        uint32_t top = mda_vocab_shape_len(text + i, mda_util_min_u32(len - i, MDA_VOCAB_SEED_MAX_LEN));
        for (uint32_t l = 1; l <= MDA_VOCAB_SEED_MAX_LEN; l += 1) {
            ids[i * MDA_VOCAB_SEED_MAX_LEN + l - 1] = mda_vocab_chunk_id(lk, text + i, top, l);
        }
    }
    return ids;
}

double *mda_vocab_forward(const mda_vocab_cands_t *c, const int32_t *ids, uint32_t len) {
    double *alpha = malloc((len + 1) * sizeof(double));
    for (uint32_t i = 0; i <= len; i += 1) {
        alpha[i] = MDA_VOCAB_NEG_INF;
    }
    alpha[0] = 0.0;
    for (uint32_t i = 0; i < len; i += 1) {
        for (uint32_t l = 1; l <= MDA_VOCAB_SEED_MAX_LEN && mda_vocab_reached(alpha[i]); l += 1) {
            int32_t id = ids[i * MDA_VOCAB_SEED_MAX_LEN + l - 1];
            if (id >= 0) {
                alpha[i + l] = mda_vocab_logaddexp(alpha[i + l], alpha[i] + c->v[id].logp);
            }
        }
    }
    return alpha;
}

double *mda_vocab_backward(const mda_vocab_cands_t *c, const int32_t *ids, uint32_t len) {
    double *beta = malloc((len + 1) * sizeof(double));
    for (uint32_t i = 0; i <= len; i += 1) {
        beta[i] = MDA_VOCAB_NEG_INF;
    }
    beta[len] = 0.0;
    for (uint32_t ii = len; ii > 0; ii -= 1) {
        uint32_t i = ii - 1;
        for (uint32_t l = 1; l <= MDA_VOCAB_SEED_MAX_LEN; l += 1) {
            int32_t id = ids[i * MDA_VOCAB_SEED_MAX_LEN + l - 1];
            if (id >= 0 && mda_vocab_reached(beta[i + l])) {
                beta[i] = mda_vocab_logaddexp(beta[i], c->v[id].logp + beta[i + l]);
            }
        }
    }
    return beta;
}

void mda_vocab_posterior(mda_vocab_cands_t *c, const int32_t *ids, const double *alpha, const double *beta, uint32_t len) {
    for (uint32_t i = 0; i < len; i += 1) {
        for (uint32_t l = 1; l <= MDA_VOCAB_SEED_MAX_LEN && mda_vocab_reached(alpha[i]); l += 1) {
            int32_t id = ids[i * MDA_VOCAB_SEED_MAX_LEN + l - 1];
            if (id >= 0) {
                c->v[id].expected += exp(alpha[i] + c->v[id].logp + beta[i + l] - alpha[len]);
            }
        }
    }
}

double mda_vocab_em_chunk(mda_vocab_cands_t *c, const mda_vocab_map_t *lk, const uint8_t *text, uint32_t len) {
    int32_t *ids = mda_vocab_chunk_ids(lk, text, len);
    double *alpha = mda_vocab_forward(c, ids, len);
    double total = alpha[len];
    if (mda_vocab_reached(total)) {
        double *beta = mda_vocab_backward(c, ids, len);
        mda_vocab_posterior(c, ids, alpha, beta, len);
        free(beta);
    }
    free(ids);
    free(alpha);
    if (mda_vocab_reached(total)) {
        return total;
    } else {
        return 0.0;
    }
}

void mda_vocab_em_round(mda_vocab_cands_t *c, const mda_vocab_text_t *t, const mda_vocab_map_t *lk) {
    for (uint32_t i = 0; i < c->n; i += 1) {
        c->v[i].expected = 0.0;
    }
    double ll = 0.0;
    for (size_t done = 0; done < t->len; done += MDA_VOCAB_EM_CHUNK) {
        ll += mda_vocab_em_chunk(c, lk, t->bytes + done, (uint32_t)mda_util_min_size(t->len - done, MDA_VOCAB_EM_CHUNK));
    }
    double total = 0.0;
    for (uint32_t i = 0; i < c->n; i += 1) {
        total += c->v[i].expected;
    }
    for (uint32_t i = 0; i < c->n; i += 1) {
        c->v[i].logp = log(fmax(c->v[i].expected, 1e-9) / total);
    }
    fprintf(stderr, "EM round: %.4f bits/byte, %.3f bytes/token, vocab %u\n", -ll / t->len / log(2.0), t->len / total, c->n);
}

void mda_vocab_em(mda_vocab_cands_t *c, const mda_vocab_text_t *t) {
    mda_vocab_map_t lk = mda_vocab_lookup(c);
    for (uint32_t it = 0; it < MDA_VOCAB_EM_ITERS; it += 1) {
        mda_vocab_em_round(c, t, &lk);
    }
    mda_vocab_map_free(&lk);
}

double mda_vocab_split_logp(const mda_vocab_cands_t *c, const mda_vocab_map_t *lk, const mda_vocab_cand_t *cand) {
    double best[MDA_VOCAB_SEED_MAX_LEN + 1];
    best[0] = 0.0;
    for (uint32_t p = 1; p <= cand->len; p += 1) {
        best[p] = MDA_VOCAB_NEG_INF;
    }
    for (uint32_t s = 0; s < cand->len; s += 1) {
        for (uint32_t l = 1; s + l <= cand->len; l += 1) {
            int32_t id = mda_vocab_map_get(lk, cand->bytes + s, l);
            if (id >= 0 && (s != 0 || l != cand->len)) {
                best[s + l] = fmax(best[s + l], best[s] + c->v[id].logp);
            }
        }
    }
    return best[cand->len];
}

void mda_vocab_prune_scores(mda_vocab_cands_t *c, const mda_vocab_map_t *lk) {
    for (uint32_t i = 0; i < c->n; i += 1) {
        mda_vocab_cand_t *cand = &c->v[i];
        if (cand->len > 1) {
            cand->score = cand->expected * (cand->logp - mda_vocab_split_logp(c, lk, cand));
        }
    }
}

double mda_vocab_prune_key(const mda_vocab_cand_t *cand) {
    if (cand->len == 1) {
        return INFINITY;
    } else {
        return cand->score;
    }
}

void mda_vocab_prune(mda_vocab_cands_t *c, uint32_t target) {
    mda_vocab_map_t lk = mda_vocab_lookup(c);
    mda_vocab_prune_scores(c, &lk);
    mda_vocab_map_free(&lk);
    mda_sort_item_t *order = malloc(c->n * sizeof(mda_sort_item_t));
    uint32_t multibyte = 0;
    for (uint32_t i = 0; i < c->n; i += 1) {
        order[i] = (mda_sort_item_t) {
            .key = mda_vocab_prune_key(&c->v[i]),
            .idx = i,
        };
        if (c->v[i].len > 1) {
            multibyte += 1;
        }
    }
    mda_sort_items(order, c->n);
    uint32_t keep = mda_util_min_u32(mda_util_max_u32((uint32_t)((double)c->n * MDA_VOCAB_PRUNE_KEEP), target), c->n);
    uint32_t drop = mda_util_min_u32(c->n - keep, multibyte);
    mda_vocab_cand_t *kept = malloc((c->n - drop) * sizeof(mda_vocab_cand_t));
    for (uint32_t r = drop; r < c->n; r += 1) {
        kept[r - drop] = c->v[order[r].idx];
    }
    free(order);
    free(c->v);
    c->v = kept;
    c->n -= drop;
}

bool mda_vocab_write(const char *path, const mda_vocab_cands_t *c) {
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        perror(path);
        return false;
    } else {
        fprintf(f, "%u\n", c->n);
        for (uint32_t i = 0; i < c->n; i += 1) {
            mda_tokenizer_write_entry(f, (float)c->v[i].logp, c->v[i].bytes, c->v[i].len);
        }
        int32_t closed = fclose(f);
        if (closed != 0) {
            perror(path);
        } else {
            fprintf(stderr, "wrote %u tokens to %s\n", c->n, path);
        }
        return closed == 0;
    }
}
