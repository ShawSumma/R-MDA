// train_tokenizer.c -- train a Unigram LM subword vocab from the corpus, the
// SentencePiece-style algorithm (EM over a unigram model + likelihood pruning).
// No BPE merges, no Python. Writes models/vocab.txt in the format tokenizer.h reads.
//
// Pipeline:
//   1. seed a large candidate set: frequent substrings (counted by hashing every
//      substring up to TOK_MAX_LEN) plus every single byte, so any text is always
//      encodable. Keep the top candidates by count*length.
//   2. EM: the E-step computes expected token counts by forward-backward over
//      sampled corpus chunks; the M-step sets each prob proportional to its count.
//   3. prune: drop the candidates that contribute least to corpus likelihood, never
//      dropping single bytes, then run more EM. Repeat until the target size.
#include "tokenizer.h"
#include <math.h>
#include <time.h>

#define TARGET_VOCAB 16000
#define SEED_KEEP 120000
#define SEED_SCAN_BYTES (4 * 1024 * 1024)
#define SEED_MAX_LEN 8
#define EM_SAMPLE_BYTES (2 * 1024 * 1024)
#define EM_ITERS_PER_ROUND 3
#define PRUNE_KEEP 0.75
#define NEG_INF (-1e30)

typedef struct cand cand_t;

struct cand {
    char bytes[TOK_MAX_LEN];
    uint32_t len;
    double logp;
    double expected;
    double score;
};

typedef struct corpus corpus_t;

struct corpus {
    unsigned char *bytes;
    size_t len;
};

static corpus_t corpus_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(n);
    fread(b, 1, n, f);
    fclose(f);
    corpus_t cp = {
        .bytes = b,
        .len = (size_t)n,
    };
    return cp;
}

static double rand_unit(void) {
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

// ---- substring hash map (counting during seeding) --------------------------

typedef struct subhash subhash_t;

struct subhash {
    char *keys;
    uint8_t *klen;
    double *count;
    uint8_t *used;
    uint32_t cap;
    uint32_t mask;
};

static subhash_t subhash_make(uint32_t cap) {
    subhash_t h;
    h.cap = cap;
    h.mask = cap - 1;
    h.keys = calloc((size_t)cap * TOK_MAX_LEN, 1);
    h.klen = calloc(cap, 1);
    h.count = calloc(cap, sizeof(double));
    h.used = calloc(cap, 1);
    return h;
}

static void subhash_add(subhash_t *h, const char *s, uint32_t len) {
    uint32_t x = 2166136261u;
    for (uint32_t j = 0; j < len; j = j + 1) {
        x = (x ^ (unsigned char)s[j]) * 16777619u;
    }
    uint32_t slot = x & h->mask;
    while (h->used[slot]) {
        if (h->klen[slot] == len && memcmp(h->keys + (size_t)slot * TOK_MAX_LEN, s, len) == 0) {
            h->count[slot] = h->count[slot] + 1.0;
            return;
        }
        slot = (slot + 1) & h->mask;
    }
    h->used[slot] = 1;
    h->klen[slot] = (uint8_t)len;
    memcpy(h->keys + (size_t)slot * TOK_MAX_LEN, s, len);
    h->count[slot] = 1.0;
}

// ---- candidate seeding -----------------------------------------------------

static int cand_by_score_desc(const void *pa, const void *pb) {
    const cand_t *a = pa;
    const cand_t *b = pb;
    if (a->score > b->score) {
        return -1;
    }
    if (a->score < b->score) {
        return 1;
    }
    return 0;
}

// hash every substring of length 2..TOK_MAX_LEN over a sampled portion of the
// corpus, then keep the SEED_KEEP highest count*len candidates, and add all 256
// single bytes. Sampling keeps the O(n*L) pass affordable on a large corpus.
static uint32_t seed_candidates(corpus_t *cp, cand_t *out) {
    subhash_t h = subhash_make(1u << 23);
    size_t scan = cp->len;
    if (scan > SEED_SCAN_BYTES) {
        scan = SEED_SCAN_BYTES;
    }
    fprintf(stderr, "seeding: hashing substrings (len 2..%u) over %zu bytes...\n", SEED_MAX_LEN, scan);
    for (size_t i = 0; i < scan; i = i + 1) {
        uint32_t cap_len = SEED_MAX_LEN;
        if (i + cap_len > scan) {
            cap_len = (uint32_t)(scan - i);
        }
        for (uint32_t l = 2; l <= cap_len; l = l + 1) {
            subhash_add(&h, (const char *)cp->bytes + i, l);
        }
    }

    uint32_t pool_cap = 1u << 21;
    cand_t *pool = malloc(pool_cap * sizeof(cand_t));
    uint32_t pool_n = 0;
    for (uint32_t slot = 0; slot < h.cap; slot = slot + 1) {
        if (h.used[slot] == 0 || h.count[slot] < 2.0) {
            continue;
        }
        if (pool_n >= pool_cap) {
            break;
        }
        cand_t c;
        c.len = h.klen[slot];
        memcpy(c.bytes, h.keys + (size_t)slot * TOK_MAX_LEN, c.len);
        c.expected = h.count[slot];
        c.score = h.count[slot] * (double)c.len;
        c.logp = 0.0;
        pool[pool_n] = c;
        pool_n = pool_n + 1;
    }
    free(h.keys);
    free(h.klen);
    free(h.count);
    free(h.used);

    fprintf(stderr, "seeding: %u raw candidates, keeping top %u + 256 bytes\n", pool_n, SEED_KEEP);
    qsort(pool, pool_n, sizeof(cand_t), cand_by_score_desc);

    uint32_t keep = (pool_n < SEED_KEEP) ? pool_n : SEED_KEEP;
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b = b + 1) {
        out[n].bytes[0] = (char)b;
        out[n].len = 1;
        out[n].expected = 1.0;
        out[n].logp = 0.0;
        out[n].score = 0.0;
        n = n + 1;
    }
    for (uint32_t i = 0; i < keep; i = i + 1) {
        out[n] = pool[i];
        n = n + 1;
    }
    free(pool);

    // initialize logp from the seed counts (normalized), then EM refines them.
    double total = 0.0;
    for (uint32_t i = 0; i < n; i = i + 1) {
        total = total + out[i].expected;
    }
    for (uint32_t i = 0; i < n; i = i + 1) {
        out[i].logp = log(out[i].expected / total);
    }
    return n;
}

// ---- EM over the current candidate set -------------------------------------

// find a candidate index for the exact substring s[0..len) by linear probe in a
// transient hash built each EM round; returns -1 if absent.
typedef struct lookup lookup_t;

struct lookup {
    int32_t *slot_id;
    char *keys;
    uint8_t *klen;
    uint8_t *used;
    uint32_t cap;
    uint32_t mask;
};

static lookup_t lookup_build(cand_t *cands, uint32_t n) {
    uint32_t cap = 1u << 20;
    while (cap < n * 2u) {
        cap = cap << 1;
    }
    lookup_t lk;
    lk.cap = cap;
    lk.mask = cap - 1;
    lk.slot_id = malloc(cap * sizeof(int32_t));
    lk.keys = calloc((size_t)cap * TOK_MAX_LEN, 1);
    lk.klen = calloc(cap, 1);
    lk.used = calloc(cap, 1);
    for (uint32_t i = 0; i < cap; i = i + 1) {
        lk.slot_id[i] = -1;
    }
    for (uint32_t i = 0; i < n; i = i + 1) {
        uint32_t x = 2166136261u;
        for (uint32_t j = 0; j < cands[i].len; j = j + 1) {
            x = (x ^ (unsigned char)cands[i].bytes[j]) * 16777619u;
        }
        uint32_t slot = x & lk.mask;
        while (lk.used[slot]) {
            slot = (slot + 1) & lk.mask;
        }
        lk.used[slot] = 1;
        lk.klen[slot] = (uint8_t)cands[i].len;
        memcpy(lk.keys + (size_t)slot * TOK_MAX_LEN, cands[i].bytes, cands[i].len);
        lk.slot_id[slot] = (int32_t)i;
    }
    return lk;
}

static void lookup_free(lookup_t *lk) {
    free(lk->slot_id);
    free(lk->keys);
    free(lk->klen);
    free(lk->used);
}

static int32_t lookup_find(const lookup_t *lk, const char *s, uint32_t len) {
    uint32_t x = 2166136261u;
    for (uint32_t j = 0; j < len; j = j + 1) {
        x = (x ^ (unsigned char)s[j]) * 16777619u;
    }
    uint32_t slot = x & lk->mask;
    while (lk->used[slot]) {
        if (lk->klen[slot] == len && memcmp(lk->keys + (size_t)slot * TOK_MAX_LEN, s, len) == 0) {
            return lk->slot_id[slot];
        }
        slot = (slot + 1) & lk->mask;
    }
    return -1;
}

// forward-backward over one chunk: accumulate expected counts of each candidate
// (the posterior probability that each token spans each position). Returns the
// chunk log-likelihood. This is the Unigram E-step.
static double em_chunk(
    cand_t *cands,
    const lookup_t *lk,
    const unsigned char *text,
    uint32_t len) {
    double *alpha = malloc((len + 1) * sizeof(double));
    double *beta = malloc((len + 1) * sizeof(double));
    for (uint32_t i = 0; i <= len; i = i + 1) {
        alpha[i] = NEG_INF;
        beta[i] = NEG_INF;
    }
    alpha[0] = 0.0;
    // forward: alpha[j] = log-sum-exp over tokens ending at j.
    for (uint32_t i = 0; i < len; i = i + 1) {
        if (alpha[i] <= NEG_INF + 1.0) {
            continue;
        }
        uint32_t cap_len = TOK_MAX_LEN;
        if (i + cap_len > len) {
            cap_len = len - i;
        }
        for (uint32_t l = 1; l <= cap_len; l = l + 1) {
            int32_t id = lookup_find(lk, (const char *)text + i, l);
            if (id < 0) {
                continue;
            }
            double s = alpha[i] + cands[id].logp;
            uint32_t j = i + l;
            if (alpha[j] <= NEG_INF + 1.0) {
                alpha[j] = s;
            } else {
                double hi = (alpha[j] > s) ? alpha[j] : s;
                alpha[j] = hi + log(exp(alpha[j] - hi) + exp(s - hi));
            }
        }
    }
    double total = alpha[len];
    if (total <= NEG_INF + 1.0) {
        free(alpha);
        free(beta);
        return 0.0;
    }
    // backward: beta[i] = log-sum-exp over tokens starting at i.
    beta[len] = 0.0;
    for (int32_t i = (int32_t)len - 1; i >= 0; i = i - 1) {
        uint32_t cap_len = TOK_MAX_LEN;
        if ((uint32_t)i + cap_len > len) {
            cap_len = len - (uint32_t)i;
        }
        for (uint32_t l = 1; l <= cap_len; l = l + 1) {
            int32_t id = lookup_find(lk, (const char *)text + i, l);
            if (id < 0) {
                continue;
            }
            uint32_t j = (uint32_t)i + l;
            if (beta[j] <= NEG_INF + 1.0) {
                continue;
            }
            double s = cands[id].logp + beta[j];
            if (beta[i] <= NEG_INF + 1.0) {
                beta[i] = s;
            } else {
                double hi = (beta[i] > s) ? beta[i] : s;
                beta[i] = hi + log(exp(beta[i] - hi) + exp(s - hi));
            }
        }
    }
    // posterior of token (i, l): alpha[i] + logp + beta[i+l] - total. accumulate.
    for (uint32_t i = 0; i < len; i = i + 1) {
        if (alpha[i] <= NEG_INF + 1.0) {
            continue;
        }
        uint32_t cap_len = TOK_MAX_LEN;
        if (i + cap_len > len) {
            cap_len = len - i;
        }
        for (uint32_t l = 1; l <= cap_len; l = l + 1) {
            int32_t id = lookup_find(lk, (const char *)text + i, l);
            if (id < 0) {
                continue;
            }
            uint32_t j = i + l;
            double post = alpha[i] + cands[id].logp + beta[j] - total;
            cands[id].expected = cands[id].expected + exp(post);
        }
    }
    free(alpha);
    free(beta);
    return total;
}

// one EM round: reset expected counts, run forward-backward over several random
// chunks, then renormalize logp from the expected counts (the M-step).
static void em_round(cand_t *cands, uint32_t n, corpus_t *cp, const lookup_t *lk) {
    for (uint32_t i = 0; i < n; i = i + 1) {
        cands[i].expected = 0.0;
    }
    uint32_t chunk = 4096;
    size_t budget = EM_SAMPLE_BYTES;
    size_t done = 0;
    double ll = 0.0;
    while (done < budget) {
        size_t start = (size_t)(rand_unit() * (double)(cp->len - chunk - 1));
        ll = ll + em_chunk(cands, lk, cp->bytes + start, chunk);
        done = done + chunk;
    }
    double total = 0.0;
    for (uint32_t i = 0; i < n; i = i + 1) {
        total = total + cands[i].expected;
    }
    for (uint32_t i = 0; i < n; i = i + 1) {
        double c = cands[i].expected;
        if (c < 1e-9) {
            c = 1e-9;
        }
        cands[i].logp = log(c / total);
    }
    fprintf(stderr, "  EM round: loglik %.1f over %zu bytes, vocab %u\n", ll, done, n);
}

// ---- pruning ---------------------------------------------------------------

static int cand_by_expected_asc(const void *pa, const void *pb) {
    const cand_t *a = pa;
    const cand_t *b = pb;
    if (a->len == 1 && b->len != 1) {
        return 1;
    }
    if (a->len != 1 && b->len == 1) {
        return -1;
    }
    if (a->expected < b->expected) {
        return -1;
    }
    if (a->expected > b->expected) {
        return 1;
    }
    return 0;
}

// keep the highest-count candidates (single bytes always survive), drop the rest.
static uint32_t prune(cand_t *cands, uint32_t n, uint32_t target) {
    qsort(cands, n, sizeof(cand_t), cand_by_expected_asc);
    uint32_t keep = (uint32_t)((double)n * PRUNE_KEEP);
    if (keep < target) {
        keep = target;
    }
    if (keep > n) {
        keep = n;
    }
    uint32_t drop = n - keep;
    // the array is ascending with single bytes forced to the high end, so dropping
    // the first `drop` entries never removes a single byte unless drop exceeds the
    // multi-byte count -- guard that.
    uint32_t multibyte = 0;
    for (uint32_t i = 0; i < n; i = i + 1) {
        if (cands[i].len > 1) {
            multibyte = multibyte + 1;
        }
    }
    if (drop > multibyte) {
        drop = multibyte;
    }
    memmove(cands, cands + drop, (n - drop) * sizeof(cand_t));
    return n - drop;
}

static void write_vocab(const char *path, cand_t *cands, uint32_t n) {
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        perror(path);
        exit(1);
    }
    fprintf(f, "%u\n", n);
    for (uint32_t i = 0; i < n; i = i + 1) {
        fprintf(f, "%.6f %u ", (float)cands[i].logp, cands[i].len);
        tok_escape(cands[i].bytes, cands[i].len, f);
        fputc('\n', f);
    }
    fclose(f);
    fprintf(stderr, "wrote %u tokens to %s\n", n, path);
}

int main(void) {
    srand(1234);
    corpus_t cp = corpus_load("data/wiki.txt");
    fprintf(stderr, "corpus: %zu bytes\n", cp.len);

    cand_t *cands = malloc((SEED_KEEP + 256) * sizeof(cand_t));
    uint32_t n = seed_candidates(&cp, cands);
    fprintf(stderr, "seeded %u candidates\n", n);

    while (1) {
        lookup_t lk = lookup_build(cands, n);
        for (uint32_t it = 0; it < EM_ITERS_PER_ROUND; it = it + 1) {
            em_round(cands, n, &cp, &lk);
        }
        lookup_free(&lk);
        if (n <= TARGET_VOCAB) {
            break;
        }
        n = prune(cands, n, TARGET_VOCAB);
        fprintf(stderr, "pruned to %u\n", n);
    }

    write_vocab("models/vocab.txt", cands, n);
    return 0;
}
