// tokenizer.h -- Unigram language-model subword tokenizer (the SentencePiece-style
// probabilistic model, not BPE). The vocab is a set of token strings, each with a
// log-probability. Encoding a string is the maximum-likelihood segmentation under
// that unigram model, found by Viterbi. The trainer that produces the vocab lives
// in train_tokenizer.c; this header is the runtime: load, encode, decode.
//
// Vocab file format (text, written by the trainer):
//   line 1:        <vocab_count>
//   lines 2..N+1:  <log_prob> <byte_length> <space> <raw_token_bytes>
// Token bytes are written raw (may contain any byte except newline is escaped as
// \n, backslash as \\), so the loader unescapes them.
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TOK_MAX_LEN 32

typedef struct tok_entry tok_entry_t;

struct tok_entry {
    char *bytes;
    uint32_t len;
    float logp;
};

// A byte trie over the vocab: walking it from a text position yields every token
// that matches a prefix there, in O(token length) with no vocab scan. Each node is
// 256 child slots (index into nodes, 0 = none) plus the token id that ends at this
// node (-1 if none). Children use index 0 as "empty" so the root is node 1.
typedef struct trie_node trie_node_t;

struct trie_node {
    uint32_t child[256];
    int32_t token;
};

typedef struct tokenizer tokenizer_t;

struct tokenizer {
    tok_entry_t *entries;
    uint32_t count;
    trie_node_t *nodes;
    uint32_t node_count;
};

static void tok_unescape(const char *src, char *dst, uint32_t *len_out) {
    uint32_t j = 0;
    for (uint32_t i = 0; src[i] != '\0'; i = i + 1) {
        if (src[i] == '\\' && src[i + 1] == 'n') {
            dst[j] = '\n';
            j = j + 1;
            i = i + 1;
        } else if (src[i] == '\\' && src[i + 1] == '\\') {
            dst[j] = '\\';
            j = j + 1;
            i = i + 1;
        } else {
            dst[j] = src[i];
            j = j + 1;
        }
    }
    dst[j] = '\0';
    *len_out = j;
}

static void tok_escape(const char *src, uint32_t len, FILE *f) {
    for (uint32_t i = 0; i < len; i = i + 1) {
        if (src[i] == '\n') {
            fputs("\\n", f);
        } else if (src[i] == '\\') {
            fputs("\\\\", f);
        } else {
            fputc(src[i], f);
        }
    }
}

static uint32_t trie_new_node(tokenizer_t *t) {
    uint32_t id = t->node_count;
    t->node_count = t->node_count + 1;
    trie_node_t *n = &t->nodes[id];
    memset(n->child, 0, sizeof(n->child));
    n->token = -1;
    return id;
}

// build the byte trie: insert every token's bytes, marking the terminal node with
// the token id. Node 0 is reserved as "empty child"; node 1 is the root.
static void tokenizer_index(tokenizer_t *t) {
    // worst case one node per byte of every token, plus root and the empty slot.
    uint32_t cap = 2;
    for (uint32_t i = 0; i < t->count; i = i + 1) {
        cap = cap + t->entries[i].len;
    }
    t->nodes = malloc(cap * sizeof(trie_node_t));
    t->node_count = 0;
    trie_new_node(t);
    uint32_t root = trie_new_node(t);
    for (uint32_t i = 0; i < t->count; i = i + 1) {
        uint32_t node = root;
        const char *b = t->entries[i].bytes;
        for (uint32_t k = 0; k < t->entries[i].len; k = k + 1) {
            unsigned char c = (unsigned char)b[k];
            if (t->nodes[node].child[c] == 0) {
                uint32_t fresh = trie_new_node(t);
                t->nodes[node].child[c] = fresh;
            }
            node = t->nodes[node].child[c];
        }
        t->nodes[node].token = (int32_t)i;
    }
}

static tokenizer_t tokenizer_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        perror(path);
        exit(1);
    }
    tokenizer_t t;
    if (fscanf(f, "%u\n", &t.count) != 1) {
        fprintf(stderr, "bad vocab file\n");
        exit(1);
    }
    t.entries = malloc(t.count * sizeof(tok_entry_t));
    char line[TOK_MAX_LEN * 4 + 64];
    for (uint32_t i = 0; i < t.count; i = i + 1) {
        float logp = 0.0f;
        uint32_t blen = 0;
        // read "<logp> <len> " then the rest of the line verbatim. %u stops at the
        // space, so we step over exactly one separator and take the rest as the raw
        // (escaped) token -- tokens may themselves start with or contain spaces.
        if (fscanf(f, "%f %u", &logp, &blen) != 2) {
            fprintf(stderr, "bad vocab entry %u\n", i);
            exit(1);
        }
        fgetc(f);
        if (fgets(line, sizeof(line), f) == NULL) {
            fprintf(stderr, "bad vocab entry %u\n", i);
            exit(1);
        }
        size_t ll = strlen(line);
        if (ll > 0 && line[ll - 1] == '\n') {
            line[ll - 1] = '\0';
        }
        char *raw = malloc(TOK_MAX_LEN + 1);
        uint32_t rlen = 0;
        tok_unescape(line, raw, &rlen);
        t.entries[i].bytes = raw;
        t.entries[i].len = rlen;
        t.entries[i].logp = logp;
    }
    fclose(f);
    tokenizer_index(&t);
    return t;
}

// Viterbi maximum-likelihood segmentation of text[0..len) into token ids. Writes
// up to max_out ids into out and returns the count. Every single byte exists as a
// token (the trainer guarantees this) so a segmentation always exists. For each
// start position we walk the trie once, relaxing every token that matches there.
static uint32_t tokenizer_encode(
    const tokenizer_t *t,
    const char *text,
    uint32_t len,
    uint32_t *out,
    uint32_t max_out) {
    float *best = malloc((len + 1) * sizeof(float));
    int32_t *back_id = malloc((len + 1) * sizeof(int32_t));
    uint32_t *back_pos = malloc((len + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= len; i = i + 1) {
        best[i] = -1e30f;
        back_id[i] = -1;
        back_pos[i] = 0;
    }
    best[0] = 0.0f;
    for (uint32_t i = 0; i < len; i = i + 1) {
        if (best[i] <= -1e29f) {
            continue;
        }
        uint32_t node = 1;
        uint32_t max_len = TOK_MAX_LEN;
        if (i + max_len > len) {
            max_len = len - i;
        }
        for (uint32_t l = 1; l <= max_len; l = l + 1) {
            unsigned char c = (unsigned char)text[i + l - 1];
            uint32_t next = t->nodes[node].child[c];
            if (next == 0) {
                break;
            }
            node = next;
            int32_t id = t->nodes[node].token;
            if (id < 0) {
                continue;
            }
            float score = best[i] + t->entries[id].logp;
            if (score > best[i + l]) {
                best[i + l] = score;
                back_id[i + l] = id;
                back_pos[i + l] = i;
            }
        }
    }
    uint32_t tmp_count = 0;
    uint32_t *tmp = malloc((len + 1) * sizeof(uint32_t));
    uint32_t pos = len;
    while (pos > 0) {
        tmp[tmp_count] = (uint32_t)back_id[pos];
        tmp_count = tmp_count + 1;
        pos = back_pos[pos];
    }
    uint32_t n = (tmp_count < max_out) ? tmp_count : max_out;
    for (uint32_t i = 0; i < n; i = i + 1) {
        out[i] = tmp[tmp_count - 1 - i];
    }
    free(best);
    free(back_id);
    free(back_pos);
    free(tmp);
    return n;
}

static void tokenizer_free(tokenizer_t *t) {
    for (uint32_t i = 0; i < t->count; i = i + 1) {
        free(t->entries[i].bytes);
    }
    free(t->entries);
    free(t->nodes);
}

#endif
