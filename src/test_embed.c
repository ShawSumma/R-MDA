// Verify embed forward (vs CPU) and backward scatter-grad (vs finite difference).
#include "vkc.h"
#include <math.h>

typedef struct emb_push emb_push_t;

struct emb_push {
    uint32_t bt;
    uint32_t d;
    uint32_t seqlen;
    uint32_t op;
};

static vkc_t c;
static vkc_kernel_t kn;
static uint32_t g_vocab = 11;
static uint32_t g_seqlen = 5;
static uint32_t g_b = 2;
static uint32_t g_d = 7;

static void run_embed(vkc_buf_t *bids, vkc_buf_t *btok, vkc_buf_t *bpos,
    vkc_buf_t *bout, uint32_t op) {
    uint32_t bt = g_b * g_seqlen;
    emb_push_t push = (emb_push_t) {
        .bt = bt,
        .d = g_d,
        .seqlen = g_seqlen,
        .op = op,
    };
    vkc_buf_t *b[4] = {
        bids,
        btok,
        bpos,
        bout,
    };
    vkc_run(&c, &kn, b, (bt * g_d + 255) / 256, 1, 1, &push);
}

int main(void) {
    c = vkc_init();
    kn = vkc_load(&c, "build/embed.spv", 4, sizeof(emb_push_t));

    uint32_t bt = g_b * g_seqlen;
    vkc_buf_t bids = vkc_alloc(&c, bt * sizeof(uint32_t));
    vkc_buf_t btok = vkc_alloc(&c, g_vocab * g_d * sizeof(float));
    vkc_buf_t bpos = vkc_alloc(&c, g_seqlen * g_d * sizeof(float));
    vkc_buf_t bout = vkc_alloc(&c, bt * g_d * sizeof(float));

    uint32_t *ids = bids.ptr;
    float *tok = btok.ptr;
    float *pos = bpos.ptr;
    float *out = bout.ptr;

    srand(2);
    for (uint32_t i = 0; i < bt; i++) {
        ids[i] = rand() % g_vocab;
    }
    float *tok0 = malloc(g_vocab * g_d * sizeof(float));
    float *pos0 = malloc(g_seqlen * g_d * sizeof(float));
    for (uint32_t i = 0; i < g_vocab * g_d; i++) {
        tok[i] = ((rand() % 200) / 100.0f - 1.0f);
        tok0[i] = tok[i];
    }
    for (uint32_t i = 0; i < g_seqlen * g_d; i++) {
        pos[i] = ((rand() % 200) / 100.0f - 1.0f);
        pos0[i] = pos[i];
    }

    // forward
    run_embed(&bids, &btok, &bpos, &bout, 0);
    double ferr = 0.0;
    for (uint32_t r = 0; r < bt; r++) {
        uint32_t t = r % g_seqlen;
        uint32_t id = ids[r];
        for (uint32_t col = 0; col < g_d; col++) {
            float ref = tok0[id * g_d + col] + pos0[t * g_d + col];
            double e = fabs(out[r * g_d + col] - ref);
            if (e > ferr) {
                ferr = e;
            }
        }
    }

    // backward: random dy in out, zero grads in tok/pos, scatter
    float *dy = malloc(bt * g_d * sizeof(float));
    for (uint32_t i = 0; i < bt * g_d; i++) {
        dy[i] = ((rand() % 200) / 100.0f - 1.0f);
        out[i] = dy[i];
    }
    for (uint32_t i = 0; i < g_vocab * g_d; i++) {
        tok[i] = 0.0f;
    }
    for (uint32_t i = 0; i < g_seqlen * g_d; i++) {
        pos[i] = 0.0f;
    }
    run_embed(&bids, &btok, &bpos, &bout, 1);

    // finite diff on tok_emb: loss = sum(dy .* fwd). d_tok = sum over rows using that id.
    double tok_err = 0.0;
    for (uint32_t p = 0; p < g_vocab * g_d; p++) {
        // analytic: sum of dy at (row, col) where ids[row]==p/g_d and col==p%g_d
        uint32_t id = p / g_d;
        uint32_t col = p % g_d;
        double num = 0.0;
        for (uint32_t r = 0; r < bt; r++) {
            if (ids[r] == id) {
                num += dy[r * g_d + col];
            }
        }
        double rel = fabs(num - tok[p]) / (fabs(num) + fabs(tok[p]) + 1e-4);
        if (rel > tok_err) {
            tok_err = rel;
        }
    }

    double pos_err = 0.0;
    for (uint32_t p = 0; p < g_seqlen * g_d; p++) {
        uint32_t t = p / g_d;
        uint32_t col = p % g_d;
        double num = 0.0;
        for (uint32_t r = 0; r < bt; r++) {
            if ((r % g_seqlen) == t) {
                num += dy[r * g_d + col];
            }
        }
        double rel = fabs(num - pos[p]) / (fabs(num) + fabs(pos[p]) + 1e-4);
        if (rel > pos_err) {
            pos_err = rel;
        }
    }

    printf("embed: fwd err = %g, d_tok rel = %g, d_pos rel = %g\n", ferr, tok_err, pos_err);
    int ok = ferr < 1e-5 && tok_err < 1e-3 && pos_err < 1e-3;
    printf("%s\n", ok ? "EMBED OK" : "EMBED WRONG");
    return ok ? 0 : 1;
}
