// Verify masked cross-entropy: loss (vs CPU) and dlogits (vs finite difference).
#include "vkc.h"
#include <math.h>

typedef struct ce_push ce_push_t;

struct ce_push {
    uint32_t rows;
    uint32_t vocab;
};

static vkc_t c;
static vkc_kernel_t kn;
static uint32_t g_rows = 4;
static uint32_t g_vocab = 17;

static void run_ce(vkc_buf_t *bl, vkc_buf_t *bt, vkc_buf_t *bm,
    vkc_buf_t *bw, vkc_buf_t *bloss) {
    ce_push_t push = (ce_push_t) {
        .rows = g_rows,
        .vocab = g_vocab,
    };
    vkc_buf_t *b[5] = {
        bl,
        bt,
        bm,
        bw,
        bloss,
    };
    vkc_run(&c, &kn, b, g_rows, 1, 1, &push);
}

static double cpu_loss(const float *logits, const uint32_t *tgt,
    const float *mask, const float *wt) {
    double total = 0.0;
    for (uint32_t row = 0; row < g_rows; row++) {
        const float *lg = logits + row * g_vocab;
        double mx = -1e30;
        for (uint32_t i = 0; i < g_vocab; i++) {
            if (lg[i] > mx) {
                mx = lg[i];
            }
        }
        double se = 0.0;
        for (uint32_t i = 0; i < g_vocab; i++) {
            se += exp(lg[i] - mx);
        }
        double logz = mx + log(se);
        double logp = lg[tgt[row]] - logz;
        total += -mask[row] * wt[row] * logp;
    }
    return total;
}

int main(void) {
    c = vkc_init();
    kn = vkc_load(&c, "build/softmax_ce.spv", 5, sizeof(ce_push_t));

    uint32_t n = g_rows * g_vocab;
    vkc_buf_t bl = vkc_alloc(&c, n * sizeof(float));
    vkc_buf_t bt = vkc_alloc(&c, g_rows * sizeof(uint32_t));
    vkc_buf_t bm = vkc_alloc(&c, g_rows * sizeof(float));
    vkc_buf_t bw = vkc_alloc(&c, g_rows * sizeof(float));
    vkc_buf_t bloss = vkc_alloc(&c, g_rows * sizeof(float));

    float *logits = bl.ptr;
    uint32_t *tgt = bt.ptr;
    float *mask = bm.ptr;
    float *wt = bw.ptr;
    float *loss = bloss.ptr;

    float *l0 = malloc(n * sizeof(float));
    srand(6);
    for (uint32_t i = 0; i < n; i++) {
        logits[i] = ((rand() % 400) / 100.0f - 2.0f);
        l0[i] = logits[i];
    }
    for (uint32_t r = 0; r < g_rows; r++) {
        tgt[r] = rand() % g_vocab;
        mask[r] = (r % 2 == 0) ? 1.0f : 0.0f;
        wt[r] = 1.0f + (rand() % 100) / 100.0f;
    }

    run_ce(&bl, &bt, &bm, &bw, &bloss);
    // loss check
    double gpu_loss = 0.0;
    for (uint32_t r = 0; r < g_rows; r++) {
        gpu_loss += loss[r];
    }
    double ref_loss = cpu_loss(l0, tgt, mask, wt);
    double lerr = fabs(gpu_loss - ref_loss) / (fabs(ref_loss) + 1e-4);

    // gradient (kernel wrote dlogits into logits) check via FD on the loss
    float *grad = malloc(n * sizeof(float));
    for (uint32_t i = 0; i < n; i++) {
        grad[i] = logits[i];
    }
    double eps = 1e-3;
    double gerr = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        float orig = l0[i];
        l0[i] = orig + (float)eps;
        double lp = cpu_loss(l0, tgt, mask, wt);
        l0[i] = orig - (float)eps;
        double lm = cpu_loss(l0, tgt, mask, wt);
        l0[i] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double rel = fabs(num - grad[i]) / (fabs(num) + fabs(grad[i]) + 1e-3);
        if (rel > gerr) {
            gerr = rel;
        }
    }

    printf("softmax_ce: loss err = %g, grad rel = %g\n", lerr, gerr);
    int ok = lerr < 1e-4 && gerr < 1e-2;
    printf("%s\n", ok ? "SOFTMAX_CE OK" : "SOFTMAX_CE WRONG");
    return ok ? 0 : 1;
}
