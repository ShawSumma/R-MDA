// generate.c -- masked-diffusion text generation by iterative denoising, and a
// measurement of real generation cost (which is N forward passes, not one). Starts
// from an all-masked sequence and, over N steps, predicts every masked position and
// unmasks the highest-confidence ones (the standard confidence schedule). Fewer steps
// = proportionally faster generation, the main inference lever for diffusion LMs.
#include "model.h"
#include "fwd.h"
#include "step.h"
#include <time.h>

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    srand(7);
    uint32_t steps = (argc > 1) ? (uint32_t)atoi(argv[1]) : 16;
    vkc_t c = vkc_init();
    corpus_t cp = corpus_load("data/wiki.txt");
    mcfg_t cfg = (mcfg_t) {
        .d = 640,
        .n_layer = 12,
        .n_head = 10,
        .seqlen = 64,
        .batch = 8,
        .ffn_hidden = 1728,
        .vocab = data_vocab,
        .vocab_in = data_vocab + 1,
    };
    model_t *m = model_init(&c, cfg);
    m->infer = 1;
    fwd_scratch_t sc = fwd_alloc(m);
    // GPU argmax over the logits each step (host loop over 16k vocab was the wall).
    vkc_kernel_t k_argmax = vkc_load(&c, "build/argmax.spv", 3, 8);
    uint32_t bt_all = cfg.batch * cfg.seqlen;
    vkc_buf_t amx_tok = vkc_alloc(&c, bt_all * sizeof(uint32_t));
    vkc_buf_t amx_conf = vkc_alloc(&c, bt_all * sizeof(float));
    {
        FILE *f = fopen("models/ckpt.bin", "rb");
        if (f != NULL) {
            uint32_t step = 0;
            fread(&step, sizeof(uint32_t), 1, f);
            for (uint32_t i = 0; i < m->n_params; ++i) {
                param_t *p = &m->params[i];
                fread(p->w.ptr, sizeof(float), p->n, f);
                fread(p->m.ptr, sizeof(float), p->n, f);
                fread(p->v.ptr, sizeof(float), p->n, f);
            }
            fclose(f);
            fprintf(stderr, "loaded checkpoint @ step %u\n", step);
        }
        m->dirty16 = 1;
    }

    uint32_t seqlen = cfg.seqlen;
    uint32_t bt = cfg.batch * seqlen;
    uint32_t *ids = sc.ids.ptr;
    float *weight = sc.weight.ptr;
    // start fully masked; the weight buffer feeds the (unused) t-cond, set to t=1.
    for (uint32_t i = 0; i < bt; ++i) {
        ids[i] = data_mask_id;
        weight[i] = 1.0f;
    }

    uint32_t cand_tok[64];
    float cand_conf[64];
    double t0 = now_sec();
    for (uint32_t s = 0; s < steps; ++s) {
        model_forward(m, &sc);
        // GPU argmax: per position, top-1 token + confidence (reads 16k logits on GPU).
        argmax_pc_t argmax_push = (argmax_pc_t) {
            .rows = bt,
            .vocab = cfg.vocab,
        };
        vkc_buf_t *argmax_bufs[3] = {
            &sc.logits_save,
            &amx_tok,
            &amx_conf,
        };
        vkc_run(&c, &k_argmax, argmax_bufs, bt, 1, 1, &argmax_push);
        uint32_t *amtok = amx_tok.ptr;
        float *amconf = amx_conf.ptr;
        // argmax each masked position ONCE this step (top-1 token + its confidence),
        // then reveal the highest-confidence ones. (Re-argmaxing per reveal was the
        // bottleneck -- host loops over the 16k-vocab logits dominate the forwards.)
        for (uint32_t b = 0; b < cfg.batch; ++b) {
            uint32_t masked = 0;
            for (uint32_t p = 0; p < seqlen; ++p) {
                if (ids[b * seqlen + p] == data_mask_id) {
                    masked = masked + 1;
                }
            }
            if (masked == 0) {
                continue;
            }
            uint32_t remaining_steps = steps - s;
            uint32_t reveal = (masked + remaining_steps - 1) / remaining_steps;
            for (uint32_t p = 0; p < seqlen; ++p) {
                uint32_t idx = b * seqlen + p;
                if (ids[idx] != data_mask_id) {
                    cand_conf[p] = -1e30f;
                    continue;
                }
                cand_tok[p] = amtok[idx];
                cand_conf[p] = amconf[idx];
            }
            // reveal the top `reveal` positions by confidence (simple selection).
            for (uint32_t r = 0; r < reveal; ++r) {
                float best = -1e30f;
                uint32_t bp = seqlen;
                for (uint32_t p = 0; p < seqlen; ++p) {
                    if (cand_conf[p] > best) {
                        best = cand_conf[p];
                        bp = p;
                    }
                }
                if (bp < seqlen) {
                    ids[b * seqlen + bp] = cand_tok[bp];
                    cand_conf[bp] = -1e30f;
                }
            }
        }
    }
    double dt = now_sec() - t0;

    // show the first sequence as decoded text.
    printf("=== generated (%u steps, %.1f ms, %.0f tok/s) ===\n", steps, dt * 1e3, bt / dt);
    for (uint32_t p = 0; p < seqlen; ++p) {
        uint32_t tok = ids[p];
        if (tok >= cfg.vocab) {
            continue;
        }
        tok_entry_t *e = &cp.tok.entries[tok];
        fwrite(e->bytes, 1, e->len, stdout);
    }
    printf("\n");
    return 0;
}
