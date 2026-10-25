#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen.h"
#include "pc.h"
#include "util.h"

mda_qwen_cfg_t mda_qwen_cfg(void) {
    return (mda_qwen_cfg_t) {
        .d = 2560,
        .n_layer = 48,
        .vocab = 248320,
        .hc_rank = 320,
        .k_heads = 16,
        .v_heads = 48,
        .lin_head = 128,
        .conv = 4,
        .heads = 24,
        .kv_heads = 2,
        .head_dim = 256,
        .rot = 64,
        .experts = 512,
        .topk = 10,
        .inter = 640,
        .shared = 640,
        .attn_every = 4,
        .ple_layer = 1,
        .ngram = 3,
        .ngram_heads = 16,
        .ple_conv = 4,
        .ple_dil = 3,
        .eos = 248044,
    };
}

uint32_t mda_qwen_round(uint32_t n, uint32_t to) {
    return (n + to - 1) / to * to;
}

uint32_t mda_qwen_qkv_cols(mda_qwen_cfg_t cfg) {
    return 2 * cfg.k_heads * cfg.lin_head + cfg.v_heads * cfg.lin_head;
}

uint32_t mda_qwen_gdn_cols(mda_qwen_cfg_t cfg) {
    return mda_qwen_round(mda_qwen_qkv_cols(cfg) + cfg.v_heads * cfg.lin_head + 2 * cfg.v_heads, MDA_QWEN_TILE);
}

uint32_t mda_qwen_attn_cols(mda_qwen_cfg_t cfg) {
    return mda_qwen_round(2 * cfg.heads * cfg.head_dim + 2 * cfg.kv_heads * cfg.head_dim, MDA_QWEN_TILE);
}

uint32_t mda_qwen_moe_cols(mda_qwen_cfg_t cfg) {
    return cfg.experts + MDA_QWEN_MOE_PAD + 2 * cfg.shared;
}

uint32_t mda_qwen_ple_cols(mda_qwen_cfg_t cfg) {
    return MDA_QWEN_HC * cfg.d + cfg.d;
}

uint32_t mda_qwen_proj_cols(mda_qwen_cfg_t cfg) {
    uint32_t a = mda_util_max_u32(mda_qwen_gdn_cols(cfg), mda_qwen_attn_cols(cfg));
    return mda_util_max_u32(a, mda_util_max_u32(mda_qwen_moe_cols(cfg), mda_qwen_ple_cols(cfg)));
}

uint32_t mda_qwen_moe_rows(mda_qwen_cfg_t cfg, uint32_t rows) {
    return mda_qwen_round(rows * cfg.topk + cfg.experts * (MDA_QWEN_TILE - 1), MDA_QWEN_TILE);
}

bool mda_qwen_is_attn(mda_qwen_cfg_t cfg, uint32_t l) {
    return (l + 1) % cfg.attn_every == 0;
}

mda_safet_tensor_t mda_qwen_find(const mda_qwen_t *q, const char *name) {
    char full[2 * MDA_SAFET_MAX_KEY];
    snprintf(full, sizeof(full), "%s%s", MDA_QWEN_PREFIX, name);
    return mda_safet_find(&q->set, full);
}

mda_safet_tensor_t mda_qwen_layer_find(const mda_qwen_t *q, uint32_t l, const char *suffix) {
    char name[MDA_SAFET_MAX_KEY];
    snprintf(name, sizeof(name), "layers.%u.%s", l, suffix);
    return mda_qwen_find(q, name);
}

bool mda_qwen_shape(mda_safet_tensor_t t, uint32_t rows, uint32_t cols, const char *what) {
    bool ok = t.ok && t.rows == rows && t.cols == cols;
    if (t.ok && ok == false) {
        fprintf(stderr, "qwen: %s is %u x %u, expected %u x %u\n", what, t.rows, t.cols, rows, cols);
    }
    return ok;
}

bool mda_qwen_check(const mda_qwen_t *q) {
    mda_qwen_cfg_t cfg = q->cfg;
    uint32_t d = cfg.d;
    uint32_t bad = 0;
    bad += mda_qwen_shape(mda_qwen_find(q, "embed_tokens.weight"), cfg.vocab, d, "embed_tokens") == false;
    bad += mda_qwen_shape(mda_qwen_layer_find(q, 0, "mlp.gate.weight"), cfg.experts, d, "router") == false;
    bad += mda_qwen_shape(mda_qwen_layer_find(q, 0, "mlp.experts.gate_up_proj"), cfg.experts * 2 * cfg.inter, d, "gate_up_proj") == false;
    bad += mda_qwen_shape(mda_qwen_layer_find(q, 0, "mlp.experts.down_proj"), cfg.experts * d, cfg.inter, "down_proj") == false;
    bad += mda_qwen_shape(mda_qwen_layer_find(q, 0, "linear_attn.in_proj_qkv.weight"), mda_qwen_qkv_cols(cfg), d, "in_proj_qkv") == false;
    bad += mda_qwen_shape(mda_qwen_layer_find(q, 0, "attn_hyper_connection.input_mix_weight_down.weight"), cfg.hc_rank, MDA_QWEN_HC * d, "input_mix_weight_down") == false;
    bad += mda_qwen_shape(mda_qwen_layer_find(q, cfg.attn_every - 1, "self_attn.q_proj.weight"), 2 * cfg.heads * cfg.head_dim, d, "q_proj") == false;
    bad += mda_qwen_shape(mda_qwen_layer_find(q, cfg.attn_every - 1, "self_attn.k_proj.weight"), cfg.kv_heads * cfg.head_dim, d, "k_proj") == false;
    bad += mda_qwen_shape(mda_qwen_layer_find(q, cfg.n_layer - 1, "mlp.gate.weight"), cfg.experts, d, "last router") == false;
    return bad == 0;
}

bool mda_qwen_geometry(uint32_t rows, uint32_t chunk, uint32_t seqlen, uint32_t head_rows) {
    bool ok = seqlen > 0 && seqlen % 64 == 0 && chunk % seqlen == 0 && chunk > 0 && rows % chunk == 0 && head_rows % 64 == 0 && head_rows > 0 && chunk % head_rows == 0;
    if (ok == false) {
        fprintf(stderr, "qwen: rows %u, chunk %u, seqlen %u, head rows %u: seqlen and head rows must be multiples of 64 dividing chunk, and chunk divide rows\n", rows, chunk, seqlen, head_rows);
    }
    return ok;
}

mda_qwen_stage_t mda_qwen_stage(mda_vkc_t *c, uint64_t bytes) {
    return (mda_qwen_stage_t) {
        .host = mda_vkc_alloc_cached(c, bytes),
        .used = 0,
    };
}

void mda_qwen_kernels(mda_qwen_t *q) {
    mda_vkc_t *c = q->c;
    q->k_mm16 = mda_vkc_load(c, "build/matmul_f16.spv", 6, sizeof(mda_pc_mm16_t));
    q->k_mm16w = mda_vkc_load(c, "build/matmul_f16_wide.spv", 6, sizeof(mda_pc_mm16_t));
    q->k_moe = mda_vkc_load(c, "build/matmul_f16_moe.spv", 5, sizeof(mda_pc_moe_mm_t));
    q->k_cast = mda_vkc_load(c, "build/cast_f16.spv", 3, sizeof(mda_pc_cast_t));
    q->k_absmax = mda_vkc_load(c, "build/absmax.spv", 2, sizeof(mda_pc_absmax_t));
    q->k_bf16 = mda_vkc_load(c, "build/qwen_bf16.spv", 1, sizeof(mda_pc_qwen_bf16_t));
    q->k_embed = mda_vkc_load(c, "build/qwen_embed.spv", 3, sizeof(mda_pc_qwen_embed_t));
    q->k_norm = mda_vkc_load(c, "build/qwen_norm.spv", 4, sizeof(mda_pc_qwen_norm_t));
    q->k_hc = mda_vkc_load(c, "build/qwen_hc.spv", 3, sizeof(mda_pc_qwen_hc_t));
    q->k_elem = mda_vkc_load(c, "build/qwen_elem.spv", 2, sizeof(mda_pc_qwen_elem_t));
    q->k_conv = mda_vkc_load(c, "build/qwen_conv.spv", 3, sizeof(mda_pc_qwen_conv_t));
    q->k_gdn = mda_vkc_load(c, "build/qwen_gdn.spv", 4, sizeof(mda_pc_qwen_gdn_t));
    q->k_attn = mda_vkc_load(c, "build/qwen_attn.spv", 4, sizeof(mda_pc_qwen_attn_t));
    q->k_ple = mda_vkc_load(c, "build/qwen_ple.spv", 4, sizeof(mda_pc_qwen_ple_t));
    q->k_route = mda_vkc_load(c, "build/moe_route.spv", 4, sizeof(mda_pc_moe_route_t));
    q->k_scan = mda_vkc_load(c, "build/moe_scan.spv", 3, sizeof(mda_pc_moe_scan_t));
    q->k_scatter = mda_vkc_load(c, "build/moe_scatter.spv", 6, sizeof(mda_pc_moe_scatter_t));
    q->k_combine = mda_vkc_load(c, "build/moe_combine.spv", 7, sizeof(mda_pc_moe_combine_t));
}

mda_qwen_scratch_t mda_qwen_scratch(mda_vkc_t *c, mda_qwen_cfg_t cfg, uint32_t chunk, uint32_t head_rows) {
    VkDeviceSize r = chunk;
    VkDeviceSize d4 = (VkDeviceSize)MDA_QWEN_HC * cfg.d;
    uint32_t mrows = mda_qwen_moe_rows(cfg, chunk);
    VkDeviceSize mr = mrows;
    return (mda_qwen_scratch_t) {
        .hn = mda_vkc_alloc(c, r * d4 * sizeof(float)),
        .t = mda_vkc_alloc(c, r * MDA_QWEN_HC_PAD * sizeof(float)),
        .u = mda_vkc_alloc(c, r * cfg.hc_rank * sizeof(float)),
        .up = mda_vkc_alloc(c, r * d4 * sizeof(float)),
        .x = mda_vkc_alloc(c, r * cfg.d * sizeof(float)),
        .p = mda_vkc_alloc(c, r * mda_qwen_proj_cols(cfg) * sizeof(float)),
        .s1 = mda_vkc_alloc(c, r * d4 * sizeof(float)),
        .s2 = mda_vkc_alloc(c, r * d4 * sizeof(float)),
        .s3 = mda_vkc_alloc(c, r * d4 * sizeof(float)),
        .y = mda_vkc_alloc(c, r * cfg.d * sizeof(float)),
        .a16 = mda_vkc_alloc(c, r * d4 * sizeof(uint16_t)),
        .idx = mda_vkc_alloc(c, r * cfg.topk * sizeof(uint32_t)),
        .wt = mda_vkc_alloc(c, r * cfg.topk * sizeof(float)),
        .slot = mda_vkc_alloc(c, r * cfg.topk * sizeof(uint32_t)),
        .count = mda_vkc_alloc(c, cfg.experts * sizeof(uint32_t)),
        .off = mda_vkc_alloc(c, cfg.experts * sizeof(uint32_t)),
        .fill = mda_vkc_alloc(c, cfg.experts * sizeof(uint32_t)),
        .tiles = mda_vkc_alloc(c, mr / MDA_QWEN_TILE * sizeof(uint32_t)),
        .xp = mda_vkc_alloc(c, mr * cfg.d * sizeof(uint16_t)),
        .g = mda_vkc_alloc(c, mr * 2 * cfg.inter * sizeof(float)),
        .hp = mda_vkc_alloc(c, mr * cfg.inter * sizeof(float)),
        .hp16 = mda_vkc_alloc(c, mr * cfg.inter * sizeof(uint16_t)),
        .ye = mda_vkc_alloc(c, mr * cfg.d * sizeof(float)),
        .hs = mda_vkc_alloc(c, r * cfg.shared * sizeof(float)),
        .sh = mda_vkc_alloc(c, r * cfg.d * sizeof(float)),
        .e = mda_vkc_alloc(c, r * cfg.d * sizeof(float)),
        .e_host = mda_vkc_alloc_cached(c, r * cfg.d * sizeof(float)),
        .logits = mda_vkc_alloc(c, (VkDeviceSize)head_rows * cfg.vocab * sizeof(float)),
        .moe_rows = mrows,
    };
}

bool mda_qwen_ple_shards(mda_qwen_ple_t *p, mda_qwen_cfg_t cfg) {
    uint32_t bad = p->n_shards == 0;
    for (uint32_t i = 0; i < p->n_shards; i += 1) {
        bad += p->shards[i].ok == false || p->shards[i].rows != p->shards[0].rows || p->shards[i].cols * cfg.ngram_heads != cfg.d;
    }
    if (bad == 0) {
        p->shard_rows = p->shards[0].rows;
        p->dim = p->shards[0].cols;
    }
    return bad == 0 && p->offs[cfg.ngram_heads - 1] + p->sizes[cfg.ngram_heads - 1] <= (uint64_t)p->shard_rows * p->n_shards;
}

bool mda_qwen_ple_open(mda_qwen_t *q) {
    mda_qwen_ple_t *p = &q->ple;
    mda_qwen_cfg_t cfg = q->cfg;
    uint32_t l = cfg.ple_layer;
    mda_safet_tensor_t mult = mda_qwen_layer_find(q, l, "ple.ple_embedding.layer_multipliers");
    mda_safet_tensor_t sizes = mda_qwen_layer_find(q, l, "ple.ple_embedding.ngram_heads_vocab_sizes");
    mda_safet_tensor_t offs = mda_qwen_layer_find(q, l, "ple.ple_embedding.ngram_heads_offsets");
    bool ok = mult.ok && sizes.ok && offs.ok && mult.dtype == MDA_SAFET_I64 && sizes.dtype == MDA_SAFET_I64 && offs.dtype == MDA_SAFET_I64 && mult.cols == cfg.ngram && sizes.cols == cfg.ngram_heads && offs.cols == cfg.ngram_heads && cfg.ngram <= MDA_QWEN_NGRAM_MAX;
    for (uint32_t i = 0; i < cfg.ngram && ok; i += 1) {
        p->mult[i] = (uint64_t)mda_safet_i64(mult, i);
    }
    for (uint32_t i = 0; i < cfg.ngram_heads && ok; i += 1) {
        p->sizes[i] = (uint64_t)mda_safet_i64(sizes, i);
        p->offs[i] = (uint64_t)mda_safet_i64(offs, i);
    }
    p->n_shards = 0;
    char name[MDA_SAFET_MAX_KEY];
    snprintf(name, sizeof(name), "%slayers.%u.ple.ple_embedding.ngram_embedding.shard_0.weight", MDA_QWEN_PREFIX, l);
    while (p->n_shards < MDA_QWEN_PLE_SHARDS && mda_safet_set_has(&q->set, name)) {
        p->shards[p->n_shards] = mda_safet_find(&q->set, name);
        p->n_shards += 1;
        snprintf(name, sizeof(name), "%slayers.%u.ple.ple_embedding.ngram_embedding.shard_%u.weight", MDA_QWEN_PREFIX, l, p->n_shards);
    }
    return ok && mda_qwen_ple_shards(p, cfg);
}

mda_qwen_t *mda_qwen_build(mda_qwen_t *q, uint32_t rows, uint32_t chunk, uint32_t seqlen, uint32_t head_rows) {
    mda_vkc_t *c = q->c;
    mda_qwen_cfg_t cfg = q->cfg;
    mda_safet_tensor_t gu = mda_qwen_layer_find(q, 0, "mlp.experts.gate_up_proj");
    mda_safet_tensor_t dn = mda_qwen_layer_find(q, 0, "mlp.experts.down_proj");
    q->rows = rows;
    q->chunk = chunk;
    q->seqlen = seqlen;
    q->head_rows = head_rows;
    VkDeviceSize nd = MDA_QWEN_DENSE_CAP * sizeof(uint16_t);
    VkDeviceSize ng = (VkDeviceSize)gu.rows * gu.cols * sizeof(uint16_t);
    VkDeviceSize nn = (VkDeviceSize)dn.rows * dn.cols * sizeof(uint16_t);
    VkDeviceSize nv = MDA_QWEN_VEC_CAP * sizeof(float);
    for (uint32_t i = 0; i < 2; i += 1) {
        q->bank[i].dense = mda_qwen_stage(c, nd);
        q->bank[i].gu = mda_qwen_stage(c, ng);
        q->bank[i].down = mda_qwen_stage(c, nn);
        q->bank[i].vec = mda_qwen_stage(c, nv);
    }
    q->wd = mda_vkc_alloc(c, nd);
    q->wgu = mda_vkc_alloc(c, ng);
    q->wdown = mda_vkc_alloc(c, nn);
    q->wvec = mda_vkc_alloc(c, nv);
    q->h = mda_vkc_alloc(c, (VkDeviceSize)rows * MDA_QWEN_HC * cfg.d * sizeof(float));
    q->ids = mda_vkc_alloc(c, (VkDeviceSize)rows * sizeof(uint32_t));
    q->amax = mda_vkc_alloc(c, MDA_QWEN_MAX_SLOTS * sizeof(uint32_t));
    q->sc = mda_qwen_scratch(c, cfg, chunk, head_rows);
    mda_qwen_kernels(q);
    if (c->fail) {
        return NULL;
    } else {
        fprintf(stderr, "qwen: %u layers, d %u, %u experts, %u rows in chunks of %u, %.2f GB on the GPU\n", cfg.n_layer, cfg.d, cfg.experts, rows, chunk, (double)c->dev_bytes / 1e9);
        return q;
    }
}

mda_qwen_t *mda_qwen_open(mda_vkc_t *c, const char *dir, uint32_t rows, uint32_t chunk, uint32_t seqlen, uint32_t head_rows) {
    mda_qwen_t *q = calloc(1, sizeof(mda_qwen_t));
    q->c = c;
    q->cfg = mda_qwen_cfg();
    q->set = mda_safet_open_set(dir);
    bool ok = q->set.n > 0 && mda_qwen_geometry(rows, chunk, seqlen, head_rows) && mda_qwen_check(q) && mda_qwen_ple_open(q);
    if (ok) {
        return mda_qwen_build(q, rows, chunk, seqlen, head_rows);
    } else {
        fprintf(stderr, "qwen: %s does not hold the expected model\n", dir);
        return NULL;
    }
}

void *mda_qwen_copy_run(void *arg) {
    mda_qwen_copy_t *j = arg;
    memcpy(j->dst, j->src, j->n);
    return NULL;
}

void mda_qwen_copy_par(uint8_t *dst, const uint8_t *src, uint64_t n) {
    mda_qwen_copy_t jobs[MDA_QWEN_THREADS];
    pthread_t th[MDA_QWEN_THREADS];
    uint64_t per = (n + MDA_QWEN_THREADS - 1) / MDA_QWEN_THREADS;
    for (uint32_t i = 0; i < MDA_QWEN_THREADS; i += 1) {
        uint64_t lo = mda_util_min_size(i * per, n);
        uint64_t hi = mda_util_min_size(lo + per, n);
        jobs[i] = (mda_qwen_copy_t) {
            .dst = dst + lo,
            .src = src + lo,
            .n = hi - lo,
        };
        pthread_create(&th[i], NULL, mda_qwen_copy_run, &jobs[i]);
    }
    for (uint32_t i = 0; i < MDA_QWEN_THREADS; i += 1) {
        pthread_join(th[i], NULL);
    }
}

bool mda_qwen_room(mda_qwen_t *q, const mda_qwen_stage_t *s, uint64_t bytes) {
    bool ok = s->used + bytes <= s->host.sz;
    if (ok == false) {
        fprintf(stderr, "qwen: staging buffer of %llu bytes is too small\n", (unsigned long long)s->host.sz);
        q->c->fail = true;
    }
    return ok;
}

uint64_t mda_qwen_put(mda_qwen_t *q, mda_qwen_stage_t *s, mda_safet_tensor_t t) {
    uint64_t bytes = (uint64_t)t.rows * t.cols * sizeof(uint16_t);
    uint64_t at = s->used;
    if (t.ok == false || t.dtype != MDA_SAFET_BF16) {
        q->c->fail = true;
    } else if (mda_qwen_room(q, s, bytes)) {
        mda_qwen_copy_par((uint8_t *)s->host.ptr + at, t.data, bytes);
        s->used += bytes;
    }
    return at / sizeof(uint16_t);
}

void mda_qwen_pad(mda_qwen_t *q, mda_qwen_stage_t *s, uint64_t bytes) {
    if (mda_qwen_room(q, s, bytes)) {
        memset((uint8_t *)s->host.ptr + s->used, 0, bytes);
        s->used += bytes;
    }
}

void mda_qwen_pad_to(mda_qwen_t *q, mda_qwen_stage_t *s, uint64_t start, uint32_t rows, uint32_t cols) {
    uint64_t want = (start + (uint64_t)rows * cols) * sizeof(uint16_t);
    if (want > s->used) {
        mda_qwen_pad(q, s, want - s->used);
    }
}

uint32_t mda_qwen_put_vec(mda_qwen_t *q, mda_qwen_bank_t *b, mda_safet_tensor_t t) {
    mda_qwen_stage_t *s = &b->vec;
    uint64_t n = (uint64_t)t.rows * t.cols;
    uint64_t at = s->used / sizeof(float);
    if (t.ok == false || t.dtype != MDA_SAFET_BF16) {
        q->c->fail = true;
    } else if (mda_qwen_room(q, s, n * sizeof(float))) {
        float *dst = (float *)s->host.ptr + at;
        for (uint64_t i = 0; i < n; i += 1) {
            dst[i] = mda_safet_bf16(t.data, i);
        }
        s->used += n * sizeof(float);
    }
    return (uint32_t)at;
}

void mda_qwen_stage_reset(mda_qwen_bank_t *b) {
    b->dense.used = 0;
    b->gu.used = 0;
    b->down.used = 0;
    b->vec.used = 0;
}

void mda_qwen_stage_copy(mda_qwen_t *q, mda_qwen_stage_t *s, mda_vkc_buf_t *dev) {
    if (s->used > 0) {
        mda_vkc_buf_t src = mda_vkc_view(&s->host, 0, s->used);
        mda_vkc_buf_t dst = mda_vkc_view(dev, 0, s->used);
        mda_vkc_copy(q->c, &dst, &src, s->used);
    }
}

void mda_qwen_convert(mda_qwen_t *q, const mda_qwen_stage_t *s, mda_vkc_buf_t *dev) {
    uint32_t n = (uint32_t)(s->used / sizeof(uint32_t));
    if (n > 0) {
        mda_vkc_buf_t w = mda_vkc_view(dev, 0, s->used);
        mda_pc_qwen_bf16_t pc = {
            .n = n,
        };
        mda_vkc_buf_t *bufs[1] = {
            &w,
        };
        mda_vkc_run(q->c, &q->k_bf16, bufs, (n + 255) / 256, 1, 1, &pc);
    }
}

void mda_qwen_upload(mda_qwen_t *q, mda_qwen_bank_t *b, bool convert_gu) {
    mda_vkc_begin(q->c);
    mda_qwen_stage_copy(q, &b->dense, &q->wd);
    mda_qwen_stage_copy(q, &b->gu, &q->wgu);
    mda_qwen_stage_copy(q, &b->down, &q->wdown);
    mda_qwen_stage_copy(q, &b->vec, &q->wvec);
    mda_qwen_convert(q, &b->dense, &q->wd);
    if (convert_gu) {
        mda_qwen_convert(q, &b->gu, &q->wgu);
    }
    mda_qwen_convert(q, &b->down, &q->wdown);
    mda_vkc_flush(q->c);
    q->ly = b->ly;
    q->mixer = b->mixer;
}

mda_qwen_res_t mda_qwen_put_res(mda_qwen_t *q, mda_qwen_bank_t *b, const char *base, bool inject) {
    mda_qwen_cfg_t cfg = q->cfg;
    char name[MDA_SAFET_MAX_KEY];
    mda_qwen_res_t r;
    snprintf(name, sizeof(name), "%s.input_mix_weight_down.weight", base);
    r.down = mda_qwen_put(q, &b->dense, mda_qwen_find(q, name));
    if (inject) {
        snprintf(name, sizeof(name), "%s.block_inject_weight.weight", base);
        mda_qwen_put(q, &b->dense, mda_qwen_find(q, name));
    }
    mda_qwen_pad_to(q, &b->dense, r.down, MDA_QWEN_HC_PAD, MDA_QWEN_HC * cfg.d);
    snprintf(name, sizeof(name), "%s.input_mix_weight_up.weight", base);
    r.up = mda_qwen_put(q, &b->dense, mda_qwen_find(q, name));
    snprintf(name, sizeof(name), "%s.hc_norm.weight", base);
    r.norm = mda_qwen_put_vec(q, b, mda_qwen_find(q, name));
    return r;
}

void mda_qwen_put_gdn(mda_qwen_t *q, mda_qwen_bank_t *b, uint32_t l) {
    mda_qwen_layer_t *ly = &b->ly;
    mda_qwen_cfg_t cfg = q->cfg;
    ly->mix = mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "linear_attn.in_proj_qkv.weight"));
    mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "linear_attn.in_proj_z.weight"));
    mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "linear_attn.in_proj_b.weight"));
    mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "linear_attn.in_proj_a.weight"));
    mda_qwen_pad_to(q, &b->dense, ly->mix, mda_qwen_gdn_cols(cfg), cfg.d);
    ly->out = mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "linear_attn.out_proj.weight"));
    ly->conv = mda_qwen_put_vec(q, b, mda_qwen_layer_find(q, l, "linear_attn.conv1d.weight"));
    ly->a_log = mda_qwen_put_vec(q, b, mda_qwen_layer_find(q, l, "linear_attn.A_log"));
    ly->dt_bias = mda_qwen_put_vec(q, b, mda_qwen_layer_find(q, l, "linear_attn.dt_bias"));
    ly->gnorm = mda_qwen_put_vec(q, b, mda_qwen_layer_find(q, l, "linear_attn.norm.weight"));
}

void mda_qwen_put_attn(mda_qwen_t *q, mda_qwen_bank_t *b, uint32_t l) {
    mda_qwen_layer_t *ly = &b->ly;
    mda_qwen_cfg_t cfg = q->cfg;
    ly->mix = mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "self_attn.q_proj.weight"));
    mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "self_attn.k_proj.weight"));
    mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "self_attn.v_proj.weight"));
    mda_qwen_pad_to(q, &b->dense, ly->mix, mda_qwen_attn_cols(cfg), cfg.d);
    ly->out = mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "self_attn.o_proj.weight"));
    ly->qnorm = mda_qwen_put_vec(q, b, mda_qwen_layer_find(q, l, "self_attn.q_norm.weight"));
    ly->knorm = mda_qwen_put_vec(q, b, mda_qwen_layer_find(q, l, "self_attn.k_norm.weight"));
}

void mda_qwen_put_moe(mda_qwen_t *q, mda_qwen_bank_t *b, uint32_t l) {
    mda_qwen_layer_t *ly = &b->ly;
    mda_qwen_cfg_t cfg = q->cfg;
    ly->moe = mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "mlp.gate.weight"));
    mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "mlp.shared_expert_gate.weight"));
    mda_qwen_pad_to(q, &b->dense, ly->moe, cfg.experts + MDA_QWEN_MOE_PAD, cfg.d);
    mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "mlp.shared_expert.gate_proj.weight"));
    mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "mlp.shared_expert.up_proj.weight"));
    ly->sdown = mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "mlp.shared_expert.down_proj.weight"));
    mda_qwen_put(q, &b->gu, mda_qwen_layer_find(q, l, "mlp.experts.gate_up_proj"));
    mda_qwen_put(q, &b->down, mda_qwen_layer_find(q, l, "mlp.experts.down_proj"));
}

void mda_qwen_put_ple(mda_qwen_t *q, mda_qwen_bank_t *b, uint32_t l) {
    mda_qwen_layer_t *ly = &b->ly;
    ly->ple = mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "ple.key_proj.weight"));
    mda_qwen_put(q, &b->dense, mda_qwen_layer_find(q, l, "ple.value_proj.weight"));
    ly->pkey = mda_qwen_put_vec(q, b, mda_qwen_layer_find(q, l, "ple.norm_key.weight"));
    ly->pquery = mda_qwen_put_vec(q, b, mda_qwen_layer_find(q, l, "ple.norm_query.weight"));
    ly->pnorm = mda_qwen_put_vec(q, b, mda_qwen_layer_find(q, l, "ple.norm_conv.weight"));
    ly->pconv = mda_qwen_put_vec(q, b, mda_qwen_layer_find(q, l, "ple.conv1d.weight"));
}

void mda_qwen_load_layer(mda_qwen_t *q, mda_qwen_bank_t *b, uint32_t l) {
    mda_qwen_cfg_t cfg = q->cfg;
    char base[MDA_SAFET_MAX_KEY];
    b->ly.attn = mda_qwen_is_attn(cfg, l);
    b->ly.ple_on = l == cfg.ple_layer;
    snprintf(base, sizeof(base), "layers.%u.attn_hyper_connection", l);
    b->ly.ahc = mda_qwen_put_res(q, b, base, true);
    snprintf(base, sizeof(base), "layers.%u.mlp_hyper_connection", l);
    b->ly.mhc = mda_qwen_put_res(q, b, base, true);
    if (b->ly.attn) {
        mda_qwen_put_attn(q, b, l);
    } else {
        mda_qwen_put_gdn(q, b, l);
    }
    mda_qwen_put_moe(q, b, l);
    if (b->ly.ple_on) {
        mda_qwen_put_ple(q, b, l);
    }
}

void mda_qwen_load_head(mda_qwen_t *q, mda_qwen_bank_t *b) {
    b->mixer = mda_qwen_put_res(q, b, "hyper_connection_mixer", false);
    mda_qwen_put(q, &b->gu, mda_safet_find(&q->set, "lm_head.weight"));
}

void *mda_qwen_load_run(void *arg) {
    mda_qwen_job_t *j = arg;
    mda_qwen_stage_reset(j->b);
    if (j->item < j->q->cfg.n_layer) {
        mda_qwen_load_layer(j->q, j->b, j->item);
    } else {
        mda_qwen_load_head(j->q, j->b);
    }
    return NULL;
}

void mda_qwen_slots_begin(mda_qwen_t *q) {
    q->n_slots = 0;
    mda_vkc_fill(q->c, &q->amax, 0);
}

uint32_t mda_qwen_slot(mda_qwen_t *q) {
    if (q->n_slots == MDA_QWEN_MAX_SLOTS) {
        fprintf(stderr, "qwen: out of fp16 scale slots\n");
        q->c->fail = true;
        return MDA_PC_NO_SLOT;
    } else {
        uint32_t s = q->n_slots;
        q->n_slots += 1;
        return s;
    }
}

uint32_t mda_qwen_cast(mda_qwen_t *q, mda_vkc_buf_t *src, uint32_t n) {
    uint32_t slot = mda_qwen_slot(q);
    mda_pc_absmax_t pa = {
        .n = n,
        .slot = slot,
    };
    mda_vkc_buf_t *ba[2] = {
        src,
        &q->amax,
    };
    mda_vkc_run(q->c, &q->k_absmax, ba, mda_util_min_u32((n + 255) / 256, 256), 1, 1, &pa);
    mda_pc_cast_t pc = {
        .n = n,
        .slot = slot,
    };
    mda_vkc_buf_t *bc[3] = {
        src,
        &q->sc.a16,
        &q->amax,
    };
    mda_vkc_run(q->c, &q->k_cast, bc, (n + 255) / 256, 1, 1, &pc);
    return slot;
}

void mda_qwen_mm(mda_qwen_t *q, uint32_t sa, mda_vkc_buf_t *w, uint64_t w_off, mda_vkc_buf_t *out, uint32_t rows, uint32_t n, uint32_t k) {
    mda_vkc_buf_t b = mda_vkc_view(w, w_off * sizeof(uint16_t), (VkDeviceSize)n * k * sizeof(uint16_t));
    mda_pc_mm16_t pc = {
        .m = rows,
        .n = n,
        .k = k,
        .sa = sa,
        .sb = MDA_PC_NO_SLOT,
        .ta = 0,
        .tb = 1,
        .store = 1,
        .alpha = 1.0f,
        .beta = 0.0f,
    };
    mda_vkc_buf_t *bufs[6] = {
        &q->sc.a16,
        &b,
        out,
        &q->amax,
        out,
        out,
    };
    uint32_t gx = (n + MDA_QWEN_TILE - 1) / MDA_QWEN_TILE;
    uint32_t gy = (rows + MDA_QWEN_TILE - 1) / MDA_QWEN_TILE;
    if (gx * gy >= MDA_QWEN_WIDE_GROUPS) {
        mda_vkc_run(q->c, &q->k_mm16w, bufs, gx, gy, 1, &pc);
    } else {
        mda_vkc_run(q->c, &q->k_mm16, bufs, n / 64, rows / 64, 1, &pc);
    }
}

void mda_qwen_norm(mda_qwen_t *q, mda_vkc_buf_t *x, mda_vkc_buf_t *y, mda_vkc_buf_t *z, mda_pc_qwen_norm_t pc) {
    mda_vkc_buf_t *bufs[4] = {
        x,
        &q->wvec,
        y,
        z,
    };
    mda_vkc_run(q->c, &q->k_norm, bufs, pc.rows * pc.groups, 1, 1, &pc);
}

mda_pc_qwen_norm_t mda_qwen_norm_pc(const mda_qwen_t *q, uint32_t rows, uint32_t groups, uint32_t gsize, uint32_t op) {
    return (mda_pc_qwen_norm_t) {
        .rows = rows,
        .groups = groups,
        .gsize = gsize,
        .in_stride = groups * gsize,
        .in_off = 0,
        .in_gstride = gsize,
        .out_stride = groups * gsize,
        .out_off = 0,
        .out_gstride = gsize,
        .w_off = 0,
        .wstride = gsize,
        .op = op,
        .rot = 0,
        .seqlen = q->seqlen,
        .z_stride = 0,
        .z_off = 0,
        .z_gstride = 0,
        .eps = MDA_QWEN_EPS,
        .scale = 1.0f,
        .theta = MDA_QWEN_THETA,
    };
}

void mda_qwen_elem(mda_qwen_t *q, mda_vkc_buf_t *a, mda_vkc_buf_t *y, mda_pc_qwen_elem_t pc) {
    mda_vkc_buf_t *bufs[2] = {
        a,
        y,
    };
    mda_vkc_run(q->c, &q->k_elem, bufs, (pc.rows * pc.n + 255) / 256, 1, 1, &pc);
}

void mda_qwen_hc(mda_qwen_t *q, mda_vkc_buf_t *a, mda_vkc_buf_t *b, mda_vkc_buf_t *h, uint32_t rows, uint32_t op, uint32_t n) {
    mda_pc_qwen_hc_t pc = {
        .rows = rows,
        .d = q->cfg.d,
        .hc = MDA_QWEN_HC,
        .op = op,
        .inj_stride = MDA_QWEN_HC_PAD,
        .inj_off = q->cfg.hc_rank,
    };
    mda_vkc_buf_t *bufs[3] = {
        a,
        b,
        h,
    };
    mda_vkc_run(q->c, &q->k_hc, bufs, (n + 255) / 256, 1, 1, &pc);
}

void mda_qwen_conv(mda_qwen_t *q, mda_vkc_buf_t *x, mda_vkc_buf_t *y, mda_pc_qwen_conv_t pc) {
    mda_vkc_buf_t *bufs[3] = {
        x,
        &q->wvec,
        y,
    };
    mda_vkc_run(q->c, &q->k_conv, bufs, (pc.rows * pc.ch + 255) / 256, 1, 1, &pc);
}

void mda_qwen_res_rec(mda_qwen_t *q, mda_vkc_buf_t *h, uint32_t rows, const mda_qwen_res_t *r) {
    mda_qwen_cfg_t cfg = q->cfg;
    mda_qwen_scratch_t *sc = &q->sc;
    uint32_t d4 = MDA_QWEN_HC * cfg.d;
    mda_pc_qwen_norm_t pn = mda_qwen_norm_pc(q, rows, MDA_QWEN_HC, cfg.d, 0);
    pn.w_off = r->norm;
    mda_qwen_norm(q, h, &sc->hn, h, pn);
    uint32_t s = mda_qwen_cast(q, &sc->hn, rows * d4);
    mda_qwen_mm(q, s, &q->wd, r->down, &sc->t, rows, MDA_QWEN_HC_PAD, d4);
    mda_pc_qwen_elem_t pe = {
        .rows = rows,
        .n = cfg.hc_rank,
        .astride = MDA_QWEN_HC_PAD,
        .ao = 0,
        .bo = 0,
        .op = 1,
        .scale = 1.0f / MDA_QWEN_HC,
    };
    mda_qwen_elem(q, &sc->t, &sc->u, pe);
    uint32_t su = mda_qwen_cast(q, &sc->u, rows * cfg.hc_rank);
    mda_qwen_mm(q, su, &q->wd, r->up, &sc->up, rows, d4, cfg.hc_rank);
    mda_qwen_hc(q, &sc->up, &sc->hn, &sc->x, rows, 0, rows * cfg.d);
}

void mda_qwen_gdn_rec(mda_qwen_t *q, uint32_t rows) {
    mda_qwen_cfg_t cfg = q->cfg;
    mda_qwen_scratch_t *sc = &q->sc;
    mda_qwen_layer_t *ly = &q->ly;
    uint32_t pcols = mda_qwen_gdn_cols(cfg);
    uint32_t qkv = mda_qwen_qkv_cols(cfg);
    uint32_t kw = cfg.k_heads * cfg.lin_head;
    uint32_t vw = cfg.v_heads * cfg.lin_head;
    uint32_t sx = mda_qwen_cast(q, &sc->x, rows * cfg.d);
    mda_qwen_mm(q, sx, &q->wd, ly->mix, &sc->p, rows, pcols, cfg.d);
    mda_pc_qwen_conv_t pc = {
        .rows = rows,
        .ch = qkv,
        .seqlen = q->seqlen,
        .k = cfg.conv,
        .dil = 1,
        .in_stride = pcols,
        .in_off = 0,
        .w_off = ly->conv,
    };
    mda_qwen_conv(q, &sc->p, &sc->s1, pc);
    mda_pc_qwen_norm_t pq = mda_qwen_norm_pc(q, rows, cfg.k_heads, cfg.lin_head, 1);
    pq.in_stride = qkv;
    pq.out_stride = qkv;
    pq.scale = 1.0f / sqrtf((float)cfg.lin_head);
    mda_qwen_norm(q, &sc->s1, &sc->s1, &sc->s1, pq);
    pq.in_off = kw;
    pq.out_off = kw;
    pq.scale = 1.0f;
    mda_qwen_norm(q, &sc->s1, &sc->s1, &sc->s1, pq);
    mda_pc_qwen_gdn_t pg = {
        .seqlen = q->seqlen,
        .vheads = cfg.v_heads,
        .kheads = cfg.k_heads,
        .qkv_stride = qkv,
        .k_off = kw,
        .v_off = 2 * kw,
        .ps = pcols,
        .boff = qkv + vw,
        .aoff = qkv + vw + cfg.v_heads,
        .a_log = ly->a_log,
        .dt_bias = ly->dt_bias,
    };
    mda_vkc_buf_t *bufs[4] = {
        &sc->s1,
        &sc->p,
        &q->wvec,
        &sc->s2,
    };
    mda_vkc_run(q->c, &q->k_gdn, bufs, rows / q->seqlen * cfg.v_heads, 1, 1, &pg);
    mda_pc_qwen_norm_t pz = mda_qwen_norm_pc(q, rows, cfg.v_heads, cfg.lin_head, 2);
    pz.w_off = ly->gnorm;
    pz.wstride = 0;
    pz.z_stride = pcols;
    pz.z_off = qkv;
    pz.z_gstride = cfg.lin_head;
    mda_qwen_norm(q, &sc->s2, &sc->s2, &sc->p, pz);
    uint32_t so = mda_qwen_cast(q, &sc->s2, rows * vw);
    mda_qwen_mm(q, so, &q->wd, ly->out, &sc->y, rows, cfg.d, vw);
}

void mda_qwen_attn_rec(mda_qwen_t *q, uint32_t rows) {
    mda_qwen_cfg_t cfg = q->cfg;
    mda_qwen_scratch_t *sc = &q->sc;
    mda_qwen_layer_t *ly = &q->ly;
    uint32_t pcols = mda_qwen_attn_cols(cfg);
    uint32_t hd = cfg.head_dim;
    uint32_t qw = cfg.heads * hd;
    uint32_t sx = mda_qwen_cast(q, &sc->x, rows * cfg.d);
    mda_qwen_mm(q, sx, &q->wd, ly->mix, &sc->p, rows, pcols, cfg.d);
    mda_pc_qwen_norm_t pq = mda_qwen_norm_pc(q, rows, cfg.heads, hd, 3);
    pq.in_stride = pcols;
    pq.in_gstride = 2 * hd;
    pq.w_off = ly->qnorm;
    pq.wstride = 0;
    pq.rot = cfg.rot;
    mda_qwen_norm(q, &sc->p, &sc->s1, &sc->p, pq);
    mda_pc_qwen_norm_t pk = mda_qwen_norm_pc(q, rows, cfg.kv_heads, hd, 3);
    pk.in_stride = pcols;
    pk.in_off = 2 * qw;
    pk.w_off = ly->knorm;
    pk.wstride = 0;
    pk.rot = cfg.rot;
    mda_qwen_norm(q, &sc->p, &sc->s2, &sc->p, pk);
    mda_pc_qwen_attn_t pa = {
        .seqlen = q->seqlen,
        .heads = cfg.heads,
        .kv_heads = cfg.kv_heads,
        .p_stride = pcols,
        .v_off = 2 * qw + cfg.kv_heads * hd,
        .g_off = hd,
        .g_hstride = 2 * hd,
        .scale = 1.0f / sqrtf((float)hd),
    };
    mda_vkc_buf_t *bufs[4] = {
        &sc->s1,
        &sc->s2,
        &sc->p,
        &sc->s3,
    };
    mda_vkc_run(q->c, &q->k_attn, bufs, rows / q->seqlen * cfg.heads * (q->seqlen / 64), 1, 1, &pa);
    uint32_t so = mda_qwen_cast(q, &sc->s3, rows * qw);
    mda_qwen_mm(q, so, &q->wd, ly->out, &sc->y, rows, cfg.d, qw);
}

void mda_qwen_moe_mm(mda_qwen_t *q, mda_vkc_buf_t *a, uint32_t sa, mda_vkc_buf_t *w, mda_vkc_buf_t *out, uint32_t m, uint32_t n, uint32_t k) {
    mda_pc_moe_mm_t pc = {
        .m = m,
        .n = n,
        .k = k,
        .sa = sa,
        .sb = MDA_PC_NO_SLOT,
    };
    mda_vkc_buf_t *bufs[5] = {
        a,
        w,
        out,
        &q->amax,
        &q->sc.tiles,
    };
    mda_vkc_run(q->c, &q->k_moe, bufs, n / MDA_QWEN_TILE, m / MDA_QWEN_TILE, 1, &pc);
}

void mda_qwen_route_rec(mda_qwen_t *q, uint32_t rows, uint32_t mrows) {
    mda_qwen_cfg_t cfg = q->cfg;
    mda_qwen_scratch_t *sc = &q->sc;
    mda_vkc_buf_t tiles = mda_vkc_view(&sc->tiles, 0, (VkDeviceSize)mrows / MDA_QWEN_TILE * sizeof(uint32_t));
    mda_vkc_buf_t xp = mda_vkc_view(&sc->xp, 0, (VkDeviceSize)mrows * cfg.d * sizeof(uint16_t));
    mda_vkc_set_nobar(q->c, true);
    mda_vkc_fill(q->c, &sc->count, 0);
    mda_vkc_fill(q->c, &sc->fill, 0);
    mda_vkc_fill(q->c, &tiles, 0xffffffffu);
    mda_vkc_set_nobar(q->c, false);
    mda_vkc_fill(q->c, &xp, 0);
    mda_pc_moe_route_t pr = {
        .rows = rows,
        .experts = cfg.experts,
        .topk = cfg.topk,
        .stride = mda_qwen_moe_cols(cfg),
    };
    mda_vkc_buf_t *br[4] = {
        &sc->p,
        &sc->idx,
        &sc->wt,
        &sc->count,
    };
    mda_vkc_run(q->c, &q->k_route, br, rows, 1, 1, &pr);
    mda_pc_moe_scan_t ps = {
        .experts = cfg.experts,
        .tile = MDA_QWEN_TILE,
    };
    mda_vkc_buf_t *bs[3] = {
        &sc->count,
        &sc->off,
        &sc->tiles,
    };
    mda_vkc_run(q->c, &q->k_scan, bs, 1, 1, 1, &ps);
    mda_pc_moe_scatter_t pt = {
        .rows = rows,
        .topk = cfg.topk,
        .d8 = cfg.d / 8,
    };
    mda_vkc_buf_t *bt[6] = {
        &sc->idx,
        &sc->off,
        &sc->fill,
        &sc->a16,
        &sc->xp,
        &sc->slot,
    };
    mda_vkc_run(q->c, &q->k_scatter, bt, rows * cfg.topk, 1, 1, &pt);
}

void mda_qwen_moe_rec(mda_qwen_t *q, mda_vkc_buf_t *h, uint32_t rows) {
    mda_qwen_cfg_t cfg = q->cfg;
    mda_qwen_scratch_t *sc = &q->sc;
    mda_qwen_layer_t *ly = &q->ly;
    uint32_t pcols = mda_qwen_moe_cols(cfg);
    uint32_t mrows = mda_qwen_moe_rows(cfg, rows);
    uint32_t sx = mda_qwen_cast(q, &sc->x, rows * cfg.d);
    mda_qwen_mm(q, sx, &q->wd, ly->moe, &sc->p, rows, pcols, cfg.d);
    mda_qwen_route_rec(q, rows, mrows);
    mda_qwen_moe_mm(q, &sc->xp, sx, &q->wgu, &sc->g, mrows, 2 * cfg.inter, cfg.d);
    mda_pc_qwen_elem_t pe = {
        .rows = mrows,
        .n = cfg.inter,
        .astride = 2 * cfg.inter,
        .ao = 0,
        .bo = cfg.inter,
        .op = 0,
        .scale = 1.0f,
    };
    mda_qwen_elem(q, &sc->g, &sc->hp, pe);
    uint32_t sh = mda_qwen_slot(q);
    mda_pc_absmax_t pa = {
        .n = mrows * cfg.inter,
        .slot = sh,
    };
    mda_vkc_buf_t *ba[2] = {
        &sc->hp,
        &q->amax,
    };
    mda_vkc_run(q->c, &q->k_absmax, ba, 256, 1, 1, &pa);
    mda_pc_cast_t pc = {
        .n = mrows * cfg.inter,
        .slot = sh,
    };
    mda_vkc_buf_t *bc[3] = {
        &sc->hp,
        &sc->hp16,
        &q->amax,
    };
    mda_vkc_run(q->c, &q->k_cast, bc, (mrows * cfg.inter + 255) / 256, 1, 1, &pc);
    mda_qwen_moe_mm(q, &sc->hp16, sh, &q->wdown, &sc->ye, mrows, cfg.d, cfg.inter);
    pe.rows = rows;
    pe.n = cfg.shared;
    pe.astride = pcols;
    pe.ao = cfg.experts + MDA_QWEN_MOE_PAD;
    pe.bo = cfg.experts + MDA_QWEN_MOE_PAD + cfg.shared;
    mda_qwen_elem(q, &sc->p, &sc->hs, pe);
    uint32_t ss = mda_qwen_cast(q, &sc->hs, rows * cfg.shared);
    mda_qwen_mm(q, ss, &q->wd, ly->sdown, &sc->sh, rows, cfg.d, cfg.shared);
    mda_pc_moe_combine_t pm = {
        .rows = rows,
        .d = cfg.d,
        .topk = cfg.topk,
        .pstride = pcols,
        .sg_off = cfg.experts,
        .inj_stride = MDA_QWEN_HC_PAD,
        .inj_off = cfg.hc_rank,
        .hc = MDA_QWEN_HC,
    };
    mda_vkc_buf_t *bm[7] = {
        &sc->wt,
        &sc->slot,
        &sc->ye,
        &sc->p,
        &sc->sh,
        &sc->t,
        h,
    };
    mda_vkc_run(q->c, &q->k_combine, bm, (rows * cfg.d + 255) / 256, 1, 1, &pm);
}

uint32_t mda_qwen_shifted(const uint32_t *seq, uint32_t p, uint32_t k, uint32_t eos) {
    uint32_t hits = 0;
    for (uint32_t i = 1; i <= k && i <= p; i += 1) {
        hits += seq[p - i] == eos;
    }
    if (p >= k && hits == 0) {
        return seq[p - k];
    } else {
        return eos;
    }
}

uint64_t mda_qwen_ngram_id(const mda_qwen_t *q, uint32_t row, uint32_t h) {
    mda_qwen_cfg_t cfg = q->cfg;
    const mda_qwen_ple_t *pl = &q->ple;
    uint32_t p = row % q->seqlen;
    const uint32_t *seq = q->host_ids + (row - p);
    uint32_t n = 2 + h / (cfg.ngram_heads / (cfg.ngram - 1));
    uint64_t mixed = (uint64_t)mda_qwen_shifted(seq, p, 0, cfg.eos) * pl->mult[0];
    for (uint32_t k = 1; k < n; k += 1) {
        mixed = mixed ^ ((uint64_t)mda_qwen_shifted(seq, p, k, cfg.eos) * pl->mult[k]);
    }
    return mixed % pl->sizes[h] + pl->offs[h];
}

void *mda_qwen_gather_run(void *arg) {
    mda_qwen_gather_t *j = arg;
    const mda_qwen_t *q = j->q;
    const mda_qwen_ple_t *pl = &q->ple;
    uint32_t d = q->cfg.d;
    for (uint32_t r = j->lo; r < j->hi; r += 1) {
        for (uint32_t h = 0; h < q->cfg.ngram_heads; h += 1) {
            uint64_t id = mda_qwen_ngram_id(q, j->row0 + r, h);
            const mda_safet_tensor_t *t = &pl->shards[id / pl->shard_rows];
            uint64_t base = (id % pl->shard_rows) * pl->dim;
            for (uint32_t c = 0; c < pl->dim; c += 1) {
                j->e[(uint64_t)r * d + h * pl->dim + c] = mda_safet_bf16(t->data, base + c);
            }
        }
    }
    return NULL;
}

void mda_qwen_gather(mda_qwen_t *q, uint32_t row0, uint32_t rows) {
    mda_qwen_gather_t jobs[MDA_QWEN_THREADS];
    pthread_t th[MDA_QWEN_THREADS];
    uint32_t per = (rows + MDA_QWEN_THREADS - 1) / MDA_QWEN_THREADS;
    for (uint32_t i = 0; i < MDA_QWEN_THREADS; i += 1) {
        uint32_t lo = mda_util_min_u32(i * per, rows);
        jobs[i] = (mda_qwen_gather_t) {
            .q = q,
            .e = q->sc.e_host.ptr,
            .row0 = row0,
            .lo = lo,
            .hi = mda_util_min_u32(lo + per, rows),
        };
        pthread_create(&th[i], NULL, mda_qwen_gather_run, &jobs[i]);
    }
    for (uint32_t i = 0; i < MDA_QWEN_THREADS; i += 1) {
        pthread_join(th[i], NULL);
    }
}

void mda_qwen_ple_rec(mda_qwen_t *q, mda_vkc_buf_t *h, uint32_t rows) {
    mda_qwen_cfg_t cfg = q->cfg;
    mda_qwen_scratch_t *sc = &q->sc;
    mda_qwen_layer_t *ly = &q->ly;
    uint32_t d = cfg.d;
    uint32_t d4 = MDA_QWEN_HC * d;
    uint32_t pcols = mda_qwen_ple_cols(cfg);
    VkDeviceSize eb = (VkDeviceSize)rows * d * sizeof(float);
    mda_vkc_buf_t src = mda_vkc_view(&sc->e_host, 0, eb);
    mda_vkc_buf_t dst = mda_vkc_view(&sc->e, 0, eb);
    mda_vkc_copy(q->c, &dst, &src, eb);
    uint32_t se = mda_qwen_cast(q, &sc->e, rows * d);
    mda_qwen_mm(q, se, &q->wd, ly->ple, &sc->p, rows, pcols, d);
    mda_pc_qwen_norm_t pk = mda_qwen_norm_pc(q, rows, MDA_QWEN_HC, d, 0);
    pk.in_stride = pcols;
    pk.w_off = ly->pkey;
    mda_qwen_norm(q, &sc->p, &sc->s1, &sc->p, pk);
    mda_pc_qwen_norm_t pq = mda_qwen_norm_pc(q, rows, MDA_QWEN_HC, d, 0);
    pq.w_off = ly->pquery;
    mda_qwen_norm(q, h, &sc->s2, h, pq);
    mda_pc_qwen_ple_t pp = {
        .rows = rows,
        .d = d,
        .hc = MDA_QWEN_HC,
        .pstride = pcols,
        .voff = d4,
    };
    mda_vkc_buf_t *bp[4] = {
        &sc->s1,
        &sc->s2,
        &sc->p,
        &sc->s3,
    };
    mda_vkc_run(q->c, &q->k_ple, bp, rows * MDA_QWEN_HC, 1, 1, &pp);
    mda_pc_qwen_norm_t pn = mda_qwen_norm_pc(q, rows, MDA_QWEN_HC, d, 0);
    pn.w_off = ly->pnorm;
    mda_qwen_norm(q, &sc->s3, &sc->s1, &sc->s3, pn);
    mda_pc_qwen_conv_t pc = {
        .rows = rows,
        .ch = d4,
        .seqlen = q->seqlen,
        .k = cfg.ple_conv,
        .dil = cfg.ple_dil,
        .in_stride = d4,
        .in_off = 0,
        .w_off = ly->pconv,
    };
    mda_qwen_conv(q, &sc->s1, &sc->s2, pc);
    mda_qwen_hc(q, &sc->s3, &sc->s2, h, rows, 2, rows * d4);
}

mda_vkc_buf_t mda_qwen_rows(mda_qwen_t *q, uint32_t row0, uint32_t rows) {
    VkDeviceSize per = (VkDeviceSize)MDA_QWEN_HC * q->cfg.d * sizeof(float);
    return mda_vkc_view(&q->h, row0 * per, rows * per);
}

void mda_qwen_layer_rec(mda_qwen_t *q, uint32_t row0, uint32_t rows) {
    mda_qwen_scratch_t *sc = &q->sc;
    mda_vkc_buf_t h = mda_qwen_rows(q, row0, rows);
    mda_qwen_slots_begin(q);
    if (q->ly.ple_on) {
        mda_qwen_ple_rec(q, &h, rows);
    }
    mda_qwen_res_rec(q, &h, rows, &q->ly.ahc);
    if (q->ly.attn) {
        mda_qwen_attn_rec(q, rows);
    } else {
        mda_qwen_gdn_rec(q, rows);
    }
    mda_qwen_hc(q, &sc->y, &sc->t, &h, rows, 1, rows * q->cfg.d);
    mda_qwen_res_rec(q, &h, rows, &q->ly.mhc);
    mda_qwen_moe_rec(q, &h, rows);
}

void mda_qwen_embed(mda_qwen_t *q) {
    mda_qwen_cfg_t cfg = q->cfg;
    mda_qwen_bank_t *b = &q->bank[1];
    mda_qwen_stage_reset(b);
    mda_qwen_put(q, &b->gu, mda_qwen_find(q, "embed_tokens.weight"));
    mda_qwen_upload(q, b, false);
    for (uint32_t row0 = 0; row0 < q->rows && q->c->fail == false; row0 += q->chunk) {
        mda_vkc_buf_t h = mda_qwen_rows(q, row0, q->chunk);
        mda_vkc_buf_t ids = mda_vkc_view(&q->ids, (VkDeviceSize)row0 * sizeof(uint32_t), (VkDeviceSize)q->chunk * sizeof(uint32_t));
        mda_pc_qwen_embed_t pc = {
            .rows = q->chunk,
            .d = cfg.d,
            .hc = MDA_QWEN_HC,
        };
        mda_vkc_buf_t *bufs[3] = {
            &ids,
            &q->wgu,
            &h,
        };
        mda_vkc_run(q->c, &q->k_embed, bufs, (q->chunk * cfg.d + 255) / 256, 1, 1, &pc);
    }
}

void mda_qwen_layer(mda_qwen_t *q) {
    for (uint32_t row0 = 0; row0 < q->rows && q->c->fail == false; row0 += q->chunk) {
        if (q->ly.ple_on) {
            mda_qwen_gather(q, row0, q->chunk);
        }
        mda_vkc_begin(q->c);
        mda_qwen_layer_rec(q, row0, q->chunk);
        mda_vkc_flush(q->c);
    }
}

bool mda_qwen_head(mda_qwen_t *q, mda_qwen_sink_t sink, void *ctx) {
    mda_qwen_cfg_t cfg = q->cfg;
    mda_qwen_res_t mixer = q->mixer;
    uint32_t sunk = 0;
    uint32_t hr = q->head_rows;
    for (uint32_t row0 = 0; row0 < q->rows && q->c->fail == false; row0 += hr) {
        mda_vkc_buf_t h = mda_qwen_rows(q, row0, hr);
        mda_vkc_begin(q->c);
        mda_qwen_slots_begin(q);
        mda_qwen_res_rec(q, &h, hr, &mixer);
        uint32_t sx = mda_qwen_cast(q, &q->sc.x, hr * cfg.d);
        mda_qwen_mm(q, sx, &q->wgu, 0, &q->sc.logits, hr, cfg.vocab, cfg.d);
        mda_vkc_flush(q->c);
        if (q->c->fail == false) {
            bool took = sink(ctx, q, row0, hr);
            sunk += took;
        }
    }
    return q->c->fail == false && sunk == q->rows / hr;
}

void mda_qwen_prefetch(mda_qwen_t *q, mda_qwen_job_t *job, pthread_t *th, uint32_t item) {
    *job = (mda_qwen_job_t) {
        .q = q,
        .b = &q->bank[item % 2],
        .item = item,
    };
    pthread_create(th, NULL, mda_qwen_load_run, job);
}

bool mda_qwen_steps(mda_qwen_t *q, mda_qwen_sink_t sink, void *ctx) {
    uint32_t n = q->cfg.n_layer;
    mda_qwen_job_t jobs[2];
    pthread_t th[2];
    uint32_t sunk = 0;
    double wait = 0.0;
    mda_qwen_prefetch(q, &jobs[0], &th[0], 0);
    for (uint32_t i = 0; i <= n; i += 1) {
        double t0 = mda_util_now();
        pthread_join(th[i % 2], NULL);
        wait += mda_util_now() - t0;
        mda_qwen_upload(q, &q->bank[i % 2], true);
        if (i < n) {
            mda_qwen_prefetch(q, &jobs[(i + 1) % 2], &th[(i + 1) % 2], i + 1);
        }
        if (q->c->fail == false && i < n) {
            mda_qwen_layer(q);
        } else if (q->c->fail == false) {
            bool took = mda_qwen_head(q, sink, ctx);
            sunk += took;
        }
    }
    fprintf(stderr, "qwen: %.1f s waiting on weight loads\n", wait);
    return sunk == 1 && q->c->fail == false;
}

bool mda_qwen_run(mda_qwen_t *q, const uint32_t *ids, mda_qwen_sink_t sink, void *ctx) {
    double t0 = mda_util_now();
    q->host_ids = ids;
    mda_vkc_write(q->c, &q->ids, ids);
    mda_qwen_embed(q);
    bool ok = q->c->fail == false && mda_qwen_steps(q, sink, ctx);
    fprintf(stderr, "qwen: %u rows in %.1f s\n", q->rows, mda_util_now() - t0);
    return ok;
}
