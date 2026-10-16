# RWKV-7 + Masked-Diffusion Denoiser — Architecture & Optimization Log

A from-scratch language model: **RWKV-7 ("Goose") blocks trained as a masked-text
diffusion denoiser**, implemented entirely in **raw Vulkan compute** (C host + GLSL
compute shaders). No Python, no PyTorch, no GGML, no autograd framework — forward and
backward are hand-written and finite-difference verified. Target hardware: AMD Ryzen
AI MAX+ 395 "Strix Halo", Radeon 8060S iGPU (gfx1151, RDNA 3.5, 40 CUs, 128 GB unified
memory).

This document describes the system as it exists and records what was tried during the
performance work (and why each thing did or didn't help) so future work doesn't repeat
dead ends.

---

## 1. Model

**Config (src/train.c, sample.c, bench.c, generate.c):**
`d=640, n_layer=12, n_head=10 (head_dim=64), seqlen=64, batch=8, ffn_hidden=1728`,
vocab=16000 subword + 1 mask id. **~120M parameters.** fp32 master weights, fp16
mirrors for the matmuls.

### 1.1 Tokenizer — Unigram LM subword (src/tokenizer.h, src/train_tokenizer.c)
- A **Unigram language-model** subword vocab (the SentencePiece-style probabilistic
  model — NOT BPE), trained from `data/wiki.txt` by the real algorithm: seed frequent
  substrings, then **EM (forward-backward expected counts) + likelihood pruning** down
  to 16k tokens. Built in C, no Python, no external vocab file.
- Runtime encode is **Viterbi max-likelihood segmentation** using a **byte trie** for
  the per-position token lookup (the trie took corpus tokenization from 0.1 MB/s ->
  30 MB/s, ~300x). ~4.5 bytes/token compression.
- Why subword and not bytes: the original model was byte-level, which is the worst
  corner of the design space for diffusion — the 2026 paper "The Efficiency Gap in Byte
  Modeling" shows byte+masked-diffusion is "doubly destructive" (~20x slower at equal
  quality than tokens) because diffusion masks scatter context AND bytes are
  semantically vacuous. Subword fixed the fundamental quality problem.

### 1.2 RWKV-7 block (src/fwd.h, src/step.h)
Per layer, **pre-norm**, two sub-blocks:
- **Time mixing (the WKV-7 recurrence):** rmsnorm -> token-shift -> project to
  k,v,r,w,a -> per-head L2-normalize k -> the delta-rule recurrence -> readout. The
  state `S` is a `[head_dim x head_dim]` matrix per head, evolved over time:
  ```
  Sk_t[j]  = sum_i S_{t-1}[i,j] * k_t[i]
  S_t[i,j] = w_t[j]*S_{t-1}[i,j] + a_t[i]*k_t[i]*(v_t[j] - Sk_t[j])
  out_t[j] = sum_i r_t[i] * S_t[i,j]
  ```
  `w` is a per-column decay, `a` a per-key-channel "in-context learning rate" (RWKV-7's
  `a*k`), so every channel of the `a` projection is used.
  **Bidirectional**: run L->R and R->L, **average** the outputs (needed for a
  denoiser — every position sees past and future; averaging not summing, or the
  magnitude doubles and drifts).
- **Channel mixing (FFN):** rmsnorm -> gate & up projections -> silu(gate)*up -> down.

**CRITICAL stability fact:** the delta-rule recurrence diverges (state -> 1e37 -> NaN)
unless `k` is **L2-normalized per head** before the recurrence. With k̂=k/‖k‖ the
rank-1 correction is a bounded projection. The backward needs the proper Jacobian
`dk = (dk̂ - k̂(k̂·dk̂))/‖k‖`, not identity.

### 1.3 Masked diffusion (src/data.h)
Absorbing-state masked diffusion (MDLM): each training sequence samples a mask rate
`t ∈ [0.02, 1]`, masks each token independently with prob t, and the model predicts the
originals. Loss is **1/t-weighted cross-entropy on masked positions only**. The 1/t
weighting produces high-variance gradients (handled by gradient clipping). adaLN-Zero
timestep conditioning: sinusoidal embed of t -> t_proj -> per-block scale/shift on both
norms. Training and inference share one forward (`forward_rec` in src/fwd.h), so eval
and generation run exactly the trained network; generation conditions each sequence
on its current masked fraction.

### 1.4 Generation (src/generate.c)
Iterative denoising: start fully masked, run **N forward passes**, each predicting all
masked positions and revealing the highest-confidence ones (confidence-based parallel
unmasking). Cost is **linear in N**. `build/generate <N>`.

---

## 2. Vulkan compute harness (src/vkc.h)

- One device, **host-visible coherent buffers** (the iGPU is UMA, so host-visible is
  the fast path — there is no separate device memory to copy to).
- Kernels loaded from `build/*.spv`, dispatched by handle.
- **Command batching:** `vkc_begin` / record many dispatches (with a compute->compute
  memory barrier between dependent ones) / `vkc_flush` submits once. This replaced a
  per-dispatch `vkQueueSubmit + vkQueueWaitIdle` that dominated with ~250 small kernels.
  The inference forward, the whole training step (forward + backward), the grad-norm
  reduction and the clip + AdamW update are each one batch.
- **wave32 is forced** at pipeline creation (`requiredSubgroupSize=32` +
  `subgroupSizeControl` feature). RDNA3 WMMA runs in wave32 but the device defaults to
  wave64; multi-subgroup matmul kernels that assume 32-lane waves silently break (half
  the output uncomputed) without this.
- Device features enabled: cooperative-matrix, shaderFloat16, 16-bit storage,
  subgroup-size-control, atomic-float. API version 1.3; glslc uses
  `--target-env=vulkan1.3`.

---

## 3. Kernels (shader/*.comp)

| kernel | purpose |
|---|---|
| `embed` | token + position embedding lookup |
| `tokenshift` | RWKV token shift |
| `wkv7_fwd` / `wkv7_bwd` | the WKV-7 recurrence (forward saves states for backward) |
| `matmul` / `matmul_bw` | scalar fp32 matmul (backward, the batch-row t_proj matmul, and the fp32 reference mode) |
| `matmul_f16` | **fp16 cooperative-matrix matmul, 64x64 block** (per-layer matmuls; the ada projections with cond padded to 64 rows) |
| `matmul_f16_wide` | fp16 coopmat, 256x64 / 4x4 block (the large-n vocab head) |
| `cast_f16` | fp32 <-> fp16 cast (weight mirrors + activation casts) |
| `temb` | sinusoidal embedding of the mask rate t (from the 1/t weight) |
| `rmsnorm`, `adaln` | normalization (forward + backward); adaln reads/writes the ada6 slots in place |
| `softmax_ce` | masked cross-entropy + dlogits |
| `adamw` | optimizer step |
| `vecops` | elementwise glue: add, mul, sigmoid, average, copy, sigmoid', scale |
| `repack` | [bt,d] <-> [b,head,t,dh] layout for wkv |
| `sumsq` | GPU grad-norm reduction (for clipping) |
| `normheads` | per-head L2 normalization of k (+ saves 1/‖k‖) and its backward |
| `argmax` | per-row top-1 token + confidence (generation sampling) |

Shape limits (checked by `mcfg_check` in src/model.h): head_dim d/n_head must be 64
(wkv7_fwd/bwd), and ffn_hidden, vocab and batch*seqlen must be multiples of 64 (the
fp16 matmul blocks).

Binaries: `train`, `sample`, `bench`, `generate`, `probe_coopmat`, `test_matmul_f16`,
`test_grad`, `train_tokenizer`. `test_grad` finite-difference checks every parameter's
gradient on a tiny model in the fp32 reference mode (`model_t.ref32`) and checks that
train_step and model_forward compute the same loss.

---

## 4. Performance — current state

Measured on the 8060S, batch=8, seqlen=64 (512 tokens/forward). These numbers (and the
step-1 loss below) predate the forward gaining adaLN conditioning in inference,
per-channel `a`, and the batched training step; re-measure before relying on them.

| metric | start | now | factor |
|---|---|---|---|
| forward pass | 578 ms | **~14.2 ms** | 41x |
| inference throughput | 886 tok/s | **~36,000 tok/s** | 41x |
| training step | 22.2 s | **~2.9 s** | 7.7x |
| generation (16 steps) | n/a | 234 ms (16 forwards + ~0.6 ms/step) | — |
| matmul (head shape) | 5.4 TFLOP/s | **19 TFLOP/s** | 3.5x |

**Forward breakdown (all isolation-measured):** matmuls ~9-10 ms, wkv ~4-5 ms,
everything else (rmsnorm, tokenshift, casts, embed, glue) each <0.2 ms (negligible).

**Quality:** correctness is verified by held-out bits/byte (random-init 16k-vocab model
= 3.1172, stable across all kernel changes) and by the training loss trajectory being
bit-identical after every optimization (step 1 loss 23.8789). The model has only been
trained briefly; held-out generalization is modest and not the focus of this log.

---

## 5. What was tried (and the result) — the optimization log

### 5.1 Things that WORKED (and are deployed)
1. **Host glue -> GPU.** The #1 early win. Host loops over Vulkan mapped (write-combined)
   memory are ~50000x slower than RAM. Moving per-head L2-norm (`host_norm_heads` alone
   was ~75% of the original forward), sigmoid, residual adds, silu-mul, and buffer copies
   to GPU kernels. 578 -> ~139 ms.
2. **A large host memcpy of a GPU buffer is also catastrophic** — the logits snapshot
   (512x16000 fp32 = 32 MB) was ~91 ms; a GPU copy is ~free.
3. **Command batching** (one submit per forward instead of ~250 submit+wait). 45 -> 30 ms.
4. **fp16 cooperative-matrix matmul.** The matrix cores. Lazy weight mirror (`dirty16`,
   re-cast only when weights change). 7.4x over scalar in isolation.
5. **Register-blocked GEMM.** The big mid-session win after re-measuring the hardware
   ceiling. Swept block sizes (all bit-verified): simple-16x16 = 5.4 TFLOP/s, 1x4 = 10.7,
   2x8 = 17.9, **4x4 (16 acc) = 19.0**, 4x8 = 5.9 (register spill). 4x4 square reuse is
   the peak for the VGPR budget. **No shared memory** (RDNA3 L2 is good enough).
6. **Shape-dispatched matmul.** The wide 4x4 block (256x64) wins on the large-n vocab
   head but STARVES small per-layer matmuls (too few workgroups). The 64x64 block wins on
   the per-layer shapes. `mm16_pre` picks by shape (`n>=4096 && rows%256==0` -> wide).
   ~7% forward win (only visible with min-of-N benchmarking; the mean is too noisy).
7. **wkv 4-lane reductions + bank-conflict padding.** The Sk and out reductions use all
   256 threads (4 lanes per column). S padded to `[64x65]` so the column-strided read
   has stride 65 (coprime with 32 banks) — conflict-free. Both bit-verified.
8. **GPU argmax** for generation sampling (`argmax.comp`). The host argmax over the 16k
   vocab was 40 s for 8 steps; GPU dropped it to 0.17 s (236x).
9. **CE-skip in inference mode** (`m->infer`): generation only needs logits, not the CE
   loss + host reduce. ~14% off each generation step.
10. **GPU-resident training activations.** `act_t` was host `float*` arrays, so the
    forward did ~600 GPU<->host activation copies/step (~55% of a training step). Made
    them GPU buffers the backward reads directly. Plus all the host-glue conversions.
    Training 22.2 -> 2.9 s/step.

### 5.2 Things that DID NOT help (measured, then reverted)
- **int8 cooperative-matrix.** The hardware exposes 16x16x16 s8/s8->s32, but measured
  only 1.12x over fp16 (5.7 vs 5.0 TOP/s) — RADV runs both at ~the same rate. Not worth
  the quantization accuracy risk.
- **LDS-staged matmul** (stage K-strips in shared memory to cut global loads). Tried
  twice (64x16 strip, 64x32 K-chunk): both SLOWER (0.66x, 0.59x). RDNA3 L2 + the
  coopmat's own load handling beat manual LDS staging; the LDS round-trip + barriers cost
  more than they save.
- **K-loop double-buffering** (prefetch next A/B fragments). 19 -> 7.8 TFLOP/s — the
  doubled fragment registers spill.
- **Transposed-B / column-major load** (the layout the AMD WMMA docs prefer). Bit-correct
  but 0.35x — RADV's `gl_CooperativeMatrixLayoutColumnMajor` does a slow transposing
  gather.
- **8-accumulator and 4x8 blocks.** Register spill, collapse to ~6 TFLOP/s.
- **Batch scaling** (run batch=16/32 for more throughput). The forward scales ~linearly
  past batch=8 (amortized-refresh microbench), i.e. the GPU is already saturated at
  batch=8. No throughput gain.
- **wkv kvrwa matmul fusion** (5 small matmuls -> 1 big). 1.46x in isolation but ~0 in
  the batched forward (the dispatches already overlap). Reverted (dead code removed).
- **wkv double-buffered per-step vectors** (drop one barrier/step). Bit-identical but
  0.93x — the extra LDS cut occupancy.
- **wkv simple (1-barrier) reductions** instead of 4-lane. 0.59x — the 4-lane's extra
  parallelism beats the barrier savings.
- **HIP/rocWMMA backend** (ROCm 7.1.1 is installed). Measured vanilla hipBLAS on our
  shapes: 6.6-7.3 TFLOP/s — our Vulkan kernel is **~3x faster than the vendor library**
  on every shape we use. A HIP backend would be slower. Dead end.

### 5.3 Things PROVEN impossible / not applicable (by derivation or measurement)
- **Chunked WKV7 to parallelize over time.** Derived and numerically verified a chunked
  formulation for our recurrence (notes/wkv7_chunked.md + .c reference, matches the serial
  kernel to ~1e-7). Built the kernel (notes/wkv7_chunk_fwd.comp.wip) — correct but SLOWER
  than the serial kernel (higher LDS -> lower occupancy; the within-chunk delta-rule solve
  is still sequential). **The standard DeltaNet/GLA "UT transform" (matrix-inverse
  parallelization) does NOT apply to our recurrence**: it requires scalar decay, but ours
  decays per-COLUMN (`w[j]`), so the intra-chunk coupling is DH separate systems, not one
  [CxC] solve. The within-chunk serialization is fundamental; only a parallel scan with a
  flop-heavy DHxDH transition-operator composition could break it (not worth it).
- **wkv occupancy.** Measured batch 8/16/32 -> time scales linearly, so the 80
  workgroups (batch*heads) already saturate the 40 CUs. Not occupancy-bound; it's
  reduction/recurrence-bound. The ~1.1 TFLOP/s is because Sᵀk and rᵀS are matrix-VECTORs
  (can't use matrix cores), which is inherent.
- **The matmul ceiling.** Raw WMMA peak measured at **56.8 TFLOP/s** (HIP microbench,
  back-to-back wmma with operands resident in registers, zero memory traffic — matches
  the 59.4 datasheet). Our GEMM at 19 is ~33% of that; the gap is the A/B load path,
  which a real streaming GEMM cannot avoid. So **19 TFLOP/s is the realistic GEMM ceiling
  on this hardware** (and we beat hipBLAS 3x getting there). The 57 figure is unreachable
  by any real matmul.

### 5.4 Hardware ground truth (verified, not assumed)
- fp16 peak: **59.4 TFLOP/s with WMMA** (the often-quoted 29.7 is the *non-WMMA halved*
  number). Real raw-WMMA measured: 56.8.
- Memory bandwidth: 212 GB/s measured (256 theoretical). The forward is NOT
  bandwidth-bound (matmul traffic ~472 MB/forward = ~1.8 ms at 212 GB/s).
- coopmat configs (probe_coopmat.c): 16x16x16, f16/f16->f32 (used), and s8/s8->s32.

---

## 6. Training stability (history, for context)

The masked-diffusion 1/t loss has huge gradient variance. Working config:
**gradient-clip global-norm <= 2** (GPU-reduced via sumsq.comp), **constant lr 3e-5**
after warmup, **weight-decay 0** (wd interacting with lr-decay was un-learning the
model). There was a recurring late-training instability (model descends then drifts up)
that no knob-tuning fully fixed; it was handled operationally by a best-checkpoint +
auto-restart wrapper earlier in the project. Don't change the loss objective mid-run.

---

## 7. What remains (honest)

The **kernel-level forward is at its measured hardware ceiling** — every component
isolated and at its limit (matmul 3x hipBLAS / 33% of the unreachable register-resident
WMMA peak; wkv reduction-bound and proven non-time-parallelizable; glue negligible).
Every kernel hypothesis has been ruled out by direct measurement.

The genuine remaining levers are **not pure-kernel speed**:
1. **Fewer denoising steps** (distillation / better schedule, e.g. T3D self-distillation,
   dKV-cache-style reuse of fixed-position state). This is the biggest *generation*
   lever — generation cost is linear in N — but it trades quality for speed and needs a
   trained model to validate.
2. **A different matmul backend** is ruled out (hipBLAS is 3x slower here).
3. The model itself is undertrained; real generalization needs more training (now 7.7x
   more feasible) and/or scale.

## 8. Key lessons (so they aren't relearned)
- **Verify hardware peak from datasheets before declaring a ceiling.** "vector peak" vs
  "WMMA peak" vs "what the driver delivers naively" vs "what a real GEMM can reach" are
  four different numbers. I called "ceiling" wrong several times; measurement (hardware
  specs, raw-WMMA microbench, min-of-N benchmarking, isolated-vs-batched profiling,
  vendor-library comparison) corrected each one and took the matmul 4x further.
- **Host loops over mapped GPU memory are the recurring villain** on this UMA part —
  catastrophically slow (write-combined). Never leave an elementwise op or a buffer copy
  on the host in a hot path.
- **Isolated kernel speedups often vanish in the batched forward** (dispatches overlap).
  Always confirm a win in the full forward, with min-of-N (the mean is too noisy at ~14 ms).
- **Verify the algorithm in C before writing a GPU kernel** (the chunked-wkv derivation
  was proven numerically correct before — and despite — the kernel attempt).
