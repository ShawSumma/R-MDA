# Chunked WKV7 — verified reference

Goal: replace the serial 64-step wkv recurrence (barrier/LDS-bound, ~4.6ms) with a
chunked formulation whose heavy ops are matmuls (matrix cores) + only T/C sequential
chunk carries. Verified bit-correct in notes/wkv7_chunked_reference.c (max diff vs the
serial recurrence ~1e-7 across w in [0.01, 0.99], no non-finite).

## The recurrence (per (b,h), state S is [DH x DH])
  Sk_t[j]  = sum_i S_{t-1}[i,j] * k_t[i]
  S_t[i,j] = w_t[j]*S_{t-1}[i,j] - a_t*k_t[i]*Sk_t[j] + a_t*k_t[i]*v_t[j]
  out_t[j] = sum_i r_t[i] * S_t[i,j]
Note: w decays COLUMNS (j); the k-dot is over ROWS (i). a_t is a per-(t) scalar.

## The key trick: factor the per-column cumulative decay
Within a chunk [c0, c0+C), let cw[u][j] = prod_{s=0..u} W[c0+s][j] (inclusive cumulative
column decay). A correction added at local step s, observed at step u (s<=u), is decayed
on column j by cw[u][j]/cw[s][j]. This makes the inter-step coupling factor cleanly.

Define the rank-1 "write" at step u: U[u][j] = a_u * (v_u[j] - Sk_u[j]). Then
  Sk_u[j] = carrySk_u[j] + intraSk_u[j]
  carrySk_u[j] = cw_excl[u][j] * (S0[:,j] . k_u)         # cw_excl = cw[u-1] (1 if u=0)
  intraSk_u[j] = sum_{s<u} (k_s . k_u) * U[s][j] * (cw_excl[u][j]/cw[s][j])
  out_u[j] = cw[u][j]*(r_u . S0[:,j]) + sum_{s<=u} (r_u . k_s) * U[s][j] * (cw[u][j]/cw[s][j])
Chunk-end carry:
  S_C[i,j] = cw[C-1][j]*S0[i,j] + sum_s k_s[i]*U[s][j]*(cw[C-1][j]/cw[s][j])

## Matmul shapes (the GPU win)
  (S0 . k)  readouts -> [C x DH] @ [DH x DH]   (carry)
  (k_s . k_u), (r_u . k_s) -> [C x DH] @ [DH x C]   (intra Gram matrices, C x C)
  final state -> [DH x C] @ [C x DH]
C=16 (RWKV-7 default, tuned for DH=64). T=64 -> 4 chunks (4 sequential carries vs 64 steps).

## Stability
cw[u]/cw[s] with s<=u is always <=1 (no blowup). Tested w down to 0.01: max diff ~1e-7,
no non-finite. Keep state/accumulation in fp32 (the serial kernel already does).

## Status: ALGORITHM VERIFIED IN C. GPU kernel not yet written. Build it as a separate
## kernel (wkv7_chunk_fwd.comp) and verify bit-identical (max abs diff < ~1e-4 on random
## input) against wkv7_fwd before swapping. The backward (save states) needs the same
## chunked treatment -- do forward first, prove inference speedup, then backward.

## UPDATE: kernel built and measured — correct but SLOWER (needs more work)
Built notes/wkv7_chunk_fwd.comp.wip from the verified algorithm. It is CORRECT
(max abs diff 1.7e-5 vs the serial kernel on random input). Optimized with per-chunk
Gram matrices (KK=k.k, RK=r.k) and carry readouts (CS, RS) precomputed once so the
inner step loops are shared-memory lookups, not DH-dot recomputes (that naive version
was 30x slower). Result: isolated wkv 1.10x faster (0.318 vs 0.349ms), BUT the FULL
forward got SLOWER (21-22ms vs 18.3ms serial). Why: the kernel's LDS use ballooned
(S+Snew+KK+RK+RS+CS ~35KB) -> fewer workgroups/CU; and the within-chunk U[u] recurrence
is still 16 serial barrier-bound sub-steps (each U[u] needs all U[s<u]).
TO ACTUALLY WIN you must (a) remove the within-chunk serialization via the triangular
solve U = (I - tril(A))^-1 RHS done as a batched [CxC] op (the DeltaNet "UT transform"),
and (b) budget LDS tightly (don't keep Snew; reuse buffers) to preserve occupancy, and
ideally (c) use the coopmat matrix cores for the [CxDH]@[DHxC] Grams and [CxDH]@[DHxDH]
carry. That's a substantial kernel. The serial wkv7_fwd (with bank-conflict padding)
remains the fastest verified option for now. The chunked SOURCE is kept here as a
correct, ready-to-optimize starting point.

## DEFINITIVE: no parallel-within-chunk solve exists for OUR recurrence (derivation)
The standard way to parallelize the intra-chunk delta-rule (DeltaNet/GLA) is to write the
coupling as a scalar [CxC] lower-triangular system and apply (I + tril(A))^-1 as a MATMUL
(the "UT transform"). I derived that this REQUIRES scalar decay. Our recurrence decays
COLUMNS: w_u[j]. Working it through, the intra-chunk coupling matrix L[u][s] = (a_u/w_u[j])
KK[u][s] depends on j (the column) -> it is NOT a single [CxC] system, it is DH different
systems. So you cannot collapse it to one [CxC] inverse + matmul; the best is vectorized
FORWARD-SUBSTITUTION: sequential in u (16 steps), parallel over j (DH). That is exactly what
the chunked kernel already did. CONCLUSION: the within-chunk 16-step serialization is
FUNDAMENTAL to the column-decay delta rule; there is no matmul reformulation that removes it.
Combined with the chunked kernel's higher LDS use (lower occupancy), the serial wkv7_fwd
(bank-conflict padded) is and will remain the fastest option for this recurrence on this hw,
short of changing the architecture (e.g. scalar decay -> then DeltaNet UT-transform applies).
